#!/usr/bin/env python3
"""Audio narration -> timed transcript -> optional illustrations -> video/draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from drawthings_cli import generate_image


@dataclass
class Caption:
    start: float
    end: float
    text: str


@dataclass
class Scene:
    start: float
    end: float
    prompt: str
    image: str | None = None
    character_present: bool = False


def compose_scene_prompt(prompt: str, suffix: str, character_name: str,
                         character_present: bool, has_references: bool) -> str:
    parts = [prompt.strip()]
    if character_present and has_references:
        parts.append(f"Use image 1 as the identity reference for {character_name}. "
                     "Create the same person with the same face, facial structure, hair, age and distinguishing "
                     "features in this scene. Change pose and surroundings as the scene requires. "
                     "Do not copy the reference photo's studio background.")
    if suffix.strip():
        parts.append(suffix.strip())
    return "\n".join(part for part in parts if part)


def image_request_fingerprint(prompt: str, model: str, width: int, height: int,
                              steps: int | None, seed: int | None, references: list[Path]) -> str:
    data = {"prompt": prompt, "model": model, "width": width, "height": height,
            "steps": steps, "seed": seed,
            "reference_hashes": [hashlib.sha256(path.read_bytes()).hexdigest() for path in references]}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def review_scenes(scenes: list[Scene], output: Path, character_name: str,
                  poll_interval: float = 0.5) -> list[Scene]:
    review_path = output / "scene_review.json"
    review_path.unlink(missing_ok=True)
    (output / "scenes_draft.json").write_text(
        json.dumps({"character_name": character_name, "scenes": [asdict(scene) for scene in scenes]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress(output, "等待确认场景", 64)
    print("场景计划已生成，请在页面确认主角出镜场景和提示词后继续。", flush=True)
    while not review_path.is_file():
        time.sleep(poll_interval)
    reviewed = json.loads(review_path.read_text(encoding="utf-8"))
    items = reviewed.get("scenes") if isinstance(reviewed, dict) else None
    if not isinstance(items, list) or len(items) != len(scenes):
        raise ValueError("场景确认数量与场景计划不一致")
    for scene, item in zip(scenes, items):
        if not isinstance(item, dict) or not isinstance(item.get("character_present"), bool):
            raise ValueError("场景出镜标记无效")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("场景图片提示词不能为空")
        scene.prompt = prompt.strip()
        scene.character_present = item["character_present"]
    return scenes


def snapshot_character_profile(folder: Path, output: Path) -> tuple[str, list[Path]]:
    profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    if profile.get("views", {}).get("front", {}).get("status") != "approved":
        raise ValueError("角色档案需要已确认的正面半身标准照")
    snapshot = output / "character_reference"
    snapshot.mkdir(exist_ok=True)
    references = []
    for view in ("front", "three_quarter", "full_body"):
        source = folder / f"{view}.png"
        if profile["views"].get(view, {}).get("status") == "approved" and source.is_file():
            destination = snapshot / source.name
            shutil.copy2(source, destination)
            references.append(destination)
    if not references:
        raise ValueError("角色档案没有可用的参考图")
    (snapshot / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile["name"], references


def simplify_texts(texts: list[str]) -> list[str]:
    """Convert Chinese text to Simplified Chinese in one ICU invocation."""
    if not texts or not any("\u3400" <= char <= "\u9fff" for value in texts for char in value):
        return texts
    try:
        result = subprocess.run(["uconv", "-x", "Traditional-Simplified"],
                                input="\0".join(texts), text=True, encoding="utf-8",
                                capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("中文简体转换需要 ICU 的 uconv 命令") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"中文简体转换失败：{exc.stderr.strip()}") from exc
    converted = result.stdout.split("\0")
    if len(converted) != len(texts):
        raise RuntimeError("中文简体转换后的段落数量不一致")
    return converted


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, **kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"缺少程序：{command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"命令失败：{' '.join(command[:3])}（退出码 {exc.returncode}）") from exc


def audio_duration(path: Path) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", str(path)], capture_output=True, text=True)
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError("音频时长必须大于零")
    return duration


def split_caption(start: float, end: float, text: str, max_chars: int = 22) -> list[Caption]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text or end <= start:
        return []
    parts = [p.strip() for p in re.findall(r"[^，。！？；,.!?;]+[，。！？；,.!?;]?", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts:
        while len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(part[:max_chars])
            part = part[max_chars:]
        if len(current) + len(part) > max_chars and current:
            chunks.append(current)
            current = ""
        current += part
    if current:
        chunks.append(current)
    total = sum(len(c) for c in chunks)
    cursor = start
    result = []
    for i, chunk in enumerate(chunks):
        next_time = end if i == len(chunks) - 1 else start + (end - start) * sum(len(c) for c in chunks[: i + 1]) / total
        result.append(Caption(cursor, next_time, chunk))
        cursor = next_time
    return result


def decode_audio_ffmpeg(audio: Path) -> bytes:
    """Decode to 16 kHz mono float32 PCM so ASR does not depend on PyAV's decoder API."""
    return run(["ffmpeg", "-v", "error", "-i", str(audio), "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"], capture_output=True).stdout


def download_whisper_model(model_name: str, cache_dir: Path) -> str:
    """Resolve a Whisper model from the local Hugging Face cache only.

    The regular faster-whisper constructor downloads missing files in worker
    threads. A stalled Hugging Face connection therefore leaves a job looking
    as if it is frozen forever. The local-only lookup lets us fail before ASR
    starts and keeps the web worker responsive.
    """
    from faster_whisper.utils import download_model

    return str(download_model(model_name, cache_dir=str(cache_dir), local_files_only=True))


def resolve_whisper_model(model_name: str, cache_dir: Path | None = None) -> str:
    """Return a usable local model directory or raise a clear cache error."""
    candidate = Path(model_name).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())
    if candidate.exists():
        raise RuntimeError(f"Whisper 模型路径不是目录：{candidate}")
    if cache_dir is None:
        hf_home = Path(os.environ.get("HF_HOME", Path(__file__).resolve().parent / ".hf-cache"))
        cache_dir = hf_home / "hub"
    try:
        return download_whisper_model(model_name, cache_dir)
    except Exception as exc:
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        raise RuntimeError(
            f"Whisper 模型 {model_name} 未完整下载或缓存不可用（{detail}）。"
            "请检查 Hugging Face 网络后重试；也可以先使用 tiny 或 small。"
        ) from exc


def transcribe(audio: Path, model_name: str, device: str, language: str,
               progress: Callable[[float], None] | None = None, duration: float | None = None) -> list[Caption]:
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf-cache"))
    try:
        from faster_whisper import WhisperModel
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("转写需要 faster-whisper：pip install faster-whisper") from exc
    model_path = resolve_whisper_model(model_name, Path(os.environ["HF_HOME"]) / "hub")
    model = WhisperModel(model_path, device=device, compute_type="int8" if device == "cpu" else "float16",
                         local_files_only=True)
    pcm = np.frombuffer(decode_audio_ffmpeg(audio), dtype=np.float32)
    segments, _ = model.transcribe(pcm, language=language or None, vad_filter=True, beam_size=5)
    captions = []
    for segment in segments:
        captions.extend(split_caption(segment.start, segment.end, segment.text))
        if progress and duration:
            progress(min(1.0, segment.end / duration))
    if not captions:
        raise RuntimeError("音频未识别出文字")
    return captions


def load_captions(path: Path, duration: float) -> list[Caption]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("captions", raw) if isinstance(raw, dict) else raw
    captions = [Caption(float(item["start"]), float(item["end"]), str(item["text"]).strip()) for item in items]
    if not captions or any(not c.text or c.start < 0 or c.end <= c.start or c.end > duration + 0.5 for c in captions):
        raise ValueError("字幕 JSON 包含空内容或无效时间")
    if any(a.end > b.start + 0.02 for a, b in zip(captions, captions[1:])):
        raise ValueError("字幕 JSON 时间重叠或顺序错误")
    return captions


def srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(captions: list[Caption], path: Path) -> None:
    path.write_text("\n".join(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text}\n" for i, c in enumerate(captions, 1)), encoding="utf-8")


def build_scenes(captions: list[Caption], duration: float, target_seconds: float = 25.0) -> list[Scene]:
    if not captions:
        return [Scene(0, duration, "")]
    boundaries = [0.0]
    for caption in captions[1:]:
        if caption.start - boundaries[-1] >= target_seconds:
            boundaries.append(caption.start)
    boundaries.append(duration)
    return [Scene(start, end, "Cinematic story illustration, no text, no subtitles. Scene: " + " ".join(c.text for c in captions if start <= c.start < end)[:250]) for start, end in zip(boundaries, boundaries[1:])]


def parse_llm_scenes(response: str, captions: list[Caption], duration: float, start_time: float = 0.0,
                     allow_small_tail: bool = False,
                     tail_repaired: Callable[[int], None] | None = None) -> list[Scene]:
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.I)
    raw = json.loads(response)
    items = raw["scenes"]
    if not isinstance(items, list) or not items or not captions:
        raise ValueError("场景结果为空")
    expected = 0
    groups = []
    for item in items:
        first, last = int(item["start_index"]), int(item["end_index"])
        if first != expected or last < first or last >= len(captions):
            raise ValueError("场景索引没有连续覆盖全部字幕")
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("场景图片提示词为空")
        groups.append((first, prompt, item.get("character_present", False) is True))
        expected = last + 1
    if expected != len(captions):
        missing = len(captions) - expected
        if not allow_small_tail or missing > 2:
            raise ValueError(f"场景索引没有覆盖末尾字幕：漏掉 {missing} 条")
        if tail_repaired:
            tail_repaired(missing)
    starts = [start_time] + [captions[first].start for first, _, _ in groups[1:]]
    ends = starts[1:] + [duration]
    return [Scene(start, end, prompt, character_present=present)
            for start, end, (_, prompt, present) in zip(starts, ends, groups)]


def post_json(url: str, payload: dict, api_key: str | None = None, timeout: int = 600) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"API 请求失败：{url}：{exc}") from exc


def analyze_scenes(captions: list[Caption], duration: float, base_url: str, model: str, api_key: str | None,
                   chunk_size: int = 80, visual_style: str = "",
                   trace: Callable[[str], None] | None = None,
                   protagonist_name: str = "") -> list[Scene]:
    if not model:
        raise ValueError("LLM 场景分析需要 --llm-model")
    if chunk_size < 1:
        raise ValueError("LLM 单批字幕数量必须大于零")
    all_scenes = []
    endpoint = base_url.rstrip("/") + "/chat/completions"
    batches: list[tuple[int, list[Caption]]] = []
    current: list[Caption] = []
    text_chars = 0
    for index, caption in enumerate(captions):
        if len(caption.text) > 2000:
            raise ValueError(f"第 {index + 1} 条字幕超过 2000 字，请先拆分该字幕再进行语义分段")
        if current and (len(current) >= chunk_size or text_chars + len(caption.text) > 2000):
            batches.append((index - len(current), current))
            current, text_chars = [], 0
        current.append(caption)
        text_chars += len(caption.text)
    if current:
        batches.append((len(captions) - len(current), current))
    for batch_number, (offset, chunk) in enumerate(batches, 1):
        next_offset = offset + len(chunk)
        end_time = captions[next_offset].start if next_offset < len(captions) else duration
        start_time = 0.0 if offset == 0 else chunk[0].start
        entries = [{"index": i, "text": c.text} for i, c in enumerate(chunk)]
        style = visual_style.strip()
        style_block = f"全局画面设定（所有场景都必须遵循）：\n{style}\n" if style else ""
        character_block = (f"故事主角是「{protagonist_name}」。每个场景增加 character_present 布尔值，"
                           "只有主角确实出镜时才填 true；空镜、其他人物和主角未出镜时填 false。\n") if protagonist_name else ""
        instruction = (style_block + character_block + "把旁白字幕按画面场景切成连续片段，每段建议 15–40 秒。只返回 JSON："
                       '{"scenes":[{"start_index":0,"end_index":3,"prompt":"English visual prompt","character_present":false}]}。'
                       "索引从 0 开始，每条字幕必须恰好出现一次，区间连续，第一段从 0 开始，最后一段结束于最后索引。"
                       "prompt 描述可生成的画面，不要包含文字或字幕。")
        payload = {"model": model, "temperature": 0.2, "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(entries, ensure_ascii=False)},
        ]}
        if trace:
            trace(f"=== 第 {batch_number} 批 LLM 输入 ===\n"
                  f"系统提示词：\n{instruction}\n"
                  f"用户字幕：\n{payload['messages'][1]['content']}\n")
        for attempt in range(2):
            if attempt:
                payload["messages"][0]["content"] = (
                    instruction + f"\n上一次结果校验失败：{error}。请重新输出完整 JSON，"
                    f"最后一个场景的 end_index 必须为 {len(chunk) - 1}。"
                )
                if trace:
                    trace(f"=== 第 {batch_number} 批 LLM 重试输入 ===\n"
                          f"系统提示词：\n{payload['messages'][0]['content']}\n"
                          f"用户字幕：\n{payload['messages'][1]['content']}\n")
            try:
                data = post_json(endpoint, payload, api_key, timeout=180)
            except Exception as exc:
                if trace:
                    trace(f"=== 第 {batch_number} 批 LLM 请求失败 ===\n{exc}\n")
                raise
            content = data["choices"][0]["message"]["content"]
            if trace:
                trace(f"=== 第 {batch_number} 批 LLM {'重试' if attempt else ''}输出 ===\n{content}\n")
            try:
                scenes = parse_llm_scenes(
                    content, chunk, end_time, start_time, allow_small_tail=True,
                    tail_repaired=(lambda missing: trace(
                        f"第 {batch_number} 批模型漏掉末尾 {missing} 条字幕；已将其时间归入最后一个场景。\n"
                    )) if trace else None,
                )
                all_scenes.extend(scenes)
                break
            except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
                error = str(exc)
                if trace:
                    trace(f"第 {batch_number} 批场景校验失败：{error}\n")
        else:
            if trace:
                trace(f"第 {batch_number} 批两次输出均不合格，改用固定时长分段。\n")
            local_captions = [Caption(c.start - start_time, c.end - start_time, c.text) for c in chunk]
            fallback = build_scenes(local_captions, end_time - start_time)
            for scene in fallback:
                scene.start += start_time
                scene.end += start_time
            all_scenes.extend(fallback)
    return all_scenes


def write_progress(output: Path, stage: str, percent: int | None) -> None:
    """Publish progress atomically for the web UI."""
    temp = output / "progress.json.tmp"
    temp.write_text(json.dumps({"stage": stage, "percent": percent}, ensure_ascii=False), encoding="utf-8")
    temp.replace(output / "progress.json")


def _ffconcat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def prepare_images(scenes: list[Scene], output: Path, width: int, height: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    black = output / "black.png"
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d=0.04", "-frames:v", "1", str(black)])
    result = []
    for i, scene in enumerate(scenes):
        if not scene.image or not Path(scene.image).is_file():
            result.append(black)
            continue
        normalized = output / f"scene_{i:03}.png"
        run(["ffmpeg", "-y", "-v", "error", "-i", str(Path(scene.image).resolve()), "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1", "-frames:v", "1", str(normalized)])
        result.append(normalized)
    return result


def render_mp4(audio: Path, scenes: list[Scene], srt: Path, output: Path, width: int, height: int, fps: int,
               progress: Callable[[float], None] | None = None) -> None:
    if not scenes:
        raise ValueError("至少需要一个场景")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avrender-") as tmp:
        work = Path(tmp)
        images = prepare_images(scenes, work / "images", width, height)
        concat = work / "frames.ffconcat"
        lines = ["ffconcat version 1.0"]
        for scene, image in zip(scenes, images):
            lines += [f"file '{_ffconcat_path(image)}'", f"duration {scene.end - scene.start:.6f}"]
        lines.append(f"file '{_ffconcat_path(images[-1])}'")
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
        shutil.copyfile(srt, work / "captions.srt")
        command = ["ffmpeg", "-y", "-v", "error", "-safe", "0", "-f", "concat", "-i", str(concat), "-i", str(audio.resolve()),
             "-vf", f"fps={fps},subtitles=captions.srt:force_style='FontSize=26,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,MarginV=32'",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{scenes[-1].end:.3f}", "-movflags", "+faststart", str(output.resolve())]
        if progress is None:
            run(command, cwd=work)
        else:
            command[-1:-1] = ["-progress", "pipe:1", "-nostats"]
            recent = []
            with subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    recent.append(line.strip())
                    recent = recent[-12:]
                    if line.startswith("out_time_us="):
                        try:
                            progress(min(1.0, float(line.split("=", 1)[1]) / 1_000_000 / scenes[-1].end))
                        except ValueError:
                            pass
                    elif line.startswith("progress=end"):
                        progress(1.0)
                if process.wait() != 0:
                    raise RuntimeError(f"视频合成失败：{' '.join(recent)[-500:]}")


def export_jianying(audio: Path, scenes: list[Scene], srt: Path, output: Path, width: int, height: int) -> Path:
    try:
        import pyJianYingDraft as draft
    except ImportError as exc:
        raise RuntimeError("剪映草稿需要 pyJianYingDraft：pip install pyJianYingDraft") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"剪映草稿目录已存在，请换一个 --output-dir：{output}")
    script = draft.DraftFolder(str(output.parent)).create_draft(output.name, width, height)
    media = output / "media"
    media.mkdir(exist_ok=True)
    audio_copy = media / audio.name
    if audio.resolve() != audio_copy.resolve():
        shutil.copy2(audio, audio_copy)
    prepared = prepare_images(scenes, media / "prepared", width, height)
    script.append_tracks([draft.TrackSpec(draft.TrackType.video, "visual"), draft.TrackSpec(draft.TrackType.audio, "narration")])
    for scene, image in zip(scenes, prepared):
        script.add_segment(draft.VideoSegment(str(image), draft.trange_seconds(scene.start, end=scene.end)), track="visual")
    audio_material = draft.AudioMaterial(str(audio_copy))
    audio_end_us = min(round(scenes[-1].end * 1_000_000), audio_material.duration)
    script.add_segment(draft.AudioSegment(audio_material, draft.Timerange(0, audio_end_us)), track="narration")
    script.import_srt(str(srt.resolve()), track_name="subtitle")
    script.save()
    return output / "draft_content.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="音频转文字、场景图片、字幕视频/剪映草稿")
    parser.add_argument("audio", type=Path, help="输入音频（m4a/mp3/wav 等）")
    parser.add_argument("--output-dir", type=Path, help="输出目录，默认 <音频名>_output")
    parser.add_argument("--output", choices=["mp4", "jianying", "both", "assets"], default="mp4")
    parser.add_argument("--transcript-json", type=Path, help="复用已校对的字幕 JSON，跳过转写")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper 模型名称或路径")
    parser.add_argument("--device", default="cpu", help="faster-whisper 设备：cpu/cuda")
    parser.add_argument("--language", default="zh", help="识别语言，空字符串为自动识别")
    parser.add_argument("--scene-mode", choices=["none", "fixed", "llm"], default="none")
    parser.add_argument("--scene-seconds", type=float, default=25.0)
    parser.add_argument("--visual-style", default=os.getenv("VISUAL_STYLE", ""), help="LLM 场景分析使用的全局画面设定")
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--generate-images", action="store_true", help="通过 Draw Things 为场景生图")
    parser.add_argument("--images-dir", type=Path, help="复用 scene_001.png、scene_002.png 等图片")
    parser.add_argument("--drawthings-model", help="本机已下载的 Draw Things 模型文件名")
    parser.add_argument("--image-steps", type=int, help="生图步数；留空使用模型推荐值")
    parser.add_argument("--image-seed", type=int, default=271829, help="场景生图使用的固定 seed，默认 271829")
    parser.add_argument("--image-prompt-suffix", default="", help="附加到每个场景生图提示词末尾的固定文本")
    parser.add_argument("--character-profile", type=Path, help="已确认的单主角角色档案目录")
    parser.add_argument("--review-scenes", action="store_true", help="场景分析后等待页面确认主角出镜标记")
    parser.add_argument("--max-seconds", type=float, help="只处理音频前 N 秒，用于快速试运行")
    parser.add_argument("--jianying-drafts-dir", type=Path, help="注册时使用的剪映草稿根目录；留空自动读取剪映设置")
    parser.add_argument("--register-jianying", action="store_true", help="将兼容格式的草稿登记到剪映首页")
    parser.add_argument("--draft-name", help="剪映草稿名称，默认音频文件名")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args(argv)
    output = None
    try:
        audio = args.audio.resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"音频不存在：{audio}")
        original_audio_name = audio.stem
        if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2 or args.fps <= 0 or args.scene_seconds <= 0:
            raise ValueError("宽高必须为正偶数，帧率和场景秒数必须为正数")
        if args.max_seconds is not None and args.max_seconds <= 0:
            raise ValueError("试运行秒数必须大于零")
        if not 0 <= args.image_seed <= 4294967295:
            raise ValueError("生图 seed 须在 0 到 4294967295 之间")
        if args.review_scenes and not args.character_profile:
            raise ValueError("场景确认需要选择角色档案")
        if args.generate_images and args.scene_mode == "none":
            raise ValueError("生图需要 --scene-mode fixed 或 llm")
        if args.register_jianying and args.output not in ("jianying", "both"):
            raise ValueError("注册剪映需要 --output jianying 或 both")
        if args.register_jianying:
            from jianying_register import registration_status
            status = registration_status(args.jianying_drafts_dir)
            if not status["supported"]:
                raise RuntimeError(status["reason"])
        output = (args.output_dir or audio.with_name(audio.stem + "_output")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        character_name, character_references = ("", [])
        if args.character_profile:
            character_name, character_references = snapshot_character_profile(args.character_profile.resolve(), output)
        write_progress(output, "准备音频", 2)
        duration = audio_duration(audio)
        if args.max_seconds is not None and args.max_seconds < duration:
            preview = output / "preview_audio.m4a"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-t", str(args.max_seconds), "-vn", "-c:a", "aac", str(preview)])
            audio = preview
            duration = audio_duration(audio)
        if args.transcript_json:
            write_progress(output, "读取已有字幕", 52)
            captions = load_captions(args.transcript_json, duration)
        else:
            write_progress(output, "检查转写模型缓存", None)
            captions = transcribe(audio, args.whisper_model, args.device, args.language,
                                  lambda fraction: write_progress(output, "转写音频", 15 + round(fraction * 40)), duration)
        write_progress(output, "生成字幕", 58)
        converted = simplify_texts([caption.text for caption in captions])
        captions = [Caption(caption.start, caption.end, text) for caption, text in zip(captions, converted)]
        (output / "transcript.json").write_text(json.dumps({"audio": str(audio), "duration": duration, "captions": [asdict(c) for c in captions]}, ensure_ascii=False, indent=2), encoding="utf-8")
        srt = output / "subtitles.srt"
        write_srt(captions, srt)
        (output / "transcript.txt").write_text("".join(c.text for c in captions), encoding="utf-8")
        write_progress(output, "分析场景", 64)
        if args.scene_mode == "none":
            scenes = [Scene(0, duration, "")]
        elif args.scene_mode == "fixed":
            scenes = build_scenes(captions, duration, args.scene_seconds)
        else:
            trace_path = output / "llm_trace.log"
            trace_path.write_text("", encoding="utf-8")

            def record_llm_trace(message: str) -> None:
                with trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(message)
                    if not message.endswith("\n"):
                        trace_file.write("\n")
                print(message, end="" if message.endswith("\n") else "\n", flush=True)

            scenes = analyze_scenes(captions, duration, args.llm_base_url, args.llm_model, args.llm_api_key,
                                    visual_style=args.visual_style, trace=record_llm_trace,
                                    protagonist_name=character_name)
        for scene, prompt in zip(scenes, simplify_texts([scene.prompt for scene in scenes])):
            scene.prompt = prompt
        if args.review_scenes:
            if args.scene_mode == "fixed":
                for scene in scenes:
                    scene.character_present = True
            review_scenes(scenes, output, character_name)
        for i, scene in enumerate(scenes, 1):
            print(f"场景 {i} [{scene.start:.2f}s - {scene.end:.2f}s] 提示词：{scene.prompt}", flush=True)
        images_dir = (args.images_dir or output / "images").resolve()
        if args.generate_images:
            images_dir.mkdir(parents=True, exist_ok=True)
        image_records = []
        for i, scene in enumerate(scenes, 1):
            write_progress(output, f"处理图片 {i}/{len(scenes)}", 65 + round((i - 1) / len(scenes) * 20))
            image = images_dir / f"scene_{i:03}.png"
            references = character_references if scene.character_present else []
            final_prompt = compose_scene_prompt(scene.prompt, args.image_prompt_suffix, character_name,
                                                scene.character_present, bool(references))
            fingerprint = image_request_fingerprint(final_prompt, args.drawthings_model or "", args.width,
                                                    args.height, args.image_steps, args.image_seed, references)
            metadata = images_dir / f"scene_{i:03}.request.json"
            record = {"scene": i, "prompt": final_prompt, "seed": args.image_seed,
                      "references": [str(path) for path in references], "model": args.drawthings_model,
                      "steps": args.image_steps, "fingerprint": fingerprint, "status": "pending"}
            print(f"场景 {i} 最终生图提示词：{final_prompt}\nseed：{args.image_seed}；参考图：{record['references']}", flush=True)
            if image.is_file() and (args.images_dir is not None or
                                    (metadata.is_file() and json.loads(metadata.read_text(encoding="utf-8")).get("fingerprint") == fingerprint)):
                scene.image = str(image)
                record["status"] = "reused"
            elif args.generate_images:
                try:
                    temporary_image = images_dir / f"scene_{i:03}.pending.png"
                    generate_image(final_prompt, temporary_image, args.drawthings_model or "", args.width, args.height,
                                   args.image_steps,
                                   progress=lambda percent: write_progress(output, f"生成图片 {i}/{len(scenes)}",
                                                                           65 + round(((i - 1) + percent / 100) / len(scenes) * 20)),
                                   seed=args.image_seed, references=references)
                    temporary_image.replace(image)
                    record["status"] = "generated"
                    metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    scene.image = str(image)
                except Exception as exc:
                    print(f"警告：第 {i} 段生图失败，将使用黑色背景：{exc}", file=sys.stderr)
                    record["status"] = "failed"
                    record["error"] = str(exc)
            image_records.append(record)
        (output / "image_generation.json").write_text(json.dumps(image_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "scenes.json").write_text(json.dumps([asdict(s) for s in scenes], ensure_ascii=False, indent=2), encoding="utf-8")
        if args.output in ("mp4", "both"):
            write_progress(output, "合成视频", 88)
            render_mp4(audio, scenes, srt, output / "video.mp4", args.width, args.height, args.fps,
                       progress=lambda fraction: write_progress(output, "合成视频", 88 + round(fraction * 5)))
        if args.output in ("jianying", "both"):
            write_progress(output, "生成剪映草稿", 94)
            draft_path = output / (args.draft_name or "jianying_draft")
            draft_file = export_jianying(audio, scenes, srt, draft_path, args.width, args.height)
            if args.register_jianying:
                from jianying_register import register_draft
                registered = register_draft(draft_file.parent, args.jianying_drafts_dir, args.draft_name or original_audio_name)
                draft_file = registered / ("draft_info.json" if (registered / "draft_info.json").is_file() else "draft_content.json")
            (output / "jianying_draft_path.txt").write_text(str(draft_file.parent) + "\n", encoding="utf-8")
        write_progress(output, "已完成", 100)
        print(f"输出完成：{output}")
        return 0
    except (RuntimeError, ValueError, OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
        if output is not None and output.is_dir():
            write_progress(output, "失败", None)
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
