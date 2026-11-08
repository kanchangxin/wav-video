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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from drawthings_cli import generate_image, list_models, upscaler_available
from parallax_layers import apply_foreground_mask, background_prompt as parallax_background_prompt, foreground_prompt as parallax_foreground_prompt, foreground_mode, preferred_edit_model
from scene_motion import (LLM_PRESETS, PARALLAX_BACKGROUND_EFFECTS, PARALLAX_FOREGROUND_EFFECTS,
                          PARALLAX_PLACEMENTS, PRESETS, choose_preset, generation_size, motion_for,
                          normalize_effects, parallax_scene_numbers, parse_scene_numbers)


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
    motion: str = "static"
    background: str | None = None
    foreground: str | None = None
    motion_reason: str = ""
    motion_effects: dict[str, str] = field(default_factory=dict)
    foreground_effects: dict[str, str] = field(default_factory=dict)
    foreground_method: str = ""
    independent_layer_motion: bool = False

    @property
    def background_motion_preset(self) -> str:
        return "static" if self.motion == "parallax" and self.independent_layer_motion else self.motion


def compose_scene_prompt(prompt: str, suffix: str, character_name: str,
                         character_present: bool, has_references: bool) -> str:
    parts = [prompt.strip()]
    if character_present and has_references:
        parts.append(f"【人物参考】Use image 1 as the character identity reference.")
    if suffix.strip():
        parts.append(suffix.strip())
    return "\n".join(part for part in parts if part)


def image_request_fingerprint(prompt: str, model: str, width: int, height: int,
                              steps: int | None, seed: int | None, references: list[Path],
                              upscale_x2: bool = False) -> str:
    data = {"prompt": prompt, "model": model, "width": width, "height": height,
            "steps": steps, "seed": seed,
            "reference_hashes": [hashlib.sha256(path.read_bytes()).hexdigest() for path in references]}
    if upscale_x2:
        data["upscale_x2"] = True
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def image_cache_matches(metadata: Path, fingerprint: str) -> bool:
    try:
        return json.loads(metadata.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
    except (OSError, ValueError, AttributeError):
        return False


def validate_foreground(path: Path) -> None:
    """Reject almost empty or opaque masks before using a cutout as a layer."""
    try:
        result = run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                      "alphaextract,signalstats,metadata=print:file=-", "-frames:v", "1", "-f", "null", "-"],
                     capture_output=True, text=True)
    except RuntimeError as exc:
        raise RuntimeError("前景图片没有可用的透明通道；请用透明 PNG 或重新生成") from exc
    match = re.search(r"lavfi\.signalstats\.YAVG=([\d.]+)", result.stdout)
    if not match or not 3 <= float(match.group(1)) <= 220:
        raise RuntimeError("人物抠图透明度异常，可能未识别出完整人物")


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
                           "只有主角不得不出镜时才填 true；\n") if protagonist_name else ""
        instruction = (style_block + character_block + "把旁白字幕按画面场景切成连续片段，每段建议 3–7 秒。只返回 JSON："
                       '{"scenes":[{"start_index":0,"end_index":3,"prompt":"English visual prompt","character_present":false}]}。'
                       "索引从 0 开始，每条字幕必须恰好出现一次，区间连续，第一段从 0 开始，最后一段结束于最后索引。"
                       "prompt 按照下面格式，给出绘画提示词， 语言精炼（"+
    """
【风格】
中国古代魏晋南北朝时期，写实人物比例的 2D 游戏叙事插画，半厚涂技法，戏剧性电影光影，高细节。高清画质。
【画面内容】
场景：【地点、时间、环境】。
主角：【动作、姿势、表情、视线、服装】。
其他人物：【数量、位置、行为】。
关键道具：【物品与空间关系】。
叙事重点：【通过具体视觉细节体现的剧情情绪，3个词概括】。
【镜头构图】
景别：【远景 / 全景 / 中景 / 近景 / 特写】。
机位：【正面 / 侧面 / 3/4侧面 / 背面】。
镜头高度：【平视 / 低机位 / 俯视】。
构图：【主体位置、前后景关系】。
【光影情绪】
光线：【时间、光源、光线方向、色调】。
氛围：【核心情绪，以及通过人物与环境表现的具体细节】。
    """
 +" ）")
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


def recommend_scene_motions(scenes: list[Scene], base_url: str, model: str, api_key: str | None,
                            captions: list[Caption] | None = None, visual_style: str = "",
                            trace: Callable[[str], None] | None = None) -> None:
    """Ask the LLM to choose one editable camera move per scene in bounded batches."""
    if not model:
        raise ValueError("自动运镜需要 --llm-model")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    instruction = (
        "你是视频分镜师。为每个场景选择且只选择一种镜头运动。"
        "可选 preset：static（静止，空白/无图片或安静停顿）、push_in（推近，聚焦、紧张）、"
        "pull_out（拉远，揭示环境或疏离）、pan_left（向左平移，向左探索）、"
        "pan_right（向右平移，向右探索）。根据画面内容和旁白意图选择，不要轮换或组合。"
        "左右平移仅在画面具备宽阔空间时使用；特写优先推近或静止。"
        '只返回 JSON：{"motions":[{"index":0,"preset":"push_in","reason":"简短中文理由"}]}。'
        "motions 必须与输入场景一一对应、index 从 0 连续递增。"
    )
    if visual_style.strip():
        instruction += "\n全局画面设定：" + visual_style.strip()[:1000]
    for batch_number, start in enumerate(range(0, len(scenes), 12), 1):
        batch = scenes[start:start + 12]
        entries = []
        for index, scene in enumerate(batch):
            spoken = "".join(c.text for c in captions or [] if scene.start <= c.start < scene.end)
            entries.append({"index": index, "start": round(scene.start, 2), "end": round(scene.end, 2),
                            "visual_prompt": scene.prompt[:500], "narration": spoken[:350]})
        user_content = json.dumps(entries, ensure_ascii=False)
        error = ""
        for attempt in range(2):
            system_content = instruction + (f"\n上次结果不合格：{error}。请完整重写本批结果。" if attempt else "")
            payload = {"model": model, "temperature": 0.2, "messages": [
                {"role": "system", "content": system_content}, {"role": "user", "content": user_content}]}
            if trace:
                trace(f"=== 第 {batch_number} 批自动运镜输入（第 {attempt + 1} 次）===\n"
                      f"系统提示词：\n{system_content}\n场景：\n{user_content}\n")
            data = post_json(endpoint, payload, api_key, timeout=180)
            content = data["choices"][0]["message"]["content"]
            if trace:
                trace(f"=== 第 {batch_number} 批自动运镜输出 ===\n{content}\n")
            try:
                cleaned = re.sub(r"<think>.*?</think>", "", content.strip(), flags=re.S | re.I).strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
                items = json.loads(cleaned)["motions"]
                if not isinstance(items, list) or len(items) != len(batch):
                    raise ValueError("运镜数量与场景数量不一致")
                for index, item in enumerate(items):
                    if not isinstance(item, dict) or item.get("index") != index or item.get("preset") not in LLM_PRESETS:
                        raise ValueError(f"第 {index + 1} 段运镜无效")
                for scene, item in zip(batch, items):
                    scene.motion = item["preset"]
                    scene.motion_reason = str(item.get("reason") or "")[:200]
                break
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                error = str(exc)
                if trace:
                    trace(f"第 {batch_number} 批自动运镜校验失败：{error}\n")
        else:
            for scene in batch:
                scene.motion = "static"
                scene.motion_reason = "模型输出无效，改用静止镜头"
            if trace:
                trace(f"第 {batch_number} 批自动运镜两次输出无效，全部改用静止镜头。\n")
        for index, scene in enumerate(batch, start + 1):
            if trace:
                trace(f"场景 {index} 运镜：{scene.motion}；原因：{scene.motion_reason}\n")


def recommend_character_presence(scenes: list[Scene], base_url: str, model: str,
                                 api_key: str | None, character_name: str,
                                 trace: Callable[[str], None] | None = None) -> None:
    """Classify fixed-duration scenes without changing their time boundaries."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    instruction = (f"判断故事主角「{character_name}」是否应出现在每段画面中。"
                   "只根据该段旁白与画面内容判断，无法确定时返回 false。"
                   '只返回 JSON：{"presence":[{"index":0,"present":false}]}，索引从 0 连续。')
    for batch_number, start in enumerate(range(0, len(scenes), 12), 1):
        batch = scenes[start:start + 12]
        entries = [{"index": i, "prompt": scene.prompt[:500]} for i, scene in enumerate(batch)]
        payload = {"model": model, "temperature": 0.1, "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(entries, ensure_ascii=False)}]}
        if trace:
            trace(f"=== 第 {batch_number} 批主角出镜判断输入 ===\n{json.dumps(payload, ensure_ascii=False)}\n")
        try:
            response = post_json(endpoint, payload, api_key, timeout=180)
            content = response["choices"][0]["message"]["content"]
            if trace:
                trace(f"=== 第 {batch_number} 批主角出镜判断输出 ===\n{content}\n")
            cleaned = re.sub(r"<think>.*?</think>", "", content.strip(), flags=re.S | re.I).strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
            items = json.loads(cleaned)["presence"]
            if not isinstance(items, list) or len(items) != len(batch):
                raise ValueError("出镜判断数量不匹配")
            for index, item in enumerate(items):
                if item.get("index") != index or not isinstance(item.get("present"), bool):
                    raise ValueError("出镜判断索引或数值无效")
            for scene, item in zip(batch, items):
                scene.character_present = item["present"]
        except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
            if trace:
                trace(f"第 {batch_number} 批主角出镜判断失败，默认不出镜：{exc}\n")


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


def parallax_layers_ready(scene: Scene) -> bool:
    return (scene.motion == "parallax" and bool(scene.background and Path(scene.background).is_file())
            and bool(scene.foreground and Path(scene.foreground).is_file())
            and (foreground_mode(scene.prompt) != "ai" or scene.foreground_method in {"ai", "upload"}))


def prepare_foregrounds(scenes: list[Scene], output: Path, width: int, height: int) -> list[Path | None]:
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for i, scene in enumerate(scenes):
        if not parallax_layers_ready(scene):
            result.append(None)
            continue
        normalized = output / f"foreground_{i:03}.png"
        run(["ffmpeg", "-y", "-v", "error", "-i", str(Path(scene.foreground).resolve()), "-vf",
             f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
             f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0,format=rgba",
             "-frames:v", "1", str(normalized)])
        result.append(normalized)
    return result


def prepare_motion_backgrounds(scenes: list[Scene], output: Path, width: int, height: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    black = output / "black.png"
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"color=c=black:s={width}x{height}:d=0.04", "-frames:v", "1", str(black)])
    result = []
    for i, scene in enumerate(scenes):
        source = scene.background if parallax_layers_ready(scene) else scene.image
        if not source or not Path(source).is_file():
            result.append(black)
            continue
        canvas_width, canvas_height = generation_size(width, height, scene.background_motion_preset, scene.motion_effects)
        normalized = output / f"background_{i:03}.png"
        run(["ffmpeg", "-y", "-v", "error", "-i", str(Path(source).resolve()), "-vf",
             f"scale={canvas_width}:{canvas_height}:force_original_aspect_ratio=increase,"
             f"crop={canvas_width}:{canvas_height},setsar=1", "-frames:v", "1", str(normalized)])
        result.append(normalized)
    return result


def render_motion_clip(image: Path, foreground: Path | None, scene: Scene, output: Path,
                       width: int, height: int, fps: int, frame_count: int) -> None:
    """Render one scene from the same linear keyframes used by the draft."""
    end_frame = max(1, frame_count - 1)
    motion = motion_for(scene.background_motion_preset, scene.motion_effects)
    if motion.horizontal_pan_only:
        x_position = f"({motion.x_start}+({motion.x_end - motion.x_start})*n/{end_frame})"
        crop_x = f"clip((iw-ow)/2-{x_position}*ow,0,iw-ow)"
        filter_graph = (f"[0:v]crop=w={width}:h={height}:x='{crop_x}':y=0:exact=1,"
                        "setsar=1,format=yuv420p[background]")
    else:
        progress = f"on/{end_frame}"
        zoom = f"({motion.scale_start}+({motion.scale_end - motion.scale_start})*{progress})"
        x_position = f"({motion.x_start}+({motion.x_end - motion.x_start})*{progress})"
        y_position = f"({motion.y_start}+({motion.y_end - motion.y_start})*{progress})"
        crop_x = f"(iw-iw/zoom)/2-({x_position})*iw/zoom"
        crop_y = f"(ih-ih/zoom)/2+({y_position})*ih/zoom"
        filter_graph = (f"[0:v]zoompan=z='{zoom}':x='{crop_x}':y='{crop_y}':"
                        f"d=1:s={width}x{height}:fps={fps},format=yuv420p[background]")
    command = ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(fps), "-i", str(image)]
    last_label = "background"
    if foreground is not None:
        if motion.fade != "none":
            duration = frame_count / fps
            fade_duration = min(0.5, duration / 3)
            fades = []
            if motion.fade in ("in", "in_out"):
                fades.append(f"fade=t=in:st=0:d={fade_duration}")
            if motion.fade in ("out", "in_out"):
                fades.append(f"fade=t=out:st={max(0, duration - fade_duration)}:d={fade_duration}")
            filter_graph += f";[background]{','.join(fades)}[faded_background]"
            last_label = "faded_background"
        command += ["-loop", "1", "-framerate", str(fps), "-i", str(foreground)]
        fg_motion = motion_for("static", scene.foreground_effects)
        fg_x = (f"(W-w)/2+W*({motion.foreground_x_start + fg_motion.x_start}+"
                f"({motion.foreground_x_end - motion.foreground_x_start + fg_motion.x_end - fg_motion.x_start})*n/{end_frame})")
        fg_y = (f"(H-h)/2-H*({fg_motion.y_start}+({fg_motion.y_end - fg_motion.y_start})*n/{end_frame})")
        fg_scale = (f"({fg_motion.scale_start}+"
                    f"({fg_motion.scale_end - fg_motion.scale_start})*n/{end_frame})")
        fg_filter = (f"format=rgba,scale=w='iw*{fg_scale}':h='ih*{fg_scale}':eval=frame,format=rgba"
                     if fg_motion.scale_start != 1.0 or fg_motion.scale_end != 1.0 else "format=rgba")
        fg_fade = fg_motion.fade
        if fg_fade == "none" and not scene.independent_layer_motion:
            fg_fade = motion.fade
        if fg_fade != "none":
            duration = frame_count / fps
            fade_duration = min(0.5, duration / 3)
            if fg_fade in ("in", "in_out"):
                fg_filter += f",fade=t=in:st=0:d={fade_duration}:alpha=1"
            if fg_fade in ("out", "in_out"):
                fg_filter += f",fade=t=out:st={max(0, duration - fade_duration)}:d={fade_duration}:alpha=1"
        filter_graph += f";[1:v]{fg_filter}[person];[{last_label}][person]overlay=x='{fg_x}':y='{fg_y}':eval=frame:format=auto[composed]"
        last_label = "composed"
    if motion.fade != "none" and foreground is None:
        duration = frame_count / fps
        fade_duration = min(0.5, duration / 3)
        fades = []
        if motion.fade in ("in", "in_out"):
            fades.append(f"fade=t=in:st=0:d={fade_duration}")
        if motion.fade in ("out", "in_out"):
            fades.append(f"fade=t=out:st={max(0, duration - fade_duration)}:d={fade_duration}")
        filter_graph += f";[{last_label}]{','.join(fades)}[faded]"
        last_label = "faded"
    command += ["-filter_complex", filter_graph, "-map", f"[{last_label}]", "-frames:v", str(frame_count),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    run(command)


def render_mp4(audio: Path, scenes: list[Scene], srt: Path, output: Path, width: int, height: int, fps: int,
               progress: Callable[[float], None] | None = None) -> None:
    if not scenes:
        raise ValueError("至少需要一个场景")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avrender-") as tmp:
        work = Path(tmp)
        if any(scene.motion != "static" or scene.motion_effects for scene in scenes):
            backgrounds = prepare_motion_backgrounds(scenes, work / "images", width, height)
            foregrounds = prepare_foregrounds(scenes, work / "foregrounds", width, height)
            images = []
            for i, (scene, background, foreground) in enumerate(zip(scenes, backgrounds, foregrounds)):
                clip = work / f"motion_{i:03}.mp4"
                frame_count = max(1, round(scene.end * fps) - round(scene.start * fps))
                render_motion_clip(background, foreground, scene, clip, width, height, fps, frame_count)
                images.append(clip)
                if progress:
                    progress(0.8 * (i + 1) / len(scenes))
        else:
            images = prepare_images(scenes, work / "images", width, height)
        concat = work / "frames.ffconcat"
        lines = ["ffconcat version 1.0"]
        if any(scene.motion != "static" or scene.motion_effects for scene in scenes):
            lines += [f"file '{_ffconcat_path(image)}'" for image in images]
        else:
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
                            progress(0.8 + 0.2 * min(1.0, float(line.split("=", 1)[1]) / 1_000_000 / scenes[-1].end)
                                     if any(scene.motion != "static" or scene.motion_effects for scene in scenes) else
                                     min(1.0, float(line.split("=", 1)[1]) / 1_000_000 / scenes[-1].end))
                        except ValueError:
                            pass
                    elif line.startswith("progress=end"):
                        progress(1.0)
                if process.wait() != 0:
                    raise RuntimeError(f"视频合成失败：{' '.join(recent)[-500:]}")


def fade_keyframes(segment, draft, fade: str, end_us: int) -> None:
    if fade == "none":
        return
    ramp = min(500_000, max(1, end_us // 3))
    points = []
    if fade in ("in", "in_out"):
        points.extend([(0, 0.0), (ramp, 1.0)])
    if fade in ("out", "in_out"):
        points.extend([(end_us - ramp, 1.0), (end_us, 0.0)])
    for offset, alpha in points:
        segment.add_keyframe(draft.KeyframeProperty.alpha, offset, alpha)


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
    moving = any(scene.motion != "static" or scene.motion_effects for scene in scenes)
    prepared = (prepare_motion_backgrounds(scenes, media / "prepared", width, height) if moving else
                prepare_images(scenes, media / "prepared", width, height))
    foregrounds = prepare_foregrounds(scenes, media / "foregrounds", width, height) if moving else [None] * len(scenes)
    tracks = [draft.TrackSpec(draft.TrackType.video, "visual")]
    if any(foreground is not None for foreground in foregrounds):
        tracks.append(draft.TrackSpec(draft.TrackType.video, "foreground"))
    tracks.append(draft.TrackSpec(draft.TrackType.audio, "narration"))
    script.append_tracks(tracks)
    for scene, image, foreground in zip(scenes, prepared, foregrounds):
        segment = draft.VideoSegment(str(image), draft.trange_seconds(scene.start, end=scene.end))
        motion = motion_for(scene.background_motion_preset, scene.motion_effects)
        if scene.motion != "static" or scene.motion_effects:
            end_us = round((scene.end - scene.start) * 1_000_000)
            for offset, scale, x, y in ((0, motion.scale_start, motion.x_start, motion.y_start),
                                        (end_us, motion.scale_end, motion.x_end, motion.y_end)):
                if motion.horizontal_pan_only:
                    # Restore canvas height after Jianying fits the wide material to its width.
                    scale = segment.material_size[0] / width
                    x *= 2  # Jianying measures position in half canvas widths.
                segment.add_keyframe(draft.KeyframeProperty.uniform_scale, offset, scale)
                segment.add_keyframe(draft.KeyframeProperty.position_x, offset, x)
                segment.add_keyframe(draft.KeyframeProperty.position_y, offset, y)
            fade_keyframes(segment, draft, motion.fade, end_us)
        script.add_segment(segment, track="visual")
        if foreground is not None:
            person = draft.VideoSegment(str(foreground), draft.trange_seconds(scene.start, end=scene.end))
            fg_motion = motion_for("static", scene.foreground_effects)
            end_us = round((scene.end - scene.start) * 1_000_000)
            for offset, fraction in ((0, 0), (end_us, 1)):
                x = (motion.foreground_x_start + (motion.foreground_x_end - motion.foreground_x_start) * fraction
                     + fg_motion.x_start + (fg_motion.x_end - fg_motion.x_start) * fraction)
                person.add_keyframe(draft.KeyframeProperty.position_x, offset, x)
                person.add_keyframe(draft.KeyframeProperty.position_y, offset,
                                    fg_motion.y_start + (fg_motion.y_end - fg_motion.y_start) * fraction)
                person.add_keyframe(draft.KeyframeProperty.uniform_scale, offset,
                                    fg_motion.scale_start + (fg_motion.scale_end - fg_motion.scale_start) * fraction)
            fg_fade = fg_motion.fade
            if fg_fade == "none" and not scene.independent_layer_motion:
                fg_fade = motion.fade
            fade_keyframes(person, draft, fg_fade, end_us)
            script.add_segment(person, track="foreground")
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
    parser.add_argument("--parallax-placement", choices=sorted(PARALLAX_PLACEMENTS), default="none")
    parser.add_argument("--parallax-scenes", default="", help="手动指定分层视差场景序号，从 1 开始")
    parser.add_argument("--parallax-background-effects", type=json.loads, default=PARALLAX_BACKGROUND_EFFECTS)
    parser.add_argument("--parallax-foreground-effects", type=json.loads, default=PARALLAX_FOREGROUND_EFFECTS)
    parser.add_argument("--scene-seconds", type=float, default=25.0)
    parser.add_argument("--motion-preset", choices=sorted(PRESETS), default="static",
                        help="镜头动效：静止、大模型逐场景推荐、推拉、平移或分层视差")
    parser.add_argument("--motion-effects", default="{}", help="组合动效 JSON：fade、horizontal、vertical、zoom")
    parser.add_argument("--visual-style", default=os.getenv("VISUAL_STYLE", ""), help="LLM 场景分析使用的全局画面设定")
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--generate-images", action="store_true", help="通过 Draw Things 为场景生图")
    parser.add_argument("--images-dir", type=Path, help="复用 scene_001.png、scene_002.png 等图片")
    parser.add_argument("--drawthings-model", help="本机已下载的 Draw Things 模型文件名")
    parser.add_argument("--upscale-x2", action="store_true", help="按指定宽高生图，再用本地 Real-ESRGAN X2+ 放大到 2 倍")
    parser.add_argument("--image-steps", type=int, help="生图步数；留空使用模型推荐值")
    parser.add_argument("--image-seed", type=int, default=271829, help="场景生图使用的固定 seed，默认 271829")
    parser.add_argument("--image-prompt-suffix", default="", help="附加到每个场景生图提示词末尾的固定文本")
    parser.add_argument("--character-profile", type=Path, help="已确认的单主角角色档案目录")
    parser.add_argument("--review-scenes", action="store_true", help="已停用，仅为兼容旧命令保留")
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
        if args.generate_images and args.scene_mode == "none":
            raise ValueError("生图需要 --scene-mode fixed 或 llm")
        if args.parallax_placement != "none" and (args.scene_mode != "llm" or not args.generate_images):
            raise ValueError("指定分层视差位置需要大模型语义分段并开启 Draw Things 生图")
        manual_parallax = parse_scene_numbers(args.parallax_scenes) if args.parallax_placement == "manual" else set()
        parallax_background = normalize_effects(args.parallax_background_effects)
        parallax_foreground = normalize_effects(args.parallax_foreground_effects)
        if args.generate_images and args.upscale_x2 and not upscaler_available():
            raise ValueError("未找到本地 Real-ESRGAN X2+ 模型文件")
        if args.upscale_x2 and not args.generate_images:
            raise ValueError("使用 Real-ESRGAN X2+ 需要开启 Draw Things 生图")
        if args.motion_preset == "parallax" and (not args.generate_images or not args.character_profile):
            raise ValueError("分层视差需要开启 Draw Things 生图并选择角色档案")
        if args.motion_preset == "auto" and not args.llm_model:
            raise ValueError("自动运镜需要 --llm-model")
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
        record_llm_trace = None
        if args.scene_mode == "llm" or args.motion_preset == "auto" or (args.scene_mode == "fixed" and args.llm_model and character_name):
            trace_path = output / "llm_trace.log"
            trace_path.write_text("", encoding="utf-8")

            def record_llm_trace(message: str) -> None:
                with trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(message)
                    if not message.endswith("\n"):
                        trace_file.write("\n")
                print(message, end="" if message.endswith("\n") else "\n", flush=True)
        if args.scene_mode == "none":
            scenes = [Scene(0, duration, "")]
        elif args.scene_mode == "fixed":
            scenes = build_scenes(captions, duration, args.scene_seconds)
        else:
            scenes = analyze_scenes(captions, duration, args.llm_base_url, args.llm_model, args.llm_api_key,
                                    visual_style=args.visual_style, trace=record_llm_trace,
                                    protagonist_name=character_name)
        for scene, prompt in zip(scenes, simplify_texts([scene.prompt for scene in scenes])):
            scene.prompt = prompt
        if args.scene_mode == "fixed" and args.llm_model and character_name:
            recommend_character_presence(scenes, args.llm_base_url, args.llm_model,
                                         args.llm_api_key, character_name, record_llm_trace)
        if args.review_scenes:
            print("提示：出镜确认参数已停用；固定分段无 LLM 时默认主角不出镜。", flush=True)
        selected_effects = normalize_effects(json.loads(args.motion_effects)) if args.scene_mode != "none" else {}
        if args.motion_preset == "auto":
            write_progress(output, "推荐场景运镜", 65)
            recommend_scene_motions(scenes, args.llm_base_url, args.llm_model, args.llm_api_key,
                                    captions=captions, visual_style=args.visual_style, trace=record_llm_trace)
        else:
            for i, scene in enumerate(scenes):
                scene.motion = choose_preset(args.motion_preset, i, scene.character_present)
        forced_numbers = parallax_scene_numbers(args.parallax_placement, manual_parallax, len(scenes))
        if missing := sorted(manual_parallax - forced_numbers):
            print(f"忽略超出本次分段范围的分层视差序号：{missing}", flush=True)
        base_motions = {}
        for number in sorted(forced_numbers):
            scene = scenes[number - 1]
            base_motions[number] = scene.motion
            scene.motion = "parallax"
            scene.motion_reason = "按指定位置使用分层视差"
        for number, scene in enumerate(scenes, 1):
            if number in forced_numbers:
                scene.independent_layer_motion = True
                scene.motion_effects = parallax_background.copy()
                scene.foreground_effects = parallax_foreground.copy()
                continue
            scene.motion_effects = selected_effects.copy()
            if scene.motion == "parallax":
                scene.foreground_effects = {key: value for key, value in selected_effects.items() if key != "fade"}
        for i, scene in enumerate(scenes, 1):
            print(f"场景 {i} [{scene.start:.2f}s - {scene.end:.2f}s] 提示词：{scene.prompt}\n"
                  f"场景 {i} 运镜：{scene.motion}；原因：{scene.motion_reason}", flush=True)
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
            image_width, image_height = generation_size(args.width, args.height, scene.motion, scene.motion_effects)
            needs_layers = scene.motion == "parallax"
            if needs_layers:
                image_width, image_height = args.width, args.height
            if args.upscale_x2:
                image_width = (image_width + 63) // 64 * 64
                image_height = (image_height + 63) // 64 * 64
            fingerprint = image_request_fingerprint(final_prompt, args.drawthings_model or "", image_width,
                                                    image_height, args.image_steps, args.image_seed, references,
                                                    args.upscale_x2)
            metadata = images_dir / f"scene_{i:03}.request.json"
            record = {"scene": i, "prompt": final_prompt, "seed": args.image_seed,
                      "references": [str(path) for path in references], "model": args.drawthings_model,
                      "steps": args.image_steps, "width": image_width, "height": image_height,
                      "upscale_x2": args.upscale_x2,
                      "fingerprint": fingerprint, "status": "pending"}
            print(f"场景 {i} 最终生图提示词：{final_prompt}\nseed：{args.image_seed}；参考图：{record['references']}", flush=True)
            if args.generate_images and args.upscale_x2:
                print(f"场景 {i} 基础生图尺寸：{image_width}×{image_height}；X2+ 输出尺寸："
                      f"{image_width * 2}×{image_height * 2}", flush=True)
            if image.is_file() and (args.images_dir is not None or image_cache_matches(metadata, fingerprint)):
                scene.image = str(image)
                record["status"] = "reused"
            elif args.generate_images:
                try:
                    temporary_image = images_dir / f"scene_{i:03}.pending.png"
                    generate_image(final_prompt, temporary_image, args.drawthings_model or "", image_width, image_height,
                                   args.image_steps,
                                   progress=lambda percent: write_progress(output, f"生成图片 {i}/{len(scenes)}",
                                                                           65 + round(((i - 1) + percent / (200 if needs_layers else 100)) / len(scenes) * 20)),
                                   seed=args.image_seed, references=references, upscale_x2=args.upscale_x2)
                    temporary_image.replace(image)
                    record["status"] = "generated"
                    metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    scene.image = str(image)
                except Exception as exc:
                    print(f"警告：第 {i} 段生图失败，将使用黑色背景：{exc}", file=sys.stderr)
                    record["status"] = "failed"
                    record["error"] = str(exc)
            if needs_layers and scene.image:
                background = images_dir / f"scene_{i:03}_background.png"
                foreground = images_dir / f"scene_{i:03}_foreground.png"
                try:
                    edit_model = preferred_edit_model(list_models(include_edit=True), args.drawthings_model or "")
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    print(f"场景 {i} 无法读取编辑模型列表，使用场景模型：{exc}", file=sys.stderr, flush=True)
                    edit_model = args.drawthings_model or ""
                background_prompt = parallax_background_prompt()
                motion_width, motion_height = generation_size(args.width, args.height,
                                                              scene.background_motion_preset, scene.motion_effects)
                background_width = max(image_width, motion_width)
                background_height = max(image_height, motion_height)
                source_image = Path(scene.image)
                background_fingerprint = image_request_fingerprint(background_prompt, edit_model,
                                                                    background_width, background_height, args.image_steps,
                                                                    args.image_seed, [source_image], args.upscale_x2)
                background_metadata = images_dir / f"scene_{i:03}_background.request.json"
                try:
                    print(f"场景 {i} 空背景提示词：{background_prompt}\n参考图：{source_image}；模型：{edit_model}", flush=True)
                    if not background.is_file() or not image_cache_matches(background_metadata, background_fingerprint):
                        pending_background = images_dir / f"scene_{i:03}_background.pending.png"
                        generate_image(background_prompt, pending_background, edit_model,
                                       background_width, background_height, args.image_steps, seed=args.image_seed,
                                       references=[source_image], reference_mode="instruction", upscale_x2=args.upscale_x2,
                                       progress=lambda percent: write_progress(output, f"生成空背景 {i}/{len(scenes)}",
                                                                               65 + round(((i - 1) + 0.5 + percent / 200) / len(scenes) * 20)))
                        pending_background.replace(background)
                        background_metadata.write_text(json.dumps({"fingerprint": background_fingerprint,
                                                                   "prompt": background_prompt,
                                                                   "model": edit_model, "width": background_width,
                                                                   "height": background_height, "steps": args.image_steps,
                                                                   "seed": args.image_seed, "references": [str(source_image)],
                                                                   "reference_mode": "instruction", "method": "source_edit",
                                                                   "upscale_x2": args.upscale_x2}, ensure_ascii=False,
                                                                  indent=2), encoding="utf-8")
                    foreground_text = parallax_foreground_prompt()
                    foreground_fingerprint = image_request_fingerprint(foreground_text, edit_model,
                                                                        image_width, image_height, args.image_steps,
                                                                        args.image_seed, [source_image], args.upscale_x2)
                    foreground_metadata = images_dir / f"scene_{i:03}_foreground.request.json"
                    if not foreground.is_file() or not image_cache_matches(foreground_metadata, foreground_fingerprint):
                        pending_foreground = images_dir / f"scene_{i:03}_foreground.pending.png"
                        print(f"场景 {i} AI 前景提示词：{foreground_text}\n参考图：{source_image}；模型：{edit_model}", flush=True)
                        generate_image(foreground_text, pending_foreground, edit_model, image_width, image_height,
                                       args.image_steps, seed=args.image_seed, references=[source_image],
                                       reference_mode="instruction", upscale_x2=args.upscale_x2)
                        validate_foreground(pending_foreground)
                        masked_foreground = images_dir / f"scene_{i:03}_foreground.masked.png"
                        apply_foreground_mask(source_image, pending_foreground, masked_foreground)
                        masked_foreground.replace(pending_foreground)
                        pending_foreground.replace(foreground)
                        foreground_metadata.write_text(json.dumps({"fingerprint": foreground_fingerprint,
                                                                    "prompt": foreground_text, "model": edit_model,
                                                                    "width": image_width, "height": image_height,
                                                                    "steps": args.image_steps, "seed": args.image_seed,
                                                                    "references": [str(source_image)], "method": "ai",
                                                                    "reference_mode": "instruction", "upscale_x2": args.upscale_x2,
                                                                    "mask_from_source": True}, ensure_ascii=False,
                                                                   indent=2), encoding="utf-8")
                    validate_foreground(foreground)
                    scene.background, scene.foreground = str(background), str(foreground)
                    scene.foreground_method = "ai"
                    record["foreground_method"] = "ai"
                    record["layer_status"] = "ready"
                    print(f"场景 {i} 已生成分层素材：{background.name}、{foreground.name}", flush=True)
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    record["layer_status"] = "failed"
                    record["layer_error"] = str(exc)
                    print(f"警告：第 {i} 段分层素材失败，改用单图动效：{exc}", file=sys.stderr)
            if needs_layers and not scene.foreground:
                scene.motion = choose_preset(base_motions.get(i, "parallax"), i - 1, False)
                scene.motion_effects = selected_effects.copy()
                scene.foreground_effects = {}
                scene.independent_layer_motion = False
                scene.motion_reason = "分层素材不可用，改用单图运镜"
            if not scene.image and (scene.motion != "static" or scene.motion_effects):
                scene.motion = "static"
                scene.motion_effects = {}
                scene.motion_reason = "无图片可供运镜，使用黑色静止画面"
                print(f"场景 {i} 无图片可供运镜，改用黑色静止画面。", flush=True)
            record["motion"] = scene.motion
            image_records.append(record)
        (output / "image_generation.json").write_text(json.dumps(image_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "scenes.json").write_text(json.dumps([asdict(s) for s in scenes], ensure_ascii=False, indent=2), encoding="utf-8")
        canvas_width = args.width * (2 if args.upscale_x2 else 1)
        canvas_height = args.height * (2 if args.upscale_x2 else 1)
        if args.output in ("mp4", "both"):
            write_progress(output, "合成视频", 88)
            render_mp4(audio, scenes, srt, output / "video.mp4", canvas_width, canvas_height, args.fps,
                       progress=lambda fraction: write_progress(output, "合成视频", 88 + round(fraction * 5)))
        if args.output in ("jianying", "both"):
            write_progress(output, "生成剪映草稿", 94)
            draft_path = output / (args.draft_name or "jianying_draft")
            draft_file = export_jianying(audio, scenes, srt, draft_path, canvas_width, canvas_height)
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
