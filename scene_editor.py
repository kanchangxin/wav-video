"""Versioned scene artwork edits with a single background Draw Things worker."""

from __future__ import annotations

import json
import queue
import shlex
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from audio_video import Caption, Scene, compose_scene_prompt, export_jianying, render_mp4, validate_foreground, write_srt
from drawthings_cli import X2_CONFIG, cli_path, generate_image, list_models, upscaler_available
from jianying_register import register_draft
from scene_motion import PRESETS, generation_size, normalize_effects


LAYER_KEYS = {"image": "image", "background": "background", "foreground": "foreground"}
IMAGE_FLAGS = {"--model", "--prompt", "--width", "--height", "--output", "--steps", "--seed", "--image", "--config-json"}
SWITCHES = {"--local", "--offline", "--no-download-missing", "--disable-preview"}


class SceneEditor:
    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.RLock()
        self.pending: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker = threading.Thread(target=self._work, daemon=True, name="scene-image-worker")
        self.worker.start()
        for folder in self.manager.runs.glob("*/edits/tasks/*.json"):
            try:
                task = json.loads(folder.read_text(encoding="utf-8"))
                if task.get("status") in ("queued", "running"):
                    task["status"] = "queued"
                    folder.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
                    self.pending.put((folder.parents[2].name, task["id"]))
            except (OSError, ValueError, KeyError):
                continue

    def _folder(self, job_id: str) -> Path:
        job = self.manager.get(job_id)
        if job["status"] != "completed":
            raise ValueError("请等待项目完成后再编辑场景")
        folder = Path(job["output_dir"])
        if not (folder / "scenes.json").is_file():
            raise ValueError("项目没有场景数据")
        return folder

    def _scenes(self, folder: Path) -> list[dict]:
        return json.loads((folder / "scenes.json").read_text(encoding="utf-8"))

    def _scene(self, folder: Path, index: int) -> dict:
        scenes = self._scenes(folder)
        if not isinstance(index, int) or not 1 <= index <= len(scenes):
            raise ValueError("场景编号无效")
        return scenes[index - 1]

    def _task_path(self, folder: Path, task_id: str) -> Path:
        if not task_id or any(ch not in "0123456789abcdef" for ch in task_id):
            raise ValueError("候选任务编号无效")
        return folder / "edits" / "tasks" / f"{task_id}.json"

    def _save_task(self, folder: Path, task: dict) -> None:
        path = self._task_path(folder, task["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def task(self, job_id: str, task_id: str) -> dict:
        folder = self._folder(job_id)
        path = self._task_path(folder, task_id)
        if not path.is_file():
            raise KeyError(task_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def has_active(self, job_id: str) -> bool:
        folder = self.manager.runs / job_id / "edits" / "tasks"
        return any(json.loads(path.read_text(encoding="utf-8")).get("status") in ("queued", "running")
                   for path in folder.glob("*.json")) if folder.is_dir() else False

    def _record(self, folder: Path, index: int) -> dict:
        path = folder / "image_generation.json"
        if path.is_file():
            records = json.loads(path.read_text(encoding="utf-8"))
            if len(records) >= index:
                return records[index - 1]
        return {}

    def _params(self, folder: Path, index: int, layer: str, *,
                scene: dict | None = None, record: dict | None = None,
                options: dict | None = None) -> dict:
        scene = scene if scene is not None else self._scene(folder, index)
        record = record if record is not None else self._record(folder, index)
        options = options if options is not None else self.manager.get(folder.name).get("options", {})
        background_record = {}
        if layer == "background":
            metadata = folder / "images" / f"scene_{index:03}_background.request.json"
            background_record = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
            prompt = background_record.get("prompt") or scene["prompt"] + "\nEmpty environment, no people, no text."
            references = background_record.get("references") or []
            width = int(background_record.get("width") or record.get("width") or options.get("width") or 1280)
            height = int(background_record.get("height") or record.get("height") or options.get("height") or 720)
            existing = scene.get("background")
            if existing and Path(existing).is_file() and not background_record.get("width"):
                size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                       "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", existing],
                                      capture_output=True, text=True, timeout=10)
                if size.returncode == 0:
                    try:
                        width, height = map(int, size.stdout.strip().split("x"))
                    except ValueError:
                        pass
        else:
            prompt = record.get("prompt") or scene["prompt"]
            references = record.get("references") or []
            width = int(record.get("width") or options.get("width") or 1280)
            height = int(record.get("height") or options.get("height") or 720)
            if scene.get("character_present") and not references and not record.get("references_customized"):
                snapshot = folder / "character_reference"
                refs = [snapshot / name for name in ("front.png", "three_quarter.png", "full_body.png")]
                references = [str(path) for path in refs if path.is_file()]
                profile = snapshot / "profile.json"
                if references and profile.is_file():
                    name = json.loads(profile.read_text(encoding="utf-8")).get("name", "主角")
                    prompt = compose_scene_prompt(scene["prompt"], str(options.get("image_prompt_suffix") or ""),
                                                  name, True, True)
        source_record = background_record or record
        upscale_x2 = bool(source_record.get("upscale_x2", options.get("upscale_x2", False)))
        if layer != "foreground" and (layer == "background" or scene.get("motion") != "parallax"):
            needed_width, needed_height = generation_size(int(options.get("width") or 1280),
                                                           int(options.get("height") or 720),
                                                           scene.get("motion") or "static",
                                                           scene.get("motion_effects") or {})
            width, height = max(width, needed_width), max(height, needed_height)
        if upscale_x2:
            width, height = (width + 63) // 64 * 64, (height + 63) // 64 * 64
        else:
            width, height = max(64, round(width / 64) * 64), max(64, round(height / 64) * 64)
        target = scene.get("image" if layer == "foreground" else layer)
        if layer == "foreground":
            target = scene.get("image")
        if not target:
            target = str(folder / "images" / f"scene_{index:03}{'_background' if layer == 'background' else ''}.png")
        return {"prompt": prompt, "model": source_record.get("model") or record.get("model") or options.get("drawthings_model") or "",
                "width": width, "height": height, "steps": source_record.get("steps") or record.get("steps") or options.get("image_steps"),
                "seed": source_record.get("seed") if source_record.get("seed") is not None else
                (record.get("seed") if record.get("seed") is not None else options.get("image_seed", 271829)),
                "references": references, "references_customized": bool(source_record.get("references_customized")),
                "output": target, "upscale_x2": upscale_x2}

    def _command(self, params: dict) -> str:
        args = [str(cli_path()), "generate", "--local", "--offline", "--no-download-missing",
                "--disable-preview", "--model", str(params["model"]), "--prompt", str(params["prompt"]),
                "--width", str(params["width"]), "--height", str(params["height"]),
                "--output", str(params["output"])]
        if params.get("steps") not in (None, ""):
            args += ["--steps", str(params["steps"])]
        if params.get("seed") not in (None, ""):
            args += ["--seed", str(params["seed"])]
        if params.get("upscale_x2"):
            args += ["--config-json", json.dumps(X2_CONFIG, separators=(",", ":"))]
        for path in params.get("references") or []:
            args += ["--image", str(path)]
        return shlex.join(args)

    def _parse_command(self, text: str, original: dict, *, validate_references: bool = True) -> dict:
        if len(text) > 20_000:
            raise ValueError("Draw Things 命令不能超过 20000 个字符")
        try:
            tokens = shlex.split(text)
        except ValueError as exc:
            raise ValueError("Draw Things 命令引号不完整") from exc
        if len(tokens) < 3 or tokens[:2] != [str(cli_path()), "generate"]:
            raise ValueError("只能编辑当前 Draw Things generate 命令")
        found: dict[str, str | list[str]] = {}
        i = 2
        while i < len(tokens):
            flag = tokens[i]
            if flag in SWITCHES:
                if flag in found:
                    raise ValueError("命令参数重复")
                found[flag] = "true"
                i += 1
                continue
            if flag not in IMAGE_FLAGS or i + 1 >= len(tokens):
                raise ValueError(f"不支持的命令参数：{flag}")
            if flag == "--image":
                found.setdefault(flag, [])
                found[flag].append(tokens[i + 1])
            elif flag in found:
                raise ValueError("命令参数重复")
            else:
                found[flag] = tokens[i + 1]
            i += 2
        if found.get("--output") != original["output"]:
            raise ValueError("输出路径不能修改；候选图片由系统另存")
        references = (self._validate_references(found.get("--image", [])) if validate_references else
                      found.get("--image", []))
        try:
            width, height = int(found["--width"]), int(found["--height"])
            seed = int(found["--seed"]) if "--seed" in found else None
            steps = int(found["--steps"]) if "--steps" in found else None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("宽高、步数或 seed 无效") from exc
        if not (64 <= width <= 4096 and 64 <= height <= 4096 and width % 64 == 0 and height % 64 == 0):
            raise ValueError("生图宽高必须是 64 的倍数，范围 64–4096")
        if steps is not None and not 1 <= steps <= 150:
            raise ValueError("生图步数须在 1–150")
        if seed is not None and not 0 <= seed <= 4294967295:
            raise ValueError("seed 超出范围")
        model = str(found.get("--model") or "")
        if not model or not isinstance(found.get("--prompt"), str) or not found["--prompt"].strip():
            raise ValueError("模型和提示词不能为空")
        if len(found["--prompt"]) > 4000:
            raise ValueError("提示词不能超过 4000 个字符")
        upscale_x2 = False
        if "--config-json" in found:
            try:
                upscale_x2 = json.loads(str(found["--config-json"])) == X2_CONFIG
            except ValueError as exc:
                raise ValueError("放大配置不是有效 JSON") from exc
            if not upscale_x2:
                raise ValueError("只支持 Real-ESRGAN X2+ 放大配置")
            if not upscaler_available():
                raise ValueError("未找到本地 Real-ESRGAN X2+ 模型文件")
        if model != original["model"] and model not in {item["id"] for item in list_models()}:
            raise ValueError("请选择本机已下载的 Draw Things 模型")
        return {"prompt": found["--prompt"], "model": model, "width": width, "height": height,
                "steps": steps, "seed": seed, "references": references, "output": original["output"],
                "upscale_x2": upscale_x2}

    def _validate_references(self, references: list[str]) -> list[str]:
        if not isinstance(references, list) or len(references) > 8 or any(
                not isinstance(path, str) or not path.strip() for path in references):
            raise ValueError("参考图路径须为图片列表，最多 8 张")
        checked = []
        for raw in references:
            path = Path(raw.strip()).expanduser()
            if not path.is_absolute():
                path = self.manager.root / path
            path = path.resolve()
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or not path.is_file():
                raise ValueError(f"参考图不存在或格式不支持：{raw}")
            try:
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                                       capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError(f"无法读取参考图：{raw}") from exc
            if probe.returncode or not probe.stdout.strip():
                raise ValueError(f"参考图不是有效图片：{raw}")
            checked.append(str(path))
        return checked

    def inventory(self, job_id: str) -> dict:
        folder = self._folder(job_id)
        scenes = self._scenes(folder)
        transcript_path = folder / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8")) if transcript_path.is_file() else {}
        captions = transcript.get("captions", []) if isinstance(transcript, dict) else []
        record_path = folder / "image_generation.json"
        records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else []
        options = self.manager.get(job_id).get("options", {})
        result = []
        for i, scene in enumerate(scenes, 1):
            layers = []
            for layer in LAYER_KEYS:
                if layer != "image" and not (scene.get(layer) or scene.get("motion") == "parallax"):
                    continue
                path = scene.get(layer)
                params = self._params(folder, i, layer, scene=scene,
                                      record=records[i - 1] if i <= len(records) else {}, options=options)
                local = bool(path and Path(path).resolve().is_relative_to(folder.resolve()))
                cutout_command = None
                if layer == "foreground":
                    tool = Path(__file__).resolve().parent / "tools" / "drawthings" / "person_cutout"
                    cutout_command = shlex.join([str(tool), params["output"],
                                                  str(path or folder / "images" / f"scene_{i:03}_foreground.png")])
                layers.append({"name": layer, "url": f"/api/jobs/{job_id}/editor/assets/{i}/{layer}" if local else None,
                               "command": self._command(params), "source": path,
                               "references": params["references"],
                               "cutout_command": cutout_command,
                               "derived": layer == "foreground",
                               "upscale_x2": params["upscale_x2"],
                               "motion_effects": scene.get("foreground_effects", {}) if layer == "foreground" else scene.get("motion_effects", {})})
            result.append({"index": i, "start": scene["start"], "end": scene["end"],
                           "prompt": scene["prompt"], "character_present": scene.get("character_present", False),
                           "motion": scene.get("motion", "static"), "motion_reason": scene.get("motion_reason", ""),
                           "motion_effects": scene.get("motion_effects", {}),
                           "foreground_effects": scene.get("foreground_effects", {}),
                           "captions": [{"index": j, **caption} for j, caption in enumerate(captions)
                                        if scene["start"] <= float(caption["start"]) < scene["end"]],
                           "layers": layers})
        return {"scenes": result, **self.tasks(job_id)}

    def update_captions(self, job_id: str, index: int, edits: list[dict]) -> dict:
        folder = self._folder(job_id)
        with self.lock:
            scene = self._scene(folder, index)
            path = folder / "transcript.json"
            if not path.is_file():
                raise ValueError("此任务没有可编辑的字幕 JSON")
            transcript = json.loads(path.read_text(encoding="utf-8"))
            captions = transcript.get("captions", [])
            if not isinstance(edits, list) or not edits:
                raise ValueError("请选择要保存的字幕")
            changed = set()
            for edit in edits:
                if not isinstance(edit, dict) or type(edit.get("index")) is not int:
                    raise ValueError("字幕编号无效")
                position = edit["index"]
                if position in changed or not 0 <= position < len(captions):
                    raise ValueError("字幕编号无效或重复")
                caption = captions[position]
                if not scene["start"] <= float(caption["start"]) < scene["end"]:
                    raise ValueError("字幕不属于当前场景")
                value = edit.get("text")
                if not isinstance(value, str) or not value.strip() or len(value) > 500 or "\n" in value or "\r" in value:
                    raise ValueError("字幕文字须为单行且不超过 500 字")
                changed.add(position)
            for edit in edits:
                captions[edit["index"]]["text"] = edit["text"].strip()
            prepared = [Caption(float(item["start"]), float(item["end"]), item["text"]) for item in captions]
            json_pending = path.with_suffix(".json.tmp")
            srt = folder / "subtitles.srt"
            srt_pending = srt.with_suffix(".srt.tmp")
            text = folder / "transcript.txt"
            text_pending = text.with_suffix(".txt.tmp")
            json_pending.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
            write_srt(prepared, srt_pending)
            text_pending.write_text("".join(item.text for item in prepared), encoding="utf-8")
            json_pending.replace(path)
            srt_pending.replace(srt)
            text_pending.replace(text)
            return {"saved": True, "rebuild": self._queue_rebuild(job_id)}

    def tasks(self, job_id: str) -> dict:
        folder = self._folder(job_id)
        tasks_dir = folder / "edits" / "tasks"
        tasks = [json.loads(p.read_text(encoding="utf-8")) for p in tasks_dir.glob("*.json")] if tasks_dir.is_dir() else []
        tasks.sort(key=lambda task: task.get("created", ""), reverse=True)
        job = self.manager.get(job_id)
        active = [task for task in tasks if task.get("status") in ("queued", "running", "ready", "failed")]
        recent = [task for task in tasks if task.get("status") not in ("queued", "running", "ready", "failed")][:100]
        visible = active + recent
        visible.sort(key=lambda task: task.get("created", ""), reverse=True)
        return {"tasks": visible, "registered_draft": job.get("registered_draft"),
                "needs_sync": bool(job.get("registered_draft") and (folder / "edits" / "needs_sync").is_file())}

    def asset(self, job_id: str, index: int, layer: str) -> Path:
        folder = self._folder(job_id)
        if layer not in LAYER_KEYS:
            raise ValueError("图层无效")
        path = self._scene(folder, index).get(layer)
        if not path:
            raise KeyError(layer)
        resolved = Path(path).resolve()
        if not resolved.is_file() or not resolved.is_relative_to(folder.resolve()):
            raise KeyError(layer)
        return resolved

    def _new_task(self, job_id: str, index: int, layer: str, kind: str, enqueue: bool = True, **fields) -> dict:
        folder = self._folder(job_id)
        self._scene(folder, index)
        if layer not in LAYER_KEYS:
            raise ValueError("图层无效")
        task_id = uuid.uuid4().hex[:16]
        candidate = folder / "edits" / "candidates" / f"{task_id}.png"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        task = {"id": task_id, "scene": index, "layer": layer, "kind": kind, "status": "queued",
                "candidate": str(candidate), "url": f"/api/jobs/{job_id}/editor/candidates/{task_id}/image",
                "created": datetime.now(timezone.utc).isoformat(), **fields}
        with self.lock:
            self._save_task(folder, task)
            if enqueue:
                self.pending.put((job_id, task_id))
        return task

    def generate(self, job_id: str, index: int, layer: str, command: str,
                 references: list[str] | None = None) -> dict:
        folder = self._folder(job_id)
        if layer not in LAYER_KEYS:
            raise ValueError("图层无效")
        original = self._params(folder, index, layer)
        params = self._parse_command(command, original, validate_references=references is None)
        changed_in_command = params["references"] != original["references"]
        if references is not None:
            params["references"] = self._validate_references(references)
        params["references_customized"] = bool(original["references_customized"] or
                                                changed_in_command or references is not None)
        return self._new_task(job_id, index, layer, "generate", params=params)

    def upload(self, job_id: str, index: int, layer: str, stream: BinaryIO, filename: str) -> dict:
        if Path(filename).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("请上传 PNG、JPEG 或 WebP 图片")
        folder = self._folder(job_id)
        task = self._new_task(job_id, index, layer, "upload", enqueue=False)
        path = Path(task["candidate"])
        total = 0
        try:
            with path.open("wb") as target:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > 40 * 1024 * 1024:
                        raise ValueError("单张图片不能超过 40 MB")
                    target.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            with self.lock:
                task["status"] = "failed"
                task["error"] = "图片上传失败或超过 40 MB"
                self._save_task(folder, task)
            raise
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True, timeout=20)
        if probe.returncode or not probe.stdout.strip():
            path.unlink(missing_ok=True)
            with self.lock:
                task["status"] = "failed"
                task["error"] = "上传内容不是有效图片"
                self._save_task(folder, task)
            raise ValueError("上传内容不是有效图片")
        try:
            width, height = map(int, probe.stdout.strip().replace(",", "x").split("x"))
        except ValueError as exc:
            raise ValueError("无法读取上传图片尺寸") from exc
        if not 64 <= width <= 8192 or not 64 <= height <= 8192:
            path.unlink(missing_ok=True)
            with self.lock:
                task["status"] = "failed"
                task["error"] = "上传图片尺寸须在 64–8192 像素"
                self._save_task(folder, task)
            raise ValueError("上传图片尺寸须在 64–8192 像素")
        with self.lock:
            params = self._params(folder, index, layer)
            params["width"], params["height"] = width, height
            task["params"] = params
            self._save_task(folder, task)
        self.pending.put((job_id, task["id"]))
        return task

    def _work(self) -> None:
        while True:
            job_id, task_id = self.pending.get()
            try:
                folder = self._folder(job_id)
                with self.lock:
                    task = self.task(job_id, task_id)
                    if task["status"] != "queued":
                        continue
                    task["status"] = "running"
                    self._save_task(folder, task)
                if task["kind"] == "generate":
                    params = task["params"]
                    target = Path(task["candidate"])
                    source = target.with_name(target.stem + "_source.png") if task["layer"] == "foreground" else target
                    generate_image(params["prompt"], source, params["model"], params["width"], params["height"],
                                   params["steps"], seed=params["seed"],
                                   references=[Path(path) for path in params["references"]],
                                   upscale_x2=params["upscale_x2"],
                                   progress=lambda percent: self._set_progress(job_id, task_id, percent))
                if task["kind"] == "upload":
                    candidate = Path(task["candidate"])
                    codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                                     "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
                                                     str(candidate)], text=True, timeout=20).strip()
                    if codec != "png":
                        converted = candidate.with_name(candidate.stem + "_converted.png")
                        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(candidate),
                                        "-frames:v", "1", str(converted)], check=True, timeout=60)
                        converted.replace(candidate)
                if task["layer"] == "foreground" and task["kind"] == "upload":
                    validate_foreground(Path(task["candidate"]))
                elif task["kind"] in ("generate", "upload") and (task["layer"] == "foreground" or (task["layer"] == "image" and
                        (self._scene(folder, task["scene"]).get("foreground") or
                         self._scene(folder, task["scene"]).get("motion") == "parallax"))):
                    source = (Path(task["candidate"]).with_name(Path(task["candidate"]).stem + "_source.png")
                              if task["layer"] == "foreground" and task["kind"] == "generate" else Path(task["candidate"]))
                    cutout = (Path(task["candidate"]) if task["layer"] == "foreground" else
                              Path(task["candidate"]).with_name(Path(task["candidate"]).stem + "_cutout.png"))
                    tool = Path(__file__).resolve().parent / "tools" / "drawthings" / "person_cutout"
                    subprocess.run([str(tool), str(source), str(cutout)], check=True, timeout=300)
                    validate_foreground(cutout)
                if task["kind"] == "rebuild":
                    self._rebuild(job_id, task_id)
                with self.lock:
                    task["status"] = "completed" if task["kind"] == "rebuild" else "ready"
                    self._save_task(folder, task)
            except Exception as exc:
                try:
                    with self.lock:
                        task = self.task(job_id, task_id)
                        task["status"] = "failed"
                        task["error"] = str(exc)
                        self._save_task(self._folder(job_id), task)
                except Exception:
                    pass
            finally:
                self.pending.task_done()

    def _set_progress(self, job_id: str, task_id: str, percent: int) -> None:
        with self.lock:
            task = self.task(job_id, task_id)
            if task["status"] == "running":
                task["progress"] = max(0, min(100, int(percent)))
                self._save_task(self._folder(job_id), task)

    def _replace(self, folder: Path, original: Path, candidate: Path) -> None:
        if not candidate.is_file():
            raise ValueError("候选图片尚未准备好")
        revision = folder / "edits" / "revisions" / f"{original.stem}_{uuid.uuid4().hex[:10]}{original.suffix}"
        revision.parent.mkdir(parents=True, exist_ok=True)
        if original.is_file():
            shutil.copy2(original, revision)
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, original.with_suffix(original.suffix + ".new"))
        original.with_suffix(original.suffix + ".new").replace(original)

    def accept(self, job_id: str, task_id: str) -> dict:
        folder = self._folder(job_id)
        with self.lock:
            task = self.task(job_id, task_id)
            if task["status"] != "ready":
                raise ValueError("候选图片尚未生成或已处理")
            scenes = self._scenes(folder)
            scene = scenes[task["scene"] - 1]
            layer = task["layer"]
            current = scene.get(layer) or str(folder / "images" / f"scene_{task['scene']:03}{'_' + layer if layer != 'image' else ''}.png")
            original = Path(current).resolve()
            if not original.is_relative_to(folder.resolve()) or original.suffix.lower() != ".png":
                original = folder / "images" / f"scene_{task['scene']:03}{'_' + layer if layer != 'image' else ''}.png"
            self._replace(folder, original, Path(task["candidate"]))
            scene[layer] = str(original)
            if layer == "foreground" and task["kind"] == "generate":
                source = Path(task["candidate"]).with_name(Path(task["candidate"]).stem + "_source.png")
                if scene.get("image"):
                    source_target = Path(scene["image"]).resolve()
                    if not source_target.is_relative_to(folder.resolve()) or source_target.suffix.lower() != ".png":
                        source_target = folder / "images" / f"scene_{task['scene']:03}.png"
                    self._replace(folder, source_target, source)
                    scene["image"] = str(source_target)
            if layer == "image" and (scene.get("foreground") or scene.get("motion") == "parallax"):
                cutout = Path(task["candidate"]).with_name(Path(task["candidate"]).stem + "_cutout.png")
                foreground_target = Path(scene["foreground"]).resolve() if scene.get("foreground") else folder / "images" / f"scene_{task['scene']:03}_foreground.png"
                if not foreground_target.is_relative_to(folder.resolve()) or foreground_target.suffix.lower() != ".png":
                    foreground_target = folder / "images" / f"scene_{task['scene']:03}_foreground.png"
                self._replace(folder, foreground_target, cutout)
                scene["foreground"] = str(foreground_target)
            (folder / "scenes.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
            if task.get("params"):
                record_path = folder / "image_generation.json"
                records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else []
                if len(records) >= task["scene"] and (layer == "image" or (layer == "foreground" and task["kind"] == "generate")):
                    records[task["scene"] - 1].update({key: value for key, value in task["params"].items() if key != "output"})
                    record_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                elif layer == "background":
                    metadata = folder / "images" / f"scene_{task['scene']:03}_background.request.json"
                    metadata.write_text(json.dumps(task["params"], ensure_ascii=False, indent=2), encoding="utf-8")
            task["status"] = "accepted"
            self._save_task(folder, task)
            return {"accepted": True, "rebuild": self._queue_rebuild(job_id)}

    def discard(self, job_id: str, task_id: str) -> dict:
        folder = self._folder(job_id)
        with self.lock:
            task = self.task(job_id, task_id)
            if task["status"] not in ("ready", "failed"):
                raise ValueError("当前候选图片不能舍弃")
            Path(task["candidate"]).unlink(missing_ok=True)
            task["status"] = "discarded"
            self._save_task(folder, task)
            return task

    def update_motion(self, job_id: str, index: int, layer: str, effects: dict, motion: str | None = None,
                      character_present: bool | None = None) -> dict:
        folder = self._folder(job_id)
        with self.lock:
            scenes = self._scenes(folder)
            if not 1 <= index <= len(scenes) or layer not in LAYER_KEYS:
                raise ValueError("场景或图层无效")
            scene = scenes[index - 1]
            clean = normalize_effects(effects)
            scene["foreground_effects" if layer == "foreground" else "motion_effects"] = clean
            if motion is not None:
                if motion not in PRESETS or motion == "auto":
                    raise ValueError("场景运镜无效")
                scene["motion"] = motion
            if character_present is not None:
                if not isinstance(character_present, bool):
                    raise ValueError("主角出镜标记无效")
                scene["character_present"] = character_present
            (folder / "scenes.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"saved": True, "rebuild": self._queue_rebuild(job_id)}

    def _queue_rebuild(self, job_id: str) -> dict | None:
        folder = self._folder(job_id)
        output = self.manager.get(job_id)["options"].get("output")
        if output not in ("mp4", "jianying", "both"):
            return None
        if self.manager.get(job_id).get("registered_draft"):
            marker = folder / "edits" / "needs_sync"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("1")
        return self._new_task(job_id, 1, "image", "rebuild")

    def rebuild(self, job_id: str) -> dict:
        with self.lock:
            task = self._queue_rebuild(job_id)
            if task is None:
                raise ValueError("此项目没有需要重建的 MP4 或剪映草稿")
            return task

    def _rebuild(self, job_id: str, task_id: str) -> None:
        folder = self._folder(job_id)
        job = self.manager.get(job_id)
        options = job["options"]
        scenes = [Scene(**scene) for scene in self._scenes(folder)]
        audio = self.manager.resolve_audio(job["audio_id"])
        srt = folder / "subtitles.srt"
        width, height, fps = int(options.get("width", 1280)), int(options.get("height", 720)), int(options.get("fps", 25))
        if options.get("upscale_x2"):
            width, height = width * 2, height * 2
        pending = folder / f"video_{task_id}.mp4"
        draft = folder / f"jianying_draft_edit_{task_id}"
        try:
            if options.get("output") in ("mp4", "both"):
                render_mp4(audio, scenes, srt, pending, width, height, fps)
            if options.get("output") in ("jianying", "both"):
                export_jianying(audio, scenes, srt, draft, width, height)
            if options.get("output") in ("mp4", "both"):
                pending.replace(folder / "video.mp4")
            if options.get("output") in ("jianying", "both"):
                pointer = folder / "jianying_draft_path.txt"
                temporary = pointer.with_suffix(".txt.tmp")
                temporary.write_text(str(draft) + "\n", encoding="utf-8")
                temporary.replace(pointer)
        except Exception:
            pending.unlink(missing_ok=True)
            if draft.is_dir():
                shutil.rmtree(draft)
            raise

    def sync_jianying(self, job_id: str) -> dict:
        folder = self._folder(job_id)
        with self.lock:
            job = self.manager.get(job_id)
            if not job.get("registered_draft"):
                raise ValueError("该项目尚未注册到剪映")
            marker = folder / "edits" / "needs_sync"
            if not marker.is_file():
                raise ValueError("没有待同步的剪映修改")
            tasks_dir = folder / "edits" / "tasks"
            rebuilds = [item for path in tasks_dir.glob("*.json")
                        if (item := json.loads(path.read_text(encoding="utf-8"))).get("kind") == "rebuild"]
            latest = max(rebuilds, key=lambda item: item.get("created", "")) if rebuilds else None
            if latest is None or latest.get("status") != "completed":
                raise ValueError("请等待剪映草稿重建成功后再同步")
            source = Path((folder / "jianying_draft_path.txt").read_text(encoding="utf-8").strip())
            if not source.resolve().is_relative_to(folder.resolve()) or not (source / "draft_content.json").is_file():
                raise ValueError("请等待剪映草稿重建完成")
            previous = Path(job["registered_draft"])
            name = str(job["options"].get("draft_name") or Path(job["audio_name"]).stem)
            destination = register_draft(source, previous.parent, name)
            with self.manager.lock:
                self.manager.jobs[job_id]["registered_draft"] = str(destination)
                self.manager._persist(self.manager.jobs[job_id])
            marker.unlink(missing_ok=True)
            return {"registered_draft": str(destination)}
