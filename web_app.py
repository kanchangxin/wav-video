#!/usr/bin/env python3
"""Local browser UI for audio_video.py. Run with: uv run python web_app.py"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, jsonify, render_template, request, send_from_directory

from jianying_register import register_draft, registration_status
from drawthings_cli import list_models


SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".mp4"}
OUTPUT_TYPES = {"mp4", "jianying", "both", "assets"}
SCENE_MODES = {"none", "fixed", "llm"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_audio_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ]", "_", name, flags=re.UNICODE).strip(" .")
    if not name or Path(name).suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError("请选择 m4a、mp3、wav、aac、flac、ogg 或 mp4 音频")
    return name


class JobManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.uploads = self.root / "uploads"
        self.runs = self.root / "runs"
        self.uploads.mkdir(exist_ok=True)
        self.runs.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        for metadata in self.runs.glob("*/job.json"):
            try:
                state = json.loads(metadata.read_text(encoding="utf-8"))
                if state.get("status") == "running":
                    state["status"] = "interrupted"
                self.jobs[state["id"]] = state
            except (OSError, ValueError, KeyError):
                continue

    def audio_files(self) -> list[dict[str, str]]:
        result = []
        for prefix, folder in (("root", self.root), ("upload", self.uploads)):
            for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                    result.append({"id": f"{prefix}:{path.name}", "name": path.name})
        return result

    def resolve_audio(self, audio_id: str) -> Path:
        if not isinstance(audio_id, str) or ":" not in audio_id:
            raise ValueError("请选择音频文件")
        prefix, name = audio_id.split(":", 1)
        if prefix not in ("root", "upload") or name != Path(name).name or "\\" in name:
            raise ValueError("音频文件标识无效")
        folder = self.root if prefix == "root" else self.uploads
        path = folder / name
        if path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("音频文件不存在")
        return path

    def _command(self, data: dict, output: Path) -> tuple[list[str], dict[str, str], dict]:
        audio_id = data.get("audio_id", "")
        audio = self.resolve_audio(audio_id)
        output_type = data.get("output", "mp4")
        scene_mode = data.get("scene_mode", "none")
        if output_type not in OUTPUT_TYPES or scene_mode not in SCENE_MODES:
            raise ValueError("输出类型或场景模式无效")
        generate_images = data.get("generate_images", False)
        if not isinstance(generate_images, bool) or generate_images and scene_mode == "none":
            raise ValueError("生图需要选择定时或大模型分段")
        register_jianying = data.get("register_jianying", False)
        if not isinstance(register_jianying, bool) or register_jianying and output_type not in ("jianying", "both"):
            raise ValueError("注册剪映需要选择剪映或两者输出")
        def number(key: str, default: int | float, minimum: float, maximum: float, integer: bool = False):
            try:
                value = int(data.get(key, default)) if integer else float(data.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} 不是有效数字") from exc
            if not minimum <= value <= maximum:
                raise ValueError(f"{key} 超出范围 {minimum}–{maximum}")
            return value
        width = number("width", 1280, 160, 4096, True)
        height = number("height", 720, 160, 4096, True)
        if width % 2 or height % 2:
            raise ValueError("宽高必须是偶数")
        fps = number("fps", 25, 1, 60, True)
        scene_seconds = number("scene_seconds", 25, 1, 600)
        image_steps = data.get("image_steps")
        if image_steps not in (None, ""):
            image_steps = number("image_steps", 20, 1, 150, True)
        else:
            image_steps = None
        max_seconds = data.get("max_seconds")
        if max_seconds not in (None, ""):
            max_seconds = number("max_seconds", 30, 1, 3600)
        else:
            max_seconds = None
        llm_model = str(data.get("llm_model", "")).strip()
        if scene_mode == "llm" and not llm_model:
            raise ValueError("大模型分段需要填写 LLM 模型名称")
        visual_style = str(data.get("visual_style", "")).strip()
        if len(visual_style) > 4000:
            raise ValueError("全局画面设定不能超过 4000 个字符")
        draft_name = str(data.get("draft_name", "")).strip()
        if draft_name and (draft_name in (".", "..") or "/" in draft_name or "\\" in draft_name):
            raise ValueError("剪映草稿名称不能包含路径分隔符")
        transcript = str(data.get("transcript_json", "")).strip()
        if transcript:
            transcript_path = (self.root / transcript).resolve()
            if not transcript_path.is_relative_to(self.root) or transcript_path.suffix.lower() != ".json" or not transcript_path.is_file():
                raise ValueError("字幕 JSON 必须是项目目录内已存在的文件")
        command = [sys.executable, str(SCRIPT_DIR / "audio_video.py"), str(audio), "--output-dir", str(output),
                   "--output", output_type, "--scene-mode", scene_mode, "--whisper-model", str(data.get("whisper_model") or "small"),
                   "--language", str(data.get("language", "zh")), "--width", str(width), "--height", str(height), "--fps", str(fps),
                   "--scene-seconds", str(scene_seconds)]
        if image_steps is not None:
            command += ["--image-steps", str(image_steps)]
        if transcript:
            command += ["--transcript-json", str(transcript_path)]
        if max_seconds is not None:
            command += ["--max-seconds", str(max_seconds)]
        if generate_images:
            command.append("--generate-images")
        if register_jianying:
            command.append("--register-jianying")
        if generate_images:
            selected_model = str(data.get("drawthings_model") or "").strip()
            try:
                available = {item["id"] for item in list_models()}
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                raise ValueError(f"无法读取 Draw Things 本地模型：{exc}") from exc
            if selected_model not in available:
                raise ValueError("请选择本机已下载的 Draw Things 模型")
        for key, flag in (("llm_base_url", "--llm-base-url"), ("llm_model", "--llm-model"),
                          ("visual_style", "--visual-style"),
                          ("drawthings_model", "--drawthings-model"),
                          ("jianying_drafts_dir", "--jianying-drafts-dir"), ("draft_name", "--draft-name")):
            value = str(data.get(key, "")).strip()
            if value:
                command += [flag, value]
        env = os.environ.copy()
        key = str(data.get("llm_api_key", "")).strip()
        if key:
            env["LLM_API_KEY"] = key
        public_options = {k: v for k, v in data.items() if k != "llm_api_key"}
        return command, env, {"audio_id": audio_id, "audio_name": audio.name, "options": public_options}

    def _persist(self, job: dict) -> None:
        path = self.runs / job["id"] / "job.json"
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def start(self, data: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        output = self.runs / job_id
        command, env, details = self._command(data, output)
        output.mkdir()
        state = {"id": job_id, "status": "running", "created_at": now_iso(), "finished_at": None,
                 "exit_code": None, **details}
        with self.lock:
            self.jobs[job_id] = state
            self._persist(state)
        threading.Thread(target=self._run, args=(job_id, command, env), daemon=True).start()
        return self.get(job_id)

    def _run(self, job_id: str, command: list[str], env: dict[str, str]) -> None:
        output = self.runs / job_id
        with self.lock:
            if self.jobs[job_id]["status"] == "cancelled":
                self.jobs[job_id].update(exit_code=-15, finished_at=now_iso())
                self._persist(self.jobs[job_id])
                return
        try:
            with subprocess.Popen(command, cwd=SCRIPT_DIR, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True) as process:
                with self.lock:
                    self.processes[job_id] = process
                    cancelled = self.jobs[job_id]["status"] == "cancelled"
                if cancelled:
                    os.killpg(process.pid, signal.SIGTERM)
                with (output / "run.log").open("w", encoding="utf-8") as log:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                exit_code = process.wait()
            status = "completed" if exit_code == 0 else "failed"
        except Exception as exc:
            (output / "run.log").write_text(f"启动失败：{exc}\n", encoding="utf-8")
            exit_code, status = 1, "failed"
        with self.lock:
            self.processes.pop(job_id, None)
            state = self.jobs[job_id]
            if state["status"] == "cancelled":
                status = "cancelled"
            state.update(status=status, exit_code=exit_code, finished_at=now_iso())
            if status == "completed" and "--register-jianying" in command:
                pointer = output / "jianying_draft_path.txt"
                if pointer.is_file():
                    state["registered_draft"] = pointer.read_text(encoding="utf-8").strip()
            self._persist(state)

    def get(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            state = dict(self.jobs[job_id])
        output = self.runs / job_id
        if not output.is_dir():
            raise KeyError(job_id)
        state["output_dir"] = str(output)
        try:
            state["files"] = sorted(p.name for p in output.iterdir() if p.is_file() and p.name not in ("job.json", "progress.json", "progress.json.tmp"))
        except FileNotFoundError as exc:
            raise KeyError(job_id) from exc
        log = output / "run.log"
        state["log"] = log.read_text(encoding="utf-8", errors="replace")[-12000:] if log.is_file() else ""
        progress_file = output / "progress.json"
        try:
            state["progress"] = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.is_file() else {"stage": "等待启动", "percent": 0}
        except (OSError, ValueError):
            state["progress"] = {"stage": "处理中", "percent": None}
        if state["status"] == "completed":
            state["progress"] = {"stage": "已完成", "percent": 100}
        return state

    def list(self) -> list[dict]:
        with self.lock:
            ids = sorted(self.jobs, key=lambda k: self.jobs[k]["created_at"], reverse=True)
        result = []
        for job_id in ids:
            try:
                result.append(self.get(job_id))
            except KeyError:
                continue
        return result

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            if self.jobs[job_id]["status"] != "running":
                raise ValueError("任务已经结束")
            self.jobs[job_id]["status"] = "cancelled"
            process = self.processes.get(job_id)
            self._persist(self.jobs[job_id])
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return self.get(job_id)

    def delete(self, job_id: str) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
            if state is None:
                raise KeyError(job_id)
            if state["status"] == "running":
                raise ValueError("请先取消任务，待处理结束后再删除")
            folder = self.runs / job_id
            if not folder.resolve().is_relative_to(self.runs.resolve()) or folder.resolve() == self.runs.resolve():
                raise ValueError("任务目录无效")
            if folder.is_dir():
                shutil.rmtree(folder)
            self.jobs.pop(job_id)

    def open_output(self, job_id: str, target: str = "output") -> None:
        job = self.get(job_id)
        if target == "output":
            folder = Path(job["output_dir"])
        elif target == "registered" and job.get("registered_draft"):
            folder = Path(job["registered_draft"])
        else:
            raise ValueError("没有可打开的目录")
        if not folder.is_dir():
            raise ValueError("目录不存在")
        subprocess.Popen(["open", str(folder)])

    def register(self, job_id: str, draft_root: str = "") -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            if self.jobs[job_id]["status"] != "completed":
                raise ValueError("请等待草稿生成任务完成")
            if self.jobs[job_id].get("registered_draft"):
                raise ValueError("该草稿已注册到剪映")
        output = self.runs / job_id
        source = output / "jianying_draft"
        if not (source / "draft_content.json").is_file():
            candidates = [p for p in output.iterdir() if p.is_dir() and (p / "draft_content.json").is_file()]
            if len(candidates) != 1:
                raise ValueError("此任务没有可注册的剪映草稿")
            source = candidates[0]
        selected_root = Path(draft_root).expanduser() if draft_root else None
        name = str(self.jobs[job_id]["options"].get("draft_name") or Path(self.jobs[job_id]["audio_name"]).stem)
        destination = register_draft(source, selected_root, name)
        (output / "jianying_draft_path.txt").write_text(str(destination) + "\n", encoding="utf-8")
        with self.lock:
            self.jobs[job_id]["registered_draft"] = str(destination)
            self._persist(self.jobs[job_id])
        return self.get(job_id)


def create_app(project_root: Path | None = None) -> Flask:
    root = (project_root or SCRIPT_DIR).resolve()
    app = Flask(__name__, template_folder=str(SCRIPT_DIR / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    app.config["TRUSTED_HOSTS"] = ["127.0.0.1", "localhost"]
    manager = JobManager(root)
    app.extensions["job_manager"] = manager

    @app.before_request
    def check_origin():
        if request.method in ("POST", "PUT", "DELETE"):
            origin = request.headers.get("Origin")
            if origin and (urlparse(origin).scheme != "http" or urlparse(origin).netloc != request.host):
                return jsonify(error="请求来源不匹配"), 403

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="上传文件超过 1 GB"), 413

    @app.errorhandler(500)
    def internal_error(_):
        if request.path.startswith("/api/"):
            return jsonify(error="服务器内部错误，请查看服务日志"), 500
        return "服务器内部错误", 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/audio")
    def audio_files():
        return jsonify(files=manager.audio_files())

    @app.get("/api/drawthings/models")
    def drawthings_models():
        try:
            return jsonify(models=list_models())
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return jsonify(error=f"无法读取 Draw Things 本地模型：{exc}"), 503

    @app.get("/api/jianying/status")
    def jianying_status():
        selected = request.args.get("root", "").strip()
        return jsonify(registration_status(Path(selected) if selected else None))

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify(error="请选择音频文件"), 400
        try:
            name = safe_audio_name(file.filename)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        stored = f"{uuid.uuid4().hex[:10]}_{name}"
        file.save(manager.uploads / stored)
        return jsonify(id=f"upload:{stored}", name=name), 201

    @app.get("/api/jobs")
    def jobs():
        return jsonify(jobs=manager.list())

    @app.post("/api/jobs")
    def start_job():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="请提交 JSON 配置"), 400
        try:
            return jsonify(manager.start(data)), 202
        except (ValueError, OSError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        try:
            return jsonify(manager.get(job_id))
        except KeyError:
            return jsonify(error="任务不存在"), 404

    @app.delete("/api/jobs/<job_id>")
    def delete_job(job_id: str):
        try:
            manager.delete(job_id)
            return jsonify(deleted=True)
        except KeyError:
            return jsonify(error="任务不存在"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/jobs/<job_id>/open")
    def open_job(job_id: str):
        try:
            manager.open_output(job_id, str((request.get_json(silent=True) or {}).get("target") or "output"))
            return jsonify(opened=True)
        except KeyError:
            return jsonify(error="任务不存在"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except OSError as exc:
            return jsonify(error=f"无法打开目录：{exc}"), 500

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        try:
            return jsonify(manager.cancel(job_id))
        except KeyError:
            return jsonify(error="任务不存在"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/jobs/<job_id>/register")
    def register_job(job_id: str):
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(manager.register(job_id, str(data.get("draft_root") or "")))
        except KeyError:
            return jsonify(error="任务不存在"), 404
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/jobs/<job_id>/files/<path:name>")
    def output_file(job_id: str, name: str):
        try:
            job = manager.get(job_id)
        except KeyError:
            return jsonify(error="任务不存在"), 404
        if name not in job["files"] or name == "run.log":
            return jsonify(error="文件不存在"), 404
        return send_from_directory(job["output_dir"], name, as_attachment=name not in ("video.mp4", "transcript.txt", "subtitles.srt"))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="本地音频视频制作页面")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = create_app()
    print(f"打开 http://127.0.0.1:{args.port}", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
