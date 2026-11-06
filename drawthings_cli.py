"""Local Draw Things CLI integration."""

from __future__ import annotations

import json
import os
import re
import fcntl
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable



DEFAULT_CLI = Path(__file__).resolve().parent / "tools" / "drawthings" / "draw-things-cli"
LOCK_PATH = Path(tempfile.gettempdir()) / "wav-video-drawthings.lock"
X2_UPSCALER = "realesrgan_x2plus_f16.ckpt"
X2_CONFIG = {"upscaler": X2_UPSCALER, "upscalerScaleFactor": 2}


def _thermal_pause_seconds() -> float:
    clock = os.environ.get("WAV_GPU_PAUSE_CLOCK")
    if clock:
        try:
            return float(Path(clock).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return 0.0


def upscaler_available() -> bool:
    configured = os.getenv("DRAWTHINGS_MODELS_DIR")
    models = Path(configured).expanduser() if configured else (
        Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models")
    return (models / X2_UPSCALER).is_file()


def generation_base_size(width: int, height: int, upscale_x2: bool) -> tuple[int, int]:
    """Return diffusion dimensions that cover the requested size after 2x export."""
    if not upscale_x2:
        return width, height
    return (max(64, ((width + 127) // 128) * 64),
            max(64, ((height + 127) // 128) * 64))


def cli_path() -> Path:
    if DEFAULT_CLI.is_file():
        return DEFAULT_CLI
    found = shutil.which("draw-things-cli")
    if found:
        return Path(found)
    raise RuntimeError("未找到 draw-things-cli，请将它安装到 PATH 或 tools/drawthings/")


def list_models(cli: Path | None = None, include_edit: bool = False) -> list[dict[str, str]]:
    selected = cli or cli_path()
    result = subprocess.run([str(selected), "models", "list", "--downloaded-only", "--offline"],
                            capture_output=True, text=True, check=True, timeout=30)
    models = []
    for line in result.stdout.splitlines():
        fields = re.split(r"\s{2,}", line.strip())
        model_id = fields[0] if fields else ""
        specialized = model_id.startswith(("ltx_", "wan_")) or (not include_edit and
                        ("_edit_" in model_id or "inpaint" in model_id))
        if len(fields) >= 4 and model_id.endswith(".ckpt") and fields[3].lower() == "yes" and not specialized:
            models.append({"id": fields[0], "name": fields[1]})
    return models


def generate_image(prompt: str, path: Path, model: str, width: int, height: int,
                   steps: int | None = None, cli: Path | None = None,
                   progress: Callable[[int], None] | None = None,
                   seed: int | None = None, references: list[Path] | None = None,
                   upscale_x2: bool = False, reference_mode: str = "moodboard") -> None:
    if not model:
        raise ValueError("请选择已下载的 Draw Things 模型")
    if reference_mode not in {"moodboard", "edit", "instruction"}:
        raise ValueError("参考图模式无效")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(cli or cli_path()), "generate", "--local", "--offline", "--no-download-missing",
               "--disable-preview", "--model", model, "--prompt", prompt,
               "--width", str(max(64, round(width / 64) * 64)),
               "--height", str(max(64, round(height / 64) * 64)), "--output", str(path)]
    if steps is not None:
        command += ["--steps", str(steps)]
    if seed is not None:
        command += ["--seed", str(seed)]
    if upscale_x2:
        command += ["--config-json", json.dumps(X2_CONFIG, separators=(",", ":"))]
    prepared_reference = None
    selected_references = list(references or [])
    if selected_references and reference_mode in {"moodboard", "instruction"}:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                                str(selected_references[0])], capture_output=True, text=True, timeout=10)
        if probe.returncode == 0:
            try:
                reference_width, reference_height = map(int, probe.stdout.strip().split("x"))
            except ValueError:
                reference_width = reference_height = 0
            output_width = max(64, round(width / 64) * 64)
            output_height = max(64, round(height / 64) * 64)
            if reference_width and reference_height and reference_width * output_height == reference_height * output_width:
                with tempfile.NamedTemporaryFile(prefix="character_reference_", suffix=".png",
                                                  dir=path.parent, delete=False) as temporary:
                    prepared_reference = Path(temporary.name)
                if reference_mode == "instruction":
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(selected_references[0]),
                                    "-vf", "pad=iw:ih+64:0:32:color=black,fillborders=top=32:bottom=32:mode=smear",
                                    "-frames:v", "1", str(prepared_reference)],
                                   capture_output=True, text=True, check=True, timeout=60)
                else:
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(selected_references[0]),
                                    "-vf", "pad=iw:ih+64:0:32:color=white", "-frames:v", "1",
                                    str(prepared_reference)], capture_output=True, text=True, check=True, timeout=60)
                selected_references[0] = prepared_reference
    for reference in selected_references:
        if not reference.is_file():
            raise FileNotFoundError(f"参考图不存在：{reference}")
        command += ["--image", str(reference)]
    recent = []
    last_percent = -1
    timed_out = threading.Event()
    try:
        with LOCK_PATH.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    finished = threading.Event()
                    started_at = time.monotonic()
                    initial_pause = _thermal_pause_seconds()
                    def stop() -> None:
                        remaining = 1800.0
                        while not finished.wait(remaining):
                            cooling = max(0, _thermal_pause_seconds() - initial_pause)
                            remaining = 1800 - (time.monotonic() - started_at - cooling)
                            if remaining <= 0:
                                if process.poll() is None:
                                    timed_out.set()
                                    process.kill()
                                return
                    timer = threading.Thread(target=stop, daemon=True)
                    timer.start()
                    try:
                        for line in process.stdout:
                            recent.append(line.strip())
                            recent = recent[-15:]
                            match = re.search(r"(\d{1,3})\s*%", line)
                            if match and progress:
                                current = min(100, int(match.group(1)))
                                if current > last_percent:
                                    last_percent = current
                                    progress(current)
                        exit_code = process.wait()
                    finally:
                        finished.set()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        if prepared_reference:
            prepared_reference.unlink(missing_ok=True)
    if timed_out.is_set():
        raise RuntimeError("Draw Things CLI 生图超时")
    if exit_code != 0:
        raise RuntimeError(f"Draw Things CLI 生图失败：{' '.join(recent)[-500:]}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Draw Things CLI 未生成图片")
