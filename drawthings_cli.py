"""Local Draw Things CLI integration."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable


DEFAULT_CLI = Path(__file__).resolve().parent / "tools" / "drawthings" / "draw-things-cli"


def cli_path() -> Path:
    if DEFAULT_CLI.is_file():
        return DEFAULT_CLI
    found = shutil.which("draw-things-cli")
    if found:
        return Path(found)
    raise RuntimeError("未找到 draw-things-cli，请将它安装到 PATH 或 tools/drawthings/")


def list_models(cli: Path | None = None) -> list[dict[str, str]]:
    selected = cli or cli_path()
    result = subprocess.run([str(selected), "models", "list", "--downloaded-only", "--offline"],
                            capture_output=True, text=True, check=True, timeout=30)
    models = []
    for line in result.stdout.splitlines():
        fields = re.split(r"\s{2,}", line.strip())
        model_id = fields[0] if fields else ""
        specialized = model_id.startswith(("ltx_", "wan_")) or "_edit_" in model_id or "inpaint" in model_id
        if len(fields) >= 4 and model_id.endswith(".ckpt") and fields[3].lower() == "yes" and not specialized:
            models.append({"id": fields[0], "name": fields[1]})
    return models


def generate_image(prompt: str, path: Path, model: str, width: int, height: int,
                   steps: int | None = None, cli: Path | None = None,
                   progress: Callable[[int], None] | None = None,
                   seed: int | None = None, references: list[Path] | None = None) -> None:
    if not model:
        raise ValueError("请选择已下载的 Draw Things 模型")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(cli or cli_path()), "generate", "--local", "--offline", "--no-download-missing",
               "--disable-preview", "--model", model, "--prompt", prompt,
               "--width", str(max(64, round(width / 64) * 64)),
               "--height", str(max(64, round(height / 64) * 64)), "--output", str(path)]
    if steps is not None:
        command += ["--steps", str(steps)]
    if seed is not None:
        command += ["--seed", str(seed)]
    for reference in references or []:
        if not reference.is_file():
            raise FileNotFoundError(f"参考图不存在：{reference}")
        command += ["--image", str(reference)]
    recent = []
    last_percent = -1
    timed_out = threading.Event()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        def stop() -> None:
            timed_out.set()
            process.kill()
        timer = threading.Timer(1800, stop)
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
            timer.cancel()
    if timed_out.is_set():
        raise RuntimeError("Draw Things CLI 生图超时")
    if exit_code != 0:
        raise RuntimeError(f"Draw Things CLI 生图失败：{' '.join(recent)[-500:]}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Draw Things CLI 未生成图片")
