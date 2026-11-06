"""Prompts and model choices for scene-image guided parallax layers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


QWEN_EDIT_MODEL = "qwen_image_2.1_turbo_i8x.ckpt"
QWEN_2511_EDIT_MODEL = "qwen_image_edit_2511_i8x.ckpt"
COMPOSITE_SUBJECT = re.compile(
    r"骑马|坐骑|马背|马匹|战马|骏马|骑乘|骑车|自行车|摩托|驾车|马车|手持|握着|拿着|长矛|佩剑|持剑|弓箭|"
    r"horse|mount|riding|bicycle|motorcycle|holding|spear|sword|shield|bow", re.IGNORECASE)


def preferred_edit_model(models: list[dict], scene_model: str) -> str:
    available = {str(item.get("id") or "") for item in models}
    for candidate in (QWEN_EDIT_MODEL, QWEN_2511_EDIT_MODEL):
        if candidate in available:
            return candidate
    return scene_model


def foreground_mode(prompt: str) -> str:
    """A conservative hint; the editor lets users override this first choice."""
    return "ai" if COMPOSITE_SUBJECT.search(prompt) else "cutout"


def background_prompt() -> str:
    return "把 image 1 前景删除，只保留背景，并补全背景"


def foreground_prompt() -> str:
    return "把image1 背景透明"


def apply_foreground_mask(source: Path, ai_foreground: Path, output: Path) -> None:
    """Use the AI image's alpha as a mask over the unchanged source pixels."""
    dimensions = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height", "-of", "csv=p=0", str(ai_foreground),
    ], text=True, timeout=20).strip()
    width, height = (int(value) for value in dimensions.split(","))
    if width <= 0 or height <= 0:
        raise ValueError("AI 前景尺寸无效")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source), "-i", str(ai_foreground),
        "-filter_complex", f"[0:v]scale={width}:{height}:flags=lanczos,format=rgb24[original];"
                           "[1:v]alphaextract[mask];[original][mask]alphamerge",
        "-frames:v", "1", str(output),
    ], check=True, timeout=120)
