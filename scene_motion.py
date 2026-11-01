"""Shared, editable camera motion plans for FFmpeg and Jianying drafts."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


PRESETS = {"static", "auto", "push_in", "pull_out", "pan_left", "pan_right", "parallax"}
LLM_PRESETS = {"static", "push_in", "pull_out", "pan_left", "pan_right"}


@dataclass(frozen=True)
class Motion:
    scale_start: float = 1.0
    scale_end: float = 1.0
    x_start: float = 0.0
    x_end: float = 0.0
    foreground_x_start: float = 0.0
    foreground_x_end: float = 0.0


MOTIONS = {
    "static": Motion(),
    "push_in": Motion(1.0, 1.18),
    "pull_out": Motion(1.28, 1.0),
    "pan_left": Motion(1.32, 1.32, 0.12, -0.12),
    "pan_right": Motion(1.32, 1.32, -0.12, 0.12),
    "parallax": Motion(1.32, 1.32, 0.10, -0.10, -0.06, 0.06),
}


def choose_preset(requested: str, scene_index: int, has_foreground: bool) -> str:
    if requested not in PRESETS:
        raise ValueError(f"未知动效预设：{requested}")
    if requested == "auto":
        raise ValueError("自动运镜需要先由大模型逐场景推荐")
    if requested == "parallax" and not has_foreground:
        return "pan_left" if scene_index % 2 == 0 else "pan_right"
    return requested


def motion_for(preset: str) -> Motion:
    return MOTIONS[preset]


def generation_size(width: int, height: int, preset: str) -> tuple[int, int]:
    """Reserve enough source pixels for pull-out and horizontal camera moves."""
    if preset in {"pull_out", "pan_left", "pan_right", "parallax"}:
        return ceil(width * 1.25 / 64) * 64, ceil(height * 1.25 / 64) * 64
    return width, height
