"""Shared, editable camera motion plans for FFmpeg and Jianying drafts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil


PRESETS = {"static", "auto", "push_in", "pull_out", "pan_left", "pan_right", "parallax"}
LLM_PRESETS = {"static", "push_in", "pull_out", "pan_left", "pan_right"}
PARALLAX_PLACEMENTS = {"all", "odd", "spaced_odd", "even", "none", "manual"}
PARALLAX_BACKGROUND_EFFECTS = {"horizontal": "right"}
PARALLAX_FOREGROUND_EFFECTS = {"fade": "in_out"}


def parse_scene_numbers(value: str) -> set[int]:
    if not isinstance(value, str) or not re.fullmatch(r"\s*[0-9]+(?:[\s,，]+[0-9]+)*\s*", value):
        raise ValueError("请输入场景序号，用逗号或空格分隔，例如 1, 5, 9")
    numbers = {int(number) for number in re.split(r"[\s,，]+", value.strip())}
    if 0 in numbers:
        raise ValueError("场景序号从 1 开始")
    return numbers


def parallax_scene_numbers(placement: str, manual: set[int], count: int) -> set[int]:
    if placement == "manual":
        return {number for number in manual if number <= count}
    if placement == "none":
        return set()
    start, step = {"all": (1, 1), "odd": (1, 2), "spaced_odd": (1, 4), "even": (2, 2)}[placement]
    return set(range(start, count + 1, step))


@dataclass(frozen=True)
class Motion:
    scale_start: float = 1.0
    scale_end: float = 1.0
    x_start: float = 0.0
    x_end: float = 0.0
    foreground_x_start: float = 0.0
    foreground_x_end: float = 0.0
    y_start: float = 0.0
    y_end: float = 0.0
    fade: str = "none"

    @property
    def horizontal_pan_only(self) -> bool:
        return (self.x_start != self.x_end and self.y_start == self.y_end == 0
                and self.scale_start == self.scale_end)


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


EFFECT_CHOICES = {
    "fade": {"none", "in", "out", "in_out"},
    "horizontal": {"none", "left", "right"},
    "vertical": {"none", "up", "down"},
    "zoom": {"none", "in", "out"},
}


def normalize_effects(raw: dict | None) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or any(key not in EFFECT_CHOICES for key in raw):
        raise ValueError("动效设置无效")
    result = {}
    for key, value in raw.items():
        if not isinstance(value, str) or value not in EFFECT_CHOICES[key]:
            raise ValueError(f"{key} 动效无效")
        if value != "none":
            result[key] = value
    return result


def motion_for(preset: str, effects: dict | None = None) -> Motion:
    base = MOTIONS[preset]
    selected = normalize_effects(effects)
    x_start, x_end = base.x_start, base.x_end
    y_start, y_end = base.y_start, base.y_end
    scale_start, scale_end = base.scale_start, base.scale_end
    if selected.get("horizontal") == "left":
        x_start, x_end = 0.12, -0.12
    elif selected.get("horizontal") == "right":
        x_start, x_end = -0.12, 0.12
    if selected.get("vertical") == "up":
        y_start, y_end = -0.12, 0.12
    elif selected.get("vertical") == "down":
        y_start, y_end = 0.12, -0.12
    if selected.get("zoom") == "in":
        scale_start, scale_end = 1.0, 1.18
    elif selected.get("zoom") == "out":
        scale_start, scale_end = 1.28, 1.0
    if selected.get("horizontal") or selected.get("vertical"):
        if scale_start == scale_end:
            scale_start = scale_end = max(1.32, scale_start)
        elif scale_start < scale_end:
            delta = scale_end - scale_start
            scale_start = max(1.25, scale_start)
            scale_end = max(scale_end, scale_start + delta)
        else:
            delta = scale_start - scale_end
            scale_end = max(1.25, scale_end)
            scale_start = max(scale_start, scale_end + delta)
    return Motion(scale_start, scale_end, x_start, x_end,
                  base.foreground_x_start, base.foreground_x_end,
                  y_start, y_end, selected.get("fade", "none"))


def generation_size(width: int, height: int, preset: str, effects: dict | None = None) -> tuple[int, int]:
    """Reserve source pixels for the axes used by the camera move."""
    motion = motion_for(preset, effects)
    if motion.horizontal_pan_only:
        return ceil(width * 1.25 / 64) * 64, height
    if motion.scale_start > 1 or motion.x_start != motion.x_end or motion.y_start != motion.y_end:
        return ceil(width * 1.25 / 64) * 64, ceil(height * 1.25 / 64) * 64
    return width, height
