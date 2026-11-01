"""One-protagonist reference image profiles for local Draw Things generation."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO

from drawthings_cli import generate_image


VIEWS = ("front", "three_quarter", "full_body")
VIEW_SIZE = {"front": (768, 1024), "three_quarter": (768, 960), "full_body": (768, 1152)}
DEFAULT_FRONT_PROMPT = (
    "A single fictional adult protagonist, front-facing half-body studio portrait, "
    "neutral expression, clearly visible face, natural skin texture, consistent facial features, "
    "simple neutral clothing, plain light gray background, soft even lighting, no text, no extra people."
)


class CharacterStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.characters = self.root / "characters"
        self.characters.mkdir(exist_ok=True)
        self.lock = threading.RLock()

    def _folder(self, profile_id: str) -> Path:
        if not profile_id or profile_id in (".", "..") or "/" in profile_id or "\\" in profile_id:
            raise ValueError("角色档案编号无效")
        folder = (self.characters / profile_id).resolve()
        if folder.parent != self.characters.resolve():
            raise ValueError("角色档案必须位于 characters 目录内")
        return folder

    def _save(self, profile: dict) -> None:
        path = self._folder(profile["id"]) / "profile.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def create(self, name: str, folder: str, model: str, seed: int = 271829,
               front_prompt: str = DEFAULT_FRONT_PROMPT) -> dict:
        name, model, front_prompt = name.strip(), model.strip(), front_prompt.strip()
        if not name or not model or not front_prompt:
            raise ValueError("请填写角色名称、模型和正面照提示词")
        if not isinstance(seed, int) or not 0 <= seed <= 4294967295:
            raise ValueError("seed 必须是 0 到 4294967295 的整数")
        requested = Path(folder).expanduser()
        resolved = (requested if requested.is_absolute() else self.root / requested).resolve()
        if resolved.parent != self.characters.resolve():
            raise ValueError("角色档案路径须为项目 characters 目录下的单独文件夹")
        profile_id = resolved.name
        with self.lock:
            if resolved.exists():
                raise ValueError("角色档案目录已存在")
            resolved.mkdir()
            profile = {"id": profile_id, "name": name, "path": str(resolved), "model": model,
                       "seed": seed, "front_prompt": front_prompt,
                       "views": {view: {"status": "empty", "prompt": ""} for view in VIEWS},
                       "generation": None}
            self._save(profile)
        return profile

    def get(self, profile_id: str) -> dict:
        path = self._folder(profile_id) / "profile.json"
        if not path.is_file():
            raise KeyError(profile_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        result = []
        for path in sorted(self.characters.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_dir() and (path / "profile.json").is_file():
                try:
                    result.append(self.get(path.name))
                except (ValueError, OSError, KeyError):
                    continue
        return result

    def suggested_prompt(self, profile_id: str, view: str) -> str:
        profile = self.get(profile_id)
        if view == "front":
            return profile["front_prompt"]
        if view == "three_quarter":
            return (f"The same person as in image 1, preserve the exact facial identity and distinguishing features. "
                    f"Three-quarter view studio portrait of {profile['name']}, upper body, face clearly visible, "
                    "neutral expression, plain light gray background, soft even lighting, no extra people, no text.")
        if view == "full_body":
            return (f"The same person as in image 1, preserve the exact facial identity and distinguishing features. "
                    f"Full-body standing studio portrait of {profile['name']}, head to toe with feet visible, "
                    "plain light gray background, soft even lighting, no extra people, no text.")
        raise ValueError("未知的角色视角")

    def generate_preview(self, profile_id: str, view: str, prompt: str,
                         steps: int | None = None, reserved: bool = False) -> dict:
        if view not in VIEWS:
            raise ValueError("未知的角色视角")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        with self.lock:
            profile = self.get(profile_id)
            if view != "front" and profile["views"]["front"]["status"] != "approved":
                raise ValueError("请先确认正面半身标准照")
            if not reserved and profile["generation"] and profile["generation"]["status"] in ("queued", "running"):
                raise ValueError("该角色正在生图")
            profile["generation"] = {"view": view, "status": "running", "percent": 0, "error": ""}
            self._save(profile)
        folder = self._folder(profile_id)
        preview = folder / f"draft_{view}.png"
        references = [] if view == "front" else [folder / "front.png"]
        width, height = VIEW_SIZE[view]

        def on_progress(percent: int) -> None:
            with self.lock:
                current = self.get(profile_id)
                if current["generation"] and current["generation"]["view"] == view:
                    current["generation"]["percent"] = percent
                    self._save(current)

        try:
            generate_image(prompt, preview, profile["model"], width, height, steps,
                           progress=on_progress, seed=profile["seed"], references=references)
        except Exception as exc:
            with self.lock:
                current = self.get(profile_id)
                current["generation"] = {"view": view, "status": "failed", "percent": 0, "error": str(exc)}
                self._save(current)
            raise
        with self.lock:
            current = self.get(profile_id)
            current["views"][view] = {"status": "draft", "prompt": prompt,
                                      "width": width, "height": height, "steps": steps}
            current["generation"] = {"view": view, "status": "completed", "percent": 100, "error": ""}
            self._save(current)
            return current

    def start_generation(self, profile_id: str, view: str, prompt: str,
                         steps: int | None = None) -> dict:
        if view not in VIEWS or not prompt.strip():
            raise ValueError("请选择角色视角并填写提示词")
        with self.lock:
            profile = self.get(profile_id)
            if view != "front" and profile["views"]["front"]["status"] != "approved":
                raise ValueError("请先确认正面半身标准照")
            if profile["generation"] and profile["generation"]["status"] in ("queued", "running"):
                raise ValueError("该角色正在生图")
            profile["generation"] = {"view": view, "status": "queued", "percent": 0, "error": ""}
            self._save(profile)

        def work() -> None:
            try:
                self.generate_preview(profile_id, view, prompt, steps, reserved=True)
            except Exception:
                pass  # generate_preview persists the error for polling clients.

        threading.Thread(target=work, daemon=True).start()
        return self.get(profile_id)

    def import_image(self, profile_id: str, view: str, stream: BinaryIO, filename: str) -> dict:
        if view not in VIEWS:
            raise ValueError("未知的角色视角")
        extension = Path(filename).suffix.lower()
        if extension not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ValueError("只支持 PNG、JPEG 或 WebP 图片")
        with self.lock:
            profile = self.get(profile_id)
            if view != "front" and profile["views"]["front"]["status"] != "approved":
                raise ValueError("请先导入或确认正面半身标准照")
            if profile["generation"] and profile["generation"]["status"] in ("queued", "running"):
                raise ValueError("该角色正在生图")
            folder = self._folder(profile_id)
            source = folder / f"upload_{view}{extension}"
            converted = folder / f"import_{view}.png"
            try:
                remaining = 20 * 1024 * 1024 + 1
                with source.open("wb") as target:
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        target.write(chunk)
                        remaining -= len(chunk)
                if remaining == 0:
                    raise ValueError("参考图不能超过 20MB")
                result = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source),
                                         "-frames:v", "1", str(converted)], capture_output=True,
                                        text=True, timeout=60)
                if result.returncode or not converted.is_file():
                    raise ValueError("图片文件无法读取")
                converted.replace(folder / f"{view}.png")
                profile["views"][view] = {"status": "approved", "prompt": "用户导入的参考图"}
                if view == "front":
                    for other in VIEWS[1:]:
                        if profile["views"][other]["status"] in ("draft", "approved"):
                            profile["views"][other]["stale_kind"] = profile["views"][other]["status"]
                            profile["views"][other]["status"] = "stale"
                self._save(profile)
                return profile
            finally:
                source.unlink(missing_ok=True)
                converted.unlink(missing_ok=True)

    def approve(self, profile_id: str, view: str) -> dict:
        if view not in VIEWS:
            raise ValueError("未知的角色视角")
        with self.lock:
            profile = self.get(profile_id)
            if profile["views"][view]["status"] != "draft":
                raise ValueError("请先生成预览图")
            folder = self._folder(profile_id)
            preview = folder / f"draft_{view}.png"
            if not preview.is_file():
                raise ValueError("预览图不存在")
            preview.replace(folder / f"{view}.png")
            profile["views"][view]["status"] = "approved"
            if view == "front":
                for other in VIEWS[1:]:
                    if profile["views"][other]["status"] in ("draft", "approved"):
                        profile["views"][other]["stale_kind"] = profile["views"][other]["status"]
                        profile["views"][other]["status"] = "stale"
            self._save(profile)
            return profile

    def approved_references(self, profile_id: str) -> list[Path]:
        profile = self.get(profile_id)
        folder = self._folder(profile_id)
        return [folder / f"{view}.png" for view in VIEWS
                if profile["views"][view]["status"] == "approved" and (folder / f"{view}.png").is_file()]

    def open_folder(self, profile_id: str) -> None:
        folder = self._folder(profile_id)
        if not (folder / "profile.json").is_file():
            raise KeyError(profile_id)
        subprocess.Popen(["open", str(folder)])
