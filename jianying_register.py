"""Register compatible, locally generated Jianying drafts in the desktop draft list."""

from __future__ import annotations

import copy
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


DEFAULT_ROOT = Path.home() / "Movies/JianyingPro/User Data/Projects/com.lveditor.draft"
PREFERENCES = Path.home() / "Library/Preferences/com.bytedance.JianyingPro.plist"
APP_INFO = Path("/Applications/VideoFusion-macOS.app/Contents/Info.plist")


def detect_draft_root() -> Path:
    """Use the draft directory selected in Jianying preferences, if available."""
    if sys.platform == "darwin" and PREFERENCES.is_file():
        try:
            prefs = plistlib.loads(PREFERENCES.read_bytes())
            selected = prefs.get("GlobalSettings.History.currentCustomDraftPath")
            if isinstance(selected, str) and selected:
                return Path(selected).expanduser().resolve()
        except (OSError, ValueError, plistlib.InvalidFileException):
            pass
    return DEFAULT_ROOT.expanduser().resolve()


def installed_version() -> str | None:
    if not APP_INFO.is_file():
        return None
    try:
        return plistlib.loads(APP_INFO.read_bytes()).get("CFBundleShortVersionString")
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None


def registration_status(root: Path | None = None) -> dict:
    root = (root or detect_draft_root()).expanduser().resolve()
    version = installed_version()
    major = int(version.split(".", 1)[0]) if version and version.split(".", 1)[0].isdigit() else None
    compatible = major is not None and (major < 6 or major == 11)
    reason = None
    if not root.is_dir():
        reason = "剪映草稿目录不存在，请先在剪映中确认草稿位置"
    elif not (root / "root_meta_info.json").is_file():
        reason = "草稿目录缺少 root_meta_info.json，无法安全登记"
    elif major is None:
        reason = "未检测到剪映版本，无法确认草稿格式"
    elif not compatible:
        reason = f"尚未验证剪映 {version} 的草稿注册格式"
    return {"root": str(root), "version": version, "supported": reason is None,
            "format": "v11" if major == 11 else "legacy", "reason": reason}


def _rewrite_paths(value, old: Path, new: Path):
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, old, new) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            return str(new / Path(value).resolve().relative_to(old))
        except ValueError:
            pass
    return value


def _make_cover(staging: Path) -> None:
    cover = staging / "draft_cover.jpg"
    if cover.is_file():
        return
    images = sorted((staging / "media" / "prepared").glob("*.png"))
    if images:
        source = ["-i", str(images[0])]
    else:
        source = ["-f", "lavfi", "-i", "color=c=black:s=288x162:d=0.1"]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *source,
                    "-frames:v", "1", "-vf", "scale=288:-1", str(cover)],
                   check=True, capture_output=True)


def register_draft(source: Path, root: Path | None = None, name: str | None = None) -> Path:
    """Copy a draft, convert its entry files and add it to Jianying's home index."""
    source = source.expanduser().resolve()
    status = registration_status(root)
    if not status["supported"]:
        raise RuntimeError(status["reason"])
    root = Path(status["root"])
    if not source.is_dir() or not (source / "draft_content.json").is_file() or not (source / "draft_meta_info.json").is_file():
        raise ValueError("找不到完整的剪映明文草稿")
    name = name or source.name
    if name in (".", "..") or not name or "/" in name or "\\" in name:
        raise ValueError("草稿名称不能包含路径分隔符")
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"剪映中已有同名草稿：{name}")
    index_path = root / "root_meta_info.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index.get("all_draft_store")
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        raise ValueError("剪映草稿索引格式不受支持")
    if any(record.get("draft_name") == name for record in records):
        raise FileExistsError(f"剪映索引中已有同名草稿：{name}")
    content = json.loads((source / "draft_content.json").read_text(encoding="utf-8"))
    meta = json.loads((source / "draft_meta_info.json").read_text(encoding="utf-8"))
    if not isinstance(content, dict) or not isinstance(meta, dict):
        raise ValueError("草稿 JSON 格式无效")
    draft_id = str(meta.get("draft_id") or uuid.uuid4()).upper()
    if any(record.get("draft_id") == draft_id for record in records):
        draft_id = str(uuid.uuid4()).upper()
    staging = root / (".registering-" + uuid.uuid4().hex)
    shutil.copytree(source, staging)
    try:
        modern = status["format"] == "v11"
        old, new = source, destination
        content = _rewrite_paths(content, old, new)
        meta = _rewrite_paths(meta, old, new)
        now = int(time.time() * 1_000_000)
        content.update(id=draft_id, name=name, create_time=now, update_time=now)
        if modern:
            platform = {"app_id": 3704, "app_source": "lv", "app_version": status["version"], "os": "mac"}
            content["platform"] = dict(platform)
            content["last_modified_platform"] = dict(platform)
        content_file = "draft_info.json" if modern else "draft_content.json"
        (staging / content_file).write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        if modern:
            (staging / "draft_content.json").unlink()
        _make_cover(staging)
        meta.update(draft_id=draft_id, draft_name=name, draft_fold_path=str(new),
                    draft_root_path=str(root), draft_cover=str(new / "draft_cover.jpg"),
                    tm_duration=int(content.get("duration", 0)))
        (staging / "draft_meta_info.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        entry = copy.deepcopy(records[0])
        entry.update(draft_id=draft_id, draft_name=name, draft_fold_path=str(new), draft_root_path=str(root),
                     draft_json_file=str(destination / content_file),
                     draft_cover=str(destination / "draft_cover.jpg"), draft_new_version="",
                     tm_draft_create=now, tm_draft_modified=now, tm_draft_removed=0,
                     tm_duration=int(content.get("duration", 0)), draft_is_invisible=False,
                     draft_timeline_materials_size=sum(p.stat().st_size for p in (staging / "media").rglob("*") if p.is_file()),
                     cloud_draft_sync=False, cloud_draft_cover=False,
                     draft_is_cloud_temp_draft=False, streaming_edit_draft_ready=True,
                     tm_draft_cloud_entry_id=-1, tm_draft_cloud_parent_entry_id=-1,
                     tm_draft_cloud_space_id=-1, tm_draft_cloud_user_id=-1,
                     tm_draft_cloud_modified=0, tm_draft_cloud_completed="")
        index["all_draft_store"] = [*records, entry]
        index["root_path"] = str(root)
        backup = root / ("root_meta_info.json.bak-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        shutil.copy2(index_path, backup)
        if destination.exists():
            raise FileExistsError(f"剪映中已有同名草稿：{name}")
        staging.rename(destination)
        try:
            # Keep the index inode and macOS xattrs, matching the proven local workflow.
            with index_path.open("w", encoding="utf-8") as handle:
                json.dump(index, handle, ensure_ascii=False, indent=1)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            shutil.copy2(backup, index_path)
            shutil.rmtree(destination)
            raise
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
