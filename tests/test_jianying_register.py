import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jianying_register import register_draft, registration_status


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "jianying"
        self.root.mkdir()
        self.source = self.base / "output" / "jianying_draft"
        (self.source / "media").mkdir(parents=True)
        (self.source / "media" / "voice.m4a").write_bytes(b"audio")
        (self.source / "draft_content.json").write_text(json.dumps({"duration": 3_000_000, "materials": [{"path": str(self.source / "media" / "voice.m4a")}]}, ensure_ascii=False))
        (self.source / "draft_meta_info.json").write_text(json.dumps({"draft_id": "source-id", "draft_name": ""}))
        (self.root / "root_meta_info.json").write_text(json.dumps({"all_draft_store": [{"draft_id": "old-id", "draft_name": "old", "draft_fold_path": str(self.root / "old")}], "draft_ids": 1, "root_path": str(self.root)}))

    def test_register_copies_media_rewrites_paths_and_indexes(self):
        with patch("jianying_register.installed_version", return_value="5.5.0"):
            dest = register_draft(self.source, self.root, "新草稿")
        self.assertTrue((dest / "media" / "voice.m4a").is_file())
        self.assertTrue((self.source / "draft_content.json").is_file())
        content = json.loads((dest / "draft_content.json").read_text())
        self.assertEqual(content["materials"][0]["path"], str(dest / "media" / "voice.m4a"))
        index = json.loads((self.root / "root_meta_info.json").read_text())
        self.assertEqual(index["all_draft_store"][-1]["draft_name"], "新草稿")
        self.assertEqual(index["all_draft_store"][-1]["draft_json_file"], str(dest / "draft_content.json"))
        self.assertEqual(index["draft_ids"], 1)
        self.assertTrue(list(self.root.glob("root_meta_info.json.bak-*")))

    def test_modern_draft_uses_info_file_and_mac_platform(self):
        with patch("jianying_register.installed_version", return_value="11.7.0"):
            status = registration_status(self.root)
            self.assertTrue(status["supported"])
            dest = register_draft(self.source, self.root, "新草稿")
        info = json.loads((dest / "draft_info.json").read_text())
        meta = json.loads((dest / "draft_meta_info.json").read_text())
        index = json.loads((self.root / "root_meta_info.json").read_text())
        self.assertEqual(info["id"], meta["draft_id"])
        self.assertEqual(info["name"], "新草稿")
        self.assertEqual(info["platform"]["os"], "mac")
        self.assertEqual(info["platform"]["app_version"], "11.7.0")
        self.assertEqual(index["all_draft_store"][-1]["draft_json_file"], str(dest / "draft_info.json"))
        self.assertEqual(index["draft_ids"], 1)

    def test_existing_name_does_not_get_overwritten(self):
        (self.root / "taken").mkdir()
        with patch("jianying_register.installed_version", return_value="5.5.0"):
            with self.assertRaises(FileExistsError):
                register_draft(self.source, self.root, "taken")
