import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from character_profiles import CharacterStore


class CharacterStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = CharacterStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_rejects_path_outside_character_folder(self):
        with self.assertRaises(ValueError):
            self.store.create("主角", "../outside", "flux.ckpt", 271829, "portrait")

    def test_x2_character_preview_uses_half_base_and_keeps_final_view_size(self):
        profile = self.store.create("高欢", "characters/gaohuan", "flux.ckpt", 271829, "portrait")
        calls = []
        def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
            calls.append((width, height, kwargs.get("upscale_x2")))
            path.write_bytes(b"image")
        with patch("character_profiles.generate_image", side_effect=fake_generate):
            result = self.store.generate_preview(profile["id"], "front", "portrait", upscale_x2=True)
        self.assertEqual(calls, [(384, 512, True)])
        self.assertEqual((result["views"]["front"]["width"], result["views"]["front"]["height"]), (768, 1024))

    def test_side_requires_approved_front_and_uses_front_as_reference(self):
        profile = self.store.create("高欢", "characters/gaohuan", "flux.ckpt", 271829, "front prompt")
        with self.assertRaises(ValueError):
            self.store.generate_preview(profile["id"], "three_quarter", "side prompt")
        calls = []
        def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
            calls.append((prompt, model, kwargs.get("seed"), kwargs.get("references")))
            path.write_bytes(b"fake image")
        with patch("character_profiles.generate_image", side_effect=fake_generate):
            self.store.generate_preview(profile["id"], "front", "front prompt")
            self.store.approve(profile["id"], "front")
            self.store.generate_preview(profile["id"], "three_quarter", "side prompt")
        self.assertEqual(calls[0][3], [])
        self.assertEqual(calls[1][3], [self.root.resolve() / "characters/gaohuan/front.png"])
        self.assertEqual(calls[1][2], 271829)
        self.assertEqual(self.store.get(profile["id"])["views"]["front"]["status"], "approved")
        self.assertEqual(self.store.get(profile["id"])["views"]["front"]["width"], 768)
        self.assertEqual(self.store.get(profile["id"])["views"]["front"]["height"], 1024)

    def test_replacing_front_invalidates_older_side_draft(self):
        profile = self.store.create("高欢", "characters/gaohuan", "flux.ckpt", 271829, "portrait")
        def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
            path.write_bytes(b"image")
        with patch("character_profiles.generate_image", side_effect=fake_generate):
            self.store.generate_preview(profile["id"], "front", "first")
            self.store.approve(profile["id"], "front")
            self.store.generate_preview(profile["id"], "three_quarter", "side")
            self.store.generate_preview(profile["id"], "front", "revised")
            self.store.approve(profile["id"], "front")
        self.assertEqual(self.store.get(profile["id"])["views"]["three_quarter"]["status"], "stale")
        self.assertEqual(self.store.get(profile["id"])["views"]["three_quarter"]["stale_kind"], "draft")
        with self.assertRaises(ValueError):
            self.store.approve(profile["id"], "three_quarter")

    def test_changing_profile_model_controls_next_generation(self):
        profile = self.store.create("高欢", "characters/gaohuan", "first.ckpt", 271829, "portrait")
        updated = self.store.update_model(profile["id"], "second.ckpt")
        self.assertEqual(updated["model"], "second.ckpt")
        used = []
        def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
            used.append(model)
            path.write_bytes(b"image")
        with patch("character_profiles.generate_image", side_effect=fake_generate):
            self.store.generate_preview(profile["id"], "front", "portrait")
        self.assertEqual(used, ["second.ckpt"])

    def test_delete_archives_only_selected_profile_and_rejects_active_generation(self):
        first = self.store.create("高欢", "characters/gaohuan", "first.ckpt", 271829, "portrait")
        second = self.store.create("陈霸先", "characters/chen", "first.ckpt", 271829, "portrait")
        folder = self.root / "characters" / first["id"]
        (folder / "front.png").write_bytes(b"image")
        manifest = self.store.get(first["id"])
        manifest["generation"] = {"view": "front", "status": "running", "percent": 1}
        self.store._save(manifest)
        with self.assertRaisesRegex(ValueError, "正在生图"):
            self.store.delete(first["id"])
        manifest["generation"] = None
        self.store._save(manifest)
        archived = self.store.delete(first["id"])
        self.assertFalse(folder.exists())
        self.assertTrue((archived / "front.png").is_file())
        self.assertEqual([item["id"] for item in self.store.list()], [second["id"]])

    def test_delete_rejects_archive_symlink_outside_project(self):
        profile = self.store.create("高欢", "characters/gaohuan", "first.ckpt", 271829, "portrait")
        outside = self.root / "outside"
        outside.mkdir()
        (self.store.characters / ".deleted").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "归档目录"):
            self.store.delete(profile["id"])
        self.assertTrue((self.store.characters / profile["id"] / "profile.json").is_file())
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
