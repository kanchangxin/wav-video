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


if __name__ == "__main__":
    unittest.main()
