import tempfile
import unittest
from pathlib import Path

from drawthings_cli import list_models, generate_image


class DrawThingsCliTests(unittest.TestCase):
    def test_lists_only_downloaded_model_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = Path(tmp) / "draw-things-cli"
            cli.write_text("#!/bin/sh\ncat <<'EOF'\nModels directory: /tmp/Models\nMODEL  NAME  SOURCE  DOWNLOADED  HUGGING_FACE\n-----  ----  ------  ----------  ------------\nz_image_turbo_1.0_i8x.ckpt  Z Image Turbo  official  yes  -\nwan_v2.2_5b_ti2v_q8p.ckpt  Wan Video  official  yes  -\nmissing.ckpt  Missing Model  official  no  -\nEOF\n")
            cli.chmod(0o755)
            models = list_models(cli)
        self.assertEqual(models, [{"id": "z_image_turbo_1.0_i8x.ckpt", "name": "Z Image Turbo"}])

    def test_generate_uses_offline_cli_and_model_recommendations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "draw-things-cli"
            cli.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$DRAWTHINGS_TEST_ARGS\"\necho 'Sampling... 1 / 2 [ ] 40 %'\necho 'Sampling... 2 / 2 [ ] 80 %'\necho 'Finishing... 50 %'\nwhile [ \"$1\" != '--output' ]; do shift; done\nprintf image > \"$2\"\n")
            cli.chmod(0o755)
            out = root / "scene.png"
            args_file = root / "args.txt"
            updates = []
            from unittest.mock import patch
            with patch.dict("os.environ", {"DRAWTHINGS_TEST_ARGS": str(args_file)}):
                generate_image("一座城市", out, "z_image_turbo_1.0_i8x.ckpt", 1280, 720, None, cli, updates.append)
            args = args_file.read_text().splitlines()
            self.assertEqual(out.read_bytes(), b"image")
            self.assertIn("--offline", args)
            self.assertIn("--no-download-missing", args)
            self.assertIn("--local", args)
            self.assertNotIn("--steps", args)
            self.assertEqual(args[args.index("--height") + 1], "704")
            self.assertEqual(updates, [40, 80])

    def test_generate_passes_fixed_seed_and_ordered_reference_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "draw-things-cli"
            cli.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$DRAWTHINGS_TEST_ARGS\"\nwhile [ \"$1\" != '--output' ]; do shift; done\nprintf image > \"$2\"\n")
            cli.chmod(0o755)
            first, second = root / "front.png", root / "side.png"
            first.write_bytes(b"front")
            second.write_bytes(b"side")
            from unittest.mock import patch
            with patch.dict("os.environ", {"DRAWTHINGS_TEST_ARGS": str(root / "args.txt")}):
                generate_image("same person", root / "out.png", "flux.ckpt", 1280, 720,
                               cli=cli, seed=271829, references=[first, second])
            args = (root / "args.txt").read_text().splitlines()
            self.assertEqual(args[args.index("--seed") + 1], "271829")
            self.assertEqual([args[i + 1] for i, value in enumerate(args) if value == "--image"],
                             [str(first), str(second)])
