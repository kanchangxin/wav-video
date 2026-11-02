import tempfile
import unittest
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from drawthings_cli import list_models, generate_image, generation_base_size


class DrawThingsCliTests(unittest.TestCase):
    def test_x2_upscaler_uses_half_size_diffusion_and_cli_config(self):
        self.assertEqual(generation_base_size(1280, 720, True), (640, 384))
        self.assertEqual(generation_base_size(1280, 720, False), (1280, 720))
        from scene_motion import generation_size
        target = generation_size(1280, 720, "pan_left")
        base = generation_base_size(*target, True)
        self.assertTrue(all(2 * source >= required for source, required in zip(base, target)))
        self.assertTrue(all(source < required for source, required in zip(base, target)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "draw-things-cli"
            cli.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$DRAWTHINGS_TEST_ARGS\"\nwhile [ \"$1\" != '--output' ]; do shift; done\nprintf image > \"$2\"\n")
            cli.chmod(0o755)
            with patch.dict("os.environ", {"DRAWTHINGS_TEST_ARGS": str(root / "args.txt")}):
                generate_image("city", root / "scene.png", "model.ckpt", 640, 384,
                               cli=cli, upscale_x2=True)
            args = (root / "args.txt").read_text().splitlines()
            import json
            config = json.loads(args[args.index("--config-json") + 1])
            self.assertEqual(config, {"upscaler": "realesrgan_x2plus_f16.ckpt", "upscalerScaleFactor": 2})
            self.assertEqual(args[args.index("--width") + 1], "640")

    def test_concurrent_local_generations_are_serialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "draw-things-cli"
            cli.write_text("#!/bin/sh\necho start >> \"$DRAWTHINGS_TEST_LOG\"\nsleep 0.15\necho end >> \"$DRAWTHINGS_TEST_LOG\"\nwhile [ \"$1\" != '--output' ]; do shift; done\nprintf image > \"$2\"\n")
            cli.chmod(0o755)
            log = root / "events.log"
            with patch.dict("os.environ", {"DRAWTHINGS_TEST_LOG": str(log)}), \
                 patch("drawthings_cli.LOCK_PATH", root / "lock", create=True):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda i: generate_image("scene", root / f"{i}.png", "model.ckpt", 320, 192, cli=cli), range(2)))
            self.assertEqual(log.read_text().splitlines(), ["start", "end", "start", "end"])

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

    def test_matching_aspect_reference_is_padded_to_avoid_canvas_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "front.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.04",
                            "-frames:v", "1", str(reference)], check=True)
            cli = root / "draw-things-cli"
            cli.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$DRAWTHINGS_TEST_ARGS\"\nwhile [ \"$1\" != '--output' ]; do shift; done\nprintf image > \"$2\"\n")
            cli.chmod(0o755)
            from unittest.mock import patch
            with patch.dict("os.environ", {"DRAWTHINGS_TEST_ARGS": str(root / "args.txt")}):
                generate_image("same person", root / "out.png", "flux.ckpt", 512, 512,
                               cli=cli, references=[reference])
            args = (root / "args.txt").read_text().splitlines()
            self.assertNotEqual(args[args.index("--image") + 1], str(reference))
