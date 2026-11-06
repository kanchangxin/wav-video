import json
import io
import subprocess
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from audio_video import Caption, Scene, analyze_scenes, audio_duration, build_scenes, decode_audio_ffmpeg, export_jianying, main, parse_llm_scenes, resolve_whisper_model, write_srt, render_mp4, simplify_texts, compose_scene_prompt, image_request_fingerprint, review_scenes, validate_foreground, recommend_scene_motions, recommend_character_presence, prepare_foregrounds, prepare_motion_backgrounds
from scene_motion import choose_preset, motion_for, generation_size, normalize_effects
from parallax_layers import background_prompt, foreground_prompt, foreground_mode, preferred_edit_model, apply_foreground_mask


class TimelineTests(unittest.TestCase):
    def test_forced_parallax_uses_one_based_scene_numbers(self):
        from scene_motion import parallax_scene_numbers, parse_scene_numbers
        manual = parse_scene_numbers("5，2, 5  12")
        for mode, expected in {"all": set(range(1, 10)), "odd": {1, 3, 5, 7, 9}, "spaced_odd": {1, 5, 9},
                               "even": {2, 4, 6, 8}, "none": set(), "manual": {2, 5}}.items():
            self.assertEqual(parallax_scene_numbers(mode, manual, 9), expected)
        for invalid in ("", "0", "-1", "1.5", "1,a"):
            with self.assertRaises(ValueError):
                parse_scene_numbers(invalid)

    def test_ai_foreground_mask_preserves_original_subject_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            ai = root / "ai.png"
            result = root / "result.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=red:s=64x64:d=0.04", "-frames:v", "1", str(source)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=blue:s=64x64:d=0.04", "-vf",
                            "format=rgba,geq=r='0':g='0':b='255':a='if(lt(X,32),255,0)'",
                            "-frames:v", "1", str(ai)], check=True)
            apply_foreground_mask(source, ai, result)
            rgb = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(result),
                                           "-vf", "crop=1:1:8:8,format=rgb24", "-frames:v", "1",
                                           "-f", "rawvideo", "-"])
            alpha = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(result),
                                             "-vf", "alphaextract,crop=1:1:56:8", "-frames:v", "1",
                                             "-f", "rawvideo", "-"])
            self.assertGreater(rgb[0], 240)
            self.assertEqual(rgb[1:], bytes([0, 0]))
            self.assertEqual(alpha, bytes([0]))

    def test_parallax_prompts_target_same_composite_subject(self):
        prompt = "高欢骑马冲入战场，手持长矛，远处有士兵"
        self.assertEqual(foreground_mode(prompt), "ai")
        self.assertEqual(background_prompt(), "把 image 1 前景删除，只保留背景，并补全背景")
        self.assertEqual(foreground_prompt(), "把image1 背景透明")
        self.assertEqual(foreground_mode("高欢站在城门前"), "cutout")

    def test_parallax_edit_model_prefers_downloaded_qwen(self):
        models = [{"id": "z_image.ckpt"}, {"id": "qwen_image_2.1_turbo_i8x.ckpt"}]
        self.assertEqual(preferred_edit_model(models, "z_image.ckpt"), "qwen_image_2.1_turbo_i8x.ckpt")
        self.assertEqual(preferred_edit_model([{"id": "qwen_image_edit_2511_i8x.ckpt"}], "z_image.ckpt"),
                         "qwen_image_edit_2511_i8x.ckpt")
        self.assertEqual(preferred_edit_model(models[:1], "z_image.ckpt"), "z_image.ckpt")

    def test_identity_prompt_only_applies_when_protagonist_is_present(self):
        with_character = compose_scene_prompt("ancient battlefield", "cinematic light", "高欢", True, True)
        without_character = compose_scene_prompt("empty battlefield", "cinematic light", "高欢", False, True)
        self.assertIn("Use image 1 as", with_character)
        self.assertIn("identity reference", with_character)
        self.assertTrue(with_character.endswith("cinematic light"))
        self.assertNotIn("identity reference", without_character)
        self.assertTrue(without_character.endswith("cinematic light"))

    def test_image_fingerprint_changes_with_reference_content_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "front.png"
            image.write_bytes(b"first portrait")
            first = image_request_fingerprint("a scene", "flux.ckpt", 1280, 720, None, 271829, [image])
            self.assertNotEqual(first, image_request_fingerprint("a scene", "flux.ckpt", 1280, 720, None, 271830, [image]))
            image.write_bytes(b"revised portrait")
            self.assertNotEqual(first, image_request_fingerprint("a scene", "flux.ckpt", 1280, 720, None, 271829, [image]))

    def test_image_fingerprint_changes_when_x2_upscaler_is_enabled(self):
        plain = image_request_fingerprint("scene", "model.ckpt", 640, 384, None, 42, [])
        enlarged = image_request_fingerprint("scene", "model.ckpt", 640, 384, None, 42, [], upscale_x2=True)
        self.assertNotEqual(plain, enlarged)
        import hashlib
        legacy = {"prompt": "scene", "model": "model.ckpt", "width": 640, "height": 384,
                  "steps": None, "seed": 42, "reference_hashes": []}
        expected = hashlib.sha256(json.dumps(legacy, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.assertEqual(plain, expected)

    def test_x2_scene_generation_doubles_entered_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 1, "text": "city"}]))
            output = root / "out"
            calls = []
            def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
                calls.append((width, height, kwargs.get("upscale_x2")))
                path.write_bytes(b"image")
            with patch("audio_video.generate_image", side_effect=fake_generate), \
                 patch("audio_video.upscaler_available", return_value=True):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "assets", "--scene-mode", "fixed", "--generate-images",
                               "--drawthings-model", "test.ckpt", "--upscale-x2"])
            self.assertEqual(result, 0)
            self.assertEqual(calls, [(1280, 768, True)])
            record = json.loads((output / "image_generation.json").read_text())[0]
            self.assertEqual((record["width"], record["height"], record["upscale_x2"]), (1280, 768, True))

    def test_x2_output_canvas_is_twice_entered_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 1, "text": "city"}]))
            output = root / "out"
            sizes = []
            def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
                path.write_bytes(b"image")
            def fake_render(audio, scenes, srt, path, width, height, fps, **kwargs):
                sizes.append(("mp4", width, height))
            def fake_draft(audio, scenes, srt, path, width, height):
                sizes.append(("jianying", width, height))
                return path / "draft_content.json"
            with patch("audio_video.generate_image", side_effect=fake_generate), \
                 patch("audio_video.upscaler_available", return_value=True), \
                 patch("audio_video.render_mp4", side_effect=fake_render), \
                 patch("audio_video.export_jianying", side_effect=fake_draft):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "both", "--scene-mode", "fixed", "--generate-images",
                               "--drawthings-model", "test.ckpt", "--upscale-x2", "--width", "640", "--height", "384"])
            self.assertEqual(result, 0)
            self.assertEqual(sizes, [("mp4", 1280, 768), ("jianying", 1280, 768)])

    def test_scene_review_applies_explicit_presence_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            scenes = [Scene(0, 1, "city"), Scene(1, 2, "empty road")]
            def answer():
                import time
                draft = output / "scenes_draft.json"
                for _ in range(100):
                    if draft.is_file():
                        break
                    time.sleep(0.01)
                (output / "scene_review.json").write_text(json.dumps({"scenes": [
                    {"character_present": True, "prompt": "hero in city"},
                    {"character_present": False, "prompt": "empty road"},
                ]}))
            import threading
            worker = threading.Thread(target=answer)
            worker.start()
            reviewed = review_scenes(scenes, output, "高欢", poll_interval=0.01)
            worker.join()
            self.assertEqual([(s.prompt, s.character_present) for s in reviewed],
                             [("hero in city", True), ("empty road", False)])

    def test_fixed_scenes_without_llm_do_not_show_protagonist_or_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([
                {"start": 0, "end": 0.8, "text": "高欢在城中"},
                {"start": 1, "end": 1.8, "text": "空旷街道"},
            ], ensure_ascii=False), encoding="utf-8")
            profile = root / "characters" / "hero"
            profile.mkdir(parents=True)
            (profile / "front.png").write_bytes(b"reference")
            (profile / "profile.json").write_text(json.dumps({"name": "高欢", "views": {
                "front": {"status": "approved"}, "three_quarter": {"status": "empty"},
                "full_body": {"status": "empty"}}}), encoding="utf-8")
            output = root / "out"
            calls = []
            def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
                calls.append((prompt, kwargs["references"], kwargs["seed"]))
                path.write_bytes(b"generated")
            with patch("audio_video.generate_image", side_effect=fake_generate):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "assets", "--scene-mode", "fixed", "--scene-seconds", "1",
                               "--generate-images", "--drawthings-model", "flux.ckpt", "--character-profile", str(profile),
                               "--image-prompt-suffix", "cinematic light"])
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual([len(call[1]) for call in calls], [0, 0])
            self.assertNotIn("identity reference", calls[0][0])
            self.assertNotIn("identity reference", calls[1][0])
            self.assertTrue(all(call[0].endswith("cinematic light") and call[2] == 271829 for call in calls))
            self.assertFalse((output / "scenes_draft.json").exists())

    def test_fixed_scenes_with_llm_apply_automatic_presence_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 1, "text": "主角出场"}], ensure_ascii=False))
            profile = root / "profile"
            profile.mkdir()
            (profile / "front.png").write_bytes(b"reference")
            (profile / "profile.json").write_text(json.dumps({"name": "主角", "views": {
                "front": {"status": "approved"}, "three_quarter": {"status": "empty"},
                "full_body": {"status": "empty"}}}))
            answer = {"choices": [{"message": {"content": '{"presence":[{"index":0,"present":true}]}'}}]}
            calls = []
            def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
                calls.append(kwargs.get("references", []))
                path.write_bytes(b"image")
            output = root / "out"
            with patch("audio_video.post_json", return_value=answer), \
                 patch("audio_video.generate_image", side_effect=fake_generate):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "assets", "--scene-mode", "fixed", "--llm-model", "model",
                               "--generate-images", "--drawthings-model", "test.ckpt", "--character-profile", str(profile)])
            self.assertEqual(result, 0)
            self.assertEqual(len(calls[0]), 1)
            self.assertTrue(json.loads((output / "scenes.json").read_text())[0]["character_present"])

    def test_changed_seed_regenerates_existing_scene_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 0.8, "text": "city"}]))
            output = root / "out"
            calls = []
            def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
                calls.append(kwargs["seed"])
                path.write_bytes(str(kwargs["seed"]).encode())
            base = [str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                    "--output", "assets", "--scene-mode", "fixed", "--generate-images",
                    "--drawthings-model", "flux.ckpt"]
            with patch("audio_video.generate_image", side_effect=fake_generate):
                self.assertEqual(main(base), 0)
                self.assertEqual(main(base), 0)
                self.assertEqual(main(base + ["--image-seed", "271830"]), 0)
            self.assertEqual(calls, [271829, 271830])
            self.assertEqual((output / "images/scene_001.png").read_bytes(), b"271830")
            metadata = json.loads((output / "images/scene_001.request.json").read_text())
            self.assertEqual(metadata["status"], "generated")
            (output / "images/scene_001.request.json").write_text("{broken")
            with patch("audio_video.generate_image", side_effect=fake_generate):
                self.assertEqual(main(base + ["--image-seed", "271830"]), 0)
            self.assertEqual(calls, [271829, 271830, 271830])

    def test_incomplete_whisper_model_fails_with_actionable_error(self):
        with patch("audio_video.download_whisper_model", side_effect=RuntimeError("model.bin incomplete")):
            with self.assertRaisesRegex(RuntimeError, "medium.*未完整下载"):
                resolve_whisper_model("medium", Path("/tmp/whisper-cache"))

    @unittest.skipUnless(shutil.which("uconv"), "ICU uconv 未安装")
    def test_simplified_chinese_conversion_preserves_caption_boundaries(self):
        self.assertEqual(simplify_texts(["你有沒有想過", "這個世界運轉", "Hello world"]),
                         ["你有没有想过", "这个世界运转", "Hello world"])

    def test_fixed_scenes_cover_audio_even_with_silence(self):
        captions = [Caption(1.0, 3.0, "开场"), Caption(5.0, 7.0, "转折")]
        scenes = build_scenes(captions, 9.0, target_seconds=4.0)
        self.assertEqual([(s.start, s.end) for s in scenes], [(0.0, 5.0), (5.0, 9.0)])

    def test_llm_boundaries_are_checked_and_use_actual_audio_times(self):
        captions = [Caption(1, 2, "甲"), Caption(3, 4, "乙"), Caption(5, 6, "丙")]
        scenes = parse_llm_scenes(
            json.dumps({"scenes": [
                {"start_index": 0, "end_index": 1, "prompt": "A"},
                {"start_index": 2, "end_index": 2, "prompt": "B"},
            ]}), captions, 8,
        )
        self.assertEqual([(s.start, s.end, s.prompt) for s in scenes], [(0, 5, "A"), (5, 8, "B")])
        with self.assertRaises(ValueError):
            parse_llm_scenes('{"scenes":[{"start_index":1,"end_index":2,"prompt":"x"}]}', captions, 8)

    def test_srt_is_millisecond_timed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.srt"
            write_srt([Caption(0.125, 1.5, "你好")], path)
            self.assertIn("00:00:00,125 --> 00:00:01,500", path.read_text())

    def test_local_llm_http_contract(self):
        requests = []
        def fake_urlopen(request, timeout):
            payload = json.loads(request.data)
            requests.append((request.full_url, payload))
            response = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":1,"prompt":"ancient city"}]}'}}]}
            return io.BytesIO(json.dumps(response).encode())

        with patch("audio_video.urllib.request.urlopen", side_effect=fake_urlopen):
            scenes = analyze_scenes([Caption(0, 1, "甲"), Caption(1, 2, "乙")], 2, "http://localhost:1234/v1", "test-model", None)
            self.assertEqual(requests[0][1]["model"], "test-model")
            self.assertEqual((scenes[0].start, scenes[0].end), (0, 2))
            self.assertEqual(scenes[0].prompt, "ancient city")

    def test_llm_prompt_includes_global_visual_style_and_logs_exchange(self):
        traces = []

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data)
            self.assertIn("时代：北魏", payload["messages"][0]["content"])
            self.assertIn("气氛：肃杀", payload["messages"][0]["content"])
            response = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":0,"prompt":"ancient battlefield"}]}'}}]}
            return io.BytesIO(json.dumps(response).encode())

        with patch("audio_video.urllib.request.urlopen", side_effect=fake_urlopen):
            scenes = analyze_scenes([Caption(0, 1, "战场")], 1, "http://localhost:1234/v1", "test-model", None,
                                    visual_style="时代：北魏\n气氛：肃杀", trace=traces.append)
        self.assertEqual(scenes[0].prompt, "ancient battlefield")
        self.assertTrue(any("LLM 输入" in item and "时代：北魏" in item for item in traces))
        self.assertTrue(any("LLM 输出" in item and "ancient battlefield" in item for item in traces))

    def test_llm_small_terminal_omission_keeps_full_timeline_and_logs_repair(self):
        captions = [Caption(i, i + 1, f"字幕{i}") for i in range(80)]
        traces = []
        response = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":78,"prompt":"ancient city"}]}'}}]}
        with patch("audio_video.post_json", return_value=response):
            scenes = analyze_scenes(captions, 80, "http://localhost:1234/v1", "test-model", None, trace=traces.append)
        self.assertEqual([(scene.start, scene.end) for scene in scenes], [(0, 80)])
        self.assertTrue(any("漏掉末尾 1 条字幕" in item for item in traces))

    def test_llm_batches_are_bounded_by_caption_text_length(self):
        captions = [Caption(i, i + 1, "长" * 1200) for i in range(3)]
        requests = []

        def fake_post(url, payload, api_key, timeout):
            entries = json.loads(payload["messages"][1]["content"])
            requests.append(entries)
            return {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":0,"prompt":"scene"}]}'}}]}

        with patch("audio_video.post_json", side_effect=fake_post):
            scenes = analyze_scenes(captions, 3, "http://localhost:1234/v1", "test-model", None)
        self.assertEqual(len(requests), 3)
        self.assertEqual([(scene.start, scene.end) for scene in scenes], [(0, 1), (1, 2), (2, 3)])

    def test_llm_rejects_single_caption_over_batch_text_limit(self):
        with self.assertRaisesRegex(ValueError, "第 1 条字幕.*2000"):
            analyze_scenes([Caption(0, 1, "长" * 2001)], 1,
                           "http://localhost:1234/v1", "test-model", None)

    def test_llm_retries_invalid_scene_coverage_once(self):
        captions = [Caption(i, i + 1, f"字幕{i}") for i in range(5)]
        bad = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":1,"prompt":"partial"}]}'}}]}
        good = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":4,"prompt":"complete"}]}'}}]}
        with patch("audio_video.post_json", side_effect=[bad, good]) as post:
            scenes = analyze_scenes(captions, 5, "http://localhost:1234/v1", "test-model", None)
        self.assertEqual(post.call_count, 2)
        self.assertEqual([(scene.start, scene.end, scene.prompt) for scene in scenes], [(0, 5, "complete")])

    def test_llm_uses_fixed_batch_fallback_after_two_invalid_outputs(self):
        captions = [Caption(i, i + 1, f"字幕{i}") for i in range(5)]
        bad = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":1,"prompt":"partial"}]}'}}]}
        traces = []
        with patch("audio_video.post_json", return_value=bad) as post:
            scenes = analyze_scenes(captions, 5, "http://localhost:1234/v1", "test-model", None, trace=traces.append)
        self.assertEqual(post.call_count, 2)
        self.assertEqual([(scene.start, scene.end) for scene in scenes], [(0, 5)])
        self.assertTrue(any("改用固定时长分段" in item for item in traces))

    def test_llm_fallback_uses_local_time_in_later_batch(self):
        captions = [Caption(0, 1, "开场"), Caption(1, 2, "过渡"),
                    Caption(100, 101, "结尾甲"), Caption(101, 102, "结尾乙")]
        good = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":1,"prompt":"opening"}]}'}}]}
        bad = {"choices": [{"message": {"content": '{"scenes":[{"start_index":1,"end_index":2,"prompt":"invalid"}]}'}}]}
        with patch("audio_video.post_json", side_effect=[good, bad, bad]):
            scenes = analyze_scenes(captions, 102, "http://localhost:1234/v1", "test-model", None, chunk_size=2)
        self.assertEqual([(scene.start, scene.end) for scene in scenes], [(0, 100), (100, 102)])


class RenderTests(unittest.TestCase):
    def test_incomplete_parallax_uses_original_image_in_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.png"
            empty = root / "empty.png"
            foreground = root / "foreground.png"
            for path, color in ((original, "red"), (empty, "green"), (foreground, "blue@0.5")):
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                f"color=c={color}:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(path)], check=True)
            scene = Scene(0, 1, "主角骑马，手持长矛", image=str(original), background=str(empty),
                          foreground=str(foreground), motion="parallax", character_present=True)
            prepared = prepare_motion_backgrounds([scene], root / "prepared", 320, 192)[0]
            rgb = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(prepared),
                                           "-vf", "crop=1:1:8:8,format=rgb24", "-frames:v", "1",
                                           "-f", "rawvideo", "-"])
            self.assertGreater(rgb[0], 240)
            self.assertEqual(rgb[1:], bytes([0, 0]))

    def test_parallax_without_background_does_not_overlay_duplicate_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreground = root / "foreground.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=red@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(foreground)], check=True)
            scene = Scene(0, 1, "scene", image=str(foreground), character_present=True,
                          motion="parallax", foreground=str(foreground))
            self.assertEqual(prepare_foregrounds([scene], root / "prepared", 320, 192), [None])

    def test_composite_parallax_requires_complete_foreground_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreground = root / "foreground.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=red@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(foreground)], check=True)
            scene = Scene(0, 1, "主角骑马，手持长矛", image=str(foreground), character_present=True,
                          motion="parallax", background=str(foreground), foreground=str(foreground))
            self.assertEqual(prepare_foregrounds([scene], root / "prepared", 320, 192), [None])
            scene.foreground_method = "ai"
            self.assertTrue(prepare_foregrounds([scene], root / "prepared", 320, 192)[0])

    def test_fixed_scene_character_presence_uses_llm_when_configured(self):
        scenes = [Scene(0, 10, "a general enters"), Scene(10, 20, "empty street")]
        answer = {"choices": [{"message": {"content": '{"presence":[{"index":0,"present":true},{"index":1,"present":false}]}'}}]}
        with patch("audio_video.post_json", return_value=answer):
            recommend_character_presence(scenes, "http://localhost:1234/v1", "model", None, "高欢")
        self.assertEqual([s.character_present for s in scenes], [True, False])

    def test_combined_motion_effects_have_independent_axes(self):
        effects = normalize_effects({"fade": "in_out", "horizontal": "left", "vertical": "up", "zoom": "out"})
        motion = motion_for("static", effects)
        self.assertLess(motion.x_end, motion.x_start)
        self.assertGreater(motion.y_end, motion.y_start)
        self.assertGreater(motion.scale_start, motion.scale_end)
        self.assertEqual(motion.fade, "in_out")
        with self.assertRaises(ValueError):
            normalize_effects({"horizontal": "diagonal"})
        with self.assertRaises(ValueError):
            normalize_effects({"horizontal": ["left", "right"]})
        self.assertGreater(motion_for("static", {"horizontal": "left"}).scale_start, 1.2)
        self.assertGreater(motion_for("static", {"vertical": "up"}).scale_end, 1.2)
        self.assertGreater(motion_for("static", {"horizontal": "left", "zoom": "out"}).scale_end, 1.2)

    def test_combined_motion_draft_has_vertical_and_alpha_keyframes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                            "-c:a", "aac", str(audio)], check=True)
            image = root / "scene.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=448x256:d=0.04",
                            "-frames:v", "1", str(image)], check=True)
            srt = root / "sub.srt"
            write_srt([Caption(0, 2, "测试")], srt)
            scene = Scene(0, 2, "scene", str(image), motion_effects={"fade": "in_out", "vertical": "up"})
            content = json.loads(export_jianying(audio, [scene], srt, root / "draft", 320, 180).read_text())
            segment = next(t for t in content["tracks"] if t["name"] == "visual")["segments"][0]
            props = {p["property_type"]: p["keyframe_list"] for p in segment["common_keyframes"]}
            self.assertIn("KFTypePositionY", props)
            self.assertIn("KFTypeAlpha", props)
            self.assertEqual(props["KFTypeAlpha"][0]["values"], [0.0])

    def test_nearly_empty_character_mask_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = Path(tmp) / "empty.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black@0:s=320x180:d=0.04,format=rgba", "-frames:v", "1", str(mask)], check=True)
            with self.assertRaisesRegex(RuntimeError, "抠图透明度异常"):
                validate_foreground(mask)

    def test_opaque_foreground_has_clear_alpha_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "opaque.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=red:s=320x192:d=0.04", "-frames:v", "1", str(image)], check=True)
            with self.assertRaisesRegex(RuntimeError, "透明通道"):
                validate_foreground(image)

    def test_motion_plan_falls_back_to_single_image_without_foreground(self):
        self.assertEqual(choose_preset("parallax", 0, False), "pan_left")
        self.assertEqual(choose_preset("parallax", 1, True), "parallax")
        self.assertLess(motion_for("parallax").x_end, motion_for("parallax").x_start)

    def test_auto_motion_uses_model_choice_for_each_scene_and_logs_it(self):
        scenes = [Scene(0, 8, "soldiers approach"), Scene(8, 20, "reveal the wide battlefield")]
        calls = []
        traces = []
        def answer(url, payload, api_key, timeout):
            calls.append(payload)
            return {"choices": [{"message": {"content": json.dumps({"motions": [
                {"index": 0, "preset": "push_in", "reason": "紧张"},
                {"index": 1, "preset": "pull_out", "reason": "展示全景"},
            ]}, ensure_ascii=False)}}]}
        with patch("audio_video.post_json", side_effect=answer):
            recommend_scene_motions(scenes, "http://localhost:1234/v1", "model", None, trace=traces.append)
        self.assertEqual([scene.motion for scene in scenes], ["push_in", "pull_out"])
        self.assertEqual(len(calls), 1)
        self.assertIn("soldiers approach", str(calls[0]))
        self.assertIn("展示全景", "".join(traces))

    def test_auto_motion_invalid_result_uses_static_instead_of_cycling(self):
        scenes = [Scene(0, 5, "scene one"), Scene(5, 10, "scene two")]
        invalid = {"choices": [{"message": {"content": '{"motions":[{"index":0,"preset":"orbit"}]}'}}]}
        with patch("audio_video.post_json", return_value=invalid):
            recommend_scene_motions(scenes, "http://localhost:1234/v1", "model", None)
        self.assertEqual([scene.motion for scene in scenes], ["static", "static"])

    def test_pan_and_pull_out_reserve_generation_space(self):
        self.assertEqual(generation_size(320, 180, "push_in"), (320, 180))
        for preset in ("pan_left", "pan_right", "parallax"):
            self.assertEqual(generation_size(768, 512, preset), (960, 512))
        self.assertEqual(generation_size(768, 512, "pan_right", {"vertical": "up"}), (960, 640))
        self.assertEqual(generation_size(768, 512, "pan_right", {"zoom": "out"}), (960, 640))
        self.assertEqual(generation_size(768, 512, "pull_out"), (960, 640))

    def test_draft_pan_uses_visible_keyframes_and_large_prepared_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                            "-c:a", "aac", str(audio)], check=True)
            image = root / "scene.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=448x256:d=0.04",
                            "-frames:v", "1", str(image)], check=True)
            srt = root / "subtitles.srt"
            write_srt([Caption(0, 2, "测试")], srt)
            content = json.loads(export_jianying(audio, [Scene(0, 2, "scene", str(image), motion="pan_left")],
                                                srt, root / "draft", 320, 180).read_text())
            prepared = root / "draft/media/prepared/background_000.png"
            dimensions = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                                  "-show_entries", "stream=width,height", "-of", "csv=p=0",
                                                  str(prepared)], text=True).strip()
            self.assertEqual(dimensions, "448,180")
            segment = next(track for track in content["tracks"] if track["name"] == "visual")["segments"][0]
            props = {item["property_type"]: item["keyframe_list"] for item in segment["common_keyframes"]}
            self.assertEqual([frame["values"][0] for frame in props["KFTypePositionX"]], [0.24, -0.24])
            self.assertEqual([frame["values"][0] for frame in props["KFTypeScaleX"]], [1.4, 1.4])

    def test_horizontal_pan_crops_wide_source_without_vertical_zoom(self):
        from audio_video import render_motion_clip
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, clip = root / "wide.png", root / "pan.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=green:s=448x192",
                            "-vf", "drawbox=x=0:y=12:w=448:h=16:color=blue:t=fill,"
                                   "drawbox=x=216:y=50:w=16:h=90:color=red:t=fill",
                            "-frames:v", "1", str(source)], check=True)
            render_motion_clip(source, None, Scene(0, 1, "scene", motion="pan_right"), clip, 320, 192, 10, 10)
            raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(clip), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
            def pixel(frame, x, y):
                offset = ((frame * 192 + y) * 320 + x) * 3
                return raw[offset:offset + 3]
            self.assertGreater(pixel(0, 122, 96)[0], 220)
            self.assertGreater(pixel(9, 198, 96)[0], 220)
            self.assertGreater(pixel(0, 160, 18)[2], 220)
            self.assertGreater(pixel(9, 160, 18)[2], 220)

    def test_failed_image_generation_disables_invisible_black_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 1, "text": "battlefield"}]))
            output = root / "out"
            with patch("audio_video.generate_image", side_effect=RuntimeError("generation failed")):
                code = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                             "--output", "assets", "--scene-mode", "fixed", "--motion-preset", "push_in",
                             "--generate-images", "--drawthings-model", "test.ckpt"])
            self.assertEqual(code, 0)
            scene = json.loads((output / "scenes.json").read_text())[0]
            self.assertEqual(scene["motion"], "static")
            self.assertIn("无图片", scene["motion_reason"])

    def test_parallax_draft_has_separate_editable_tracks_and_keyframes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                            "-c:a", "aac", str(audio)], check=True)
            background, foreground = root / "background.png", root / "foreground.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=0.04",
                            "-frames:v", "1", str(background)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red@0.5:s=320x180:d=0.04,format=rgba",
                            "-frames:v", "1", str(foreground)], check=True)
            srt = root / "subtitles.srt"
            write_srt([Caption(0, 2, "测试")], srt)
            scene = Scene(0, 2, "scene", str(background), True, "parallax", str(background), str(foreground),
                          motion_effects={"fade": "in_out", "vertical": "up", "zoom": "out"},
                          foreground_effects={"vertical": "down", "zoom": "in"})
            draft_file = export_jianying(audio, [scene], srt, root / "draft", 320, 180)
            content = json.loads(draft_file.read_text(encoding="utf-8"))
            tracks = {track["name"]: track for track in content["tracks"]}
            self.assertEqual(len(tracks["visual"]["segments"]), 1)
            self.assertEqual(len(tracks["foreground"]["segments"]), 1)
            for name in ("visual", "foreground"):
                segment = tracks[name]["segments"][0]
                properties = {item["property_type"]: item["keyframe_list"] for item in segment["common_keyframes"]}
                self.assertIn("KFTypePositionX", properties)
                self.assertIn("KFTypePositionY", properties)
                self.assertIn("KFTypeAlpha", properties)
                self.assertIn("KFTypeScaleX", properties)
                self.assertEqual([item["time_offset"] for item in properties["KFTypePositionX"]], [0, 2_000_000])

    def test_motion_mp4_renders_background_and_transparent_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            background, foreground = root / "background.png", root / "foreground.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=0.1",
                            "-frames:v", "1", str(background)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red@0.5:s=320x180:d=0.04,format=rgba",
                            "-frames:v", "1", str(foreground)], check=True)
            srt = root / "sub.srt"
            write_srt([Caption(0, 1, "测试")], srt)
            scene = Scene(0, 1, "scene", str(background), True, "parallax", str(background), str(foreground))
            output = root / "video.mp4"
            render_mp4(audio, [scene], srt, output, 320, 180, 10)
            probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                             "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1", str(output)], text=True)
            self.assertEqual(int(probe.strip()), 10)
            def frame(at):
                return subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(output),
                                                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            self.assertNotEqual(frame(0.05), frame(0.85))

    def test_pipeline_builds_layered_assets_and_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 1, "text": "人物出场"}]))
            background, foreground = root / "background.png", root / "foreground.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x192:d=0.1",
                            "-frames:v", "1", str(background)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red@0.5:s=320x192:d=0.04,format=rgba",
                            "-frames:v", "1", str(foreground)], check=True)
            generated = []
            def fake_generate(prompt, path, model, width, height, steps, **kwargs):
                generated.append((path.name, model, kwargs, prompt))
                shutil.copy2(foreground if path.name.endswith("_foreground.pending.png") else background, path)
            from audio_video import run as original_run
            def fake_run(command, **kwargs):
                if Path(command[0]).name == "person_cutout":
                    shutil.copy2(foreground, command[2])
                    return subprocess.CompletedProcess(command, 0)
                return original_run(command, **kwargs)
            output = root / "out"
            def recommend(scenes, *args, **kwargs):
                for scene in scenes:
                    scene.motion = "push_in"
            with patch("audio_video.analyze_scenes", return_value=[Scene(0, .5, "traveler in city"), Scene(.5, 1, "street")]), \
                 patch("audio_video.recommend_scene_motions", side_effect=recommend), \
                 patch("audio_video.generate_image", side_effect=fake_generate), \
                 patch("audio_video.list_models", return_value=[{"id": "qwen_image_2.1_turbo_i8x.ckpt"}]), \
                 patch("audio_video.run", side_effect=fake_run):
                code = main([str(audio), "--output-dir", str(output), "--output", "both", "--transcript-json", str(transcript),
                             "--scene-mode", "llm", "--llm-model", "test", "--generate-images", "--drawthings-model", "test.ckpt",
                             "--motion-preset", "auto", "--parallax-placement", "odd", "--width", "320", "--height", "180", "--fps", "10",
                             "--image-prompt-suffix", "cinematic light"])
            self.assertEqual(code, 0)
            self.assertEqual([scene["motion"] for scene in json.loads((output / "scenes.json").read_text())],
                             ["parallax", "push_in"])
            self.assertTrue((output / "video.mp4").is_file())
            content = json.loads((output / "jianying_draft/draft_content.json").read_text())
            tracks = {track["name"]: track for track in content["tracks"]}
            def first_keyframes(track):
                return {entry["property_type"]: [frame["values"][0] for frame in entry["keyframe_list"]]
                        for entry in tracks[track]["segments"][0]["common_keyframes"]}
            background_frames = first_keyframes("visual")
            foreground_frames = first_keyframes("foreground")
            self.assertEqual(background_frames["KFTypePositionX"], [-0.24, 0.24])
            self.assertNotIn("KFTypeAlpha", background_frames)
            self.assertEqual(foreground_frames["KFTypePositionX"], [0.0, 0.0])
            self.assertEqual(foreground_frames["KFTypeAlpha"], [0.0, 1.0, 1.0, 0.0])
            edit_call = next((model, kwargs, prompt) for name, model, kwargs, prompt in generated
                             if name.endswith("_background.pending.png"))
            self.assertEqual(edit_call[0], "qwen_image_2.1_turbo_i8x.ckpt")
            self.assertEqual(edit_call[1]["references"], [(output / "images/scene_001.png").resolve()])
            self.assertEqual(edit_call[1]["reference_mode"], "instruction")
            self.assertEqual(edit_call[2], "把 image 1 前景删除，只保留背景，并补全背景")
            background_request = json.loads((output / "images/scene_001_background.request.json").read_text())
            self.assertEqual((background_request["width"], background_request["height"]), (448, 180))
            foreground_call = next((kwargs, prompt) for name, _, kwargs, prompt in generated
                                   if name.endswith("_foreground.pending.png"))
            self.assertEqual(foreground_call[0]["references"], [(output / "images/scene_001.png").resolve()])
            self.assertEqual(foreground_call[0]["reference_mode"], "instruction")
            self.assertEqual(foreground_call[1], "把image1 背景透明")

    def test_composite_subject_opaque_ai_foreground_falls_back_to_single_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1",
                            "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 1, "text": "主角骑马"}]))
            profile = root / "character"
            profile.mkdir()
            (profile / "front.png").write_bytes(b"reference")
            (profile / "profile.json").write_text(json.dumps({"name": "主角", "views": {
                "front": {"status": "approved"}, "three_quarter": {"status": "empty"},
                "full_body": {"status": "empty"}}}))
            opaque = root / "opaque.png"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x192:d=0.04",
                            "-frames:v", "1", str(opaque)], check=True)
            calls = []
            def fake_generate(prompt, path, model, width, height, steps, **kwargs):
                calls.append(kwargs)
                shutil.copy2(opaque, path)
            output = root / "out"
            with patch("audio_video.analyze_scenes", return_value=[Scene(0, 1, "主角骑马，手持长矛", character_present=True)]), \
                 patch("audio_video.generate_image", side_effect=fake_generate), \
                 patch("audio_video.list_models", return_value=[{"id": "qwen_image_2.1_turbo_i8x.ckpt"}]), \
                 patch("audio_video.upscaler_available", return_value=True):
                code = main([str(audio), "--output-dir", str(output), "--output", "assets", "--transcript-json", str(transcript),
                             "--scene-mode", "llm", "--llm-model", "test", "--generate-images", "--drawthings-model", "test.ckpt",
                             "--character-profile", str(profile), "--motion-preset", "parallax", "--width", "320", "--height", "192",
                             "--upscale-x2"])
            self.assertEqual(code, 0)
            scene = json.loads((output / "scenes.json").read_text())[0]
            self.assertNotEqual(scene["motion"], "parallax")
            self.assertIsNone(scene["foreground"])
            self.assertEqual(calls[-1]["reference_mode"], "instruction")
            self.assertTrue(calls[-1]["upscale_x2"])

    def test_jianying_clamps_audio_to_its_reported_material_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.000084",
                            "-c:a", "pcm_s16le", str(audio)], check=True)
            duration = audio_duration(audio)
            srt = root / "subtitles.srt"
            write_srt([Caption(0, 1, "测试")], srt)
            draft_file = export_jianying(audio, [Scene(0, duration, "")], srt, root / "draft", 320, 180)
            content = json.loads(draft_file.read_text(encoding="utf-8"))
            narration = next(track for track in content["tracks"] if track["type"] == "audio")
            self.assertEqual(narration["segments"][0]["target_timerange"]["duration"], 1_000_000)

    def test_jianying_three_consecutive_scenes_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "aac", str(audio)], check=True)
            srt = root / "subtitles.srt"
            write_srt([Caption(0, 1, "甲"), Caption(1, 2, "乙"), Caption(2, 3, "丙")], srt)
            draft_file = export_jianying(audio, [Scene(0, 1, "A"), Scene(1, 2, "B"), Scene(2, 3, "C")],
                                         srt, root / "draft", 320, 180)
            content = json.loads(draft_file.read_text(encoding="utf-8"))
            visual = next(track for track in content["tracks"] if track["type"] == "video")
            ranges = [segment["target_timerange"] for segment in visual["segments"]]
            self.assertEqual([(item["start"], item["duration"]) for item in ranges],
                             [(0, 1_000_000), (1_000_000, 1_000_000), (2_000_000, 1_000_000)])

    @unittest.skipUnless(shutil.which("uconv"), "ICU uconv 未安装")
    def test_reused_traditional_transcript_outputs_simplified_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 0.8, "text": "你有沒有想過這個世界"}]), encoding="utf-8")
            output = root / "out"
            self.assertEqual(main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output), "--output", "assets"]), 0)
            self.assertEqual((output / "transcript.txt").read_text(), "你有没有想过这个世界")
            self.assertIn("你有没有想过这个世界", (output / "subtitles.srt").read_text())
            self.assertIn("你有没有想过这个世界", (output / "transcript.json").read_text())
            self.assertEqual(json.loads((output / "progress.json").read_text())["percent"], 100)

    def test_fixed_scene_prompts_are_printed_to_task_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 0.8, "text": "ancient city"}]), encoding="utf-8")
            output = root / "out"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "assets", "--scene-mode", "fixed"])
            self.assertEqual(result, 0)
            self.assertIn("场景 1 [0.00s - 1.00s] 提示词：", stdout.getvalue())
            self.assertIn("Scene: ancient city", stdout.getvalue())

    def test_llm_exchange_is_saved_to_output_trace_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 0.8, "text": "战场"}], ensure_ascii=False), encoding="utf-8")
            output = root / "out"

            def fake_urlopen(request, timeout):
                response = {"choices": [{"message": {"content": '{"scenes":[{"start_index":0,"end_index":0,"prompt":"ancient battlefield"}]}'}}]}
                return io.BytesIO(json.dumps(response).encode())

            with patch("audio_video.urllib.request.urlopen", side_effect=fake_urlopen):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "assets", "--scene-mode", "llm", "--llm-model", "test-model",
                               "--visual-style", "时代：北魏；气氛：肃杀"])
            self.assertEqual(result, 0)
            trace = (output / "llm_trace.log").read_text(encoding="utf-8")
            self.assertIn("LLM 输入", trace)
            self.assertIn("时代：北魏", trace)
            self.assertIn("LLM 输出", trace)
            self.assertIn("ancient battlefield", trace)

    def test_cli_registers_draft_with_v11_entry_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 0.8, "text": "剪映测试"}]), encoding="utf-8")
            draft_root = root / "drafts"
            draft_root.mkdir()
            (draft_root / "root_meta_info.json").write_text(json.dumps({"all_draft_store": [{"draft_id": "old", "draft_name": "old"}], "draft_ids": 17, "root_path": str(draft_root)}))
            output = root / "output"
            with patch("jianying_register.installed_version", return_value="11.7.0"):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "jianying", "--register-jianying", "--jianying-drafts-dir", str(draft_root),
                               "--draft-name", "验证草稿", "--width", "320", "--height", "180"])
            self.assertEqual(result, 0)
            self.assertTrue((draft_root / "验证草稿" / "draft_info.json").is_file())
            self.assertEqual(Path((output / "jianying_draft_path.txt").read_text().strip()).resolve(), (draft_root / "验证草稿").resolve())
            self.assertEqual(json.loads((draft_root / "root_meta_info.json").read_text())["draft_ids"], 17)

    def test_preview_limit_trims_source_audio_and_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(audio)], check=True)
            transcript = root / "captions.json"
            transcript.write_text(json.dumps([{"start": 0, "end": 0.8, "text": "测试"}]), encoding="utf-8")
            output = root / "out"
            result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output), "--max-seconds", "1", "--width", "320", "--height", "180", "--fps", "10"])
            self.assertEqual(result, 0)
            duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", str(output / "video.mp4")], text=True))
            self.assertAlmostEqual(duration, 1, delta=0.15)

    def test_ffmpeg_decodes_audio_for_whisper_without_pyav(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "tone.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)], check=True)
            pcm = decode_audio_ffmpeg(audio)
            self.assertGreaterEqual(len(pcm), 16000 * 4)
            self.assertLess(len(pcm), 17000 * 4)

    def test_black_fallback_renders_audio_and_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            out = root / "video.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(audio)], check=True)
            srt = root / "sub.srt"
            write_srt([Caption(0, 1.5, "测试")], srt)
            updates = []
            render_mp4(audio, [Scene(0, 2, "", None)], srt, out, 320, 180, 10, progress=updates.append)
            self.assertTrue(any(0 < value <= 1 for value in updates))
            probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)], text=True)
            self.assertIn("video", probe)
            self.assertIn("audio", probe)

    def test_scene_images_switch_at_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "voice.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(audio)], check=True)
            images = []
            for color in ("red", "blue"):
                path = root / f"{color}.png"
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s=64x64:d=0.04", "-frames:v", "1", str(path)], check=True)
                images.append(path)
            srt = root / "sub.srt"
            write_srt([Caption(0, 2, "测试")], srt)
            output = root / "video.mp4"
            render_mp4(audio, [Scene(0, 1, "", str(images[0])), Scene(1, 2, "", str(images[1]))], srt, output, 320, 180, 10)
            def pixel(at):
                return subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(output), "-frames:v", "1", "-vf", "crop=1:1:160:90,format=rgb24", "-f", "rawvideo", "pipe:1"])[:3]
            self.assertGreater(pixel(0.5)[0], 200)
            self.assertGreater(pixel(1.5)[2], 200)


if __name__ == "__main__":
    unittest.main()
