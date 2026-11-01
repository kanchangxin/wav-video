import json
import io
import subprocess
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from audio_video import Caption, Scene, analyze_scenes, audio_duration, build_scenes, decode_audio_ffmpeg, export_jianying, main, parse_llm_scenes, resolve_whisper_model, write_srt, render_mp4, simplify_texts, compose_scene_prompt, image_request_fingerprint, review_scenes


class TimelineTests(unittest.TestCase):
    def test_identity_prompt_only_applies_when_protagonist_is_present(self):
        with_character = compose_scene_prompt("ancient battlefield", "cinematic light", "高欢", True, True)
        without_character = compose_scene_prompt("empty battlefield", "cinematic light", "高欢", False, True)
        self.assertIn("Use image 1 as the identity reference", with_character)
        self.assertIn("高欢", with_character)
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

    def test_pipeline_sends_reference_only_for_confirmed_protagonist_scene(self):
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
            import threading
            def confirm():
                import time
                for _ in range(100):
                    if (output / "scenes_draft.json").is_file():
                        break
                    time.sleep(0.01)
                (output / "scene_review.json").write_text(json.dumps({"scenes": [
                    {"prompt": "hero scene", "character_present": True},
                    {"prompt": "empty street", "character_present": False},
                ]}))
            calls = []
            def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
                calls.append((prompt, kwargs["references"], kwargs["seed"]))
                path.write_bytes(b"generated")
            worker = threading.Thread(target=confirm)
            worker.start()
            with patch("audio_video.generate_image", side_effect=fake_generate):
                result = main([str(audio), "--transcript-json", str(transcript), "--output-dir", str(output),
                               "--output", "assets", "--scene-mode", "fixed", "--scene-seconds", "1",
                               "--generate-images", "--drawthings-model", "flux.ckpt", "--character-profile", str(profile),
                               "--review-scenes", "--image-prompt-suffix", "cinematic light"])
            worker.join()
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual([len(call[1]) for call in calls], [1, 0])
            self.assertIn("identity reference", calls[0][0])
            self.assertNotIn("identity reference", calls[1][0])
            self.assertTrue(all(call[0].endswith("cinematic light") and call[2] == 271829 for call in calls))

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
