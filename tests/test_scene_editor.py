import io
import json
import shlex
import shutil
import subprocess
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from web_app import create_app


class SceneEditorTests(unittest.TestCase):
    def test_saves_batch_into_one_sync_and_concurrent_edits_stay_dirty(self):
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"]["options"]["output"] = "both"
        manager.jobs["testjob"]["registered_draft"] = str(self.root / "drafts" / "old")
        for effects in ({"fade": "in"}, {"zoom": "out"}):
            response = self.client.patch("/api/jobs/testjob/editor/motion/1/image", json={"effects": effects})
            self.assertIsNone(response.get_json()["rebuild"])
        self.client.patch("/api/jobs/testjob/editor/captions/1", json={"captions": [{"index": 0, "text": "合并字幕"}]})
        self.assertEqual(self.client.get("/api/jobs/testjob/editor/tasks").get_json()["tasks"], [])
        started, release = threading.Event(), threading.Event()
        def render(audio, scenes, srt, output, width, height, fps):
            started.set()
            release.wait(5)
            output.write_bytes(b"video")
        with patch("scene_editor.render_mp4", side_effect=render), patch("scene_editor.register_draft", return_value=self.root / "drafts" / "new") as register:
            first = self.client.post("/api/jobs/testjob/editor/sync-jianying")
            self.assertEqual(first.status_code, 202)
            self.assertTrue(started.wait(3))
            second = self.client.post("/api/jobs/testjob/editor/sync-jianying")
            self.assertEqual(first.get_json()["id"], second.get_json()["id"])
            self.client.patch("/api/jobs/testjob/editor/captions/1", json={"captions": [{"index": 0, "text": "下一次修改"}]})
            release.set()
            for _ in range(150):
                task = self.client.get('/api/jobs/testjob/editor/candidates/' + first.get_json()["id"]).get_json()
                if task["status"] in ("completed", "failed"):
                    break
                time.sleep(0.02)
            self.assertEqual(task["status"], "completed", task)
            self.assertEqual(register.call_count, 1)
        draft = Path((self.output / "jianying_draft_path.txt").read_text().strip())
        self.assertIn("合并字幕", (draft / "draft_content.json").read_text())
        self.assertNotIn("下一次修改", (draft / "draft_content.json").read_text())
        state = self.client.get("/api/jobs/testjob/editor/tasks").get_json()
        self.assertTrue(state["needs_sync"])
        self.assertEqual(len(state["tasks"]), 1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.audio = self.root / "voice.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-c:a", "aac", str(self.audio)], check=True)
        self.app = create_app(self.root)
        self.client = self.app.test_client()
        self.output = self.root / "runs" / "testjob"
        images = self.output / "images"
        images.mkdir(parents=True)
        self.image = images / "scene_001.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x192:d=0.04",
                        "-frames:v", "1", str(self.image)], check=True)
        self.original = self.image.read_bytes()
        (self.output / "scenes.json").write_text(json.dumps([{"start": 0, "end": 1, "prompt": "red city",
            "image": str(self.image), "character_present": False, "motion": "static", "background": None,
            "foreground": None, "motion_reason": "", "motion_effects": {}, "foreground_effects": {}}]))
        (self.output / "image_generation.json").write_text(json.dumps([{"scene": 1, "prompt": "red city",
            "model": "test.ckpt", "width": 320, "height": 192, "steps": 20, "seed": 271829, "references": []}]))
        (self.output / "subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntest\n")
        (self.output / "transcript.json").write_text(json.dumps({"audio": str(self.audio), "duration": 1,
            "captions": [{"start": 0, "end": 1, "text": "test"}]}))
        (self.output / "transcript.txt").write_text("test")
        state = {"id": "testjob", "status": "completed", "created_at": "2026-11-01T00:00:00",
                 "finished_at": "2026-11-01T00:01:00", "exit_code": 0, "audio_id": "root:voice.m4a",
                 "audio_name": "voice.m4a", "options": {"output": "assets", "width": 320, "height": 192, "fps": 10}}
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"] = state
        manager._persist(state)

    def tearDown(self):
        self.tmp.cleanup()

    def test_inventory_exposes_preview_and_editable_full_command(self):
        response = self.client.get("/api/jobs/testjob/editor")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        layer = response.get_json()["scenes"][0]["layers"][0]
        self.assertIn("--prompt 'red city'", layer["command"])
        self.assertIn("--width 320", layer["command"])
        with self.client.get(layer["url"]) as preview:
            self.assertEqual(preview.status_code, 200)
        page = self.client.get("/jobs/testjob/scenes")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn("场景图片编辑", html)
        self.assertIn("重新生成候选图", html)
        self.assertIn("同步到剪映", html)
        self.assertIn('id="scene-nav"', html)
        self.assertIn('id="scene-current"', html)
        self.assertIn('id="scene-search"', html)
        self.assertIn('本场景字幕', html)
        self.assertIn('用原图补空背景', html)
        self.assertIn('AI 提取整组前景', html)
        self.assertIn('从原图抠人物', html)
        self.assertIn('缺少空背景', html)
        self.assertIn('前景未包含坐骑或道具', html)
        self.assertNotIn('id="reference-paths"', html)
        self.assertEqual(self.client.get("/api/jobs/testjob/editor/tasks").status_code, 200)

    def test_parallax_suggestions_use_source_image_and_ai_foreground_keeps_source(self):
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0].update({"motion": "parallax", "character_present": True,
                          "prompt": "主角骑马，手持长矛"})
        (self.output / "scenes.json").write_text(json.dumps(scenes, ensure_ascii=False))
        self.app.extensions["job_manager"].jobs["testjob"]["options"]["image_prompt_suffix"] = "积雪覆盖军营"
        request_path = self.output / "images" / "scene_001_foreground.request.json"
        request_path.write_text(json.dumps({
            "method": "ai_edit", "references": [str(self.image)],
            "prompt": "以 image 1 为原图，精确提取前景主体：主角及其坐骑、手持武器和相关道具。"
                      "保留原有面容、服饰、姿态、相对位置和轮廓。删除所有背景，背景像素必须真正透明（PNG alpha），"
                      "不要绘制棋盘格、纯色背景、文字或新物体；主体保持原图中的位置和大小。"
                      "\n场景设定：主角骑马，手持长矛\n积雪覆盖军营",
        }, ensure_ascii=False))
        inventory = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]
        background = next(layer for layer in inventory["layers"] if layer["name"] == "background")
        foreground = next(layer for layer in inventory["layers"] if layer["name"] == "foreground")
        self.assertIn("--image", background["source_edit_command"])
        self.assertIn(str(self.image.resolve()), background["source_edit_command"])
        self.assertIn("--image", foreground["ai_edit_command"])
        for command in (foreground["command"], foreground["ai_edit_command"]):
            tokens = shlex.split(command)
            self.assertEqual(tokens[tokens.index("--prompt") + 1], "把image1 背景透明")
            self.assertEqual(tokens[tokens.index("--image") + 1], str(self.image.resolve()))
        transparent = self.root / "transparent.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(transparent)], check=True)
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: shutil.copy2(transparent, path)) as generated:
            response = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "foreground", "command": foreground["command"].replace("--width 320", "--width 384")})
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            task_id = response.get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertTrue(task["params"]["mask_from_source"])
        candidate_rgb = subprocess.check_output(["ffmpeg", "-v", "error", "-i", task["candidate"],
                                                  "-vf", "crop=1:1:8:8,format=rgb24", "-frames:v", "1",
                                                  "-f", "rawvideo", "-"])
        self.assertGreater(candidate_rgb[0], 240)
        self.assertEqual(candidate_rgb[1:], bytes([0, 0]))
        self.assertEqual(generated.call_args.kwargs["reference_mode"], "instruction")
        self.assertEqual(self.client.post(f"/api/jobs/testjob/editor/candidates/{task_id}/accept").status_code, 200)
        self.assertEqual(self.image.read_bytes(), self.original)
        updated = next(layer for layer in self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"]
                       if layer["name"] == "foreground")
        self.assertIn("--width 384", updated["command"])

    def test_cutout_from_scene_queues_foreground_candidate(self):
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0]["motion"] = "parallax"
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        transparent = self.root / "transparent.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=red@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(transparent)], check=True)
        real_run = subprocess.run
        def fake_cutout(command, **kwargs):
            if Path(command[0]).name != "person_cutout":
                return real_run(command, **kwargs)
            shutil.copy2(transparent, command[2])
            return subprocess.CompletedProcess(command, 0)
        with patch("scene_editor.subprocess.run", side_effect=fake_cutout):
            response = self.client.post("/api/jobs/testjob/editor/candidates/cutout", json={"scene": 1})
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            task_id = response.get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(task["layer"], "foreground")

    def test_background_source_edit_stays_candidate_until_accepted(self):
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0]["motion"] = "parallax"
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        background = next(layer for layer in self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"]
                          if layer["name"] == "background")
        expected_prompt = "把 image 1 前景删除，只保留背景，并补全背景"
        self.assertEqual(background["references"], [str(self.image.resolve())])
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: shutil.copy2(self.image, path)) as generated:
            queued = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "background", "command": background["source_edit_command"]})
            self.assertEqual(queued.status_code, 202, queued.get_data(as_text=True))
            task_id = queued.get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(generated.call_args.args[0], expected_prompt)
        self.assertEqual(generated.call_args.args[3:5], (448, 192))
        self.assertEqual(generated.call_args.kwargs["references"], [self.image.resolve()])
        self.assertEqual(generated.call_args.kwargs["reference_mode"], "instruction")
        self.assertIsNone(json.loads((self.output / "scenes.json").read_text())[0]["background"])
        self.assertEqual(self.client.post(f"/api/jobs/testjob/editor/candidates/{task_id}/accept").status_code, 200)
        self.assertTrue(json.loads((self.output / "scenes.json").read_text())[0]["background"])

    def test_scene_subtitle_edit_updates_transcript_and_srt(self):
        scene = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]
        self.assertEqual(scene["captions"], [{"index": 0, "start": 0, "end": 1, "text": "test"}])
        saved = self.client.patch("/api/jobs/testjob/editor/captions/1", json={
            "captions": [{"index": 0, "text": "修正错字"}]})
        self.assertEqual(saved.status_code, 200, saved.get_data(as_text=True))
        self.assertEqual(json.loads((self.output / "transcript.json").read_text())["captions"][0]["text"], "修正错字")
        self.assertIn("修正错字", (self.output / "subtitles.srt").read_text())
        self.assertEqual((self.output / "transcript.txt").read_text(), "修正错字")
        invalid = self.client.patch("/api/jobs/testjob/editor/captions/1", json={
            "captions": [{"index": 99, "text": "不应保存"}]})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual((self.output / "transcript.txt").read_text(), "修正错字")

    def test_subtitle_edit_rebuilds_video_and_jianying_draft(self):
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"]["options"]["output"] = "both"
        manager._persist(manager.jobs["testjob"])
        saved = self.client.patch("/api/jobs/testjob/editor/captions/1", json={
            "captions": [{"index": 0, "text": "新字幕"}]})
        self.assertEqual(saved.status_code, 200, saved.get_data(as_text=True))
        self.assertIsNone(saved.get_json()["rebuild"])
        task_id = self.client.post("/api/jobs/testjob/editor/rebuild").get_json()["id"]
        for _ in range(200):
            task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
            if task["status"] in ("completed", "failed"):
                break
            time.sleep(0.025)
        self.assertEqual(task["status"], "completed", task)
        self.assertTrue((self.output / "video.mp4").is_file())
        draft = Path((self.output / "jianying_draft_path.txt").read_text().strip())
        self.assertIn("新字幕", (draft / "draft_content.json").read_text())

    def test_edited_reference_paths_reach_generation_and_are_saved_on_accept(self):
        reference = self.root / "new reference.png"
        shutil.copy2(self.image, reference)
        layer = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        self.assertEqual(layer["references"], [])
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)) as generate:
            response = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "image", "command": layer["command"],
                "references": [str(reference)]})
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            task_id = response.get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(generate.call_args.kwargs["references"], [reference.resolve()])
        accepted = self.client.post(f"/api/jobs/testjob/editor/candidates/{task_id}/accept")
        self.assertEqual(accepted.status_code, 200, accepted.get_data(as_text=True))
        record = json.loads((self.output / "image_generation.json").read_text())[0]
        self.assertEqual(record["references"], [str(reference.resolve())])
        updated = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        self.assertIn(str(reference.resolve()), updated["command"])

    def test_reference_path_can_be_changed_in_full_command_and_must_be_an_image(self):
        reference = self.root / "reference.png"
        shutil.copy2(self.image, reference)
        layer = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)) as generate:
            changed = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "image", "command": layer["command"] + f" --image {reference}"})
            self.assertEqual(changed.status_code, 202, changed.get_data(as_text=True))
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{changed.get_json()['id']}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(generate.call_args.kwargs["references"], [reference.resolve()])
        invalid = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
            "scene": 1, "layer": "image", "command": layer["command"],
            "references": [str(self.root / "missing.png")]})
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("参考图", invalid.get_json()["error"])

    def test_background_reference_path_persists_after_accept(self):
        reference = self.root / "plate_reference.png"
        shutil.copy2(self.image, reference)
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0]["motion"] = "parallax"
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        background = next(layer for layer in self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"]
                          if layer["name"] == "background")
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)):
            queued = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "background", "command": background["command"],
                "references": ["plate_reference.png"]})
            self.assertEqual(queued.status_code, 202, queued.get_data(as_text=True))
            task_id = queued.get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(self.client.post(f"/api/jobs/testjob/editor/candidates/{task_id}/accept").status_code, 200)
        updated = next(layer for layer in self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"]
                       if layer["name"] == "background")
        self.assertEqual(updated["references"], [str(reference.resolve())])

    def test_inventory_supports_more_than_one_hundred_scenes(self):
        scenes = json.loads((self.output / "scenes.json").read_text())
        scene = scenes[0]
        scenes = [{**scene, "start": index, "end": index + 1, "prompt": f"scene {index + 1}"}
                  for index in range(130)]
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        response = self.client.get("/api/jobs/testjob/editor")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(len(response.get_json()["scenes"]), 130)

    def test_reference_paths_can_be_cleared_for_character_scene(self):
        snapshot = self.output / "character_reference"
        snapshot.mkdir()
        shutil.copy2(self.image, snapshot / "front.png")
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0]["character_present"] = True
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        layer = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        self.assertTrue(layer["references"])
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)):
            response = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "image", "command": layer["command"], "references": []})
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            task_id = response.get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(self.client.post(f"/api/jobs/testjob/editor/candidates/{task_id}/accept").status_code, 200)
        updated = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        self.assertEqual(updated["references"], [])

    def test_missing_old_reference_can_be_replaced_from_path_field(self):
        records = json.loads((self.output / "image_generation.json").read_text())
        records[0]["references"] = [str(self.root / "deleted.png")]
        (self.output / "image_generation.json").write_text(json.dumps(records))
        replacement = self.root / "replacement.png"
        shutil.copy2(self.image, replacement)
        layer = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)) as generate:
            response = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "image", "command": layer["command"],
                "references": [str(replacement)]})
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{response.get_json()['id']}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(generate.call_args.kwargs["references"], [replacement.resolve()])

    def test_editor_exposes_and_accepts_x2_upscaler_command(self):
        records = json.loads((self.output / "image_generation.json").read_text())
        records[0].update(width=192, height=128, upscale_x2=True)
        (self.output / "image_generation.json").write_text(json.dumps(records))
        layer = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        self.assertIn("--config-json", layer["command"])
        self.assertIn("--width 320", layer["command"])
        self.assertIn("--height 192", layer["command"])
        self.assertTrue(layer["upscale_x2"])
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)) as generate:
            submitted = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "image", "command": layer["command"]})
            self.assertEqual(submitted.status_code, 202, submitted.get_data(as_text=True))
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{submitted.get_json()['id']}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertTrue(generate.call_args.kwargs["upscale_x2"])

    def test_upload_candidate_keeps_original_until_accept_and_saves_revision(self):
        blue = self.root / "blue.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x192:d=0.04",
                        "-frames:v", "1", str(blue)], check=True)
        response = self.client.post("/api/jobs/testjob/editor/candidates/upload", data={
            "scene": "1", "layer": "image", "file": (io.BytesIO(blue.read_bytes()), "blue.png")})
        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        candidate = response.get_json()
        self.assertEqual(self.image.read_bytes(), self.original)
        for _ in range(100):
            state = self.client.get(f"/api/jobs/testjob/editor/candidates/{candidate['id']}").get_json()
            if state["status"] in ("ready", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(state["status"], "ready", state)
        with self.client.get(candidate["url"]) as preview:
            self.assertEqual(preview.status_code, 200)
        accepted = self.client.post(f"/api/jobs/testjob/editor/candidates/{candidate['id']}/accept")
        self.assertEqual(accepted.status_code, 200, accepted.get_data(as_text=True))
        self.assertEqual(self.image.read_bytes(), blue.read_bytes())
        self.assertTrue(list((self.output / "edits" / "revisions").glob("*.png")))

    def test_generation_uses_whitelisted_command_and_finishes_in_background(self):
        layer = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        rejected = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
            "scene": 1, "layer": "image", "command": layer["command"] + " ; touch /tmp/unsafe"})
        self.assertEqual(rejected.status_code, 400)
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)):
            queued = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "image", "command": layer["command"]})
            self.assertEqual(queued.status_code, 202, queued.get_data(as_text=True))
            task_id = queued.get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(self.image.read_bytes(), self.original)

    def test_jpeg_upload_is_converted_to_png_candidate(self):
        jpeg = self.root / "input.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.image), "-frames:v", "1", str(jpeg)], check=True)
        submitted = self.client.post("/api/jobs/testjob/editor/candidates/upload", data={
            "scene": "1", "layer": "image", "file": (io.BytesIO(jpeg.read_bytes()), "input.jpg")}).get_json()
        for _ in range(100):
            task = self.client.get(f"/api/jobs/testjob/editor/candidates/{submitted['id']}").get_json()
            if task["status"] in ("ready", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                         "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
                                         task["candidate"]], text=True).strip()
        self.assertEqual(probe, "png")

    def test_edited_prompt_and_size_are_parsed_without_shell_execution(self):
        layer = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"][0]
        command = layer["command"].replace("'red city'", "'blue city'").replace("--width 320", "--width 448")
        with patch("scene_editor.generate_image", side_effect=lambda prompt, path, model, width, height, steps, **kw: path.write_bytes(self.original)) as generate:
            submitted = self.client.post("/api/jobs/testjob/editor/candidates/generate", json={
                "scene": 1, "layer": "image", "command": command})
            self.assertEqual(submitted.status_code, 202, submitted.get_data(as_text=True))
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{submitted.get_json()['id']}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(generate.call_args.args[0], "blue city")
        self.assertEqual(generate.call_args.args[3], 448)

    def test_parallax_layers_can_be_previewed_and_foreground_uploaded_independently(self):
        background = self.output / "images" / "scene_001_background.png"
        foreground = self.output / "images" / "scene_001_foreground.png"
        background.write_bytes(self.original)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=red@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(foreground)], check=True)
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0].update(motion="parallax", background=str(background), foreground=str(foreground))
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        layers = self.client.get("/api/jobs/testjob/editor").get_json()["scenes"][0]["layers"]
        self.assertEqual([layer["name"] for layer in layers], ["image", "background", "foreground"])
        self.assertIn("person_cutout", layers[2]["cutout_command"])
        new_foreground = self.root / "new_foreground.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(new_foreground)], check=True)
        submitted = self.client.post("/api/jobs/testjob/editor/candidates/upload", data={
            "scene": "1", "layer": "foreground", "file": (io.BytesIO(new_foreground.read_bytes()), "person.png")}).get_json()
        for _ in range(100):
            task = self.client.get(f"/api/jobs/testjob/editor/candidates/{submitted['id']}").get_json()
            if task["status"] in ("ready", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(self.client.post(f"/api/jobs/testjob/editor/candidates/{task['id']}/accept").status_code, 200)
        self.assertEqual(foreground.read_bytes(), new_foreground.read_bytes())
        self.assertEqual(self.image.read_bytes(), self.original)

    def test_replacing_parallax_source_also_refreshes_derived_foreground(self):
        background = self.output / "images" / "scene_001_background.png"
        foreground = self.output / "images" / "scene_001_foreground.png"
        background.write_bytes(self.original)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=red@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(foreground)], check=True)
        replacement_mask = self.root / "replacement_mask.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(replacement_mask)], check=True)
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0].update(motion="parallax", background=str(background), foreground=str(foreground))
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        replacement = self.root / "replacement.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=320x192:d=0.04",
                        "-frames:v", "1", str(replacement)], check=True)
        def fake_generate(prompt, path, *args, **kwargs):
            shutil.copy2(replacement_mask, path)
        with patch("scene_editor.generate_image", side_effect=fake_generate) as generated, \
             patch("scene_editor.list_models", return_value=[{"id": "qwen_image_2.1_turbo_i8x.ckpt"}]):
            submitted = self.client.post("/api/jobs/testjob/editor/candidates/upload", data={
                "scene": "1", "layer": "image", "file": (io.BytesIO(replacement.read_bytes()), "replacement.png")}).get_json()
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{submitted['id']}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(generated.call_args.args[0], "把image1 背景透明")
        self.assertEqual(generated.call_args.kwargs["reference_mode"], "instruction")
        self.assertEqual(generated.call_args.kwargs["references"], [Path(task["candidate"])])
        self.assertEqual(self.client.post(f"/api/jobs/testjob/editor/candidates/{task['id']}/accept").status_code, 200)
        self.assertEqual(self.image.read_bytes(), replacement.read_bytes())
        self.assertEqual(json.loads((self.output / "scenes.json").read_text())[0]["foreground_method"], "ai")

    def test_motion_update_rebuilds_mp4_and_jianying_draft_in_queue(self):
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"]["options"]["output"] = "both"
        manager._persist(manager.jobs["testjob"])
        response = self.client.patch("/api/jobs/testjob/editor/motion/1/image", json={
            "effects": {"fade": "in_out", "vertical": "up", "zoom": "out"}})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertIsNone(response.get_json()["rebuild"])
        rebuild = self.client.post("/api/jobs/testjob/editor/rebuild").get_json()
        for _ in range(200):
            task = self.client.get(f"/api/jobs/testjob/editor/candidates/{rebuild['id']}").get_json()
            if task["status"] in ("completed", "failed"):
                break
            time.sleep(0.025)
        self.assertEqual(task["status"], "completed", task)
        self.assertTrue((self.output / "video.mp4").is_file())
        draft = Path((self.output / "jianying_draft_path.txt").read_text().strip())
        self.assertTrue((draft / "draft_content.json").is_file())
        segment = next(t for t in json.loads((draft / "draft_content.json").read_text())["tracks"]
                       if t["name"] == "visual")["segments"][0]
        self.assertIn("KFTypeAlpha", {item["property_type"] for item in segment["common_keyframes"]})

    def test_parallax_motion_rebuild_does_not_run_cutout_on_rebuild_task(self):
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"]["options"]["output"] = "both"
        manager._persist(manager.jobs["testjob"])
        background = self.output / "images" / "scene_001_background.png"
        foreground = self.output / "images" / "scene_001_foreground.png"
        background.write_bytes(self.original)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=red@0.5:s=320x192:d=0.04,format=rgba", "-frames:v", "1", str(foreground)], check=True)
        scenes = json.loads((self.output / "scenes.json").read_text())
        scenes[0].update(motion="parallax", background=str(background), foreground=str(foreground))
        (self.output / "scenes.json").write_text(json.dumps(scenes))
        response = self.client.patch("/api/jobs/testjob/editor/motion/1/foreground", json={"effects": {"vertical": "up"}})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        task_id = self.client.post("/api/jobs/testjob/editor/rebuild").get_json()["id"]
        for _ in range(150):
            task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
            if task["status"] in ("completed", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(task["status"], "completed", task)

    def test_sync_never_uses_registered_old_copy_as_new_draft(self):
        manager = self.app.extensions["job_manager"]
        external = self.root / "jianying" / "old"
        external.mkdir(parents=True)
        (external / "draft_content.json").write_text("{}")
        manager.jobs["testjob"]["registered_draft"] = str(external)
        manager._persist(manager.jobs["testjob"])
        (self.output / "jianying_draft_path.txt").write_text(str(external))
        marker = self.output / "edits" / "needs_sync"
        marker.parent.mkdir(parents=True)
        marker.write_text("1")
        response = self.client.post("/api/jobs/testjob/editor/sync-jianying")
        self.assertEqual(response.status_code, 400)
        self.assertTrue(marker.is_file())

    def test_failed_draft_rebuild_keeps_previous_mp4(self):
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"]["options"]["output"] = "both"
        manager._persist(manager.jobs["testjob"])
        previous = self.output / "video.mp4"
        previous.write_bytes(b"previous video")
        with patch("scene_editor.render_mp4", side_effect=lambda audio, scenes, srt, output, width, height, fps: output.write_bytes(b"new video")), \
             patch("scene_editor.export_jianying", side_effect=RuntimeError("draft failed")):
            response = self.client.patch("/api/jobs/testjob/editor/motion/1/image", json={"effects": {"fade": "in"}})
            task_id = self.client.post("/api/jobs/testjob/editor/rebuild").get_json()["id"]
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{task_id}").get_json()
                if task["status"] in ("completed", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "failed", task)
        self.assertEqual(previous.read_bytes(), b"previous video")

    def test_rebuild_can_be_retried_after_failure(self):
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"]["options"]["output"] = "both"
        manager._persist(manager.jobs["testjob"])
        response = self.client.post("/api/jobs/testjob/editor/rebuild")
        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["kind"], "rebuild")


if __name__ == "__main__":
    unittest.main()
