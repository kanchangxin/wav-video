import io
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from web_app import create_app


class SceneEditorTests(unittest.TestCase):
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
        self.assertEqual(self.client.get("/api/jobs/testjob/editor/tasks").status_code, 200)

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
        original_run = subprocess.run
        def fake_cutout(command, **kwargs):
            if Path(command[0]).name == "person_cutout":
                shutil.copy2(replacement_mask, command[2])
                return subprocess.CompletedProcess(command, 0)
            return original_run(command, **kwargs)
        with patch("scene_editor.subprocess.run", side_effect=fake_cutout):
            submitted = self.client.post("/api/jobs/testjob/editor/candidates/upload", data={
                "scene": "1", "layer": "image", "file": (io.BytesIO(replacement.read_bytes()), "replacement.png")}).get_json()
            for _ in range(100):
                task = self.client.get(f"/api/jobs/testjob/editor/candidates/{submitted['id']}").get_json()
                if task["status"] in ("ready", "failed"):
                    break
                time.sleep(0.02)
        self.assertEqual(task["status"], "ready", task)
        self.assertEqual(self.client.post(f"/api/jobs/testjob/editor/candidates/{task['id']}/accept").status_code, 200)
        self.assertEqual(self.image.read_bytes(), replacement.read_bytes())
        self.assertEqual(foreground.read_bytes(), replacement_mask.read_bytes())

    def test_motion_update_rebuilds_mp4_and_jianying_draft_in_queue(self):
        manager = self.app.extensions["job_manager"]
        manager.jobs["testjob"]["options"]["output"] = "both"
        manager._persist(manager.jobs["testjob"])
        response = self.client.patch("/api/jobs/testjob/editor/motion/1/image", json={
            "effects": {"fade": "in_out", "vertical": "up", "zoom": "out"}})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        rebuild = response.get_json()["rebuild"]
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
        task_id = response.get_json()["rebuild"]["id"]
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
            task_id = response.get_json()["rebuild"]["id"]
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
