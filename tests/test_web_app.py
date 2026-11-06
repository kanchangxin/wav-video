import io
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from web_app import JobManager, create_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.audio = self.root / "中文旁白.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(self.audio)], check=True)
        self.app = create_app(self.root)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_page_lists_existing_audio(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("开始处理", page.get_data(as_text=True))
        self.assertIn("全局画面设定", page.get_data(as_text=True))
        self.assertIn("Real-ESRGAN X2+", page.get_data(as_text=True))
        audio = self.client.get("/api/audio").get_json()["files"]
        self.assertIn({"id": "root:中文旁白.m4a", "name": "中文旁白.m4a"}, audio)

    def test_x2_option_is_validated_and_passed_to_pipeline(self):
        manager = self.app.extensions["job_manager"]
        data = {"audio_id": "root:中文旁白.m4a", "output": "assets", "scene_mode": "fixed",
                "generate_images": True, "drawthings_model": "test.ckpt", "upscale_x2": True}
        with patch("web_app.list_models", return_value=[{"id": "test.ckpt", "name": "Test"}]), \
             patch("web_app.upscaler_available", return_value=True):
            command, _, saved = manager._command(data, self.root / "runs" / "upscale-test")
        self.assertIn("--upscale-x2", command)
        self.assertTrue(saved["options"]["upscale_x2"])
        with self.assertRaisesRegex(ValueError, "放大选项"):
            manager._command({**data, "upscale_x2": "true"}, self.root / "runs" / "upscale-test")
        with patch("web_app.upscaler_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Real-ESRGAN X2"):
                manager._command(data, self.root / "runs" / "upscale-test")

    def test_character_page_and_profile_creation(self):
        self.assertEqual(self.client.get("/characters").status_code, 200)
        with patch("web_app.list_models", return_value=[{"id": "flux.ckpt", "name": "FLUX"}]):
            created = self.client.post("/api/characters", json={
                "name": "高欢", "path": "characters/gaohuan", "model": "flux.ckpt",
                "seed": 271829, "front_prompt": "portrait of a general",
            })
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        self.assertEqual(created.get_json()["name"], "高欢")
        detail = self.client.get("/api/characters/gaohuan").get_json()
        self.assertIn("高欢", detail["suggested_prompts"]["three_quarter"])
        self.assertEqual(len(self.client.get("/api/characters").get_json()["characters"]), 1)
        self.assertEqual(self.client.post("/api/characters", json={
            "name": "escape", "path": "../escape", "model": "flux.ckpt", "front_prompt": "portrait",
        }).status_code, 400)
        with patch("web_app.list_models", return_value=[{"id": "flux.ckpt", "name": "FLUX"}]):
            invalid_seed = self.client.post("/api/characters", json={
                "name": "invalid", "path": "characters/invalid", "model": "flux.ckpt",
                "seed": None, "front_prompt": "portrait",
            })
        self.assertEqual(invalid_seed.status_code, 400)
        self.assertEqual(invalid_seed.mimetype, "application/json")

    def test_character_generation_requires_front_approval_and_serves_preview(self):
        with patch("web_app.list_models", return_value=[{"id": "flux.ckpt", "name": "FLUX"},
                                                         {"id": "z_image.ckpt", "name": "Z Image"}]):
            profile = self.client.post("/api/characters", json={
                "name": "高欢", "path": "characters/gaohuan", "model": "flux.ckpt", "front_prompt": "portrait",
            }).get_json()
        profile_id = profile["id"]
        side = self.client.post(f"/api/characters/{profile_id}/generate", json={"view": "three_quarter", "prompt": "side"})
        self.assertEqual(side.status_code, 400)
        def fake_generate(prompt, path, model, width, height, steps=None, **kwargs):
            path.write_bytes(b"preview image")
        with patch("character_profiles.generate_image", side_effect=fake_generate):
            started = self.client.post(f"/api/characters/{profile_id}/generate", json={"view": "front", "prompt": "portrait"})
            self.assertEqual(started.status_code, 202, started.get_data(as_text=True))
            for _ in range(100):
                state = self.client.get(f"/api/characters/{profile_id}").get_json()
                if state["views"]["front"]["status"] == "draft":
                    break
                time.sleep(0.01)
        self.assertEqual(state["views"]["front"]["status"], "draft")
        preview = self.client.get(f"/api/characters/{profile_id}/image/front/draft")
        self.assertEqual(preview.data, b"preview image")
        preview.close()
        approved = self.client.post(f"/api/characters/{profile_id}/approve", json={"view": "front"})
        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.get_json()["views"]["front"]["status"], "approved")

    def test_import_existing_png_rejects_invalid_image(self):
        with patch("web_app.list_models", return_value=[{"id": "flux.ckpt", "name": "FLUX"}]):
            profile = self.client.post("/api/characters", json={
                "name": "高欢", "path": "characters/gaohuan", "model": "flux.ckpt", "front_prompt": "portrait",
            }).get_json()
        profile_id = profile["id"]
        invalid = self.client.post(f"/api/characters/{profile_id}/import",
                                   data={"view": "front", "file": (io.BytesIO(b"not an image"), "bad.png")},
                                   content_type="multipart/form-data")
        self.assertEqual(invalid.status_code, 400)
        source = self.root / "valid.png"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.04",
                        "-frames:v", "1", str(source)], check=True)
        imported = self.client.post(f"/api/characters/{profile_id}/import",
                                    data={"view": "front", "file": (io.BytesIO(source.read_bytes()), "front.png")},
                                    content_type="multipart/form-data")
        self.assertEqual(imported.status_code, 200, imported.get_data(as_text=True))
        self.assertEqual(imported.get_json()["views"]["front"]["status"], "approved")

    def test_job_command_passes_character_profile_seed_and_suffix(self):
        with patch("web_app.list_models", return_value=[{"id": "flux.ckpt", "name": "FLUX"},
                                                         {"id": "z_image.ckpt", "name": "Z Image"}]):
            profile = self.client.post("/api/characters", json={
                "name": "高欢", "path": "characters/gaohuan", "model": "flux.ckpt", "front_prompt": "portrait",
            }).get_json()
            folder = self.root / "characters" / profile["id"]
            (folder / "front.png").write_bytes(b"approved")
            manifest = json.loads((folder / "profile.json").read_text())
            manifest["views"]["front"]["status"] = "approved"
            (folder / "profile.json").write_text(json.dumps(manifest))
            manager = self.app.extensions["job_manager"]
            command, _, _ = manager._command({
                "audio_id": "root:中文旁白.m4a", "output": "assets", "scene_mode": "fixed",
                "generate_images": True, "drawthings_model": "flux.ckpt",
                "character_profile_id": profile["id"], "image_seed": 271829,
                "image_prompt_suffix": "cinematic light",
            }, self.root / "runs" / "character-test")
            different_command, _, _ = manager._command({
                    "audio_id": "root:中文旁白.m4a", "output": "assets", "scene_mode": "fixed",
                    "generate_images": True, "drawthings_model": "z_image.ckpt",
                    "character_profile_id": profile["id"],
                }, self.root / "runs" / "character-test")
        self.assertNotIn("--review-scenes", command)
        self.assertEqual(command[command.index("--character-profile") + 1], str(folder.resolve()))
        self.assertEqual(command[command.index("--image-seed") + 1], "271829")
        self.assertEqual(command[command.index("--image-prompt-suffix") + 1], "cinematic light")
        self.assertEqual(different_command[different_command.index("--drawthings-model") + 1], "z_image.ckpt")

    def test_motion_preset_passes_to_pipeline_and_parallax_requires_profile(self):
        manager = self.app.extensions["job_manager"]
        command, _, _ = manager._command({"audio_id": "root:中文旁白.m4a", "output": "both",
                                          "scene_mode": "fixed", "motion_preset": "push_in"}, self.root / "runs" / "motion")
        self.assertEqual(command[command.index("--motion-preset") + 1], "push_in")
        with self.assertRaisesRegex(ValueError, "分层视差需要"):
            manager._command({"audio_id": "root:中文旁白.m4a", "output": "both",
                              "scene_mode": "fixed", "motion_preset": "parallax"}, self.root / "runs" / "motion")
        with self.assertRaisesRegex(ValueError, "自动运镜需要"):
            manager._command({"audio_id": "root:中文旁白.m4a", "output": "both",
                              "scene_mode": "fixed", "motion_preset": "auto"}, self.root / "runs" / "motion")

    def test_combined_motion_is_passed_and_no_scene_review_pause(self):
        manager = self.app.extensions["job_manager"]
        with patch("web_app.list_models", return_value=[{"id": "test.ckpt", "name": "Test"}]), \
             patch.object(manager.characters, "get", return_value={"views": {"front": {"status": "approved"}}}), \
             patch.object(manager.characters, "approved_references", return_value=[Path("front.png")]):
            command, _, _ = manager._command({"audio_id": "root:中文旁白.m4a", "output": "assets",
                "scene_mode": "fixed", "generate_images": True, "drawthings_model": "test.ckpt",
                "character_profile_id": "hero", "motion_effects": {"fade": "in", "vertical": "up"}},
                self.root / "runs" / "combined")
        self.assertNotIn("--review-scenes", command)
        self.assertEqual(json.loads(command[command.index("--motion-effects") + 1]),
                         {"fade": "in", "vertical": "up"})

    def test_manual_parallax_placement_keeps_base_motion_and_passes_scene_numbers(self):
        manager = self.app.extensions["job_manager"]
        data = {"audio_id": "root:中文旁白.m4a", "output": "assets", "scene_mode": "llm",
                "llm_model": "test", "motion_preset": "pan_left", "parallax_placement": "manual",
                "parallax_scenes": "5，1, 5", "generate_images": True, "drawthings_model": "test.ckpt",
                "parallax_background_effects": {"vertical": "down", "horizontal": "none"},
                "parallax_foreground_effects": {"fade": "none"}}
        with patch("web_app.list_models", return_value=[{"id": "test.ckpt"}]):
            command, _, _ = manager._command(data, self.root / "runs" / "parallax")
        self.assertEqual(command[command.index("--motion-preset") + 1], "pan_left")
        self.assertEqual(command[command.index("--parallax-placement") + 1], "manual")
        self.assertEqual(command[command.index("--parallax-scenes") + 1], "1,5")
        self.assertEqual(json.loads(command[command.index("--parallax-background-effects") + 1]), {"vertical": "down"})
        self.assertEqual(json.loads(command[command.index("--parallax-foreground-effects") + 1]), {})
        with self.assertRaisesRegex(ValueError, "大模型语义分段"):
            manager._command(dict(data, scene_mode="fixed"), self.root / "runs" / "parallax")

    def test_profile_model_can_be_changed_and_profile_can_be_deleted(self):
        models = [{"id": "flux.ckpt", "name": "FLUX"}, {"id": "z_image.ckpt", "name": "Z Image"}]
        with patch("web_app.list_models", return_value=models):
            created = self.client.post("/api/characters", json={
                "name": "高欢", "path": "characters/gaohuan", "model": "flux.ckpt", "front_prompt": "portrait",
            }).get_json()
            changed = self.client.patch("/api/characters/gaohuan", json={"model": "z_image.ckpt"})
            self.assertEqual(changed.status_code, 200)
            self.assertEqual(self.client.get("/api/characters/gaohuan").get_json()["model"], "z_image.ckpt")
            invalid = self.client.patch("/api/characters/gaohuan", json={"model": "missing.ckpt"})
            self.assertEqual(invalid.status_code, 400)
        deleted = self.client.delete("/api/characters/gaohuan")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/characters/gaohuan").status_code, 404)
        self.assertTrue(Path(deleted.get_json()["archived_path"]).is_dir())
        self.assertEqual(created["id"], "gaohuan")

    def test_scene_review_endpoint_validates_and_writes_decision(self):
        manager = self.app.extensions["job_manager"]
        job_id = "scene-review"
        folder = manager.runs / job_id
        folder.mkdir()
        manager.jobs[job_id] = {"id": job_id, "status": "running", "created_at": "2026-01-01T00:00:00+00:00"}
        (folder / "progress.json").write_text(json.dumps({"stage": "等待确认场景", "percent": 64}))
        (folder / "scenes_draft.json").write_text(json.dumps({"character_name": "高欢", "scenes": [
            {"start": 0, "end": 1, "prompt": "city", "character_present": True},
        ]}))
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/scenes").get_json()["scenes"][0]["prompt"], "city")
        invalid = self.client.post(f"/api/jobs/{job_id}/scenes", json={"scenes": [{"prompt": "", "character_present": True}]})
        self.assertEqual(invalid.status_code, 400)
        accepted = self.client.post(f"/api/jobs/{job_id}/scenes", json={"scenes": [
            {"prompt": "hero in city", "character_present": False},
        ]})
        self.assertEqual(accepted.status_code, 200)
        self.assertFalse(json.loads((folder / "scene_review.json").read_text())["scenes"][0]["character_present"])

    def test_job_command_passes_global_visual_style(self):
        manager = self.app.extensions["job_manager"]
        command, _, _ = manager._command({
            "audio_id": "root:中文旁白.m4a", "output": "assets", "scene_mode": "llm",
            "llm_model": "qwen", "visual_style": "时代：北魏；地点：邺城；气氛：肃杀",
        }, self.root / "runs" / "style-test")
        self.assertEqual(command[-2:], ["--visual-style", "时代：北魏；地点：邺城；气氛：肃杀"])

    def test_upload_stays_inside_uploads_and_rejects_non_audio(self):
        response = self.client.post("/api/upload", data={"file": (io.BytesIO(b"audio"), "../../声音.m4a")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201)
        audio_id = response.get_json()["id"]
        self.assertTrue(audio_id.startswith("upload:"))
        self.assertTrue((self.root / "uploads" / audio_id.split(":", 1)[1]).is_file())
        bad = self.client.post("/api/upload", data={"file": (io.BytesIO(b"x"), "bad.exe")}, content_type="multipart/form-data")
        self.assertEqual(bad.status_code, 400)

    def test_job_runs_and_exposes_outputs(self):
        transcript = self.root / "captions.json"
        transcript.write_text(json.dumps([{"start": 0, "end": 0.8, "text": "测试字幕"}]), encoding="utf-8")
        created = self.client.post("/api/jobs", json={
            "audio_id": "root:中文旁白.m4a", "transcript_json": "captions.json", "output": "assets",
            "scene_mode": "none", "whisper_model": "tiny", "max_seconds": 1,
        })
        self.assertEqual(created.status_code, 202, created.get_data(as_text=True))
        job_id = created.get_json()["id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").get_json()
            if job["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "completed", job)
        self.assertIn("transcript.txt", job["files"])
        text = self.client.get(f"/api/jobs/{job_id}/files/transcript.txt")
        self.assertEqual(text.get_data(as_text=True), "测试字幕")
        text.close()

    def test_rejects_paths_outside_project(self):
        response = self.client.post("/api/jobs", json={"audio_id": "root:../other.m4a", "output": "assets"})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/jobs", json={"audio_id": "root:中文旁白.m4a", "transcript_json": "../other.json", "output": "assets"})
        self.assertEqual(response.status_code, 400)

    def test_jianying_status_accepts_verified_v11_format(self):
        draft_root = self.root / "drafts"
        draft_root.mkdir()
        (draft_root / "root_meta_info.json").write_text('{"all_draft_store": []}')
        with patch("jianying_register.installed_version", return_value="11.7.0"):
            result = self.client.get("/api/jianying/status", query_string={"root": str(draft_root)})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.get_json()["supported"])

    def test_cancel_before_worker_starts_skips_process(self):
        manager = JobManager(self.root)
        job_id = "cancelled-test"
        (manager.runs / job_id).mkdir()
        manager.jobs[job_id] = {"id": job_id, "status": "cancelled", "created_at": "2026-01-01T00:00:00+00:00"}
        with patch("web_app.subprocess.Popen", side_effect=AssertionError("worker launched after cancellation")) as popen:
            manager._run(job_id, ["python", "audio_video.py"], {})
            popen.assert_not_called()

    def test_deleted_job_directory_is_not_returned_as_html_500(self):
        manager = self.app.extensions["job_manager"]
        job_id = "removed-run"
        folder = manager.runs / job_id
        folder.mkdir()
        manager.jobs[job_id] = {"id": job_id, "status": "completed", "created_at": "2026-01-01T00:00:00+00:00"}
        shutil.rmtree(folder)
        listing = self.client.get("/api/jobs")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.get_json()["jobs"], [])
        detail = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(detail.status_code, 404)
        self.assertIn("error", detail.get_json())

    def test_unexpected_api_error_is_json(self):
        with patch.object(self.app.extensions["job_manager"], "list", side_effect=RuntimeError("测试异常")):
            response = self.client.get("/api/jobs")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.mimetype, "application/json")
        self.assertIn("error", response.get_json())

    def test_delete_finished_job_removes_only_its_run_directory(self):
        manager = self.app.extensions["job_manager"]
        job_id = "finished-run"
        folder = manager.runs / job_id
        folder.mkdir()
        (folder / "job.json").write_text("{}")
        outside = self.root / "keep.txt"
        outside.write_text("keep")
        manager.jobs[job_id] = {"id": job_id, "status": "completed", "created_at": "2026-01-01T00:00:00+00:00"}
        response = self.client.delete(f"/api/jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(folder.exists())
        self.assertTrue(outside.is_file())
        self.assertNotIn(job_id, manager.jobs)

    def test_running_job_cannot_be_deleted(self):
        manager = self.app.extensions["job_manager"]
        job_id = "running-run"
        folder = manager.runs / job_id
        folder.mkdir()
        manager.jobs[job_id] = {"id": job_id, "status": "running", "created_at": "2026-01-01T00:00:00+00:00"}
        response = self.client.delete(f"/api/jobs/{job_id}")
        self.assertEqual(response.status_code, 400)
        self.assertTrue(folder.is_dir())

    def test_open_job_directory_uses_os_file_browser(self):
        manager = self.app.extensions["job_manager"]
        job_id = "open-run"
        folder = manager.runs / job_id
        folder.mkdir()
        manager.jobs[job_id] = {"id": job_id, "status": "completed", "created_at": "2026-01-01T00:00:00+00:00"}
        with patch("web_app.subprocess.Popen") as popen:
            response = self.client.post(f"/api/jobs/{job_id}/open")
        self.assertEqual(response.status_code, 200)
        popen.assert_called_once_with(["open", str(folder)])

    def test_job_status_exposes_progress_file(self):
        manager = self.app.extensions["job_manager"]
        job_id = "progress-run"
        folder = manager.runs / job_id
        folder.mkdir()
        (folder / "progress.json").write_text(json.dumps({"stage": "转写中", "percent": 42}))
        manager.jobs[job_id] = {"id": job_id, "status": "running", "created_at": "2026-01-01T00:00:00+00:00"}
        response = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(response.get_json()["progress"], {"stage": "转写中", "percent": 42})


if __name__ == "__main__":
    unittest.main()
