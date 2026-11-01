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
        audio = self.client.get("/api/audio").get_json()["files"]
        self.assertIn({"id": "root:中文旁白.m4a", "name": "中文旁白.m4a"}, audio)

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
