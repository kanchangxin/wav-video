# Scene Editor and Composable Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users combine four camera effect groups and edit every completed scene asset without blocking the page.

**Architecture:** Keep the existing Flask and Python pipeline. Store structured effect settings in scenes.json and expose validated Draw Things commands. Use one persistent worker for candidate generation and rebuild outputs only after an accepted change.

**Tech Stack:** Python 3.14, Flask, FFmpeg, Draw Things CLI, pyJianYingDraft, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-11-01-scene-editor-design.md`

## Global Constraints

- Preserve existing projects and the current single-preset CLI.
- Default fixed segmentation without an LLM to no protagonist.
- Never execute edited command text through a shell.
- Keep `characters/` and `runs/` as local user data.

## Review Focus

- An interrupted candidate job must not replace accepted artwork.
- A path in edited command or upload must not escape the selected job.
- Replacing a parallax source image must regenerate or invalidate its derived foreground.
- Rebuilding a registered draft must not silently overwrite the copy open in Jianying.
- New keyframes must produce visibly different MP4 frames and valid draft properties.

---

### Task 1: Compose motion groups

**Files:** `scene_motion.py`, `audio_video.py`, `web_app.py`, `templates/index.html`, `tests/test_pipeline.py`, `tests/test_web_app.py`.

- [x] Write tests for combined fade, horizontal, vertical, zoom, and per-layer motion.
- [x] Run the tests and observe feature-related failure.
- [x] Add validated structured motion settings and preserve existing preset behavior.
- [x] Update FFmpeg and Jianying writers to use the shared motion plan.
- [x] Expose four group controls on the home page and verify focused tests.

### Task 2: Remove protagonist confirmation pause

**Files:** `audio_video.py`, `web_app.py`, `templates/index.html`, `tests/test_pipeline.py`, `tests/test_web_app.py`.

- [x] Write a failing test for fixed segmentation with no LLM and profile selected.
- [x] Remove blocking review from new jobs and default fixed/no LLM scenes to no protagonist.
- [x] Keep model-derived presence for semantic scenes and verify focused tests.

### Task 3: Candidate assets and background queue

**Files:** `scene_editor.py`, `web_app.py`, `tests/test_scene_editor.py`.

- [x] Write failing API tests for scene inventory, command parsing, candidate generation, upload, status, discard and accept.
- [x] Build a single-worker queue with persisted task state and safe command validation.
- [x] Keep accepted images untouched until candidate acceptance; save the old revision.
- [x] Rebuild local outputs after accepted changes; provide explicit sync for registered drafts.
- [x] Verify API and pipeline tests.

### Task 4: Scene editor page

**Files:** `templates/scenes.html`, `templates/index.html`, `web_app.py`, `tests/test_web_app.py`.

- [x] Write failing route and HTML behavior tests.
- [x] Show scene/layer previews, editable command and motion controls, background job status, accept/discard/upload actions.
- [x] Verify page behavior with local API plus full test suite.

### Task 5: Final verification

- [x] Run all unit tests, JS syntax check and diff check.
- [x] Inspect a short rendered MP4 and draft keyframes for the four effect groups.
- [x] Review changed files and document remaining limits.
