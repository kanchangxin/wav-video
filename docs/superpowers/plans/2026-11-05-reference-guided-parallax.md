# Reference Guided Parallax Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make empty backgrounds and optional compound foregrounds from the original scene image while retaining the existing pixel-preserving person cutout.

**Architecture:** Add an explicit Draw Things reference mode so edit inputs stay a canvas. Keep prompt/model selection and subject mode in a small parallax-layer helper, then call it from the batch pipeline and the existing scene-editor queue. All user-facing edits remain candidates until accepted.

**Tech Stack:** Python 3.11, Draw Things CLI, Apple Vision person cutout, Flask/vanilla JS, FFmpeg, unittest.

**Spec:** `docs/superpowers/specs/2026-11-05-reference-guided-parallax-design.md`

## Global Constraints

- Scene image is `--image` 1 for edit operations; identity references keep their current moodboard behavior.
- Background filling uses the source framing. Existing scale/crop motion supplies the pan margin.
- Single-person foreground uses original pixels; composite subject foreground tries AI and must have valid alpha.
- Batch failure returns to a single-image motion; editor failure retains the current layer.
- MP4 and Jianying use the same accepted layers and motion settings.

## Review Focus

- Identical source/output aspect ratio must remain an edit canvas rather than being padded into a moodboard.
- A horse or held prop must not silently disappear through person-only cutout fallback.
- Opaque AI output must not be accepted as a transparent foreground.
- Replacing the source image must invalidate derived background/foreground request fingerprints.
- Old jobs without new metadata must remain editable and renderable.

---

### Task 1: Explicit Draw Things reference mode

**Files:** Modify `drawthings_cli.py`; test `tests/test_drawthings_cli.py`.

**Interfaces:** `generate_image(..., references: list[Path] | None = None, reference_mode: str = "moodboard") -> None`, where `reference_mode="instruction"` adds 64 pixels of edge-filled padding so Qwen receives the first image as a moodboard input; `"edit"` retains the literal canvas for other uses.

- [x] Add failing tests that edit mode keeps the literal canvas and instruction mode adds edge-filled padding without white bars.
- [x] Run `uv run python -m unittest tests.test_drawthings_cli` and confirm the new tests fail.
- [x] Implement and validate the reference modes in `drawthings_cli.py`.
- [x] Run the test module and confirm pass.

### Task 2: Batch parallax layer generation

**Files:** Create `parallax_layers.py`; modify `audio_video.py`, `web_app.py` if settings need exposing; test `tests/test_audio_video.py` or the matching pipeline test module.

**Interfaces:** `parallax_layers.background_request(scene, image, model, size, seed, suffix) -> dict`, `foreground_mode(scene) -> str`, and `foreground_prompt(scene) -> str`. Batch pipeline records mode, source reference, fingerprint, prompt, and output. Use the downloaded Qwen Image 2.1 Turbo as background default when present; scene image model remains independent.

- [ ] Write failing tests for background prompt/source reference, model independence, composite subject detection, and single-image fallback on invalid/opaque AI foreground.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement helpers and integrate batch generation with `reference_mode="edit"`; preserve old non-parallax behavior and metadata.
- [ ] Run targeted tests and confirm pass.

### Task 3: Scene editor candidate operations and UI

**Files:** Modify `scene_editor.py`, `templates/scene_workspace.html`, `web_app.py`; test `tests/test_scene_editor.py`.

**Interfaces:** `SceneEditor.generate(job_id, index, layer, command, references=None, mode="standard") -> dict` with modes `source_edit` and `ai_edit`; `SceneEditor.cutout_from_scene(job_id, index) -> dict`; re-use the persistent queue, candidate acceptance, and rebuild semantics.

- [ ] Write failing tests for each new operation, command preview with `--image <scene image>`, alpha rejection for composite AI output, candidate acceptance, and old-job compatibility.
- [ ] Run targeted tests and confirm failure.
- [ ] Add queue operations, endpoints, and compact buttons in the layer editor; show model/prompt/command and failure details.
- [ ] Run targeted tests and confirm pass.

### Task 4: End-to-end verification

**Files:** Modify `README.md` and tests only if verification exposes gaps.

- [ ] Run `uv run python -m unittest discover -s tests`, `node --check` on the extracted scene-editor script, and `git diff --check`.
- [ ] On the user's sample job, queue one empty-background candidate and one foreground candidate with Qwen; inspect output, alpha, and UI without overwriting accepted layers.
- [ ] Document the tested model behavior and known limitations in `README.md`.
- [ ] Commit only feature files, preserving unrelated workspace changes.
