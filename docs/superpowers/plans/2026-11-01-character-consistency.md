# 单主角角色档案 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本地页面创建、确认和复用主角参考图，并在场景生图时使用其身份、固定 seed 和提示词后缀。

**Architecture:** `character_profiles.py` 管理档案和生成状态；`drawthings_cli.py` 扩展 seed/多参考图；`audio_video.py` 负责场景确认、参数快照与缓存指纹；Flask API 和页面负责交互。

**Tech Stack:** Python 3.11、Flask、Draw Things CLI、原生 HTML/JS、unittest。

**Spec:** `docs/superpowers/specs/2026-11-01-character-consistency-design.md`

## Global Constraints

- 单主角；seed 默认 271829，参考图和后缀可为空。
- 档案存于可修改的项目内 `characters/` 子目录。
- 侧脸和全身只参考已确认正面照。
- 保留无角色档案的旧流程。

## Review Focus

- 路径逃逸、伪造图片和未确认参考图应被拒绝。
- 正面照未确认时，侧脸与全身生成应被拒绝。
- 场景没有主角时，不传角色图和身份提示词。
- 图片参数或参考图变化时，旧图必须失效。
- 生成失败或任务取消时，确认和后续操作不应卡住。

---

### Task 1: Draw Things 参数和角色档案

**Files:** `drawthings_cli.py`、`character_profiles.py`、`tests/test_drawthings_cli.py`、`tests/test_character_profiles.py`

- [ ] 写测试并确认失败：seed、多参考图顺序、模型能力和档案路径/审批状态。
- [ ] 实现 CLI 参数与角色档案管理。
- [ ] 运行相关测试并提交。

### Task 2: 场景生图与确认

**Files:** `audio_video.py`、`tests/test_pipeline.py`

- [ ] 写测试并确认失败：场景出镜、最终提示词、快照、指纹失效。
- [ ] 实现场景确认文件和生图参数传递。
- [ ] 运行相关测试并提交。

### Task 3: Flask API 与页面

**Files:** `web_app.py`、`templates/index.html`、`templates/characters.html`、`tests/test_web_app.py`

- [ ] 写测试并确认失败：角色创建/生成/确认/导入/预览、任务参数与场景确认。
- [ ] 实现 API、角色页面和主页面入口。
- [ ] 运行全套测试、手动检查页面和文档，并提交。
