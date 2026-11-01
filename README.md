# 音频转字幕视频 / 剪映草稿

输入 m4a、mp3 或 wav 等音频，输出带时间戳的文字、SRT 字幕、场景计划，并可输出 MP4 或剪映草稿。没有图片的场景使用黑色背景。

## 安装

需要 `uv`、Python 3.11+、ICU 的 `uconv`，以及 `ffmpeg`、`ffprobe`（ffmpeg 需带 `libx264` 与 `subtitles/libass`）。`uconv` 用于将转写、字幕和场景文字统一转换为简体中文。Python 依赖统一写在 [pyproject.toml](pyproject.toml)，版本锁定在 [uv.lock](uv.lock)。

```bash
uv sync --locked
```

以后增加依赖用 `uv add 包名`，运行命令用 `uv run`；不再单独维护 `requirements.txt`。

Whisper 模型保存在项目的 `.hf-cache`。页面和命令行会先检查缓存是否完整，再开始转写；缺少 `model.bin` 时会立即给出错误，不会因为 Hugging Face 网络超时而一直卡在“下载模型”。首次使用某个模型，可先在网络正常时执行：

```bash
HF_HOME=.hf-cache uv run python -c "from faster_whisper.utils import download_model; print(download_model('medium', cache_dir='.hf-cache/hub'))"
```

也可以给 `--whisper-model` 一个本地模型目录。Mac 上默认使用 CPU/int8，`small` 模型适合先检查质量；需要更准确的中文识别可改成 `medium` 或 `large-v3`。

## 基本使用

```bash
uv run python audio_video.py "信使高欢的逆袭.m4a" --output mp4
```

默认产生 `<音频名>_output/`，其中包含 `transcript.txt`、`transcript.json`、`subtitles.srt`、`scenes.json`、`video.mp4`。默认画面为黑色，字幕和原音频会保留。

只转写、不合成视频：

```bash
uv run python audio_video.py "信使高欢的逆袭.m4a" --output assets
```

`transcript.json` 可手工校对，再用它跳过转写：

```bash
uv run python audio_video.py "信使高欢的逆袭.m4a" \
  --transcript-json "信使高欢的逆袭_output/transcript.json" \
  --output-dir "信使高欢的逆袭_修订版" --output mp4
```

## 场景和图片

固定时长分段，不调用大模型：

```bash
uv run python audio_video.py "信使高欢的逆袭.m4a" \
  --scene-mode fixed --scene-seconds 25 --generate-images --output mp4
```

使用 OpenAI 兼容的聊天接口进行语义分段和图片提示词生成：

```bash
export LLM_BASE_URL="http://127.0.0.1:1234/v1"
export LLM_MODEL="你的模型名称"
export LLM_API_KEY="可选密钥"
uv run python audio_video.py "信使高欢的逆袭.m4a" \
  --scene-mode llm --visual-style "时代：北魏；地点：邺城；气氛：肃杀；画风：历史写实" \
  --generate-images --output both
```

`--visual-style` 是可选的全局画面设定，会加入每一批语义分段的系统提示词。网页中对应“全局画面设定”输入框，留空时由模型自行判断。每次请求的系统提示词、字幕输入和模型原始输出会保存到输出目录的 `llm_trace.log`，网页任务日志也会同步显示。

长音频会按批调用 LLM：每批最多 80 条字幕，且字幕正文合计最多 2000 字；每批索引从 0 重新开始，场景时间再拼接回完整音频。超过 2000 字的单条字幕需先拆分。模型若只漏掉末尾 1–2 条字幕，程序会把其时间归入最后一段并记录修正；其他索引错误会重试一次，两次都不合格则仅该批改用固定时长分段并记录原因。批次之间没有传递前一批的语义摘要；不同模型的上下文窗口也不同，因此 2000 字限制是请求大小保护，并非通用的 token 上限保证。

生图直接调用本机 `draw-things-cli`，优先使用 `tools/drawthings/draw-things-cli`，也支持安装到 `PATH`。页面自动列出 CLI 检测到的已下载模型；步数留空时使用模型推荐值。命令行可用 `--drawthings-model` 和可选的 `--image-steps`。生成时启用本地离线模式，不会下载额外模型。图片保存在输出目录 `images/scene_001.png` 等，重跑时会复用；生图失败会提示并使用黑色画面。[CLI 官方说明](https://github.com/drawthingsai/draw-things-community#quick-start-with-draw-things-cli-on-macos)。

`tools/drawthings/draw-things-cli` 是本机安装的 277MB 可执行文件，不纳入 GitHub；模型、输入音频和生成结果也属于本地资产。重新克隆项目后，请按 [Draw Things CLI 说明](https://github.com/drawthingsai/draw-things-community#quick-start-with-draw-things-cli-on-macos) 安装 CLI，或将可执行文件放回该路径，程序代码和配置即可恢复。

也可自行放入图片，然后使用 `--images-dir 图片目录 --scene-mode fixed`；按 `scene_001.png`、`scene_002.png` 顺序命名，缺少的场景自动变黑。

## 剪映草稿

```bash
uv run python audio_video.py "信使高欢的逆袭.m4a" \
  --scene-mode fixed --output jianying \
  --register-jianying --draft-name "高欢解说"
```

勾选页面里的“生成后注册到剪映首页”，或使用 `--register-jianying`，程序会读取剪映设置中的草稿目录，复制草稿及媒体、修正素材路径、为剪映 v11 生成 `draft_info.json` 和封面、备份并更新 `root_meta_info.json`。注册后重新打开剪映查看。也可以在页面任务结果中对已生成的本地草稿点击“注册此草稿到剪映”。`--jianying-drafts-dir` 可覆盖自动检测的目录，仅在注册时使用。已有同名草稿不会覆盖。

**格式适配：**参考本机已成功注册并由剪映打开的 `video_pipeline/draft_stage.py`，剪映 v11 在首次打开时可读取明文 `draft_info.json`，随后会自行加密；`root_meta_info.json` 的 `draft_ids` 是计数器，注册时保持原值。程序支持检测到的 5.x 与 11.x，其他版本会提示尚未验证。剪映库的[项目说明](https://github.com/GuanYixuan/pyJianYingDraft/blob/main/README.md)注明 Mac/Linux 可生成草稿，但平台自动导出能力有限。

## 常用参数

| 参数 | 用途 |
| --- | --- |
| `--output mp4\|jianying\|both\|assets` | 选择成片、草稿、两者或仅转写资源 |
| `--width 1280 --height 720 --fps 25` | 视频尺寸和帧率，宽高必须为偶数 |
| `--whisper-model small --device cpu --language zh` | 本地转写模型和语言 |
| `--scene-mode none\|fixed\|llm` | 不分段、定时分段、LLM 语义分段 |
| `--transcript-json 文件` | 使用已校对字幕，跳过 ASR |
| `--register-jianying` | 将草稿登记到检测到的剪映草稿列表（支持 5.x 与 11.x） |

用 `--max-seconds 20` 可只处理前 20 秒，快速检查转写、画面和导出。运行 `uv run python audio_video.py --help` 查看全部参数。

## 本地操作页面

```bash
uv run python web_app.py
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。页面可选择项目目录里的音频、上传新音频、设置转写模型、场景拆分、Draw Things、剪映草稿与输出格式；默认勾选“只处理前 20 秒”方便试运行。上传时显示实际上传百分比；任务运行时显示处理阶段和进度，模型缓存检查期间显示等待进度，缓存不完整时会在日志中说明原因。固定时长分段生成的每条图片提示词会逐条写入任务日志，也会保存在 `scenes.json`。任务详情可直接打开输出目录及已注册的剪映草稿目录，任务列表可删除已结束任务及其 `runs/` 输出文件（不会删除剪映中已注册的草稿）。右侧显示任务日志、文字预览、视频预览和下载链接。上传文件在 `uploads/`。页面只监听 `127.0.0.1`。

## 验证

```bash
uv run python -m unittest discover -s tests -v
```
