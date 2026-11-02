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

可在场景生图、角色基准图及已完成项目的场景图片编辑页勾选 **Real-ESRGAN X2+ 放大**；命令行对应 `--upscale-x2`。主页面的宽高表示基础生图尺寸：勾选后 Draw Things 在同一次调用中放大 2 倍，MP4 和剪映画布也扩大 2 倍。例如输入 640×384，输出 1280×768；运镜需要余量时，场景原图可能更大。角色基准图页使用预设的最终尺寸，勾选时基础图约为预设尺寸的一半。必须已在本机安装 `realesrgan_x2plus_f16.ckpt`；没有勾选时沿用原有尺寸与缓存。X2+ 是后处理放大，细节质量仍取决于基础图。

`tools/drawthings/draw-things-cli` 是本机安装的 277MB 可执行文件，不纳入 GitHub；模型、输入音频和生成结果也属于本地资产。重新克隆项目后，请按 [Draw Things CLI 说明](https://github.com/drawthingsai/draw-things-community#quick-start-with-draw-things-cli-on-macos) 安装 CLI，或将可执行文件放回该路径，程序代码和配置即可恢复。

### 单主角角色基准图

勾选“调用 Draw Things 为场景生图”后，可以选择已确认的角色档案，设定固定 seed（默认 `271829`）和生图提示词固定后缀；三项设置彼此独立。点击“配置基准图”进入角色页面，创建档案并修改存储路径（限项目 `characters/` 下的单独目录）。页面提供可修改的正面半身照提示词案例；生成预览并确认正面照后，可分别以它为参考生成 3/4 侧脸照和全身照。三个视角都可以单独重做，也可以导入 PNG、JPEG、WebP 图片，只有确认过的图片会进入场景生图。更新正面照后，已有的侧脸和全身照会标为过期，需重新确认或生成。

角色页面的“当前档案的基准图模型”可单独修改并保存，后续生成基准图会使用新模型；主页面的场景生图模型独立选择，可以与基准图模型不同。删除档案会将其目录移至 `characters/.deleted/`，方便手动恢复；正在生图或被运行中任务使用的档案不能删除。任务会在场景分析后暂停，页面列出每段时间、提示词和“主角在画面中”勾选框。确认后继续生图；未勾选的场景不会传入角色图和身份提示词。每次运行会将已确认参考图复制到任务输出的 `character_reference/`，并在 `image_generation.json` 和运行日志里保存最终提示词、seed、模型、参考图和生成状态。图片复用会检查这些参数及参考图内容；改变任一项就重新生图。Draw Things CLI 在第一张参考图与输出画幅相同时会将其视作待编辑画布，程序会给参考图加边框使它保持参考图用途。

命令行可用 `--character-profile characters/角色目录 --image-seed 271829 --image-prompt-suffix "cinematic realism"`。主角出镜由模型自动判断；固定分段且没有配置大模型时默认不出镜。旧参数 `--review-scenes` 已停用。固定 seed 和参考图只能提高角色相似度，不能保证所有镜头完全一致。

也可自行放入图片，然后使用 `--images-dir 图片目录 --scene-mode fixed`；按 `scene_001.png`、`scene_002.png` 顺序命名，缺少的场景自动变黑。

### 场景动效

页面“场景动效”可选择静止、大模型逐场景自动推荐、缓慢推近、拉远、左右平移或分层视差；默认保持静止。命令行对应 `--motion-preset static|auto|push_in|pull_out|pan_left|pan_right|parallax`。自动推荐需要填写 LLM API 和模型名，模型为每段选择一种运镜；输入、输出记录在 `llm_trace.log`，最终选择和理由记录在 `scenes.json`。拉远和平移的生图宽高至少增加 25% 并向上取整到 64 的倍数，剪映草稿保留大图并写入可编辑关键帧。

还可在“淡入淡出、水平移动、垂直移动、推近拉远”四组中分别选一项，组间可组合，作用于基础运镜。CLI 使用 `--motion-effects '{"fade":"in_out","horizontal":"left","vertical":"up","zoom":"out"}'`；各项都可省略。动效同时写入 MP4 和剪映的位置、缩放、透明度关键帧。固定时长分段如未配置 LLM，主角默认不出镜；配置 LLM 和角色档案时由模型逐场景判断，无需等待人工确认。

分层视差需要开启 Draw Things 生图并选择角色档案。系统判断主角在画面中后，程序会从该场景图片抠出透明人物，同时额外生成无人空背景；在剪映草稿里保存为背景与人物两条可编辑视频轨道，分别添加原生位移、缩放关键帧。无主角的场景使用单图平移；某段分层素材失败或抠图近乎全透明时，仅该段回退到单图平移。分层图片保存在 `images/scene_001_background.png`、`images/scene_001_foreground.png` 等路径。MP4 使用同一组运动参数合成。当前抠图工具只处理单个人物主体；头发边缘、遮挡和空背景与人物的透视一致性需要人工检查。已在本机剪映 11.7.0 打开双图层测试草稿，并确认背景和前景的位置关键帧随时间变化。

### 场景图片编辑

任务完成后，点击任务详情中的“编辑场景图片与动效”。新页面逐场景展示原图及分层背景、透明人物，可分别修改四组动效。每张图旁展示可编辑的完整 Draw Things CLI 命令：可改提示词、模型、宽高（64 的倍数）、步数和 seed；输出路径与参考图路径由系统固定。命令按参数解析和校验，不经 Shell 执行。透明人物图可由新人物源图重新生成并抠图，也可直接上传透明 PNG。

重新生成或上传会创建候选图，在后台串行处理；页面切换和刷新不会影响任务。比较原图与候选图后选择“采用新图”或“舍弃候选”。采用新图会把原图保存到 `edits/revisions/`，更新素材并在后台重建 MP4 与本地剪映草稿；音频转写不会重新运行。若旧草稿已经注册到剪映，页面会在重建完成后提示“同步到剪映”，这会新增带编号的草稿，不覆盖剪映中可能正打开的旧草稿。上传支持 PNG、JPEG、WebP，单图上限 40 MB。

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
| `--motion-preset auto` | 由大模型逐场景选择运镜，需要 `--llm-model`；默认 `static` |
| `--transcript-json 文件` | 使用已校对字幕，跳过 ASR |
| `--register-jianying` | 将草稿登记到检测到的剪映草稿列表（支持 5.x 与 11.x） |

用 `--max-seconds 20` 可只处理前 20 秒，快速检查转写、画面和导出。运行 `uv run python audio_video.py --help` 查看全部参数。

## 本地操作页面

```bash
uv run python web_app.py
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。页面可选择项目目录里的音频、上传新音频、设置转写模型、场景拆分、Draw Things、剪映草稿与输出格式；默认勾选“只处理前 20 秒”方便试运行。主页面的表单设置会自动保存在当前浏览器，刷新或从角色页面返回后恢复；“重设配置”清除保存值，API Key 不写入浏览器存储。上传时显示实际上传百分比；任务运行时显示处理阶段和进度，模型缓存检查期间显示等待进度，缓存不完整时会在日志中说明原因。固定时长分段生成的每条图片提示词会逐条写入任务日志，也会保存在 `scenes.json`。任务详情可直接打开输出目录及已注册的剪映草稿目录，任务列表可删除已结束任务及其 `runs/` 输出文件（不会删除剪映中已注册的草稿）。右侧显示任务日志、文字预览、视频预览和下载链接。上传文件在 `uploads/`。页面只监听 `127.0.0.1`。

## 验证

```bash
uv run python -m unittest discover -s tests -v
```
