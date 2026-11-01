# Draw Things 本机 CLI 测试报告

测试日期：2026-10-31。机器：Apple M4 Max、128 GB 内存、macOS 26.5.2。所有 CLI 生成均使用 `--local --offline --no-download-missing`，调用本机已有模型；没有云端生成或额外模型下载。下面时间是 CLI 输出的「总生成时间（含模型加载）」；终端进程总耗时通常再多约 3–4 秒。单次测试的时间不是稳定吞吐量基准。

## 安装核验

- 可执行文件：`/Users/jim/.local/bin/draw-things-cli`，已在 `PATH` 中。
- 来源：[Draw Things 官方 v26.1023.0 发布资产](https://github.com/drawthingsai/draw-things-community/releases/tag/v26.1023.0)。官方资产大小 290,110,032 字节，SHA-256 为 `5900f7a2f42e00b3b028c0fbcfcb799bd1da8905a2901b6b5bfbd32088c4a049`；本机副本两项均一致。
- `draw-things-cli --version` 显示 `dev`，但发布资产的哈希已核对。`--help`、模型列表和实际本机生图均可用。
- 本机模型目录：`/Users/jim/Library/Containers/com.liuliu.draw-things/Data/Documents/Models`。

`draw-things-cli models list --downloaded-only` 列出 11 个已下载模型：FLUX.2 Klein 4B、Z Image Turbo、Qwen Image 2.1 Turbo、Qwen Image 2.1、Qwen Image Edit 2511、ERNIE Image Turbo、HiDream I1 Dev、SDXL Base、Fooocus Inpaint SDXL、LTX-2.3 视频模型、Wan 2.2 视频模型。本次优先测低步数的三个图像模型。模型列表只代表 CLI 识别到的完整下载模型。

## 实测结果

| 任务 | 模型和设置 | 尺寸 | CLI 时间 | 观察 |
|---|---|---:|---:|---|
| 文生图 | FLUX.2 Klein 4B i8x，4 步，CFG 1 | 512² | **9.01 秒** | 红壶、木桌、蓝背景均符合提示词；本次最快且构图稳定。 |
| 文生图 | Z Image Turbo i8x，8 步，CFG 1 | 512² | 9.92 秒 | 构图正确；壶偏棕红，颜色遵循略弱。 |
| 文生图 | Qwen Image 2.1 Turbo i8x，8 步，CFG 1 | 512² | 13.78 秒 | 红壶、木桌、蓝背景符合；速度较慢。 |
| 改图：红壶改绿壶 | FLUX.2 Klein 4B，4 步，`--strength 1` | 512² | **9.96 秒** | 成功变绿，壶形与场景基本保留。 |
| 同一改图 | Qwen Image 2.1 Turbo，8 步，`--strength 1` | 512² | 25.80 秒 | 成功变绿并保留构图；本例没有速度优势。 |
| 参考图生图：红壶移到厨房 | FLUX.2 Klein 4B，4 步，输入 512²，输出 768×512 | 768×512 | **12.81 秒** | CLI 明确将不同画幅的输入当作 *moodboard*；红壶特征保留，厨房和大理石台面自然。 |
| 扩图尝试：白色扩展画布 | FLUX.2 Klein 4B，4 步，`--strength 0.7` | 768² | 10.85 秒 | **失败**：白边基本未填充；实际只运行了 2 个采样步。 |
| 扩图尝试：白色扩展画布 | FLUX.2 Klein 4B，4 步，`--strength 1` | 768² | **10.19 秒** | 蓝背景和木桌补全，原壶外观接近；整幅图经过重绘。将原图以 RGB 无损叠回中央后，原 512² 区域的像素与输入完全一致。 |

原图和核心输出：

| 原图 | 改图 | 参考图 | 扩图（原图中心已叠回） |
|---|---|---|---|
| ![原图](results/cli_flux_txt2img.png) | ![改图](results/cli_flux_edit.png) | ![参考图](results/cli_flux_reference.png) | ![扩图](results/cli_flux_outpaint_preserved.png) |

另用 Draw Things 图形界面的扩图/修复操作试过 FLUX.2 Klein 4B：产出 [GUI 扩图样张](results/flux_klein_outpaint.png)，原图中心保住了，但四周有明显接缝。这一次的 CLI 重绘加原图叠回效果更好；它仍可能在复杂图片边界出现接缝，需要逐图查看。

## 速度优先的方案

1. **文生图**：先用 FLUX.2 Klein 4B i8x，4 步、CFG 1。只有当具体提示词效果不佳时再试 Z Image Turbo 或 Qwen Image 2.1 Turbo。
2. **改图**：先用 FLUX.2 Klein 4B、`--image`、`--strength 1`、4 步。低强度 0.6 的 HTTP API 对照试验未把红壶改绿；强度应按改动幅度调整。复杂指令或保真度要求高时可试 Qwen Image 2.1 Turbo，但本例耗时约为 Flux 的 2.6 倍。
3. **参考图生图**：用 FLUX.2 Klein 4B 的 `--image`，使输出画幅与参考图不同；CLI 会把它作为 moodboard 重构新场景。若要保持原画布而只是改动内容，使用相同画幅并设置 `--strength`。
4. **扩图**：当前 CLI 帮助未提供 `--mask` 参数。实测可先将原图居中扩成白色画布，以 `--strength 1` 生成，再将原图无损叠回中央。这个方案适合背景简单、速度优先的图片；它不是受掩码约束的原生扩图，复杂边界要人工检查。

以上“优先”只针对这台机器和本报告里的样例。11 个已下载模型没有全部逐一比较，不能将单张测试推广成所有题材的绝对排名。

## 可复现命令

在 `/Users/jim/codex_space/wav生成视频` 执行：

```bash
draw-things-cli models list --downloaded-only

draw-things-cli generate --local --offline --no-download-missing \
  --model flux_2_klein_4b_i8x.ckpt --prompt 'A small red ceramic teapot on a wooden table, soft studio light, plain blue background, product photo' \
  --steps 4 --cfg 1 --width 512 --height 512 --seed 42 \
  --output tools/drawthings/results/cli_flux_txt2img.png

draw-things-cli generate --local --offline --no-download-missing \
  --model flux_2_klein_4b_i8x.ckpt --prompt 'Change the red teapot to vivid emerald green. Preserve the teapot shape, wooden table, camera angle and blue background.' \
  --image tools/drawthings/results/cli_flux_txt2img.png --strength 1 \
  --steps 4 --cfg 1 --seed 42 --output tools/drawthings/results/cli_flux_edit.png

draw-things-cli generate --local --offline --no-download-missing \
  --model flux_2_klein_4b_i8x.ckpt --prompt 'Use the same red ceramic teapot from the reference image, now placed in a sunlit kitchen on a white marble counter, preserve its recognizable shape and red color, photorealistic' \
  --image tools/drawthings/results/cli_flux_txt2img.png \
  --width 768 --height 512 --steps 4 --cfg 1 --seed 123 \
  --output tools/drawthings/results/cli_flux_reference.png
```

扩图的测试命令：

```bash
ffmpeg -hide_banner -loglevel error -y \
  -i tools/drawthings/results/cli_flux_txt2img.png \
  -vf 'pad=768:768:128:128:white' \
  tools/drawthings/results/cli_outpaint_canvas.png

draw-things-cli generate --local --offline --no-download-missing \
  --model flux_2_klein_4b_i8x.ckpt \
  --prompt 'A red ceramic teapot centered on a wide wooden tabletop against a continuous blue studio background, no white border, photorealistic.' \
  --image tools/drawthings/results/cli_outpaint_canvas.png \
  --strength 1 --steps 4 --cfg 1 --seed 42 \
  --output tools/drawthings/results/cli_flux_outpaint_strength1.png

ffmpeg -hide_banner -loglevel error -y \
  -i tools/drawthings/results/cli_flux_outpaint_strength1.png \
  -i tools/drawthings/results/cli_flux_txt2img.png \
  -filter_complex 'overlay=128:128:format=rgb,format=rgb24' -pix_fmt rgb24 \
  tools/drawthings/results/cli_flux_outpaint_preserved.png
```

扩图最后一步的中央 512×512 像素用 `ffmpeg` PSNR 对照验证为 `inf`（逐像素相同）。

## CLI 能力边界

CLI 本地模式直接使用 Draw Things 的模型与推理引擎，不等同于操作正在打开的图形界面。官方 CLI 的 `--image` 可用于图生图和多张参考图；在测试版本的帮助中没有掩码输入。若任务必须用精确掩码保护原图并修复复杂接缝，可改用 Draw Things 图形界面的画布/蒙版功能。CLI 参考图行为见 [官方 CLI 源码](https://github.com/drawthingsai/draw-things-community/blob/main/Apps/DrawThingsCLI/DrawThingsCLI.swift)。
