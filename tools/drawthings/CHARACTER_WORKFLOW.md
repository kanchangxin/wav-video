# Draw Things CLI 人物资产工作流

## 当前环境结论

- `draw-things-cli generate` 没有透明背景或 alpha 参数；FLUX.2 Klein 4B 的直接输出是 RGB PNG。
- 本机可用 Draw Things CLI 生成人物，再用 macOS Vision 人像分割输出透明 PNG。本目录的 `person_cutout.m` 已编译为 `person_cutout` 并用测试人物验证；测试输出为 RGBA，画布角落 alpha=0、人物中心 alpha=255。
- 固定 seed 只能复现同一请求，不能保证换姿势、镜头或场景后脸完全一样。长期保持身份，需要保存参考图、固定生成配置；要求更高时训练人物 LoRA。

## 透明人物

先用简单、对比清楚的纯色背景生成人物，避免头发、衣服与背景颜色太接近，确保全身和手脚完整。然后运行：

```bash
draw-things-cli generate --local --offline --no-download-missing \
  --model flux_2_klein_4b_i8x.ckpt \
  --prompt 'Full-body portrait of one person, standing, feet visible, plain light gray background, even studio lighting' \
  --steps 4 --cfg 1 --width 512 --height 768 --seed 271828 \
  --output person_rgb.png

/Users/jim/codex_space/wav生成视频/tools/drawthings/person_cutout \
  person_rgb.png person_transparent.png
```

若需重新编译抠图工具：

```bash
clang -fobjc-arc -framework Foundation -framework CoreImage -framework Vision \
  -framework ImageIO -framework CoreVideo -framework CoreGraphics \
  /Users/jim/codex_space/wav生成视频/tools/drawthings/person_cutout.m \
  -o /Users/jim/codex_space/wav生成视频/tools/drawthings/person_cutout
```

人物头发边缘、透明衣料、手指间隙等应逐张检查；Vision 人像分割不是精修蒙版。

## 跨对话保持人物身份

为每个人物在固定位置保存一个资产包，例如：

```text
characters/lin/
  identity.md              固定脸型、五官、发型、特殊标记；区分可变的衣服和场景
  master_front_rgb.png     正面基准图
  master_three_quarter.png 斜侧面基准图
  master_full_body.png     全身基准图
  generation.json          模型文件名、步数、CFG、尺寸、seed、参考图顺序、LoRA 配置
  lora.ckpt                训练后才有
```

在新对话中明确给出资产包的**绝对路径**，要求先读 `identity.md` 和 `generation.json`，再以同一套参考图生成。不要只说“沿用上个对话的人物”。保存原始 RGB 参考图用于 CLI `--image`；透明 PNG 主要用于排版和合成。

参考图模式示例：

```bash
draw-things-cli generate --local --offline --no-download-missing \
  --model flux_2_klein_4b_i8x.ckpt \
  --image characters/lin/master_front_rgb.png \
  --prompt 'The same person from the reference, preserve the exact face and distinguishing features, now walking in a park' \
  --width 512 --height 512 --steps 4 --cfg 1 --seed 271829 \
  --output scene_park.png
```

当输出画幅与第一张参考图比例不同时，CLI 将它作为 moodboard，而非待编辑画布；这种方式允许重构新场景，也会带来人脸漂移。用相同画幅和合适的 `--strength` 可让原图作为画布进行图生图。`--image` 可重复以加入更多参考图，但模型是否善用多图须实测。

如果要在很多姿势、表情和场景中尽量保持脸，收集同一人物的多角度高质量图并训练人物 LoRA；把 LoRA 文件、触发词和权重一同保存在资产包中。生成时继续搭配参考图并检查关键特征。即使有 LoRA，也不能保证每一张完全一致。

## 本次验证样张

- [CLI 原始人物](results/character_test_rgb.png)
- [透明 PNG](results/character_test_transparent.png)
- [参考图换场景](results/character_test_reference.png)：外观相似，但五官有轻微变化，说明单张参考图并非身份锁定。
