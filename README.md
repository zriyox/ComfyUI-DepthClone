# ComfyUI-DepthClone

深度复刻（视频主体替换）预处理用的三个 ComfyUI 节点：把参考视频转成逐帧深度图、做人体分割，以及只保留人物与人物前方的「前景深度」。生成模型拿这些深度图做动作和构图参考。

算法与千谲本地脚本 `depth_gpu.py` / `depth_gpu_fg.py` 一致（本地脚本同样是 GPU fp16 批量推理）。依赖 ComfyUI 环境自带的 `torch`，人体分割另用 `transformers`。

## 速度

RTX 5070 Ti、720×1280、160 帧（5 秒）实测：

| 步骤 | 原做法 | 本包 | 说明 |
|---|---|---|---|
| 深度推理+归一化 | fp32：8.5 秒 | fp16：2.6 秒 | 与 fp32 的深度差平均 0.05（0～255） |
| 人体分割 | `segformer_b2_clothes` 节点逐帧跑 CPU：434 毫秒/帧 | GPU fp16 批量：7 毫秒/帧 | 人物区域与旧节点重合度 0.99 |
| 前景深度合成 | — | 0.4 秒 | |
| 前景深度整条工作流 | 75.5 秒 | 6.8 秒（含冷启动加载模型） | 与本地脚本输出平均差 1.24，提速前 1.25 |

深度推理可把 `precision` 设为 `fp32`（慢约 3 倍）。bf16 误差明显更大（平均 0.43、最大 39），没有提供。

## 节点

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| 深度推理+归一化(深度复刻) `DepthCloneDepth` | `da_v2_model`（DAMODEL）、`images`（IMAGE）、`precision`（fp16 / fp32，默认 fp16） | 深度（IMAGE，近亮远暗） | Depth Anything V2 推理（短边 518），逐帧取 1% / 99% 分位、按 0.9 / 0.1 滑动平均归一化，输出长边不超过 960 |
| 人体分割(深度复刻) `DepthClonePersonMask` | `images`（IMAGE） | 人体分割（IMAGE，黑=人、白=背景） | segformer_b2_clothes，推理尺寸 512×512，非背景类别都算人；可直接替换 `segformer_b2_clothes` 节点（13 个类别全开时输出相同含义） |
| 前景深度合成(深度复刻) `DepthCloneForeground` | `depth`（IMAGE）、`person_mask`（IMAGE，黑=人、白=背景）、`percentile`（FLOAT，外接输入，默认 0.15） | 前景深度（IMAGE） | 人物掩码膨胀；取人物区域内深度的第 `percentile` 分位作阈值；保留「人物 ∪ 深度 ≥ 阈值」，高斯羽化后乘到深度上；人物占画面不到 1% 时输出完整深度。`percentile` 越小保留越多 |

## 依赖的其它节点

- `DownloadAndLoadDepthAnythingV2Model`：[kijai/ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2)，模型用 `depth_anything_v2_vitb_fp32.safetensors`
- 人体分割模型：放在 `ComfyUI/models/segformer_b2_clothes`（与 [segformer_b2_clothes 节点](https://github.com/StartHua/Comfyui_segformer_b2_clothes) 共用同一份，来源 [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)）；目录不存在时按仓库名从 Hugging Face 下载
- 视频读写：[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## 典型连线

普通深度：

```
加载视频(VHS_LoadVideo) ──IMAGE──┐
深度模型(DownloadAndLoadDepthAnythingV2Model) ──DAMODEL──┴─> 深度推理+归一化 ──> 合并为视频(VHS_VideoCombine)
```

前景深度：

```
深度推理+归一化 ──深度──────────┐
人体分割(深度复刻) ─────────────┼─> 前景深度合成 ──> 合并为视频
阈值分位(PrimitiveFloat) ────────┘
```

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zriyox/ComfyUI-DepthClone.git
```

重启 ComfyUI 即可。

## License

MIT
