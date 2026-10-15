# ComfyUI-DepthClone

深度复刻（视频主体替换）预处理用的两个 ComfyUI 节点：把参考视频转成逐帧深度图，以及只保留人物与人物前方的「前景深度」。生成模型拿这些深度图做动作和构图参考。

算法与千谲本地脚本 `depth_gpu.py` / `depth_gpu_fg.py` 逐行一致，线上线下出来的深度图相同。只用 ComfyUI 自带的 `torch`，不引入额外依赖。

## 节点

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| 深度推理+归一化(深度复刻) `DepthCloneDepth` | `da_v2_model`（DAMODEL）、`images`（IMAGE） | 深度（IMAGE，近亮远暗） | Depth Anything V2 推理（短边 518），逐帧取 1% / 99% 分位、按 0.9 / 0.1 滑动平均归一化，输出长边不超过 960 |
| 前景深度合成(深度复刻) `DepthCloneForeground` | `depth`（IMAGE）、`person_mask`（IMAGE，黑=人、白=背景）、`percentile`（FLOAT，外接输入，默认 0.15） | 前景深度（IMAGE） | 人物掩码膨胀；取人物区域内深度的第 `percentile` 分位作阈值；保留「人物 ∪ 深度 ≥ 阈值」，高斯羽化后乘到深度上；人物占画面不到 1% 时输出完整深度。`percentile` 越小保留越多 |

## 依赖的其它节点

- `DownloadAndLoadDepthAnythingV2Model`：[kijai/ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2)，模型用 `depth_anything_v2_vitb_fp32.safetensors`
- 人体分割（前景深度用）：`segformer_b2_clothes`，输出黑=人、白=背景的分割图
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
人体分割(segformer_b2_clothes) ──┼─> 前景深度合成 ──> 合并为视频
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
