"""深度复刻预处理节点：普通深度（Depth Anything V2 推理 + 归一化）、人体分割、前景深度合成。

算法与千谲本地脚本 depth_gpu.py / depth_gpu_fg.py 逐行一致（原先放在 RunningHub 的 RB_Code 代码节点里，
该节点下线后改成固定输入输出的正式节点）。改参数前先和本地脚本对齐，否则线上线下深度图会不一致。
"""

import contextlib
import os

import torch
import torch.nn.functional as F

# 推理短边与输出长边上限，同本地 depth_gpu.py。
SHORT_SIDE = 518
MAX_OUTPUT_SIDE = 960
# Depth Anything V2 的 ImageNet 归一化参数。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH = 8
# 人体分割：segformer_b2_clothes，推理尺寸 512x512，同本地 depth_gpu_fg.py。
SEG_MODEL_DIR = "segformer_b2_clothes"
SEG_REPO_ID = "mattmdjaga/segformer_b2_clothes"
SEG_SIZE = 512


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def depth_normalize(da_model, images, precision="fp16"):
    """images: [B,H,W,3] 0~1；返回 [B,oh,ow,3] 深度（近亮远暗），1%/99% 分位 + 0.9/0.1 滑动平均归一化。

    precision="fp16"：模型推理走 fp16 混合精度（本地脚本同样用 fp16 推理），5070 Ti 实测比 fp32 快约 3.3 倍，
    与 fp32 的深度差平均 0.05/255；分位、归一化始终在 fp32 下计算。"""
    model = da_model["model"] if isinstance(da_model, dict) else da_model
    B, H, W, _ = images.shape
    sc = SHORT_SIDE / min(W, H)
    iw, ih = max(14, round(W * sc / 14) * 14), max(14, round(H * sc / 14) * 14)
    ow, oh = W // 2 * 2, H // 2 * 2
    if max(ow, oh) > MAX_OUTPUT_SIDE:
        k = MAX_OUTPUT_SIDE / max(ow, oh)
        ow, oh = int(ow * k) // 2 * 2, int(oh * k) // 2 * 2
    dev = _device()
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    model.to(dev)
    p0 = next(model.parameters())
    lo = hi = None
    res = []
    amp = precision == "fp16" and dev.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
    with torch.no_grad():
        for s in range(0, B, BATCH):
            x = images[s:s + BATCH].to(dev).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)
            with autocast:
                d = model(((x - mean) / std).to(p0.dtype))
            d = d.float()
            if d.dim() == 4:
                d = d[:, 0]
            d = F.interpolate(d[:, None], size=(oh, ow), mode="bicubic", align_corners=False)[:, 0]
            for f in d:
                flat = f.flatten()
                a = float(torch.quantile(flat, 0.01))
                b = float(torch.quantile(flat, 0.99))
                if lo is None:
                    lo, hi = a, b
                else:
                    lo, hi = lo * 0.9 + a * 0.1, hi * 0.9 + b * 0.1
                res.append(((f - lo) / max(1e-6, hi - lo)).clamp(0, 1).cpu())
    return torch.stack(res)[..., None].repeat(1, 1, 1, 3)


_seg_model = None


def _load_seg_model():
    """优先读 ComfyUI/models/segformer_b2_clothes（与 segformer_b2_clothes 节点共用同一份模型），没有再按仓库名取。"""
    global _seg_model
    if _seg_model is None:
        from transformers import AutoModelForSemanticSegmentation
        src = SEG_REPO_ID
        try:
            import folder_paths
            local = os.path.join(folder_paths.models_dir, SEG_MODEL_DIR)
            if os.path.isfile(os.path.join(local, "config.json")):
                src = local
        except ImportError:
            pass
        _seg_model = AutoModelForSemanticSegmentation.from_pretrained(src).eval()
    return _seg_model


def person_mask(images):
    """images: [B,H,W,3] 0~1；返回 [B,H,W,3] 人体分割图（黑=人、白=背景），类别非背景即算人。
    GPU 上 fp16 按批推理（同本地 depth_gpu_fg.py）；替代逐帧在 CPU 上跑的 segformer_b2_clothes 节点。"""
    B, H, W, _ = images.shape
    dev = _device()
    half = dev.type == "cuda"
    model = _load_seg_model().to(dev)
    model = model.half() if half else model.float()
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    res = []
    with torch.no_grad():
        for s in range(0, B, BATCH):
            x = images[s:s + BATCH].to(dev).permute(0, 3, 1, 2).float()
            xs = F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
            xs = (xs - mean) / std
            lg = model(pixel_values=xs.half() if half else xs).logits.float()
            pm = F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False).argmax(1) != 0
            res.append((~pm).float().cpu())
    return torch.cat(res)[..., None].repeat(1, 1, 1, 3)


def foreground_depth(depth, person_mask, percentile=0.15):
    """depth: 归一化深度 [B,H,W,3]；person_mask: 人体分割图（黑=人、白=背景）；
    人物掩码膨胀，取人物区域内深度第 percentile 分位作阈值，保留「人物 ∪ 深度>=阈值」，高斯羽化后乘到深度上；
    人占比不到 1% 时输出完整深度。"""
    d = depth[..., 0].float()
    seg = person_mask[..., 0].float()
    pct = min(max(float(percentile) if percentile is not None else 0.15, 0.0), 1.0)
    B, oh, ow = d.shape
    dev = _device()
    kd = max(5, ow // 60) | 1
    sigma = max(2.0, ow / 200)
    r = int(round(sigma * 4))
    xs = torch.arange(-r, r + 1, device=dev).float()
    g1 = torch.exp(-xs * xs / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    res = []
    for i in range(B):
        g = d[i].to(dev)
        si = min(i, seg.shape[0] - 1)
        pm = F.interpolate(seg[si:si + 1, None].to(dev), size=(oh, ow), mode="bilinear", align_corners=False)[0, 0] < 0.5
        if float(pm.float().mean()) > 0.01:
            ref = float(torch.quantile(g[pm], pct))
            dil = F.max_pool2d(pm.float()[None, None], kd, 1, kd // 2)[0, 0]
            keep = torch.maximum(dil, (g >= ref).float())[None, None]
            keep = F.conv2d(F.pad(keep, (r, r, 0, 0), mode="reflect"), g1.view(1, 1, 1, -1))
            keep = F.conv2d(F.pad(keep, (0, 0, r, r), mode="reflect"), g1.view(1, 1, -1, 1))[0, 0]
            g = g * keep
        res.append(g.cpu())
    return torch.stack(res)[..., None].repeat(1, 1, 1, 3)


class DepthCloneDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "da_v2_model": ("DAMODEL",),
                "images": ("IMAGE",),
                "precision": (["fp16", "fp32"], {"default": "fp16", "tooltip": "fp16 混合精度约 3 倍速度、深度差可忽略；fp32 与本地脚本逐值一致"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("深度",)
    FUNCTION = "run"
    CATEGORY = "深度复刻"
    DESCRIPTION = "Depth Anything V2 推理 + 1%/99% 分位滑动平均归一化（近亮远暗），与千谲本地 depth_gpu.py 同算法；默认 fp16 混合精度推理。"

    def run(self, da_v2_model, images, precision="fp16"):
        return (depth_normalize(da_v2_model, images, precision),)


class DepthClonePersonMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("人体分割",)
    FUNCTION = "run"
    CATEGORY = "深度复刻"
    DESCRIPTION = "segformer_b2_clothes 人体分割（黑=人、白=背景），GPU fp16 批量推理，与千谲本地 depth_gpu_fg.py 同算法。"

    def run(self, images):
        return (person_mask(images),)


class DepthCloneForeground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
                "person_mask": ("IMAGE",),
                "percentile": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("前景深度",)
    FUNCTION = "run"
    CATEGORY = "深度复刻"
    DESCRIPTION = "人物掩码膨胀 + 自适应深度阈值 + 高斯羽化，只保留人物与人物前方的深度，与千谲本地 depth_gpu_fg.py 同算法。"

    def run(self, depth, person_mask, percentile):
        return (foreground_depth(depth, person_mask, percentile),)


NODE_CLASS_MAPPINGS = {
    "DepthCloneDepth": DepthCloneDepth,
    "DepthClonePersonMask": DepthClonePersonMask,
    "DepthCloneForeground": DepthCloneForeground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthCloneDepth": "深度推理+归一化(深度复刻)",
    "DepthClonePersonMask": "人体分割(深度复刻)",
    "DepthCloneForeground": "前景深度合成(深度复刻)",
}
