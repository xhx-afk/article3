"""
DEIM × DINOv3 训练期特征蒸馏（M2：最朴素版，插值对齐）
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
对齐损失写法参考 lightly-train (https://github.com/lightly-ai/lightly-train)。
教师仅在训练循环出现，推理路径不加载、不前向、不计入 Params/GFLOPs。
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register

__all__ = ['DINOv3Teacher', 'FeatureDistiller', 'EncoderFeatureHook']

# DEIM 的 ConvertPILImage 只做 /255，没有 mean/std 归一化（见现状 F8）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DTYPES = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}


# --------------------------------------------------------------------------- #
# 教师
# --------------------------------------------------------------------------- #
@register()
class DINOv3Teacher(nn.Module):
    """冻结的 DINOv3 教师。输入 [0,1] 的 RGB batch，输出 [B, D, H/p, W/p] 的 patch 特征图。

    两种后端：
      source='hf'  : transformers.AutoModel，model_id 可以是 HF repo id 或本地目录
      source='hub' : torch.hub.load(repo_dir, hub_entry, source='local', weights=...)
    """

    def __init__(self,
                 source: str = 'hf',
                 model_id: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                 repo_dir: Optional[str] = None,
                 hub_entry: str = 'dinov3_vitb16',
                 weights: Optional[str] = None,
                 dtype: str = 'float16',
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None):
        super().__init__()
        assert source in ('hf', 'hub'), source
        self.source = source
        self.autocast_dtype = _DTYPES[dtype]

        if source == 'hf':
            from transformers import AutoModel          # 惰性 import，满足 C4
            model = AutoModel.from_pretrained(model_id)
            self.patch_size = int(model.config.patch_size)
            self.embed_dim = int(model.config.hidden_size)
            # 归一化参数优先取官方 preprocessor，读不到再落回 ImageNet 常数。
            # 硬编码常数与 F8 属于同一类坑：不报错，只让教师特征悄悄退化。
            # 但配置里显式写了 mean/std 时不许覆盖 —— 否则那两个配置项静默失效，
            # 又是同一类坑的另一个版本。
            if mean is None or std is None:
                try:
                    from transformers import AutoImageProcessor
                    proc = AutoImageProcessor.from_pretrained(model_id)
                    mean = mean or tuple(proc.image_mean)
                    std = std or tuple(proc.image_std)
                    print(f'     ### teacher norm from preprocessor: mean={mean} std={std} ###')
                except Exception as e:                   # noqa: BLE001
                    print(f'     ### preprocessor 读取失败({e})，回退 ImageNet 常数 ###')
        else:
            assert repo_dir is not None and weights is not None, \
                "source='hub' 时必须给 repo_dir（dinov3 仓库路径）和 weights（.pth 路径）"
            model = torch.hub.load(repo_dir, hub_entry, source='local', weights=weights)
            self.patch_size = int(getattr(model, 'patch_size', 16))
            self.embed_dim = int(getattr(model, 'embed_dim', model.norm.normalized_shape[0]))
        # 注意：不要保存 num_prefix。register/storage token 的字段名在 DINOv3 各版本里
        # 叫过 num_register_tokens / n_storage_tokens，猜错会得到 num_prefix=1，
        # 然后 unflatten 抛一个语义不明的 RuntimeError。forward 里按 token 数反推。

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        # 关键：绕开 nn.Module 的子模块注册。
        # 目的：教师不进 parameters() / state_dict() / DDP 广播（见现状 F5、约束 C2/C6）
        object.__setattr__(self, 'model', model)

        mean = mean if mean is not None else IMAGENET_MEAN
        std = std if std is not None else IMAGENET_STD
        self.register_buffer('_mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    # --- 让教师跟着 distiller 走 device / eval，但不进 state_dict ---
    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self.model._apply(fn, *args, **kwargs)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()          # 教师永远 eval
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B,3,H,W]，取值 [0,1]（DEIM 数据管线的原始输出）。"""
        b, _, h, w = images.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, \
            f'教师 patch={p}，输入 {h}x{w} 不能整除。多尺度是 32 的倍数，正常不会触发'

        x = (images - self._mean) / self._std
        gh, gw = h // p, w // p

        with torch.autocast(device_type=images.device.type, dtype=self.autocast_dtype):
            if self.source == 'hf':
                tokens = self.model(pixel_values=x).last_hidden_state   # [B, prefix+N, D]
                # prefix（cls + register/storage token）按 token 数反推，不猜字段名
                n_prefix = tokens.shape[1] - gh * gw
                assert n_prefix >= 0, (
                    f'教师输出 {tokens.shape[1]} 个 token < 期望 patch 数 {gh * gw}，'
                    'patch_size 解析可能有误')
                feat = tokens[:, n_prefix:, :].unflatten(1, (gh, gw)).permute(0, 3, 1, 2)
            else:
                feat = self.model.get_intermediate_layers(
                    x, n=1, reshape=True, norm=True)[-1]                # [B, D, gh, gw]

        assert feat.shape[-2:] == (gh, gw), f'教师输出网格 {tuple(feat.shape[-2:])} != 期望 {(gh, gw)}'
        return feat.float()


# --------------------------------------------------------------------------- #
# 对齐层：M2 = 1×1 卷积 + 双线性插值。M3 在这里换成频域对齐（创新点 2）
# --------------------------------------------------------------------------- #
class InterpolateAligner(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False)

    def forward(self, s: torch.Tensor, t: torch.Tensor, exact: bool = False):
        s = self.proj(s)
        if s.shape[-2:] != t.shape[-2:]:
            assert not exact, (
                f'该层被标记为精确对齐层，但学生 {tuple(s.shape[-2:])} != 教师 {tuple(t.shape[-2:])}。'
                '检查 teacher patch_size 与该层 stride 是否一致')
            # 下采样方向必须开 antialias。PyTorch 的 bilinear 2× 下采样默认不抗混叠，
            # 会把教师高频混叠成噪声，人为压低"朴素插值"基线 ——
            # 那样 M3 频域对齐的增益里就混着"修好了一个 bug"，创新点 2 变成和稻草人比较。
            downsampling = t.shape[-2] > s.shape[-2]
            t = F.interpolate(t, size=s.shape[-2:], mode='bilinear',
                              align_corners=False, antialias=downsampling)
        return s, t


def _feature_loss(s: torch.Tensor, t: torch.Tensor, kind: str) -> torch.Tensor:
    """s, t: [B, D, H, W]，fp32。

    注意归约方式：不要用 F.mse_loss —— 它对通道维也取均值，归一化特征下量级 ~2/D，
    在 D=768 时典型值 0.0013，会让 loss_weight 需要给到 1000 量级。
    """
    if kind == 'cosine':
        s = F.normalize(s, dim=1)
        t = F.normalize(t, dim=1)
        return (1.0 - (s * t).sum(dim=1)).mean()            # ∈ [0, 2]
    if kind == 'mse_norm':
        s = F.normalize(s, dim=1)
        t = F.normalize(t, dim=1)
        return (s - t).pow(2).sum(dim=1).mean()             # = 2 * cosine 版本
    if kind == 'smooth_l1':
        return F.smooth_l1_loss(s, t, beta=1.0)             # 未归一化，权重需另调
    raise ValueError(f'unknown loss_type: {kind}')


# --------------------------------------------------------------------------- #
# 蒸馏器
# --------------------------------------------------------------------------- #
@register()
class FeatureDistiller(nn.Module):
    __inject__ = ['teacher']

    def __init__(self,
                 teacher: nn.Module,
                 layers: Sequence[int] = (0, 1, 2),
                 student_channels: Sequence[int] = (256, 256, 256),
                 student_strides: Sequence[int] = (8, 16, 32),
                 align: str = 'interpolate',
                 loss_type: str = 'cosine',
                 loss_weight: float = 1.0,
                 start_epoch: int = 0,
                 stop_epoch: Optional[int] = None,
                 aligner_lr: Optional[float] = None):
        super().__init__()
        assert align == 'interpolate', 'M2 只实现朴素插值；M3 在这里接频域对齐'
        object.__setattr__(self, 'teacher', teacher)   # 同样不注册为子模块

        self.layers = list(layers)
        self.student_channels = list(student_channels)
        self.student_strides = list(student_strides)
        self.loss_type = loss_type
        self.loss_weight = float(loss_weight)
        # start_epoch / stop_epoch 在 DDP 下要小心：蒸馏未激活时 forward 返回 {}，
        # aligner 拿不到梯度，DDP 会报
        # `Expected to have finished reduction in the prior iteration`。
        # M2 默认 start_epoch: 0 + stop_epoch: ~（全程激活）所以不会触发。
        # 将来真要用这两个开关，二选一：给 distiller 的 warp_model 传
        # find_unused_parameters=True，或在未激活的 epoch 里直接把 distiller
        # 置为 None 传给 train_one_epoch。
        self.start_epoch = int(start_epoch)
        self.stop_epoch = stop_epoch
        self.aligner_lr = aligner_lr

        # 只为选中的层建 aligner → DDP 不会出现未用参数（find_unused_parameters 可保持 False）
        self.aligners = nn.ModuleDict({
            str(i): InterpolateAligner(student_channels[i], teacher.embed_dim)
            for i in self.layers
        })
        # stride == teacher.patch_size 的层要求逐像素精确对齐
        self.exact = {i: (student_strides[i] == teacher.patch_size) for i in self.layers}

    def to(self, *args, **kwargs):
        self.teacher.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def active(self, epoch: int) -> bool:
        if epoch < self.start_epoch:
            return False
        if self.stop_epoch is not None and epoch >= self.stop_epoch:
            return False
        return True

    def forward(self, images: torch.Tensor, student_feats: List[torch.Tensor],
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        if not self.active(epoch):
            return {}

        t_feat = self.teacher(images)                       # [B, D, H/p, W/p]，fp32，无梯度

        total = images.new_zeros(())
        for i in self.layers:
            s, t = self.aligners[str(i)](student_feats[i].float(), t_feat, exact=self.exact[i])
            total = total + _feature_loss(s, t, self.loss_type)
        total = total / max(len(self.layers), 1)

        return {'loss_distill': self.loss_weight * total}


# --------------------------------------------------------------------------- #
# 抓编码器输出（不改 HybridEncoder，满足 C1）
# --------------------------------------------------------------------------- #
class EncoderFeatureHook:
    def __init__(self, encoder: nn.Module):
        self.feats: Optional[List[torch.Tensor]] = None
        self.enabled = False
        self._handle = encoder.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.enabled:
            self.feats = output

    def pop(self):
        feats, self.feats = self.feats, None
        assert feats is not None, 'hook 没抓到编码器输出：确认 enabled=True 且 hook 挂在 de_parallel(model).encoder 上'
        return feats

    def remove(self):
        self._handle.remove()
