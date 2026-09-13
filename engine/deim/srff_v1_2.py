"""
SRFF-V1.2: 冻结基线之上的“证据条件化可学习双专家残差适配器”。

依据 ``SRFF-V1.2-冻结基线可学习双专家适配器-实现文档-Agent.md``。上一轮无重训容量拆分已证伪
“固定 Gaussian/trimmed 基函数本身就有足够检测增益”（GaussianBestGDC=+0.0074、TrimmedBestPDC=+0.0731、
NO_FIXED_BASIS_SIGNAL）。因此 V1.2 不再注入固定算子输出，而是：

* 用**固定算子只生成证据与专家输入**（r_g=low-Gaussian(low)、r_p=low-Trimmed(low)、r_cross=low-high_up）；
* 两个**可学习有符号残差专家**（Gaussian-like / Impulse-like），末层零初始化 → 初始 ``low_out`` 与 ``low``
  逐元素完全相同（adapter-off 严格还原 baseline）；
* **无域标签的受约束路由**：由 impulse evidence 的每图变异系数 cv 经两个可学习标量（threshold、
  raw_temperature）得每图标量权重 ``w_p=sigmoid((log cv - threshold)/temperature)``、``w_g=1-w_p``；
* **全局安全门**：复用 V1.1 已验证的无标签 stationarity evidence + 固定 tau=[0.78,0.80]，只缩放 adapter 残差；
* **相对 RMS 残差预算**：``budget=alpha_max*low_rms``，``budget_scale=clamp(budget/delta_rms,max=1).detach()``。

本模块**不继承 V1/V1.1**（避免带入被证伪的固定输出注入与旧门控参数），仅复制无参数统计函数并独立测试。
门模式仅评估期允许 force_off/force_on；训练期只允许 auto，否则报错。所有固定 kernel 为 persistent=False buffer。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FrozenBaseEvidenceConditionedResidualAdapter']

_ALLOWED_GATE_MODES = ('auto', 'force_off', 'force_on')
_ALLOWED_EXPERT_MODES = ('dual', 'gaussian_only', 'impulse_only', 'equal_weight')
_TEMP_FLOOR = 0.05


def _build_pascal_kernel(channels: int, kernel_size: int) -> torch.Tensor:
    """归一化 2D Pascal（近似 Gaussian）depthwise 核，形状 (C,1,k,k)。迭代构造，不依赖 math.comb。"""
    row = [1.0]
    for _ in range(kernel_size - 1):
        row = [1.0] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1.0]
    row = torch.tensor(row, dtype=torch.float32)
    k2d = torch.outer(row, row)
    k2d = k2d / k2d.sum()
    return k2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1).contiguous()


def _inverse_softplus(y: float) -> float:
    """softplus(x)=y 的反函数 x=log(exp(y)-1)；要求 y>0。"""
    if y <= 0:
        raise ValueError(f'inverse-softplus 需要 y>0，得到 {y}')
    return math.log(math.expm1(y))


class _ResidualExpert(nn.Module):
    """可学习有符号残差专家（无 BN/Dropout/stochastic depth）。

    Conv1x1(2C->bn,bias=False) → GroupNorm(8,bn) → SiLU → DepthwiseConv(k) →
    GroupNorm(8,bn) → SiLU → Conv1x1(bn->C,bias=True)【weight/bias 零初始化】。
    末层零初始化保证初始输出恒为 0（从而 low_out==low）；输出为有符号残差，禁止 ReLU 截断。
    """

    def __init__(self, channels: int, bottleneck: int, depthwise_kernel: int, norm_groups: int = 8):
        super().__init__()
        assert bottleneck % norm_groups == 0, \
            f'bottleneck({bottleneck}) 必须能被 norm_groups({norm_groups}) 整除'
        assert depthwise_kernel % 2 == 1 and depthwise_kernel > 0, \
            f'depthwise_kernel 必须为正奇数，得到 {depthwise_kernel}'
        self.in_proj = nn.Conv2d(2 * channels, bottleneck, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(norm_groups, bottleneck)
        self.act1 = nn.SiLU()
        self.dw = nn.Conv2d(bottleneck, bottleneck, kernel_size=depthwise_kernel,
                            padding=depthwise_kernel // 2, groups=bottleneck, bias=False)
        self.gn2 = nn.GroupNorm(norm_groups, bottleneck)
        self.act2 = nn.SiLU()
        self.out_proj = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=True)
        # 末层零初始化：adapter 初始 low_out 与 low 逐元素完全相同。
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.gn1(self.in_proj(x)))
        x = self.act2(self.gn2(self.dw(x)))
        return self.out_proj(x)   # 有符号残差，无激活截断


class FrozenBaseEvidenceConditionedResidualAdapter(nn.Module):
    """冻结基线之上的证据条件化可学习双专家残差适配器（SRFF-V1.2）。

    参数
    ----
    channels: high/low 通道数（baseline 统一 256）。
    bottleneck_channels: 专家瓶颈通道（默认 64，须被 8 整除）。
    alpha_max: 相对 RMS 残差预算上限（默认 0.10）。
    global_tau: 全局门 smoothstep 阈值 (tau_low, tau_high)，默认 (0.78, 0.80)。
    router_threshold_init / router_temperature_init: 路由两个可学习标量的初值；温度经 inverse-softplus
        初始化，使 ``softplus(raw)+0.05`` 的初始有效温度约等于 router_temperature_init（须 >0.05）。
    gate_mode: 'auto'（训练/主实验）/ 'force_off'（严格返回 low，不跑专家）/ 'force_on'（仅推理诊断）。
    expert_mode: 'dual'（训练）/ 'gaussian_only' / 'impulse_only' / 'equal_weight'（消融）。
    """

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int = 64,
        alpha_max: float = 0.10,
        global_tau=(0.78, 0.80),
        router_threshold_init: float = 0.0,
        router_temperature_init: float = 0.25,
        gate_mode: str = 'auto',
        expert_mode: str = 'dual',
        gaussian_kernel: int = 5,
        trim_kernel: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert isinstance(channels, int) and channels > 0, f'channels 必须为正整数，得到 {channels}'
        assert gate_mode in _ALLOWED_GATE_MODES, f'gate_mode 必须是 {_ALLOWED_GATE_MODES} 之一，得到 {gate_mode!r}'
        assert expert_mode in _ALLOWED_EXPERT_MODES, \
            f'expert_mode 必须是 {_ALLOWED_EXPERT_MODES} 之一，得到 {expert_mode!r}'
        assert gaussian_kernel % 2 == 1 and gaussian_kernel > 0, f'gaussian_kernel 必须为正奇数，得到 {gaussian_kernel}'
        assert trim_kernel == 3, f'SRFF-V1.2 trimmed 只支持 trim_kernel=3，得到 {trim_kernel}'
        tau_low, tau_high = float(global_tau[0]), float(global_tau[1])
        assert 0.0 <= tau_low < tau_high <= 1.0, f'要求 0<=tau_low<tau_high<=1，得到 {global_tau}'
        assert float(alpha_max) > 0, f'alpha_max 必须为正，得到 {alpha_max}'
        assert float(router_temperature_init) > _TEMP_FLOOR, \
            f'router_temperature_init 必须 > {_TEMP_FLOOR}，得到 {router_temperature_init}'

        self.channels = channels
        self.bottleneck_channels = int(bottleneck_channels)
        self.alpha_max = float(alpha_max)
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.gaussian_kernel = int(gaussian_kernel)
        self.trim_kernel = int(trim_kernel)
        self.eps = float(eps)
        # 运行期模式：普通属性，不注册 parameter/buffer、不入 state_dict。
        self.gate_mode = gate_mode
        self.expert_mode = expert_mode

        # 固定 kernel（persistent=False，不入 checkpoint）
        self.register_buffer('gaussian_weight', _build_pascal_kernel(channels, self.gaussian_kernel), persistent=False)

        # 两个可学习有符号残差专家（末层零初始化；不共享末层）
        self.expert_gaussian = _ResidualExpert(channels, self.bottleneck_channels, depthwise_kernel=5)
        self.expert_impulse = _ResidualExpert(channels, self.bottleneck_channels, depthwise_kernel=3)

        # 受约束路由：仅两个可学习标量
        self.router_threshold = nn.Parameter(torch.tensor(float(router_threshold_init), dtype=torch.float32))
        raw_temp0 = _inverse_softplus(float(router_temperature_init) - _TEMP_FLOOR)
        self.router_raw_temperature = nn.Parameter(torch.tensor(raw_temp0, dtype=torch.float32))

    # ------------------------------------------------------------------
    # 固定算子（复制自 V1 的无参数统计实现，独立测试）
    # ------------------------------------------------------------------
    @staticmethod
    def _replicate_pad(x, ph, pw):
        return F.pad(x, (pw, pw, ph, ph), mode='replicate')

    def _gaussian(self, x):
        pad = self.gaussian_kernel // 2
        w = self.gaussian_weight.to(dtype=x.dtype)
        return F.conv2d(self._replicate_pad(x, pad, pad), w, groups=x.shape[1])

    def _trimmed_mean(self, x):
        k = self.trim_kernel
        pad = k // 2
        n = k * k
        xp = self._replicate_pad(x, pad, pad)
        avg = F.avg_pool2d(xp, k, stride=1)
        maxp = F.max_pool2d(xp, k, stride=1)
        minp = -F.max_pool2d(-xp, k, stride=1)
        return (n * avg - maxp - minp) / (n - 2)

    def _normalize(self, e):
        m = e.mean(dim=(2, 3), keepdim=True)
        return e / (m + self.eps)

    def _coh(self, a, b):
        return torch.clamp(1.0 - (a - b).abs() / (a + b + self.eps), 0.0, 1.0)

    def _avg_pool(self, x, kh, kw):
        return F.avg_pool2d(self._replicate_pad(x, kh // 2, kw // 2), (kh, kw), stride=1)

    @torch.no_grad()
    def _stationarity(self, high, low, gaussian_low):
        """复用 V1.1 已验证的无标签 stationarity evidence（=V1 的 structure s），供全局门使用。

        全程 no_grad（固定启发式、无可学习参数），归一化统计在 FP32 完成。
        """
        high_f = high.float()
        low_f = low.float()
        g_low_f = gaussian_low.float()
        g_high_f = self._gaussian(high_f)
        e_l = (low_f - g_low_f).abs().mean(dim=1, keepdim=True)
        e_h = (high_f - g_high_f).abs().mean(dim=1, keepdim=True)
        e_h_up = F.interpolate(e_h, size=low_f.shape[-2:], mode='bilinear', align_corners=False)
        n_l = self._normalize(e_l)
        n_h = self._normalize(e_h_up)
        c = torch.clamp((n_l - n_h).abs() / (n_l + n_h + self.eps), 0.0, 1.0)
        l_coh = self._coh(n_l, self._avg_pool(n_l, 3, 3))
        h_coh = self._coh(n_l, self._avg_pool(n_l, 1, 3))
        v_coh = self._coh(n_l, self._avg_pool(n_l, 3, 1))
        d = torch.maximum(h_coh, v_coh)
        p = 1.0 - c
        return torch.clamp(p * torch.maximum(l_coh, d), 0.0, 1.0)

    def _smooth_global_gate(self, score):
        u = ((score - self.tau_low) / (self.tau_high - self.tau_low)).clamp(0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def _route(self, r_p):
        """无域标签受约束路由：返回 w_g, w_p, cv, temperature（均 [N,1,1,1]，w_g+w_p=1）。"""
        e = r_p.abs().mean(dim=1, keepdim=True)                       # (B,1,H,W)
        ef = e.float()
        mu = ef.mean(dim=(-2, -1), keepdim=True)                       # (B,1,1,1)
        std = ((ef - mu).square().mean(dim=(-2, -1), keepdim=True) + self.eps).sqrt()
        cv = std / (mu + self.eps)                                     # (B,1,1,1)
        temperature = F.softplus(self.router_raw_temperature) + _TEMP_FLOOR
        w_p = torch.sigmoid((torch.log(cv + self.eps) - self.router_threshold) / temperature)
        w_g = 1.0 - w_p
        return w_g, w_p, cv, temperature

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------
    def _core(self, high: torch.Tensor, low: torch.Tensor) -> dict:
        # 训练期只允许 auto（force_off/on 仅推理诊断）
        if self.training and self.gate_mode != 'auto':
            raise RuntimeError(
                f'SRFF-V1.2 训练期只允许 gate_mode=auto，得到 {self.gate_mode!r}'
                '（force_off/force_on 仅用于推理诊断）')

        eps = self.eps
        # 固定证据（输入 dtype 计算算子；统计在 FP32）
        high_up = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        gaussian_low = self._gaussian(low)
        trimmed_low = self._trimmed_mean(low)
        r_g = low - gaussian_low
        r_p = low - trimmed_low
        r_cross = low - high_up

        # stationarity + 全局门
        stationarity = self._stationarity(high, low, gaussian_low)                 # (B,1,H,W)
        global_score = stationarity.float().mean(dim=(2, 3), keepdim=True)          # (B,1,1,1)

        # force_off：严格返回 low，且不运行专家（§2.5 / 测试 item 2）
        if self.gate_mode == 'force_off':
            return {
                'low_out': low, 'gate_mode': self.gate_mode, 'expert_mode': self.expert_mode,
                'stationarity': stationarity, 'global_score': global_score,
                'global_gate': torch.zeros_like(global_score),
                'ran_experts': False,
            }

        if self.gate_mode == 'force_on':
            global_gate = torch.ones_like(global_score)
        else:  # auto
            global_gate = self._smooth_global_gate(global_score)

        # 两个专家的有符号残差
        delta_g = self.expert_gaussian(torch.cat([r_g, r_cross], dim=1))
        delta_p = self.expert_impulse(torch.cat([r_p, r_cross], dim=1))

        # 路由权重
        w_g, w_p, cv, temperature = self._route(r_p)
        w_g = w_g.to(delta_g.dtype)
        w_p = w_p.to(delta_p.dtype)

        em = self.expert_mode
        if em == 'dual':
            delta_mix = w_g * delta_g + w_p * delta_p
        elif em == 'gaussian_only':
            delta_mix = delta_g
        elif em == 'impulse_only':
            delta_mix = delta_p
        else:  # equal_weight
            delta_mix = 0.5 * delta_g + 0.5 * delta_p

        # 相对 RMS 残差预算（FP32 统计；budget_scale detach）
        low_rms = (low.float().square().mean(dim=(1, 2, 3), keepdim=True) + eps).sqrt()
        delta_rms_pre = (delta_mix.float().square().mean(dim=(1, 2, 3), keepdim=True) + eps).sqrt()
        budget = self.alpha_max * low_rms
        budget_scale = torch.clamp(budget / (delta_rms_pre + eps), max=1.0).detach()
        delta_cap = delta_mix * budget_scale.to(delta_mix.dtype)
        delta_rms_post = (delta_cap.float().square().mean(dim=(1, 2, 3), keepdim=True) + eps).sqrt()
        clamp_fraction = (budget_scale < 1.0).float().mean()

        low_out = low + global_gate.to(delta_cap.dtype) * delta_cap
        final_gate = global_gate  # 门只缩放 adapter 残差，不改 baseline 融合路径

        return {
            'low_out': low_out, 'gate_mode': self.gate_mode, 'expert_mode': self.expert_mode,
            'ran_experts': True,
            'gaussian_low': gaussian_low, 'trimmed_low': trimmed_low,
            'r_g': r_g, 'r_p': r_p, 'r_cross': r_cross, 'high_up': high_up,
            'stationarity': stationarity, 'global_score': global_score, 'global_gate': global_gate,
            'cv': cv, 'w_g': w_g, 'w_p': w_p, 'temperature': temperature,
            'delta_g': delta_g, 'delta_p': delta_p, 'delta_mix': delta_mix, 'delta_cap': delta_cap,
            'low_rms': low_rms, 'delta_rms_pre': delta_rms_pre, 'delta_rms_post': delta_rms_post,
            'budget_scale': budget_scale, 'clamp_fraction': clamp_fraction,
            'final_gate': final_gate,
        }

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        assert high.dim() == 4 and low.dim() == 4, \
            f'SRFF-V1.2 期望 4D (B,C,H,W)，得到 high{tuple(high.shape)} low{tuple(low.shape)}'
        assert high.shape[1] == self.channels and low.shape[1] == self.channels, \
            f'SRFF-V1.2 通道不匹配：期望 {self.channels}，得到 high={high.shape[1]} low={low.shape[1]}'
        assert high.shape[0] == low.shape[0], f'batch 不匹配：high={high.shape[0]} low={low.shape[0]}'
        return self._core(high, low)['low_out']
