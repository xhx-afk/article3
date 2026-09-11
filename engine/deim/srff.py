"""
SRFF v1: Selective Robust Frequency Fusion (选择性鲁棒频率融合)

第一创新点的独立模块实现。设计冻结说明见仓库根目录
``SRFF-V1-Agent-Implementation.md``。

核心思想
--------
SRFF v1 只依赖当前批次的特征，不依赖清洁图、同源配对、退化类型标签或教师网络。
它在 ``HybridEncoder`` 的自顶向下融合路径中，对低层特征 ``low`` 做一次保守的、
近似恒等初始化的选择性校正：

* 当低层特征呈现高频噪声、且缺少跨尺度与方向结构一致性时，才用固定鲁棒基底
  （Gaussian 平滑 / 3x3 trimmed-mean）做小幅校正；
* 当结构可信时（``S -> 1``）尽量保留原始特征（校正门 ``A -> 0``）。

所有噪声/结构证据在送入门控与路由前均与检测梯度解耦（在 ``torch.no_grad()`` 下计算，
语义等价于对 Z 与 S 执行 ``detach()``），目的是避免 backbone 通过操纵噪声统计走捷径；
检测梯度仍然通过原始残差通路、鲁棒基底与门控参数传播。

本模块不使用 BatchNorm、不使用 ``unfold``、不引入 mmcv 或自定义 CUDA 算子。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SelectiveRobustFrequencyFusion']


class SelectiveRobustFrequencyFusion(nn.Module):
    """选择性鲁棒频率融合（SRFF v1）。

    参数
    ----
    channels:
        ``high`` 与 ``low`` 的通道数（当前 baseline 统一为 256）。
    gaussian_kernel:
        固定 Gaussian 鲁棒基底的核尺寸，必须为正奇数，默认 5（5x5 Pascal 核）。
    trim_kernel:
        固定 trimmed-mean 鲁棒基底的核尺寸，v1 只允许 3。
    gate_hidden:
        两个轻量网络（gate/router）的隐藏通道数，必须为正。
    gate_init_bias:
        ``gate_net`` 末层 bias 初始值，默认 -4.0，使 ``sigmoid(-4)≈0.018``，
        从而初始最大校正比例约为 1.8%。这是保护 ODC 与 baseline 初始行为的关键，
        不得改成 0。
    eps:
        所有归一化/一致性计算的分母保护项。
    """

    def __init__(
        self,
        channels: int,
        gaussian_kernel: int = 5,
        trim_kernel: int = 3,
        gate_hidden: int = 16,
        gate_init_bias: float = -4.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        # ---- 尽早校验非法参数 ----
        # eps / gate_init_bias 可能来自 YAML：科学计数法若误写为 1e-6 会被 PyYAML 解析成
        # 字符串，这里统一强制转 float，避免比较时抛 TypeError。
        eps = float(eps)
        gate_init_bias = float(gate_init_bias)
        assert isinstance(channels, int) and channels > 0, \
            f'channels must be a positive int, got {channels}'
        assert isinstance(gaussian_kernel, int) and gaussian_kernel > 0 \
            and gaussian_kernel % 2 == 1, \
            f'gaussian_kernel must be a positive odd int, got {gaussian_kernel}'
        assert trim_kernel == 3, \
            f'SRFF v1 only supports trim_kernel=3, got {trim_kernel}'
        assert isinstance(gate_hidden, int) and gate_hidden > 0, \
            f'gate_hidden must be a positive int, got {gate_hidden}'
        assert eps > 0, f'eps must be positive, got {eps}'

        self.channels = channels
        self.gaussian_kernel = gaussian_kernel
        self.trim_kernel = trim_kernel
        self.gate_hidden = gate_hidden
        self.gate_init_bias = gate_init_bias
        self.eps = eps

        # ---- 固定 Gaussian 鲁棒基底（buffer，不参与训练）----
        # 形状 (channels, 1, k, k)，配合 F.conv2d(..., groups=channels) 做 depthwise。
        # persistent=False：它完全由配置决定，无需写入 checkpoint，也避免 dtype/device 漂移。
        gaussian_weight = self._build_pascal_kernel(channels, gaussian_kernel)
        self.register_buffer('gaussian_weight', gaussian_weight, persistent=False)

        # ---- 两个轻量网络，均不使用 BatchNorm ----
        # gate_net:   Conv3x3(4->gate_hidden) -> SiLU -> Conv1x1(gate_hidden->1)
        # router_net: Conv3x3(4->gate_hidden) -> SiLU -> Conv1x1(gate_hidden->2)
        self.gate_net = nn.Sequential(
            nn.Conv2d(4, gate_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, kernel_size=1),
        )
        self.router_net = nn.Sequential(
            nn.Conv2d(4, gate_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, 2, kernel_size=1),
        )

        self._reset_parameters()

    # ------------------------------------------------------------------
    # 初始化与固定核构造
    # ------------------------------------------------------------------
    def _reset_parameters(self):
        """按冻结设计初始化 gate/router 网络。

        * 两个首层卷积沿用 PyTorch 默认的 Kaiming 初始化，仅将 bias 置 0；
        * 两个末层卷积权重使用 ``normal_(std=1e-3)``；
        * ``gate_net`` 末层 bias 固定为 ``gate_init_bias``（默认 -4.0）；
        * ``router_net`` 末层 bias 置 0，使 Gaussian/trimmed 初始权重接近各 0.5。
        """
        for net in (self.gate_net, self.router_net):
            first_conv = net[0]
            last_conv = net[-1]
            # 首层：保留 PyTorch 默认 Kaiming 权重，bias 置 0。
            nn.init.zeros_(first_conv.bias)
            # 末层：小幅正态权重，保证近恒等初始化。
            nn.init.normal_(last_conv.weight, mean=0.0, std=1e-3)

        nn.init.constant_(self.gate_net[-1].bias, self.gate_init_bias)
        nn.init.zeros_(self.router_net[-1].bias)

    @staticmethod
    def _build_pascal_kernel(channels: int, kernel_size: int) -> torch.Tensor:
        """构造归一化 2D Pascal（近似 Gaussian）depthwise 核，形状 (C,1,k,k)。

        一维核取 Pascal 三角形第 ``kernel_size`` 行（k=5 -> [1,4,6,4,1]），
        外积后归一化使其和为 1。使用迭代构造，避免依赖 math.comb。
        """
        row = [1.0]
        for _ in range(kernel_size - 1):
            row = [1.0] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1.0]
        row = torch.tensor(row, dtype=torch.float32)
        kernel_2d = torch.outer(row, row)
        kernel_2d = kernel_2d / kernel_2d.sum()
        weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        return weight.contiguous()

    # ------------------------------------------------------------------
    # 固定鲁棒算子（全部 replicate padding，保持空间尺寸）
    # ------------------------------------------------------------------
    @staticmethod
    def _replicate_pad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')

    def _gaussian(self, x: torch.Tensor) -> torch.Tensor:
        """G(x)：固定 Gaussian 平滑，depthwise conv，不产生 unfold 大张量。"""
        pad = self.gaussian_kernel // 2
        weight = self.gaussian_weight.to(dtype=x.dtype)
        xp = self._replicate_pad(x, pad, pad)
        return F.conv2d(xp, weight, groups=x.shape[1])

    def _trimmed_mean(self, x: torch.Tensor) -> torch.Tensor:
        """T(x)：常数内存池化实现的 3x3 trimmed-mean，抑制稀疏极值。

        T(x) = (n * AvgPool(x) - MaxPool(x) - MinPool(x)) / (n - 2)，n = k*k。
        MinPool(x) = -MaxPool(-x)。这不是图像恢复目标。
        """
        k = self.trim_kernel
        pad = k // 2
        n = k * k
        xp = self._replicate_pad(x, pad, pad)
        avg = F.avg_pool2d(xp, k, stride=1)
        maxp = F.max_pool2d(xp, k, stride=1)
        minp = -F.max_pool2d(-xp, k, stride=1)
        return (n * avg - maxp - minp) / (n - 2)

    def _avg_pool(self, x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
        """带 replicate padding 的均值池化，保持空间尺寸不变。"""
        pad_h, pad_w = kh // 2, kw // 2
        xp = self._replicate_pad(x, pad_h, pad_w)
        return F.avg_pool2d(xp, (kh, kw), stride=1)

    # ------------------------------------------------------------------
    # 无标签证据与门控核心计算
    # ------------------------------------------------------------------
    def _normalize(self, e: torch.Tensor) -> torch.Tensor:
        """按样本做尺度归一化：N(E) = E / (mean_hw(E) + eps)。"""
        m = e.mean(dim=(2, 3), keepdim=True)
        return e / (m + self.eps)

    def _coh(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """coh(a,b) = clamp(1 - |a-b|/(a+b+eps), 0, 1)。"""
        return torch.clamp(1.0 - (a - b).abs() / (a + b + self.eps), 0.0, 1.0)

    def _core(self, high: torch.Tensor, low: torch.Tensor) -> dict:
        """共享的核心计算，供 forward 与 analyze 复用。

        返回的字典包含最终输出 ``low_out`` 与全部中间证据/门控张量。
        证据统计（Z、S）在送入门控前已 detach，检测梯度只通过残差、鲁棒基底与
        门控参数传播。
        """
        eps = self.eps

        # 固定鲁棒基底 G(low)/T(low)：需要梯度，参与融合 F_rob，
        # 检测梯度经此与原始残差通路传播（文档 5.6）。
        g_low = self._gaussian(low)
        t_low = self._trimmed_mean(low)

        # 无标签噪声/结构证据：这些统计量只用于门控与路由，必须在送入小网络前 detach
        # （文档 5.6），以避免 backbone 通过操纵噪声统计走捷径。整体置于 no_grad 下，
        # 语义等价于对 Z 与 S 执行 detach()，同时省去一份用完即弃的计算图（降低开销）。
        with torch.no_grad():
            g_high = self._gaussian(high)

            # 5.4 通道平均的高频残差能量，形状 (B,1,H,W)。
            e_l = (low - g_low).abs().mean(dim=1, keepdim=True)
            e_t = (low - t_low).abs().mean(dim=1, keepdim=True)
            e_h = (high - g_high).abs().mean(dim=1, keepdim=True)

            # 将 E_h 用 bilinear(align_corners=False) 对齐到 low 尺寸（不假设边长固定）。
            e_h_up = F.interpolate(e_h, size=low.shape[-2:], mode='bilinear', align_corners=False)

            # 每个能量图按样本做尺度归一化。
            n_l = self._normalize(e_l)
            n_t = self._normalize(e_t)
            n_h = self._normalize(e_h_up)

            # 跨尺度不一致性 C。
            c = torch.clamp((n_l - n_h).abs() / (n_l + n_h + eps), 0.0, 1.0)

            # 5.5 无标签结构证据（在 N(E_l) 上计算局部/方向一致性）。
            l_coh = self._coh(n_l, self._avg_pool(n_l, 3, 3))
            h_coh = self._coh(n_l, self._avg_pool(n_l, 1, 3))
            v_coh = self._coh(n_l, self._avg_pool(n_l, 3, 1))
            d = torch.maximum(h_coh, v_coh)
            p = 1.0 - c
            s = torch.clamp(p * torch.maximum(l_coh, d), 0.0, 1.0)

            # Z = concat[N(E_l), N(E_t), C, 1-S]（已无梯度）。
            z = torch.cat([n_l, n_t, c, 1.0 - s], dim=1)

        # 5.6 可学习门控与鲁棒基底路由（两个轻量网络均不使用 BatchNorm）。
        weights = torch.softmax(self.router_net(z), dim=1)
        w_g = weights[:, 0:1]
        w_t = weights[:, 1:2]
        f_rob = w_g * g_low + w_t * t_low

        a_raw = torch.sigmoid(self.gate_net(z))
        a = a_raw * (1.0 - s)

        low_out = low + a * (f_rob - low)

        return {
            'gaussian_low': g_low,
            'trimmed_low': t_low,
            'noise_l': n_l,
            'noise_t': n_t,
            'cross_scale': c,
            'structure': s,
            'gaussian_weight': w_g,
            'trimmed_weight': w_t,
            'gate_raw': a_raw,
            'gate': a,
            'low_out': low_out,
        }

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """对低层特征做选择性鲁棒频率校正。

        Args:
            high: 高层特征 (B, C, H_h, W_h)，已经过 lateral conv，仅作结构证据。
            low:  低层特征 (B, C, H_l, W_l)，已经过 input proj，是被校正对象。

        Returns:
            与 ``low`` 形状/dtype/device 一致的张量。
        """
        assert high.dim() == 4 and low.dim() == 4, \
            f'SRFF expects 4D (B,C,H,W) tensors, got high{tuple(high.shape)} low{tuple(low.shape)}'
        assert high.shape[1] == self.channels and low.shape[1] == self.channels, \
            f'SRFF channel mismatch: expect {self.channels}, got high={high.shape[1]} low={low.shape[1]}'
        assert high.shape[0] == low.shape[0], \
            f'SRFF batch mismatch: high={high.shape[0]} low={low.shape[0]}'

        return self._core(high, low)['low_out']

    @torch.no_grad()
    def analyze(self, high: torch.Tensor, low: torch.Tensor) -> dict:
        """诊断接口，仅供烟测/离线分析复用与 forward 相同的内部计算。

        返回一组 detached 的 0-dim 张量标量；不持久保存特征图，不默认打印。
        不得在每个训练 iteration 调用（会同步 GPU 标量、拖慢训练）。
        """
        out = self._core(high, low)
        gate = out['gate']
        structure = out['structure']
        low_out = out['low_out']

        gate_flat = gate.detach().flatten().float()
        numel = gate_flat.numel()
        # 用排序取分位，避免 torch.quantile 在大张量上的元素数上限。
        sorted_gate, _ = torch.sort(gate_flat)
        p95_index = min(numel - 1, int(round(0.95 * (numel - 1)))) if numel > 0 else 0
        gate_p95 = sorted_gate[p95_index] if numel > 0 else gate_flat.new_zeros(())

        low_out_f = low_out.detach().float()
        low_f = low.detach().float()
        relative_delta = (low_out_f - low_f).abs().mean() / (low_f.abs().mean() + self.eps)

        return {
            'gate_mean': gate_flat.mean(),
            'gate_p95': gate_p95,
            'gate_max': gate_flat.max() if numel > 0 else gate_flat.new_zeros(()),
            'structure_mean': structure.detach().float().mean(),
            'gaussian_weight_mean': out['gaussian_weight'].detach().float().mean(),
            'trimmed_weight_mean': out['trimmed_weight'].detach().float().mean(),
            'relative_delta': relative_delta,
        }
