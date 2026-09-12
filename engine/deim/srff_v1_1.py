"""
SRFF-V1.1: 全局退化触发的单层安全残差融合。

在已冻结的 SRFF-V1 之上做**单原因、可归因**的改进（依据
``SRFF-V1.1-改进实现文档-Agent.md`` 与 fast72 seed0 训练后诊断）：

* **保留** V1 已在 GDC/PDC 产生正收益的局部鲁棒校正——Gaussian / trimmed-mean /
  router / gate_net / A_pre 的计算**完全不改**（直接复用父类 ``_core``）；
* 在其**外部**增加一个**按样本**计算的全局退化触发门 ``m``：把 V1 的 structure map
  重新解释为 stationarity evidence（诊断显示它在退化域 GDC/PDC 更高、清洁域更低），
  逐样本空间平均得 ``q``，经固定 smoothstep(``tau_low``,``tau_high``) 得 ``m∈[0,1]``；
  ``q<=tau_low`` → ``m=0`` 严格走恒等路径（保护 clean 域）；``q>=tau_high`` → ``m=1``
  完整保留 V1 校正（保住 robust 域已验证收益）；
* 单层启用（只保留 P5→P4）由 ``HybridEncoder.srff_active_levels`` 控制，本类不感知层数。

不依赖清洁图 / 同源配对 / 教师网络 / 域标签 / 文件名；全局门**无可训练参数、不新增
checkpoint 参数**（阈值是普通 float 配置属性）。诊断证据方向见改进文档 §2。
"""

import torch

from .srff import SelectiveRobustFrequencyFusion

__all__ = ['SelectiveRobustFrequencyFusionV11']


class SelectiveRobustFrequencyFusionV11(SelectiveRobustFrequencyFusion):
    """SRFF-V1.1：在 V1 局部校正外层叠加按样本的全局退化触发安全门。

    继承 V1，复用其 fixed filters / router / gate_net / 初始化；仅重写 ``_core``，
    在 V1 输出之上增加全局门 ``m`` 与安全残差。

    新增参数
    --------
    global_threshold_low / global_threshold_high:
        smoothstep 触发阈值 ``tau_low``/``tau_high``（默认 0.78 / 0.80），来自本次
        seed0 val 诊断中 clean 最大域均值(0.76683)与 robust 最小域均值(0.81068)之间
        的保守中段。是普通配置属性，**不注册为 parameter/buffer，不进 checkpoint**。
        要求 ``0 <= tau_low < tau_high <= 1``。
    """

    def __init__(
        self,
        channels: int,
        gaussian_kernel: int = 5,
        trim_kernel: int = 3,
        gate_hidden: int = 16,
        gate_init_bias: float = -4.0,
        eps: float = 1e-6,
        global_threshold_low: float = 0.78,
        global_threshold_high: float = 0.80,
    ):
        # 复用 V1 的全部结构、固定算子与初始化（不复制 Gaussian/trimmed 实现）。
        super().__init__(
            channels=channels,
            gaussian_kernel=gaussian_kernel,
            trim_kernel=trim_kernel,
            gate_hidden=gate_hidden,
            gate_init_bias=gate_init_bias,
            eps=eps,
        )
        tau_low = float(global_threshold_low)
        tau_high = float(global_threshold_high)
        assert 0.0 <= tau_low < tau_high <= 1.0, \
            f'require 0 <= tau_low < tau_high <= 1, got tau_low={tau_low}, tau_high={tau_high}'
        # 纯配置属性（普通 float），不注册为 parameter/buffer，因此不新增 checkpoint 参数。
        self.tau_low = tau_low
        self.tau_high = tau_high

    def _smooth_global_gate(self, score: torch.Tensor) -> torch.Tensor:
        """固定 smoothstep 触发门 m：score<=tau_low→0，>=tau_high→1，中间连续过渡。

        无可训练参数、无域标签；输入形状任意（此处为逐样本 [B,1,1,1]），输出同形状。
        """
        u = ((score - self.tau_low) / (self.tau_high - self.tau_low)).clamp(0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def _core(self, high: torch.Tensor, low: torch.Tensor) -> dict:
        """复用 V1 的 ``_core``，再叠加按样本全局门与安全残差。

        返回 V1 的全部字段，并新增/覆盖：
          * ``pre_global_gate``：V1 原 gate（A_pre）；
          * ``stationarity``：原 ``structure``（V1.1 中用作 stationarity 触发证据）；
          * ``global_score``：逐样本 ``q``（[B,1,1,1]）；
          * ``global_gate``：``m``（[B,1,1,1]）；
          * ``gate``：V1.1 最终 gate = ``m * A_pre``；
          * ``low_out``：V1.1 最终输出 = ``low + m * (Y_v1 - low)``。
        """
        out = super()._core(high, low)

        # V1 的 structure map 在 V1.1 中重新解释为 stationarity evidence（越高越像
        # 跨尺度持续、空间平稳的高频扰动=退化）。逐样本空间平均，绝不用 batch/running mean，
        # 因此 batch 组成、大小变化都不影响单张图的触发值，也不需要域标签。
        stationarity = out['structure']                                       # (B,1,H,W)
        global_score = stationarity.float().mean(dim=(2, 3), keepdim=True)     # (B,1,1,1)
        global_gate = self._smooth_global_gate(global_score)                   # (B,1,1,1) in [0,1]

        # 安全残差：global_gate=0 → low_out 与 low 数值严格相等；=1 → 数值等于 V1 输出。
        # global_gate 以 fp32 计算后按目标 dtype 转换，保证 AMP 下输出 dtype 与 low 一致。
        gate_pre = out['gate']                                                # V1 原 gate A_pre
        delta_v1 = out['low_out'] - low
        low_out = low + global_gate.to(delta_v1.dtype) * delta_v1
        gate = gate_pre * global_gate.to(gate_pre.dtype)

        out['pre_global_gate'] = gate_pre
        out['stationarity'] = stationarity
        out['global_score'] = global_score
        out['global_gate'] = global_gate
        out['gate'] = gate          # 覆盖为 V1.1 最终 gate
        out['low_out'] = low_out    # 覆盖为 V1.1 最终输出
        # 注：out['structure'] 保留为兼容诊断字段；在 V1.1 中它是 stationarity 触发证据，
        # 不再单独解释为语义结构置信度。
        return out
