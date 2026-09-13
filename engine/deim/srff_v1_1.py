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

# 全局门的三态推理干预模式（仅诊断用；纯运行期配置，绝不进 checkpoint）。
_ALLOWED_GATE_MODES = ('auto', 'force_off', 'force_on')
# 局部门 / 专家路由的诊断 override 允许值（仅评估期；纯运行期配置，绝不进 checkpoint）。
_ALLOWED_LOCAL_GATE_MODES = ('learned', 'constant')
_ALLOWED_ROUTER_MODES = ('learned', 'gaussian_only', 'trimmed_only')


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
    global_gate_mode:
        ``'auto'``（默认，smoothstep 全局门，等价既有 V1.1 行为）/ ``'force_off'``（严格恒等，
        诊断用）/ ``'force_on'``（严格复用 V1 输出，诊断用）。普通字符串属性，不入 checkpoint，
        仅由运行配置决定；三态用于对同一 checkpoint 做推理期因果干预，不改变任何权重。
    diagnostic_local_gate_mode / diagnostic_local_gate_value / diagnostic_router_mode:
        局部门与双专家容量拆分的**评估期诊断 override**（默认 ``learned``/``0.02``/``learned``，
        即完全不激活、逐位保持既有 V1.1 行为）。``local_gate_mode='constant'`` 时把 learned
        local gate 替换为常数 ``local_gate_value``（α）；``router_mode='gaussian_only'/'trimmed_only'``
        时把 router 权重强制为 1/0 或 0/1。均为普通属性，不入 checkpoint；激活（任一非 learned）
        时若 ``self.training`` 为真，forward 直接抛 RuntimeError（diagnostic override is evaluation-only）。
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
        global_gate_mode: str = 'auto',
        diagnostic_local_gate_mode: str = 'learned',
        diagnostic_local_gate_value: float = 0.02,
        diagnostic_router_mode: str = 'learned',
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
        assert global_gate_mode in _ALLOWED_GATE_MODES, \
            f'global_gate_mode 必须是 {_ALLOWED_GATE_MODES} 之一，得到 {global_gate_mode!r}'
        # 三态门控模式：普通字符串属性，不注册 parameter/buffer、不入 state_dict，仅由运行配置决定。
        self.global_gate_mode = global_gate_mode
        # 诊断 override（局部门 / 专家路由）：均为普通属性，不注册 parameter/buffer、不入 state_dict。
        assert diagnostic_local_gate_mode in _ALLOWED_LOCAL_GATE_MODES, \
            f'diagnostic_local_gate_mode 必须是 {_ALLOWED_LOCAL_GATE_MODES} 之一，得到 {diagnostic_local_gate_mode!r}'
        assert diagnostic_router_mode in _ALLOWED_ROUTER_MODES, \
            f'diagnostic_router_mode 必须是 {_ALLOWED_ROUTER_MODES} 之一，得到 {diagnostic_router_mode!r}'
        dgv = float(diagnostic_local_gate_value)
        assert 0.0 <= dgv <= 1.0, f'diagnostic_local_gate_value 必须在 [0,1]，得到 {dgv}'
        self.diagnostic_local_gate_mode = diagnostic_local_gate_mode
        self.diagnostic_local_gate_value = dgv
        self.diagnostic_router_mode = diagnostic_router_mode

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
        # 训练期硬阻断（§5.1）：诊断 override 仅限评估期；默认 learned+learned 不激活，训练不受影响。
        override_active = (self.diagnostic_local_gate_mode != 'learned') \
            or (self.diagnostic_router_mode != 'learned')
        if self.training and override_active:
            raise RuntimeError(
                'SRFF-V1.1 diagnostic override is evaluation-only：'
                '训练期禁止激活 local_gate/router 诊断 override（默认 learned+learned 不受影响）')

        out = super()._core(high, low)

        # V1 的 structure map 在 V1.1 中重新解释为 stationarity evidence（越高越像
        # 跨尺度持续、空间平稳的高频扰动=退化）。逐样本空间平均，绝不用 batch/running mean，
        # 因此 batch 组成、大小变化都不影响单张图的触发值，也不需要域标签。
        stationarity = out['structure']                                       # (B,1,H,W)
        global_score = stationarity.float().mean(dim=(2, 3), keepdim=True)     # (B,1,1,1)
        gate_pre = out['gate']  # V1 原 gate A_pre = pre_global_gate
        v1_low_out = out['low_out']  # V1 原输出

        mode = self.global_gate_mode
        if mode == 'force_off':
            # 严格恒等：显式 low_out = low（不用 low + 0*delta，避免异常值传播），gate 全 0。
            global_gate = torch.zeros_like(global_score)
            low_out = low
            gate = torch.zeros_like(gate_pre)
        elif mode == 'force_on':
            # 严格复用 V1 输出：low_out = v1_low_out（不重算，避免数值漂移），gate = pre_global_gate。
            global_gate = torch.ones_like(global_score)
            low_out = v1_low_out
            gate = gate_pre
        else:  # 'auto'：保留原 V1.1 行为
            # 安全残差：global_gate=0 → low_out 与 low 严格相等；=1 → 数值等于 V1 输出。
            # global_gate 以 fp32 计算后按目标 dtype 转换，保证 AMP 下输出 dtype 与 low 一致。
            global_gate = self._smooth_global_gate(global_score)  # (B,1,1,1) in [0,1]
            delta_v1 = v1_low_out - low
            low_out = low + global_gate.to(delta_v1.dtype) * delta_v1
            gate = gate_pre * global_gate.to(gate_pre.dtype)

        out['pre_global_gate'] = gate_pre
        out['stationarity'] = stationarity
        out['global_score'] = global_score
        out['global_gate'] = global_gate
        out['gate'] = gate          # 覆盖为 V1.1 最终 gate
        out['low_out'] = low_out    # 覆盖为 V1.1 最终输出

        # ---- 诊断 override（局部门 / 专家路由）：仅评估期。默认 learned+learned 不激活，
        # 上面 auto/force_off/force_on 的原有数值路径完全不被重构（避免浮点顺序变化）。
        learned_wg = out['gaussian_weight']  # V1 learned router 权重 w_g
        learned_wt = out['trimmed_weight']  # V1 learned router 权重 w_t
        learned_local_gate = gate_pre  # V1 learned local gate a（= pre_global_gate）
        out['learned_gaussian_weight'] = learned_wg
        out['learned_trimmed_weight'] = learned_wt
        out['learned_local_gate'] = learned_local_gate
        out['diagnostic_override_active'] = override_active
        if override_active:
            # 容量实验必须 global_gate_mode=force_on，否则组合不可解释，立即报错（§5.5）。
            if self.global_gate_mode != 'force_on':
                raise RuntimeError(
                    'SRFF-V1.1 diagnostic override 要求 global_gate_mode=force_on，'
                    f'当前为 {self.global_gate_mode!r}')
            # Router override：实际使用的专家权重（learned 原值已在上面保留）。
            if self.diagnostic_router_mode == 'gaussian_only':
                used_wg = torch.ones_like(learned_wg)
                used_wt = torch.zeros_like(learned_wt)
            elif self.diagnostic_router_mode == 'trimmed_only':
                used_wg = torch.zeros_like(learned_wg)
                used_wt = torch.ones_like(learned_wt)
            else:  # 'learned'
                used_wg, used_wt = learned_wg, learned_wt
            # Local gate override：实际使用的局部残差系数（constant 时为固定 α）。
            if self.diagnostic_local_gate_mode == 'constant':
                local_gate_used = torch.full_like(learned_local_gate, self.diagnostic_local_gate_value)
            else:  # 'learned'
                local_gate_used = learned_local_gate
            # 用实际专家 + 固定 α 重构（global_gate 此时为 force_on 的全 1，形状 (B,1,1,1)）。
            g_low = out['gaussian_low']
            t_low = out['trimmed_low']
            f_rob_used = used_wg * g_low + used_wt * t_low
            delta_used = f_rob_used - low
            low_out = low + global_gate.to(delta_used.dtype) * local_gate_used.to(delta_used.dtype) * delta_used
            final_gate = global_gate.to(local_gate_used.dtype) * local_gate_used
            out['gate'] = final_gate
            out['low_out'] = low_out
            out['gaussian_weight'] = used_wg
            out['trimmed_weight'] = used_wt
            out['local_gate_used'] = local_gate_used
        # 注：out['structure'] 保留为兼容诊断字段；在 V1.1 中它是 stationarity 触发证据，
        # 不再单独解释为语义结构置信度。
        return out
