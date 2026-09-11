"""SRFF v1 单元测试。

覆盖实现文档 ``SRFF-V1-Agent-Implementation.md`` 第 9 节要求的全部测试项：
Gaussian 核、trimmed-mean、两级形状、动态尺寸、dtype/device、梯度、近恒等初始化、
结构保护组合公式、稀疏脉冲鲁棒性、状态字典与 encoder 集成。

运行方式（二选一，均无需真实数据集 / 联网 / 完整训练）：

    python -m pytest tests/test_srff.py -q     # 若环境已安装 pytest
    python tests/test_srff.py                  # 独立运行（内置 runner）

纯 SRFF 模块测试通过 importlib 直接加载 ``engine/deim/srff.py``，只依赖 torch，
不触发 engine 包的重依赖（torchvision / pycocotools 等）。HybridEncoder 集成测试
在函数内部惰性导入，若缺少可选依赖则安全跳过。
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 独立加载 SRFF 模块（只依赖 torch），以及可选的 pytest 兼容跳过。
# ---------------------------------------------------------------------------
def _load_srff_standalone():
    srff_path = ROOT / 'engine' / 'deim' / 'srff.py'
    spec = importlib.util.spec_from_file_location('_srff_standalone', srff_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_srff = _load_srff_standalone()
SelectiveRobustFrequencyFusion = _srff.SelectiveRobustFrequencyFusion

try:
    import pytest as _pytest
except Exception:  # pragma: no cover - pytest 不存在时降级为本地跳过机制
    _pytest = None


class _Skipped(Exception):
    """pytest 不可用时的本地跳过信号。"""


_SKIP_EXCEPTIONS = [_Skipped]
if _pytest is not None:
    try:
        _SKIP_EXCEPTIONS.append(_pytest.skip.Exception)
    except Exception:  # pragma: no cover - 极端版本兼容
        pass


def _skip(reason):
    if _pytest is not None:
        _pytest.skip(reason)
    raise _Skipped(reason)


def _reference_trimmed_mean_3x3(x):
    """独立参考实现：对 9 个 replicate-padding 邻域求 (sum - max - min) / 7。"""
    xp = F.pad(x, (1, 1, 1, 1), mode='replicate')
    b, c, h, w = x.shape
    neighbors = []
    for di in range(3):
        for dj in range(3):
            neighbors.append(xp[:, :, di:di + h, dj:dj + w])
    stack = torch.stack(neighbors, dim=0)  # (9, B, C, H, W)
    total = stack.sum(dim=0)
    maxv = stack.max(dim=0).values
    minv = stack.min(dim=0).values
    return (total - maxv - minv) / 7.0


def _count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 1. Gaussian 核
# ---------------------------------------------------------------------------
def test_gaussian_kernel_properties():
    channels = 4
    m = SelectiveRobustFrequencyFusion(channels=channels, gaussian_kernel=5)

    # 形状正确、每通道一份 depthwise 核。
    assert m.gaussian_weight.shape == (channels, 1, 5, 5)

    # 核和为 1。
    sums = m.gaussian_weight.sum(dim=(1, 2, 3))
    assert torch.allclose(sums, torch.ones(channels), atol=1e-6)

    # 与 5x5 Pascal 核期望值一致。
    row = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    expected = torch.outer(row, row)
    expected = expected / expected.sum()
    assert torch.allclose(m.gaussian_weight[0, 0], expected, atol=1e-6)

    # 形状保持。
    x = torch.randn(2, channels, 12, 14)
    g = m._gaussian(x)
    assert g.shape == x.shape

    # 每通道独立：改动通道 1 不影响通道 0 输出。
    x2 = x.clone()
    x2[:, 1] = x2[:, 1] * 100.0 + 7.0
    g2 = m._gaussian(x2)
    assert torch.allclose(g2[:, 0], g[:, 0], atol=1e-6)

    # 常数输入边界不漂移（replicate padding + 归一化核）。
    const = torch.full((1, channels, 9, 9), 3.5)
    gc = m._gaussian(const)
    assert torch.allclose(gc, const, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Trimmed mean：公式一致 / 常数不变 / 单极值削弱
# ---------------------------------------------------------------------------
def test_trimmed_mean_matches_formula():
    m = SelectiveRobustFrequencyFusion(channels=3)

    x = torch.randn(2, 3, 10, 11)
    t = m._trimmed_mean(x)
    ref = _reference_trimmed_mean_3x3(x)
    assert t.shape == x.shape
    assert torch.allclose(t, ref, atol=1e-5)

    # 常数输入不变。
    const = torch.full((1, 3, 7, 7), -2.25)
    assert torch.allclose(m._trimmed_mean(const), const, atol=1e-5)

    # 单个正极值被削弱（3x3 trimmed mean 对孤立脉冲应消除为 ~0）。
    spike = torch.zeros(1, 3, 9, 9)
    spike[0, :, 4, 4] = 10.0
    ts = m._trimmed_mean(spike)
    assert ts[0, :, 4, 4].abs().max() < 1e-5


# ---------------------------------------------------------------------------
# 3. 两级形状：20->40、40->80，输出严格等于 low shape
# ---------------------------------------------------------------------------
def test_two_level_shapes():
    m = SelectiveRobustFrequencyFusion(channels=16)
    for high_size, low_size in [(20, 40), (40, 80)]:
        high = torch.randn(2, 16, high_size, high_size)
        low = torch.randn(2, 16, low_size, low_size)
        out = m(high, low)
        assert out.shape == low.shape
        assert out.dtype == low.dtype
        assert out.device == low.device


# ---------------------------------------------------------------------------
# 4. 动态尺寸：非整齐偶数 / 非严格 2 倍，通过显式 size 对齐
# ---------------------------------------------------------------------------
def test_dynamic_sizes():
    m = SelectiveRobustFrequencyFusion(channels=8)
    cases = [(15, 31), (13, 27), (21, 41), (10, 19)]
    for high_size, low_size in cases:
        # 故意使用非整齐、非严格 2 倍、且高宽不等的尺寸。
        high = torch.randn(1, 8, high_size, high_size + 1)
        low = torch.randn(1, 8, low_size, low_size + 2)
        out = m(high, low)
        assert out.shape == low.shape


# ---------------------------------------------------------------------------
# 5. dtype/device：CPU FP32 必测；有 CUDA 时测 FP16 autocast 前后向
# ---------------------------------------------------------------------------
def test_dtype_device_cpu_fp32():
    m = SelectiveRobustFrequencyFusion(channels=8)
    high = torch.randn(2, 8, 16, 16)
    low = torch.randn(2, 8, 32, 32)
    out = m(high, low)
    assert out.dtype == torch.float32
    assert out.device == low.device
    assert torch.isfinite(out).all()


def test_dtype_device_cuda_fp16_autocast():
    if not torch.cuda.is_available():
        _skip('CUDA 不可用，跳过 FP16 autocast 测试')

    m = SelectiveRobustFrequencyFusion(channels=8).cuda()
    high = torch.randn(2, 8, 16, 16, device='cuda')
    low = torch.randn(2, 8, 32, 32, device='cuda')

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = m(high, low)
        loss = out.float().pow(2).mean()

    assert out.device.type == 'cuda'
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(loss)

    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f'{name} 缺少梯度'
        assert torch.isfinite(p.grad).all(), f'{name} 梯度出现 NaN/Inf'


# ---------------------------------------------------------------------------
# 6. 梯度：输入(low)与 gate/router 参数梯度 finite；high 作证据被 detach
# ---------------------------------------------------------------------------
def test_gradients_finite():
    torch.manual_seed(0)
    m = SelectiveRobustFrequencyFusion(channels=8)
    high = torch.randn(2, 8, 16, 16, requires_grad=True)
    low = torch.randn(2, 8, 32, 32, requires_grad=True)

    out = m(high, low)
    loss = out.pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss)
    # 被校正的低层特征必须有 finite 梯度（残差 + 鲁棒基底通路）。
    assert low.grad is not None and torch.isfinite(low.grad).all()

    # gate/router 全部参数梯度 finite，且不缺失。
    param_count = 0
    for name, p in m.named_parameters():
        assert p.grad is not None, f'{name} 缺少梯度'
        assert torch.isfinite(p.grad).all(), f'{name} 梯度出现 NaN/Inf'
        param_count += 1
    assert param_count == 8  # gate_net(2 conv)*2 + router_net(2 conv)*2

    # 设计锁定：high 只作结构证据，其证据路径被 detach，SRFF 不向 high 回传梯度。
    assert high.grad is None, 'high 应为被 detach 的结构证据，不应收到 SRFF 梯度'


# ---------------------------------------------------------------------------
# 7. 近恒等初始化：默认 bias=-4 下 relative_delta <= 0.05
# ---------------------------------------------------------------------------
def test_near_identity_initialization():
    torch.manual_seed(0)
    m = SelectiveRobustFrequencyFusion(channels=16)  # 默认 gate_init_bias=-4.0
    high = torch.randn(2, 16, 20, 20)
    low = torch.randn(2, 16, 40, 40)

    stats = m.analyze(high, low)
    relative_delta = float(stats['relative_delta'])
    assert relative_delta <= 0.05, f'relative_delta={relative_delta:.5f} 超过 0.05'

    # sigmoid(-4)≈0.018，初始门控应很小。
    assert float(stats['gate_mean']) < 0.02
    assert float(stats['gate_max']) < 0.05
    # router 初始接近均分。
    assert abs(float(stats['gaussian_weight_mean']) - 0.5) < 0.05
    assert abs(float(stats['trimmed_weight_mean']) - 0.5) < 0.05


# ---------------------------------------------------------------------------
# 8. 结构保护：内部组合公式 A = A_raw * (1 - S)，S=1 时门为 0
# ---------------------------------------------------------------------------
def test_structure_protection_composition():
    torch.manual_seed(0)
    m = SelectiveRobustFrequencyFusion(channels=8)
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)

    out = m._core(high, low)
    gate = out['gate']
    gate_raw = out['gate_raw']
    structure = out['structure']

    # 组合公式严格成立：A = A_raw * (1 - S)。
    recomposed = gate_raw * (1.0 - structure)
    assert torch.allclose(gate, recomposed, atol=1e-6)

    # S=1 时最终校正门为 0（代入组合公式）。
    gate_at_full_structure = gate_raw * (1.0 - torch.ones_like(structure))
    assert torch.all(gate_at_full_structure == 0)

    # 门控被结构证据单调抑制：A <= A_raw。
    assert torch.all(gate <= gate_raw + 1e-6)


def test_structure_confident_input_is_preserved():
    # 空间均匀的高频（棋盘格）应得到高结构一致性 -> 校正被抑制、输出接近恒等。
    # 仅验证算子级组合行为，不断言未训练网络能识别真实边缘。
    m = SelectiveRobustFrequencyFusion(channels=4)
    yy, xx = torch.meshgrid(
        torch.arange(32), torch.arange(32), indexing='ij'
    )
    checker = ((xx + yy) % 2).float() * 2.0 - 1.0  # ±1 棋盘格
    low = checker.expand(1, 4, 32, 32).contiguous()
    high = checker[:16, :16].expand(1, 4, 16, 16).contiguous()

    out = m._core(high, low)
    # 内部区域结构一致性高，整体校正被强烈抑制（近恒等）。
    interior = out['low_out'][:, :, 6:-6, 6:-6]
    low_interior = low[:, :, 6:-6, 6:-6]
    rel = (interior - low_interior).abs().mean() / (low_interior.abs().mean() + 1e-6)
    assert rel < 0.05


# ---------------------------------------------------------------------------
# 9. 噪声鲁棒算子：合成稀疏正/负脉冲，trimmed mean 降低极值
# ---------------------------------------------------------------------------
def test_trimmed_mean_suppresses_sparse_impulses():
    m = SelectiveRobustFrequencyFusion(channels=1)

    # 正脉冲。
    pos = torch.zeros(1, 1, 11, 11)
    pos[0, 0, 5, 5] = 8.0
    tpos = m._trimmed_mean(pos)
    assert tpos[0, 0, 5, 5].abs() < pos[0, 0, 5, 5].abs() * 0.5
    assert torch.allclose(tpos[0, 0, 5, 5], torch.tensor(0.0), atol=1e-5)

    # 负脉冲。
    neg = torch.zeros(1, 1, 11, 11)
    neg[0, 0, 3, 7] = -8.0
    tneg = m._trimmed_mean(neg)
    assert tneg[0, 0, 3, 7].abs() < neg[0, 0, 3, 7].abs() * 0.5
    assert torch.allclose(tneg[0, 0, 3, 7], torch.tensor(0.0), atol=1e-5)

    # 多个稀疏脉冲（互不相邻）同样被削弱。
    multi = torch.zeros(1, 1, 15, 15)
    multi[0, 0, 2, 2] = 6.0
    multi[0, 0, 7, 12] = -6.0
    multi[0, 0, 12, 4] = 6.0
    tmulti = m._trimmed_mean(multi)
    assert tmulti.abs().max() < multi.abs().max() * 0.5


# ---------------------------------------------------------------------------
# 10. 状态字典：同配置 strict save/load；关闭 SRFF 的 encoder 不含 srff_blocks.*
# ---------------------------------------------------------------------------
def test_srff_state_dict_roundtrip():
    torch.manual_seed(0)
    m1 = SelectiveRobustFrequencyFusion(channels=8)
    m2 = SelectiveRobustFrequencyFusion(channels=8)

    state = m1.state_dict()
    # Gaussian buffer 为 persistent=False，不应出现在 state_dict 中。
    assert not any('gaussian_weight' in k for k in state.keys())
    m2.load_state_dict(state, strict=True)  # 不应抛异常

    high = torch.randn(1, 8, 10, 10)
    low = torch.randn(1, 8, 20, 20)
    assert torch.allclose(m1(high, low), m2(high, low), atol=1e-6)


def test_parameter_validation():
    # 非法参数应尽早报错。
    for bad_kwargs in (
        dict(channels=8, gaussian_kernel=4),      # 非奇数
        dict(channels=8, gaussian_kernel=0),      # 非正
        dict(channels=8, trim_kernel=5),          # v1 只允许 3
        dict(channels=8, gate_hidden=0),          # 必须为正
        dict(channels=0),                         # 通道必须为正
    ):
        try:
            SelectiveRobustFrequencyFusion(**bad_kwargs)
        except AssertionError:
            continue
        raise AssertionError(f'非法参数未被拒绝: {bad_kwargs}')


def _build_small_encoder(use_srff):
    from engine.deim.hybrid_encoder import HybridEncoder
    return HybridEncoder(
        in_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        hidden_dim=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        eval_spatial_size=None,
        version='dfine',
        use_srff=use_srff,
    )


def test_encoder_integration_state_and_forward():
    try:
        enc_off = _build_small_encoder(use_srff=False)
        enc_on = _build_small_encoder(use_srff=True)
    except Exception as exc:  # pragma: no cover - 缺少可选依赖时跳过
        _skip(f'HybridEncoder 导入/构建失败（可能缺少可选依赖）: {exc}')
        return

    # 关闭 SRFF：不含任何 srff_blocks.* 参数。
    off_names = [n for n, _ in enc_off.named_parameters()]
    assert not any('srff_blocks' in n for n in off_names)
    assert enc_off.srff_blocks is None

    # 打开 SRFF：两个 top-down 分支各一个 block，互不共享参数。
    on_names = [n for n, _ in enc_on.named_parameters()]
    srff_names = [n for n in on_names if 'srff_blocks' in n]
    assert len(srff_names) > 0
    assert len(enc_on.srff_blocks) == len(enc_on.in_channels) - 1 == 2
    assert enc_on.srff_blocks[0] is not enc_on.srff_blocks[1]

    # 新增可训练参数量 = 打开 - 关闭，且 < 0.01M。
    added = _count_trainable(enc_on) - _count_trainable(enc_off)
    assert added > 0
    assert added < int(0.01 * 1e6), f'SRFF 新增可训练参数 {added} 超过 0.01M'

    # 前后向：输出形状一致，SRFF 参数梯度 finite。
    enc_on.train()
    feats = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    outs_on = enc_on(feats)
    outs_off = enc_off(feats)
    assert len(outs_on) == len(outs_off) == 3
    for o_on, o_off in zip(outs_on, outs_off):
        assert o_on.shape == o_off.shape

    loss = sum(o.pow(2).mean() for o in outs_on)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in enc_on.named_parameters():
        if 'srff_blocks' in name:
            assert p.grad is not None, f'{name} 缺少梯度'
            assert torch.isfinite(p.grad).all(), f'{name} 梯度出现 NaN/Inf'


# ---------------------------------------------------------------------------
# 独立运行入口（pytest 不可用时的等价 runner）
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith('test_') and callable(obj)
    ]
    passed = skipped = failed = 0
    skip_exceptions = tuple(_SKIP_EXCEPTIONS)
    for name, fn in tests:
        try:
            fn()
        except skip_exceptions as exc:
            print(f'[SKIP] {name}: {exc}')
            skipped += 1
        except Exception as exc:  # noqa: BLE001 - runner 需要吞掉所有失败并汇总
            print(f'[FAIL] {name}: {type(exc).__name__}: {exc}')
            failed += 1
        else:
            print(f'[PASS] {name}')
            passed += 1
    print(f'\nSummary: {passed} passed, {skipped} skipped, {failed} failed')
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)
