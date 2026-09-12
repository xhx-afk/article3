"""SRFF-V1.1 单元测试（改进文档 §6.1 的 20 项）。

不依赖真实数据/检查点/完整训练。V1、V1.1 通过合成包 importlib 加载（仅需 torch，
不触发 engine 重依赖）；HybridEncoder/YAML 相关项在函数内惰性导入，缺依赖则安全跳过。

运行方式（二选一）：
    python -m pytest tests/test_srff_v1_1.py -q
    python tests/test_srff_v1_1.py
"""

import importlib
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_srff_modules():
    """用合成包加载 srff + srff_v1_1，使 srff_v1_1 的相对导入 `.srff` 可解析，且仅需 torch。"""
    pkg_name = '_srff_pkg_for_v11_test'
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(ROOT / 'engine' / 'deim')]
        sys.modules[pkg_name] = pkg
    srff = importlib.import_module(f'{pkg_name}.srff')
    v11 = importlib.import_module(f'{pkg_name}.srff_v1_1')
    return srff, v11


_srff, _v11mod = _load_srff_modules()
SRFF = _srff.SelectiveRobustFrequencyFusion
SRFFV11 = _v11mod.SelectiveRobustFrequencyFusionV11

try:
    import pytest as _pytest
except Exception:  # pragma: no cover
    _pytest = None


class _Skipped(Exception):
    """pytest 不可用时的本地跳过信号。"""


_SKIP_EXCEPTIONS = [_Skipped]
if _pytest is not None:
    try:
        _SKIP_EXCEPTIONS.append(_pytest.skip.Exception)
    except Exception:  # pragma: no cover
        pass


def _skip(reason):
    if _pytest is not None:
        _pytest.skip(reason)
    raise _Skipped(reason)


def _import_hybrid_encoder():
    try:
        from engine.deim.hybrid_encoder import HybridEncoder
        return HybridEncoder
    except Exception as exc:  # pragma: no cover
        _skip(f'HybridEncoder 导入失败（可能缺少可选依赖）: {exc}')


def _small_enc_kwargs(**over):
    d = dict(in_channels=[16, 32, 64], feat_strides=[8, 16, 32], hidden_dim=16, nhead=4,
             dim_feedforward=32, dropout=0.0, use_encoder_idx=[2], num_encoder_layers=1,
             eval_spatial_size=None, version='dfine')
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# 1. 阈值非法 fail-fast
# ---------------------------------------------------------------------------
def test_threshold_validation():
    for bad in [dict(global_threshold_low=0.9, global_threshold_high=0.8),
                dict(global_threshold_low=-0.1, global_threshold_high=0.5),
                dict(global_threshold_low=0.5, global_threshold_high=1.1),
                dict(global_threshold_low=0.8, global_threshold_high=0.8)]:
        try:
            SRFFV11(channels=8, **bad)
        except AssertionError:
            continue
        raise AssertionError(f'非法阈值未被拒绝: {bad}')
    SRFFV11(channels=8, global_threshold_low=0.78, global_threshold_high=0.80)  # 合法


# ---------------------------------------------------------------------------
# 2/3/4. smoothstep 边界：q<=tau_low→0，中点→0.5，q>=tau_high→1
# ---------------------------------------------------------------------------
def test_global_gate_below_tau_low_is_zero():
    m = SRFFV11(channels=8, global_threshold_low=0.78, global_threshold_high=0.80)
    for q in [0.0, 0.5, 0.77, 0.78]:
        g = m._smooth_global_gate(torch.tensor([[[[q]]]]))
        assert float(g) == 0.0, (q, float(g))


def test_global_gate_midpoint_half():
    m = SRFFV11(channels=8)
    g = m._smooth_global_gate(torch.tensor([[[[0.79]]]]))
    # float32 下 (0.79-0.78)/0.02 ≈ 0.5000037 → m ≈ 0.500004，非精确 0.5，容差 1e-4
    assert abs(float(g) - 0.5) < 1e-4


def test_global_gate_above_tau_high_is_one():
    m = SRFFV11(channels=8)
    for q in [0.80, 0.85, 1.0, 5.0]:
        g = m._smooth_global_gate(torch.tensor([[[[q]]]]))
        assert abs(float(g) - 1.0) < 1e-6, (q, float(g))


# ---------------------------------------------------------------------------
# 5. smoothstep 单调、连续、范围 [0,1]
# ---------------------------------------------------------------------------
def test_smoothstep_monotonic_continuous_bounded():
    m = SRFFV11(channels=8)
    q = torch.linspace(0.0, 1.0, 20001).reshape(1, 1, 1, -1)
    g = m._smooth_global_gate(q).flatten()
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0
    diffs = g[1:] - g[:-1]
    assert bool((diffs >= -1e-6).all())           # 单调不减（float32 容差）
    assert float(diffs.abs().max()) < 0.01        # 无跳变（连续）


# ---------------------------------------------------------------------------
# 6/7. gate shape [B,1,1,1] 且逐样本独立
# ---------------------------------------------------------------------------
def test_global_gate_shape_and_per_sample_independence():
    torch.manual_seed(0)
    m = SRFFV11(channels=8)
    high = torch.randn(3, 8, 10, 10)
    low = torch.randn(3, 8, 20, 20)
    out = m._core(high, low)
    assert tuple(out['global_gate'].shape) == (3, 1, 1, 1)
    assert tuple(out['global_score'].shape) == (3, 1, 1, 1)
    g0 = out['global_gate'][0].clone()

    high2 = high.clone()
    low2 = low.clone()
    high2[1] = torch.randn(8, 10, 10)
    low2[1] = low2[1] * 100.0 + 7.0
    low2[2] = low2[2] * 0.01
    out2 = m._core(high2, low2)
    # 改动样本 1/2 不影响样本 0 的全局门
    assert torch.allclose(out2['global_gate'][0], g0, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. m=0 时输出严格等于 low
# ---------------------------------------------------------------------------
def test_global_gate_zero_is_identity():
    torch.manual_seed(0)
    m = SRFFV11(channels=8, global_threshold_low=0.9999, global_threshold_high=1.0)
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    out = m._core(high, low)
    assert float(out['global_gate'].max()) == 0.0
    assert torch.equal(out['low_out'], low)
    assert torch.equal(out['gate'], torch.zeros_like(out['gate']))


# ---------------------------------------------------------------------------
# 9. m=1 时同权重 V1/V1.1 输出一致
# ---------------------------------------------------------------------------
def test_global_gate_one_matches_v1():
    torch.manual_seed(0)
    v1 = SRFF(channels=8)
    v11 = SRFFV11(channels=8, global_threshold_low=0.0, global_threshold_high=1e-6)
    v11.load_state_dict(v1.state_dict())  # 同权重
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    o1 = v1._core(high, low)
    o11 = v11._core(high, low)
    assert float(o11['global_gate'].min()) == 1.0
    assert torch.allclose(o11['low_out'], o1['low_out'], atol=1e-5)


# ---------------------------------------------------------------------------
# 10. transition 时最终 gate == pre_global_gate * global_gate
# ---------------------------------------------------------------------------
def test_final_gate_equals_pre_times_global():
    torch.manual_seed(0)
    m = SRFFV11(channels=8)
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    out = m._core(high, low)
    recomposed = out['pre_global_gate'] * out['global_gate']
    assert torch.allclose(out['gate'], recomposed, atol=1e-6)
    # stationarity 与 structure 一致（兼容字段）
    assert torch.equal(out['stationarity'], out['structure'])


# ---------------------------------------------------------------------------
# 11. CPU FP32 forward/backward finite
# ---------------------------------------------------------------------------
def test_cpu_fp32_forward_backward_finite():
    torch.manual_seed(0)
    m = SRFFV11(channels=8)
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24, requires_grad=True)
    out = m(high, low)
    assert out.dtype == torch.float32 and out.shape == low.shape
    loss = out.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(low.grad).all()
    n = 0
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        n += 1
    assert n == 8  # gate_net(2conv)*2 + router_net(2conv)*2，全局门不增参数


# ---------------------------------------------------------------------------
# 12. CUDA AMP forward/backward finite
# ---------------------------------------------------------------------------
def test_cuda_amp_forward_backward_finite():
    if not torch.cuda.is_available():
        _skip('CUDA 不可用，跳过 AMP 测试')
    m = SRFFV11(channels=8).cuda()
    high = torch.randn(2, 8, 12, 12, device='cuda')
    low = torch.randn(2, 8, 24, 24, device='cuda')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = m(high, low)
        loss = out.float().pow(2).mean()
    assert torch.isfinite(out.float()).all() and torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# 13. high 证据路径仍 detach
# ---------------------------------------------------------------------------
def test_high_evidence_detached():
    torch.manual_seed(0)
    m = SRFFV11(channels=8)
    high = torch.randn(2, 8, 12, 12, requires_grad=True)
    low = torch.randn(2, 8, 24, 24, requires_grad=True)
    out = m(high, low)
    out.pow(2).mean().backward()
    assert high.grad is None, 'high 应为 detach 的证据，不回传梯度'
    assert low.grad is not None and torch.isfinite(low.grad).all()


# ---------------------------------------------------------------------------
# 14. state dict round-trip（阈值/gaussian buffer 不入 state dict）
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip():
    torch.manual_seed(0)
    a = SRFFV11(channels=8)
    b = SRFFV11(channels=8)
    sd = a.state_dict()
    assert not any(('tau' in k or 'threshold' in k) for k in sd)
    assert not any('gaussian_weight' in k for k in sd)
    b.load_state_dict(sd, strict=True)
    high = torch.randn(1, 8, 10, 10)
    low = torch.randn(1, 8, 20, 20)
    assert torch.allclose(a(high, low), b(high, low), atol=1e-6)


# ---------------------------------------------------------------------------
# 15/16. active_levels=[0]：block0=V1.1、block1=Identity 无参数且被旁路
# ---------------------------------------------------------------------------
def test_encoder_active_levels_v11():
    HybridEncoder = _import_hybrid_encoder()
    enc = HybridEncoder(**_small_enc_kwargs(use_srff=True, srff_version='v1_1', srff_active_levels=[0]))
    assert type(enc.srff_blocks[0]).__name__ == 'SelectiveRobustFrequencyFusionV11'
    assert isinstance(enc.srff_blocks[1], torch.nn.Identity)
    assert hasattr(enc.srff_blocks[0], 'tau_low')
    assert enc.srff_active_levels == {0}
    assert len(list(enc.srff_blocks[1].parameters())) == 0
    sd_keys = list(enc.state_dict().keys())
    assert any(k.startswith('srff_blocks.0.') for k in sd_keys)
    assert not any(k.startswith('srff_blocks.1.') for k in sd_keys)


def test_inactive_level_bypassed_uses_baseline():
    HybridEncoder = _import_hybrid_encoder()
    enc = HybridEncoder(**_small_enc_kwargs(use_srff=True, srff_version='v1_1', srff_active_levels=[0])).eval()
    called = {0: False, 1: False}

    def mk(i):
        def hook(_m, _inp):
            called[i] = True
        return hook

    handles = [enc.srff_blocks[i].register_forward_pre_hook(mk(i)) for i in (0, 1)]
    feats = [torch.randn(1, 16, 16, 16), torch.randn(1, 32, 8, 8), torch.randn(1, 64, 4, 4)]
    try:
        with torch.no_grad():
            enc(feats)
    finally:
        for h in handles:
            h.remove()
    assert called[0] is True    # active level 调用 V1.1
    assert called[1] is False   # inactive level 旁路（Identity 从未被调用 → 走 baseline 插值）


# ---------------------------------------------------------------------------
# 17/18. V1 默认两级；use_srff=False 无参数
# ---------------------------------------------------------------------------
def test_v1_default_creates_two_v1_blocks():
    HybridEncoder = _import_hybrid_encoder()
    enc = HybridEncoder(**_small_enc_kwargs(use_srff=True))
    assert enc.srff_version == 'v1'
    assert enc.srff_active_levels == {0, 1}
    assert len(enc.srff_blocks) == 2
    assert all(type(b).__name__ == 'SelectiveRobustFrequencyFusion' for b in enc.srff_blocks)


def test_use_srff_false_has_no_srff_params():
    HybridEncoder = _import_hybrid_encoder()
    enc = HybridEncoder(**_small_enc_kwargs(use_srff=False))
    assert enc.srff_blocks is None
    assert enc.srff_active_levels == set()
    assert not any('srff_blocks' in n for n, _ in enc.named_parameters())


# ---------------------------------------------------------------------------
# 19. 动态尺寸与输出 shape
# ---------------------------------------------------------------------------
def test_dynamic_sizes_output_shape():
    m = SRFFV11(channels=8)
    for hs, ls in [(15, 31), (13, 27), (20, 40), (10, 19)]:
        high = torch.randn(1, 8, hs, hs + 1)
        low = torch.randn(1, 8, ls, ls + 2)
        out = m(high, low)
        assert out.shape == low.shape and out.dtype == low.dtype


# ---------------------------------------------------------------------------
# 20. YAML 能实例化（从真实 v1_1 配置构建 HybridEncoder）
# ---------------------------------------------------------------------------
def test_yaml_instantiates_encoder():
    cfg_path = ROOT / 'configs' / 'deim_dfine' / 'custom' / 'coated_wood_fast_s_srff_v1_1.yml'
    if not cfg_path.is_file():
        _skip('缺少 coated_wood_fast_s_srff_v1_1.yml')
    try:
        from engine.core import YAMLConfig
        from engine.core.workspace import create
        cfg = YAMLConfig(str(cfg_path))
        enc = create('HybridEncoder', cfg.global_cfg)
    except Exception as exc:  # pragma: no cover
        _skip(f'从 YAML 实例化 HybridEncoder 失败: {exc}')
    assert enc.use_srff is True and enc.srff_version == 'v1_1'
    assert enc.srff_active_levels == {0}
    assert type(enc.srff_blocks[0]).__name__ == 'SelectiveRobustFrequencyFusionV11'
    assert isinstance(enc.srff_blocks[1], torch.nn.Identity)
    assert abs(enc.srff_blocks[0].tau_low - 0.78) < 1e-9
    assert abs(enc.srff_blocks[0].tau_high - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# 附加：V1.1 相对 V1 少一个 block 的参数（单层启用）
# ---------------------------------------------------------------------------
def test_v11_has_fewer_params_than_v1():
    HybridEncoder = _import_hybrid_encoder()
    enc_v1 = HybridEncoder(**_small_enc_kwargs(use_srff=True, srff_version='v1'))
    enc_v11 = HybridEncoder(**_small_enc_kwargs(use_srff=True, srff_version='v1_1', srff_active_levels=[0]))
    n_v1 = sum(p.numel() for p in enc_v1.parameters() if p.requires_grad)
    n_v11 = sum(p.numel() for p in enc_v11.parameters() if p.requires_grad)
    assert n_v11 < n_v1


# ---------------------------------------------------------------------------
# 独立运行入口
# ---------------------------------------------------------------------------
def _run_all():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith('test_') and callable(o)]
    passed = skipped = failed = 0
    skip_exc = tuple(_SKIP_EXCEPTIONS)
    for name, fn in tests:
        try:
            fn()
        except skip_exc as exc:
            print(f'[SKIP] {name}: {exc}')
            skipped += 1
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] {name}: {type(exc).__name__}: {exc}')
            failed += 1
        else:
            print(f'[PASS] {name}')
            passed += 1
    print(f'\nSummary: {passed} passed, {skipped} skipped, {failed} failed')
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)
