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
# 三态门控因果干预（实现文档 §8 items 1-9）
# ---------------------------------------------------------------------------
def test_gate_mode_validation():
    # item 6：非法 mode 明确报错；合法三态可构造
    for ok in ('auto', 'force_off', 'force_on'):
        SRFFV11(channels=8, global_gate_mode=ok)
    for bad in ['off', 'on', 'AUTO', '', 'force', None]:
        try:
            SRFFV11(channels=8, global_gate_mode=bad)
        except AssertionError:
            continue
        raise AssertionError(f'非法 mode 未被拒绝: {bad!r}')


def test_three_modes_same_global_score_and_state_dict():
    # item 4/5/9：三态 global_score/stationarity/pre_global_gate 相同；state dict 完全一致且不含 mode/tau
    torch.manual_seed(0)
    base = SRFFV11(channels=8)
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    blocks, outs = {}, {}
    for mode in ('auto', 'force_off', 'force_on'):
        b = SRFFV11(channels=8, global_gate_mode=mode)
        b.load_state_dict(base.state_dict())
        blocks[mode] = b
        outs[mode] = b._core(high, low)
    for mode in ('force_off', 'force_on'):
        assert torch.equal(outs[mode]['global_score'], outs['auto']['global_score'])
        assert torch.equal(outs[mode]['stationarity'], outs['auto']['stationarity'])
        assert torch.allclose(outs[mode]['pre_global_gate'], outs['auto']['pre_global_gate'], atol=1e-6)
    ref_keys = set(blocks['auto'].state_dict().keys())
    for mode in ('force_off', 'force_on'):
        assert set(blocks[mode].state_dict().keys()) == ref_keys
    nparam = {m: sum(1 for _ in blocks[m].parameters()) for m in blocks}
    nbuf = {m: sum(1 for _ in blocks[m].buffers()) for m in blocks}
    assert len(set(nparam.values())) == 1 and len(set(nbuf.values())) == 1
    assert not any(('global_gate_mode' in k or 'tau' in k or 'threshold' in k) for k in ref_keys)
    # item 9：与 V1 同构，旧 V1.1 checkpoint key 集不因本补丁改变
    assert ref_keys == set(SRFF(channels=8).state_dict().keys())


def test_force_off_strict_identity():
    # item 2：force_off 下 low_out 严格等于 low（同一张量），最终 gate 全 0
    torch.manual_seed(0)
    b = SRFFV11(channels=8, global_gate_mode='force_off')
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    out = b._core(high, low)
    assert out['low_out'] is low
    assert torch.equal(out['low_out'], low)
    assert float(out['gate'].abs().max()) == 0.0
    assert float(out['global_gate'].abs().max()) == 0.0
    assert torch.equal(b(high, low), low)


def test_force_on_strict_v1():
    # item 3：force_on 下 low_out 严格等于父类 V1 输出（复用非重算），gate == pre_global_gate
    torch.manual_seed(0)
    v1 = SRFF(channels=8)
    b = SRFFV11(channels=8, global_gate_mode='force_on')
    b.load_state_dict(v1.state_dict())
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    out = b._core(high, low)
    ov1 = v1._core(high, low)
    assert torch.equal(out['low_out'], ov1['low_out'])
    assert torch.equal(out['gate'], out['pre_global_gate'])
    assert float(out['global_gate'].min()) == 1.0
    assert torch.equal(b(high, low), ov1['low_out'])


def test_auto_matches_original_formula():
    # item 1：auto 与修改前公式一致（gate=pre*m，low_out=low+m*(v1_low-low)）
    torch.manual_seed(0)
    b = SRFFV11(channels=8, global_gate_mode='auto')
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    out = b._core(high, low)
    assert torch.allclose(out['gate'], out['pre_global_gate'] * out['global_gate'], atol=1e-6)
    v1 = SRFF(channels=8)
    v1.load_state_dict(b.state_dict())
    v1_low = v1._core(high, low)['low_out']
    recon = low + out['global_gate'] * (v1_low - low)
    assert torch.allclose(out['low_out'], recon, atol=1e-5)


def test_encoder_passes_gate_mode():
    # item 7/8：HybridEncoder 向 active block 传递 mode；inactive P4→P3 仍为 Identity/baseline
    HybridEncoder = _import_hybrid_encoder()
    for mode in ('auto', 'force_off', 'force_on'):
        enc = HybridEncoder(**_small_enc_kwargs(use_srff=True, srff_version='v1_1',
                                                srff_active_levels=[0], srff_global_gate_mode=mode))
        assert enc.srff_blocks[0].global_gate_mode == mode
        assert isinstance(enc.srff_blocks[1], torch.nn.Identity)
    try:
        HybridEncoder(**_small_enc_kwargs(use_srff=True, srff_version='v1_1',
                                          srff_active_levels=[0], srff_global_gate_mode='bad'))
    except AssertionError:
        pass
    else:
        raise AssertionError('encoder 未拒绝非法 srff_global_gate_mode')


# ---------------------------------------------------------------------------
# 局部门 / 双专家容量诊断 override（实现文档 §8 items 1-10）
# ---------------------------------------------------------------------------
def _hl(c=8):
    return torch.randn(2, c, 12, 12), torch.randn(2, c, 24, 24)


def test_diag_default_preserves_auto():
    # item 1：默认 diagnostic 字段（learned/0.02/learned）不改变旧 V1.1 auto 输出
    torch.manual_seed(0)
    b = SRFFV11(channels=8)
    high, low = _hl()
    out = b._core(high, low)
    assert out['diagnostic_override_active'] is False
    v1 = SRFF(channels=8)
    v1.load_state_dict(b.state_dict())
    ov1 = v1._core(high, low)
    m = out['global_gate']
    assert torch.allclose(out['low_out'], low + m * (ov1['low_out'] - low), atol=1e-5)
    assert torch.allclose(out['gate'], out['pre_global_gate'] * m, atol=1e-6)
    assert torch.equal(out['learned_gaussian_weight'], ov1['gaussian_weight'])
    assert torch.equal(out['learned_trimmed_weight'], ov1['trimmed_weight'])
    assert torch.equal(out['learned_local_gate'], ov1['gate'])


def test_diag_training_override_raises():
    # item 2：训练模式激活 override 必然报错（含固定字样）；eval 不报错
    high, low = _hl()
    for kwargs in ({'diagnostic_local_gate_mode': 'constant'},
                   {'diagnostic_router_mode': 'gaussian_only'}):
        b = SRFFV11(channels=8, global_gate_mode='force_on', **kwargs)
        b.train()
        try:
            b(high, low)
        except RuntimeError as e:
            assert 'diagnostic override is evaluation-only' in str(e)
        else:
            raise AssertionError(f'训练期 override 未报错: {kwargs}')
    b = SRFFV11(channels=8, global_gate_mode='force_on', diagnostic_local_gate_mode='constant')
    b.eval()
    b(high, low)


def test_diag_constant_local_gate_exact():
    # item 3：constant=0.02/0.05/0.10 产生逐元素常数 local gate（final gate 也等于 α）
    for alpha in (0.02, 0.05, 0.10):
        b = SRFFV11(channels=8, global_gate_mode='force_on',
                    diagnostic_local_gate_mode='constant', diagnostic_local_gate_value=alpha)
        b.eval()
        out = b._core(*_hl())
        assert out['diagnostic_override_active'] is True
        assert torch.all(out['local_gate_used'] == alpha)
        assert torch.all(out['gate'] == alpha)   # global_gate(force_on)=1 → final_gate=α


def test_diag_gaussian_only_weights():
    # item 4：gaussian-only 实际权重严格 1/0，learned 原值保留
    torch.manual_seed(0)
    b = SRFFV11(channels=8, global_gate_mode='force_on', diagnostic_router_mode='gaussian_only',
                diagnostic_local_gate_mode='constant', diagnostic_local_gate_value=0.05)
    b.eval()
    high, low = _hl()
    out = b._core(high, low)
    assert torch.all(out['gaussian_weight'] == 1.0) and torch.all(out['trimmed_weight'] == 0.0)
    v1 = SRFF(channels=8)
    v1.load_state_dict(b.state_dict())
    ov1 = v1._core(high, low)
    assert torch.equal(out['learned_gaussian_weight'], ov1['gaussian_weight'])
    assert not torch.all(out['learned_gaussian_weight'] == 1.0)


def test_diag_trimmed_only_weights():
    # item 5：trimmed-only 实际权重严格 0/1
    b = SRFFV11(channels=8, global_gate_mode='force_on', diagnostic_router_mode='trimmed_only',
                diagnostic_local_gate_mode='constant', diagnostic_local_gate_value=0.05)
    b.eval()
    out = b._core(*_hl())
    assert torch.all(out['gaussian_weight'] == 0.0) and torch.all(out['trimmed_weight'] == 1.0)


def test_diag_learned_router_keeps_weights():
    # item 6：learned router 保持原权重（仅 local gate 被 override）
    b = SRFFV11(channels=8, global_gate_mode='force_on', diagnostic_local_gate_mode='constant',
                diagnostic_local_gate_value=0.05, diagnostic_router_mode='learned')
    b.eval()
    out = b._core(*_hl())
    assert out['diagnostic_override_active'] is True
    assert torch.equal(out['gaussian_weight'], out['learned_gaussian_weight'])
    assert torch.equal(out['trimmed_weight'], out['learned_trimmed_weight'])


def test_diag_override_output_formula():
    # item 7：override 输出严格满足 low + α*(f_rob_used - low)（gaussian_only → f_rob_used=g_low）
    torch.manual_seed(0)
    alpha = 0.05
    b = SRFFV11(channels=8, global_gate_mode='force_on', diagnostic_local_gate_mode='constant',
                diagnostic_local_gate_value=alpha, diagnostic_router_mode='gaussian_only')
    b.eval()
    high, low = _hl()
    out = b._core(high, low)
    v1 = SRFF(channels=8)
    v1.load_state_dict(b.state_dict())
    g_low = v1._core(high, low)['gaussian_low']
    expect = low + alpha * (g_low - low)   # global_gate=1
    assert torch.allclose(out['low_out'], expect, atol=1e-5)


def test_diag_override_requires_force_on():
    # item 8：global mode 非 force_on 时激活 override 被拒绝
    for gmode in ('auto', 'force_off'):
        b = SRFFV11(channels=8, global_gate_mode=gmode,
                    diagnostic_local_gate_mode='constant', diagnostic_local_gate_value=0.05)
        b.eval()
        try:
            b._core(*_hl())
        except RuntimeError as e:
            assert 'force_on' in str(e)
        else:
            raise AssertionError(f'{gmode} + override 未报错')


def test_diag_state_dict_unchanged():
    # item 9：三种 router × 三种 α 不改变 state dict / 参数数 / buffer 数
    ref = SRFFV11(channels=8)
    ref_keys = set(ref.state_dict().keys())
    ref_np = sum(1 for _ in ref.parameters())
    ref_nb = sum(1 for _ in ref.buffers())
    for router in ('learned', 'gaussian_only', 'trimmed_only'):
        for alpha in (0.02, 0.05, 0.10):
            b = SRFFV11(channels=8, global_gate_mode='force_on',
                        diagnostic_local_gate_mode='constant', diagnostic_local_gate_value=alpha,
                        diagnostic_router_mode=router)
            assert set(b.state_dict().keys()) == ref_keys
            assert sum(1 for _ in b.parameters()) == ref_np
            assert sum(1 for _ in b.buffers()) == ref_nb
    # 诊断属性不入 state dict（router_net.* 是 V1 合法参数，故只查 diagnostic/tau/global_gate_mode）
    assert not any(('diagnostic' in k or 'tau' in k or 'global_gate_mode' in k) for k in ref_keys)


def test_diag_v1_baseline_unaffected():
    # item 10：V1 类无 diagnostic 字段；v1/baseline encoder 路径不受影响
    v1 = SRFF(channels=8)
    out = v1._core(*_hl())
    assert 'diagnostic_override_active' not in out
    HybridEncoder = _import_hybrid_encoder()
    enc0 = HybridEncoder(**_small_enc_kwargs(use_srff=False))
    assert enc0.srff_blocks is None
    enc1 = HybridEncoder(**_small_enc_kwargs(use_srff=True, srff_version='v1',
                                             srff_diagnostic_router_mode='gaussian_only'))
    assert not hasattr(enc1.srff_blocks[0], 'diagnostic_router_mode')


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
