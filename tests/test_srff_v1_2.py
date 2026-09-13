"""SRFF-V1.2 单元测试（实现文档 §6 items 1-16,18,19,22）。

模块与冻结/训练语义测试；仅需 torch（srff_v1_2 + adapter_freeze 经合成包 importlib 加载）。
HybridEncoder 相关项（9/19/22）惰性导入，缺依赖则安全跳过。items 17/20/21（stage restart、
审计工具 git/tamper）见 test_srff_v12_tools.py。

运行方式（二选一）：
    python -m pytest tests/test_srff_v1_2.py -q
    python tests/test_srff_v1_2.py
"""

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'engine')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_modules():
    pkg_name = '_srff_pkg_for_v12_test'
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(ROOT / 'engine' / 'deim')]
        sys.modules[pkg_name] = pkg
    v12 = importlib.import_module(f'{pkg_name}.srff_v1_2')
    return v12


_v12 = _load_modules()
Adapter = _v12.FrozenBaseEvidenceConditionedResidualAdapter

# adapter_freeze 依赖 torch.nn，直接从 engine.misc 加载
from engine.misc import adapter_freeze as AF  # noqa: E402

PATTERNS = [r'^encoder\.srff_blocks\.0\.']

try:
    import pytest as _pytest
except Exception:  # pragma: no cover
    _pytest = None


class _Skipped(Exception):
    pass


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
        _skip(f'HybridEncoder 导入失败: {exc}')


def _hl(c=8, hs=12, ls=24):
    return torch.randn(2, c, hs, hs), torch.randn(2, c, ls, ls)


# ---------------------------------------------------------------------------
# 1-8：模块级
# ---------------------------------------------------------------------------
def test_01_zero_init_identity():
    torch.manual_seed(0)
    a = Adapter(channels=8, bottleneck_channels=8)
    a.eval()
    high, low = _hl()
    out = a._core(high, low)
    assert torch.equal(out['low_out'], low)          # 末层零初始化 → 逐元素恒等
    assert float(out['delta_g'].abs().max()) == 0.0
    assert float(out['delta_p'].abs().max()) == 0.0


def test_02_force_off_identity_no_experts():
    a = Adapter(channels=8, bottleneck_channels=8, gate_mode='force_off')
    a.eval()
    high, low = _hl()
    out = a._core(high, low)
    assert out['low_out'] is low
    assert out['ran_experts'] is False               # 不运行专家
    assert 'delta_g' not in out


def test_03_training_rejects_non_auto():
    for gm in ('force_off', 'force_on'):
        a = Adapter(channels=8, bottleneck_channels=8, gate_mode=gm)
        a.train()
        try:
            a(*_hl())
        except RuntimeError as e:
            assert 'auto' in str(e)
        else:
            raise AssertionError(f'训练期 gate_mode={gm} 未被拒绝')
    # auto 训练不报错
    a = Adapter(channels=8, bottleneck_channels=8, gate_mode='auto')
    a.train()
    a(*_hl())


def test_04_experts_signed_residual():
    torch.manual_seed(0)
    a = Adapter(channels=8, bottleneck_channels=8)
    a.eval()
    # 打破末层零初始化，验证专家能输出正负残差（无 ReLU 截断）
    with torch.no_grad():
        for e in (a.expert_gaussian, a.expert_impulse):
            e.out_proj.weight.normal_(0, 0.05)
            e.out_proj.bias.normal_(0, 0.01)
    high, low = _hl()
    out = a._core(high, low)
    for key in ('delta_g', 'delta_p'):
        d = out[key]
        assert float(d.min()) < 0 < float(d.max()), f'{key} 应同时含正负值'


def test_05_router_weights_sum_one():
    a = Adapter(channels=8, bottleneck_channels=8)
    a.eval()
    out = a._core(*_hl())
    s = out['w_g'] + out['w_p']
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
    assert float(out['w_g'].min()) >= 0.0 and float(out['w_g'].max()) <= 1.0
    assert out['w_g'].shape[-1] == 1 and out['w_g'].dim() == 4     # [N,1,1,1]


def test_06_wp_monotonic_in_cv():
    a = Adapter(channels=8, bottleneck_channels=8)
    # 直接检验路由函数：cv 越大 w_p 单调不减
    cvs = torch.tensor([[[[0.05]]], [[[0.2]]], [[[1.0]]], [[[5.0]]]], dtype=torch.float32)
    # 构造 r_p 使 cv 递增较难；改为直接调用 sigmoid 公式验证单调性
    threshold = a.router_threshold
    temperature = torch.nn.functional.softplus(a.router_raw_temperature) + 0.05
    wp = torch.sigmoid((torch.log(cvs + a.eps) - threshold) / temperature)
    assert torch.all(wp[1:] >= wp[:-1] - 1e-7)
    # _route 端到端：两种 r_p（低/高变异）→ 高变异 w_p 更大
    low_var = torch.ones(2, 8, 24, 24)
    hi_var = torch.zeros(2, 8, 24, 24)
    hi_var[:, :, ::2, ::2] = 5.0
    _, wp_lo, _, _ = a._route(low_var)
    _, wp_hi, _, _ = a._route(hi_var)
    assert float(wp_hi.mean()) >= float(wp_lo.mean()) - 1e-6


def test_07_temperature_above_floor():
    a = Adapter(channels=8, bottleneck_channels=8, router_temperature_init=0.25)
    temp = torch.nn.functional.softplus(a.router_raw_temperature) + 0.05
    assert float(temp) > 0.05
    assert abs(float(temp) - 0.25) < 1e-4          # inverse-softplus 正确初始化
    # 非法初值（<=0.05）应报错
    try:
        Adapter(channels=8, bottleneck_channels=8, router_temperature_init=0.05)
    except AssertionError:
        pass
    else:
        raise AssertionError('router_temperature_init<=0.05 未报错')


def test_08_rms_budget_not_exceeded():
    torch.manual_seed(0)
    a = Adapter(channels=8, bottleneck_channels=8, alpha_max=0.10)
    with torch.no_grad():
        for e in (a.expert_gaussian, a.expert_impulse):
            e.out_proj.weight.normal_(0, 0.5)      # 放大专家输出以触发预算裁剪
    a.eval()
    high, low = _hl()
    out = a._core(high, low)
    tol = 1e-4
    ok = bool(torch.all(out['delta_rms_post'] <= a.alpha_max * out['low_rms'] + tol))
    assert ok, (out['delta_rms_post'], a.alpha_max * out['low_rms'])
    assert float(out['budget_scale'].max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 9/19/22：encoder 级
# ---------------------------------------------------------------------------
def _small_enc_kwargs(**over):
    d = dict(in_channels=[16, 32, 64], feat_strides=[8, 16, 32], hidden_dim=16, nhead=4,
             dim_feedforward=32, dropout=0.0, use_encoder_idx=[2], num_encoder_layers=1,
             eval_spatial_size=None, version='dfine')
    d.update(over)
    return d


def test_09_only_block0_active():
    HE = _import_hybrid_encoder()
    enc = HE(**_small_enc_kwargs(use_srff=True, srff_version='v1_2', srff_active_blocks=[0]))
    assert type(enc.srff_blocks[0]).__name__ == 'FrozenBaseEvidenceConditionedResidualAdapter'
    assert isinstance(enc.srff_blocks[1], nn.Identity)
    assert enc.srff_active_levels == {0}


def test_19_module_prefix_and_22_baseline_no_regression():
    HE = _import_hybrid_encoder()
    # item 22：use_srff=False 时无 srff block，baseline 路径不受影响
    enc0 = HE(**_small_enc_kwargs(use_srff=False))
    assert enc0.srff_blocks is None
    # item 19：DDP module. 前缀不破坏白名单（is_adapter_name 去前缀）
    assert AF.is_adapter_name('module.encoder.srff_blocks.0.router_threshold', PATTERNS)
    assert AF.is_adapter_name('encoder.srff_blocks.0.expert_gaussian.in_proj.weight', PATTERNS)
    assert not AF.is_adapter_name('module.encoder.srff_blocks.1.x', PATTERNS)
    assert not AF.is_adapter_name('backbone.stage1.0.weight', PATTERNS)


# ---------------------------------------------------------------------------
# 10-16：冻结 / optimizer / 训练 / 审计
# ---------------------------------------------------------------------------
def _syn_model(C=8):
    m = nn.Module()
    m.backbone = nn.Sequential(nn.Conv2d(3, C, 3, padding=1), nn.BatchNorm2d(C))
    enc = nn.Module()
    enc.srff_blocks = nn.ModuleList([
        Adapter(channels=C, bottleneck_channels=8, global_tau=(0.0, 1e-3)),  # 低 tau → 门≈1，便于训练测试
        nn.Identity()])
    m.encoder = enc
    m.head = nn.Conv2d(C, 4, 1)
    return m


def test_10_11_freeze_and_optimizer_only_adapter():
    torch.manual_seed(0)
    m = _syn_model()
    # 冻结前：optimizer 若用全部参数会含非 adapter
    all_params = list(m.parameters())
    tr, fz = AF.freeze_non_adapter(m, PATTERNS)
    assert fz > 0 and len(tr) > 0
    # item 11：optimizer 只含 adapter 参数且无重复
    opt_params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=1e-3)
    n_opt = sum(len(g['params']) for g in opt.param_groups)
    adapter_named = {n for n, p in m.named_parameters() if AF.is_adapter_name(n, PATTERNS)}
    assert n_opt == len(adapter_named) == len(tr)
    ids = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids) == len(set(ids))                 # 无重复
    # item 10：verify_trainable_set 校验（冻结在 optimizer 之前完成的等价断言）
    AF.verify_trainable_set(m, PATTERNS)
    # 非 adapter 仍 requires_grad → 应报错
    m.head.weight.requires_grad_(True)
    try:
        AF.verify_trainable_set(m, PATTERNS)
    except RuntimeError:
        pass
    else:
        raise AssertionError('越界 requires_grad 未被 verify 捕获')


def test_12_14_15_backward_grads_and_frozen_bitwise():
    torch.manual_seed(0)
    m = _syn_model()
    AF.freeze_non_adapter(m, PATTERNS)
    AF.apply_adapter_train_mode(m, PATTERNS)          # item 13 前置
    # item 13：非 adapter BN 仍 eval
    assert m.backbone[1].training is False
    assert m.encoder.srff_blocks[0].training is True
    assert AF.count_non_adapter_train_mode(m, PATTERNS) == 0

    frozen_before = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    frozen_buf_before = {n: b.detach().clone() for n, b in m.named_buffers()
                         if not AF.is_adapter_name(n, PATTERNS)}
    adapter_before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)

    for _ in range(2):
        opt.zero_grad()
        high, low = _hl(8)
        out = m.encoder.srff_blocks[0](high, low)
        loss = out.square().mean() + m.head(m.backbone(torch.randn(2, 3, 24, 24))).square().mean()
        loss.backward()
        # item 12：非 adapter grad 全为 None
        for n, p in m.named_parameters():
            if not AF.is_adapter_name(n, PATTERNS):
                assert p.grad is None, f'非 adapter 参数 {n} 不应有 grad'
        opt.step()

    # item 14：frozen parameter / buffer bitwise 不变
    for n, p in m.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p.detach(), frozen_before[n]), f'frozen 参数 {n} 被改动'
    for n, b in m.named_buffers():
        if not AF.is_adapter_name(n, PATTERNS):
            assert torch.equal(b.detach(), frozen_buf_before[n]), f'frozen buffer {n} 被改动'
    # item 15：至少一个 adapter 参数变化
    changed = any(not torch.equal(p.detach(), adapter_before[n])
                  for n, p in m.named_parameters() if p.requires_grad)
    assert changed, '两步训练后 adapter 参数应发生变化'


def test_16_baseline_load_audit_fail():
    # audit_baseline_load：缺非 adapter key / shape mismatch / 错误来源 → 报错
    import tempfile, os
    m = _syn_model()
    good = {'ema': {'module': {k: v.detach().clone() for k, v in m.state_dict().items()}}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'ck.pth')
        torch.save(good, p)
        aud = AF.audit_baseline_load(m, p, PATTERNS)   # 正常应通过
        assert aud['ok'] and aud['source'] == 'ema.module'
        # 删掉一个非 adapter key → missing 非 adapter → 失败
        bad = {'ema': {'module': {k: v for k, v in good['ema']['module'].items()
                                  if k != 'head.weight'}}}
        p2 = os.path.join(td, 'ck2.pth'); torch.save(bad, p2)
        try:
            AF.audit_baseline_load(m, p2, PATTERNS)
        except RuntimeError:
            pass
        else:
            raise AssertionError('缺非 adapter key 未报错')
        # shape mismatch
        bad2 = {'ema': {'module': dict(good['ema']['module'])}}
        bad2['ema']['module']['head.weight'] = torch.zeros(3, 3)
        p3 = os.path.join(td, 'ck3.pth'); torch.save(bad2, p3)
        try:
            AF.audit_baseline_load(m, p3, PATTERNS)
        except RuntimeError:
            pass
        else:
            raise AssertionError('shape mismatch 未报错')


def test_18_adapter_checkpoint_no_fake_ema():
    m = _syn_model()
    AF.freeze_non_adapter(m, PATTERNS)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    ck = AF.build_adapter_checkpoint(m, opt, 5, PATTERNS, baseline_sha256='abc', config_sha256='cfg',
                                     git='deadbeef', extra={'ap_overall': 1.0})
    assert ck['use_ema'] is False
    assert 'ema' not in ck                            # 无伪 EMA 字段
    assert ck['epoch'] == 5 and ck['baseline_sha256'] == 'abc' and ck['git_commit'] == 'deadbeef'
    # adapter-only state 只含 adapter key
    assert all(AF.is_adapter_name(k, PATTERNS) for k in ck['adapter'])
    assert len(ck['adapter']) > 0


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
            print(f'[SKIP] {name}: {exc}'); skipped += 1
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] {name}: {type(exc).__name__}: {exc}'); failed += 1
        else:
            print(f'[PASS] {name}'); passed += 1
    print(f'\nSummary: {passed} passed, {skipped} skipped, {failed} failed')
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)
