"""SRFF-V1.2 工具测试（实现文档 §6 items 17/20/21 + summarize_pilot select/final）。

summarize_pilot 部分为纯 stdlib（本地可跑）；audit 工具部分需 torch（无 torch 时安全跳过）。

运行方式（二选一）：
    python -m pytest tests/test_srff_v12_tools.py -q
    python tests/test_srff_v12_tools.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TOOLS = ROOT / 'tools' / 'wood'
for _p in (str(ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import summarize_srff_v12_pilot as P  # noqa: E402

PILOT = str(_TOOLS / 'summarize_srff_v12_pilot.py')
AUDIT = str(_TOOLS / 'audit_srff_v12_frozen_checkpoint.py')
DOMS = ['ODC', 'LDC', 'DDC', 'GDC', 'PDC']

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _torch = None
    _HAS_TORCH = False

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


def _J(ap, scratch=None):
    cats = []
    if scratch is not None:
        cats = [{'category_name': c, 'AP': (scratch if c == 'scratch' else ap)}
                for c in ['blister', 'crack', 'hole', 'scratch']]
    return {'coco_eval_bbox': {'AP': ap, 'AP50': ap, 'AP75': ap, 'AR100': ap},
            'per_category_bbox': cats}


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# summarize_pilot final（§11 指标 + 标签）
# ---------------------------------------------------------------------------
def test_pilot_final_strong_pass():
    with tempfile.TemporaryDirectory() as td:
        edir = os.path.join(td, 'eval_final')
        for st in ('force_off', 'auto', 'baseline'):
            os.makedirs(os.path.join(edir, st), exist_ok=True)
        base = {'overall': 0.3832, 'ODC': 0.5209, 'LDC': 0.5091, 'DDC': 0.5095, 'GDC': 0.2436, 'PDC': 0.1479}
        off = dict(base)                                   # force_off == baseline → TrainingEffect=0
        auto = {'overall': 0.3860, 'ODC': 0.5215, 'LDC': 0.5100, 'DDC': 0.5100, 'GDC': 0.2470, 'PDC': 0.1540}
        for scope in ['overall'] + DOMS:
            json.dump(_J(base[scope], base[scope] if scope == 'overall' else None),
                      open(os.path.join(edir, 'baseline', f'{scope}.json'), 'w'))
            json.dump(_J(off[scope]), open(os.path.join(edir, 'force_off', f'{scope}.json'), 'w'))
            json.dump(_J(auto[scope], 0.30 if scope == 'overall' else None),
                      open(os.path.join(edir, 'auto', f'{scope}.json'), 'w'))
        # baseline overall scratch = 0.300，auto = 0.30 → Δscratch=0（满足 STRONG 门 >=-0.20）
        json.dump(_J(base['overall'], 0.300), open(os.path.join(edir, 'baseline', 'overall.json'), 'w'))
        out = os.path.join(td, 'r.json'); omd = os.path.join(td, 'r.md')
        p = _run([sys.executable, PILOT, 'final', '--baseline-dir', edir, '--eval-dir', edir,
                  '--output', out, '--markdown', omd, '--state-tpl', '{state}/{scope}.json'])
        assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
        r = json.load(open(out, encoding='utf-8'))
        c = r['causal']
        assert abs(c['total_effect']['GDC'] - (24.70 - 24.36)) < 1e-6
        assert abs(c['total_effect']['PDC'] - (15.40 - 14.79)) < 1e-6
        assert abs(c['training_effect']['overall']) < 1e-6          # off==baseline → 0
        assert abs(c['adapter_effect']['overall'] - (38.60 - 38.32)) < 1e-6
        mm = r['mechanism']
        assert abs(mm['robust_mean'] - ((24.70 - 24.36) + (15.40 - 14.79)) / 2) < 1e-6
        assert 'STRONG_DIRECTIONAL_PASS' in r['decision']['labels'], r['decision']['labels']
        assert r['decision']['verdict'] == 'STRONG_PASS'
        # §9 结构键齐全
        for k in ('identity', 'baseline_load', 'freeze_audit', 'checkpoint_selection',
                  'metrics', 'causal', 'mechanism', 'decision'):
            assert k in r


def test_pilot_final_fail_negative_gdc():
    with tempfile.TemporaryDirectory() as td:
        edir = os.path.join(td, 'eval_final')
        for st in ('force_off', 'auto', 'baseline'):
            os.makedirs(os.path.join(edir, st), exist_ok=True)
        base = {'overall': 0.3832, 'ODC': 0.52, 'LDC': 0.51, 'DDC': 0.51, 'GDC': 0.2436, 'PDC': 0.1479}
        auto = dict(base); auto['GDC'] = 0.2400            # ΔGDC<0
        for scope in ['overall'] + DOMS:
            json.dump(_J(base[scope]), open(os.path.join(edir, 'baseline', f'{scope}.json'), 'w'))
            json.dump(_J(base[scope]), open(os.path.join(edir, 'force_off', f'{scope}.json'), 'w'))
            json.dump(_J(auto[scope]), open(os.path.join(edir, 'auto', f'{scope}.json'), 'w'))
        out = os.path.join(td, 'r.json')
        p = _run([sys.executable, PILOT, 'final', '--baseline-dir', edir, '--eval-dir', edir, '--output', out])
        assert p.returncode == 0
        r = json.load(open(out, encoding='utf-8'))
        assert 'FAIL_GDC_NEGATIVE' in r['decision']['labels']
        assert r['decision']['verdict'] == 'FAIL'


# ---------------------------------------------------------------------------
# summarize_pilot select
# ---------------------------------------------------------------------------
def test_pilot_select():
    with tempfile.TemporaryDirectory() as td:
        edir = os.path.join(td, 'eval_select'); os.makedirs(edir)
        adir = os.path.join(td, 'audit'); os.makedirs(adir)
        bdir = os.path.join(td, 'base'); os.makedirs(bdir)
        json.dump({'subsets': {'GDC': {'baseline_AP': 24.36}, 'PDC': {'baseline_AP': 14.79}}},
                  open(os.path.join(bdir, 'init_equivalence.json'), 'w'))
        # epoch024 最优：GDC+0.30 PDC+0.50；epoch012 小；epoch036 PDC 负
        vals = {'012': {'GDC': 0.2446, 'PDC': 0.1489}, '024': {'GDC': 0.2466, 'PDC': 0.1529},
                '036': {'GDC': 0.2460, 'PDC': 0.1470}}
        for ep, vv in vals.items():
            for dom in ('GDC', 'PDC'):
                json.dump(_J(vv[dom]), open(os.path.join(edir, f'e{ep}_{dom}_auto.json'), 'w'))
            json.dump({'ok': True}, open(os.path.join(adir, f'adapter_epoch{ep}.json'), 'w'))
        out = os.path.join(td, 'sel.json'); omd = os.path.join(td, 'sel.md')
        p = _run([sys.executable, PILOT, 'select', '--baseline-dir', bdir, '--eval-dir', edir,
                  '--audit-dir', adir, '--output', out, '--markdown', omd])
        assert p.returncode == 0, p.stdout[-1000:] + p.stderr[-1000:]
        r = json.load(open(out, encoding='utf-8'))
        assert r['selected_epoch'] == '024', r
        assert r['selected_checkpoint'] == 'adapter_epoch024.pth'


# ---------------------------------------------------------------------------
# audit 工具（items 20/21，需 torch）
# ---------------------------------------------------------------------------
def _mk_ckpts(td, tamper=None, git='abc123'):
    base = {'ema': {'module': {
        'backbone.conv.weight': _torch.zeros(4, 3, 3, 3),
        'backbone.bn.running_mean': _torch.zeros(4),
        'encoder.lateral.weight': _torch.ones(4, 4, 1, 1),
    }}}
    model = {k: v.clone() for k, v in base['ema']['module'].items()}
    model['encoder.srff_blocks.0.expert_gaussian.out_proj.weight'] = _torch.ones(4, 8, 1, 1) * 0.01
    model['encoder.srff_blocks.0.expert_gaussian.out_proj.bias'] = _torch.zeros(4)
    if tamper:
        model[tamper] = model[tamper] + 1.0
    ad = {'model': model, 'adapter': {'encoder.srff_blocks.0.expert_gaussian.out_proj.weight': model[
        'encoder.srff_blocks.0.expert_gaussian.out_proj.weight']},
        'optimizer': {}, 'epoch': 12, 'use_ema': False, 'git_commit': git,
        'baseline_sha256': 'b' * 64, 'config_sha256': 'c' * 64}
    bp = os.path.join(td, 'base.pth'); ap_ = os.path.join(td, 'adapter.pth')
    _torch.save(base, bp); _torch.save(ad, ap_)
    return bp, ap_


def test_audit_git_none_fails():
    if not _HAS_TORCH:
        _skip('需要 torch')
    with tempfile.TemporaryDirectory() as td:
        bp, ap_ = _mk_ckpts(td, git=None)
        out = os.path.join(td, 'audit.json')
        p = _run([sys.executable, AUDIT, '--baseline-checkpoint', bp, '--adapter-checkpoint', ap_, '--output', out])
        assert p.returncode == 2                       # item 20：git_commit=None → 失败
        r = json.load(open(out, encoding='utf-8'))
        assert r['CHECK'] == 'FAIL'
        # --allow-missing-git 则放行（其余应 PASS）
        p2 = _run([sys.executable, AUDIT, '--baseline-checkpoint', bp, '--adapter-checkpoint', ap_,
                   '--output', out, '--allow-missing-git'])
        assert p2.returncode == 0, p2.stdout + p2.stderr


def test_audit_detects_tampered_buffer():
    if not _HAS_TORCH:
        _skip('需要 torch')
    with tempfile.TemporaryDirectory() as td:
        bp, ap_ = _mk_ckpts(td, tamper='backbone.bn.running_mean')   # 篡改一个 frozen buffer
        out = os.path.join(td, 'audit.json')
        p = _run([sys.executable, AUDIT, '--baseline-checkpoint', bp, '--adapter-checkpoint', ap_, '--output', out])
        assert p.returncode == 2                       # item 21：捕获篡改
        r = json.load(open(out, encoding='utf-8'))
        assert r['frozen_buffer_mismatch_count'] >= 1
        assert r['CHECK'] == 'FAIL'
    # 未篡改时应 PASS
    with tempfile.TemporaryDirectory() as td2:
        bp2, ap2 = _mk_ckpts(td2)
        out2 = os.path.join(td2, 'audit.json')
        p2 = _run([sys.executable, AUDIT, '--baseline-checkpoint', bp2, '--adapter-checkpoint', ap2, '--output', out2])
        assert p2.returncode == 0, p2.stdout + p2.stderr
        r2 = json.load(open(out2, encoding='utf-8'))
        assert r2['CHECK'] == 'PASS' and r2['adapter_changed_count'] >= 1


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
