"""容量拆分工具的单元测试（实现文档 §8 items 11-14）。

覆盖 summarize_srff_v11_expert_capacity.py 与 create_srff_v11_capacity_manifest.py：
  11. 汇总工具正确计算 ΔAP、best α 与全部分流标签；
  12. 缺文件 / NaN / 错误身份 / 参照未复现 → 非零退出；
  13. 报告中不存在把 ref_learned-ref_force_off 称为 FullCapacity 的内容；
  14. 身份工具 before 拒绝覆盖，after 能发现 checkpoint/config/code/git 变化。

运行方式（二选一）：
    python -m pytest tests/test_srff_v11_expert_capacity.py -q
    python tests/test_srff_v11_expert_capacity.py
"""

import hashlib
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

import summarize_srff_v11_expert_capacity as S  # noqa: E402

SUMM = str(_TOOLS / 'summarize_srff_v11_expert_capacity.py')
MANI = str(_TOOLS / 'create_srff_v11_capacity_manifest.py')
DOMAINS = S.DOMAINS
STATES = S.ALL_STATES
CATS = S.CATS
EXP = S.EXPECTED_CKPT_SHA

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


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def _aps():
    """11 状态 × GDC/PDC 的 AP（fraction）。设计成触发已知标签组合。"""
    a = {st: {} for st in STATES}
    a['ref_force_off'] = {'GDC': 0.2444, 'PDC': 0.1577}
    a['ref_learned'] = {'GDC': 0.2446, 'PDC': 0.1581}
    a['gaussian_a002'] = {'GDC': 0.2479, 'PDC': 0.1577}   # GDC Δ+0.35
    a['gaussian_a005'] = {'GDC': 0.2469, 'PDC': 0.1577}   # Δ+0.25
    a['gaussian_a010'] = {'GDC': 0.2439, 'PDC': 0.1577}   # Δ-0.05
    a['trimmed_a002'] = {'GDC': 0.2444, 'PDC': 0.1582}    # PDC Δ+0.05
    a['trimmed_a005'] = {'GDC': 0.2444, 'PDC': 0.1585}    # Δ+0.08
    a['trimmed_a010'] = {'GDC': 0.2444, 'PDC': 0.1583}    # Δ+0.06
    a['mix_a002'] = {'GDC': 0.2446, 'PDC': 0.1580}
    a['mix_a005'] = {'GDC': 0.2448, 'PDC': 0.1582}
    a['mix_a010'] = {'GDC': 0.2450, 'PDC': 0.1584}
    return a


def _metrics(ap_frac):
    ap = ap_frac * 100.0
    return {'AP': ap, 'AP50': ap, 'AP75': ap, 'AR100': ap, 'cat': {c: ap for c in CATS}}


def _M(aps=None):
    aps = aps or _aps()
    return {st: {dom: _metrics(aps[st][dom]) for dom in DOMAINS} for st in STATES}


def _J(ap_frac):
    ap = ap_frac
    return {'coco_eval_bbox': {'AP': ap, 'AP50': ap, 'AP75': ap, 'AR100': ap},
            'per_category_bbox': [{'category_name': c, 'AP': ap} for c in CATS]}


def _manifest(sha, training=False):
    return {'phase': 'after', 'experiment_type': 'eval_only', 'training_performed': training,
            'checkpoint_source': 'ema', 'seed': 0, 'expect_checkpoint_sha256': sha,
            'git_commit': None, 'git_branch': None, 'git_dirty': None,
            'checkpoint_sha256_before': sha, 'checkpoint_sha256_after': sha,
            'states': STATES, 'domains': DOMAINS}


def _write_evals(cdir, aps):
    for st in STATES:
        for dom in DOMAINS:
            with open(os.path.join(cdir, f'{st}_{dom}.json'), 'w', encoding='utf-8') as f:
                json.dump(_J(aps[st][dom]), f)


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# 11. 派生量 / α 曲线 / 标签（函数级）
# ---------------------------------------------------------------------------
def test_build_and_labels_direct():
    M = _M()
    res = S.build(M)
    dv = res['derived']
    assert abs(dv['GlobalGateOnEffect']['GDC'] - 0.02) < 1e-6
    assert abs(dv['GlobalGateOnEffect']['PDC'] - 0.04) < 1e-6
    assert abs(dv['GaussianBestGDC'] - 0.35) < 1e-6
    assert abs(dv['TrimmedBestPDC'] - 0.08) < 1e-6
    assert abs(res['alpha_curves']['gaussian']['GDC']['a002'] - 0.35) < 1e-6
    assert abs(res['alpha_curves']['trimmed']['PDC']['a005'] - 0.08) < 1e-6
    assert abs(res['diff_vs_ref_off']['ref_learned']['GDC']['AP'] - 0.02) < 1e-6
    labs = set(S.build_labels(M, res)['labels'])
    assert 'GAUSSIAN_BASIS_POSITIVE' in labs
    assert 'TRIMMED_BASIS_WEAK' in labs
    assert 'OVERCORRECTION' in labs
    assert 'NO_FIXED_BASIS_SIGNAL' not in labs
    assert 'BORDERLINE_TRIMMED' not in labs


# ---------------------------------------------------------------------------
# 11 + 13. 汇总工具端到端（含无 FullCapacity 命名）
# ---------------------------------------------------------------------------
def test_summarizer_happy_path():
    with tempfile.TemporaryDirectory() as td:
        cdir = os.path.join(td, 'eval'); os.makedirs(cdir)
        _write_evals(cdir, _aps())
        mfp = os.path.join(td, 'manifest.json')
        with open(mfp, 'w', encoding='utf-8') as f:
            json.dump(_manifest(EXP), f)
        omd = os.path.join(td, 'r.md'); ojson = os.path.join(td, 'r.json')
        p = _run([sys.executable, SUMM, '--eval-dir', cdir, '--identity-manifest', mfp,
                  '--out-md', omd, '--out-json', ojson])
        assert p.returncode == 0, (p.stdout[-1500:] + p.stderr[-1500:])
        r = json.load(open(ojson, encoding='utf-8'))
        assert abs(r['derived']['GaussianBestGDC'] - 0.35) < 1e-6
        md = open(omd, encoding='utf-8').read()
        assert 'FullCapacity' not in md          # item 13
        assert 'GlobalGateOnEffect' in md
        assert '命中标签' in md


# ---------------------------------------------------------------------------
# 12. 完整性门禁
# ---------------------------------------------------------------------------
def _summ_with(td, cdir, mfp):
    omd = os.path.join(td, 'r.md'); ojson = os.path.join(td, 'r.json')
    return _run([sys.executable, SUMM, '--eval-dir', cdir, '--identity-manifest', mfp,
                 '--out-md', omd, '--out-json', ojson])


def test_gate_missing_file():
    with tempfile.TemporaryDirectory() as td:
        cdir = os.path.join(td, 'eval'); os.makedirs(cdir)
        aps = _aps(); _write_evals(cdir, aps)
        os.remove(os.path.join(cdir, 'gaussian_a010_GDC.json'))
        mfp = os.path.join(td, 'm.json')
        with open(mfp, 'w', encoding='utf-8') as f:
            json.dump(_manifest(EXP), f)
        assert _summ_with(td, cdir, mfp).returncode == S.EXIT_GATE


def test_gate_ref_repro_fail():
    with tempfile.TemporaryDirectory() as td:
        cdir = os.path.join(td, 'eval'); os.makedirs(cdir)
        aps = _aps(); aps['ref_force_off']['GDC'] = 0.30   # 偏离 24.44
        _write_evals(cdir, aps)
        mfp = os.path.join(td, 'm.json')
        with open(mfp, 'w', encoding='utf-8') as f:
            json.dump(_manifest(EXP), f)
        assert _summ_with(td, cdir, mfp).returncode == S.EXIT_GATE


def test_gate_training_manifest():
    with tempfile.TemporaryDirectory() as td:
        cdir = os.path.join(td, 'eval'); os.makedirs(cdir)
        _write_evals(cdir, _aps())
        mfp = os.path.join(td, 'm.json')
        with open(mfp, 'w', encoding='utf-8') as f:
            json.dump(_manifest(EXP, training=True), f)
        assert _summ_with(td, cdir, mfp).returncode == S.EXIT_GATE


def test_gate_nan():
    with tempfile.TemporaryDirectory() as td:
        cdir = os.path.join(td, 'eval'); os.makedirs(cdir)
        _write_evals(cdir, _aps())
        with open(os.path.join(cdir, 'mix_a002_GDC.json'), 'w', encoding='utf-8') as f:
            f.write('{"coco_eval_bbox": {"AP": NaN}, "per_category_bbox": []}')
        mfp = os.path.join(td, 'm.json')
        with open(mfp, 'w', encoding='utf-8') as f:
            json.dump(_manifest(EXP), f)
        assert _summ_with(td, cdir, mfp).returncode == S.EXIT_GATE


# ---------------------------------------------------------------------------
# 14. 身份清单工具
# ---------------------------------------------------------------------------
def test_manifest_tool_lifecycle():
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, 'c.yml')
        with open(cfg, 'w', encoding='utf-8') as f:
            f.write('HybridEncoder: {}\n')
        ck = os.path.join(td, 'ck.pth')
        with open(ck, 'wb') as f:
            f.write(b'dummy-checkpoint-bytes')
        cksha = hashlib.sha256(b'dummy-checkpoint-bytes').hexdigest()
        out = os.path.join(td, 'manifest.json')
        base = [sys.executable, MANI, '--config', cfg, '--checkpoint', ck, '--out', out,
                '--repo', str(ROOT)]
        # before + 默认预期 sha → 拒绝（dummy != 0c6434...）
        assert _run(base + ['--phase', 'before']).returncode == 2
        # before + 正确预期 sha → 成功
        assert _run(base + ['--phase', 'before', '--expect-checkpoint-sha', cksha]).returncode == 0
        assert os.path.isfile(out)
        # before 再次 → 拒绝覆盖
        assert _run(base + ['--phase', 'before', '--expect-checkpoint-sha', cksha]).returncode == 2
        # after 未改 → 通过
        assert _run(base + ['--phase', 'after']).returncode == 0
        mf = json.load(open(out, encoding='utf-8'))
        assert mf['checkpoint_sha256_after'] == cksha and mf['phase'] == 'after'
        # 改 checkpoint → after 发现不一致（非零）
        with open(ck, 'wb') as f:
            f.write(b'CHANGED-checkpoint')
        assert _run(base + ['--phase', 'after']).returncode == 3


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
