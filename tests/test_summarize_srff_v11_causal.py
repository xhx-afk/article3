"""summarize_srff_v11_causal.py 的单元测试（实现文档 §8 items 10-12）。

纯 stdlib（因果汇总工具只依赖标准库），用合成 JSON 校验：
  10. 完整性门禁：缺文件 / NaN / 错误 D 关身份 → 非零退出；
  11. 因果分解公式（O-B / N-O / I-O / I-N）；
  12. AP [0,1] -> AP point 换算。

运行方式（二选一）：
    python -m pytest tests/test_summarize_srff_v11_causal.py -q
    python tests/test_summarize_srff_v11_causal.py
"""

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TOOLS = ROOT / 'tools' / 'wood'
for _p in (str(ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import summarize_srff_v11_causal as C  # noqa: E402

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
# 合成 JSON / 状态构造
# ---------------------------------------------------------------------------
def _mkj(ap, scratch=None):
    cats = []
    if scratch is not None:
        cats = [{'category_name': 'scratch', 'AP': scratch},
                {'category_name': 'crack', 'AP': ap}]
    return {'coco_eval_bbox': {'AP': ap, 'AP50': ap, 'AP75': ap, 'AR100': ap,
                               'AP_small': ap, 'AP_medium': ap, 'AP_large': ap},
            'per_category_bbox': cats}


def _state(ov_ap, dom_ap, clean_scratch, robust_scratch=0.30):
    doms = {}
    for d in C.DOMS:
        sc = clean_scratch if d in C.CLEAN else robust_scratch
        doms[d] = _mkj(dom_ap[d], sc)
    return {'tag': 'X', 'overall': _mkj(ov_ap), 'domains': doms}


def _four_states():
    dom_ap = {
        'B': {'ODC': 0.5209, 'LDC': 0.5091, 'DDC': 0.5095, 'GDC': 0.2436, 'PDC': 0.1479},
        'N': {'ODC': 0.5200, 'LDC': 0.5100, 'DDC': 0.5100, 'GDC': 0.2445, 'PDC': 0.1576},
        'O': {'ODC': 0.5200, 'LDC': 0.5100, 'DDC': 0.5100, 'GDC': 0.2440, 'PDC': 0.1500},
        'I': {'ODC': 0.5200, 'LDC': 0.5100, 'DDC': 0.5100, 'GDC': 0.2480, 'PDC': 0.1590},
    }
    B = _state(0.3832, dom_ap['B'], 0.400)
    N = _state(0.3864, dom_ap['N'], 0.388)
    O = _state(0.3850, dom_ap['O'], 0.390)
    I = _state(0.3900, dom_ap['I'], 0.395)
    return B, N, O, I


# ---------------------------------------------------------------------------
# 11. 因果分解公式
# ---------------------------------------------------------------------------
def test_decomp_formula():
    d = C._decomp(10.0, 12.0, 11.0, 13.0)   # b, n, o, i
    assert d['JointTrainingDriftEstimate'] == 1.0    # O-B
    assert d['NormalGateContribution'] == 1.0         # N-O
    assert d['FullSRFFCapacity'] == 2.0               # I-O
    assert d['GateSuppression'] == 1.0                # I-N
    d2 = C._decomp(None, 12.0, 11.0, 13.0)
    assert d2['JointTrainingDriftEstimate'] is None   # None 传播
    assert d2['GateSuppression'] == 1.0


def test_build_results_overall_and_gdc():
    B, N, O, I = _four_states()
    res = C.build_results(B, N, O, I)
    ov = res['overall']['AP']
    assert abs(ov['JointTrainingDriftEstimate'] - (38.50 - 38.32)) < 1e-6
    assert abs(ov['NormalGateContribution'] - (38.64 - 38.50)) < 1e-6
    assert abs(ov['FullSRFFCapacity'] - (39.00 - 38.50)) < 1e-6
    assert abs(ov['GateSuppression'] - (39.00 - 38.64)) < 1e-6
    gdc = res['gdc_pdc']['GDC']
    assert abs(gdc['FullSRFFCapacity_I_minus_O'] - (24.80 - 24.40)) < 1e-6
    assert abs(gdc['GateSuppression_I_minus_N'] - (24.80 - 24.45)) < 1e-6


def test_clean_scratch_avg_decomp():
    B, N, O, I = _four_states()
    res = C.build_results(B, N, O, I)
    cs = res['clean_scratch_avg']
    assert abs(cs['states']['B'] - 40.0) < 1e-6
    assert abs(cs['states']['O'] - 39.0) < 1e-6
    assert abs(cs['decomp']['JointTrainingDriftEstimate'] - (-1.0)) < 1e-6   # O-B
    assert abs(cs['decomp']['NormalGateContribution'] - (-0.2)) < 1e-6       # N-O


# ---------------------------------------------------------------------------
# 12. AP [0,1] -> AP point 换算
# ---------------------------------------------------------------------------
def test_ap_scaling_to_point():
    j = _mkj(0.3864)
    assert abs(C.gv(j, 'AP') - 38.64) < 1e-6
    assert abs(C.per_category(_mkj(0.5, 0.3811))['scratch'] - 38.11) < 1e-6


# ---------------------------------------------------------------------------
# 分流判据（§6.4）
# ---------------------------------------------------------------------------
def test_verdict_labels():
    B, N, O, I = _four_states()
    v = C.build_verdicts(B, N, O, I, 38.64)
    labels = set(v['labels'])
    # clean scratch O-B=-1.0<=-0.50 -> SHARED_DRIFT；N-O=-0.2 -> 不触发 CLEAN_FALSE_ACTIVATION
    assert 'SHARED_DRIFT' in labels
    assert 'CLEAN_FALSE_ACTIVATION' not in labels
    # GDC I-N=+0.35>=0.30 -> GDC_GATE_SUPPRESSION；I-O=+0.40>0.20 -> 不触发 GDC_EXPERT_WEAK
    assert 'GDC_GATE_SUPPRESSION' in labels
    assert 'GDC_EXPERT_WEAK' not in labels
    # PDC N-O=+0.76>0 且 I-N=+0.14<=0.20 -> PDC_PATH_RETAINED
    assert 'PDC_PATH_RETAINED' in labels
    # N overall AP=38.64 -> auto 复现通过
    assert 'AUTO_REPRO_FAIL' not in labels


def test_auto_repro_fail_label():
    B, N, O, I = _four_states()
    N['overall'] = _mkj(0.3900)   # N AP=39.00，偏离 38.64 达 0.36 > 0.02
    v = C.build_verdicts(B, N, O, I, 38.64)
    assert 'AUTO_REPRO_FAIL' in v['labels']


# ---------------------------------------------------------------------------
# D 关诊断提取
# ---------------------------------------------------------------------------
def test_extract_diagnostic():
    gd = {d: {'all': {'quantities': {'global_score': {'mean': 0.78}, 'global_gate': {'mean': 0.5},
                                     'pre_global_gate': {'mean': 0.012}, 'gate': {'mean': 0.006}}}}
          for d in C.DOMS}
    diag = {'metadata': {'srff_version': 'v1_1', 'srff_global_gate_mode': 'auto',
                         'srff_global_threshold_low': 0.78, 'srff_global_threshold_high': 0.80,
                         'srff_active_block_count': 1, 'checkpoint_source': 'ema',
                         'srff_blocks': [{'diag_name': 'block0_p5_to_p4'}]},
            'groups': {'by_block_domain_region': {'block0_p5_to_p4': gd}},
            'comparisons': {'block0_p5_to_p4': {'final_gate_robust_over_clean': 2.12}}}
    x = C.extract_diagnostic(diag)
    assert x['per_domain_gate']['ODC']['global_gate'] == 0.5
    assert x['per_domain_gate']['GDC']['gate'] == 0.006
    assert x['comparisons']['final_gate_robust_over_clean'] == 2.12
    assert x['metadata']['srff_global_gate_mode'] == 'auto'


# ---------------------------------------------------------------------------
# 10. 完整性门禁
# ---------------------------------------------------------------------------
def test_check_diagnostic_gate():
    good = {'metadata': {'partial_run': False, 'num_images': 2760, 'srff_active_block_count': 1,
                         'srff_global_gate_mode': 'auto', 'domain_audit': {'audit_ok': True}}}
    errs = []
    C.check_diagnostic(good, errs)
    assert errs == []
    bad_mode = json.loads(json.dumps(good)); bad_mode['metadata']['srff_global_gate_mode'] = 'force_off'
    e2 = []; C.check_diagnostic(bad_mode, e2); assert any('mode' in x for x in e2)
    bad_part = json.loads(json.dumps(good)); bad_part['metadata']['partial_run'] = True
    bad_part['metadata']['num_images'] = 16
    e3 = []; C.check_diagnostic(bad_part, e3); assert e3
    bad_blk = json.loads(json.dumps(good)); bad_blk['metadata']['srff_active_block_count'] = 2
    e4 = []; C.check_diagnostic(bad_blk, e4); assert any('block' in x for x in e4)


def test_load_state_missing_file_records_error():
    errs = []
    with tempfile.TemporaryDirectory() as td:
        st = C.load_state(td, 'nope_overall.json', 'nope_{dom}.json', 'X', errs)
    assert st['overall'] is None
    assert len(errs) >= 1 and any('缺失' in e for e in errs)


def test_load_checked_nan_records_error():
    errs = []
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'nan.json')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('{"coco_eval_bbox": {"AP": NaN}}')
        j = C._load_checked(p, errs)
    assert any('NaN' in e or 'Infinity' in e or '损坏' in e for e in errs)


def _run_main(argv):
    old = sys.argv
    sys.argv = argv
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            C.main()
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old


def test_main_missing_file_exits_nonzero():
    with tempfile.TemporaryDirectory() as td:
        bdir = os.path.join(td, 'b'); cdir = os.path.join(td, 'c')
        os.makedirs(bdir); os.makedirs(cdir)
        djson = os.path.join(td, 'd.json')
        code = _run_main(['prog', '--baseline-dir', bdir, '--causal-dir', cdir,
                          '--diagnostic-json', djson, '--out-md', os.path.join(td, 'o.md'),
                          '--out-json', os.path.join(td, 'o.json')])
        assert code == C.EXIT_GATE


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
