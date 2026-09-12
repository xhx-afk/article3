"""SRFF-V1/V1.1 训练检查点诊断工具的单元测试（文档 §7 的 14 项 + V1/V1.1 复算一致性）。

不依赖真实数据、真实检查点或 engine 重依赖：通过 importlib 直接加载
``tools/wood/inspect_srff_checkpoint.py``（其纯工具层仅需 torch）与 ``engine/deim/srff.py``。

运行方式（二选一）：
    python -m pytest tests/test_srff_checkpoint_diagnostics.py -q
    python tests/test_srff_checkpoint_diagnostics.py
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load(ROOT / 'tools' / 'wood' / 'inspect_srff_checkpoint.py', '_inspect_srff_ckpt_standalone')
srff = _load(ROOT / 'engine' / 'deim' / 'srff.py', '_srff_standalone_for_diag')


def _load_v11_class():
    """用合成包加载 srff_v1_1（其 `from .srff import` 需包上下文），仅需 torch。"""
    import types
    pkg_name = '_srff_pkg_for_diag_test'
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(ROOT / 'engine' / 'deim')]
        sys.modules[pkg_name] = pkg
    importlib.import_module(f'{pkg_name}.srff')
    v11 = importlib.import_module(f'{pkg_name}.srff_v1_1')
    return v11.SelectiveRobustFrequencyFusionV11


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


# ---------------------------------------------------------------------------
# 1. 文件名解析与域/source 提取
# ---------------------------------------------------------------------------
def test_resolve_identity():
    assert diag.resolve_identity('00004_DDC_3.jpg') == ('00004', 'DDC', 3)
    assert diag.resolve_identity('00069_PDC_8.jpg') == ('00069', 'PDC', 8)
    assert diag.resolve_identity('00123_ODC_1.png') == ('00123', 'ODC', 1)
    for bad in ['00004_XXX_3.jpg', 'nope.jpg', '00004_DDC.jpg', '']:
        try:
            diag.resolve_identity(bad)
        except ValueError:
            continue
        raise AssertionError(f'应拒绝非法文件名: {bad!r}')


# ---------------------------------------------------------------------------
# 2. COCO xywh -> feature mask：缩放 / floor / ceil / clamp
# ---------------------------------------------------------------------------
def test_box_to_feature_mask_scaling_and_clamp():
    m = diag.box_to_feature_mask((0, 0, 50, 50), 100, 100, 10, 10)
    assert m.shape == (10, 10)
    assert bool(m[0:5, 0:5].all())
    assert not bool(m[5:, :].any())
    assert not bool(m[:, 5:].any())
    # 越界框被 clamp 回网格内
    m2 = diag.box_to_feature_mask((90, 90, 50, 50), 100, 100, 10, 10)
    assert m2 is not None and int(m2.sum()) >= 1
    assert bool(m2[9, 9])
    # 非法宽高
    assert diag.box_to_feature_mask((10, 10, 0, 5), 100, 100, 10, 10) is None
    assert diag.box_to_feature_mask((10, 10, 5, -1), 100, 100, 10, 10) is None


# ---------------------------------------------------------------------------
# 3. 极小合法框至少映射一个 pixel
# ---------------------------------------------------------------------------
def test_tiny_box_at_least_one_pixel():
    m = diag.box_to_feature_mask((50, 50, 1, 1), 600, 600, 80, 80)
    assert m is not None
    assert int(m.sum()) >= 1
    # 角落极小框
    m2 = diag.box_to_feature_mask((0, 0, 0.5, 0.5), 640, 640, 80, 80)
    assert m2 is not None and int(m2.sum()) >= 1


# ---------------------------------------------------------------------------
# 4. 多框重叠按并集计数一次
# ---------------------------------------------------------------------------
def test_multi_box_union_foreground():
    anns = [
        {'bbox': [0, 0, 20, 20], 'category_id': 0, 'iscrowd': 0},
        {'bbox': [10, 10, 20, 20], 'category_id': 0, 'iscrowd': 0},
    ]
    masks, ign, inv = diag.build_region_masks(anns, 40, 40, 10, 10, {0: 'hole'}, 0.10)
    fg = masks['foreground']
    # box1 -> [0:5,0:5]=25, box2 -> [2:8,2:8]=36, 重叠 [2:5,2:5]=9 -> 并集 52
    assert int(fg.sum()) == 52
    assert ign == 0 and inv == 0


# ---------------------------------------------------------------------------
# 5. 分类别 mask
# ---------------------------------------------------------------------------
def test_per_class_masks():
    anns = [
        {'bbox': [0, 0, 20, 20], 'category_id': 0, 'iscrowd': 0},
        {'bbox': [20, 20, 20, 20], 'category_id': 1, 'iscrowd': 0},
    ]
    cat = {0: 'hole', 1: 'blister'}
    masks, _, _ = diag.build_region_masks(anns, 40, 40, 10, 10, cat, 0.10)
    assert 'class:hole' in masks and 'class:blister' in masks
    assert int(masks['class:hole'].sum()) == 25
    assert int(masks['class:blister'].sum()) == 25
    assert bool(masks['class:hole'][0, 0]) and not bool(masks['class:hole'][9, 9])
    assert bool(masks['class:blister'][9, 9]) and not bool(masks['class:blister'][0, 0])


# ---------------------------------------------------------------------------
# 6. 带 margin 的 background mask
# ---------------------------------------------------------------------------
def test_background_with_margin():
    anns = [{'bbox': [16, 16, 8, 8], 'category_id': 0, 'iscrowd': 0}]
    masks, _, _ = diag.build_region_masks(anns, 40, 40, 10, 10, {0: 'hole'}, 0.10)
    fg, bg = masks['foreground'], masks['background']
    assert int(fg.sum()) == 4                    # [4:6,4:6]
    # 扩张框 [3:7,3:7]=16 -> bg=100-16=84
    assert int(bg.sum()) == 84
    assert not bool((bg & fg).any())             # bg 与 fg 不相交
    assert int(masks['all'].sum()) == 100
    # margin=0 时 bg = ~fg
    masks0, _, _ = diag.build_region_masks(anns, 40, 40, 10, 10, {0: 'hole'}, 0.0)
    assert int(masks0['background'].sum()) == 96


# ---------------------------------------------------------------------------
# 7. 空 mask 处理（无 NaN，count=0 -> null）
# ---------------------------------------------------------------------------
def test_empty_mask_and_empty_stats():
    masks, ign, inv = diag.build_region_masks([], 40, 40, 10, 10, {0: 'hole'}, 0.10)
    assert int(masks['foreground'].sum()) == 0
    assert int(masks['background'].sum()) == 100
    assert int(masks['all'].sum()) == 100
    st = diag.StreamingStats(0.0, 1.0, 100, False, 'cpu')
    d = st.to_dict()
    assert d['count'] == 0 and d['mean'] is None and d['p95'] is None and d['max'] is None
    # iscrowd 被忽略并计数
    anns = [{'bbox': [0, 0, 10, 10], 'category_id': 0, 'iscrowd': 1}]
    _, ign2, _ = diag.build_region_masks(anns, 40, 40, 10, 10, {0: 'hole'}, 0.10)
    assert ign2 == 1


# ---------------------------------------------------------------------------
# 8. 流式 mean/std 与精确结果一致
# ---------------------------------------------------------------------------
def test_streaming_mean_std_exact():
    torch.manual_seed(0)
    x = torch.rand(37, dtype=torch.float64)
    st = diag.StreamingStats(0.0, 1.0, 1000, False, 'cpu')
    st.update(x)
    assert st.count == 37
    assert abs(st.mean() - float(x.mean())) < 1e-9
    assert abs(st.std() - float(x.std(unbiased=False))) < 1e-9
    # 分批 update 与一次性一致
    st2 = diag.StreamingStats(0.0, 1.0, 1000, False, 'cpu')
    st2.update(x[:10])
    st2.update(x[10:])
    assert st2.count == 37 and abs(st2.mean() - st.mean()) < 1e-9
    # merge 等价
    a = diag.StreamingStats(0.0, 1.0, 1000, False, 'cpu'); a.update(x[:20])
    b = diag.StreamingStats(0.0, 1.0, 1000, False, 'cpu'); b.update(x[20:])
    a.merge(b)
    assert a.count == 37 and abs(a.mean() - st.mean()) < 1e-9


# ---------------------------------------------------------------------------
# 9. histogram 分位误差不超过一个 bin
# ---------------------------------------------------------------------------
def test_histogram_quantile_within_one_bin():
    bins = 1000
    x = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    st = diag.StreamingStats(0.0, 1.0, bins, False, 'cpu')
    st.update(x)
    binw = 1.0 / bins
    for q in (0.5, 0.9, 0.95, 0.99):
        est = st.quantile(q)
        exact = float(torch.quantile(x, q))
        assert est is not None
        assert abs(est - exact) <= binw + 1e-9, (q, est, exact)


# ---------------------------------------------------------------------------
# 10. relative-delta 用总 numerator/denominator 计算
# ---------------------------------------------------------------------------
def test_relative_delta_total_ratio():
    rd = diag.RelativeDelta('cpu', 1000)
    rd.add(torch.tensor(2.0, dtype=torch.float64), torch.tensor(8.0, dtype=torch.float64))
    rd.add(torch.tensor(1.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    val = rd.value()
    assert abs(val - 3.0 / 9.0) < 1e-6          # 总量比
    assert abs(val - 0.625) > 0.1               # 不是每图 ratio 的均值
    d = rd.to_dict()
    assert d['relative_delta'] is not None and d['per_image']['count'] == 2
    # 空 -> null
    empty = diag.RelativeDelta('cpu', 100)
    assert empty.value() is None


# ---------------------------------------------------------------------------
# 11. domain/source 聚合不串组
# ---------------------------------------------------------------------------
def test_group_aggregation_no_cross_contamination():
    store = {}
    a = diag.get_group_acc(store, 'ODC', ('gate',), 'cpu', 100)
    b = diag.get_group_acc(store, 'GDC', ('gate',), 'cpu', 100)
    assert a is not b
    a.stats['gate'].update(torch.tensor([0.8, 0.9]))
    assert a.stats['gate'].count == 2
    assert b.stats['gate'].count == 0 and b.stats['gate'].mean() is None
    a2 = diag.get_group_acc(store, 'ODC', ('gate',), 'cpu', 100)
    assert a2 is a and a2.stats['gate'].count == 2


# ---------------------------------------------------------------------------
# 12. 反事实 gate 范围为 [0,1]
# ---------------------------------------------------------------------------
def test_counterfactual_gate_range():
    torch.manual_seed(0)
    for _ in range(20):
        cs = torch.rand(2, 1, 8, 8)
        lc = torch.rand(2, 1, 8, 8)
        dmax = torch.rand(2, 1, 8, 8)
        dmin = torch.rand(2, 1, 8, 8)
        gate_raw = torch.rand(2, 1, 8, 8)
        cand = diag.counterfactual_structure_v11(cs, lc, dmax, dmin)
        cfg = diag.counterfactual_gate_v11(gate_raw, cand)
        assert float(cand.min()) >= 0.0 and float(cand.max()) <= 1.0
        assert float(cfg.min()) >= 0.0 and float(cfg.max()) <= 1.0


# ---------------------------------------------------------------------------
# 13. JSON/CSV 不产生 NaN/Infinity
# ---------------------------------------------------------------------------
def test_json_csv_no_nan_inf():
    obj = {'a': float('nan'), 'b': float('inf'), 'c': 1.5,
           'd': [float('nan'), 2.0], 'e': None, 'f': -float('inf')}
    safe = diag.to_json_safe(obj)
    assert safe['a'] is None and safe['b'] is None and safe['f'] is None
    assert safe['c'] == 1.5 and safe['d'][0] is None and safe['d'][1] == 2.0
    s = json.dumps(safe, allow_nan=False)       # 不抛异常
    assert 'NaN' not in s and 'Infinity' not in s
    assert diag.csv_cell(float('nan')) == '' and diag.csv_cell(float('inf')) == ''
    assert diag.csv_cell(None) == '' and diag.csv_cell(1.5) == '1.5'
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'x.json')
        diag.dump_json_strict(obj, p)
        loaded = json.load(open(p, encoding='utf-8'))
        assert loaded['a'] is None and loaded['c'] == 1.5


# ---------------------------------------------------------------------------
# 14. 部分运行 metadata 标记
# ---------------------------------------------------------------------------
def test_partial_run_flag():
    assert diag.is_partial_run(16) is True
    assert diag.is_partial_run(1) is True
    assert diag.is_partial_run(None) is False
    assert diag.is_partial_run(0) is False


# ---------------------------------------------------------------------------
# 15.（附加）复算量与真实 SRFF block 的 _core 一致，且不触发漂移失败
# ---------------------------------------------------------------------------
def test_compute_batch_quantities_matches_core():
    torch.manual_seed(0)
    blk = srff.SelectiveRobustFrequencyFusion(channels=8)
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    maps = blk._core(high, low)
    qb, adf, alf = diag.compute_batch_quantities(blk, maps, low)  # 不一致会 SystemExit
    global_q = ('pre_global_gate', 'global_score', 'global_gate')
    v1_quantities = tuple(q for q in diag.ALL_QUANTITIES if q not in global_q)
    for q in v1_quantities:
        assert q in qb, q
        assert tuple(qb[q].shape) == (2, 24, 24)
    for q in global_q:                      # V1 checkpoint 无全局门量
        assert q not in qb
    assert torch.allclose(qb['gate'], maps['gate'][:, 0], atol=1e-6)
    assert torch.allclose(qb['structure'], maps['structure'][:, 0], atol=1e-6)
    assert float(qb['counterfactual_gate_v11'].min()) >= -1e-6
    assert float(qb['counterfactual_gate_v11'].max()) <= 1.0 + 1e-6
    assert tuple(adf.shape) == (2, 8, 24, 24) and tuple(alf.shape) == (2, 8, 24, 24)


# ---------------------------------------------------------------------------
# 16.（附加）V1.1 block 的 _core 全局门量被正确复算与展开
# ---------------------------------------------------------------------------
def test_compute_batch_quantities_v11_global_fields():
    SRFFV11 = _load_v11_class()
    torch.manual_seed(0)
    blk = SRFFV11(channels=8)
    high = torch.randn(2, 8, 12, 12)
    low = torch.randn(2, 8, 24, 24)
    maps = blk._core(high, low)
    qb, adf, alf = diag.compute_batch_quantities(blk, maps, low)
    for q in ('pre_global_gate', 'global_score', 'global_gate'):
        assert q in qb and tuple(qb[q].shape) == (2, 24, 24)
    gg = qb['global_gate']
    # global_gate 逐样本常量展开（每图内 HxW 均相同）
    assert torch.allclose(gg[0], gg[0].flatten()[0].expand_as(gg[0]), atol=0.0)
    # 最终 gate == pre_global_gate * global_gate
    assert torch.allclose(qb['gate'], qb['pre_global_gate'] * gg, atol=1e-5)
    assert float(gg.min()) >= 0.0 and float(gg.max()) <= 1.0


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
