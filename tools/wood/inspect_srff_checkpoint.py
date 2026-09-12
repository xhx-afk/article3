"""SRFF-V1 / V1.1 训练检查点诊断工具（只读）。

加载一个**真实训练好的 SRFF 检查点**（V1 两个 block，或 V1.1 单 active block），在**真实
验证集**上前向，用 forward_pre_hook 捕获各 active SRFF block 的 (high, low)，在诊断路径中
复算 ``block._core`` 的全部门控/证据/路由量（V1.1 额外含 pre_global_gate/global_score/
global_gate），并按 block × domain × region/class/source 做**流式**统计，输出 JSON + 3 份 CSV。

严格只读：不改模型、不改训练/推理生产路径、不改数据。诊断计算只在独立脚本内进行，
不进入 forward。设计依据见 ``SRFF-V1-训练检查点诊断-实现文档.md``。

用法示例::

    python tools/wood/inspect_srff_checkpoint.py \
      -c configs/deim_dfine/custom/coated_wood_fast_s_srff_v1.yml \
      -r /abs/path/best_stg2.pth \
      -o /abs/path/trained_gate_diagnostics \
      -d cuda --seed 0 --hist-bins 1000 --bg-margin 0.10 --print-freq 50

本模块的**纯工具层**（文件名解析、框->feature 掩码、流式统计器、反事实结构项、
JSON/CSV 安全写出）不依赖 engine，可被 ``tests/test_srff_checkpoint_diagnostics.py``
在仅安装 torch 的环境下单元测试；engine 相关导入全部延迟到 main() 内部。
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用 split_val_domains.parse_file_name（避免文件名解析逻辑漂移）。用 importlib 从同级
# 文件加载，使本模块即便被单测以文件方式加载也能工作，不依赖 tools.wood 是否为包。
_SVD_PATH = Path(__file__).resolve().parent / 'split_val_domains.py'
_svd_spec = importlib.util.spec_from_file_location('_srff_split_val_domains', _SVD_PATH)
_svd = importlib.util.module_from_spec(_svd_spec)
_svd_spec.loader.exec_module(_svd)
parse_file_name = _svd.parse_file_name

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
EPS = 1e-6

DOMAINS = ('ODC', 'LDC', 'DDC', 'GDC', 'PDC')
CLEAN_GROUP = ('ODC', 'LDC', 'DDC')
ROBUST_GROUP = ('GDC', 'PDC')
GROUP_DEFS = {'clean_group': CLEAN_GROUP, 'robust_group': ROBUST_GROUP}
REGIONS = ('all', 'foreground', 'background')

# [0,1] 线性直方图量（含 V1.1 全局门量；V1 checkpoint 下这些量为空→输出 null）
Q_UNIT = (
    'gate', 'gate_raw', 'structure', 'cross_scale', 'cross_persistence',
    'gaussian_weight', 'trimmed_weight',
    'local_coherence', 'horizontal_coherence', 'vertical_coherence',
    'directional_max', 'directional_min', 'directional_anisotropy',
    'candidate_structure_v11', 'counterfactual_gate_v11',
    'pre_global_gate', 'global_score', 'global_gate',
)
# [-1,1] 线性（gate_reduction 可正可负）
Q_SIGNED = ('gate_reduction_v11',)
# 非负、上界不定：对数 bins
Q_UNBOUNDED = ('noise_l', 'noise_t', 'smooth_gap', 'abs_delta')
ALL_QUANTITIES = tuple(Q_UNIT) + tuple(Q_SIGNED) + tuple(Q_UNBOUNDED)

# source 维度精简量（文档 4.9：只保留必要字段，避免文件过大）
Q_SOURCE = ('gate', 'structure', 'gaussian_weight', 'trimmed_weight',
            'candidate_structure_v11', 'counterfactual_gate_v11',
            'pre_global_gate', 'global_gate')

LOG_LO = 1e-6
LOG_HI = 1e2
LOG_BINS = 512

# 两个 SRFF block 的诊断名（按 named_modules 顺序）；顺序不可靠时回退为 block0/block1。
BLOCK_DIAG_NAMES = ('block0_p5_to_p4', 'block1_p4_to_p3')
BLOCK_FALLBACK_NAMES = ('block0', 'block1')


# ---------------------------------------------------------------------------
# 身份解析
# ---------------------------------------------------------------------------
def resolve_identity(file_name):
    """从 file_name 解析 (source_id, domain, index)。域非法或无法解析时抛 ValueError。"""
    parsed = parse_file_name(file_name)
    if parsed is None:
        raise ValueError(f'无法解析 file_name: {file_name!r}')
    source, domain, index = parsed
    if domain not in DOMAINS:
        raise ValueError(f'未知域 {domain!r}（file_name={file_name!r}），允许集={DOMAINS}')
    return source, domain, index


def is_partial_run(max_images):
    """--max-images 被设置（正整数）时为部分运行。"""
    return bool(max_images) and int(max_images) > 0


# ---------------------------------------------------------------------------
# 框 -> feature 掩码
# ---------------------------------------------------------------------------
def box_to_feature_mask(box_xywh, img_w, img_h, feat_w, feat_h, device='cpu'):
    """把原始像素 xywh 框映射到 feature 网格 bool 掩码（floor/ceil + clamp）。

    合法非空框至少覆盖 1 个 feature pixel；非法（宽高<=0 或尺寸非法）返回 None。
    掩码建在 device 上，以便与 GPU 特征张量做布尔索引而不发生设备不匹配。
    """
    if not (img_w > 0 and img_h > 0 and feat_w > 0 and feat_h > 0):
        return None
    x, y, w, h = (float(v) for v in box_xywh[:4])
    if w <= 0 or h <= 0:
        return None
    x0 = math.floor(x / img_w * feat_w)
    y0 = math.floor(y / img_h * feat_h)
    x1 = math.ceil((x + w) / img_w * feat_w)
    y1 = math.ceil((y + h) / img_h * feat_h)
    # clamp 并保证至少 1 pixel
    x0 = max(0, min(x0, feat_w - 1))
    y0 = max(0, min(y0, feat_h - 1))
    x1 = max(x0 + 1, min(x1, feat_w))
    y1 = max(y0 + 1, min(y1, feat_h))
    mask = torch.zeros(feat_h, feat_w, dtype=torch.bool, device=device)
    mask[y0:y1, x0:x1] = True
    return mask


def build_region_masks(anns, img_w, img_h, feat_w, feat_h, cat_id_to_name, bg_margin, device='cpu'):
    """构造 {region_name: bool mask(feat_h,feat_w)}。

    region 含 'all'、'foreground'、'background' 以及每个出现类别的 'class:<name>'。
    background = 所有 GT 框按宽高各向外扩张 bg_margin 后并集的补集。
    忽略 iscrowd=1。返回 (masks, ignored_crowd, invalid_box)。
    """
    fg = torch.zeros(feat_h, feat_w, dtype=torch.bool, device=device)
    fg_dilated = torch.zeros(feat_h, feat_w, dtype=torch.bool, device=device)
    class_masks = {}
    ignored_crowd = 0
    invalid_box = 0

    for ann in anns:
        if int(ann.get('iscrowd', 0)) == 1:
            ignored_crowd += 1
            continue
        bbox = ann.get('bbox')
        if bbox is None or len(bbox) < 4:
            invalid_box += 1
            continue
        x, y, w, h = (float(v) for v in bbox[:4])
        m = box_to_feature_mask((x, y, w, h), img_w, img_h, feat_w, feat_h, device)
        if m is None:
            invalid_box += 1
            continue
        fg |= m
        cname = cat_id_to_name.get(ann.get('category_id'), str(ann.get('category_id')))
        cname = str(cname)
        if cname not in class_masks:
            class_masks[cname] = torch.zeros(feat_h, feat_w, dtype=torch.bool, device=device)
        class_masks[cname] |= m
        md = box_to_feature_mask(
            (x - bg_margin * w, y - bg_margin * h, w * (1.0 + 2.0 * bg_margin),
             h * (1.0 + 2.0 * bg_margin)),
            img_w, img_h, feat_w, feat_h, device)
        if md is not None:
            fg_dilated |= md

    masks = {
        'all': torch.ones(feat_h, feat_w, dtype=torch.bool, device=device),
        'foreground': fg,
        'background': ~fg_dilated,
    }
    for cname, cm in class_masks.items():
        masks['class:' + cname] = cm
    return masks, ignored_crowd, invalid_box


# ---------------------------------------------------------------------------
# 反事实结构项（仅诊断用，禁止进入模型 forward）
# ---------------------------------------------------------------------------
def counterfactual_structure_v11(cross_scale, local_coherence, directional_max, directional_min):
    """候选方向连续性结构分数（文档 4.7）。返回 clamp 到 [0,1] 的张量。"""
    p = 1.0 - cross_scale
    cand = 1.0 - (1.0 - p * local_coherence) * (1.0 - directional_max * (1.0 - directional_min))
    return cand.clamp(0.0, 1.0)


def counterfactual_gate_v11(gate_raw, candidate_structure):
    """反事实门控 = gate_raw * (1 - candidate_structure)，范围 [0,1]。"""
    return gate_raw * (1.0 - candidate_structure)


# ---------------------------------------------------------------------------
# 流式统计器
# ---------------------------------------------------------------------------
def quantity_bin_config(qname, hist_bins):
    """返回 (lo, hi, bins, log) 直方图配置。"""
    if qname in Q_UNIT:
        return (0.0, 1.0, int(hist_bins), False)
    if qname in Q_SIGNED:
        return (-1.0, 1.0, int(hist_bins), False)
    return (LOG_LO, LOG_HI, LOG_BINS, True)  # Q_UNBOUNDED


class StreamingStats:
    """可合并的流式统计器：count/sum/sum_sq/min/max + 固定区间直方图（近似分位数）。

    所有累加在 device 上用 float64，避免逐值 CPU 传输；只在 to_dict/quantile 时取标量。
    空统计（count=0）的 value 一律为 None，绝不产生 NaN/Infinity。
    """

    def __init__(self, lo, hi, bins, log=False, device='cpu'):
        self.lo = float(lo)
        self.hi = float(hi)
        self.bins = int(bins)
        self.log = bool(log)
        self.device = device
        if self.log and self.lo <= 0:
            raise ValueError('log bins 要求 lo>0')
        self.count = 0
        self.sum = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros((), dtype=torch.float64, device=device)
        self.min = None
        self.max = None
        self.hist = torch.zeros(self.bins, dtype=torch.float64, device=device)
        self.under = torch.zeros((), dtype=torch.float64, device=device)
        self.over = torch.zeros((), dtype=torch.float64, device=device)
        if self.log:
            self._log_lo = math.log(self.lo)
            self._log_hi = math.log(self.hi)

    def _bin_index(self, v):
        if self.log:
            vc = v.clamp(min=self.lo)
            frac = (torch.log(vc) - self._log_lo) / (self._log_hi - self._log_lo)
        else:
            frac = (v - self.lo) / (self.hi - self.lo)
        return torch.floor(frac * self.bins).long()

    def update(self, v):
        if v is None:
            return
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        v = v.detach().reshape(-1).to(torch.float64).to(self.device)
        n = int(v.numel())
        if n == 0:
            return
        self.count += n
        self.sum += v.sum()
        self.sum_sq += (v * v).sum()
        vmin = v.min()
        vmax = v.max()
        self.min = vmin if self.min is None else torch.minimum(self.min, vmin)
        self.max = vmax if self.max is None else torch.maximum(self.max, vmax)
        self.under += (v < self.lo).sum().to(torch.float64)
        self.over += (v > self.hi).sum().to(torch.float64)
        in_range = (v >= self.lo) & (v <= self.hi)
        if bool(in_range.any()):
            idx = self._bin_index(v[in_range]).clamp(0, self.bins - 1)
            self.hist += torch.bincount(idx, minlength=self.bins).to(torch.float64)

    def merge(self, other):
        key = (self.lo, self.hi, self.bins, self.log)
        if key != (other.lo, other.hi, other.bins, other.log):
            raise ValueError(f'merge 配置不一致: {key} vs {(other.lo, other.hi, other.bins, other.log)}')
        self.count += other.count
        self.sum += other.sum.to(self.device)
        self.sum_sq += other.sum_sq.to(self.device)
        self.hist += other.hist.to(self.device)
        self.under += other.under.to(self.device)
        self.over += other.over.to(self.device)
        if other.min is not None:
            om = other.min.to(self.device)
            self.min = om if self.min is None else torch.minimum(self.min, om)
        if other.max is not None:
            om = other.max.to(self.device)
            self.max = om if self.max is None else torch.maximum(self.max, om)
        return self

    def _bin_value(self, idx):
        if self.log:
            return math.exp(self._log_lo + (idx + 0.5) * (self._log_hi - self._log_lo) / self.bins)
        return self.lo + (idx + 0.5) * (self.hi - self.lo) / self.bins

    def mean(self):
        return None if self.count == 0 else float(self.sum) / self.count

    def std(self):
        if self.count == 0:
            return None
        m = float(self.sum) / self.count
        var = float(self.sum_sq) / self.count - m * m
        return math.sqrt(var) if var > 0.0 else 0.0

    def quantile(self, q):
        if self.count == 0:
            return None
        total = float(self.hist.sum())
        if total <= 0:
            return None
        cum = torch.cumsum(self.hist, 0)
        target = torch.tensor([q * total], dtype=cum.dtype, device=cum.device)
        idx = int(torch.searchsorted(cum, target, side='left').item())
        idx = max(0, min(idx, self.bins - 1))
        return self._bin_value(idx)

    def to_dict(self):
        base = {'lo': self.lo, 'hi': self.hi, 'bins': self.bins, 'log': self.log,
                'under': int(self.under), 'over': int(self.over)}
        if self.count == 0:
            base.update({'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None,
                         'p50': None, 'p90': None, 'p95': None, 'p99': None})
            return base
        base.update({
            'count': self.count,
            'mean': self.mean(), 'std': self.std(),
            'min': float(self.min), 'max': float(self.max),
            'p50': self.quantile(0.50), 'p90': self.quantile(0.90),
            'p95': self.quantile(0.95), 'p99': self.quantile(0.99),
        })
        return base


class RelativeDelta:
    """按总量聚合的 relative_delta = sum(abs_delta)/(sum(abs_low)+eps)，并保留每图分布。"""

    def __init__(self, device='cpu', hist_bins=1000):
        self.device = device
        self.sum_num = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_den = torch.zeros((), dtype=torch.float64, device=device)
        self.per_image = StreamingStats(0.0, 1.0, hist_bins, False, device)

    def add(self, num, den):
        num = num.detach().to(torch.float64).to(self.device) if torch.is_tensor(num) \
            else torch.tensor(float(num), dtype=torch.float64, device=self.device)
        den = den.detach().to(torch.float64).to(self.device) if torch.is_tensor(den) \
            else torch.tensor(float(den), dtype=torch.float64, device=self.device)
        self.sum_num += num
        self.sum_den += den
        self.per_image.update((num / (den + EPS)).reshape(1))

    def merge(self, other):
        self.sum_num += other.sum_num.to(self.device)
        self.sum_den += other.sum_den.to(self.device)
        self.per_image.merge(other.per_image)
        return self

    def value(self):
        den = float(self.sum_den)
        if den <= EPS:
            return None
        return float(self.sum_num) / (den + EPS)

    def to_dict(self):
        return {'relative_delta': self.value(), 'per_image': self.per_image.to_dict(),
                'sum_abs_delta': float(self.sum_num), 'sum_abs_low': float(self.sum_den)}


def make_stats(qname, hist_bins, device='cpu'):
    lo, hi, bins, log = quantity_bin_config(qname, hist_bins)
    return StreamingStats(lo, hi, bins, log, device)


# ---------------------------------------------------------------------------
# 分组累加器
# ---------------------------------------------------------------------------
class GroupAccumulator:
    """一个聚合组（如 block×domain×region）的统计容器。"""

    def __init__(self, quantities, device='cpu', hist_bins=1000):
        self.quantities = tuple(quantities)
        self.device = device
        self.hist_bins = hist_bins
        self.stats = {q: make_stats(q, hist_bins, device) for q in self.quantities}
        self.rel_delta = RelativeDelta(device, hist_bins)
        self.image_count = 0
        self.pixel_count = 0

    def add_observation(self, values_by_q, mask_count, rel_num=None, rel_den=None):
        for q in self.quantities:
            v = values_by_q.get(q)
            if v is not None:
                self.stats[q].update(v)
        self.image_count += 1
        self.pixel_count += int(mask_count)
        if rel_num is not None and rel_den is not None:
            self.rel_delta.add(rel_num, rel_den)

    def merge(self, other):
        for q in self.quantities:
            if q in other.stats:
                self.stats[q].merge(other.stats[q])
        self.rel_delta.merge(other.rel_delta)
        self.image_count += other.image_count
        self.pixel_count += other.pixel_count
        return self

    def mean_of(self, q):
        st = self.stats.get(q)
        return None if st is None else st.mean()

    def to_dict(self):
        return {
            'image_count': self.image_count,
            'pixel_count': self.pixel_count,
            'quantities': {q: self.stats[q].to_dict() for q in self.quantities},
            'relative_delta': self.rel_delta.to_dict(),
        }


def get_group_acc(store, key, quantities, device, hist_bins):
    """从嵌套 dict 取（或创建）GroupAccumulator；不同 key 互不串组。"""
    acc = store.get(key)
    if acc is None:
        acc = GroupAccumulator(quantities, device, hist_bins)
        store[key] = acc
    return acc


# ---------------------------------------------------------------------------
# JSON / CSV 安全写出（禁止 NaN/Infinity）
# ---------------------------------------------------------------------------
def to_json_safe(obj):
    """递归把非有限 float 转为 None，保证严格 JSON 序列化不抛错、不产出 NaN/Inf。"""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


def dump_json_strict(obj, path):
    safe = to_json_safe(obj)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(safe, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def csv_cell(x):
    """CSV 单元格：None/非有限 -> 空串；其余原样字符串化（绝不写 nan/inf）。"""
    if x is None:
        return ''
    if isinstance(x, float):
        return '' if not math.isfinite(x) else repr(x)
    return str(x)


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([csv_cell(c) for c in row])


# ---------------------------------------------------------------------------
# 全量审计期望值（编码自 audited coated-wood val：2760 图 / 5 域各 552 / 69 source / 每 source 8）
# 仅用于正式全量运行的守卫；partial 运行不做全量断言。
# ---------------------------------------------------------------------------
EXPECTED_TOTAL_IMAGES = 2760
EXPECTED_PER_DOMAIN_IMAGES = 552
EXPECTED_PER_DOMAIN_SOURCES = 69
EXPECTED_PER_SOURCE_IMAGES = 8

HIGH_FREQ_CLASSES = ('crack', 'scratch')
LOW_FREQ_CLASSES = ('blister', 'hole')

QUANT_ORDER = ALL_QUANTITIES


# ---------------------------------------------------------------------------
# 元数据辅助
# ---------------------------------------------------------------------------
def _git_info():
    try:
        c = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', 'status', '--porcelain'], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return {'commit': (c.stdout.strip() or None) if c.returncode == 0 else None,
                'dirty': (bool(s.stdout.strip()) if s.returncode == 0 else None)}
    except Exception:
        return {'commit': None, 'dirty': None}


def _file_sha256(path, chunk=1 << 20):
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for blk in iter(lambda: f.read(chunk), b''):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 批量量计算（复用 block._coh/_avg_pool，避免公式漂移）
# ---------------------------------------------------------------------------
def compute_batch_quantities(block, maps, low):
    """从 _core 输出复算全部每像素诊断量。返回 (qbatch{q:(B,H,W)}, abs_delta_full, abs_low_full)。"""
    eps = block.eps
    gate = maps['gate']
    gate_raw = maps['gate_raw']
    structure = maps['structure']
    cross_scale = maps['cross_scale']
    cross_persistence = 1.0 - cross_scale
    gw = maps['gaussian_weight']
    tw = maps['trimmed_weight']
    n_l = maps['noise_l']
    n_t = maps['noise_t']

    local_coh = block._coh(n_l, block._avg_pool(n_l, 3, 3))
    h_coh = block._coh(n_l, block._avg_pool(n_l, 1, 3))
    v_coh = block._coh(n_l, block._avg_pool(n_l, 3, 1))
    dir_max = torch.maximum(h_coh, v_coh)
    dir_min = torch.minimum(h_coh, v_coh)
    dir_aniso = (h_coh - v_coh).abs() / (h_coh + v_coh + eps)

    # 一致性校验：复算 structure 必须与 _core 输出一致，否则说明公式漂移，立即失败。
    s_recomp = (cross_persistence * torch.maximum(local_coh, dir_max)).clamp(0.0, 1.0)
    if not torch.allclose(s_recomp, structure, atol=1e-4, rtol=1e-3):
        raise SystemExit('复算 structure 与 _core 输出不一致（公式漂移），中止诊断')

    cand = counterfactual_structure_v11(cross_scale, local_coh, dir_max, dir_min)
    cf_gate = counterfactual_gate_v11(gate_raw, cand)
    gate_reduction = gate - cf_gate

    f_rob = gw * maps['gaussian_low'] + tw * maps['trimmed_low']       # (B,C,H,W)
    smooth_gap = (f_rob - low).abs().mean(dim=1, keepdim=True)         # (B,1,H,W)
    abs_delta_full = (maps['low_out'] - low).abs()                     # (B,C,H,W)
    abs_delta = abs_delta_full.mean(dim=1, keepdim=True)               # (B,1,H,W)
    abs_low_full = low.abs()                                           # (B,C,H,W)

    qbatch = {
        'gate': gate[:, 0], 'gate_raw': gate_raw[:, 0], 'structure': structure[:, 0],
        'cross_scale': cross_scale[:, 0], 'cross_persistence': cross_persistence[:, 0],
        'gaussian_weight': gw[:, 0], 'trimmed_weight': tw[:, 0],
        'local_coherence': local_coh[:, 0], 'horizontal_coherence': h_coh[:, 0],
        'vertical_coherence': v_coh[:, 0], 'directional_max': dir_max[:, 0],
        'directional_min': dir_min[:, 0], 'directional_anisotropy': dir_aniso[:, 0],
        'candidate_structure_v11': cand[:, 0], 'counterfactual_gate_v11': cf_gate[:, 0],
        'gate_reduction_v11': gate_reduction[:, 0],
        'noise_l': n_l[:, 0], 'noise_t': n_t[:, 0],
        'smooth_gap': smooth_gap[:, 0], 'abs_delta': abs_delta[:, 0],
    }
    # V1.1 额外量：pre_global_gate(V1 原 gate) / global_score(q) / global_gate(m)。
    # global_score、global_gate 是逐样本 [B,1,1,1] 常量，展开到 (B,H,W) 复用同一套区域聚合；
    # V1 checkpoint 无这些键，qbatch 不含它们，累加时按 present 过滤 → 输出 null（不破坏 V1 读取）。
    if 'global_gate' in maps:
        bsz, _, fh, fw = low.shape
        qbatch['pre_global_gate'] = maps['pre_global_gate'][:, 0]
        qbatch['global_score'] = maps['global_score'].reshape(bsz, 1, 1).expand(bsz, fh, fw)
        qbatch['global_gate'] = maps['global_gate'].reshape(bsz, 1, 1).expand(bsz, fh, fw)
    return qbatch, abs_delta_full, abs_low_full


def _update_acc(acc, q_stacked, quant_order, mask, abs_delta_full_b, abs_low_full_b):
    """把一个 region mask 内的像素量累加进 GroupAccumulator；空 mask 直接跳过（不产生 NaN）。

    用 vals.shape[1] 取像素数（来自 shape，不触发 GPU 同步）。
    """
    vals = q_stacked[:, mask]                     # (Nq, n)；空 mask -> n=0
    n = vals.shape[1]
    if n == 0:
        return
    for qi, q in enumerate(quant_order):
        acc.stats[q].update(vals[qi])
    num = abs_delta_full_b[:, mask].sum()
    den = abs_low_full_b[:, mask].sum()
    acc.rel_delta.add(num, den)
    acc.image_count += 1
    acc.pixel_count += n


def _merge_group_accs(accs, quantities, device, hist_bins):
    merged = None
    for acc in accs:
        if acc is None:
            continue
        if merged is None:
            merged = GroupAccumulator(quantities, device, hist_bins)
        merged.merge(acc)
    return merged


# ---------------------------------------------------------------------------
# 对比量（文档 4.10，纯描述性，无伪阈值）
# ---------------------------------------------------------------------------
def _safe_diff(a, b):
    return None if (a is None or b is None) else a - b


def _safe_ratio(a, b, warnings, label):
    if a is None or b is None:
        return None
    if abs(b) < EPS:
        warnings.append(f'{label}: 分母接近 0，结果记为 null')
        return None
    return a / b


def build_comparisons(acc_region, acc_class, block_names, device, hist_bins, warnings):
    comps = {}
    for i, bname in enumerate(block_names):
        reg = acc_region.get(i, {})

        def _all(doms, region):
            return _merge_group_accs([reg.get(d, {}).get(region) for d in doms], QUANT_ORDER, device, hist_bins)

        clean_all = _all(CLEAN_GROUP, 'all')
        robust_all = _all(ROBUST_GROUP, 'all')
        alldom_all = _all(DOMAINS, 'all')
        alldom_fg = _all(DOMAINS, 'foreground')
        alldom_bg = _all(DOMAINS, 'background')
        hf = _merge_group_accs([acc_class.get(i, {}).get(d, {}).get(c)
                                for d in DOMAINS for c in HIGH_FREQ_CLASSES], QUANT_ORDER, device, hist_bins)
        lf = _merge_group_accs([acc_class.get(i, {}).get(d, {}).get(c)
                                for d in DOMAINS for c in LOW_FREQ_CLASSES], QUANT_ORDER, device, hist_bins)

        clean_gate = clean_all.mean_of('gate') if clean_all else None
        robust_gate = robust_all.mean_of('gate') if robust_all else None
        clean_pre = clean_all.mean_of('pre_global_gate') if clean_all else None
        robust_pre = robust_all.mean_of('pre_global_gate') if robust_all else None
        clean_gg = clean_all.mean_of('global_gate') if clean_all else None
        robust_gg = robust_all.mean_of('global_gate') if robust_all else None
        comps[bname] = {
            'domain_gate_diff_robust_minus_clean': _safe_diff(robust_gate, clean_gate),
            'domain_gate_ratio_robust_over_clean': _safe_ratio(robust_gate, clean_gate, warnings, f'{bname}.domain_gate_ratio'),
            'foreground_protection_diff_bg_minus_fg': _safe_diff(
                alldom_bg.mean_of('gate') if alldom_bg else None,
                alldom_fg.mean_of('gate') if alldom_fg else None),
            'candidate_protection_retention_cfovercur': _safe_ratio(
                alldom_all.mean_of('counterfactual_gate_v11') if alldom_all else None,
                alldom_all.mean_of('gate') if alldom_all else None,
                warnings, f'{bname}.candidate_retention'),
            'gaussian_specialization_gdc_minus_pdc': _safe_diff(
                reg.get('GDC', {}).get('all').mean_of('gaussian_weight') if reg.get('GDC', {}).get('all') else None,
                reg.get('PDC', {}).get('all').mean_of('gaussian_weight') if reg.get('PDC', {}).get('all') else None),
            'trimmed_specialization_pdc_minus_gdc': _safe_diff(
                reg.get('PDC', {}).get('all').mean_of('trimmed_weight') if reg.get('PDC', {}).get('all') else None,
                reg.get('GDC', {}).get('all').mean_of('trimmed_weight') if reg.get('GDC', {}).get('all') else None),
            'highfreq_class_gate_minus_lowfreq': _safe_diff(
                hf.mean_of('gate') if hf else None, lf.mean_of('gate') if lf else None),
            # V1.1 新增对比量；V1 checkpoint 下 pre/gg 为 null → 结果为 null（不破坏旧读取）
            'global_gate_robust_minus_clean': _safe_diff(robust_gg, clean_gg),
            'final_gate_robust_over_clean': _safe_ratio(robust_gate, clean_gate, warnings, f'{bname}.final_gate_ratio'),
            'robust_gate_retention_final_over_pre': _safe_ratio(robust_gate, robust_pre, warnings, f'{bname}.robust_retention'),
            'clean_gate_retention_final_over_pre': _safe_ratio(clean_gate, clean_pre, warnings, f'{bname}.clean_retention'),
        }
    return comps


# ---------------------------------------------------------------------------
# 输出写出
# ---------------------------------------------------------------------------
DOMAIN_CSV_HEADER = ['block', 'module_name', 'domain', 'region', 'image_count', 'pixel_count',
                     'gate_mean', 'gate_p95', 'gate_p99', 'gate_raw_mean', 'structure_mean',
                     'cross_persistence_mean', 'gaussian_weight_mean', 'trimmed_weight_mean',
                     'relative_delta', 'candidate_structure_v11_mean', 'counterfactual_gate_v11_mean',
                     'pre_global_gate_mean', 'global_score_mean', 'global_gate_mean']
CLASS_CSV_HEADER = ['block', 'module_name', 'domain', 'class_name', 'image_count', 'pixel_count',
                    'gate_mean', 'gate_p95', 'structure_mean', 'local_coherence_mean',
                    'horizontal_coherence_mean', 'vertical_coherence_mean',
                    'directional_anisotropy_mean', 'relative_delta',
                    'candidate_structure_v11_mean', 'counterfactual_gate_v11_mean',
                    'pre_global_gate_mean', 'global_gate_mean']
SOURCE_CSV_HEADER = ['block', 'module_name', 'domain', 'source_id', 'image_count',
                     'gate_mean', 'gate_p95', 'structure_mean', 'gaussian_weight_mean',
                     'trimmed_weight_mean', 'relative_delta', 'counterfactual_gate_v11_mean',
                     'global_gate_mean']


def _m(acc, q):
    return acc.stats[q].mean() if (acc and q in acc.stats) else None


def _q(acc, q, p):
    return acc.stats[q].quantile(p) if (acc and q in acc.stats) else None


def _rd(acc):
    return acc.rel_delta.value() if acc else None


def write_outputs(out_dir, metadata, acc_region, acc_class, acc_source, acc_group,
                  comparisons, block_names, warnings):
    os.makedirs(out_dir, exist_ok=True)
    groups = {'by_block_domain_region': {}, 'by_block_domain_class': {},
              'by_block_domain_source': {}, 'by_block_domain_group_region': {}}
    for i, bname in enumerate(block_names):
        groups['by_block_domain_region'][bname] = {
            d: {r: acc_region[i][d][r].to_dict() for r in REGIONS if r in acc_region.get(i, {}).get(d, {})}
            for d in DOMAINS if d in acc_region.get(i, {})}
        groups['by_block_domain_class'][bname] = {
            d: {cn: a.to_dict() for cn, a in acc_class[i][d].items()}
            for d in DOMAINS if d in acc_class.get(i, {})}
        groups['by_block_domain_source'][bname] = {
            d: {s: a.to_dict() for s, a in acc_source[i][d].items()}
            for d in DOMAINS if d in acc_source.get(i, {})}
        groups['by_block_domain_group_region'][bname] = {
            g: {r: acc_group[i][g][r].to_dict() for r in REGIONS if r in acc_group.get(i, {}).get(g, {})}
            for g in GROUP_DEFS if g in acc_group.get(i, {})}

    payload = {'metadata': metadata, 'groups': groups, 'comparisons': comparisons, 'warnings': warnings}
    dump_json_strict(payload, os.path.join(out_dir, 'srff_checkpoint_diagnostics.json'))

    rows = []
    for i, bname in enumerate(block_names):
        mname = metadata['srff_blocks'][i]['module_name']
        for d in DOMAINS:
            for r in REGIONS:
                a = acc_region.get(i, {}).get(d, {}).get(r)
                if a is None:
                    continue
                rows.append([bname, mname, d, r, a.image_count, a.pixel_count,
                             _m(a, 'gate'), _q(a, 'gate', 0.95), _q(a, 'gate', 0.99),
                             _m(a, 'gate_raw'), _m(a, 'structure'), _m(a, 'cross_persistence'),
                             _m(a, 'gaussian_weight'), _m(a, 'trimmed_weight'), _rd(a),
                             _m(a, 'candidate_structure_v11'), _m(a, 'counterfactual_gate_v11'),
                             _m(a, 'pre_global_gate'), _m(a, 'global_score'), _m(a, 'global_gate')])
    write_csv(os.path.join(out_dir, 'srff_checkpoint_domain_summary.csv'), DOMAIN_CSV_HEADER, rows)
    n_domain_rows = len(rows)

    rows = []
    for i, bname in enumerate(block_names):
        mname = metadata['srff_blocks'][i]['module_name']
        for d in DOMAINS:
            for cn, a in sorted(acc_class.get(i, {}).get(d, {}).items()):
                rows.append([bname, mname, d, cn, a.image_count, a.pixel_count,
                             _m(a, 'gate'), _q(a, 'gate', 0.95), _m(a, 'structure'),
                             _m(a, 'local_coherence'), _m(a, 'horizontal_coherence'),
                             _m(a, 'vertical_coherence'), _m(a, 'directional_anisotropy'), _rd(a),
                             _m(a, 'candidate_structure_v11'), _m(a, 'counterfactual_gate_v11'),
                             _m(a, 'pre_global_gate'), _m(a, 'global_gate')])
    write_csv(os.path.join(out_dir, 'srff_checkpoint_class_summary.csv'), CLASS_CSV_HEADER, rows)
    n_class_rows = len(rows)

    rows = []
    for i, bname in enumerate(block_names):
        mname = metadata['srff_blocks'][i]['module_name']
        for d in DOMAINS:
            for s, a in sorted(acc_source.get(i, {}).get(d, {}).items()):
                rows.append([bname, mname, d, s, a.image_count,
                             _m(a, 'gate'), _q(a, 'gate', 0.95), _m(a, 'structure'),
                             _m(a, 'gaussian_weight'), _m(a, 'trimmed_weight'), _rd(a),
                             _m(a, 'counterfactual_gate_v11'), _m(a, 'global_gate')])
    write_csv(os.path.join(out_dir, 'srff_checkpoint_source_summary.csv'), SOURCE_CSV_HEADER, rows)
    n_source_rows = len(rows)

    return {'domain_rows': n_domain_rows, 'class_rows': n_class_rows, 'source_rows': n_source_rows}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _parse_block_index(module_name):
    """从 'encoder.srff_blocks.0' 解析出 block 序号 0；解析不到返回 None。"""
    parts = module_name.split('.')
    for i, p in enumerate(parts):
        if p == 'srff_blocks' and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def locate_srff_blocks(module, srff_class, expected_count):
    """定位所有 SRFF block（V1.1 是 V1 子类，isinstance 同样命中）。

    不再硬编码“必须恰好 2 个”；按配置 active levels 得到 expected_count（V1=2，V1.1[0]=1）。
    诊断名按 srff_blocks.<idx> 决定（block0=P5→P4，block1=P4→P3）。
    """
    found = [(n, m) for n, m in module.named_modules() if isinstance(m, srff_class)]
    if len(found) != expected_count:
        raise SystemExit(f'期望 {expected_count} 个 SRFF block（按配置 active levels），'
                         f'实际 {len(found)}: {[n for n, _ in found]}')
    result = []
    for name, mod in found:
        idx = _parse_block_index(name)
        if idx is not None and idx < len(BLOCK_DIAG_NAMES):
            diag = BLOCK_DIAG_NAMES[idx]
        else:
            diag = f'block{idx}' if idx is not None else name
        result.append((diag, name, mod))
    result.sort(key=lambda t: t[0])
    return result


def unwrap_coco(dataset):
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    coco = getattr(dataset, 'coco', None)
    if coco is None:
        raise SystemExit('无法从验证集获取 .coco 元数据；拒绝用文件顺序推断身份')
    return coco


def run(args):
    from engine.core import YAMLConfig, yaml_utils
    from engine.solver import TASKS
    from engine.misc import dist_utils
    from engine.deim.srff import SelectiveRobustFrequencyFusion

    if not os.path.isfile(args.resume):
        raise SystemExit(f'检查点不存在: {args.resume}')
    if not os.path.isfile(args.config):
        raise SystemExit(f'配置不存在: {args.config}')

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    exit_code = 0
    try:
        if dist_utils.get_world_size() > 1:
            raise SystemExit('inspect_srff_checkpoint 第一版只支持单进程；world_size>1 会重复统计，请单卡运行')

        updates = yaml_utils.parse_cli(args.update)
        updates['resume'] = args.resume
        if args.device is not None:
            updates['device'] = args.device
        cfg = YAMLConfig(args.config, **updates)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        he = cfg.yaml_cfg.get('HybridEncoder', {})
        if not he.get('use_srff', False):
            raise SystemExit('配置未启用 SRFF（HybridEncoder.use_srff != True）；本工具只诊断 SRFF 检查点')
        srff_version = he.get('srff_version', 'v1')
        in_ch = he.get('in_channels') or [0, 0, 0]
        num_levels = max(1, len(in_ch) - 1)
        active_levels_cfg = he.get('srff_active_levels', None)
        if active_levels_cfg is None:
            expected_count = num_levels
        else:
            expected_count = len(set(int(x) for x in active_levels_cfg))

        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        solver.eval()
        used_ema = solver.ema is not None
        module = solver.ema.module if used_ema else solver.model
        module.eval()
        device = solver.device

        blocks = locate_srff_blocks(module, SelectiveRobustFrequencyFusion, expected_count)
        nblocks = len(blocks)
        block_names = [b[0] for b in blocks]
        print(f'[srff-diag] srff_version={srff_version} active_blocks={nblocks} '
              f'checkpoint_source={"ema" if used_ema else "model"} device={device}')
        for dn, mn, _ in blocks:
            print(f'[srff-diag] {dn} -> module_name={mn}')

        dataset = solver.val_dataloader.dataset
        coco = unwrap_coco(dataset)
        cat_id_to_name = {int(c['id']): c['name'] for c in coco.dataset['categories']}
        ds_cfg = cfg.yaml_cfg['val_dataloader']['dataset']
        ann_path = getattr(dataset, 'ann_file', None) or ds_cfg.get('ann_file')
        img_folder = getattr(dataset, 'img_folder', None) or ds_cfg.get('img_folder')

        acc_region = {i: defaultdict(dict) for i in range(nblocks)}
        acc_class = {i: defaultdict(dict) for i in range(nblocks)}
        acc_source = {i: defaultdict(dict) for i in range(nblocks)}
        H = args.hist_bins
        dev = str(device)

        captured = {}
        handles = []
        for i, (dn, mn, blk) in enumerate(blocks):
            def _mk(idx):
                def _hook(_m, inputs):
                    captured[idx] = (inputs[0], inputs[1])
                return _hook
            handles.append(blk.register_forward_pre_hook(_mk(i)))

        total_images = 0
        total_batches = 0
        ignored_crowd = 0
        invalid_box = 0
        per_domain_images = defaultdict(int)
        per_domain_sources = defaultdict(set)
        per_domain_source_images = defaultdict(int)
        block_input_shapes = [None] * nblocks
        warnings = []
        partial = is_partial_run(args.max_images)

        try:
            for batch_idx, (samples, targets) in enumerate(solver.val_dataloader):
                samples = samples.to(device)
                with torch.no_grad():
                    _ = module(samples)

                n_batch = len(targets)
                # partial(--max-images) 时按图精确截断，使 num_images 恰为 max_images
                n_proc = min(n_batch, max(0, args.max_images - total_images)) if partial else n_batch
                if n_proc <= 0:
                    break

                identities = []
                for tgt in targets[:n_proc]:
                    iid = tgt['image_id']
                    img_id = int(iid.item()) if torch.is_tensor(iid) else int(iid)
                    info = coco.imgs[img_id]
                    src, dom, idx = resolve_identity(info['file_name'])
                    identities.append((img_id, src, dom, float(info['width']), float(info['height']),
                                       coco.imgToAnns.get(img_id, [])))
                    if not partial:
                        per_domain_images[dom] += 1
                        per_domain_sources[dom].add(src)
                        per_domain_source_images[(dom, src)] += 1

                for i, (dn, mn, blk) in enumerate(blocks):
                    if captured.get(i) is None:
                        raise SystemExit(f'forward 后 block{i}({dn}) 的 hook 未捕获输入')
                    high, low = captured[i]
                    if block_input_shapes[i] is None:
                        block_input_shapes[i] = {'high': list(high.shape), 'low': list(low.shape)}
                    with torch.no_grad():
                        maps = blk._core(high.detach(), low.detach())
                        qbatch, adf, alf = compute_batch_quantities(blk, maps, low)
                    B, C, feat_h, feat_w = low.shape
                    if B != n_batch:
                        raise SystemExit(f'feature batch {B} 与 targets {n_batch} 无法对齐')
                    # 只累加 qbatch 实际提供的量（V1 无全局门量）；accumulator 仍按全集创建，缺者留空→null。
                    present_q = [q for q in QUANT_ORDER if q in qbatch]
                    present_src = [q for q in Q_SOURCE if q in qbatch]

                    for b in range(n_proc):
                        img_id, src, dom, ow, oh, anns = identities[b]
                        masks, ign, inv = build_region_masks(
                            anns, ow, oh, feat_w, feat_h, cat_id_to_name, args.bg_margin,
                            device=low.device)
                        ignored_crowd += ign
                        invalid_box += inv
                        Q = torch.stack([qbatch[q][b] for q in present_q])
                        adf_b = adf[b]
                        alf_b = alf[b]

                        for region in REGIONS:
                            acc = get_group_acc(acc_region[i][dom], region, QUANT_ORDER, dev, H)
                            _update_acc(acc, Q, present_q, masks[region], adf_b, alf_b)
                        for mname, m in masks.items():
                            if not mname.startswith('class:'):
                                continue
                            cn = mname[len('class:'):]
                            acc = get_group_acc(acc_class[i][dom], cn, QUANT_ORDER, dev, H)
                            _update_acc(acc, Q, present_q, m, adf_b, alf_b)
                        Qs = torch.stack([qbatch[q][b] for q in present_src])
                        sacc = get_group_acc(acc_source[i][dom], src, Q_SOURCE, dev, H)
                        _update_acc(sacc, Qs, present_src, masks['all'], adf_b, alf_b)

                    captured[i] = None  # 释放本 batch 引用，避免显存累积

                total_images += n_proc
                total_batches += 1
                if args.print_freq and batch_idx % args.print_freq == 0:
                    print(f'[srff-diag] batch {batch_idx} processed_images={total_images}')
                if partial and total_images >= args.max_images:
                    print(f'[srff-diag] 达到 --max-images={args.max_images}，提前结束（partial_run=True）')
                    break
        finally:
            for h in handles:
                h.remove()

        acc_group = {i: defaultdict(dict) for i in range(nblocks)}
        for i in range(nblocks):
            for gname, doms in GROUP_DEFS.items():
                for region in REGIONS:
                    merged = _merge_group_accs([acc_region[i].get(d, {}).get(region) for d in doms],
                                               QUANT_ORDER, dev, H)
                    if merged is not None:
                        acc_group[i][gname][region] = merged

        audit = {
            'partial_run': partial,
            'total_images': total_images,
            'expected_total_images': EXPECTED_TOTAL_IMAGES,
            'per_domain_images': {d: per_domain_images[d] for d in DOMAINS},
            'per_domain_sources': {d: len(per_domain_sources[d]) for d in DOMAINS},
            'expected_per_domain_images': EXPECTED_PER_DOMAIN_IMAGES,
            'expected_per_domain_sources': EXPECTED_PER_DOMAIN_SOURCES,
            'expected_per_source_images': EXPECTED_PER_SOURCE_IMAGES,
            'per_source_image_counts': {f'{d}/{s}': c for (d, s), c in sorted(per_domain_source_images.items())},
        }
        audit_ok = True
        if not partial:
            bad = []
            if total_images != EXPECTED_TOTAL_IMAGES:
                bad.append(f'total_images={total_images}!={EXPECTED_TOTAL_IMAGES}')
            for d in DOMAINS:
                if per_domain_images[d] != EXPECTED_PER_DOMAIN_IMAGES:
                    bad.append(f'{d} images={per_domain_images[d]}!={EXPECTED_PER_DOMAIN_IMAGES}')
                if len(per_domain_sources[d]) != EXPECTED_PER_DOMAIN_SOURCES:
                    bad.append(f'{d} sources={len(per_domain_sources[d])}!={EXPECTED_PER_DOMAIN_SOURCES}')
            for (d, s), c in per_domain_source_images.items():
                if c != EXPECTED_PER_SOURCE_IMAGES:
                    bad.append(f'{d}/{s} images={c}!={EXPECTED_PER_SOURCE_IMAGES}')
            audit_ok = (len(bad) == 0)
            audit['audit_ok'] = audit_ok
            audit['audit_failures'] = bad
            if not audit_ok:
                warnings.extend([f'数据审计失败: {x}' for x in bad])

        git = _git_info()
        metadata = {
            'utc_time': datetime.now(timezone.utc).isoformat(),
            'git_commit': git['commit'], 'git_dirty': git['dirty'],
            'config_path': os.path.abspath(args.config),
            'checkpoint_path': os.path.abspath(args.resume),
            'checkpoint_size_bytes': os.path.getsize(args.resume),
            'checkpoint_sha256': _file_sha256(args.resume),
            'checkpoint_source': 'ema' if used_ema else 'model',
            'device': str(device), 'seed': args.seed,
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'val_img_folder': str(img_folder), 'val_ann_file': str(ann_path),
            'srff_version': srff_version, 'srff_active_levels': active_levels_cfg,
            'srff_active_block_count': nblocks,
            'num_images': total_images, 'num_batches': total_batches,
            'partial_run': partial, 'max_images': args.max_images,
            'histogram': {
                'unit': {'lo': 0.0, 'hi': 1.0, 'bins': H, 'log': False},
                'signed': {'lo': -1.0, 'hi': 1.0, 'bins': H, 'log': False},
                'unbounded': {'lo': LOG_LO, 'hi': LOG_HI, 'bins': LOG_BINS, 'log': True},
            },
            'bg_margin': args.bg_margin,
            'categories': {str(k): v for k, v in cat_id_to_name.items()},
            'srff_blocks': [{'diag_name': blocks[i][0], 'module_name': blocks[i][1],
                             'input_high_shape': block_input_shapes[i]['high'] if block_input_shapes[i] else None,
                             'input_low_shape': block_input_shapes[i]['low'] if block_input_shapes[i] else None}
                            for i in range(nblocks)],
            'domain_audit': audit,
            'ignored_crowd': ignored_crowd,
            'invalid_box_total_over_blocks': invalid_box,
        }
        if ignored_crowd > 0:
            warnings.append(f'忽略 iscrowd=1 标注 {ignored_crowd} 个')
        if invalid_box > 0:
            warnings.append(f'丢弃无效/无法映射 box {invalid_box} 个（所有 active block 合计）')

        comparisons = build_comparisons(acc_region, acc_class, block_names, dev, H, warnings)
        counts = write_outputs(args.o, metadata, acc_region, acc_class, acc_source, acc_group,
                               comparisons, block_names, warnings)
        print(f'[srff-diag] 写出完成: {args.o}  rows={counts}')
        if not partial and not audit_ok:
            print('[srff-diag] 数据审计失败，非零退出')
            exit_code = 1
    finally:
        dist_utils.cleanup()
    return exit_code


def get_args_parser():
    p = argparse.ArgumentParser(
        description='SRFF-V1/V1.1 训练检查点只读诊断：分 block/域/区域/类别/source 统计门控与证据（V1.1 额外输出全局门量）。')
    p.add_argument('-c', '--config', required=True, help='SRFF 配置 YAML（HybridEncoder.use_srff=True）')
    p.add_argument('-r', '--resume', required=True, help='训练好的 SRFF 检查点（best_stg2.pth）')
    p.add_argument('-o', '--output-dir', dest='o', required=True, help='诊断输出目录（不存在则创建，不删除已有文件）')
    p.add_argument('-d', '--device', default=None, help='cuda 或 cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hist-bins', type=int, default=1000, help='[0,1] 量直方图 bin 数（默认 1000）')
    p.add_argument('--bg-margin', type=float, default=0.10, help='background 区域 GT 框外扩比例（默认 0.10）')
    p.add_argument('--print-freq', type=int, default=50, help='进度打印间隔（batch）')
    p.add_argument('--max-images', type=int, default=None, help='仅冒烟：限制处理图数；设置后 partial_run=true')
    p.add_argument('-u', '--update', nargs='+', help='覆盖配置，如 val_dataloader.dataset.ann_file=...')
    p.add_argument('--print-method', default='builtin')
    p.add_argument('--print-rank', type=int, default=0)
    p.add_argument('--local-rank', type=int)
    return p


if __name__ == '__main__':
    sys.exit(run(get_args_parser().parse_args()))
