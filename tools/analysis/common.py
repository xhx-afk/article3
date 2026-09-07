"""Shared helpers for the tools/analysis scripts.

All analysis scripts import from here so that file-name normalization, COCO
indexing, IoU and the standalone COCO-style AP implementation exist in exactly
one place.

Conventions (see 准备阶段Agent实现文档.md §0/§2):
- source file names are zero-padded to 6 digits, augmented files to 5 digits;
  every id is normalized via norm_id() == str(int(s)).
- iou_matrix takes xywh boxes and converts to xyxy internally; box area is
  w*h (NOT the annotation `area` field, which may be a segmentation area).
- the AP implementation follows the COCO standard strictly (IoU
  linspace(0.5, 0.95, 10), recall points linspace(0, 1, 101), area=all,
  maxDets=100 truncated per image per category).
"""

import json
import os
import re
from dataclasses import dataclass

import numpy as np

# --- file-name regexes (verbatim from the handover doc §2.4) -----------------
SOURCE_RE = re.compile(r'^(\d+)_[A-Z]{3}_\d+$')
AUG_RE = re.compile(r'_([A-Z]{3})_')
NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')

# COCO evaluation constants
IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.00, 101)
MAX_DETS = 100


def norm_id(s):
    """Normalize any id-ish string: '000123' -> '123'."""
    return str(int(s))


def _stem(file_name):
    """Strip any image extension so the anchored regexes ($-terminated)
    match real file names like '000123_ODC_00001.jpg'."""
    return os.path.splitext(str(file_name))[0]


def parse_name(file_name):
    """Split '<src>_<CODE>_<direction>' into (src, code, direction); None otherwise.

    src/direction are returned normalized (leading zeros stripped).
    """
    m = NAME_RE.match(_stem(file_name))
    if not m:
        return None
    src, code, direction = m.groups()
    return norm_id(src), code, norm_id(direction)


def load_coco(path):
    """Load a COCO-format json file as-is."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def index_gt(coco):
    """Index a COCO dict.

    Returns:
        gt_by_image: {image_id(int): [ann, ...]}   (crowd/ignore anns kept,
            callers decide how to treat them)
        images:      {image_id(int): image dict}
        categories:  {cat_id(int): category dict}
    """
    gt_by_image = {}
    for ann in coco.get('annotations', []):
        gt_by_image.setdefault(int(ann['image_id']), []).append(ann)
    images = {int(img['id']): img for img in coco.get('images', [])}
    categories = {int(c['id']): c for c in coco.get('categories', [])}
    return gt_by_image, images, categories


def iou_matrix(a_xywh, b_xywh, crowd_b=None):
    """Pairwise IoU between two sets of xywh boxes. Returns (Na, Nb) array.

    crowd_b: optional per-b boolean. Where True, the pycocotools crowd
    convention applies: the union is the DETECTION's area only
    (IoU = inter / area_a), not the geometric union.
    """
    a = np.asarray(a_xywh, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_xywh, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

    lt_x = np.maximum(ax1[:, None], bx1[None, :])
    lt_y = np.maximum(ay1[:, None], by1[None, :])
    rb_x = np.minimum(ax2[:, None], bx2[None, :])
    rb_y = np.minimum(ay2[:, None], by2[None, :])

    inter = np.clip(rb_x - lt_x, 0.0, None) * np.clip(rb_y - lt_y, 0.0, None)
    area_a = np.clip(a[:, 2], 0.0, None) * np.clip(a[:, 3], 0.0, None)
    area_b = np.clip(b[:, 2], 0.0, None) * np.clip(b[:, 3], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    if crowd_b is not None:
        crowd = np.asarray(crowd_b, dtype=bool).reshape(-1)
        if crowd.any():
            union = np.where(crowd[None, :], area_a[:, None], union)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


@dataclass
class APResult:
    """AP over the 10 COCO IoU thresholds (area=all, maxDets=100)."""
    AP: float          # mean over thresholds and categories with GT
    AP50: float        # IoU=0.50 slice
    AP75: float        # IoU=0.75 slice
    per_category: dict  # {cat_id: (AP, AP50, AP75)} for categories with GT


def _is_ignored(ann):
    return int(ann.get('iscrowd', 0)) == 1 or int(ann.get('ignore', 0)) == 1


def match_image_category(gt_boxes, dt_boxes, gt_ignore=None,
                         iou_thrs=IOU_THRS, max_dets=MAX_DETS,
                         gt_iscrowd=None):
    """Greedy COCO matching for one (image, category) pair.

    dt_boxes must already be sorted by descending score; they are truncated to
    the first `max_dets` entries here.

    Ignore semantics follow pycocotools' evaluateImg EXACTLY (this used to be
    wrong and silently turned detections on ignored GT into FPs):
    - GTs are stably reordered so non-ignored come first
      (np.argsort(-gt_ignore, kind='mergesort'));
    - an already-matched NON-ignored GT is taken out; an ignored GT (incl.
      crowd) stays matchable by later detections;
    - early termination: once a detection has matched a non-ignored GT, it
      never degrades onto an ignored one (`if m>-1 and gtIg[m]==0 and
      gtIg[gind]==1: break`);
    - best match = max IoU >= threshold, ties go to the LAST candidate
      (pycocotools updates the best on every `>=`).

    Args:
        gt_ignore: optional per-GT ignore flags (1 = ignore). `None` = none.
        gt_iscrowd: optional per-GT crowd flags; crowd GTs use the pycocotools
            IoU convention (union = detection area).

    Returns:
        (tp, dt_ig), both (min(len(dt), max_dets), len(iou_thrs)) uint8:
        tp[d, t] = matched to ANY gt; dt_ig[d, t] = ignore flag of the matched
        gt (0 when unmatched). Accumulation must count dt_ig dets in NEITHER
        TP nor FP.
    """
    n_dt = min(len(dt_boxes), max_dets)
    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 4)[:n_dt]
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    T = len(iou_thrs)
    tp = np.zeros((n_dt, T), dtype=np.uint8)
    dt_ig = np.zeros((n_dt, T), dtype=np.uint8)
    if n_dt == 0 or len(gt_boxes) == 0:
        return tp, dt_ig
    if gt_ignore is None:
        gt_ignore = np.zeros(len(gt_boxes), dtype=np.int64)
    else:
        gt_ignore = np.asarray(gt_ignore, dtype=np.int64).reshape(-1)
        if len(gt_ignore) != len(gt_boxes):
            raise ValueError('gt_ignore length != number of gt boxes')
    # stable reorder: non-ignored GTs first (pycocotools: argsort(ascending))
    order = np.argsort(gt_ignore, kind='mergesort')
    gt_boxes = gt_boxes[order]
    gt_ignore = gt_ignore[order]
    if gt_iscrowd is not None:
        gt_iscrowd = np.asarray(gt_iscrowd, dtype=bool).reshape(-1)[order]
    ious = iou_matrix(dt_boxes, gt_boxes, crowd_b=gt_iscrowd)
    for t in range(T):
        thr = min(iou_thrs[t], 1 - 1e-10)
        gt_matched = np.zeros(len(gt_boxes), dtype=np.int64)
        for d in range(n_dt):
            m = -1
            best_iou = thr
            for g in range(len(gt_boxes)):
                # an already-matched non-ignored GT is taken; ignored GTs
                # (crowd) stay matchable by later detections
                if gt_matched[g] and not gt_ignore[g]:
                    continue
                # never degrade a non-ignored match onto an ignored GT
                if m > -1 and gt_ignore[m] == 0 and gt_ignore[g] == 1:
                    break
                if ious[d, g] < best_iou:
                    continue
                best_iou = ious[d, g]
                m = g
            if m == -1:
                continue
            gt_matched[m] = 1
            tp[d, t] = 1
            dt_ig[d, t] = gt_ignore[m]
    return tp, dt_ig


def _accumulate_cat(tp_cat, dt_ig_cat, score_cat, npig, rec_thrs=REC_THRS):
    """Global score-ordered accumulation for one category.

    tp_cat / dt_ig_cat: (N, T) flags for all kept detections of this category
            across images (already per-image truncated by match_image_category).
            Detections with dt_ig == 1 count in NEITHER TP nor FP (COCO
            accumulate semantics).
    score_cat: (N,) scores.
    npig: number of (non-ignored) GT for this category.
    """
    T = tp_cat.shape[1]
    ap_thr = np.zeros(T, dtype=np.float64)
    if npig <= 0:
        return ap_thr
    if len(score_cat):
        order = np.argsort(-np.asarray(score_cat, dtype=np.float64), kind='mergesort')
        tp_sorted = tp_cat[order]
        ig_sorted = dt_ig_cat[order]
    else:
        tp_sorted = tp_cat
        ig_sorted = dt_ig_cat
    tps = (tp_sorted == 1) & (ig_sorted == 0)
    fps = (tp_sorted == 0) & (ig_sorted == 0)
    n = tp_sorted.shape[0]
    tp_cum = np.cumsum(tps, axis=0).astype(np.float64)
    fp_cum = np.cumsum(fps, axis=0).astype(np.float64)
    rec = tp_cum / float(npig)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where((tp_cum + fp_cum) > 0, tp_cum / (tp_cum + fp_cum), 0.0)
    # make precision monotone non-increasing from the tail
    for i in range(n - 2, -1, -1):
        prec[i] = np.maximum(prec[i], prec[i + 1])
    for t in range(T):
        rec_t = rec[:, t]
        pi = np.searchsorted(rec_t, rec_thrs, side='left')
        vals = np.zeros(len(rec_thrs), dtype=np.float64)
        valid = pi < n
        vals[valid] = prec[pi[valid], t]
        ap_thr[t] = vals.mean()
    return ap_thr


def _gt_ignore_flag(ann, gt_subset=None):
    """COCO ignore flag: iscrowd/ignore as before, PLUS out-of-subset GTs when
    `gt_subset` (a set of annotation ids) is given. Out-of-subset GTs are
    IGNORED, never deleted -- deleting them would turn their detections into
    FPs and collapse stratified AP."""
    if int(ann.get('iscrowd', 0)) == 1 or int(ann.get('ignore', 0)) == 1:
        return 1
    if gt_subset is not None and int(ann['id']) not in gt_subset:
        return 1
    return 0


def coco_ap(gt_index, dt, images=None, categories=None,
            iou_thrs=IOU_THRS, max_dets=MAX_DETS, gt_subset=None):
    """COCO-style AP (bbox, area=all, maxDets, all categories).

    Args:
        gt_index: {image_id: [ann, ...]} from index_gt()
        dt: flat COCO detection results list
            ([{image_id, category_id, bbox(.xywh), score}, ...])
        images/categories: optional dicts from index_gt() (unused for AP math
            except category enumeration fallback).
        gt_subset: optional {annotation_id} set; GTs outside the set are marked
            ignored (they still participate in matching, consuming detections
            as pycocotools does) instead of deleted. Used for stratified AP.
    """
    if gt_subset is not None and len(gt_subset) == 0:
        raise ValueError('coco_ap: gt_subset is empty -- an empty stratified '
                         'group is a caller bug, not an AP of 0')
    # group detections per (image, category)
    dt_by_ic = {}
    for d in dt:
        dt_by_ic.setdefault((int(d['image_id']), int(d['category_id'])), []).append(d)

    # ALL GTs per (image, category), with ignore flags (out-of-subset and
    # crowd/ignore GTs stay in the matching, unlike the pre-P5 behavior)
    gt_by_ic = {}
    for image_id, anns in gt_index.items():
        for ann in anns:
            gt_by_ic.setdefault((int(image_id), int(ann['category_id'])), []).append(
                (ann, _gt_ignore_flag(ann, gt_subset)))

    cat_ids = sorted({c for (_, c) in gt_by_ic.keys()} | {c for (_, c) in dt_by_ic.keys()})
    per_category = {}
    for cat in cat_ids:
        keys = sorted({(i, c) for (i, c) in gt_by_ic if c == cat} |
                      {(i, c) for (i, c) in dt_by_ic if c == cat})
        npig = 0
        tp_parts, ig_parts, score_parts = [], [], []
        for (image_id, c) in keys:
            entries = gt_by_ic.get((image_id, c), [])
            gtb = [a['bbox'] for (a, _) in entries]
            gti = [ig for (_, ig) in entries]
            gtc = [int(a.get('iscrowd', 0)) for (a, _) in entries]
            npig += sum(1 for ig in gti if ig == 0)
            dets = dt_by_ic.get((image_id, c), [])
            dets = sorted(dets, key=lambda d: -float(d['score']))
            boxes = [d['bbox'] for d in dets]
            scores = [float(d['score']) for d in dets]
            tp, dt_ig = match_image_category(gtb, boxes, gt_ignore=gti,
                                             iou_thrs=iou_thrs, max_dets=max_dets,
                                             gt_iscrowd=gtc)
            tp_parts.append(tp)
            ig_parts.append(dt_ig)
            score_parts.append(np.asarray(scores[:len(tp)], dtype=np.float64))
        if tp_parts:
            tp_cat = np.concatenate(tp_parts, axis=0)
            ig_cat = np.concatenate(ig_parts, axis=0)
            score_cat = np.concatenate(score_parts, axis=0)
        else:
            tp_cat = np.zeros((0, len(iou_thrs)), dtype=np.uint8)
            ig_cat = np.zeros((0, len(iou_thrs)), dtype=np.uint8)
            score_cat = np.zeros((0,), dtype=np.float64)
        ap_thr = _accumulate_cat(tp_cat, ig_cat, score_cat,
                                 npig, rec_thrs=np.linspace(0.0, 1.00, 101))
        if npig > 0:
            per_category[cat] = (float(ap_thr.mean()), float(ap_thr[0]), float(ap_thr[5]))
    if not per_category:
        return APResult(0.0, 0.0, 0.0, {})
    aps = np.array([v[0] for v in per_category.values()])
    ap50s = np.array([v[1] for v in per_category.values()])
    ap75s = np.array([v[2] for v in per_category.values()])
    return APResult(float(aps.mean()), float(ap50s.mean()), float(ap75s.mean()),
                    per_category)


def group_by_source(coco):
    """{src(str): [image_id, ...]} grouping val images by the SOURCE_RE stem."""
    gt_by_image, images, _ = index_gt(coco)
    groups = {}
    for image_id, img in images.items():
        m = SOURCE_RE.match(_stem(img.get('file_name', '')))
        if not m:
            continue
        groups.setdefault(norm_id(m.group(1)), []).append(int(image_id))
    return groups


__all__ = [
    'SOURCE_RE', 'AUG_RE', 'NAME_RE',
    'IOU_THRS', 'REC_THRS', 'MAX_DETS',
    'norm_id', 'parse_name', 'load_coco', 'index_gt', 'iou_matrix',
    'APResult', 'match_image_category', 'coco_ap', 'group_by_source',
    '_gt_ignore_flag',
]
