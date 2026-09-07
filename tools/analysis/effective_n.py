#!/usr/bin/env python3
"""Effective sample size + AP noise floor + per-augmentation-code AP +
paired method comparison.

The AP implementation is self-contained and must match pycocotools exactly
(COCO standard: IoU linspace(0.5, 0.95, 10), recall points linspace(0, 1, 101),
area=all, maxDets=100 truncated per image per category, stable mergesort by
descending score, greedy matching inside each image, global accumulation per
category).

Bootstrap is cluster-aware:
- --by-source: resample original sources (correct; sources are the independent
  unit because every augmentation of a source derives from the same image);
- --by-image: resample single images (WRONG scale, printed as a contrast).

TP/FP matching is done ONCE; each bootstrap replicate only re-accumulates with
integer source weights, which is exactly equivalent to replicating each source
w_s times (ties keep adjacent order under stable sort). 1000 replicates run in
seconds.

Paired mode (--pred-b): the SAME resample weights evaluate both predictions,
so the ΔAP distribution is the only legitimate method-comparison statistic.

Self-test:
    python tools/analysis/effective_n.py --self-test-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    AUG_RE, IOU_THRS, MAX_DETS, REC_THRS, SOURCE_RE,
    APResult, coco_ap, group_by_source, index_gt, load_coco,
    match_image_category, norm_id, parse_name,
)
from common import _gt_ignore_flag  # noqa: E402

CODES = ['ODC', 'LDC', 'DDC', 'GDC', 'PDC']


# -----------------------------------------------------------------------------
# Precomputation: TP/FP flags computed once, per resampling unit
# -----------------------------------------------------------------------------

class Precomputed:
    """Per-category detection arrays sorted by descending score, plus the
    (unit, category) GT-count matrix."""

    def __init__(self):
        self.cats = []          # category ids
        self.scores = {}        # cat -> (N,) float64, descending (stable)
        self.units = {}         # cat -> (N,) int32 unit index
        self.tp = {}            # cat -> (N, 10) uint8
        self.dt_ig = {}         # cat -> (N, 10) uint8 (ignore flag of match)
        self.tp_idx = {}        # cat -> [10 arrays of TP&~ig positions]
        self.npig = None        # (n_units, n_cats) NON-ignored GT counts
        self.unit_list = []
        self.has_ignore = False # any dt_ig set? selects fast/slow AP path


def _unit_of_image(images, unit_kind):
    """Map image_id -> resampling unit key ('source' or 'image')."""
    mapping = {}
    for iid, img in images.items():
        name = os.path.splitext(str(img.get('file_name', '')))[0]
        if unit_kind == 'source':
            m = SOURCE_RE.match(name)
            if not m:
                raise SystemExit(
                    f'[effective_n] file_name {name!r} does not match SOURCE_RE; '
                    f'--by-source requires augmented file names like '
                    f'"000123_ODC_00001.jpg". Use --by-image instead?')
            mapping[iid] = norm_id(m.group(1))
        else:
            mapping[iid] = str(iid)
    return mapping


def build_precomputed(gt_index, dt, images, unit_kind, gt_subset=None):
    """Precompute TP/ignore flags once; bootstrap replicates only re-accumulate.

    gt_subset: optional {annotation_id} set. GTs outside the set are IGNORED
    (they still participate in matching and consume detections, exactly as
    pycocotools does) instead of deleted -- deleting them would turn correct
    detections into FPs and collapse stratified AP.
    """
    pc = Precomputed()
    unit_of = _unit_of_image(images, unit_kind)
    pc.unit_list = sorted(set(unit_of.values()))
    unit_index = {u: i for i, u in enumerate(pc.unit_list)}
    n_units = len(pc.unit_list)

    dt_by_ic = {}
    for d in dt:
        dt_by_ic.setdefault((int(d['image_id']), int(d['category_id'])), []).append(d)

    gt_by_ic = {}
    for iid, anns in gt_index.items():
        if iid not in unit_of:
            continue
        for ann in anns:
            # ALL anns kept (crowd/out-of-subset stay as ignore targets)
            gt_by_ic.setdefault((iid, int(ann['category_id'])), []).append(
                (ann, _gt_ignore_flag(ann, gt_subset)))

    cats = sorted({c for (_, c) in gt_by_ic} | {c for (_, c) in dt_by_ic})
    pc.cats = cats
    cat_pos = {c: i for i, c in enumerate(cats)}

    pc.npig = np.zeros((n_units, len(cats)), dtype=np.float64)
    for (iid, c), entries in gt_by_ic.items():
        pc.npig[unit_index[unit_of[iid]], cat_pos[c]] += sum(
            1 for (_, ig) in entries if ig == 0)

    for c in cats:
        rows_score, rows_unit, rows_tp, rows_ig = [], [], [], []
        for iid in sorted(images):
            dets = dt_by_ic.get((iid, c), [])
            entries = gt_by_ic.get((iid, c), [])
            gtb = [a['bbox'] for (a, _) in entries]
            gti = [ig for (_, ig) in entries]
            gtc = [int(a.get('iscrowd', 0)) for (a, _) in entries]
            dets = sorted(dets, key=lambda d: -float(d['score']))
            boxes = [d['bbox'] for d in dets]
            scores = [float(d['score']) for d in dets]
            tp, dt_ig = match_image_category(gtb, boxes, gt_ignore=gti,
                                             iou_thrs=IOU_THRS,
                                             max_dets=MAX_DETS,
                                             gt_iscrowd=gtc)
            n = tp.shape[0]
            if n == 0:
                continue
            rows_score.append(np.asarray(scores[:n], dtype=np.float64))
            rows_unit.append(np.full(n, unit_index[unit_of[iid]], dtype=np.int32))
            rows_tp.append(tp.astype(np.uint8))
            rows_ig.append(dt_ig.astype(np.uint8))
        if rows_score:
            scores = np.concatenate(rows_score)
            units = np.concatenate(rows_unit)
            tp = np.concatenate(rows_tp, axis=0)
            dt_ig = np.concatenate(rows_ig, axis=0)
        else:
            scores = np.zeros((0,), dtype=np.float64)
            units = np.zeros((0,), dtype=np.int32)
            tp = np.zeros((0, len(IOU_THRS)), dtype=np.uint8)
            dt_ig = np.zeros((0, len(IOU_THRS)), dtype=np.uint8)
        # one single stable global sort by descending score (COCO semantics)
        order = np.argsort(-scores, kind='mergesort')
        scores, units, tp, dt_ig = scores[order], units[order], tp[order], dt_ig[order]
        pc.scores[c] = scores
        pc.units[c] = units
        pc.tp[c] = tp
        pc.dt_ig[c] = dt_ig
        if dt_ig.any():
            pc.has_ignore = True
        pc.tp_idx[c] = [np.where((tp[:, t] == 1) & (dt_ig[:, t] == 0))[0]
                        for t in range(tp.shape[1])]
    return pc


def weighted_ap(pc, weights):
    """AP for one resample given integer unit weights.

    Equivalent to replicating every unit w_s times: cumsum(w*tp) over the
    fixed score-sorted order equals the cumsum of the replicated detection
    list (replicas of equal-score dets are adjacent under stable sort).

    Detections matched to IGNORED GTs (dt_ig) count in NEITHER TP nor FP
    (COCO accumulate semantics); npig counts only non-ignored GTs.
    """
    sum_thr = np.zeros(len(IOU_THRS), dtype=np.float64)
    ap50s, ap75s = [], []
    for ci, c in enumerate(pc.cats):
        npig_c = float(weights @ pc.npig[:, ci])
        if npig_c <= 0:
            continue  # category without non-ignored GT in this resample -> excluded
        tp = pc.tp[c]
        cat_thr = np.zeros(len(IOU_THRS), dtype=np.float64)
        if tp.shape[0]:
            w_cat = weights[pc.units[c]].astype(np.float64)
            if pc.has_ignore:
                # slow path (--gt-subset mode only): separate cumsums so that
                # ignored dets drop out of BOTH the TP and the FP stream
                is_tp = (tp == 1) & (pc.dt_ig[c] == 0)
                is_fp = (tp == 0) & (pc.dt_ig[c] == 0)
                tp_cum_all = np.cumsum(w_cat[:, None] * is_tp, axis=0)
                fp_cum_all = np.cumsum(w_cat[:, None] * is_fp, axis=0)
                rec_all = tp_cum_all / npig_c
                with np.errstate(divide='ignore', invalid='ignore'):
                    prec_all = np.where(tp_cum_all + fp_cum_all > 0,
                                        tp_cum_all / (tp_cum_all + fp_cum_all), 0.0)
                for t in range(len(IOU_THRS)):
                    idx = pc.tp_idx[c][t]
                    if idx.size == 0:
                        continue
                    prec = prec_all[:, t].copy()
                    # suffix-max == monotone non-increasing fill from the tail
                    prec = np.maximum.accumulate(prec[::-1])[::-1]
                    rec_tp = rec_all[idx, t]     # recall AT the TP positions
                    prec_tp = prec[idx]          # precision AT the TP positions
                    pi = np.searchsorted(rec_tp, REC_THRS, side='left')
                    vals = np.zeros(len(REC_THRS), dtype=np.float64)
                    ok = pi < idx.size
                    vals[ok] = prec_tp[pi[ok]]
                    cat_thr[t] = vals.mean()
            else:
                tot_cum = np.cumsum(w_cat)
                for t in range(len(IOU_THRS)):
                    idx = pc.tp_idx[c][t]
                    if idx.size == 0:
                        continue  # no TP at this IoU -> all recall points give 0
                    tp_cum = np.cumsum(w_cat[idx])
                    fp_cum = tot_cum[idx] - tp_cum
                    rec_tp = tp_cum / npig_c
                    # guard zero-weight TP positions (their unit was not drawn):
                    # 0/0 would produce nan and poison the suffix-max below
                    prec = np.where(tp_cum > 0, tp_cum / (tp_cum + fp_cum), 0.0)
                    # suffix-max == monotone non-increasing fill from the tail; only
                    # TP positions can carry the running max (FPs only lower it)
                    prec = np.maximum.accumulate(prec[::-1])[::-1]
                    pi = np.searchsorted(rec_tp, REC_THRS, side='left')
                    vals = np.zeros(len(REC_THRS), dtype=np.float64)
                    ok = pi < idx.size
                    vals[ok] = prec[pi[ok]]
                    cat_thr[t] = vals.mean()
        sum_thr += cat_thr
        ap50s.append(cat_thr[0])
        ap75s.append(cat_thr[5])
    if not ap50s:
        return 0.0, 0.0, 0.0
    n_cats = len(ap50s)
    return float(sum_thr.sum() / (n_cats * len(IOU_THRS))), \
        float(sum(ap50s) / n_cats), float(sum(ap75s) / n_cats)


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def _resample_weights(rng, n_units):
    draw = rng.integers(0, n_units, size=n_units)
    return np.bincount(draw, minlength=n_units).astype(np.float64)


def bootstrap_aps(pc, n_boot, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        w = _resample_weights(rng, len(pc.unit_list))
        out[i] = weighted_ap(pc, w)[0]
    return out


def bootstrap_paired(pc_a, pc_b, n_boot, seed):
    rng = np.random.default_rng(seed)
    diff = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        w = _resample_weights(rng, len(pc_a.unit_list))
        diff[i] = weighted_ap(pc_a, w)[0] - weighted_ap(pc_b, w)[0]
    return diff


def summarize(values):
    return {
        'mean': float(np.mean(values)),
        'sd': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'p2.5': float(np.percentile(values, 2.5)),
        'p97.5': float(np.percentile(values, 97.5)),
    }


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

def probe(ann_path):
    coco = load_coco(ann_path)
    _, images, _ = index_gt(coco)
    groups = group_by_source(coco)
    n_images = len(images)
    n_groups = len(groups)
    sizes = sorted(len(v) for v in groups.values())
    med = float(np.median(sizes)) if sizes else 0.0
    print(f'[probe] {ann_path}')
    print(f'[probe] images={n_images} source-groups={n_groups} '
          f'group-size min/median/max = {min(sizes) if sizes else 0}/{med}/{max(sizes) if sizes else 0}')
    sample = sorted(groups.items(), key=lambda kv: int(kv[0]))[:3]
    for src, ids in sample:
        names = [images[i].get('file_name') for i in sorted(ids)[:3]]
        print(f'[probe]   source {src}: {names}')

    # hard failures: the SOURCE_RE grouped nothing useful
    if n_groups == n_images or n_groups == 1:
        sys.exit(f'[probe] HARD FAIL: {n_groups} groups for {n_images} images -- '
                 f'the source regex grouped nothing (or everything); '
                 f'check file_name format / SOURCE_RE.')
    # soft warnings anchored on the real dataset (335/229/69/37 sources, ~40 per)
    if n_groups not in (335, 229, 69, 37):
        print(f'[probe] WARNING: group count {n_groups} is not one of the '
              f'expected anchors 335/229/69/37')
    if med != 40:
        print(f'[probe] WARNING: median group size {med} != 40')


def _load_pair(ann_path, pred_path):
    coco = load_coco(ann_path)
    gt_index, images, _ = index_gt(coco)
    with open(pred_path, 'r', encoding='utf-8') as f:
        dt = json.load(f)
    dt_ids = {int(d['image_id']) for d in dt}
    unknown = dt_ids - set(images)
    if unknown:
        sys.exit(f'[effective_n] pred {pred_path} has image_ids not in ann '
                 f'(mismatch), e.g. {sorted(unknown)[:10]}')
    return gt_index, images, dt


def run(args):
    gt_index, images, dt = _load_pair(args.ann, args.pred)

    # single AP (cross-validation entry point)
    res = coco_ap(gt_index, dt)
    print(f'[ap] AP={res.AP:.6f} AP50={res.AP50:.6f} AP75={res.AP75:.6f} '
          f'(self-contained COCO implementation, maxDets={MAX_DETS})')

    results = {
        'ann': args.ann, 'pred': args.pred,
        'AP': res.AP, 'AP50': res.AP50, 'AP75': res.AP75,
        'max_dets': MAX_DETS,
    }

    # stratified AP over instance groups (from degradation_paired --export-groups).
    # Out-of-group GTs are IGNORED (not deleted) -- see common.coco_ap.
    if args.gt_subset:
        with open(args.gt_subset, 'r', encoding='utf-8') as f:
            gs = json.load(f)
        groups = gs.get('groups') or {}
        if not groups:
            sys.exit(f'[effective_n] --gt-subset file has no groups: {args.gt_subset}')
        gt_b = images_b = dt_b = None
        if args.pred_b:
            gt_b, images_b, dt_b = _load_pair(args.ann, args.pred_b)
        unit_kind = 'image' if args.by_image else 'source'
        layers = {}
        for gname in sorted(groups):
            ids = {int(a) for a in groups[gname]}
            if not ids:
                sys.exit(f'[effective_n] gt-subset group {gname!r} is empty')
            r_a = coco_ap(gt_index, dt, gt_subset=ids)
            entry = {'n_gt': len(ids),
                     'AP_A': r_a.AP, 'AP50_A': r_a.AP50, 'AP75_A': r_a.AP75}
            line = f'[gt-subset] {gname}: n_gt={len(ids)} AP={r_a.AP:.6f}'
            if args.pred_b:
                r_b = coco_ap(gt_b, dt_b, gt_subset=ids)
                entry.update(AP_B=r_b.AP, AP50_B=r_b.AP50, AP75_B=r_b.AP75)
                line += f' AP_B={r_b.AP:.6f}'
                if args.by_source or args.by_image:
                    pc_a = build_precomputed(gt_index, dt, images, unit_kind,
                                             gt_subset=ids)
                    pc_b = build_precomputed(gt_b, dt_b, images_b, unit_kind,
                                             gt_subset=ids)
                    if pc_a.unit_list != pc_b.unit_list:
                        sys.exit('[effective_n] paired mode requires the same ann sources')
                    diff = bootstrap_paired(pc_a, pc_b, args.bootstrap, args.seed)
                    s = summarize(diff)
                    s['ci_includes_zero'] = bool(s['p2.5'] <= 0.0 <= s['p97.5'])
                    entry['paired_delta'] = s
                    line += (f' dAP={s["mean"]:.6f} '
                             f'CI=[{s["p2.5"]:.6f}, {s["p97.5"]:.6f}]'
                             + (' CI含0' if s['ci_includes_zero'] else ' CI不含0'))
            layers[gname] = entry
            print(line)
        results['gt_subset'] = {'file': args.gt_subset, 'name': gs.get('name'),
                                'unit': unit_kind, 'layers': layers}
        if args.ap_only:
            if args.out:
                _dump(args.out, results)
            return

    if args.by_code:
        code_ap = {}
        for code in CODES:
            keep_ids = set()
            for iid, img in images.items():
                m = AUG_RE.search(str(img.get('file_name', '')))
                if m and m.group(1) == code:
                    keep_ids.add(iid)
            if not keep_ids:
                continue
            sub_gt = {iid: a for iid, a in gt_index.items() if iid in keep_ids}
            sub_dt = [d for d in dt if int(d['image_id']) in keep_ids]
            r = coco_ap(sub_gt, sub_dt)
            code_ap[code] = {'AP': r.AP, 'AP50': r.AP50, 'AP75': r.AP75,
                             'images': len(keep_ids)}
            print(f'[by-code] {code}: AP={r.AP:.6f} AP50={r.AP50:.6f} '
                  f'AP75={r.AP75:.6f} ({len(keep_ids)} images)')
        results['by_code'] = code_ap

    if args.ap_only:
        if args.out:
            _dump(args.out, results)
        return

    # bootstrap scales are opt-in: without --by-source/--by-image only the
    # single AP (and --by-code APs) are computed, so a plain `--by-code` run
    # never silently launches a 1000-replicate bootstrap
    do_image = args.by_image
    do_source = args.by_source
    if not (do_source or do_image):
        print('[bootstrap] skipped: pass --by-source (correct) and/or --by-image '
              '(wrong-scale contrast) to run the bootstrap')

    if do_source or do_image:
        n_units_src = len(group_by_source(load_coco(args.ann)))
        print(f'[bootstrap] n={args.bootstrap} seed={args.seed} '
              f'(by-source units={n_units_src}, by-image units={len(images)})')

    stats = {'bootstrap_n': args.bootstrap, 'seed': args.seed}
    pcs = {}
    if do_source:
        pcs['source'] = build_precomputed(gt_index, dt, images, 'source')
        aps = bootstrap_aps(pcs['source'], args.bootstrap, args.seed)
        s = summarize(aps)
        s['mdd'] = 1.96 * np.sqrt(2.0) * s['sd']
        s['scale'] = 'by-source (correct: cluster = original source)'
        stats['by_source'] = s
        print(f'[bootstrap] by-source (correct): sd={s["sd"]:.4f} '
              f'MDD(95%)=±{s["mdd"]:.4f}')
    if do_image:
        pcs['image'] = build_precomputed(gt_index, dt, images, 'image')
        aps = bootstrap_aps(pcs['image'], args.bootstrap, args.seed)
        s = summarize(aps)
        s['mdd'] = 1.96 * np.sqrt(2.0) * s['sd']
        s['scale'] = 'by-image (WRONG scale: ignores within-source correlation)'
        stats['by_image'] = s
        print(f'[bootstrap] by-image  (WRONG)  : sd={s["sd"]:.4f} '
              f'MDD(95%)=±{s["mdd"]:.4f}')

    if 'by_source' in stats and 'by_image' in stats:
        if stats['by_source']['sd'] > stats['by_image']['sd']:
            verdict = '【符合假设】source 口径的 sd 严格大于 image 口径'
        else:
            verdict = '【与假设相反】source 口径的 sd 没有大于 image 口径'
        stats['sd_verdict'] = verdict
        print(f'[bootstrap] {verdict}')

    # paired mode
    if args.pred_b:
        gt_b, images_b, dt_b = _load_pair(args.ann, args.pred_b)
        pc_a = pcs.get('source') or build_precomputed(gt_index, dt, images, 'source')
        pc_b = build_precomputed(gt_b, dt_b, images_b, 'source')
        if pc_a.unit_list != pc_b.unit_list:
            sys.exit('[effective_n] paired mode requires the same ann sources')
        diff = bootstrap_paired(pc_a, pc_b, args.bootstrap, args.seed)
        s = summarize(diff)
        s['ci_includes_zero'] = bool(s['p2.5'] <= 0.0 <= s['p97.5'])
        s['direction'] = ('A>B' if s['mean'] > 0 else ('B>A' if s['mean'] < 0 else 'A==B'))
        s['verdict'] = ('CI 包含 0 → 差异不显著'
                        if s['ci_includes_zero']
                        else f'CI 不包含 0 → 差异显著，方向 {s["direction"]}')
        results['paired'] = s
        print(f'[paired] ΔAP mean={s["mean"]:.6f} sd={s["sd"]:.6f} '
              f'CI=[{s["p2.5"]:.6f}, {s["p97.5"]:.6f}] '
              f'CI含0={s["ci_includes_zero"]} -> {s["verdict"]}')

    results['bootstrap'] = stats
    if args.out:
        _dump(args.out, results)


def _dump(out, results):
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[out] wrote {out}')


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _toy_gt(dt_boxes_per_img, n_images, box=(0, 0, 100, 100), cat=1):
    """gt_index with one GT per image at the given box."""
    return {i: [{'id': i, 'image_id': i, 'category_id': cat,
                 'bbox': list(box), 'area': box[2] * box[3], 'iscrowd': 0}]
            for i in range(n_images)}


def self_test():
    print('[self-test] effective_n')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    # ---------- AP correctness ----------
    # 1. perfect prediction -> AP == 1.0 exactly
    gt = _toy_gt(None, 10)
    dt = [{'image_id': i, 'category_id': 1, 'bbox': [0, 0, 100, 100],
           'score': 1.0} for i in range(10)]
    r = coco_ap(gt, dt)
    check('perfect prediction -> AP == 1.0', r.AP == 1.0, f'{r.AP}')

    # 2. every GT gets exactly one IoU=0.62 box -> AP==0.30, AP50==1.0, AP75==0
    gt = _toy_gt(None, 10, box=(0, 0, 162, 162))
    dt = [{'image_id': i, 'category_id': 1, 'bbox': [38, 0, 162, 162],
           'score': 0.9} for i in range(10)]
    r = coco_ap(gt, dt)
    check('IoU=0.62 -> AP == 0.30', abs(r.AP - 0.30) < 1e-12, f'{r.AP}')
    check('IoU=0.62 -> AP50 == 1.0', r.AP50 == 1.0, f'{r.AP50}')
    check('IoU=0.62 -> AP75 == 0.0', r.AP75 == 0.0, f'{r.AP75}')

    # 3. half GT missed -> AP50 == 51/101
    n = 100
    gt = _toy_gt(None, n)
    dt = [{'image_id': i, 'category_id': 1, 'bbox': [0, 0, 100, 100],
           'score': 0.9} for i in range(n // 2)]
    r = coco_ap(gt, dt)
    check('half missed -> AP50 == 51/101',
          abs(r.AP50 - 51.0 / 101.0) < 1e-6, f'{r.AP50}')

    # 4a. trailing duplicate boxes AFTER full recall: AP must NOT change
    #     (COCO's 101-point interpolation ignores FPs that come after
    #     recall == 1.0; the pre-audit test #4 used INTERLEAVED scores,
    #     which is a different, also-correct construction -- see README audit)
    gt = _toy_gt(None, 10)
    dt_good = [{'image_id': i, 'category_id': 1,
                'bbox': [0, 0, 100, 100], 'score': 0.9} for i in range(10)]
    dt_tail = dt_good + [{'image_id': i, 'category_id': 1,
                          'bbox': [0, 0, 100, 100], 'score': 0.5}
                         for i in range(10)]
    ap_good = coco_ap(gt, dt_good).AP
    ap_tail = coco_ap(gt, dt_tail).AP
    check('4a trailing duplicates after full recall -> AP unchanged (bit-exact)',
          ap_tail == ap_good, f'{ap_tail} vs {ap_good}')

    # 4b. high-score background FPs BEFORE the TPs: AP must DROP by >= 1.0
    dt_bg = ([{'image_id': i, 'category_id': 1,
               'bbox': [300, 300, 50, 50], 'score': 0.99} for i in range(3)]
             + dt_good)
    ap_bg = coco_ap(gt, dt_bg).AP
    check('4a/4b high-score background FP lowers AP by >= 1.0',
          (ap_good - ap_bg) * 100.0 >= 1.0, f'{ap_bg} vs {ap_good}')

    # 5. pycocotools cross-check (30 images / 4 cats / noisy preds)
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print('  [SKIP] pycocotools not installed')
    else:
        rng = np.random.default_rng(0)
        n_img, cats = 30, 4
        images = [{'id': i, 'width': 200, 'height': 200,
                   'file_name': f'{i:06d}_ODC_0.jpg'} for i in range(n_img)]
        anns, dt = [], []
        aid = 1  # COCOeval counts TP via dtMatches > 0, so ids must start at 1
        for i in range(n_img):
            for k in range(rng.integers(1, 5)):
                c = int(rng.integers(1, cats + 1))
                bx = float(rng.uniform(0, 150))
                by = float(rng.uniform(0, 150))
                bw = float(rng.uniform(10, 50))
                bh = float(rng.uniform(10, 50))
                anns.append({'id': aid, 'image_id': i, 'category_id': c,
                             'bbox': [bx, by, bw, bh],
                             'area': bw * bh, 'iscrowd': 0})
                px = float(np.clip(bx + rng.normal(0, 3), 0, 200))
                py = float(np.clip(by + rng.normal(0, 3), 0, 200))
                dt.append({'image_id': i, 'category_id': int(
                    rng.integers(1, cats + 1) if rng.random() < 0.1 else c),
                    'bbox': [px, py, bw, bh],
                    'score': float(rng.random())})
                aid += 1
        toy = {'images': images, 'annotations': anns,
               'categories': [{'id': c, 'name': str(c)}
                              for c in range(1, cats + 1)]}
        import tempfile, os
        tmpdir = tempfile.mkdtemp(prefix='effn_selftest_')
        ann_path = os.path.join(tmpdir, 'toy.json')
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(toy, f)
        coco_gt = COCO(ann_path)
        coco_dt = coco_gt.loadRes(dt)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
        gt_index, imgs, _ = index_gt(toy)
        mine = coco_ap(gt_index, dt)
        diffs = [abs(mine.AP - ev.stats[0]), abs(mine.AP50 - ev.stats[1]),
                 abs(mine.AP75 - ev.stats[2])]
        check('pycocotools cross-check AP/AP50/AP75 < 1e-6',
              max(diffs) < 1e-6, f'max diff {max(diffs):.2e}')

    # ---------- bootstrap correctness ----------
    gt = _toy_gt(None, 10)
    dt = [{'image_id': i, 'category_id': 1, 'bbox': [0, 0, 100, 100],
           'score': 0.5 + 0.01 * (i % 3)} for i in range(10)]

    # 6. all-ones weights reproduce the plain AP bit-exactly
    pc = build_precomputed(gt, dt, {i: {'file_name': f'{i:06d}_ODC_0.jpg'}
                                    for i in range(10)}, 'source')
    ones = np.ones(len(pc.unit_list), dtype=np.float64)
    ap_fast = weighted_ap(pc, ones)
    r = coco_ap(gt, dt)
    check('all-ones weighted AP == plain AP (bit-exact)',
          ap_fast[0] == r.AP and ap_fast[1] == r.AP50 and ap_fast[2] == r.AP75,
          f'{ap_fast} vs {(r.AP, r.AP50, r.AP75)}')

    # 7. by-source sd strictly > by-image sd on a clustered toy
    #    5 sources x 40 images x 10 GT; source-level miss quality varies a lot
    rng = np.random.default_rng(1)
    n_src, per_src, n_gt = 5, 40, 10
    images, gt, dt = {}, {}, []
    iid = 0
    for s in range(n_src):
        miss_q = [0.0, 0.15, 0.3, 0.45, 0.6][s]
        miss_pattern = rng.random(n_gt) < miss_q  # identical inside a source
        for _ in range(per_src):
            images[iid] = {'file_name': f'{s:06d}_ODC_{iid:05d}.jpg'}
            gt[iid] = [{'id': iid * n_gt + g, 'image_id': iid, 'category_id': 1,
                        'bbox': [g * 10.0, 0, 8, 8], 'area': 64, 'iscrowd': 0}
                       for g in range(n_gt)]
            for g in range(n_gt):
                if not miss_pattern[g]:
                    dt.append({'image_id': iid, 'category_id': 1,
                               'bbox': [g * 10.0, 0, 8, 8],
                               'score': 0.9})
            iid += 1
    pc_src = build_precomputed(gt, dt, images, 'source')
    pc_img = build_precomputed(gt, dt, images, 'image')
    aps_s = bootstrap_aps(pc_src, 300, seed=0)
    aps_i = bootstrap_aps(pc_img, 300, seed=0)
    sd_s = float(np.std(aps_s, ddof=1))
    sd_i = float(np.std(aps_i, ddof=1))
    check('by-source sd strictly > by-image sd', sd_s > sd_i,
          f'{sd_s:.6f} vs {sd_i:.6f}')

    # 8. paired mode with pred_b == pred_a -> ΔAP mean & sd both 0
    import tempfile, os
    tmpdir = tempfile.mkdtemp(prefix='effn_selftest_paired_')
    ann_path = os.path.join(tmpdir, 'ann.json')
    pa = os.path.join(tmpdir, 'a.json')
    pb = os.path.join(tmpdir, 'b.json')
    rng8 = np.random.default_rng(7)
    gt8 = {i: [{'id': i, 'image_id': i, 'category_id': 1,
                'bbox': [5.0, 5.0, 40.0, 40.0], 'area': 1600, 'iscrowd': 0}]
           for i in range(10)}
    dt8 = [{'image_id': i, 'category_id': 1,
            'bbox': [5.0, 5.0, 40.0, 40.0], 'score': float(rng8.random())}
           for i in range(10)]
    coco = {'images': [{'id': i, 'width': 100, 'height': 100,
                        'file_name': f'{i:06d}_ODC_0.jpg'} for i in range(10)],
            'annotations': [a for v in gt8.values() for a in v],
            'categories': [{'id': 1, 'name': 'x'}]}
    for path, payload in ((ann_path, coco), (pa, dt8), (pb, dt8)):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
    out_path = os.path.join(tmpdir, 'paired_result.json')
    sys.argv = ['effective_n.py', '--ann', ann_path, '--pred', pa,
                '--pred-b', pb, '--by-source', '--bootstrap', '50', '--seed', '0',
                '--out', out_path]
    ns = build_parser().parse_args(sys.argv[1:])
    ns.self_test_only = False
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(ns)
    with open(out_path, encoding='utf-8') as f:
        paired = json.load(f).get('paired')
    check('paired with identical preds -> mean==0 and sd==0',
          paired is not None and paired['mean'] == 0.0 and paired['sd'] == 0.0,
          str(paired))

    # ---------- P5: ignore semantics / stratified AP (R1-R6) ----------
    # R1. no-ignore regression: the pre-patch values are pinned exactly
    gt = _toy_gt(None, 10)
    dt = [{'image_id': i, 'category_id': 1, 'bbox': [0, 0, 100, 100],
           'score': 1.0} for i in range(10)]
    r = coco_ap(gt, dt)
    check('R1 no-ignore regression (perfect -> 1.0/1.0/1.0)',
          r.AP == 1.0 and r.AP50 == 1.0 and r.AP75 == 1.0,
          f'{(r.AP, r.AP50, r.AP75)}')
    gt = _toy_gt(None, 10, box=(0, 0, 162, 162))
    dt = [{'image_id': i, 'category_id': 1, 'bbox': [38, 0, 162, 162],
           'score': 0.9} for i in range(10)]
    r = coco_ap(gt, dt)
    check('R1 no-ignore regression (IoU=0.62 -> 0.30/1.0/0.0)',
          r.AP50 == 1.0 and r.AP75 == 0.0 and abs(r.AP - 0.30) < 1e-12,
          f'{(r.AP, r.AP50, r.AP75)}')

    # R2. all GTs in the subset -> stratified AP == full AP (bit-exact)
    gt = _toy_gt(None, 10)
    dt = [{'image_id': i, 'category_id': 1, 'bbox': [0, 0, 100, 100],
           'score': 0.5 + 0.01 * (i % 3)} for i in range(10)]
    full = coco_ap(gt, dt)
    strat = coco_ap(gt, dt,
                    gt_subset={a['id'] for v in gt.values() for a in v})
    check('R2 full subset -> bit-exact AP',
          (strat.AP, strat.AP50, strat.AP75) == (full.AP, full.AP50, full.AP75),
          f'{(strat.AP, strat.AP50, strat.AP75)} vs {(full.AP, full.AP50, full.AP75)}')

    # R3. THE regression case for the old delete-semantics bug: an in-subset
    #     GT and an ignored GT, each with a perfect detection, and the ignored
    #     one has the HIGHER score. Old code deleted the ignored GT -> its
    #     detection became an FP at the top of the score order -> AP == 0.5.
    #     Correct COCO semantics: it is ignored (neither TP nor FP) -> AP == 1.0.
    gt = {0: [{'id': 1, 'image_id': 0, 'category_id': 1,
               'bbox': [0, 0, 100, 100], 'area': 10000, 'iscrowd': 0}],
          1: [{'id': 2, 'image_id': 1, 'category_id': 1,
               'bbox': [0, 0, 100, 100], 'area': 10000, 'iscrowd': 0}]}
    dt = [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 100, 100],
           'score': 0.95},
          {'image_id': 0, 'category_id': 1, 'bbox': [0, 0, 100, 100],
           'score': 0.9}]
    r = coco_ap(gt, dt, gt_subset={1})
    check('R3 detection on ignored GT is neither TP nor FP -> AP == 1.0',
          r.AP == 1.0, f'{r.AP} (old buggy semantics gave 0.5)')

    # R4. empty subset -> hard error, never a silent 0/nan
    raised = False
    try:
        coco_ap(gt, dt, gt_subset=set())
    except Exception:
        raised = True
    check('R4 empty subset raises', raised)

    # R5. early-termination rule: the det overlaps the ignored GT MORE
    #     (IoU 0.90 vs 0.714) but must stay matched to the in-subset GT.
    #     Without the rule it degrades onto the ignored GT -> dt_ig=1 -> AP=0.
    gt = {0: [{'id': 1, 'image_id': 0, 'category_id': 1,
               'bbox': [0, 0, 100, 100], 'area': 10000, 'iscrowd': 0},
              {'id': 2, 'image_id': 0, 'category_id': 1,
               'bbox': [0, 0, 100, 126], 'area': 12600, 'iscrowd': 0}]}
    dt = [{'image_id': 0, 'category_id': 1, 'bbox': [0, 0, 100, 140],
           'score': 0.9}]
    r = coco_ap(gt, dt, gt_subset={1})
    # npig=1; det is TP at IoU 0.50..0.70 (5/10 thresholds) -> AP=0.5, AP50=1.0
    check('R5 early termination keeps the non-ignored match',
          abs(r.AP - 0.5) < 1e-12 and r.AP50 == 1.0, f'{(r.AP, r.AP50)}')

    # R6. pycocotools cross-check WITH iscrowd GTs (crowd re-matchable, ignored
    #     in accumulation)
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print('  [SKIP] pycocotools not installed (R6)')
    else:
        rng = np.random.default_rng(3)
        n_img, cats = 20, 3
        images = [{'id': i, 'width': 200, 'height': 200,
                   'file_name': f'{i:06d}_ODC_0.jpg'} for i in range(n_img)]
        anns, dt6 = [], []
        aid = 1
        for i in range(n_img):
            for k in range(rng.integers(1, 5)):
                c = int(rng.integers(1, cats + 1))
                bx = float(rng.uniform(0, 150))
                by = float(rng.uniform(0, 150))
                bw = float(rng.uniform(10, 60))
                bh = float(rng.uniform(10, 60))
                crowd = int(rng.random() < 0.25)
                anns.append({'id': aid, 'image_id': i, 'category_id': c,
                             'bbox': [bx, by, bw, bh], 'area': bw * bh,
                             'iscrowd': crowd})
                px = float(np.clip(bx + rng.normal(0, 4), 0, 200))
                py = float(np.clip(by + rng.normal(0, 4), 0, 200))
                dt6.append({'image_id': i, 'category_id': c,
                            'bbox': [px, py, bw, bh],
                            'score': float(rng.random())})
                aid += 1
        toy = {'images': images, 'annotations': anns,
               'categories': [{'id': c, 'name': str(c)}
                              for c in range(1, cats + 1)]}
        import tempfile, os
        tmpdir = tempfile.mkdtemp(prefix='effn_r6_')
        ann_path = os.path.join(tmpdir, 'toy.json')
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(toy, f)
        coco_gt = COCO(ann_path)
        coco_dt = coco_gt.loadRes(dt6)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
        gt_index, _, _ = index_gt(toy)
        mine = coco_ap(gt_index, dt6)
        diffs = [abs(mine.AP - ev.stats[0]), abs(mine.AP50 - ev.stats[1]),
                 abs(mine.AP75 - ev.stats[2])]
        check('R6 pycocotools cross-check with iscrowd < 1e-6',
              max(diffs) < 1e-6, f'max diff {max(diffs):.2e}')

    # R7. weighted_ap slow path (has_ignore) must equal coco_ap(gt_subset)
    #     bit-exactly at all-ones weights -- locks the ignore-aware accumulation
    gt7 = {0: [{'id': 1, 'image_id': 0, 'category_id': 1,
                'bbox': [0, 0, 100, 100], 'area': 10000, 'iscrowd': 0}],
           1: [{'id': 2, 'image_id': 1, 'category_id': 1,
                'bbox': [0, 0, 100, 100], 'area': 10000, 'iscrowd': 0}]}
    dt7 = [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 100, 100],
            'score': 0.95},
           {'image_id': 0, 'category_id': 1, 'bbox': [0, 0, 100, 100],
            'score': 0.9}]
    r7 = coco_ap(gt7, dt7, gt_subset={1})
    pc7 = build_precomputed(
        gt7, dt7,
        {0: {'file_name': '000000_ODC_0.jpg'},
         1: {'file_name': '000001_ODC_1.jpg'}},
        'source', gt_subset={1})
    ap7 = weighted_ap(pc7, np.ones(len(pc7.unit_list), dtype=np.float64))
    check('R7 ignore-aware weighted_ap == coco_ap(gt_subset) bit-exact',
          ap7 == (r7.AP, r7.AP50, r7.AP75),
          f'{ap7} vs {(r7.AP, r7.AP50, r7.AP75)}')

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ann', default=None, help='val ann json (all 5 codes)')
    ap.add_argument('--pred', default=None, help='pred json (method A)')
    ap.add_argument('--pred-b', default=None,
                    help='pred json (method B); enables paired ΔAP mode')
    ap.add_argument('--gt-subset', default=None,
                    help='groups json from degradation_paired --export-groups; '
                         'computes per-group stratified AP (out-of-group GTs '
                         'are ignored, never deleted)')
    ap.add_argument('--by-source', action='store_true',
                    help='bootstrap over original sources (correct)')
    ap.add_argument('--by-image', action='store_true',
                    help='bootstrap over single images (WRONG scale, contrast)')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--by-code', action='store_true',
                    help='also report per-code AP (ODC/LDC/DDC/GDC/PDC)')
    ap.add_argument('--ap-only', action='store_true',
                    help='single AP, no bootstrap (for cross-validation)')
    ap.add_argument('--probe', action='store_true',
                    help='only inspect ann source grouping, no AP')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='result json path')
    ap.add_argument('--self-test-only', action='store_true')
    return ap


def main():
    args = build_parser().parse_args()
    if args.self_test_only:
        sys.exit(self_test())
    if args.probe:
        if not args.ann:
            build_parser().error('--probe requires --ann')
        probe(args.ann)
        return
    if not args.ann or not args.pred:
        build_parser().error('--ann and --pred are required (unless '
                             '--self-test-only / --probe)')
    run(args)


if __name__ == '__main__':
    main()

