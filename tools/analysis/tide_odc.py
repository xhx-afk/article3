#!/usr/bin/env python3
"""ODC-scale TIDE error decomposition, three-seed mean, BOTH IoU thresholds.

Run AFTER the baseline dumps exist:
    python tools/analysis/tide_odc.py \
        --glob 'prep/baseline/seed*/odc' --out-dir prep/tide_odc

Why two thresholds (failure-diagnosis doc §5.3): TIDE's default pos_threshold
0.5 puts only IoU in [0.1, 0.5) into the Loc bucket, but the "boxes are
inaccurate" claim is about IoU in 0.5~0.75 -- that part never enters the Loc
bucket at 0.5. The mechanism table must use the 0.75 set. Every run therefore
produces BOTH 0.5 and 0.75 results in one json (`by_threshold`), plus the
AP50-AP75 cross-check computed in-pipeline via common.coco_ap (tidecv itself
exposes no AP50/AP75).

Requires: pip install tidecv
"""

import argparse
import glob as globmod
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import coco_ap, index_gt, load_coco  # noqa: E402

# this tidecv version's main-error short names (manual: loc/miss/both/cls/bkg/dup)
MAIN_ORDER = ['Loc', 'Both', 'Cls', 'Dupe', 'Bkg', 'Miss']
SHARE_KEYS = ['Loc', 'Both', 'Cls', 'Bkg', 'Miss']  # "recoverable dAP" pool
THRESHOLDS = (0.5, 0.75)   # §3: always produce both in one run


def _seg_valid(seg):
    """tidecv calls frPyObjects on the segmentation; empty lists ([]) or
    empty polygons crash with IndexError."""
    if not seg:
        return False
    if isinstance(seg, list):
        return any(isinstance(s, (list, tuple)) and len(s) >= 4 for s in seg)
    return True  # dict (RLE) -- pass through


def _ensure_segmentation(ann_path, out_dir):
    """tidecv's COCO loader reads ann['segmentation']; detection-only COCO
    files often omit it or leave it as []. Return a temp json with bbox
    polygons filled in for missing/empty ones (TIDE's box-mode errors are
    computed from bboxes anyway)."""
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    bad = [a for a in coco.get('annotations', [])
           if not _seg_valid(a.get('segmentation'))]
    if not bad:
        return ann_path
    for a in bad:
        x, y, w, h = [float(v) for v in a['bbox'][:4]]
        a['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, '_ann_with_segmentation.json')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(coco, f)
    print(f'[tide_odc] filled bbox-polygon segmentation for {len(bad)} '
          f'annotations (was missing/empty)')
    return tmp


def run_one(ann_path, pred_path, out_dir, pos_threshold):
    from tidecv import TIDE, datasets
    os.makedirs(out_dir, exist_ok=True)
    ann_use = _ensure_segmentation(ann_path, out_dir)
    gt = datasets.COCO(ann_use)
    det = datasets.COCOResult(pred_path)
    t = TIDE(pos_threshold=pos_threshold)
    t.evaluate(gt, det, mode=TIDE.BOX)
    t.summarize()
    try:
        t.plot(out_dir)
    except Exception as e:  # plotting needs matplotlib; numbers are what matter
        print(f'[tide_odc] plot skipped ({e})')
    errors = t.get_main_errors()
    run_name = list(errors.keys())[0]
    # tidecv's get_main_errors already returns PERCENT points
    return {k: float(errors[run_name][k]) for k in MAIN_ORDER}


def _shares(errs):
    pool = sum(errs[k] for k in SHARE_KEYS)
    return {
        'pool': pool,
        'loc_share': errs['Loc'] / pool if pool > 0 else float('nan'),
        'miss_share': errs['Miss'] / pool if pool > 0 else float('nan'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--glob', required=True,
                    help="glob of seed dirs, e.g. 'prep/baseline/seed*/odc'")
    ap.add_argument('--out-dir', required=True, help='e.g. prep/tide_odc')
    ap.add_argument('--pos-threshold', type=float, default=0.5,
                    help='PRIMARY threshold for the headline printout '
                         '(default 0.5, unchanged); BOTH 0.5 and 0.75 are '
                         'always computed and stored in by_threshold')
    args = ap.parse_args()

    try:
        import tidecv  # noqa: F401
    except ImportError:
        sys.exit('[tide_odc] tidecv not installed: pip install tidecv')

    dirs = sorted(globmod.glob(args.glob))
    if not dirs:
        sys.exit(f'[tide_odc] no directories match {args.glob}')

    per_seed = {}          # seed -> {thr(str): errs}
    ap50_75_per_seed = []
    for d in dirs:
        ann_path = os.path.join(d, 'ann.json')
        pred_path = os.path.join(d, 'pred.json')
        if not (os.path.isfile(ann_path) and os.path.isfile(pred_path)):
            sys.exit(f'[tide_odc] missing ann.json/pred.json in {d}')
        # deterministic seed key: last TWO path components (e.g. 'seed0/odc')
        _parts = d.replace('\\', '/').rstrip('/').split('/')
        seed = '/'.join(_parts[-2:]) if len(_parts) >= 2 else _parts[-1]

        # in-pipeline AP50-AP75 (tidecv exposes none); same coco_ap as
        # effective_n, so the cross-check is same-pipeline by construction
        coco = load_coco(ann_path)
        gt_index, _, _ = index_gt(coco)
        with open(pred_path, 'r', encoding='utf-8') as f:
            dt = json.load(f)
        r = coco_ap(gt_index, dt)
        gap = (r.AP50 - r.AP75) * 100.0
        ap50_75_per_seed.append(gap)

        per_thr = {}
        for thr in THRESHOLDS:
            tag = f'{thr:g}'
            print(f'\n===== TIDE: {d}  (pos_threshold={tag}) =====')
            errs = run_one(ann_path, pred_path,
                           os.path.join(args.out_dir, seed, f'thr{tag}'), thr)
            per_thr[tag] = errs
            sh = _shares(errs)
            print(f"[tide_odc] {seed} @{tag}: " +
                  '  '.join(f'{k}={errs[k]:.2f}' for k in MAIN_ORDER) +
                  f"  | Loc 占可恢复 dAP 比例 = {sh['loc_share']*100:.1f}%")
        per_seed[seed] = per_thr
        print(f'[tide_odc] {seed}: AP50-AP75 = {gap:.2f} (in-pipeline coco_ap)')

        with open(os.path.join(args.out_dir,
                               f'{seed.replace("/", "_")}_main_errors.json'),
                  'w', encoding='utf-8') as f:
            json.dump({'main_errors': per_thr[f'{args.pos_threshold:g}'],
                       'by_threshold': per_thr,
                       'ap50_minus_ap75': gap,
                       'loc_share_of_recoverable_dap':
                           _shares(per_thr[f'{args.pos_threshold:g}'])['loc_share']},
                      f, indent=2)

    n = len(per_seed)
    by_thr_mean = {}
    for thr in THRESHOLDS:
        tag = f'{thr:g}'
        mean = {k: sum(per_seed[s][tag][k] for s in per_seed) / n
                for k in MAIN_ORDER}
        sh = _shares(mean)
        by_thr_mean[tag] = {'main_errors_mean': mean, **sh}

    gap_mean = sum(ap50_75_per_seed) / n
    loc05 = by_thr_mean['0.5']['main_errors_mean']['Loc']
    loc075 = by_thr_mean['0.75']['main_errors_mean']['Loc']
    # cross-check verdict (§3): the 0.75 Loc bucket must pick up materially
    # more localization error than 0.5, and AP50-AP75 must be large -- only
    # then do the two instruments agree that "boxes are inaccurate"
    cross = (loc075 > loc05) and (gap_mean > 0)
    verdict = ('【两个口径互证】Loc@0.75 > Loc@0.5 且 AP50-AP75 > 0'
               if cross else
               '【两个口径不互证】检查 Loc@0.75 vs Loc@0.5 与 AP50-AP75')

    print(f'\n[tide_odc] ===== {n} seeds mean (percent points) =====')
    for tag in ('0.5', '0.75'):
        m = by_thr_mean[tag]
        print(f'@{tag}: ' + '  '.join(
            f'{k}={m["main_errors_mean"][k]:.2f}' for k in MAIN_ORDER))
        print(f'@{tag}: Loc 占比 = {m["loc_share"]*100:.1f}%   '
              f'Miss 占比 = {m["miss_share"]*100:.1f}%')
    print(f'[tide_odc] AP50-AP75 (mean) = {gap_mean:.2f}  '
          f'(ODC 口径历史读数 28.83，混合口径 27.64)')
    print(f'[tide_odc] Loc@0.5 = {loc05:.2f} -> Loc@0.75 = {loc075:.2f}   {verdict}')
    print('[tide_odc] 机制表用 @0.75 一套（0.5 的 Loc 桶测不到 IoU 0.5~0.75 的误差）')

    with open(os.path.join(args.out_dir, 'mean_main_errors.json'),
              'w', encoding='utf-8') as f:
        json.dump({'n_seeds': n,
                   # backward-compatible headline = primary threshold
                   'main_errors_mean':
                       by_thr_mean[f'{args.pos_threshold:g}']['main_errors_mean'],
                   'loc_share_of_recoverable_dap':
                       by_thr_mean[f'{args.pos_threshold:g}']['loc_share'],
                   'miss_share':
                       by_thr_mean[f'{args.pos_threshold:g}']['miss_share'],
                   'by_threshold': by_thr_mean,
                   'ap50_minus_ap75_mean': gap_mean,
                   'cross_check_verdict': verdict},
                  f, indent=2, ensure_ascii=False)
    print(f'[tide_odc] wrote {args.out_dir}/mean_main_errors.json')


if __name__ == '__main__':
    main()
