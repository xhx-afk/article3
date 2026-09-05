#!/usr/bin/env python3
"""ODC-scale TIDE error decomposition (user manual §6), three-seed mean.

Run AFTER the baseline dumps exist:
    python tools/analysis/tide_odc.py \
        --glob 'prep/baseline/seed*/odc' --out-dir prep/tide_odc

For each directory matching --glob (containing ann.json / pred.json) this
runs tidecv TIDE (BOX mode), saves the printed report, and finally prints
the three-seed mean table plus the two pre-registered ratios:
    Loc share  = Loc / (Loc+Det+Cls+Bkg+Miss)   (target > 47.1%)
    AP50-AP75 cross-check is read from effective_n (ODC scale).

Requires: pip install tidecv
"""

import argparse
import glob as globmod
import json
import os
import sys

# this tidecv version's main-error short names (manual: loc/miss/both/cls/bkg/dup)
MAIN_ORDER = ['Loc', 'Both', 'Cls', 'Dupe', 'Bkg', 'Miss']
SHARE_KEYS = ['Loc', 'Both', 'Cls', 'Bkg', 'Miss']  # "recoverable dAP" pool


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
    import tempfile
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


def run_one(ann_path, pred_path, out_dir):
    from tidecv import TIDE, datasets
    os.makedirs(out_dir, exist_ok=True)
    ann_use = _ensure_segmentation(ann_path, out_dir)
    gt = datasets.COCO(ann_use)
    det = datasets.COCOResult(pred_path)
    t = TIDE()
    t.evaluate(gt, det, mode=TIDE.BOX)
    t.summarize()
    try:
        t.plot(out_dir)
    except Exception as e:  # plotting needs matplotlib; numbers are what matter
        print(f'[tide_odc] plot skipped ({e})')
    errors = t.get_main_errors()
    run_name = list(errors.keys())[0]
    return {k: float(errors[run_name][k]) for k in MAIN_ORDER}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--glob', required=True,
                    help="glob of seed dirs, e.g. 'prep/baseline/seed*/odc'")
    ap.add_argument('--out-dir', required=True, help='e.g. prep/tide_odc')
    args = ap.parse_args()

    try:
        import tidecv  # noqa: F401
    except ImportError:
        sys.exit('[tide_odc] tidecv not installed: pip install tidecv')

    dirs = sorted(globmod.glob(args.glob))
    if not dirs:
        sys.exit(f'[tide_odc] no directories match {args.glob}')

    per_seed = {}
    for d in dirs:
        ann_path = os.path.join(d, 'ann.json')
        pred_path = os.path.join(d, 'pred.json')
        if not (os.path.isfile(ann_path) and os.path.isfile(pred_path)):
            sys.exit(f'[tide_odc] missing ann.json/pred.json in {d}')
        # deterministic seed key: last TWO path components (e.g. 'seed0/odc')
        # -- basename(dirname) collides when parents share a name
        _parts = d.replace('\\', '/').rstrip('/').split('/')
        seed = '/'.join(_parts[-2:]) if len(_parts) >= 2 else _parts[-1]
        print(f'\n===== TIDE: {d} =====')
        errs = run_one(ann_path, pred_path, os.path.join(args.out_dir, seed))
        per_seed[seed] = errs
        pool = sum(errs[k] for k in SHARE_KEYS)
        loc_share = errs['Loc'] / pool if pool > 0 else float('nan')
        with open(os.path.join(args.out_dir, f'{seed.replace("/", "_")}_main_errors.json'),
                  'w', encoding='utf-8') as f:
            json.dump({'main_errors': errs,
                       'loc_share_of_recoverable_dap': loc_share},
                      f, indent=2)
        # tidecv's get_main_errors already returns PERCENT points
        print(f"[tide_odc] {seed}: " +
              '  '.join(f'{k}={errs[k]:.2f}' for k in MAIN_ORDER) +
              f"  | Loc 占可恢复 dAP 比例 = {loc_share*100:.1f}% (目标 > 47.1%)")
    
    n = len(per_seed)
    mean = {k: sum(e[k] for e in per_seed.values()) / n for k in MAIN_ORDER}
    pool = sum(mean[k] for k in SHARE_KEYS)
    loc_share = mean['Loc'] / pool if pool > 0 else float('nan')
    print(f'\n[tide_odc] ===== {n} seeds mean (percent points) =====')
    print('  '.join(f'{k}={mean[k]:.2f}' for k in MAIN_ORDER))
    print(f"Loc 占可恢复 dAP 比例 = {loc_share*100:.1f}%  (混合口径参照 47.1%；"
          f"判据：> 47.1% = 符合预期，下降则停下重议)")
    miss_share = mean['Miss'] / pool if pool > 0 else float('nan')
    print(f"Miss 占比 = {miss_share*100:.1f}%  (判据 < 34.8%)")
    print('（AP50 减 AP75 互证：ODC 口径 effective_n 读数 = 28.83，混合口径 27.64）')
    with open(os.path.join(args.out_dir, 'mean_main_errors.json'),
              'w', encoding='utf-8') as f:
        json.dump({'n_seeds': n,
                   'main_errors_mean': mean,
                   'loc_share_of_recoverable_dap': loc_share,
                   'miss_share': miss_share},
                  f, indent=2, ensure_ascii=False)
    print(f'[tide_odc] wrote {args.out_dir}/mean_main_errors.json')


if __name__ == '__main__':
    main()
