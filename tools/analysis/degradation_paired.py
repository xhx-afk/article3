#!/usr/bin/env python3
"""Per-instance paired analysis across the 5 augmentation codes.

Ground-truth geometry is identical for the same (source, direction) across
ODC/LDC/DDC/GDC/PDC (verified upstream), so instances can be paired directly:

- contrast: Euclidean distance between the median CIE Lab color inside the GT
  box and in the surrounding ring, computed ON THE CLEAN ODC IMAGE.
- detection: for each instance and each code, the same-class detection with
  the maximum IoU in that code's image; det = score >= --score-thr and
  IoU >= --iou-thr.
- core statistic: survival rate (det_X still True) of instances detected on
  ODC, stratified by contrast tertile.

Lab warning (the single easiest thing to get wrong): the conversion MUST use
the float32 path `cv2.cvtColor(rgb.astype(np.float32)/255.0, COLOR_RGB2Lab)`
giving L in [0,100], a/b in [-127,127]. The uint8 path scales L by 2.55 which
distorts dE by an instance-dependent factor (x2.55 for luminance-dominated
instances, x1.0 for chroma-dominated ones). A regression assertion in the
self-test locks this behavior.

Outputs:
    paired_instances.csv  columns: src, direction, cls, area, contrast,
        contrast_tertile, then iou_X/score_X/det_X for X in the 5 codes
        (innovation 3 reads this table directly; do not rename columns)
    summary.json

Self-test:
    python tools/analysis/degradation_paired.py --self-test-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUG_RE, NAME_RE, index_gt, load_coco, norm_id, parse_name  # noqa: E402

CODES = ['ODC', 'LDC', 'DDC', 'GDC', 'PDC']
CSV_COLUMNS = (['src', 'direction', 'cls', 'area', 'contrast', 'contrast_tertile'] +
               [f'{f}_{c}' for c in CODES for f in ('iou', 'score', 'det')])

try:
    import cv2
except ImportError:  # only degradation_paired / denoise_probe need cv2
    cv2 = None


# -----------------------------------------------------------------------------
# Contrast (the only easy-to-get-wrong part)
# -----------------------------------------------------------------------------

def rgb_to_lab_f32(rgb):
    """Real CIE Lab: L in [0,100], a/b in [-127,127].

    Input uint8 RGB; the float32 path requires normalization to [0,1] --
    NEVER call cvtColor on the raw uint8 image (L gets scaled by 2.55).
    """
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def box_ring_contrast(rgb, box_xywh, ring_scale=0.6, exclude_boxes=None):
    """Median-Lab Euclidean distance between box interior and surrounding ring.

    Args:
        rgb: HxWx3 uint8 RGB image.
        box_xywh: GT box [x, y, w, h] (floats).
        ring_scale: total expansion ratio split evenly to both sides
            (0.6 -> 1.6x box size, 0.3*w / 0.3*h added per side).
        exclude_boxes: optional list of other GT xywh boxes whose interiors
            are removed from the ring.

    Returns:
        (contrast, n_ring_pixels); contrast is nan when the ring has < 50 px.
    """
    h_img, w_img = rgb.shape[:2]
    x, y, w, h = [float(v) for v in box_xywh]
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = int(np.ceil(x + w)), int(np.ceil(y + h))
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, w_img), min(y2, h_img)
    if x2c <= x1c or y2c <= y1c:
        return float('nan'), 0

    pad_x = int(round(ring_scale / 2.0 * w))
    pad_y = int(round(ring_scale / 2.0 * h))
    ox1, oy1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
    ox2, oy2 = min(x2 + pad_x, w_img), min(y2 + pad_y, h_img)

    lab = rgb_to_lab_f32(rgb)
    box_mask = np.zeros(lab.shape[:2], dtype=bool)
    box_mask[y1c:y2c, x1c:x2c] = True
    outer_mask = np.zeros(lab.shape[:2], dtype=bool)
    outer_mask[oy1:oy2, ox1:ox2] = True
    ring_mask = outer_mask & ~box_mask

    if exclude_boxes:
        for ex, ey, ew, eh in exclude_boxes:
            ex1, ey1 = max(int(np.floor(ex)), 0), max(int(np.floor(ey)), 0)
            ex2 = min(int(np.ceil(ex + ew)), w_img)
            ey2 = min(int(np.ceil(ey + eh)), h_img)
            if ex2 > ex1 and ey2 > ey1:
                ring_mask[ey1:ey2, ex1:ex2] = False

    n_ring = int(ring_mask.sum())
    if n_ring < 50:
        return float('nan'), n_ring

    med_box = np.median(lab[box_mask], axis=0)
    med_ring = np.median(lab[ring_mask], axis=0)
    return float(np.linalg.norm(med_box - med_ring)), n_ring


def contrast_tertiles(values, labels=('low', 'mid', 'high')):
    """Tertile binning robust to massive ties: rank(method='first') then qcut."""
    import pandas as pd
    s = pd.Series(np.asarray(values, dtype=np.float64))
    ranked = s.rank(method='first')
    binned = pd.qcut(ranked, 3, labels=list(labels))
    return np.asarray(binned.astype(object))


# -----------------------------------------------------------------------------
# Instance table + paired matching
# -----------------------------------------------------------------------------

def build_instance_table(gt_index, images):
    """Instances come from the clean ODC images; key = (src, direction, cls, bbox)."""
    instances = []
    for iid, img in sorted(images.items()):
        parsed = parse_name(img.get('file_name', ''))
        if not parsed:
            continue
        src, code, direction = parsed
        if code != 'ODC':
            continue
        for ann in gt_index.get(iid, []):
            if int(ann.get('iscrowd', 0)) == 1 or int(ann.get('ignore', 0)) == 1:
                continue
            bx = [float(v) for v in ann['bbox'][:4]]
            instances.append({
                'src': src, 'direction': direction,
                'cls': int(ann['category_id']),
                'area': bx[2] * bx[3],
                'bbox': bx,
                'image_id': iid,
                'contrast': float('nan'),
                'contrast_tertile': None,
            })
    return instances


def instance_key(inst):
    return (inst['src'], inst['direction'], inst['cls'],
            tuple(round(v, 4) for v in inst['bbox']))


def code_image_lookup(images):
    """{(src, code, direction): image_id}."""
    lookup = {}
    for iid, img in images.items():
        parsed = parse_name(img.get('file_name', ''))
        if parsed:
            lookup[(parsed[0], parsed[1], parsed[2])] = iid
    return lookup


def pair_and_match(instances, gt_index, images, dt, codes=CODES,
                   score_thr=0.5, iou_thr=0.7):
    """Fill iou_X / score_X / det_X per instance for each code.

    For each instance and code, the same-class detection with the maximum IoU
    in that code's image is taken (no score pre-filter, so low-score IoU is
    still recorded).
    """
    from common import iou_matrix

    dt_by_image_cat = {}
    for d in dt:
        dt_by_image_cat.setdefault(
            (int(d['image_id']), int(d['category_id'])), []).append(d)

    lookup = code_image_lookup(images)
    rows = []
    for inst in instances:
        row = {'src': inst['src'], 'direction': inst['direction'],
               'cls': inst['cls'], 'area': inst['area'],
               'contrast': inst['contrast'],
               'contrast_tertile': inst['contrast_tertile']}
        for code in codes:
            iid = lookup.get((inst['src'], code, inst['direction']))
            iou_val, score_val = float('nan'), float('nan')
            if iid is not None:
                dets = dt_by_image_cat.get((iid, inst['cls']), [])
                if dets:
                    boxes = np.asarray([d['bbox'] for d in dets], dtype=np.float64)
                    ious = iou_matrix([inst['bbox']], boxes)[0]
                    best = int(np.argmax(ious))
                    iou_val = float(ious[best])
                    score_val = float(dets[best]['score'])
            row[f'iou_{code}'] = iou_val
            row[f'score_{code}'] = score_val
            row[f'det_{code}'] = bool(
                np.isfinite(score_val) and np.isfinite(iou_val) and
                score_val >= score_thr and iou_val >= iou_thr)
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Real run
# -----------------------------------------------------------------------------

def run(args):
    import pandas as pd

    coco = load_coco(args.ann)
    gt_index, images, categories = index_gt(coco)
    with open(args.pred, 'r', encoding='utf-8') as f:
        dt = json.load(f)

    instances = build_instance_table(gt_index, images)
    if not instances:
        sys.exit('[degradation_paired] no ODC instances found -- check the '
                 'ann file_name format / img-dir')

    # contrast on the clean ODC images
    nan_count = 0
    for inst in instances:
        img_path = os.path.join(args.img_dir,
                                images[inst['image_id']].get('file_name', ''))
        if not os.path.exists(img_path):
            sys.exit(f'[degradation_paired] ODC image missing: {img_path}')
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            sys.exit(f'[degradation_paired] failed to read: {img_path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        others = [a['bbox'] for a in gt_index.get(inst['image_id'], [])
                  if a is not None and
                  [float(v) for v in a['bbox'][:4]] != inst['bbox']]
        contrast, n_ring = box_ring_contrast(
            rgb, inst['bbox'], ring_scale=args.ring_scale,
            exclude_boxes=others if args.exclude_other_gt else None)
        inst['contrast'] = contrast
        if not np.isfinite(contrast):
            nan_count += 1
    print(f'[contrast] computed for {len(instances)} instances '
          f'(nan ring<50px: {nan_count})')

    finite = [inst['contrast'] for inst in instances
              if np.isfinite(inst['contrast'])]
    tertiles = contrast_tertiles(finite)
    it = iter(tertiles)
    for inst in instances:
        inst['contrast_tertile'] = next(it) if np.isfinite(inst['contrast']) else None
    if len(finite):
        print(f'[contrast] median = {np.median(finite):.2f} '
              f'(validation anchors by tertile: low ~1.0 / mid ~2.2 / high ~6.0)')

    rows = pair_and_match(instances, gt_index, images, dt,
                          score_thr=args.score_thr, iou_thr=args.iou_thr)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'paired_instances.csv', index=False)

    # core statistics: survival of ODC-detected instances per code x tertile
    base = df[df['det_ODC'] == True]  # noqa: E712
    survival = {}
    for code in CODES:
        survival[code] = {}
        for t in ('low', 'mid', 'high'):
            sub = base[base['contrast_tertile'] == t]
            survival[code][t] = (float(sub[f'det_{code}'].mean())
                                 if len(sub) else None)
        print(f'[survival] {code}: ' +
              '  '.join(f'{t}={survival[code][t] * 100 if survival[code][t] is not None else float("nan"):.1f}%'
                        for t in ('low', 'mid', 'high')))

    # monotonicity discipline tag (survival expected to rise with contrast)
    for code in ('GDC', 'PDC'):
        v = [survival[code][t] for t in ('low', 'mid', 'high')]
        if all(x is not None for x in v):
            tag = ('【符合假设】存活率随对比度单调上升'
                   if v[0] < v[1] < v[2] else '【与假设相反】存活率未随对比度单调上升')
            print(f'[verdict] {code}: {tag}')

    per_class = {}
    for cls, sub in df.groupby('cls'):
        per_class[int(cls)] = {
            'n': int(len(sub)),
            'odc_det_rate': float(sub['det_ODC'].mean()),
            'pdc_det_rate': float(sub['det_PDC'].mean()),
            'name': categories.get(int(cls), {}).get('name', str(cls)),
        }
        print(f'[per-class] cls={cls}: ODC {per_class[int(cls)]["odc_det_rate"]*100:.1f}%'
              f' -> PDC {per_class[int(cls)]["pdc_det_rate"]*100:.1f}%')

    summary = {
        'score_thr': args.score_thr,
        'iou_thr': args.iou_thr,
        'ring_scale': args.ring_scale,
        'exclude_other_gt': bool(args.exclude_other_gt),
        'n_instances': int(len(df)),
        'n_instances_detected_on_odc': int(len(base)),
        'nan_contrast_count': int(nan_count),
        'survival_by_tertile': survival,
        'per_class': per_class,
        'anchors': {
            'n_detected_on_odc_expected': 617,
            'survival_expected': {'GDC': [0.597, 0.780, 0.932],
                                  'PDC': [0.437, 0.615, 0.830]},
        },
    }
    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[out] {out_dir / "paired_instances.csv"}')
    print(f'[out] {out_dir / "summary.json"}')


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _gray_lab_l(v):
    """Analytic CIE Lab L for an 8-bit gray value (a=b=0)."""
    c = v / 255.0
    lin = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    f = lin ** (1.0 / 3.0) if lin > 0.008856 else 7.787 * lin + 16.0 / 116.0
    return 116.0 * f - 16.0


def _contrast_uint8(rgb, box_xywh, ring_scale=0.6):
    """The WRONG uint8 cv2 path, only for the regression assertion.

    Same RGB input and channel order as the production function, so the only
    difference under test is the uint8 scaling (L <- L*255/100, a/b <- a/b+128).
    """
    x, y, w, h = [int(round(v)) for v in box_xywh]
    H, W = rgb.shape[:2]
    pad_x, pad_y = int(round(ring_scale / 2 * w)), int(round(ring_scale / 2 * h))
    ox1, oy1, ox2, oy2 = max(x - pad_x, 0), max(y - pad_y, 0), \
        min(x + w + pad_x, W), min(y + h + pad_y, H)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    box = lab[y:y + h, x:x + w]
    ring = np.concatenate([
        lab[oy1:y, ox1:ox2].reshape(-1, 3),
        lab[y + h:oy2, ox1:ox2].reshape(-1, 3),
        lab[y:y + h, ox1:x].reshape(-1, 3),
        lab[y:y + h, x + w:ox2].reshape(-1, 3),
    ], axis=0)
    if ring.shape[0] < 50:
        return float('nan')
    return float(np.linalg.norm(np.median(box.reshape(-1, 3), 0) -
                                np.median(ring, 0)))


def self_test():
    print('[self-test] degradation_paired')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    # 1. synthetic flat image: contrast matches the analytic Lab distance
    img = np.full((100, 100, 3), 255, dtype=np.uint8)  # white background
    img[30:70, 30:70] = 128                            # gray rectangle
    contrast, n_ring = box_ring_contrast(img, [30, 30, 40, 40], ring_scale=0.6)
    analytic = abs(_gray_lab_l(255) - _gray_lab_l(128))
    check('flat-image contrast matches analytic Lab distance (<0.05)',
          abs(contrast - analytic) < 0.05, f'{contrast:.4f} vs {analytic:.4f}')

    # 2. regression lock on the float32 Lab path (uint8 distortion shape)
    #    2a. pure-luminance pair: uint8/float32 ratio ~ 2.55
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    img[30:70, 30:70] = 150
    c32, _ = box_ring_contrast(img, [30, 30, 40, 40])
    c8 = _contrast_uint8(img, [30, 30, 40, 40])
    ratio_lum = c8 / c32
    check('luminance pair: uint8/float32 ~ 2.55 (+-0.05)',
          abs(ratio_lum - 2.55) < 0.05, f'{ratio_lum:.4f}')
    #    2b. pure-chroma pair (equal L): ratio ~ 1.0 -- catches a naive
    #        "fix" that just divides everything by 2.55
    img = np.full((100, 100, 3), 0, dtype=np.uint8)
    img[:, :] = (255, 0, 0)       # pure red (RGB), L ~= 53.2
    img[30:70, 30:70] = (0, 148, 0)  # pure green (RGB), matched luminance
    c32c, _ = box_ring_contrast(img, [30, 30, 40, 40])
    c8c = _contrast_uint8(img, [30, 30, 40, 40])
    ratio_chr = c8c / c32c
    check('chroma pair: uint8/float32 ~ 1.0 (+-0.05)',
          abs(ratio_chr - 1.0) < 0.05, f'{ratio_chr:.4f}')

    # 3. 300 heavily-tied contrast values: rank+qcut OK, raw qcut raises
    import pandas as pd
    tied = np.full(300, 2.0)
    binned = pd.qcut(pd.Series(tied).rank(method='first'), 3,
                     labels=['low', 'mid', 'high'])
    sizes = [int((binned == l).sum()) for l in ('low', 'mid', 'high')]
    check('rank+qcut on 300 ties: bin sizes differ <= 1',
          max(sizes) - min(sizes) <= 1, str(sizes))
    raised = False
    try:
        pd.qcut(pd.Series(tied), 3)
    except Exception:
        raised = True
    check('raw qcut on identical values raises', raised)

    # 4. pairing: 2 src x 2 dir x 5 codes, identical GT geometry
    images, gt_index = {}, {}
    iid = 0
    for s in (1, 2):
        for d in (0, 1):
            for code in CODES:
                images[iid] = {'file_name': f'{s:06d}_{code}_{d:05d}.jpg'}
                gt_index[iid] = [
                    {'image_id': iid, 'category_id': 1,
                     'bbox': [10.0, 10.0, 20.0, 20.0], 'area': 400.0,
                     'iscrowd': 0},
                    {'image_id': iid, 'category_id': 2,
                     'bbox': [50.0, 50.0, 15.0, 15.0], 'area': 225.0,
                     'iscrowd': 0},
                ]
                iid += 1
    instances = build_instance_table(gt_index, images)
    check('instance keys == 2 src x 2 dir x 2 gt',
          len({instance_key(i) for i in instances}) == 8, str(len(instances)))
    dt = [{'image_id': iid_, 'category_id': a['category_id'],
           'bbox': list(a['bbox']), 'score': 0.9}
          for iid_, anns in gt_index.items() for a in anns]
    rows = pair_and_match(instances, gt_index, images, dt,
                          score_thr=0.5, iou_thr=0.7)
    ok_codes = all(row[f'det_{c}'] for row in rows for c in CODES)
    check('all 5 codes matched for every instance', ok_codes)

    # 5. edge box near the image corner -> nan branch, no crash
    img = np.full((60, 60, 3), 200, dtype=np.uint8)
    img[0:5, 0:5] = 30
    contrast, n_ring = box_ring_contrast(img, [0, 0, 4, 4], ring_scale=0.6)
    check('edge box -> nan contrast branch',
          (not np.isfinite(contrast)) and n_ring < 50,
          f'contrast={contrast} n_ring={n_ring}')

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ann', default=None,
                    help='full val ann json (contains all 5 codes)')
    ap.add_argument('--pred', default=None, help='full val pred.json')
    ap.add_argument('--img-dir', default=None,
                    help='image dir (only ODC images are read)')
    ap.add_argument('--out-dir', default=None, required=False)
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--iou-thr', type=float, default=0.7)
    ap.add_argument('--ring-scale', type=float, default=0.6)
    ap.add_argument('--exclude-other-gt', action='store_true',
                    help='remove pixels inside other GT boxes from the ring')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()

    if args.self_test_only:
        sys.exit(self_test())
    for req in ('ann', 'pred', 'img_dir', 'out_dir'):
        if getattr(args, req) is None:
            ap.error(f'--{req.replace("_", "-")} is required '
                     f'(unless --self-test-only)')
    run(args)


if __name__ == '__main__':
    main()
