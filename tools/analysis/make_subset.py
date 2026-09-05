#!/usr/bin/env python3
"""Cut a COCO ann (and optionally one or more pred files) down to a subset
selected by a regular expression over image file_name.

Typical use (ODC-only evaluation):
    python tools/analysis/make_subset.py \
        --ann  annotations/instances_val.json \
        --pred pred_val.json --keep '_ODC_' \
        --out-dir subsets/odc

Rules (§7.1):
- --keep and --drop are mutually exclusive; the regex is matched against the
  image file_name;
- ann images/annotations AND all pred files are filtered by the SAME retained
  image_id set (ann/pred stay in sync);
- categories / info / licenses are preserved verbatim;
- image_id and annotation id are NEVER renumbered (subsets stay comparable
  with the full set);
- a pred image_id that is absent from the ann images is a hard error
  (pred/ann mismatch).

Self-test:
    python tools/analysis/make_subset.py --self-test-only
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def subset_coco(coco, keep_ids):
    """Filter a COCO dict to the given image_id set (ids preserved verbatim)."""
    out = {k: v for k, v in coco.items()
           if k not in ('images', 'annotations')}
    images = [img for img in coco.get('images', [])
              if int(img['id']) in keep_ids]
    annotations = [a for a in coco.get('annotations', [])
                   if int(a['image_id']) in keep_ids]
    out['images'] = images
    out['annotations'] = annotations
    return out


def subset_pred(pred, keep_ids):
    return [d for d in pred if int(d['image_id']) in keep_ids]


def run(args):
    keep_re = drop_re = None
    if args.keep and args.drop:
        sys.exit('[make_subset] --keep and --drop are mutually exclusive')
    if not args.keep and not args.drop:
        sys.exit('[make_subset] one of --keep / --drop is required')
    if args.keep:
        keep_re = re.compile(args.keep)
    if args.drop:
        drop_re = re.compile(args.drop)

    with open(args.ann, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    ann_image_ids = {int(img['id']) for img in coco.get('images', [])}
    name_by_id = {int(img['id']): img.get('file_name', '')
                  for img in coco.get('images', [])}

    if args.keep:
        keep_ids = {i for i in ann_image_ids if keep_re.search(name_by_id[i])}
    else:
        keep_ids = {i for i in ann_image_ids if not drop_re.search(name_by_id[i])}

    if not keep_ids:
        sys.exit(f'[make_subset] --keep/--drop matched 0 of {len(ann_image_ids)} '
                 f'images; refusing to silently produce an empty subset')

    sub = subset_coco(coco, keep_ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_out = out_dir / 'ann.json'
    with ann_out.open('w', encoding='utf-8') as f:
        json.dump(sub, f)
        f.write('\n')
    print(f'[make_subset] ann: {ann_out}  '
          f'images {len(sub["images"])}/{len(ann_image_ids)}  '
          f'gt {len(sub["annotations"])}')

    for idx, pred_path in enumerate(args.pred or []):
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        unknown = {int(d['image_id']) for d in pred} - ann_image_ids
        if unknown:
            sys.exit(f'[make_subset] pred {pred_path} contains image_ids not '
                     f'present in the ann images (pred/ann mismatch), '
                     f'e.g. {sorted(unknown)[:10]}')
        sub_pred = subset_pred(pred, keep_ids)
        if len(args.pred) == 1:
            out_path = out_dir / 'pred.json'   # single pred -> fixed name
        else:
            out_path = out_dir / (Path(pred_path).stem + '.json')
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(sub_pred, f)
            f.write('\n')
        src_names = {name_by_id[i] for i in keep_ids}
        n_sources = len({re.match(r'^(\d+)_', n).group(1)
                         for n in src_names if re.match(r'^(\d+)_', n)})
        print(f'[make_subset] pred: {out_path}  detections {len(sub_pred)}/{len(pred)}'
              f'  ({n_sources} independent sources)')


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def self_test():
    import tempfile
    print('[self-test] make_subset')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}' + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    # 20 images: 10 ODC (sources 1..10, direction 0), 10 PDC (sources 1..10, dir 1)
    images, annotations = [], []
    aid = 0
    gt_per_image = {}
    for s in range(1, 11):
        for code, direction in (('ODC', 0), ('PDC', 1)):
            iid = s * 10 + direction
            name = f'{s:06d}_{code}_{direction:05d}.jpg'
            images.append({'id': iid, 'file_name': name,
                           'width': 100, 'height': 100})
            n_gt = s % 3 + 1  # 1..3 GT per image
            gt_per_image[iid] = n_gt
            for g in range(n_gt):
                annotations.append({
                    'id': aid, 'image_id': iid, 'category_id': (g % 2) + 1,
                    'bbox': [g * 10.0, 0.0, 10.0, 10.0],
                    'area': 100.0, 'iscrowd': 0,
                })
                aid += 1
    coco = {
        'images': images, 'annotations': annotations,
        'categories': [{'id': 1, 'name': 'scratch'}, {'id': 2, 'name': 'blister'}],
        'info': {'description': 'toy'}, 'licenses': [],
    }

    pred = []
    for iid in [i['id'] for i in images]:
        pred.append({'image_id': iid, 'category_id': 1,
                     'bbox': [0.0, 0.0, 10.0, 10.0], 'score': 0.9})

    tmpdir = Path(tempfile.mkdtemp(prefix='make_subset_selftest_'))
    ann_path = tmpdir / 'ann.json'
    pred_path = tmpdir / 'pred.json'
    with ann_path.open('w', encoding='utf-8') as f:
        json.dump(coco, f)
    with pred_path.open('w', encoding='utf-8') as f:
        json.dump(pred, f)

    # --- 1. --keep '_ODC_' keeps 10 images, matching GT sum, no PDC dets
    out_dir = tmpdir / 'odc'
    sys.argv = ['make_subset.py', '--ann', str(ann_path), '--pred', str(pred_path),
                '--keep', '_ODC_', '--out-dir', str(out_dir)]
    run(argparse.Namespace(ann=str(ann_path), pred=[str(pred_path)],
                           keep='_ODC_', drop=None, out_dir=str(out_dir)))
    with (out_dir / 'ann.json').open(encoding='utf-8') as f:
        sub = json.load(f)
    with (out_dir / 'pred.json').open(encoding='utf-8') as f:
        sub_pred = json.load(f)
    check('10 ODC images kept', len(sub['images']) == 10, str(len(sub['images'])))
    expected_gt = sum(gt_per_image[i] for i in gt_per_image if i % 10 == 0)
    check('GT count == sum over kept images',
          len(sub['annotations']) == expected_gt,
          f'{len(sub["annotations"])} vs {expected_gt}')
    check('pred contains no PDC image_id',
          all(d['image_id'] % 10 == 0 for d in sub_pred))
    check('categories/info/licenses preserved',
          sub['categories'] == coco['categories'] and
          sub['info'] == coco['info'] and sub['licenses'] == coco['licenses'])
    check('ids not renumbered',
          {img['id'] for img in sub['images']} ==
          {i for i in gt_per_image if i % 10 == 0} and
          {a['id'] for a in sub['annotations']} ==
          {a['id'] for a in annotations if a['image_id'] % 10 == 0})

    # --- 2. --keep matching 0 images -> hard error
    rc = _run_capture(lambda: run(argparse.Namespace(
        ann=str(ann_path), pred=[], keep='_XXX_', drop=None,
        out_dir=str(tmpdir / 'empty'))))
    check('0-match --keep errors out', rc != 0)

    # --- 3. pred with unknown image_id -> hard error
    bad_pred_path = tmpdir / 'bad_pred.json'
    with bad_pred_path.open('w', encoding='utf-8') as f:
        json.dump(pred + [{'image_id': 99999, 'category_id': 1,
                           'bbox': [0, 0, 1, 1], 'score': 0.5}], f)
    rc = _run_capture(lambda: run(argparse.Namespace(
        ann=str(ann_path), pred=[str(bad_pred_path)], keep='_ODC_', drop=None,
        out_dir=str(tmpdir / 'bad'))))
    check('pred with unknown image_id errors out', rc != 0)

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def _run_capture(fn):
    """Run fn() with stdout suppressed; return 0 if it returned normally."""
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
        return 0
    except SystemExit as e:
        print(f'  (expected exit: {e})')
        return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ann', default=None, help='full COCO ann json')
    ap.add_argument('--pred', nargs='*', default=None,
                    help='one or more pred.json files, filtered in sync')
    ap.add_argument('--keep', help='regex; keep images whose file_name matches')
    ap.add_argument('--drop', help='regex; mutually exclusive with --keep')
    ap.add_argument('--out-dir', default=None,
                    help='writes ann.json and <pred-stem>.json inside')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()

    if args.self_test_only:
        sys.exit(self_test())
    if not args.ann or not args.out_dir:
        ap.error('--ann and --out-dir are required (unless --self-test-only)')
    run(args)


if __name__ == '__main__':
    main()
