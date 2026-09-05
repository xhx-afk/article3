#!/usr/bin/env python3
"""Dump a DEIM detection checkpoint's raw predictions as a flat COCO results list.

The official evaluation path (engine/solver/det_engine.py: evaluate() +
CocoEvaluator) builds the COCO results list but only feeds it to loadRes();
nothing in the repo persists the per-detection output. This script reuses the
exact same config/model/dataloader/postprocessor wiring as
tools/wood/eval_metrics.py, runs fp32 inference (no AMP, reproducible), and
writes:

    pred.json      [{"image_id", "category_id", "bbox"(xywh), "score"}, ...]
    pred.json.meta.json

The model label is written through unchanged: with
remap_mscoco_category: False the postprocessor label already IS the GT
category_id (engine/deim/postprocessor.py mod(index, num_classes) +
engine/data/dataset/coco_dataset.py), so no mapping is applied here.

Example:
  python tools/wood/dump_predictions.py \
      -c configs/deim_dfine/deim_hgnetv2_s_custom.yml \
      -r ./deim_outputs/.../best_stg2.pth \
      -o ./analysis/pred_val.json \
      -u val_dataloader.dataset.img_folder=... val_dataloader.dataset.ann_file=...

Self-test (no real data / no model / no torch import):
  python tools/wood/dump_predictions.py --self-test-only
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# -----------------------------------------------------------------------------
# Pure conversion helpers (torch-free so --self-test-only stays lightweight)
# -----------------------------------------------------------------------------

def results_to_coco_entries(labels, boxes_xyxy, scores, image_id, score_thr):
    """Convert one image's postprocessor output to flat COCO result entries.

    - keeps detections with score >= score_thr (boundary inclusive);
    - xyxy -> xywh with negative x/y clipped to 0 (w/h kept >= 0);
    - bbox rounded to 2 decimals, score to 5 decimals;
    - label written through as int (no category remapping).
    """
    labels = np.asarray(labels).reshape(-1)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = min(len(labels), len(boxes), len(scores))

    keep = scores[:n] >= score_thr  # inclusive boundary, asserted in self-test
    out = []
    for i in np.where(keep)[0]:
        x1, y1, x2, y2 = boxes[i]
        x = max(float(x1), 0.0)
        y = max(float(y1), 0.0)
        w = max(float(x2) - x, 0.0)
        h = max(float(y2) - y, 0.0)
        out.append({
            'image_id': int(image_id),
            'category_id': int(labels[i]),
            'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            'score': round(float(scores[i]), 5),
        })
    return out


def matched_state(state, params):
    """Shape-aware state-dict intersection (same idea as Solver._matched_state)."""
    missed, unmatched, matched = [], [], {}
    for k, v in state.items():
        if k in params:
            if v.shape == params[k].shape:
                matched[k] = params[k]
            else:
                unmatched.append(k)
        else:
            missed.append(k)
    return matched, {'missed': missed, 'unmatched': unmatched}


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        import subprocess
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _strip_module_prefix(raw):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in raw.items()}


# -----------------------------------------------------------------------------
# Real run
# -----------------------------------------------------------------------------

def run(args):
    import torch

    from engine.core import YAMLConfig, yaml_utils
    from engine.misc import dist_utils
    from engine.solver import TASKS

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    updates = yaml_utils.parse_cli(args.update)
    if args.device:
        updates['device'] = args.device

    cfg = YAMLConfig(args.config, **updates)
    # Do not download pretrained weights at inference time.
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()  # builds model / postprocessor / val_dataloader / evaluator

    # make the EFFECTIVE data paths visible in every run (catches a silently
    # ignored -u override: three dumps on three dirs must not look identical)
    _ds = dist_utils.de_parallel(solver.val_dataloader.dataset)
    print(f'[dump] val dataset: img_folder={getattr(_ds, "img_folder", "?")} '
          f'ann_file={getattr(_ds, "ann_file", "?")}')

    # ---- checkpoint loading: tolerant, shape-matched, explicit weight source
    try:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
    except Exception:
        # older torch or checkpoints with non-tensor payloads; repo-wide
        # convention (eval_metrics.py / _solver.py) uses plain torch.load
        state = torch.load(args.resume, map_location='cpu')
    has_ema = isinstance(state.get('ema'), dict) and 'module' in state['ema']
    if has_ema and not args.use_model_not_ema:
        raw = state['ema']['module']
        weight_source = 'ema'
    else:
        raw = state['model'] if 'model' in state else state
        weight_source = 'model'
    print(f'[dump] using {weight_source} weights from {args.resume}')

    model = dist_utils.de_parallel(solver.model)
    weights = _strip_module_prefix(raw)
    matched, infos = matched_state(model.state_dict(), weights)
    model.load_state_dict(matched, strict=False)
    print(f'[dump] load_state_dict strict=False: '
          f'missed={len(infos["missed"])} unmatched={len(infos["unmatched"])}')
    if infos['missed']:
        print(f'[dump]   missed  (first 10): {infos["missed"][:10]}')
    if infos['unmatched']:
        print(f'[dump]   unmatched (first 10): {infos["unmatched"][:10]}')

    postprocessor = solver.postprocessor
    device = solver.device

    model.eval()
    all_entries = []
    num_images = 0
    # fp32 inference, no AMP, no grad: identical to det_engine.evaluate() L140-154
    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            for t, r in zip(targets, results):
                image_id = int(t['image_id'].item())
                boxes = r['boxes'].detach().cpu().numpy()
                scores = r['scores'].detach().cpu().numpy()
                labels = r['labels'].detach().cpu().numpy()
                all_entries.extend(results_to_coco_entries(
                    labels, boxes, scores, image_id, args.score_thr))
                num_images += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(all_entries, f)
        f.write('\n')

    num_top_queries = getattr(postprocessor, 'num_top_queries', None)
    meta = {
        'config': args.config,
        'update': list(args.update or []),
        'resume': args.resume,
        'resume_md5': file_md5(args.resume),
        'weight_source': weight_source,
        'num_top_queries': num_top_queries,
        'eval_resize_size': cfg.yaml_cfg.get('eval_resize_size'),
        'eval_spatial_size': cfg.yaml_cfg.get('eval_spatial_size'),
        'score_thr': args.score_thr,
        'num_images': num_images,
        'num_detections': len(all_entries),
        'git_commit': git_commit(),
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with Path(str(out_path) + '.meta.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print(f'[dump] wrote {len(all_entries)} detections '
          f'for {num_images} images -> {out_path}')


# -----------------------------------------------------------------------------
# Self-test (synthetic only, no torch / no engine imports)
# -----------------------------------------------------------------------------

def self_test():
    failures = []

    def check(name, cond, detail=''):
        status = 'ok' if cond else 'FAIL'
        print(f'  [{status}] {name}' + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    print('[self-test] dump_predictions')

    # --- xyxy -> xywh conversion, incl. negative-origin clipping
    labels = np.array([0, 1, 2, 3, 0])
    boxes = np.array([
        [10.0, 20.0, 110.0, 70.0],    # normal
        [-5.0, -3.0, 10.0, 12.0],     # negative origin -> clipped to 0
        [0.0, 0.0, 0.0, 0.0],         # degenerate -> w=h=0 kept
        [4.0, 4.0, 2.0, 2.0],         # inverted -> w/h clamped to 0
        [50.0, 50.0, 60.0, 60.0],     # below thr
    ])
    scores = np.array([0.9, 0.5, 0.3, 0.1, 0.049])
    thr = 0.05
    entries = results_to_coco_entries(labels, boxes, scores, image_id=7, score_thr=thr)
    check('score-thr boundary is inclusive (>=)', len(entries) == 4,
          f'kept {len(entries)}')
    e0 = next(e for e in entries if e['score'] == 0.9)
    check('xyxy->xywh normal', e0['bbox'] == [10.0, 20.0, 100.0, 50.0], str(e0['bbox']))
    e1 = next(e for e in entries if e['score'] == 0.5)
    check('negative origin clipped to 0', e1['bbox'] == [0.0, 0.0, 10.0, 12.0],
          str(e1['bbox']))
    check('bbox 2-decimal / score 5-decimal rounding',
          all(v == round(v, 2) for e in entries for v in e['bbox']) and
          all(e['score'] == round(e['score'], 5) for e in entries))

    # --- json round-trip and schema
    with open('_dump_selftest_tmp.json', 'w', encoding='utf-8') as f:
        json.dump(entries, f)
    with open('_dump_selftest_tmp.json', 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    os.unlink('_dump_selftest_tmp.json')
    check('json round-trip', loaded == entries)
    check('every entry has exactly 4 keys',
          all(set(e.keys()) == {'image_id', 'category_id', 'bbox', 'score'}
              for e in loaded))

    # --- optional pycocotools cross-check
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print('  [SKIP] pycocotools not installed')
    else:
        toy_ann = {
            'images': [{'id': 0, 'width': 200, 'height': 200,
                        'file_name': '0_ODC_0.jpg'}],
            'annotations': [
                {'id': 1, 'image_id': 0, 'category_id': 1,
                 'bbox': [10, 10, 50, 50], 'area': 2500, 'iscrowd': 0},
                {'id': 2, 'image_id': 0, 'category_id': 2,
                 'bbox': [100, 100, 40, 40], 'area': 1600, 'iscrowd': 0},
            ],
            'categories': [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}],
        }
        toy_pred = [
            {'image_id': 0, 'category_id': 1,
             'bbox': [10.0, 10.0, 50.0, 50.0], 'score': 0.9},
            {'image_id': 0, 'category_id': 2,
             'bbox': [100.0, 100.0, 40.0, 40.0], 'score': 0.8},
        ]
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix='dump_selftest_')
        ann_path = os.path.join(tmpdir, 'toy_ann.json')
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(toy_ann, f)
        coco_gt = COCO(ann_path)
        coco_dt = coco_gt.loadRes(toy_pred)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        check('pycocotools COCOeval runs on toy ann+pred', ev.stats[0] > 0.99)

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(
        description='Dump DEIM checkpoint predictions as a flat COCO results json.')
    ap.add_argument('-c', '--config', help='training yml used for the checkpoint')
    ap.add_argument('-r', '--resume', help='checkpoint path')
    ap.add_argument('-o', '--output', help='pred.json output path')
    ap.add_argument('-d', '--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('-u', '--update', nargs='+',
                    help='yml overrides, e.g. val_dataloader.dataset.ann_file=...')
    ap.add_argument('--score-thr', type=float, default=0.0,
                    help='keep detections with score >= thr (default keeps all queries)')
    ap.add_argument('--use-model-not-ema', action='store_true',
                    help='use state["model"] even when state["ema"]["module"] exists')
    ap.add_argument('--print-method', default='builtin')
    ap.add_argument('--print-rank', type=int, default=0)
    ap.add_argument('--local-rank', type=int)
    ap.add_argument('--self-test-only', action='store_true',
                    help='run synthetic self-tests only and exit')
    args = ap.parse_args()

    if args.self_test_only:
        sys.exit(self_test())

    for req in ('config', 'resume', 'output'):
        if getattr(args, req) is None:
            ap.error(f'--{req} is required (unless --self-test-only)')
    run(args)


if __name__ == '__main__':
    main()
