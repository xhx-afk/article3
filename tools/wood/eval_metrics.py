"""Evaluate a detection checkpoint and save COCO metrics as JSON."""
"""
CUDA_VISIBLE_DEVICES=0,2 torchrun \
  --master_port=7793 \
  --nproc_per_node=2 \
  tools/wood/eval_metrics.py \
  -c configs/deim_dfine/deim_hgnetv2_l_wood.yml \
  -r ./deim_outputs_origin/deim_hgnetv2_l_wood_960/best_stg2.pth \
  -o ./deim_outputs_origin/deim_hgnetv2_l_wood_960/eval_val/metrics.json \
  --seed=0 \
  -u \
  val_dataloader.dataset.img_folder=/home/zxw4090/hjw/D-FINE/data/WoodDefect/wood_coco_all_only_defect_quick_balanced_4000/images/val/ \
  val_dataloader.dataset.ann_file=/home/zxw4090/hjw/D-FINE/data/WoodDefect/wood_coco_all_only_defect_quick_balanced_4000/annotations/instances_val.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig, yaml_utils
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.solver.det_engine import evaluate


COCO_METRIC_NAMES = (
    'AP',
    'AP50',
    'AP75',
    'AP_small',
    'AP_medium',
    'AP_large',
    'AR1',
    'AR10',
    'AR100',
    'AR_small',
    'AR_medium',
    'AR_large',
)


def mean_valid(values):
    values = np.asarray(values)
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def find_index(values, expected):
    for index, value in enumerate(values):
        if value == expected:
            return index
    raise ValueError(f'{expected!r} is not present in {list(values)!r}')


def category_precision(coco_eval, category_index, iou=None, area='all', max_dets=100):
    precision = coco_eval.eval['precision']
    area_index = find_index(coco_eval.params.areaRngLbl, area)
    max_dets_index = find_index(coco_eval.params.maxDets, max_dets)

    if iou is None:
        values = precision[:, :, category_index, area_index, max_dets_index]
    else:
        iou_indices = np.flatnonzero(np.isclose(coco_eval.params.iouThrs, iou))
        if not iou_indices.size:
            return None
        values = precision[iou_indices, :, category_index, area_index, max_dets_index]
    return mean_valid(values)


def category_recall(coco_eval, category_index, area='all', max_dets=100):
    recall = coco_eval.eval['recall']
    area_index = find_index(coco_eval.params.areaRngLbl, area)
    max_dets_index = find_index(coco_eval.params.maxDets, max_dets)
    values = recall[:, category_index, area_index, max_dets_index]
    return mean_valid(values)


def format_category_metrics(coco_evaluator):
    coco_eval = coco_evaluator.coco_eval['bbox']
    categories = {
        int(category['id']): category.get('name', str(category['id']))
        for category in coco_evaluator.coco_gt.dataset.get('categories', [])
    }

    metrics = []
    for category_index, category_id in enumerate(coco_eval.params.catIds):
        category_id = int(category_id)
        metrics.append({
            'category_id': category_id,
            'category_name': categories.get(category_id, str(category_id)),
            'AP': category_precision(coco_eval, category_index),
            'AP50': category_precision(coco_eval, category_index, iou=0.50),
            'AP75': category_precision(coco_eval, category_index, iou=0.75),
            'AP_small': category_precision(coco_eval, category_index, area='small'),
            'AP_medium': category_precision(coco_eval, category_index, area='medium'),
            'AP_large': category_precision(coco_eval, category_index, area='large'),
            'AR1': category_recall(coco_eval, category_index, max_dets=1),
            'AR10': category_recall(coco_eval, category_index, max_dets=10),
            'AR100': category_recall(coco_eval, category_index, max_dets=100),
            'AR_small': category_recall(coco_eval, category_index, area='small'),
            'AR_medium': category_recall(coco_eval, category_index, area='medium'),
            'AR_large': category_recall(coco_eval, category_index, area='large'),
        })
    return metrics


def format_metrics(test_stats, coco_evaluator):
    metrics = {}
    for name, values in test_stats.items():
        values = [float(value) for value in values]
        if name.startswith('coco_eval_') and len(values) == len(COCO_METRIC_NAMES):
            metrics[name] = dict(zip(COCO_METRIC_NAMES, values))
        else:
            metrics[name] = values

    if 'bbox' in coco_evaluator.coco_eval:
        metrics['per_category_bbox'] = format_category_metrics(coco_evaluator)
    return metrics


def save_metrics(metrics, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'{output_path.name}.tmp.{os.getpid()}')
    try:
        with temporary_path.open('w', encoding='utf-8') as file:
            json.dump(metrics, file, indent=2)
            file.write('\n')
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(args):
    dist_utils.setup_distributed(
        args.print_rank,
        args.print_method,
        seed=args.seed,
    )

    try:
        updates = yaml_utils.parse_cli(args.update)
        updates['resume'] = args.resume
        if args.device is not None:
            updates['device'] = args.device

        cfg = YAMLConfig(args.config, **updates)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        solver.eval()

        module = solver.ema.module if solver.ema else solver.model
        test_stats, coco_evaluator = evaluate(
            module,
            solver.criterion,
            solver.postprocessor,
            solver.val_dataloader,
            solver.evaluator,
            solver.device,
        )

        if dist_utils.is_main_process():
            metrics = format_metrics(test_stats, coco_evaluator)
            save_metrics(metrics, args.output_file)
            print(f'Metrics saved to {args.output_file}')
    finally:
        dist_utils.cleanup()


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a DEIM checkpoint and save COCO metrics as JSON.',
    )
    parser.add_argument('-c', '--config', required=True, help='Path to the YAML config.')
    parser.add_argument('-r', '--resume', required=True, help='Path to the trained checkpoint.')
    parser.add_argument(
        '-o',
        '--output-file',
        required=True,
        help='Destination JSON file for evaluation metrics.',
    )
    parser.add_argument('-d', '--device', help='Evaluation device, for example cuda or cpu.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '-u',
        '--update',
        nargs='+',
        help='Override config values, for example val_dataloader.dataset.ann_file=...',
    )
    parser.add_argument('--print-method', default='builtin')
    parser.add_argument('--print-rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int)
    return parser


if __name__ == '__main__':
    main(get_args_parser().parse_args())
