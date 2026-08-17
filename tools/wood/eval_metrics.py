"""Evaluate a DEIM detection checkpoint and save extended COCO diagnostics as JSON.

This script intentionally keeps the original DEIM evaluation path unchanged:
    engine.solver.det_engine.evaluate(...)

Additional diagnostics are computed as a side channel around CocoEvaluator.update(),
so the official/faster-COCO AP/AR values are produced by exactly the same evaluator,
postprocessor, dataloader and distributed synchronization used by the original script.

Example (validation):
CUDA_VISIBLE_DEVICES=0,2 torchrun \
  --master_port=7793 \
  --nproc_per_node=2 \
  tools/wood/eval_metrics.py \
  -c configs/deim_dfine/deim_hgnetv2_l_wood.yml \
  -r ./deim_outputs_origin/deim_hgnetv2_l_wood_960_seed0/best_stg2.pth \
  -o ./deim_outputs_origin/deim_hgnetv2_l_wood_960_seed0/eval_val/fix_val_metrics.json \
  --seed=0 \
  -u \
  val_dataloader.dataset.img_folder=/home/zxw4090/hjw/D-FINE/data/WoodDefect/wood_coco_all_only_defect_quick_balanced_4000/images/val \
  val_dataloader.dataset.ann_file=/home/zxw4090/hjw/D-FINE/data/WoodDefect/wood_coco_all_only_defect_quick_balanced_4000/annotations/instances_val.json

The same command can be used for test by overriding img_folder/ann_file.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
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

DEFAULT_DIAG_SCORE_THRESHOLDS = (0.05, 0.10, 0.25, 0.50, 0.75)
DEFAULT_DIAG_IOU_THRESHOLDS = (0.50, 0.75)
DEFAULT_SCORE_HIST_BINS = (
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95, 1.000001,
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def safe_f1(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mean_valid(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > -1)]
    return float(values.mean()) if values.size else None


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            'mean': None,
            'p10': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'p90': None,
        }
    return {
        'mean': float(values.mean()),
        'p10': float(np.percentile(values, 10)),
        'p25': float(np.percentile(values, 25)),
        'p50': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
    }


def find_index(values, expected):
    for index, value in enumerate(values):
        if value == expected:
            return index
    raise ValueError(f'{expected!r} is not present in {list(values)!r}')


def find_float_index(values, expected):
    values = np.asarray(values, dtype=np.float64)
    indices = np.flatnonzero(np.isclose(values, expected))
    return int(indices[0]) if indices.size else None


def as_numpy(value):
    """Convert torch/numpy/list values to a CPU numpy array without importing torch."""
    if value is None:
        return np.asarray([])
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return np.asarray([x, y, x + max(w, 0.0), y + max(h, 0.0)], dtype=np.float64)


def box_iou_matrix(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], 0.0, None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], 0.0, None
    )
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def normalize_float_list(values, minimum=0.0, maximum=1.0):
    cleaned = []
    for value in values:
        value = float(value)
        if not minimum <= value <= maximum:
            raise ValueError(f'Value {value} is outside [{minimum}, {maximum}].')
        cleaned.append(value)
    # Stable deduplication after rounding avoids duplicate JSON operating points.
    return sorted(set(round(v, 8) for v in cleaned))


# -----------------------------------------------------------------------------
# COCOeval tensor readers (do not alter COCO evaluation)
# -----------------------------------------------------------------------------

def coco_precision(coco_eval, category_index=None, iou=None, area='all', max_dets=100):
    precision = np.asarray(coco_eval.eval['precision'])
    area_index = find_index(coco_eval.params.areaRngLbl, area)
    max_dets_index = find_index(coco_eval.params.maxDets, max_dets)

    values = precision[:, :, :, area_index, max_dets_index]
    if iou is not None:
        iou_index = find_float_index(coco_eval.params.iouThrs, iou)
        if iou_index is None:
            return None
        values = values[iou_index:iou_index + 1]
    if category_index is not None:
        values = values[:, :, category_index:category_index + 1]
    return mean_valid(values)


def coco_recall(coco_eval, category_index=None, iou=None, area='all', max_dets=100):
    recall = np.asarray(coco_eval.eval['recall'])
    area_index = find_index(coco_eval.params.areaRngLbl, area)
    max_dets_index = find_index(coco_eval.params.maxDets, max_dets)

    values = recall[:, :, area_index, max_dets_index]
    if iou is not None:
        iou_index = find_float_index(coco_eval.params.iouThrs, iou)
        if iou_index is None:
            return None
        values = values[iou_index:iou_index + 1]
    if category_index is not None:
        values = values[:, category_index:category_index + 1]
    return mean_valid(values)


def precision_at_recall(coco_eval, category_index, iou, recall_target, area='all', max_dets=100):
    iou_index = find_float_index(coco_eval.params.iouThrs, iou)
    if iou_index is None:
        return None
    recall_index = find_float_index(coco_eval.params.recThrs, recall_target)
    if recall_index is None:
        return None
    area_index = find_index(coco_eval.params.areaRngLbl, area)
    max_dets_index = find_index(coco_eval.params.maxDets, max_dets)
    value = np.asarray(coco_eval.eval['precision'])[
        iou_index, recall_index, category_index, area_index, max_dets_index
    ]
    return None if value < 0 else float(value)


def best_f1_from_coco_curve(coco_eval, category_index, iou, area='all', max_dets=100):
    """Best F1 on COCO's interpolated PR curve for one category and IoU.

    If eval['scores'] is available, score_at_best_f1 is the COCO score associated
    with the selected recall point. This is a diagnostic operating point, not a
    replacement for AP and not a globally optimal multi-class threshold.
    """
    iou_index = find_float_index(coco_eval.params.iouThrs, iou)
    if iou_index is None:
        return None
    area_index = find_index(coco_eval.params.areaRngLbl, area)
    max_dets_index = find_index(coco_eval.params.maxDets, max_dets)

    precision = np.asarray(coco_eval.eval['precision'])[
        iou_index, :, category_index, area_index, max_dets_index
    ].astype(np.float64)
    recalls = np.asarray(coco_eval.params.recThrs, dtype=np.float64)
    valid = np.isfinite(precision) & (precision >= 0)
    if not valid.any():
        return None

    f1 = np.zeros_like(precision)
    denom = precision + recalls
    positive = valid & (denom > 0)
    f1[positive] = 2.0 * precision[positive] * recalls[positive] / denom[positive]
    f1[valid & ~positive] = 0.0
    f1[~valid] = -1.0
    best_index = int(np.argmax(f1))

    score_value = None
    scores = coco_eval.eval.get('scores') if isinstance(coco_eval.eval, dict) else None
    if scores is not None:
        scores = np.asarray(scores)
        if scores.ndim == 5:
            raw_score = scores[
                iou_index, best_index, category_index, area_index, max_dets_index
            ]
            if np.isfinite(raw_score) and raw_score >= 0:
                score_value = float(raw_score)

    return {
        'f1': float(f1[best_index]),
        'precision': float(precision[best_index]),
        'recall': float(recalls[best_index]),
        'score_at_best_f1': score_value,
    }


def format_iou_sweep(coco_eval):
    result = []
    for iou in np.asarray(coco_eval.params.iouThrs, dtype=np.float64):
        result.append({
            'iou': float(iou),
            'AP': coco_precision(coco_eval, iou=float(iou)),
            'AR100': coco_recall(coco_eval, iou=float(iou), max_dets=100),
        })
    return result


def format_size_iou_diagnostics(coco_eval):
    result = []
    for area in ('small', 'medium', 'large'):
        result.append({
            'area': area,
            'AP': coco_precision(coco_eval, area=area),
            'AP50': coco_precision(coco_eval, iou=0.50, area=area),
            'AP75': coco_precision(coco_eval, iou=0.75, area=area),
            'AR100': coco_recall(coco_eval, area=area, max_dets=100),
            'AR50': coco_recall(coco_eval, iou=0.50, area=area, max_dets=100),
            'AR75': coco_recall(coco_eval, iou=0.75, area=area, max_dets=100),
        })
    return result


def format_category_metrics(coco_evaluator):
    coco_eval = coco_evaluator.coco_eval['bbox']
    categories = {
        int(category['id']): category.get('name', str(category['id']))
        for category in coco_evaluator.coco_gt.dataset.get('categories', [])
    }

    metrics = []
    for category_index, category_id in enumerate(coco_eval.params.catIds):
        category_id = int(category_id)
        ap50 = coco_precision(coco_eval, category_index, iou=0.50)
        ap75 = coco_precision(coco_eval, category_index, iou=0.75)
        ap90 = coco_precision(coco_eval, category_index, iou=0.90)
        ar50 = coco_recall(coco_eval, category_index, iou=0.50, max_dets=100)
        ar75 = coco_recall(coco_eval, category_index, iou=0.75, max_dets=100)
        ar90 = coco_recall(coco_eval, category_index, iou=0.90, max_dets=100)

        item = {
            'category_id': category_id,
            'category_name': categories.get(category_id, str(category_id)),

            # Backward-compatible fields from the original script.
            'AP': coco_precision(coco_eval, category_index),
            'AP50': ap50,
            'AP75': ap75,
            'AP_small': coco_precision(coco_eval, category_index, area='small'),
            'AP_medium': coco_precision(coco_eval, category_index, area='medium'),
            'AP_large': coco_precision(coco_eval, category_index, area='large'),
            'AR1': coco_recall(coco_eval, category_index, max_dets=1),
            'AR10': coco_recall(coco_eval, category_index, max_dets=10),
            'AR100': coco_recall(coco_eval, category_index, max_dets=100),
            'AR_small': coco_recall(coco_eval, category_index, area='small'),
            'AR_medium': coco_recall(coco_eval, category_index, area='medium'),
            'AR_large': coco_recall(coco_eval, category_index, area='large'),

            # Strict-localization diagnostics.
            'AP90': ap90,
            'AP95': coco_precision(coco_eval, category_index, iou=0.95),
            'AR50': ar50,
            'AR75': ar75,
            'AR90': ar90,
            'AR95': coco_recall(coco_eval, category_index, iou=0.95, max_dets=100),
            'AP50_minus_AP75': None if ap50 is None or ap75 is None else float(ap50 - ap75),
            'AP75_over_AP50': safe_div(ap75, ap50) if ap50 not in (None, 0.0) else None,
            'AP90_over_AP50': safe_div(ap90, ap50) if ap50 not in (None, 0.0) else None,
            'AR75_over_AR50': safe_div(ar75, ar50) if ar50 not in (None, 0.0) else None,

            # COCO interpolated precision at fixed recall targets.
            'P_at_R50_IoU50': precision_at_recall(coco_eval, category_index, 0.50, 0.50),
            'P_at_R75_IoU50': precision_at_recall(coco_eval, category_index, 0.50, 0.75),
            'P_at_R90_IoU50': precision_at_recall(coco_eval, category_index, 0.50, 0.90),
            'P_at_R50_IoU75': precision_at_recall(coco_eval, category_index, 0.75, 0.50),
            'P_at_R75_IoU75': precision_at_recall(coco_eval, category_index, 0.75, 0.75),

            # Size + IoU localization diagnostics.
            'AP50_small': coco_precision(coco_eval, category_index, iou=0.50, area='small'),
            'AP50_medium': coco_precision(coco_eval, category_index, iou=0.50, area='medium'),
            'AP50_large': coco_precision(coco_eval, category_index, iou=0.50, area='large'),
            'AP75_small': coco_precision(coco_eval, category_index, iou=0.75, area='small'),
            'AP75_medium': coco_precision(coco_eval, category_index, iou=0.75, area='medium'),
            'AP75_large': coco_precision(coco_eval, category_index, iou=0.75, area='large'),
        }

        item['AP_by_IoU'] = {
            f'{float(iou):.2f}': coco_precision(coco_eval, category_index, iou=float(iou))
            for iou in np.asarray(coco_eval.params.iouThrs, dtype=np.float64)
        }
        item['AR100_by_IoU'] = {
            f'{float(iou):.2f}': coco_recall(
                coco_eval, category_index, iou=float(iou), max_dets=100
            )
            for iou in np.asarray(coco_eval.params.iouThrs, dtype=np.float64)
        }
        item['best_F1_IoU50'] = best_f1_from_coco_curve(coco_eval, category_index, 0.50)
        item['best_F1_IoU75'] = best_f1_from_coco_curve(coco_eval, category_index, 0.75)
        metrics.append(item)

    return metrics


# -----------------------------------------------------------------------------
# Dataset / GT geometry summary
# -----------------------------------------------------------------------------

def format_dataset_summary(coco_gt):
    categories = {
        int(category['id']): category.get('name', str(category['id']))
        for category in coco_gt.dataset.get('categories', [])
    }
    images = {
        int(image['id']): image
        for image in coco_gt.dataset.get('images', [])
    }

    per_category_raw = {
        category_id: {
            'width': [],
            'height': [],
            'area': [],
            'aspect_ratio_w_over_h': [],
            'relative_width': [],
            'relative_height': [],
            'relative_area': [],
            'image_ids': set(),
            'size_counts': {'small': 0, 'medium': 0, 'large': 0},
            'gt_count': 0,
            'ignored_or_crowd_count': 0,
        }
        for category_id in categories
    }

    image_instance_counts = defaultdict(int)
    total_valid = 0
    total_ignored = 0
    global_size_counts = {'small': 0, 'medium': 0, 'large': 0}

    for ann in coco_gt.dataset.get('annotations', []):
        category_id = int(ann.get('category_id', -1))
        if category_id not in per_category_raw:
            continue

        if int(ann.get('iscrowd', 0)) == 1 or int(ann.get('ignore', 0)) == 1:
            per_category_raw[category_id]['ignored_or_crowd_count'] += 1
            total_ignored += 1
            continue

        bbox = ann.get('bbox', None)
        if bbox is None or len(bbox) < 4:
            continue
        x, y, width, height = [float(v) for v in bbox[:4]]
        width = max(width, 0.0)
        height = max(height, 0.0)
        area = float(ann.get('area', width * height))
        if not math.isfinite(area) or area < 0:
            area = width * height

        image_id = int(ann.get('image_id'))
        image = images.get(image_id, {})
        image_width = float(image.get('width', 0) or 0)
        image_height = float(image.get('height', 0) or 0)

        raw = per_category_raw[category_id]
        raw['gt_count'] += 1
        raw['image_ids'].add(image_id)
        raw['width'].append(width)
        raw['height'].append(height)
        raw['area'].append(area)
        raw['aspect_ratio_w_over_h'].append(width / height if height > 0 else np.nan)
        raw['relative_width'].append(width / image_width if image_width > 0 else np.nan)
        raw['relative_height'].append(height / image_height if image_height > 0 else np.nan)
        raw['relative_area'].append(
            area / (image_width * image_height)
            if image_width > 0 and image_height > 0
            else np.nan
        )

        if area < 32.0 ** 2:
            size_name = 'small'
        elif area < 96.0 ** 2:
            size_name = 'medium'
        else:
            size_name = 'large'
        raw['size_counts'][size_name] += 1
        global_size_counts[size_name] += 1
        image_instance_counts[image_id] += 1
        total_valid += 1

    per_category = []
    nonzero_counts = []
    for category_id in categories:
        raw = per_category_raw[category_id]
        if raw['gt_count'] > 0:
            nonzero_counts.append(raw['gt_count'])
        per_category.append({
            'category_id': category_id,
            'category_name': categories[category_id],
            'gt_count': int(raw['gt_count']),
            'image_count': int(len(raw['image_ids'])),
            'ignored_or_crowd_count': int(raw['ignored_or_crowd_count']),
            'size_counts_coco_definition': raw['size_counts'],
            'bbox_width_px': percentile_summary(raw['width']),
            'bbox_height_px': percentile_summary(raw['height']),
            'bbox_area_px2': percentile_summary(raw['area']),
            'aspect_ratio_w_over_h': percentile_summary(raw['aspect_ratio_w_over_h']),
            'relative_width_to_image': percentile_summary(raw['relative_width']),
            'relative_height_to_image': percentile_summary(raw['relative_height']),
            'relative_area_to_image': percentile_summary(raw['relative_area']),
        })

    counts_all_images = [image_instance_counts.get(image_id, 0) for image_id in images]
    class_balance = {
        'num_categories': len(categories),
        'num_nonempty_categories': len(nonzero_counts),
        'min_gt_count_nonzero': int(min(nonzero_counts)) if nonzero_counts else 0,
        'max_gt_count': int(max(nonzero_counts)) if nonzero_counts else 0,
        'max_to_min_gt_ratio': (
            float(max(nonzero_counts) / min(nonzero_counts)) if nonzero_counts else None
        ),
        'coefficient_of_variation_gt_count': (
            float(np.std(nonzero_counts) / np.mean(nonzero_counts))
            if nonzero_counts and np.mean(nonzero_counts) > 0
            else None
        ),
    }

    return {
        'num_images': len(images),
        'num_images_with_valid_gt': int(sum(count > 0 for count in counts_all_images)),
        'num_valid_gt_instances': int(total_valid),
        'num_ignored_or_crowd_gt_instances': int(total_ignored),
        'instances_per_image': percentile_summary(counts_all_images),
        'size_counts_coco_definition': global_size_counts,
        'class_balance': class_balance,
        'per_category': per_category,
        'coco_size_definition_px2': {
            'small': '[0, 32^2)',
            'medium': '[32^2, 96^2)',
            'large': '[96^2, +inf)',
        },
    }


# -----------------------------------------------------------------------------
# Side-channel threshold / confusion diagnostics
# -----------------------------------------------------------------------------

class DetectionDiagnosticCollector:
    """Collect threshold-based diagnostics without changing COCO evaluator input."""

    def __init__(
        self,
        coco_gt,
        score_thresholds,
        iou_thresholds,
        primary_score_threshold=0.25,
        primary_iou_threshold=0.50,
        localization_floor_iou=0.10,
        score_hist_bins=DEFAULT_SCORE_HIST_BINS,
    ):
        self.coco_gt = coco_gt
        self.score_thresholds = normalize_float_list(score_thresholds)
        self.iou_thresholds = normalize_float_list(iou_thresholds)
        self.primary_score_threshold = float(primary_score_threshold)
        self.primary_iou_threshold = float(primary_iou_threshold)
        self.localization_floor_iou = float(localization_floor_iou)
        self.score_hist_bins = np.asarray(score_hist_bins, dtype=np.float64)

        self.category_ids = [
            int(category['id']) for category in coco_gt.dataset.get('categories', [])
        ]
        self.category_names = {
            int(category['id']): category.get('name', str(category['id']))
            for category in coco_gt.dataset.get('categories', [])
        }
        self.category_to_index = {
            category_id: index for index, category_id in enumerate(self.category_ids)
        }
        self.background_index = len(self.category_ids)
        self.reset()

    def reset(self):
        self.stats = defaultdict(
            lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0, 'gt': 0})
        )
        size = len(self.category_ids) + 1
        self.confusion = np.zeros((size, size), dtype=np.int64)
        self.error_breakdown = defaultdict(int)
        self.hard_images = []
        self.score_hist = {
            category_id: np.zeros(len(self.score_hist_bins) - 1, dtype=np.int64)
            for category_id in self.category_ids
        }
        self.unknown_prediction_labels = defaultdict(int)
        self.seen_image_ids = set()
        # Keep compact per-image diagnostics so distributed synchronization can
        # deduplicate padded/repeated image_ids exactly like COCO evaluation.
        self.image_records = {}

    @staticmethod
    def _op_key(score_threshold, iou_threshold):
        return f'score={score_threshold:.4f}|iou={iou_threshold:.4f}'

    def _gt_for_image(self, image_id):
        gt_boxes = []
        gt_labels = []
        for ann in self.coco_gt.imgToAnns.get(int(image_id), []):
            if int(ann.get('iscrowd', 0)) == 1 or int(ann.get('ignore', 0)) == 1:
                continue
            bbox = ann.get('bbox')
            if bbox is None or len(bbox) < 4:
                continue
            category_id = int(ann.get('category_id', -1))
            if category_id not in self.category_to_index:
                continue
            gt_boxes.append(xywh_to_xyxy(bbox[:4]))
            gt_labels.append(category_id)
        if gt_boxes:
            gt_boxes = np.stack(gt_boxes, axis=0)
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float64)
        return gt_boxes, np.asarray(gt_labels, dtype=np.int64)

    def _prediction_arrays(self, prediction):
        boxes = as_numpy(prediction.get('boxes')).astype(np.float64).reshape(-1, 4)
        scores = as_numpy(prediction.get('scores')).astype(np.float64).reshape(-1)
        labels = as_numpy(prediction.get('labels')).astype(np.int64).reshape(-1)
        count = min(len(boxes), len(scores), len(labels))
        boxes, scores, labels = boxes[:count], scores[:count], labels[:count]

        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        return boxes, scores, labels

    def _class_aware_counts(self, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
                            score_threshold, iou_threshold):
        counts = {}
        for category_id in self.category_ids:
            gt_mask = gt_labels == category_id
            pred_mask = (pred_labels == category_id) & (pred_scores >= score_threshold)
            category_gt = gt_boxes[gt_mask]
            category_pred = pred_boxes[pred_mask]
            category_scores = pred_scores[pred_mask]

            if len(category_pred):
                order = np.argsort(-category_scores, kind='mergesort')
                category_pred = category_pred[order]

            matched_gt = np.zeros(len(category_gt), dtype=bool)
            tp = 0
            fp = 0
            if len(category_pred) and len(category_gt):
                ious = box_iou_matrix(category_pred, category_gt)
                for pred_index in range(len(category_pred)):
                    candidates = np.where(~matched_gt)[0]
                    if not len(candidates):
                        fp += 1
                        continue
                    local = ious[pred_index, candidates]
                    best_local_index = int(np.argmax(local))
                    best_gt_index = int(candidates[best_local_index])
                    if local[best_local_index] >= iou_threshold:
                        matched_gt[best_gt_index] = True
                        tp += 1
                    else:
                        fp += 1
            else:
                fp = len(category_pred)

            fn = int(len(category_gt) - tp)
            counts[category_id] = {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'pred': int(len(category_pred)),
                'gt': int(len(category_gt)),
            }
        return counts

    def _confusion_and_errors(self, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels):
        """Return per-image primary-point confusion/error diagnostics.

        This is intentionally independent of the official COCO matching logic. It is a
        class-aware diagnostic at one explicit score/IoU operating point. All returned
        arrays/counters are local to the image so distributed synchronization can
        deduplicate by image_id before global summation.
        """
        score_threshold = self.primary_score_threshold
        iou_threshold = self.primary_iou_threshold

        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        if len(pred_scores):
            order = np.argsort(-pred_scores, kind='mergesort')
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]

        size = len(self.category_ids) + 1
        local_confusion = np.zeros((size, size), dtype=np.int64)
        local_errors = defaultdict(int)
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        ious = box_iou_matrix(pred_boxes, gt_boxes)
        per_image = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'pred': int(len(pred_boxes)),
            'gt': int(len(gt_boxes)),
            'class_confusion': 0,
            'background_fp': 0,
            'localization_fp': 0,
            'duplicate_or_competing_fp': 0,
            'other_object_overlap_fp': 0,
        }

        for pred_index in range(len(pred_boxes)):
            pred_label = int(pred_labels[pred_index])
            pred_col = self.category_to_index.get(pred_label)
            # Unknown labels are tracked separately; there is no matching matrix column
            # for them, so leave them out of this known-class confusion view.
            if pred_col is None:
                continue

            unmatched_candidates = np.where(~matched_gt)[0]
            matched = False
            if len(unmatched_candidates):
                local_ious = ious[pred_index, unmatched_candidates]
                best_local_index = int(np.argmax(local_ious))
                best_gt_index = int(unmatched_candidates[best_local_index])
                best_iou = float(local_ious[best_local_index])
                if best_iou >= iou_threshold:
                    matched_gt[best_gt_index] = True
                    gt_label = int(gt_labels[best_gt_index])
                    gt_row = self.category_to_index[gt_label]
                    local_confusion[gt_row, pred_col] += 1
                    matched = True
                    if gt_label == pred_label:
                        per_image['tp'] += 1
                        local_errors['correct_tp'] += 1
                    else:
                        # A class confusion is simultaneously an FP for the predicted
                        # class and an FN for the GT class in class-aware P/R/F1.
                        per_image['fp'] += 1
                        per_image['class_confusion'] += 1
                        local_errors['classification_confusion'] += 1

            if matched:
                continue

            # Unmatched known-class prediction -> background GT row.
            local_confusion[self.background_index, pred_col] += 1
            per_image['fp'] += 1

            best_any_iou = 0.0
            best_any_gt_label = None
            best_same_iou = 0.0
            if len(gt_boxes):
                all_ious = ious[pred_index]
                best_any_gt_index = int(np.argmax(all_ious))
                best_any_iou = float(all_ious[best_any_gt_index])
                best_any_gt_label = int(gt_labels[best_any_gt_index])
                same_indices = np.where(gt_labels == pred_label)[0]
                if len(same_indices):
                    best_same_iou = float(np.max(all_ious[same_indices]))

            if best_any_iou >= iou_threshold:
                bucket = 'duplicate_or_competing_fp'
            elif best_same_iou >= self.localization_floor_iou:
                bucket = 'localization_fp'
            elif best_any_iou >= self.localization_floor_iou and best_any_gt_label != pred_label:
                bucket = 'other_object_overlap_fp'
            else:
                bucket = 'background_fp'
            per_image[bucket] += 1
            local_errors[bucket] += 1

        unmatched_gt_indices = np.where(~matched_gt)[0]
        for gt_index in unmatched_gt_indices:
            gt_label = int(gt_labels[gt_index])
            gt_row = self.category_to_index[gt_label]
            local_confusion[gt_row, self.background_index] += 1

        unmatched_fn = int(len(unmatched_gt_indices))
        # Wrong-class matched GT are also class-aware false negatives.
        per_image['fn'] = unmatched_fn + int(per_image['class_confusion'])
        local_errors['unmatched_gt_fn'] += unmatched_fn
        local_errors['class_confusion_fn'] += int(per_image['class_confusion'])
        return per_image, local_confusion, dict(local_errors)

    def update(self, predictions):
        for image_id, prediction in predictions.items():
            image_id = int(image_id)
            gt_boxes, gt_labels = self._gt_for_image(image_id)
            pred_boxes, pred_scores, pred_labels = self._prediction_arrays(prediction)

            image_hist = {
                category_id: np.zeros(len(self.score_hist_bins) - 1, dtype=np.int64)
                for category_id in self.category_ids
            }
            for category_id in self.category_ids:
                scores = pred_scores[pred_labels == category_id]
                if len(scores):
                    hist, _ = np.histogram(scores, bins=self.score_hist_bins)
                    image_hist[category_id] += hist.astype(np.int64)

            image_unknown = defaultdict(int)
            for unknown_label in pred_labels[~np.isin(pred_labels, self.category_ids)]:
                image_unknown[int(unknown_label)] += 1

            image_op_stats = {}
            for score_threshold in self.score_thresholds:
                for iou_threshold in self.iou_thresholds:
                    op_key = self._op_key(score_threshold, iou_threshold)
                    counts = self._class_aware_counts(
                        gt_boxes,
                        gt_labels,
                        pred_boxes,
                        pred_scores,
                        pred_labels,
                        score_threshold,
                        iou_threshold,
                    )
                    image_op_stats[op_key] = {
                        str(category_id): {
                            field: int(item[field])
                            for field in ('tp', 'fp', 'fn', 'pred', 'gt')
                        }
                        for category_id, item in counts.items()
                    }

            primary_image_summary, local_confusion, local_errors = self._confusion_and_errors(
                gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels
            )
            precision = safe_div(
                primary_image_summary['tp'],
                primary_image_summary['tp'] + primary_image_summary['fp'],
            )
            recall = safe_div(
                primary_image_summary['tp'],
                primary_image_summary['tp'] + primary_image_summary['fn'],
            )
            f1 = safe_f1(precision, recall)
            image_info = self.coco_gt.imgs.get(image_id, {})
            hard = {
                'image_id': image_id,
                'file_name': image_info.get('file_name'),
                **primary_image_summary,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

            # Store one compact record per image. Assignment (rather than append) also
            # removes any accidental within-rank duplicate image_id. Global duplicates
            # caused by distributed sampler padding are removed after all_gather().
            self.image_records[image_id] = {
                'image_id': image_id,
                'op_stats': image_op_stats,
                'confusion': local_confusion.tolist(),
                'errors': {str(k): int(v) for k, v in local_errors.items()},
                'score_hist': {
                    str(category_id): hist.tolist()
                    for category_id, hist in image_hist.items()
                },
                'unknown_prediction_labels': {
                    str(label): int(count) for label, count in image_unknown.items()
                },
                'hard': hard,
            }

        # Keep local aggregates useful before synchronize() (e.g. single-process or
        # debugging). The same rebuild routine is used, so semantics are identical.
        self._rebuild_from_unique_records(self.image_records)

    def payload(self):
        return {'image_records': list(self.image_records.values())}

    def _rebuild_from_unique_records(self, records_by_id):
        merged_stats = defaultdict(
            lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0, 'gt': 0})
        )
        size = len(self.category_ids) + 1
        merged_confusion = np.zeros((size, size), dtype=np.int64)
        merged_errors = defaultdict(int)
        merged_hard_images = []
        merged_hist = {
            category_id: np.zeros(len(self.score_hist_bins) - 1, dtype=np.int64)
            for category_id in self.category_ids
        }
        merged_unknown = defaultdict(int)

        for image_id in sorted(records_by_id):
            record = records_by_id[image_id]
            for op_key, per_category in record.get('op_stats', {}).items():
                for category_id_string, values in per_category.items():
                    category_id = int(category_id_string)
                    for field in ('tp', 'fp', 'fn', 'pred', 'gt'):
                        merged_stats[op_key][category_id][field] += int(values.get(field, 0))

            local_confusion = np.asarray(record.get('confusion', []), dtype=np.int64)
            if local_confusion.shape == merged_confusion.shape:
                merged_confusion += local_confusion
            for key, value in record.get('errors', {}).items():
                merged_errors[str(key)] += int(value)
            hard = record.get('hard')
            if hard is not None:
                merged_hard_images.append(hard)
            for category_id_string, hist in record.get('score_hist', {}).items():
                category_id = int(category_id_string)
                if category_id in merged_hist:
                    merged_hist[category_id] += np.asarray(hist, dtype=np.int64)
            for label, count in record.get('unknown_prediction_labels', {}).items():
                merged_unknown[int(label)] += int(count)

        self.stats = merged_stats
        self.confusion = merged_confusion
        self.error_breakdown = merged_errors
        self.hard_images = merged_hard_images
        self.score_hist = merged_hist
        self.unknown_prediction_labels = merged_unknown
        self.seen_image_ids = set(int(v) for v in records_by_id)

    def merge_payloads(self, payloads):
        # Match COCO evaluator behavior conceptually: image_id is the unit of
        # deduplication. Rank order is deterministic; the first copy is retained.
        unique_records = {}
        for payload in payloads:
            for record in payload.get('image_records', []):
                image_id = int(record['image_id'])
                if image_id not in unique_records:
                    unique_records[image_id] = record

        self.image_records = unique_records
        self._rebuild_from_unique_records(unique_records)

    def synchronize_between_processes(self):
        payloads = dist_utils.all_gather(self.payload())
        self.merge_payloads(payloads)

    def _format_count_metrics(self, counts):
        tp = int(counts['tp'])
        fp = int(counts['fp'])
        fn = int(counts['fn'])
        pred_count = int(counts['pred'])
        gt_count = int(counts['gt'])

        # For a class with GT but no predictions, precision=0 is more useful for
        # macro diagnostics than None (which would silently exclude a failed class).
        if tp + fp == 0:
            precision = 0.0 if gt_count > 0 else None
        else:
            precision = float(tp) / float(tp + fp)
        recall = 0.0 if gt_count > 0 and tp + fn == 0 else safe_div(tp, tp + fn)
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'pred_count': pred_count,
            'gt_count': gt_count,
            'precision': precision,
            'recall': recall,
            'f1': safe_f1(precision, recall),
        }

    def format_operating_points(self):
        result = []
        for score_threshold in self.score_thresholds:
            for iou_threshold in self.iou_thresholds:
                op_key = self._op_key(score_threshold, iou_threshold)
                per_category = []
                summed = {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0, 'gt': 0}
                macro_precision = []
                macro_recall = []
                macro_f1 = []

                for category_id in self.category_ids:
                    counts = self.stats[op_key][category_id]
                    formatted = self._format_count_metrics(counts)
                    formatted.update({
                        'category_id': category_id,
                        'category_name': self.category_names[category_id],
                    })
                    per_category.append(formatted)
                    for field in summed:
                        summed[field] += int(counts[field])
                    if formatted['gt_count'] > 0:
                        if formatted['precision'] is not None:
                            macro_precision.append(formatted['precision'])
                        if formatted['recall'] is not None:
                            macro_recall.append(formatted['recall'])
                        if formatted['f1'] is not None:
                            macro_f1.append(formatted['f1'])

                micro = self._format_count_metrics(summed)
                macro = {
                    'precision': float(np.mean(macro_precision)) if macro_precision else None,
                    'recall': float(np.mean(macro_recall)) if macro_recall else None,
                    'f1': float(np.mean(macro_f1)) if macro_f1 else None,
                }
                result.append({
                    'score_threshold': float(score_threshold),
                    'iou_threshold': float(iou_threshold),
                    'micro': micro,
                    'macro_over_categories_with_gt': macro,
                    'per_category': per_category,
                })
        return result

    def format_confusion_matrix(self):
        labels = [self.category_names[cid] for cid in self.category_ids] + ['__background__']
        category_ids = self.category_ids + [None]
        matrix = self.confusion.astype(np.int64)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix.astype(np.float64),
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums > 0,
        )

        confusion_pairs = []
        for row, gt_category_id in enumerate(self.category_ids):
            for col, pred_category_id in enumerate(self.category_ids):
                if row == col:
                    continue
                count = int(matrix[row, col])
                if count > 0:
                    confusion_pairs.append({
                        'gt_category_id': gt_category_id,
                        'gt_category_name': self.category_names[gt_category_id],
                        'pred_category_id': pred_category_id,
                        'pred_category_name': self.category_names[pred_category_id],
                        'count': count,
                    })
        confusion_pairs.sort(key=lambda x: (-x['count'], x['gt_category_name'], x['pred_category_name']))

        return {
            'score_threshold': self.primary_score_threshold,
            'iou_threshold': self.primary_iou_threshold,
            'row_definition': 'ground-truth class; final row is background/unmatched prediction',
            'column_definition': 'predicted class; final column is background/unmatched ground truth',
            'category_ids': category_ids,
            'labels': labels,
            'matrix_gt_rows_pred_cols': matrix.tolist(),
            'row_normalized_matrix': normalized.tolist(),
            'class_confusion_pairs_nonzero': confusion_pairs,
        }

    def format_score_histogram(self):
        bins = [float(v) for v in self.score_hist_bins]
        per_category = []
        total_hist = np.zeros(len(bins) - 1, dtype=np.int64)
        for category_id in self.category_ids:
            hist = self.score_hist[category_id]
            total_hist += hist
            per_category.append({
                'category_id': category_id,
                'category_name': self.category_names[category_id],
                'counts': hist.tolist(),
            })
        return {
            'bin_edges': bins,
            'bin_semantics': '[left, right), except the final bin effectively includes score=1.0',
            'all_predictions_counts': total_hist.tolist(),
            'per_category': per_category,
            'unknown_prediction_labels': {
                str(label): int(count) for label, count in sorted(self.unknown_prediction_labels.items())
            },
        }

    def format_hard_images(self, limit=100):
        # Highest FN first, then FP, then lower F1; deterministic image-id tie-break.
        def sort_key(item):
            f1 = item.get('f1')
            f1_key = 1.0 if f1 is None else float(f1)
            return (-int(item.get('fn', 0)), -int(item.get('fp', 0)), f1_key, int(item['image_id']))

        items = sorted(self.hard_images, key=sort_key)
        if limit is not None and limit >= 0:
            items = items[:limit]
        return {
            'score_threshold': self.primary_score_threshold,
            'iou_threshold': self.primary_iou_threshold,
            'sort_rule': 'FN descending, then FP descending, then F1 ascending',
            'num_images_seen': len(self.seen_image_ids),
            'returned_count': len(items),
            'images': items,
        }

    def format_error_breakdown(self):
        keys = (
            'correct_tp',
            'classification_confusion',
            'duplicate_or_competing_fp',
            'localization_fp',
            'other_object_overlap_fp',
            'background_fp',
            'unmatched_gt_fn',
            'class_confusion_fn',
        )
        return {
            'score_threshold': self.primary_score_threshold,
            'matching_iou_threshold': self.primary_iou_threshold,
            'localization_floor_iou': self.localization_floor_iou,
            'definitions': {
                'correct_tp': 'Matched one-to-one with IoU >= matching threshold and class is correct.',
                'classification_confusion': 'Matched one-to-one with IoU >= matching threshold but predicted class differs from GT.',
                'duplicate_or_competing_fp': 'Unmatched prediction still overlaps some GT at IoU >= matching threshold, usually duplicate/competition.',
                'localization_fp': 'Unmatched prediction has same-class GT overlap in [localization floor, matching threshold).',
                'other_object_overlap_fp': 'Unmatched prediction overlaps another-class GT above localization floor but below matching threshold.',
                'background_fp': 'Unmatched prediction has no GT overlap above localization floor.',
                'unmatched_gt_fn': 'Ground-truth instance left unmatched at the primary operating point.',
                'class_confusion_fn': 'GT instance matched spatially but predicted as another class; this is also a class-aware FN.',
            },
            'counts': {key: int(self.error_breakdown.get(key, 0)) for key in keys},
            'note': (
                'This is a heuristic threshold-based diagnostic decomposition, not the official TIDE metric. '
                'classification_confusion contributes one FP to the predicted class and one FN to the GT class.'
            ),
        }


class DiagnosticCocoEvaluator:
    """Transparent CocoEvaluator wrapper that adds side-channel diagnostics."""

    def __init__(self, base_evaluator, collector):
        self._base_evaluator = base_evaluator
        self.collector = collector

    def __getattr__(self, name):
        return getattr(self._base_evaluator, name)

    def cleanup(self):
        self.collector.reset()
        return self._base_evaluator.cleanup()

    def update(self, predictions):
        # Diagnostics see the exact same postprocessed predictions passed to COCO.
        self.collector.update(predictions)
        return self._base_evaluator.update(predictions)

    def synchronize_between_processes(self):
        # Keep original COCO synchronization unchanged. Diagnostic synchronization
        # is called explicitly after evaluate() so collectives stay ordered.
        return self._base_evaluator.synchronize_between_processes()

    def synchronize_diagnostics(self):
        self.collector.synchronize_between_processes()


# -----------------------------------------------------------------------------
# Final JSON formatting
# -----------------------------------------------------------------------------

def diagnostic_notes():
    return {
        'official_metric_integrity': (
            'coco_eval_bbox and all metrics derived from coco_eval.eval use the original '
            'DEIM evaluate() + faster_coco_eval path. Side-channel diagnostics do not '
            'modify predictions before CocoEvaluator.update().'
        ),
        'value_scale': 'AP/AR/precision/recall/F1 values are stored on [0, 1], not percentages.',
        'why_AP50_AP75_AP90': (
            'Large AP50->AP75/AP90 drops indicate localization/box-tightness weakness more '
            'than simple object discovery weakness.'
        ),
        'why_AR_by_IoU': (
            'AR50 vs AR75/AR90 separates whether GT can be found loosely but cannot be '
            'localized precisely.'
        ),
        'why_threshold_operating_points': (
            'Precision/recall/F1 at fixed score thresholds reveal whether a change mainly '
            'adds recall, adds false positives, or shifts score calibration/ranking.'
        ),
        'why_confusion_matrix': (
            'The primary threshold confusion matrix helps separate wrong-class predictions '
            'from pure background false positives and missed GT.'
        ),
        'why_gt_geometry': (
            'Per-class GT frequency, COCO size counts, aspect ratio and relative bbox size '
            'help relate metric changes to long-tail, tiny/slender, or large defects.'
        ),
    }


def format_metrics(test_stats, coco_evaluator, collector=None, hard_image_limit=100):
    metrics = {}
    for name, values in test_stats.items():
        values = [float(value) for value in values]
        if name.startswith('coco_eval_') and len(values) == len(COCO_METRIC_NAMES):
            metrics[name] = dict(zip(COCO_METRIC_NAMES, values))
        else:
            metrics[name] = values

    metrics['diagnostic_notes'] = diagnostic_notes()
    metrics['dataset_bbox_summary'] = format_dataset_summary(coco_evaluator.coco_gt)

    if 'bbox' in coco_evaluator.coco_eval:
        coco_eval = coco_evaluator.coco_eval['bbox']
        metrics['coco_iou_sweep_bbox'] = format_iou_sweep(coco_eval)
        metrics['coco_size_iou_diagnostics_bbox'] = format_size_iou_diagnostics(coco_eval)
        metrics['per_category_bbox'] = format_category_metrics(coco_evaluator)

    if collector is not None:
        metrics['threshold_diagnostics_bbox'] = {
            'score_thresholds': [float(v) for v in collector.score_thresholds],
            'iou_thresholds': [float(v) for v in collector.iou_thresholds],
            'matching_definition': (
                'Greedy one-to-one same-class matching, predictions sorted by descending score. '
                'These threshold metrics are diagnostic and are separate from COCO AP/AR.'
            ),
            'operating_points': collector.format_operating_points(),
        }
        metrics['confusion_matrix_bbox'] = collector.format_confusion_matrix()
        metrics['heuristic_error_breakdown_bbox'] = collector.format_error_breakdown()
        metrics['prediction_score_histogram_bbox'] = collector.format_score_histogram()
        metrics['hard_images_bbox'] = collector.format_hard_images(limit=hard_image_limit)

    return metrics


def save_metrics(metrics, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'{output_path.name}.tmp.{os.getpid()}')
    try:
        with temporary_path.open('w', encoding='utf-8') as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False, allow_nan=False)
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

        evaluator_for_run = solver.evaluator
        collector = None
        if not args.disable_threshold_diagnostics:
            if evaluator_for_run is None or not hasattr(evaluator_for_run, 'coco_gt'):
                raise RuntimeError(
                    'Threshold diagnostics require a COCO evaluator with coco_gt. '
                    'Use --disable-threshold-diagnostics only if you intentionally want '
                    'COCO tensor metrics without side-channel diagnostics.'
                )
            collector = DetectionDiagnosticCollector(
                evaluator_for_run.coco_gt,
                score_thresholds=args.diag_score_thresholds,
                iou_thresholds=args.diag_iou_thresholds,
                primary_score_threshold=args.diag_primary_score,
                primary_iou_threshold=args.diag_primary_iou,
                localization_floor_iou=args.diag_localization_floor_iou,
            )
            evaluator_for_run = DiagnosticCocoEvaluator(evaluator_for_run, collector)

        # IMPORTANT: this is the same evaluate() call used by the original script.
        test_stats, coco_evaluator = evaluate(
            module,
            solver.criterion,
            solver.postprocessor,
            solver.val_dataloader,
            evaluator_for_run,
            solver.device,
        )

        # All ranks must enter this collective. It does not modify COCOeval state.
        if collector is not None:
            coco_evaluator.synchronize_diagnostics()

        if dist_utils.is_main_process():
            metrics = format_metrics(
                test_stats,
                coco_evaluator,
                collector=collector,
                hard_image_limit=args.hard_image_limit,
            )
            save_metrics(metrics, args.output_file)
            print(f'Metrics saved to {args.output_file}')
    finally:
        dist_utils.cleanup()


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a DEIM checkpoint and save extended COCO metrics as JSON.',
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

    parser.add_argument(
        '--diag-score-thresholds',
        type=float,
        nargs='+',
        default=list(DEFAULT_DIAG_SCORE_THRESHOLDS),
        help='Score thresholds for side-channel precision/recall/F1 diagnostics.',
    )
    parser.add_argument(
        '--diag-iou-thresholds',
        type=float,
        nargs='+',
        default=list(DEFAULT_DIAG_IOU_THRESHOLDS),
        help='IoU thresholds for side-channel precision/recall/F1 diagnostics.',
    )
    parser.add_argument(
        '--diag-primary-score',
        type=float,
        default=0.25,
        help='Score threshold used for confusion matrix, error buckets and hard-image ranking.',
    )
    parser.add_argument(
        '--diag-primary-iou',
        type=float,
        default=0.50,
        help='IoU threshold used for confusion matrix, error buckets and hard-image ranking.',
    )
    parser.add_argument(
        '--diag-localization-floor-iou',
        type=float,
        default=0.10,
        help='Lower IoU bound for the heuristic localization-FP bucket.',
    )
    parser.add_argument(
        '--hard-image-limit',
        type=int,
        default=100,
        help='Maximum number of hardest images written to JSON; -1 keeps all images.',
    )
    parser.add_argument(
        '--disable-threshold-diagnostics',
        action='store_true',
        help=(
            'Disable raw-prediction side-channel diagnostics. Standard COCO metrics, IoU sweep, '
            'per-category metrics and GT geometry summary are still written.'
        ),
    )
    return parser


if __name__ == '__main__':
    main(get_args_parser().parse_args())
