"""Synthetic QCCR V3 smoke tests; no dataset or checkpoint required."""

import pathlib
import sys
import types

import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _register(obj=None, **kwargs):
    return obj if obj is not None else (lambda value: value)


def _install_isolated_engine_packages():
    engine = types.ModuleType('engine'); engine.__path__ = [str(ROOT / 'engine')]
    sys.modules['engine'] = engine
    deim = types.ModuleType('engine.deim'); deim.__path__ = [str(ROOT / 'engine' / 'deim')]
    sys.modules['engine.deim'] = deim
    core = types.ModuleType('engine.core'); core.register = _register
    sys.modules['engine.core'] = core
    dist = types.ModuleType('engine.misc.dist_utils')
    dist.get_world_size = lambda: 1
    dist.is_dist_available_and_initialized = lambda: False
    sys.modules['engine.misc.dist_utils'] = dist


_install_isolated_engine_packages()

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.matcher import HungarianMatcher
from engine.deim.postprocessor import PostProcessor


def _model(**overrides):
    kwargs = dict(num_classes=3, hidden_dim=8, num_queries=3, feat_channels=[8],
                  feat_strides=[8], num_levels=1, num_points=1, nhead=2,
                  num_layers=2, dim_feedforward=16, num_denoising=0,
                  reg_max=4, layer_scale=1)
    kwargs.update(overrides)
    return DFINETransformer(**kwargs)


def _matcher():
    return HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                            use_focal_loss=True)


def _targets(boxes=None, labels=None):
    boxes = torch.empty(0, 4) if boxes is None else boxes
    labels = torch.empty(0, dtype=torch.long) if labels is None else labels
    return [{'boxes': boxes, 'labels': labels}]


def _load_yaml(path):
    raw = yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.Loader) or {}
    merged = {}
    for include in raw.get('__include__', []):
        merged.update(_load_yaml(path.parent / include))
    for key, value in raw.items():
        if key != '__include__':
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _rank_criterion(**overrides):
    kwargs = dict(matcher=_matcher(), weight_dict={}, losses=[], num_classes=3, reg_max=4,
                  qcmr_local_rank_enabled=True, qcmr_rank_start_epoch=5,
                  qcmr_rank_min_pos_iou=0.5, qcmr_rank_min_comp_iou=0.5,
                  qcmr_rank_topk_competitors=3, qcmr_rank_better_comp_tolerance=0.05,
                  qcmr_rank_temperature=0.2, qcmr_rank_loss_weight=0.05)
    kwargs.update(overrides)
    return DEIMCriterion(**kwargs)


def _rank_outputs(logits, boxes):
    return {'pred_logits': logits, 'pred_boxes': boxes}


def test_configs():
    base = ROOT / 'configs' / 'deim_dfine'
    r1 = _load_yaml(base / 'deim_hgnetv2_l_wood_qcmr_v3_r1.yml')
    r2 = _load_yaml(base / 'deim_hgnetv2_l_wood_qcmr_v3_r2.yml')
    assert r1['DFINETransformer']['qcmr_encoder_quality_enabled'] is False
    assert r2['DFINETransformer']['qcmr_encoder_quality_enabled'] is True
    assert r1['DEIMCriterion']['qcmr_local_rank_enabled'] is True
    assert r2['DEIMCriterion']['qcmr_encoder_quality_enabled'] is True
    for cfg in (r1, r2):
        assert 'qcmr_quality_calibration' not in cfg['DEIMCriterion']
        assert 'qcmr_decoder_quality_loss_weight' not in cfg['DEIMCriterion']
        assert 'qcmr_quality_calibration' not in cfg['DFINETransformer']
        assert 'qcmr_score_quality_alpha' not in cfg['DFINETransformer']
        assert 'qcmr_score_quality_beta' not in cfg['DFINETransformer']


def test_baseline_and_encoder():
    features = [torch.randn(1, 8, 4, 4)]
    baseline = _model().eval()
    assert not hasattr(baseline, 'enc_quality_head')
    assert not hasattr(baseline, 'dec_quality_head')
    with torch.no_grad():
        outputs = baseline(features)
    assert 'pred_quality' not in outputs
    encoder = _model(qcmr_encoder_quality_enabled=True).train()
    outputs = encoder(features)
    assert 'pred_quality' not in outputs
    assert outputs['enc_aux_outputs'][0]['pred_quality'].shape == (1, 3, 1)


def test_candidate_mining_and_ranking():
    # q0 is Hungarian positive (IoU .80), q1 is a valid duplicate (.70), q2 is background (.10).
    boxes = torch.tensor([[[0.5, 0.5, 0.8, 0.8], [0.5, 0.5, 0.7, 0.7],
                          [0.1, 0.1, 0.1, 0.1]]])
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    logits.data[0, :, 0] = torch.tensor([1.0, 2.0, 5.0])
    outputs = _rank_outputs(logits, boxes)
    criterion = _rank_criterion()
    targets = _targets(torch.tensor([[0.5, 0.5, 0.8, 0.8]]), torch.tensor([0]))
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    warm = criterion.loss_qcmr_competitive_rank(outputs, targets, indices, epoch=0)
    assert warm['loss_qcmr_rank'].item() == 0.0
    assert criterion.qcmr_debug_stats['qcmr_rank_active'].item() == 0.0

    losses = criterion.loss_qcmr_competitive_rank(outputs, targets, indices, epoch=5)
    assert losses['loss_qcmr_rank'].item() > 0
    assert criterion.qcmr_debug_stats['qcmr_rank_num_pairs'].item() == 1
    assert criterion.qcmr_debug_stats['qcmr_rank_num_gt_with_pairs'].item() == 1
    assert criterion.qcmr_debug_stats['qcmr_rank_violation_rate'].item() == 1.0
    assert criterion.qcmr_debug_stats['qcmr_rank_better_unmatched_count'].item() == 0
    assert criterion.qcmr_debug_stats['qcmr_rank_matched_winner_rate'].item() == 0.0
    losses['loss_qcmr_rank'].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 1, 0] > 0
    assert logits.grad[0, 2, 0] == 0


def test_exclusions_and_guards():
    criterion = _rank_criterion()
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    boxes = torch.tensor([[[0.5, 0.5, 0.8, 0.8], [0.5, 0.5, 0.7, 0.7]]])
    targets = _targets(torch.tensor([[0.5, 0.5, 0.8, 0.8], [0.5, 0.5, 0.7, 0.7]]),
                       torch.tensor([0, 1]))
    # q1 is matched to GT-B and must not become GT-A's competitor.
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    outputs = _rank_outputs(logits, boxes)
    criterion.loss_qcmr_competitive_rank(outputs, targets, indices, epoch=5)
    assert criterion.qcmr_debug_stats['qcmr_rank_num_pairs'].item() == 0

    # Better unmatched guard skips a competitor whose IoU is materially better.
    boxes = torch.tensor([[[0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.8, 0.8]]])
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    outputs = _rank_outputs(logits, boxes)
    criterion.loss_qcmr_competitive_rank(
        outputs, _targets(torch.tensor([[0.5, 0.5, 0.8, 0.8]]), torch.tensor([0])),
        [(torch.tensor([0]), torch.tensor([0]))], epoch=5)
    assert criterion.qcmr_debug_stats['qcmr_rank_num_pairs'].item() == 0
    assert criterion.qcmr_debug_stats['qcmr_rank_better_unmatched_rate'].item() == 1.0
    assert criterion.qcmr_debug_stats['qcmr_rank_better_unmatched_count'].item() == 1

    zero = criterion.loss_qcmr_competitive_rank(
        _rank_outputs(torch.zeros(1, 2, 3, requires_grad=True), torch.zeros(1, 2, 4)),
        _targets(), [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))], epoch=5)
    zero['loss_qcmr_rank'].backward()
    assert torch.isfinite(zero['loss_qcmr_rank'])


def test_inference_contract():
    model = _model().eval()
    with torch.no_grad():
        outputs = model([torch.randn(1, 8, 4, 4)])
    assert 'pred_quality' not in outputs
    postprocessor = PostProcessor(num_classes=3, num_top_queries=3)
    assert not hasattr(postprocessor, 'qcmr_quality_calibration')
    assert not hasattr(postprocessor, 'qcmr_score_quality_alpha')
    assert not hasattr(postprocessor, 'qcmr_score_quality_beta')
    results = postprocessor(outputs, torch.tensor([[32, 32]]))
    assert set(results[0]) == {'labels', 'boxes', 'scores'}


def main():
    torch.manual_seed(17)
    test_configs()
    test_baseline_and_encoder()
    test_candidate_mining_and_ranking()
    test_exclusions_and_guards()
    test_inference_contract()
    print('QCMR V3 QCCR smoke tests passed.')


if __name__ == '__main__':
    main()
