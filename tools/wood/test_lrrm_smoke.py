"""Synthetic LRRM checks; no dataset or checkpoint required."""

import pathlib
import sys
import types

import torch

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


def _model(**overrides):
    kwargs = dict(num_classes=3, hidden_dim=8, num_queries=3, feat_channels=[8],
                  feat_strides=[8], num_levels=1, num_points=1, nhead=2,
                  num_layers=2, dim_feedforward=16, num_denoising=0,
                  reg_max=4, layer_scale=1, lrrm_enabled=True)
    kwargs.update(overrides)
    return DFINETransformer(**kwargs)


def _criterion(**overrides):
    matcher = HungarianMatcher(
        weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        use_focal_loss=True)
    kwargs = dict(matcher=matcher, weight_dict={}, losses=[], num_classes=3,
                  reg_max=4, lrrm_enabled=True, lrrm_loss_weight=1.0,
                  lrrm_start_epoch=10)
    kwargs.update(overrides)
    return DEIMCriterion(**kwargs)


def _target(box):
    return [{'boxes': box, 'labels': torch.tensor([0], dtype=torch.long)}]


def test_forward_and_zero_init():
    model = _model().train()
    outputs = model([torch.randn(1, 8, 4, 4)])
    assert {'pred_logits', 'pred_boxes', 'pred_refined_boxes', 'pred_delta_boxes'} <= outputs.keys()
    torch.testing.assert_close(outputs['pred_refined_boxes'], outputs['pred_boxes'], rtol=0, atol=1e-7)
    assert outputs['pred_delta_boxes'].abs().max().item() == 0.0

    baseline = _model(lrrm_enabled=False).eval()
    baseline_outputs = baseline([torch.randn(1, 8, 4, 4)])
    assert not hasattr(baseline, 'dec_lrrm_head') or baseline.dec_lrrm_head is None
    torch.testing.assert_close(baseline_outputs['pred_refined_boxes'], baseline_outputs['pred_boxes'])


def test_refinement_loss_and_warmup():
    model = _model().train()
    outputs = model([torch.randn(1, 8, 4, 4)])
    target_box = outputs['pred_boxes'][0, 0].detach() + torch.tensor([0.05, -0.03, 0.02, -0.01])
    targets = _target(target_box.unsqueeze(0))
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    criterion = _criterion()

    warm = criterion.loss_localization_refinement(outputs, targets, indices, epoch=9)
    assert warm['loss_ref_l1'].item() == 0.0
    assert warm['loss_ref_giou'].item() == 0.0

    losses = criterion.loss_localization_refinement(outputs, targets, indices, epoch=10)
    assert losses['loss_ref_l1'].item() > 0
    assert losses['loss_ref_giou'].item() > 0
    assert 'train_loss_ref_l1' in criterion.lrrm_debug_stats
    assert 'train_loss_ref_giou' in criterion.lrrm_debug_stats
    model.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.dec_lrrm_head[-1].parameters())
    assert criterion.lrrm_debug_stats['max_delta_box'].item() == 0.0

    integration_model = _model().train()
    integration_targets = _target(torch.tensor([[0.5, 0.5, 0.4, 0.4]]))
    integration_outputs = integration_model([torch.randn(1, 8, 4, 4)], integration_targets)
    integration_losses = criterion(integration_outputs, integration_targets, epoch=10)
    assert 'loss_ref_l1' in integration_losses and 'loss_ref_giou' in integration_losses


def test_dn_and_eval_contract():
    model = _model(num_denoising=4).train()
    targets = _target(torch.tensor([[0.5, 0.5, 0.4, 0.4]]))
    outputs = model([torch.randn(1, 8, 4, 4)], targets)
    assert 'dn_outputs' in outputs
    assert outputs['pred_refined_boxes'].shape == outputs['pred_boxes'].shape

    model.eval()
    with torch.no_grad():
        eval_outputs = model([torch.randn(1, 8, 4, 4)])
    assert 'pred_refined_boxes' in eval_outputs
    assert eval_outputs['pred_refined_boxes'].shape == eval_outputs['pred_boxes'].shape


def main():
    torch.manual_seed(23)
    test_forward_and_zero_init()
    test_refinement_loss_and_warmup()
    test_dn_and_eval_contract()
    print('LRRM V1 smoke tests passed.')


if __name__ == '__main__':
    main()
