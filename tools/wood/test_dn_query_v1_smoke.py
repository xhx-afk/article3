"""Synthetic DN Query v1 checks; no dataset or checkpoint required."""

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
from engine.deim.dn_components import prepare_for_dn, dn_post_process
from engine.deim.matcher import HungarianMatcher


DN_CONFIG = dict(enabled=True, dn_number=50, label_noise_ratio=0.5,
                 box_noise_scale=0.2, dn_loss_weight=1.0)


def _model(**overrides):
    kwargs = dict(num_classes=3, hidden_dim=8, num_queries=3, feat_channels=[8],
                  feat_strides=[8], num_levels=1, num_points=1, nhead=2,
                  num_layers=2, dim_feedforward=16, num_denoising=100,
                  reg_max=4, layer_scale=1, dn_query=DN_CONFIG)
    kwargs.update(overrides)
    return DFINETransformer(**kwargs)


def _criterion():
    matcher = HungarianMatcher(
        weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        use_focal_loss=True)
    return DEIMCriterion(
        matcher=matcher, weight_dict={}, losses=[], num_classes=3,
        reg_max=4, dn_query=DN_CONFIG)


def _full_criterion():
    matcher = HungarianMatcher(
        weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        use_focal_loss=True)
    return DEIMCriterion(
        matcher=matcher,
        weight_dict={'loss_mal': 1, 'loss_bbox': 5, 'loss_giou': 2,
                     'loss_fgl': 0.15, 'loss_ddf': 1.5},
        losses=['mal', 'boxes', 'local'], num_classes=3,
        reg_max=4, dn_query=DN_CONFIG)


def _targets():
    return [{'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
             'labels': torch.tensor([0], dtype=torch.long)}]


def _mixed_targets():
    return [
        {'boxes': torch.tensor([[0.3, 0.3, 0.2, 0.2],
                                [0.7, 0.7, 0.1, 0.1]]),
         'labels': torch.tensor([0, 2], dtype=torch.long)},
        {'boxes': torch.empty(0, 4),
         'labels': torch.empty(0, dtype=torch.long)},
    ]


def test_prepare_and_mask():
    embed = torch.nn.Embedding(4, 8, padding_idx=3)
    label_query, box_query, mask, meta = prepare_for_dn(
        _targets(), embed, 1, True, 3, 3, 8, **{
            'dn_number': 50, 'label_noise_ratio': 0.5, 'box_noise_scale': 0.2})
    assert label_query.shape[:2] == box_query.shape[:2]
    dn_count, normal_count = meta['dn_num_split']
    assert normal_count == 3 and dn_count == 50
    assert meta['dn_num_group'] == 50
    assert meta['dn_positive_idx'][0].numel() == dn_count
    assert mask.shape == (dn_count + normal_count, dn_count + normal_count)
    assert mask[dn_count:, :dn_count].all()
    assert not mask[:dn_count, dn_count:].any()
    assert mask[0, 1:dn_count].all()

    mixed = _mixed_targets()
    label_query, box_query, mask, meta = prepare_for_dn(
        mixed, embed, 2, True, 3, 3, 8, dn_number=50,
        label_noise_ratio=0.5, box_noise_scale=0.2)
    assert label_query.shape == (2, 50, 8)
    assert box_query.shape == (2, 50, 4)
    assert meta['dn_num_group'] == 25
    assert meta['dn_positive_idx'][0].numel() == 50
    assert meta['dn_positive_idx'][1].numel() == 0

    empty = [{'boxes': torch.empty(0, 4),
              'labels': torch.empty(0, dtype=torch.long)}]
    assert prepare_for_dn(empty, embed, 1, True, 3, 3, 8, dn_number=50) == \
        (None, None, None, None)


def test_split_and_model_contract():
    classes = torch.randn(2, 1, 8, 3)
    boxes = torch.rand(2, 1, 8, 4)
    meta = {'dn_num_split': [5, 3]}
    dn_cls, normal_cls, dn_box, normal_box = dn_post_process(classes, boxes, meta)
    assert dn_cls.shape[-2] == 5 and normal_cls.shape[-2] == 3
    assert dn_box.shape[-2] == 5 and normal_box.shape[-2] == 3

    model = _model().train()
    outputs = model([torch.randn(1, 8, 4, 4)], _targets())
    assert 'dn_outputs' in outputs and 'dn_meta' in outputs
    assert outputs['pred_logits'].shape[1] == 3
    assert outputs['dn_outputs'][0]['pred_logits'].shape[1] > 0

    model.eval()
    with torch.no_grad():
        eval_outputs = model([torch.randn(1, 8, 4, 4)])
    assert 'dn_outputs' not in eval_outputs and 'dn_meta' not in eval_outputs


def test_legacy_config_compatibility():
    model = _model(dn_query=None, num_denoising=4).train()
    outputs = model([torch.randn(1, 8, 4, 4)], _targets())
    assert outputs['dn_meta']['dn_num_split'] == [8, 3]


def test_explicit_dn_losses():
    model = _model().train()
    targets = _targets()
    outputs = model([torch.randn(1, 8, 4, 4)], targets)
    losses = _criterion()(outputs, targets, epoch=0)
    dn_keys = {key for key in losses if key.startswith('loss_dn_')}
    assert {'loss_dn_cls', 'loss_dn_bbox', 'loss_dn_giou'} <= dn_keys
    assert not any('_dn_' in key[len('loss_dn_'):] or key.endswith('_dn_pre')
                   for key in dn_keys)
    total = sum(losses[key] for key in dn_keys)
    assert torch.isfinite(total)
    total.backward()


def test_dn_with_deim_losses():
    model = _model().train()
    targets = _targets()
    outputs = model([torch.randn(1, 8, 4, 4)], targets)
    losses = _full_criterion()(outputs, targets, epoch=0)
    assert {'loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl'} <= losses.keys()
    assert {'loss_dn_cls', 'loss_dn_bbox', 'loss_dn_giou'} <= losses.keys()
    assert all(torch.isfinite(value).all() for value in losses.values())


def main():
    torch.manual_seed(31)
    test_prepare_and_mask()
    test_split_and_model_contract()
    test_legacy_config_compatibility()
    test_explicit_dn_losses()
    test_dn_with_deim_losses()
    print('DEIM DINO DN Query v1 smoke tests passed.')


if __name__ == '__main__':
    main()
