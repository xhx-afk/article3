"""Synthetic QCMR V2 smoke tests; no dataset or checkpoint required."""

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


def _targets(empty=False):
    return [{'labels': torch.empty(0, dtype=torch.long) if empty else torch.tensor([1]),
             'boxes': torch.empty(0, 4) if empty else torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]


def _load_yaml(path):
    raw = yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.Loader) or {}
    merged = {}
    for include in raw.get('__include__', []):
        included = _load_yaml(path.parent / include)
        for key, value in included.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    for key, value in raw.items():
        if key != '__include__':
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def test_configs():
    base = ROOT / 'configs' / 'deim_dfine'
    q1, q2, q3 = [_load_yaml(base / f'deim_hgnetv2_l_wood_qcmr_q{i}.yml') for i in (1, 2, 3)]
    assert q1['DFINETransformer']['qcmr_enabled'] and not q1['DFINETransformer']['qcmr_quality_calibration']
    assert not q2['DFINETransformer']['qcmr_enabled'] and q2['DFINETransformer']['qcmr_quality_calibration']
    assert q3['DFINETransformer']['qcmr_enabled'] and q3['DFINETransformer']['qcmr_quality_calibration']
    for cfg in (q1, q2, q3):
        assert 'qcmr_use_quality_selection' not in cfg['DFINETransformer']
        matcher = cfg['DEIMCriterion'].get('matcher', {})
        assert not any(key.startswith('qcmr_') for key in matcher)


def test_model_and_losses(features):
    baseline = _model().train()
    assert len(baseline.dec_quality_head) == 0
    outputs = baseline(features)
    assert 'pred_quality' not in outputs

    encoder = _model(qcmr_enabled=True).train()
    enc_outputs = encoder(features)
    assert 'pred_quality' not in enc_outputs
    assert enc_outputs['enc_aux_outputs'][0]['pred_quality'].shape == (1, 3, 1)

    decoder = _model(qcmr_quality_calibration=True).train()
    dec_outputs = decoder(features)
    assert dec_outputs['pred_quality'].shape == (1, 3, 1)
    assert dec_outputs['aux_outputs'][0]['pred_quality'].shape == (1, 3, 1)

    criterion = DEIMCriterion(matcher=_matcher(), weight_dict={}, losses=[], num_classes=3,
                              reg_max=4, qcmr_quality_calibration=True)
    losses = criterion(dec_outputs, _targets())
    assert 'loss_qcmr_quality_decoder' in losses and torch.isfinite(losses['loss_qcmr_quality_decoder'])
    assert all(torch.isfinite(v) for v in criterion.qcmr_debug_stats.values())
    losses['loss_qcmr_quality_decoder'].backward()
    assert any(p.grad is not None for p in decoder.dec_quality_head[-1].parameters())

    enc_criterion = DEIMCriterion(matcher=_matcher(), weight_dict={}, losses=[], num_classes=3,
                                  reg_max=4, qcmr_enabled=True)
    enc_losses = enc_criterion(enc_outputs, _targets())
    assert 'loss_qcmr_quality_enc' in enc_losses
    assert 'qcmr_quality_iou_corr' in enc_criterion.qcmr_debug_stats
    return baseline, encoder, decoder


def test_calibrated_postprocessor():
    processor = PostProcessor(num_classes=2, num_top_queries=2,
                               qcmr_quality_calibration=True,
                               qcmr_score_quality_alpha=1.0,
                               qcmr_score_quality_beta=0.5)
    outputs = {'pred_logits': torch.tensor([[[4., 0.], [3., 0.]]]),
               'pred_boxes': torch.tensor([[[.5, .5, .2, .2], [.5, .5, .2, .2]]]),
               'pred_quality': torch.tensor([[[-4.], [4.]]])}
    result = processor(outputs, torch.tensor([[32, 32]]))[0]
    assert result['scores'][0] > result['scores'][1]


def main():
    torch.manual_seed(17)
    test_configs()
    models = test_model_and_losses([torch.randn(1, 8, 4, 4)])
    test_calibrated_postprocessor()
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model([torch.randn(1, 8, 4, 4)])
        assert set(outputs).issubset({'pred_logits', 'pred_boxes', 'pred_quality'})
    print('QCMR V2 smoke tests passed.')


if __name__ == '__main__':
    main()
