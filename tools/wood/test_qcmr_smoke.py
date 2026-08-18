"""Synthetic QCMR V1.1 smoke tests that do not require data or checkpoints."""

import inspect
import pathlib
import sys
import types

import torch
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _register(obj=None, **kwargs):
    if obj is not None:
        return obj
    return lambda value: value


def _install_isolated_engine_packages():
    """Load QCMR modules without importing optional training/data dependencies."""
    engine = types.ModuleType('engine')
    engine.__path__ = [str(ROOT / 'engine')]
    sys.modules['engine'] = engine

    deim = types.ModuleType('engine.deim')
    deim.__path__ = [str(ROOT / 'engine' / 'deim')]
    sys.modules['engine.deim'] = deim

    solver = types.ModuleType('engine.solver')
    solver.__path__ = [str(ROOT / 'engine' / 'solver')]
    sys.modules['engine.solver'] = solver

    core = types.ModuleType('engine.core')
    core.register = _register
    sys.modules['engine.core'] = core

    dist_utils = types.ModuleType('engine.misc.dist_utils')
    dist_utils.get_world_size = lambda: 1
    dist_utils.is_dist_available_and_initialized = lambda: False
    dist_utils.reduce_dict = lambda values: values
    dist_utils.is_main_process = lambda: True
    sys.modules['engine.misc.dist_utils'] = dist_utils

    misc = types.ModuleType('engine.misc')
    misc.__path__ = [str(ROOT / 'engine' / 'misc')]
    misc.MetricLogger = type('MetricLogger', (), {})
    misc.SmoothedValue = type('SmoothedValue', (), {})
    misc.dist_utils = dist_utils
    sys.modules['engine.misc'] = misc

    optim = types.ModuleType('engine.optim')
    optim.ModelEMA = type('ModelEMA', (), {})
    optim.Warmup = type('Warmup', (), {})
    sys.modules['engine.optim'] = optim

    data = types.ModuleType('engine.data')
    data.CocoEvaluator = type('CocoEvaluator', (), {})
    sys.modules['engine.data'] = data

    tensorboard = types.ModuleType('torch.utils.tensorboard')
    tensorboard.SummaryWriter = type('SummaryWriter', (), {})
    sys.modules['torch.utils.tensorboard'] = tensorboard


_install_isolated_engine_packages()

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer, LQE
from engine.deim.matcher import HungarianMatcher
from engine.deim.postprocessor import PostProcessor
from engine.solver.det_engine import _set_qcmr_epoch


def _model(**overrides):
    kwargs = {
        'num_classes': 3,
        'hidden_dim': 8,
        'num_queries': 3,
        'feat_channels': [8],
        'feat_strides': [8],
        'num_levels': 1,
        'num_points': 1,
        'nhead': 2,
        'num_layers': 2,
        'dim_feedforward': 16,
        'num_denoising': 0,
        'reg_max': 4,
        'reg_scale': 4.0,
        'layer_scale': 1,
    }
    kwargs.update(overrides)
    return DFINETransformer(**kwargs)


def _merge(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path):
    path = pathlib.Path(path)
    raw = yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.Loader) or {}
    expanded = {}
    for include in raw.get('__include__', []):
        _merge(expanded, _load_yaml(path.parent / include))
    return _merge(expanded, raw)


def _test_configs():
    config_dir = ROOT / 'configs' / 'deim_dfine'
    baseline = _load_yaml(config_dir / 'deim_hgnetv2_l_wood.yml')
    q1 = _load_yaml(config_dir / 'deim_hgnetv2_l_wood_qcmr_q1.yml')
    q2 = _load_yaml(config_dir / 'deim_hgnetv2_l_wood_qcmr_q2.yml')
    q3 = _load_yaml(config_dir / 'deim_hgnetv2_l_wood_qcmr_q3.yml')

    assert not baseline['DFINETransformer'].get('qcmr_enabled', False)
    assert q1['DFINETransformer']['qcmr_enabled']
    assert not q1['DFINETransformer']['qcmr_use_quality_selection']
    assert not q1['DEIMCriterion']['matcher']['qcmr_quality_matching']
    assert q2['DFINETransformer']['qcmr_enabled']
    assert q2['DFINETransformer']['qcmr_use_quality_selection']
    assert q2['DFINETransformer']['qcmr_quality_selection_start_epoch'] == 3
    assert not q2['DEIMCriterion']['matcher']['qcmr_quality_matching']
    assert q3['DFINETransformer']['qcmr_enabled']
    assert q3['DFINETransformer']['qcmr_use_quality_selection']
    assert q3['DFINETransformer']['qcmr_quality_selection_start_epoch'] == 3
    assert q3['DEIMCriterion']['matcher']['qcmr_quality_matching']
    assert q3['DEIMCriterion']['matcher']['qcmr_match_pred_quality_beta'] == 0.0


def _matcher(**overrides):
    kwargs = {
        'weight_dict': {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
        'use_focal_loss': True,
    }
    kwargs.update(overrides)
    return HungarianMatcher(**kwargs)


def _targets(empty=False):
    if empty:
        return [{
            'labels': torch.empty(0, dtype=torch.long),
            'boxes': torch.empty(0, 4),
        }]
    return [{
        'labels': torch.tensor([1]),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }]


def _test_baseline(features):
    model = _model().train()
    assert not hasattr(model, 'enc_quality_head')
    outputs = model(features)
    assert 'pred_quality' not in outputs
    assert 'qcmr_matching' not in outputs
    assert all('pred_quality' not in item for item in outputs['enc_aux_outputs'])

    criterion = DEIMCriterion(
        matcher=_matcher(),
        weight_dict={'loss_mal': 1, 'loss_bbox': 5, 'loss_giou': 2},
        losses=['mal', 'boxes'],
        gamma=1.5,
        num_classes=3,
        reg_max=4,
    )
    losses = criterion(outputs, _targets())
    assert losses and all(torch.isfinite(value) for value in losses.values())
    assert not criterion.qcmr_debug_stats

    model.eval()
    with torch.no_grad():
        eval_outputs = model(features)
    assert set(eval_outputs) == {'pred_logits', 'pred_boxes'}
    return model


def _test_q1(features):
    model = _model(qcmr_enabled=True, qcmr_use_quality_selection=False).train()
    assert hasattr(model, 'enc_quality_head')
    assert 'return_quality' not in inspect.signature(LQE.forward).parameters
    outputs = model(features)
    assert 'pred_quality' not in outputs
    assert outputs['enc_aux_outputs'][0]['pred_quality'].shape == (1, 3, 1)
    q3_matcher = _matcher(
        qcmr_quality_matching=True,
        qcmr_match_pred_quality_beta=0.0,
    )
    q3_indices = q3_matcher(outputs, _targets())['indices']
    assert len(q3_indices) == 1 and len(q3_indices[0][0]) == 1

    criterion = DEIMCriterion(
        matcher=_matcher(),
        weight_dict={},
        losses=[],
        num_classes=3,
        reg_max=4,
        qcmr_enabled=True,
    )
    losses = criterion(outputs, _targets())
    assert set(losses) == {'loss_qcmr_quality_enc'}
    assert torch.isfinite(losses['loss_qcmr_quality_enc'])
    assert set(criterion.qcmr_debug_stats) == {
        'qcmr_quality_pos_mean',
        'qcmr_quality_neg_mean',
        'qcmr_quality_iou_corr',
    }
    assert all(torch.isfinite(value) for value in criterion.qcmr_debug_stats.values())

    losses['loss_qcmr_quality_enc'].backward()
    gradients = [parameter.grad for parameter in model.enc_quality_head.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert all(parameter.grad is None for parameter in model.decoder.lqe_layers[-1].parameters())

    empty_outputs = model(features)
    empty_losses = criterion(empty_outputs, _targets(empty=True))
    assert set(empty_losses) == {'loss_qcmr_quality_enc'}
    assert torch.isfinite(empty_losses['loss_qcmr_quality_enc'])
    assert all(torch.isfinite(value) for value in criterion.qcmr_debug_stats.values())
    assert criterion.qcmr_debug_stats['qcmr_quality_iou_corr'].item() == 0.0
    return model


def _selected_ids(model, memory, logits, anchors, quality):
    selected = model._select_topk(memory, logits, anchors, 2, quality)[0]
    return selected[0, :, 0].to(torch.int64).tolist()


def _test_q2_warmup():
    model = _model(
        qcmr_enabled=True,
        qcmr_use_quality_selection=True,
        qcmr_quality_selection_start_epoch=3,
    ).train()
    memory = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1).repeat(1, 1, 8)
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0]).reshape(1, 4, 1).repeat(1, 1, 3)
    anchors = torch.zeros(1, 4, 4)
    discriminative_quality = torch.tensor([-10.0, -10.0, 10.0, 10.0]).reshape(1, 4, 1)

    model.set_qcmr_epoch(0)
    assert _selected_ids(model, memory, logits, anchors, discriminative_quality) == [0, 1]
    model.set_qcmr_epoch(2)
    assert _selected_ids(model, memory, logits, anchors, discriminative_quality) == [0, 1]
    model.set_qcmr_epoch(3)
    assert _selected_ids(model, memory, logits, anchors, discriminative_quality) == [2, 3]

    zero_quality = torch.zeros_like(discriminative_quality)
    assert _selected_ids(model, memory, logits, anchors, zero_quality) == [0, 1]

    model.set_qcmr_epoch(0)
    model.eval()
    assert _selected_ids(model, memory, logits, anchors, discriminative_quality) == [2, 3]

    wrapper = torch.nn.Module()
    wrapper.module = model
    _set_qcmr_epoch(wrapper, 2)
    assert model.qcmr_current_epoch == 2
    return model


def _test_debug_correlation():
    criterion = DEIMCriterion(
        matcher=_matcher(),
        weight_dict={},
        losses=[],
        num_classes=3,
        reg_max=4,
        qcmr_enabled=True,
    )
    outputs = {
        'pred_quality': torch.tensor([[[2.0], [-2.0], [0.0]]]),
        'pred_boxes': torch.tensor([[[0.3, 0.3, 0.2, 0.2],
                                     [0.7, 0.7, 0.1, 0.1],
                                     [0.5, 0.5, 0.1, 0.1]]]),
    }
    targets = [{
        'labels': torch.tensor([0, 1]),
        'boxes': torch.tensor([[0.3, 0.3, 0.2, 0.2],
                               [0.7, 0.7, 0.2, 0.2]]),
    }]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    loss = criterion.loss_qcmr_quality_enc(outputs, targets, indices)
    assert torch.isfinite(loss['loss_qcmr_quality_enc'])
    correlation = criterion.qcmr_debug_stats['qcmr_quality_iou_corr']
    assert torch.isfinite(correlation) and correlation.item() > 0.99


def _test_state_dict_compatibility(baseline, qcmr):
    missing, unexpected = qcmr.load_state_dict(baseline.state_dict(), strict=False)
    expected_missing = {
        'enc_quality_head.layers.0.weight',
        'enc_quality_head.layers.0.bias',
        'enc_quality_head.layers.1.weight',
        'enc_quality_head.layers.1.bias',
    }
    assert set(missing) == expected_missing
    assert not unexpected
    qcmr.load_state_dict(qcmr.state_dict(), strict=True)
    assert not any('qcmr' in key for key in baseline.state_dict())
    assert not any('enc_quality_head' in key for key in baseline.state_dict())


def _test_eval_contract(features, models):
    postprocessor = PostProcessor(num_classes=3, num_top_queries=3)
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            assert set(outputs) == {'pred_logits', 'pred_boxes'}
            results = postprocessor(outputs, torch.tensor([[32, 32]]))
        assert len(results) == 1
        assert set(results[0]) == {'labels', 'boxes', 'scores'}


def main():
    torch.manual_seed(17)
    _test_configs()
    features = [torch.randn(1, 8, 4, 4)]
    baseline = _test_baseline(features)
    q1 = _test_q1(features)
    q2 = _test_q2_warmup()
    _test_debug_correlation()
    q3 = _model(qcmr_enabled=True, qcmr_use_quality_selection=True).eval()
    _test_state_dict_compatibility(baseline, q1)
    _test_eval_contract(features, [baseline, q1, q2, q3])
    print('QCMR V1.1 smoke tests passed.')


if __name__ == '__main__':
    main()
