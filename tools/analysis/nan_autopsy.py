#!/usr/bin/env python3
"""NaN autopsy for the snapshot saved by det_engine's sentinel (NaN.pth).

When train_one_epoch detects non-finite pred_boxes it dumps the whole model
state to ./NaN.pth. This tool answers WHERE the NaNs came from:

Step 1 (weight check; CPU only, no yml needed):
    python3 tools/analysis/nan_autopsy.py --snapshot ./NaN.pth
    Reports every parameter / buffer containing NaN or Inf. If ANY weight is
    non-finite, the optimizer step polluted the weights (gradient leak through
    AMP) -- check whether it is concentrated in one module.

Step 2 (forward check; needs -c yml, GPU for fp16):
    python3 tools/analysis/nan_autopsy.py --snapshot ./NaN.pth -c <cfg.yml>
    Rebuilds the model, loads the snapshot, and runs a single 640x640 forward
    in fp32 and (if available) fp16-autocast with per-module hooks, reporting
    the FIRST module in execution order whose output is non-finite.

Self-test (no GPU, no yml):
    python3 tools/analysis/nan_autopsy.py --self-test-only
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def load_snapshot(path):
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # snapshot comes from our own training run; repo-wide convention
        # (dump_predictions / _solver) falls back to plain torch.load
        state = torch.load(path, map_location='cpu')
    sd = state.get('model', state)
    # det_engine's sentinel flattens 'module.' prefixes into the same dict;
    # strip and dedupe, keeping the original-name entries
    clean = {}
    for k, v in sd.items():
        k2 = k[7:] if k.startswith('module.') else k
        clean[k2] = v
    return clean


def scan_state_dict(sd, tag):
    """Returns list of (name, kind, n_nan, n_inf, numel) for non-finite tensors."""
    bad = []
    for name, t in sd.items():
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        if n_nan or n_inf:
            bad.append((name, tag, n_nan, n_inf, t.numel()))
    return bad


def _first_bad_module_hook(model, x):
    """Runs a forward with hooks; returns (output, first_bad_module or None)."""
    first_bad = []
    hooks = []

    def make_hook(name):
        def hook(_m, _inp, out):
            if first_bad:
                return
            tensors = out if isinstance(out, (tuple, list)) else (out,)
            for t in tensors:
                if torch.is_tensor(t) and t.is_floating_point():
                    if not torch.isfinite(t).all():
                        first_bad.append(name)
                        break
        return hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:      # leaves only
            hooks.append(m.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            out = model(x)
    finally:
        for h in hooks:
            h.remove()
    return out, (first_bad[0] if first_bad else None)


def forward_probe(sd, config, device):
    """Step 2: rebuild model, load snapshot, probe fp32 then fp16 forward."""
    import contextlib
    from engine.core import YAMLConfig
    cfg = YAMLConfig(config)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    matched = model.state_dict()
    loaded = {k: v for k, v in sd.items() if k in matched
              and matched[k].shape == v.shape}
    model.load_state_dict(loaded, strict=False)
    model.to(device).eval()

    results = {}
    x = torch.randn(1, 3, 640, 640, device=device)
    out, bad32 = _first_bad_module_hook(model, x)
    results['fp32_first_bad_module'] = bad32
    results['fp32_out_finite'] = bool(
        all(torch.isfinite(t).all() for t in
            (out if isinstance(out, (list, tuple)) else [out])
            if torch.is_tensor(t))
        ) if not isinstance(out, (list, tuple)) else \
        bool(all(torch.isfinite(t).all() for t in out if torch.is_tensor(t)))
    print(f'[autopsy] fp32 forward: first_bad_module={bad32}')

    use_amp = hasattr(torch, 'autocast')
    if use_amp and device.type == 'cuda':
        out16, bad16 = None, None
        with torch.autocast('cuda', dtype=torch.float16):
            out16, bad16 = _first_bad_module_hook(model, x)
        results['fp16_first_bad_module'] = bad16
        print(f'[autopsy] fp16 forward: first_bad_module={bad16}')
    else:
        results['fp16_first_bad_module'] = 'skipped (no torch.autocast/cuda)'
        print('[autopsy] fp16 forward skipped (needs torch>=2.0 autocast + cuda)')
    return results


def run(args):
    sd = load_snapshot(args.snapshot)
    print(f'[autopsy] snapshot: {args.snapshot}  ({len(sd)} tensors)')

    bad = scan_state_dict(sd, 'weight')
    if bad:
        print(f'[autopsy] WEIGHTS CONTAMINATED: {len(bad)} non-finite tensors')
        for name, kind, n_nan, n_inf, n in bad:
            print(f'  {name}: NaN={n_nan} Inf={n_inf} / {n}')
        # group by top-level module for a quick verdict
        groups = {}
        for name, kind, n_nan, n_inf, n in bad:
            groups.setdefault('.'.join(name.split('.')[:3]), 0)
            groups['.'.join(name.split('.')[:3])] += 1
        print(f'[autopsy] contaminated module groups: {groups}')
        print('[autopsy] verdict: optimizer step polluted the weights '
              '(non-finite gradient reached the update). Check AMP/scaler '
              'and the module list above for the leak origin.')
    else:
        print('[autopsy] weights are all finite -> NaN arises IN THE FORWARD '
              '(data spike / fp16 overflow), not from the optimizer step')

    if args.config:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = forward_probe(sd, args.config, device)
        out = Path(args.out) if args.out else Path('prep/innov1/nan_autopsy.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        import json
        with out.open('w', encoding='utf-8') as f:
            json.dump({'bad_weights': bad, 'forward_probe': results},
                      f, indent=2, ensure_ascii=False)
        print(f'[out] {out}')


def self_test():
    print('[self-test] nan_autopsy')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    # 1. scan_state_dict detects NaN / Inf / ignores ints and clean tensors
    sd = {
        'a.weight': torch.tensor([1.0, float('nan')]),
        'b.bias': torch.tensor([float('inf')]),
        'c.weight': torch.tensor([1.0, 2.0]),
        'd.running_mean': torch.tensor([0.0, float('-inf')]),
        'e.num_batches_tracked': torch.tensor([7]),   # int -> ignored
    }
    bad = scan_state_dict(sd, 'weight')
    names = {n for n, *_ in bad}
    check('scan finds NaN/Inf tensors, skips clean & int',
          names == {'a.weight', 'b.bias', 'd.running_mean'}, str(names))
    check('scan counts correctly',
          dict((n, (k, nn, ii)) for n, k, nn, ii, _ in bad)
          == {'a.weight': ('weight', 1, 0),
              'b.bias': ('weight', 0, 1),
              'd.running_mean': ('weight', 0, 1)})

    # 2. load_snapshot strips module. prefix and dedupes
    import tempfile, os
    raw = {'module.a.weight': torch.ones(2), 'a.weight': torch.ones(2),
           'model': None}
    raw['model'] = {k: v for k, v in raw.items() if k != 'model'}
    p = os.path.join(tempfile.mkdtemp(), 'NaN.pth')
    torch.save(raw, p)
    clean = load_snapshot(p)
    check('load_snapshot strips module. prefix',
          list(clean.keys()) == ['a.weight'], str(list(clean.keys())))

    # 3. forward hook probe: first bad module in execution order
    probe_model = nn.Sequential(
        nn.Linear(4, 4), nn.ReLU(),
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        nn.Linear(4, 2),
    )
    with torch.no_grad():   # inject NaN in the second Linear
        probe_model[2][0].weight.fill_(float('nan'))
    out, bad_mod = _first_bad_module_hook(probe_model, torch.randn(2, 4))
    check('hook probe localizes first non-finite module',
          bad_mod == '2.0', str(bad_mod))

    # 4. clean model -> no bad module
    with torch.no_grad():
        probe_model[2][0].weight.fill_(0.5)
    out, bad_mod = _first_bad_module_hook(probe_model, torch.randn(2, 4))
    check('hook probe passes a clean model', bad_mod is None and
          bool(torch.isfinite(out).all()))

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', default='./NaN.pth',
                    help='NaN.pth written by det_engine (repo root)')
    ap.add_argument('--config', '-c', default=None,
                    help='model yml; enables the step-2 forward probe')
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()
    if args.self_test_only:
        sys.exit(self_test())
    run(args)


if __name__ == '__main__':
    main()
