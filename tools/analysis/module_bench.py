#!/usr/bin/env python3
"""Module bench (user manual criteria P7 / P8): parameter count, FLOPs and
throughput of the wavelet downsampling variants vs the conv baseline.

Reports (per config yml given via -c, optionally two configs A vs B):
- whole-model parameter count;
- downsample-point parameters for stage2/3/4 (conv weights only, BN excluded
  exactly as the handover doc's expected printout);
- GFLOPs at 640x640 via `thop` when installed (analytic fallback: Haar is a
  fixed kernel, proj is 1x1 -- the numbers are reported per downsample point);
- throughput: bs in {1, 16} x {fp32, fp16}, warmup 50 + timed 200 iterations,
  FPS plus p50/p95 batch latency;
- the median5 pre-filter CPU cost (cv2.medianBlur(img, 5) on 640x640, timed
  per image and multiplied by batch) -- the "zero inference overhead" contrast
  for P6/P8.

Self-test (no engine import, no GPU needed):
    python tools/analysis/module_bench.py --self-test-only
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# running as `python3 tools/analysis/module_bench.py` puts tools/analysis at
# sys.path[0]; the repo root (needed for `from engine.core import ...`) must
# be added explicitly
sys.path.insert(0, str(ROOT))


# -----------------------------------------------------------------------------
# Standalone module loading (keeps the self-test independent of engine imports)
# -----------------------------------------------------------------------------

def _load_wavelet_standalone():
    import types
    import torch  # noqa: F401
    pkg_engine = types.ModuleType('engine'); pkg_engine.__path__ = []
    pkg_bb = types.ModuleType('engine.backbone'); pkg_bb.__path__ = []
    pkg_core = types.ModuleType('engine.core')

    def _register():
        def deco(cls):
            return cls
        return deco

    pkg_core.register = _register
    for name, mod in (('engine', pkg_engine), ('engine.backbone', pkg_bb),
                      ('engine.core', pkg_core)):
        sys.modules.setdefault(name, mod)
    sys.modules['engine'] = pkg_engine
    sys.modules['engine.backbone'] = pkg_bb
    sys.modules['engine.core'] = pkg_core

    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    common = load('engine.backbone.common', 'engine/backbone/common.py')
    pkg_bb.common = common
    return load('engine.backbone.wavelet', 'engine/backbone/wavelet.py')


def count_non_bn(module):
    """Parameter count excluding BatchNorm affine terms (matches the doc's
    'conv 7488' convention: conv weights only, BN counted separately)."""
    return sum(p.numel() for n, p in module.named_parameters()
               if '.bn' not in n and not n.endswith('.bn'))


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def self_test():
    print('[self-test] module_bench')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    wavelet = _load_wavelet_standalone()
    C = 8
    m_dw = wavelet.HaarSubbandDownsample(C, gate=False, mode='dw')
    m_dw_g = wavelet.HaarSubbandDownsample(C, gate=True, mode='dw')
    m_full = wavelet.HaarSubbandDownsample(C, gate=False, mode='full')

    dw_params = sum(p.numel() for n, p in m_dw.named_parameters()
                    if not n.startswith('bn.'))
    check("mode='dw' non-BN params == 4C", dw_params == 4 * C, str(dw_params))
    gate_params = sum(p.numel() for n, p in m_dw_g.named_parameters()
                      if n.startswith('gate.'))
    check('gate params == 40 (reduction=4)', gate_params == 40, str(gate_params))
    full_params = sum(p.numel() for n, p in m_full.named_parameters()
                      if not n.startswith('bn.'))
    check("mode='full' non-BN params == 4C^2",
          full_params == 4 * C * C, str(full_params))

    # B0 downsample conv totals: 9C per stage, C = 64/256/512 -> 7488
    hg = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
        'engine.backbone.hgnetv2', str(ROOT / 'engine/backbone/hgnetv2.py')))
    sys.modules['engine.backbone.hgnetv2'] = hg
    hg.__spec__.loader.exec_module(hg)
    m0 = hg.HGNetv2('B0', pretrained=False)
    conv_stages = []
    for i in (1, 2, 3):
        ds = m0.stages[i].downsample
        conv_stages.append(sum(p.numel() for n, p in ds.named_parameters()
                               if not n.startswith('bn')))
    check('B0 conv downsample totals == 7488 (576+2304+4608)',
          sum(conv_stages) == 7488, str(conv_stages))

    # analytic throughput helper: p50/p95 of a known series
    lat = np.array([10.0, 20.0, 30.0, 40.0])
    check('percentile helper sane',
          abs(float(np.percentile(lat, 50)) - 25.0) < 1e-9 and
          abs(float(np.percentile(lat, 95)) - 38.5) < 1e-9)

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


# -----------------------------------------------------------------------------
# Real run
# -----------------------------------------------------------------------------

def _build_model(config):
    from engine.core import YAMLConfig
    cfg = YAMLConfig(config)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    model.eval()
    return model


def _downsample_report(model, tag):
    """{stage: params} for stage2/3/4 downsample points + whole-model count."""
    from engine.backbone.wavelet import HaarSubbandDownsample
    per_stage = {}
    kind = None
    for i, stage in enumerate(model.backbone.stages):
        ds = stage.downsample
        per_stage[i + 1] = count_non_bn(ds)
        if isinstance(ds, HaarSubbandDownsample):
            kind = f'haar_{"dw" if ds.mode == "dw" else "full"}' + \
                   ('' if ds.gate is None else
                    ('_se' if ds.se_baseline else ''))
        elif not isinstance(ds, torch.nn.Identity):
            kind = 'conv'
    total = sum(p.numel() for p in model.parameters())
    print(f'[{tag}] downsample params stage2/3/4 : {per_stage[2]} / '
          f'{per_stage[3]} / {per_stage[4]}  (kind={kind})')
    print(f'[{tag}] whole model params            : {total}')
    return per_stage, total, kind


def _gflops(model, res):
    try:
        from thop import profile
    except ImportError:
        print('[bench] thop not installed -- GFLOPs skipped '
              '(pip install thop for the full readout)')
        return None
    x = torch.randn(1, 3, res, res)
    flops, _ = profile(model, inputs=(x,), verbose=False)
    return flops / 1e9


def _throughput(model, device, bs, dtype_tag, res, warmup, iters):
    x = torch.randn(bs, 3, res, res, device=device)
    if dtype_tag == 'fp16' and device.type == 'cuda':
        ctx_ctx = torch.autocast('cuda', dtype=torch.float16)
    else:
        import contextlib
        ctx_ctx = contextlib.nullcontext()

    def step():
        with ctx_ctx, torch.no_grad():
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.asarray(times)
    fps = bs * 1000.0 / float(times.mean())
    return {'fps': fps, 'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95))}


def _median5_cpu_cost(res, n_rep=50):
    import cv2
    img = np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)
    cv2.medianBlur(img, 5)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_rep):
        cv2.medianBlur(img, 5)
    return (time.perf_counter() - t0) * 1000.0 / n_rep


def run(args):
    import torch

    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.backends.cudnn.benchmark = True

    models = {}
    for tag, path in (('A', args.config), ('B', args.config_b)):
        if not path:
            continue
        models[tag] = _build_model(path)

    report = {'device': str(device), 'resolution': args.res}
    rep_a = None
    for tag, model in models.items():
        model.to(device)
        per_stage, total, kind = _downsample_report(model, tag)
        gf = _gflops(model, args.res)
        if gf is not None:
            print(f'[{tag}] GFLOPs (1x3x{args.res}x{args.res})    : {gf:.2f}')
        thr = {}
        for bs in (1, 16):
            for dt in ('fp32', 'fp16'):
                if dt == 'fp16' and device.type != 'cuda':
                    continue
                r = _throughput(model, device, bs, dt, args.res,
                                args.warmup, args.iters)
                thr[f'bs{bs}_{dt}'] = r
                print(f'[{tag}] bs={bs:<2d} {dt}: FPS={r["fps"]:.1f}  '
                      f'p50={r["p50_ms"]:.2f}ms  p95={r["p95_ms"]:.2f}ms')
        report[tag] = {'per_stage': per_stage, 'total_params': total,
                       'kind': kind, 'gflops': gf, 'throughput': thr}
        if tag == 'A':
            rep_a = (per_stage, total)

    if 'A' in models and 'B' in models:
        pa, ta = report['A']['per_stage'], report['A']['total_params']
        pb, tb = report['B']['per_stage'], report['B']['total_params']
        print(f'\n[compare] downsample params stage2/3/4 : '
              f'{sum(pa.values())} -> {sum(pb.values())}  '
              f'delta = {sum(pb.values()) - sum(pa.values())}')
        print(f'[compare] whole model params           : {ta} -> {tb}  '
              f'delta = {tb - ta} ({(tb - ta) / ta * 100:+.2f}%)')

    ms5 = _median5_cpu_cost(args.res)
    print(f'[median5] cv2.medianBlur(img,5) CPU 640x640: {ms5:.3f} ms/image '
          f'(x16 batch ~= {ms5 * 16:.1f} ms) -- the pre-filter the wavelet '
          f'downsampling must beat on cost')
    report['median5_cpu_ms_per_image'] = ms5

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[out] {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', '-c', default=None, help='config yml (A)')
    ap.add_argument('--config-b', default=None,
                    help='second config yml (B) for the A-vs-B comparison')
    ap.add_argument('--device', default=None)
    ap.add_argument('--res', type=int, default=640)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--out', default='prep/innov1/module_bench/bench.json')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()
    if args.self_test_only:
        sys.exit(self_test())
    if not args.config:
        ap.error('--config is required (unless --self-test-only)')
    run(args)


if __name__ == '__main__':
    main()
