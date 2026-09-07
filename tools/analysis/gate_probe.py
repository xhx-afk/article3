#!/usr/bin/env python3
"""Gate mechanism probe (user manual criterion P9).

Answers: "did the gate learn the claimed mechanism, or is it just another
attention module that happens to help?"

Loads the A2 checkpoint, runs the val set forward, and reads
`HaarSubbandDownsample.last_gate` (B, C, 4) at every downsample point for
every sample, grouped by the augmentation code of the image (via ann
file_name). Reports:
- per module x code: mean g_LL / g_LH / g_HL / g_HH;
- the AUC with which g_HH separates PDC from ODC, defined as
  P(g_HH(PDC) < g_HH(ODC)) (the gate SUPPRESSING impulse-dominated subbands);
- the correlation between g_HH and the sample's HH energy share (the gate's
  own input, SubbandEnergyGate.last_p).

Preregistered expectations (all three must hold for the mechanism claim):
    g_HH(PDC) < g_HH(ODC),  AUC >= 0.8,  corr(g_HH, HH share) < 0.
If any fails, AP gains may NOT be described as "the gate suppresses
impulse-dominated subbands" in the paper -- only as an empirical improvement.
That verdict lands in summary.json's `claim_supported` field.

Self-test (no checkpoint / no torch needed):
    python tools/analysis/gate_probe.py --self-test-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUG_RE, index_gt, load_coco  # noqa: E402

CODES = ['ODC', 'LDC', 'DDC', 'GDC', 'PDC']
SUBBANDS = ('LL', 'LH', 'HL', 'HH')


# -----------------------------------------------------------------------------
# Statistics (torch-free, covered by the self-test)
# -----------------------------------------------------------------------------

def auc_lower(a, b):
    """P(a < b) under the Mann-Whitney formulation, ties count 0.5.

    a: samples of group A (PDC), b: samples of group B (ODC).
    Returns nan when either group is empty.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float('nan')
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind='mergesort')
    sorted_v = allv[order]
    avg_ranks = np.arange(1, len(allv) + 1, dtype=np.float64)
    i = 0
    while i < len(sorted_v):                      # average ranks within ties
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            avg_ranks[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = avg_ranks
    U = ranks[:na].sum() - na * (na + 1) / 2.0    # counts pairs a > b (+0.5 tie)
    return float(1.0 - U / (na * nb))


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def aggregate(records):
    """records: list of dicts {module, code, gate (C,4)-like, p_hh float}.

    Returns {module: {code: {'g_mean': [4], 'n': int,
                             'g_hh': [...], 'p_hh': [...]}}}.
    """
    out = {}
    for r in records:
        g = np.asarray(r['gate'], dtype=np.float64).reshape(-1, 4)
        entry = out.setdefault(r['module'], {}).setdefault(r['code'], {
            'g_sum': np.zeros(4), 'n': 0, 'g_hh': [], 'p_hh': []})
        entry['g_sum'] += g.mean(axis=0)
        entry['n'] += 1
        entry['g_hh'].append(float(g[:, 3].mean()))
        entry['p_hh'].append(float(r['p_hh']))
    for module, by_code in out.items():
        for code, e in by_code.items():
            e['g_mean'] = (e.pop('g_sum') / max(e['n'], 1)).tolist()
    return out


def verdicts_for(agg, module):
    """Preregistered checks for one module; needs PDC and ODC readings."""
    by_code = agg.get(module, {})
    if 'PDC' not in by_code or 'ODC' not in by_code:
        return None
    g_hh_pdc = by_code['PDC']['g_hh']
    g_hh_odc = by_code['ODC']['g_hh']
    mean_pdc = float(np.mean(g_hh_pdc))
    mean_odc = float(np.mean(g_hh_odc))
    auc = auc_lower(g_hh_pdc, g_hh_odc)
    g_hh_all, p_hh_all = [], []
    for code, e in by_code.items():
        g_hh_all.extend(e['g_hh'])
        p_hh_all.extend(e['p_hh'])
    corr = pearson(g_hh_all, p_hh_all)
    checks = {
        'g_HH(PDC) < g_HH(ODC)': mean_pdc < mean_odc,
        'AUC(g_HH separates PDC from ODC) >= 0.8': (np.isfinite(auc) and auc >= 0.8),
        'corr(g_HH, HH share) < 0': (np.isfinite(corr) and corr < 0),
    }
    tag = '【符合假设】' if all(checks.values()) else '【与假设相反】'
    return {
        'g_hh_mean_PDC': mean_pdc,
        'g_hh_mean_ODC': mean_odc,
        'auc_PDC_below_ODC': auc,
        'corr_g_hh_vs_hh_share': corr,
        'checks': checks,
        'claim_supported': bool(all(checks.values())),
        'tag': tag,
    }


# -----------------------------------------------------------------------------
# Real run (torch + checkpoint + val dataloader)
# -----------------------------------------------------------------------------

def _strip_module_prefix(raw):
    return {(k[7:] if k.startswith('module.') else k): v
            for k, v in raw.items()}


def _matched_state(state, params):
    matched = {}
    missed, unmatched = [], []
    for k, v in state.items():
        if k in params:
            if v.shape == params[k].shape:
                matched[k] = params[k]
            else:
                unmatched.append(k)
        else:
            missed.append(k)
    return matched, {'missed': missed, 'unmatched': unmatched}


def image_id_to_code(coco):
    _, images, _ = index_gt(coco)
    mapping = {}
    for iid, img in images.items():
        m = AUG_RE.search(str(img.get('file_name', '')))
        if m:
            mapping[int(iid)] = m.group(1)
    return mapping


def run(args):
    import torch
    from engine.core import YAMLConfig, yaml_utils
    from engine.misc import dist_utils
    from engine.solver import TASKS
    from engine.backbone.wavelet import HaarSubbandDownsample

    dist_utils.setup_distributed(False, 'logo', seed=0)
    updates = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(args.config, **updates)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    model = dist_utils.de_parallel(solver.model)

    try:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
    except Exception:
        state = torch.load(args.resume, map_location='cpu')
    raw = (state['ema']['module'] if isinstance(state.get('ema'), dict)
           and 'module' in state['ema'] else
           state['model'] if 'model' in state else state)
    weights = _strip_module_prefix(raw)
    matched, infos = _matched_state(model.state_dict(), weights)
    model.load_state_dict(matched, strict=False)
    print(f'[gate_probe] loaded {args.resume}: '
          f'missed={len(infos["missed"])} unmatched={len(infos["unmatched"])}')

    modules = [(name, m) for name, m in model.named_modules()
               if isinstance(m, HaarSubbandDownsample)]
    if not modules:
        sys.exit('[gate_probe] no HaarSubbandDownsample in the model -- '
                 'this probe requires the A2 (haar_gate) config/checkpoint')
    print(f'[gate_probe] probes: {[n for n, _ in modules]}')

    id2code = image_id_to_code(load_coco(args.ann))
    device = solver.device
    model.eval()
    records = []
    n_images = 0
    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model(samples)
            for idx, t in enumerate(targets):
                image_id = int(t['image_id'].item())
                code = id2code.get(image_id)
                n_images += 1
                if code is None:
                    continue
                for name, m in modules:
                    if m.last_gate is None or m.gate is None \
                            or m.gate.last_p is None:
                        continue
                    g = m.last_gate.detach().cpu().numpy()     # (B, C, 4)
                    p = m.gate.last_p.detach().cpu().numpy()   # (B, C, 4)
                    records.append({'module': name, 'code': code,
                                    'gate': g[idx],
                                    'p_hh': float(p[idx][..., 3].mean())})
    print(f'[gate_probe] forwarded {n_images} images, '
          f'{len(records)} (module, image) readings')

    agg = aggregate(records)
    summary = {'ann': args.ann, 'resume': args.resume,
               'n_images': n_images, 'modules': {}}
    any_supported = []
    for name in agg:
        v = verdicts_for(agg, name)
        g_mean = {c: agg[name][c]['g_mean'] for c in agg[name]}
        summary['modules'][name] = {
            'g_mean_by_code': g_mean, 'verdict': v}
        for c in sorted(agg[name]):
            gm = agg[name][c]['g_mean']
            print(f'[gate] {name} {c}: ' +
                  '  '.join(f'{s}={gm[i]:.4f}' for i, s in enumerate(SUBBANDS)) +
                  f'  (n={agg[name][c]["n"]})')
        if v is not None:
            any_supported.append(v['claim_supported'])
            print(f'[gate] {name}: g_HH PDC={v["g_hh_mean_PDC"]:.4f} '
                  f'ODC={v["g_hh_mean_ODC"]:.4f} AUC={v["auc_PDC_below_ODC"]:.3f} '
                  f'corr={v["corr_g_hh_vs_hh_share"]:.3f} -> {v["tag"]}')
            for desc, ok in v['checks'].items():
                print(f'[gate]   [{"ok" if ok else "FAIL"}] {desc}')

    supported = bool(any_supported) and all(any_supported)
    summary['claim_supported'] = supported
    summary['claim_text'] = (
        '门控抑制脉冲主导子带（机制性结论）' if supported else
        '门控仅带来经验性改进，机制宣称不成立（论文措辞必须降级）')
    print(f'[gate] claim_supported = {supported}: {summary["claim_text"]}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[out] {out}')


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def self_test():
    print('[self-test] gate_probe')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    rng = np.random.default_rng(0)

    # AUC sanity
    check('AUC fully separable (a << b) == 1.0',
          auc_lower(np.arange(10) * 0.1, np.arange(10) * 0.1 + 100) == 1.0)
    check('AUC identical distributions == 0.5',
          abs(auc_lower(rng.normal(size=200), rng.normal(size=200)) - 0.5) < 0.1)
    check('AUC fully reversed == 0.0',
          auc_lower(np.arange(10) * 0.1 + 100, np.arange(10) * 0.1) == 0.0)
    check('AUC with ties handled (constants -> 0.5)',
          auc_lower(np.ones(5), np.ones(5)) == 0.5)
    check('AUC empty group -> nan', np.isnan(auc_lower([], [1.0])))

    # pearson matches numpy
    x = rng.normal(size=100)
    y = 2 * x + rng.normal(size=100)
    check('pearson matches np.corrcoef',
          abs(pearson(x, y) - np.corrcoef(x, y)[0, 1]) < 1e-12)

    # aggregation: 2 modules x 2 codes, known means
    records = []
    for code, base in (('ODC', 1.0), ('PDC', 0.5)):
        for i in range(4):
            records.append({'module': 'm/stages.2.downsample', 'code': code,
                            'gate': np.full((3, 4), base + 0.01 * i),
                            'p_hh': 0.1 * (i + 1)})
            records.append({'module': 'm/stages.4.downsample', 'code': code,
                            'gate': np.full((2, 4), base * 2), 'p_hh': 0.3})
    agg = aggregate(records)
    a2 = agg['m/stages.2.downsample']
    check('aggregate ODC mean == 1.015',
          abs(a2['ODC']['g_mean'][0] - 1.015) < 1e-9,
          str(a2['ODC']['g_mean']))
    check('aggregate PDC mean == 0.515',
          abs(a2['PDC']['g_mean'][0] - 0.515) < 1e-9,
          str(a2['PDC']['g_mean']))
    check('aggregate counts', a2['ODC']['n'] == 4 and a2['PDC']['n'] == 4)

    # verdicts: constructed to satisfy ALL preregistered checks
    rec_ok = []
    for i in range(20):
        p_hh = 0.01 + 0.02 * i
        rec_ok.append({'module': 'm', 'code': 'ODC',
                       'gate': np.full((2, 4), 1.0), 'p_hh': p_hh})
        rec_ok.append({'module': 'm', 'code': 'PDC',
                       'gate': np.full((2, 4), 0.5 - 0.01 * i), 'p_hh': p_hh})
    agg_ok = aggregate(rec_ok)
    v_ok = verdicts_for(agg_ok, 'm')
    check('verdict: supported when all checks pass',
          v_ok is not None and v_ok['claim_supported'] and
          v_ok['tag'] == '【符合假设】', str(v_ok))

    # verdicts: corr positive (gate RAISES g_HH with HH share) -> claim fails
    rec_bad = []
    for i in range(20):
        p_hh = 0.01 + 0.02 * i
        rec_bad.append({'module': 'm', 'code': 'ODC',
                        'gate': np.full((2, 4), 0.5), 'p_hh': p_hh})
        rec_bad.append({'module': 'm', 'code': 'PDC',
                        'gate': np.full((2, 4), 0.5 + 0.01 * i), 'p_hh': p_hh})
    v_bad = verdicts_for(aggregate(rec_bad), 'm')
    check('verdict: claim rejected when gate tracks HH share positively',
          v_bad is not None and not v_bad['claim_supported'] and
          v_bad['tag'] == '【与假设相反】')

    # missing code -> verdict None (no crash)
    check('verdict None when PDC/ODC missing',
          verdicts_for(aggregate(
              [{'module': 'm', 'code': 'LDC', 'gate': np.ones((1, 4)),
                'p_hh': 0.1}]), 'm') is None)

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', '-c', default=None,
                    help='A2 (haar_gate) config yml')
    ap.add_argument('--resume', '-r', default=None,
                    help='A2 checkpoint (best_stg2.pth)')
    ap.add_argument('--ann', default=None,
                    help='val ann json (for image_id -> augmentation code)')
    ap.add_argument('--update', '-u', nargs='+', default=None)
    ap.add_argument('--out', default='prep/innov1/gate_probe/summary.json')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()

    if args.self_test_only:
        sys.exit(self_test())
    missing = [k for k in ('config', 'resume', 'ann') if not getattr(args, k)]
    if missing:
        ap.error(f'missing required args: {", ".join("--" + m for m in missing)}')
    run(args)


if __name__ == '__main__':
    main()
