#!/usr/bin/env python3
"""Summarize experiment O (wavelet-suppression probe, failure-diagnosis doc §6).

Reads the per-variant `by_code_*.json` files produced by
`effective_n.py --by-code` over the directories built by wavelet_suppress.py,
and prints one table: variant x {ODC, LDC, DDC, GDC, PDC} AP (percent) plus
each variant's delta vs `id`.

Three correctness checks are run and explicitly reported (§2.3):
  1. `id`'s five-code APs match the baseline within 0.05 (codec has no effect;
     if not, every other delta must be corrected by the id delta);
  2. every variant's ODC/LDC/DDC equals the baseline BIT-EXACTLY (those images
     are symlinks -- any difference means the probe touched files it must not);
  3. best PDC of the nonlinear family (clip*/shrink*) vs the linear family
     (hh*/det*), with the §2 prediction verdict printed.

Then prints the pre-registered decision (>=35 / 25~35 / <25) WITH all three
tiers' original text, plus the "clip hurts ODC?" companion reading (§6).

Self-test:
    python tools/analysis/summarize_wavelet_probe.py --self-test-only
"""

import argparse
import glob as globmod
import json
import os
import sys

CODES = ['ODC', 'LDC', 'DDC', 'GDC', 'PDC']
VARIANT_ORDER = ['id', 'hh0', 'hh50', 'det50', 'det0',
                 'clip1.0', 'clip2.0', 'shrink1.0', 'shrink2.0']
LINEAR_PREFIXES = ('hh', 'det')        # what A1/A2 can express (§2)
NONLINEAR_PREFIXES = ('clip', 'shrink')  # what they CANNOT express (§2)
SYMLINK_CODES = ['ODC', 'LDC', 'DDC']  # never touched by the probe

# pre-registered decision tiers, verbatim from failure-diagnosis §6
TIERS = [
    ('>= 35', '（接近/超过 median5 的 39.28）频带路径有效，但必须是逐系数非线性',
     '改进 → 方向 A（SubbandCoeffClip + 落点前移，见实现文档 §4）'),
    ('25 ~ 35', '有效但上限明显低于经典去噪',
     '只能算"零开销的部分替代"，价值不足以撑主创新点 → 放弃主线，降级为消融'),
    ('< 25', '频带路径本身无效', '放弃（路径 C：只保留等价性证明作方法论注记，'
     '立刻全力转创新点 2）'),
]


def load_by_code(path):
    """by_code_*.json -> {code: AP in percent}."""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    bc = d.get('by_code', {})
    return {c: float(bc[c]['AP']) * 100.0 for c in CODES if c in bc}


def discover_variants(probe_dir):
    out = {}
    for p in sorted(globmod.glob(os.path.join(probe_dir, 'by_code_*.json'))):
        name = os.path.basename(p)[len('by_code_'):-len('.json')]
        out[name] = load_by_code(p)
    ordered = [v for v in VARIANT_ORDER if v in out]
    ordered += sorted(v for v in out if v not in VARIANT_ORDER)
    return ordered, out


def summarize(probe_dir, baseline_path):
    variants, table = discover_variants(probe_dir)
    if 'id' not in table:
        sys.exit(f'[summarize] no by_code_id.json in {probe_dir} -- the id '
                 f'control is mandatory (§6: all deltas are relative to it)')
    baseline = load_by_code(baseline_path)
    id_row = table['id']

    checks = {}
    # 1. codec neutrality: id vs baseline within 0.05 percent points
    d_id = {c: id_row.get(c, float('nan')) - baseline[c] for c in CODES
            if c in baseline and c in id_row}
    checks['id_matches_baseline'] = {
        'pass': bool(d_id) and all(abs(v) < 0.05 for v in d_id.values()),
        'delta_vs_baseline': d_id,
    }
    # 2. symlink integrity: ODC/LDC/DDC bit-exact vs baseline for EVERY variant
    bad = []
    for v, row in table.items():
        for c in SYMLINK_CODES:
            if c in baseline and c in row and row[c] != baseline[c]:
                bad.append((v, c, row[c], baseline[c]))
    checks['symlink_codes_bit_exact'] = {'pass': not bad, 'violations': bad}
    # 3. linear vs nonlinear family best (PDC)
    def best_pdc(prefixes):
        vals = [(v, row['PDC']) for v, row in table.items()
                if v.startswith(prefixes) and 'PDC' in row]
        return max(vals, key=lambda kv: kv[1]) if vals else (None, float('nan'))
    lin_v, lin = best_pdc(LINEAR_PREFIXES)
    nlin_v, nlin = best_pdc(NONLINEAR_PREFIXES)
    pred_ok = bool(nlin > lin) if (nlin == nlin and lin == lin) else None
    checks['nonlinear_beats_linear'] = {
        'linear_best': {'variant': lin_v, 'PDC': lin},
        'nonlinear_best': {'variant': nlin_v, 'PDC': nlin},
        'prediction': ('【符合 §2 预测】非线性族 > 线性族' if pred_ok
                       else '【与 §2 预测相反】非线性族未超过线性族'),
        'pass': pred_ok,
    }
    # §6 companion: does clip hurt ODC? (> 2 percent points drop = misfire risk)
    clip_odc = {v: row.get('ODC', float('nan')) - baseline.get('ODC', float('nan'))
                for v, row in table.items() if v.startswith('clip')}
    checks['clip_odc_cost'] = {
        'delta_vs_baseline': clip_odc,
        'warning': any(d < -2.0 for d in clip_odc.values()),
    }

    # decision tier on the best clip* PDC
    clip_vals = [row['PDC'] for v, row in table.items()
                 if v.startswith('clip') and 'PDC' in row]
    best_clip = max(clip_vals) if clip_vals else float('nan')
    if best_clip != best_clip:
        tier, action = 'N/A', 'no clip* variant found'
    elif best_clip >= 35:
        tier, action = TIERS[0][0], TIERS[0][2]
    elif best_clip >= 25:
        tier, action = TIERS[1][0], TIERS[1][2]
    else:
        tier, action = TIERS[2][0], TIERS[2][2]

    return {'baseline': baseline, 'variants': variants, 'table': table,
            'checks': checks,
            'decision': {'best_clip_PDC': best_clip, 'tier': tier,
                         'action': action}}


def render(s):
    table, baseline = s['table'], s['baseline']
    print('\n===== 实验 O 汇总（AP，百分点；括号内为 Δ vs id）=====')
    head = 'variant'.ljust(12) + ''.join(c.rjust(16) for c in CODES)
    print(head)
    print('baseline'.ljust(12) +
          ''.join(f'{baseline[c]:16.2f}' for c in CODES))
    for v in s['variants']:
        row = table[v]
        cells = []
        for c in CODES:
            if c not in row:
                cells.append(' ' * 16)
                continue
            d = row[c] - table['id'].get(c, float('nan'))
            cells.append(f'{row[c]:9.2f}({d:+5.2f})'.rjust(16))
        print(v.ljust(12) + ''.join(cells))

    print('\n===== 三条校验 =====')
    c1 = s['checks']['id_matches_baseline']
    print(f'[1] id vs baseline < 0.05 : {"PASS" if c1["pass"] else "FAIL"}  '
          + '  '.join(f'{c}:{d:+.3f}' for c, d in c1['delta_vs_baseline'].items()))
    if not c1['pass']:
        print('    -> 编解码本身有影响：后面所有 Δ 都要减掉 id 的 Δ')
    c2 = s['checks']['symlink_codes_bit_exact']
    print(f'[2] ODC/LDC/DDC 逐位相同  : {"PASS" if c2["pass"] else "FAIL"}')
    for v, c, got, exp in c2['violations'][:10]:
        print(f'    VIOLATION {v} {c}: {got} != baseline {exp}')
    if not c2['pass']:
        print('    -> 探针动了不该动的文件（软链被破坏），结果全部作废')
    c3 = s['checks']['nonlinear_beats_linear']
    print(f'[3] 非线性族 vs 线性族    : '
          f'线性最好 {c3["linear_best"]["variant"]}='
          f'{c3["linear_best"]["PDC"]:.2f}  非线性最好 '
          f'{c3["nonlinear_best"]["variant"]}={c3["nonlinear_best"]["PDC"]:.2f}'
          f'  -> {c3["prediction"]}')
    c4 = s['checks']['clip_odc_cost']
    print(f'[4] clip 的 ODC 代价      : '
          + '  '.join(f'{v}:{d:+.2f}' for v, d in c4['delta_vs_baseline'].items())
          + ('  -> 警告：掉超过 2，限幅会误伤缺陷，网内版本必须条件触发'
             if c4['warning'] else ''))

    print('\n===== 预注册落子（失败诊断 §6 判据表，三档原文）=====')
    d = s['decision']
    print(f'clip* 里最好的 PDC AP = {d["best_clip_PDC"]:.2f}  '
          f'(参照：median5 = 39.28，baseline = 15.86)')
    for tier, meaning, action in TIERS:
        mark = ' <== 命中' if tier == d['tier'] else ''
        print(f'  [{tier}] {meaning} -> {action}{mark}')
    print(f'落子建议：{d["action"]}')


def self_test():
    print('[self-test] summarize_wavelet_probe')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    import tempfile

    def write_probe(root, table, baseline):
        os.makedirs(root, exist_ok=True)
        for v, row in table.items():
            payload = {'by_code': {c: {'AP': row[c] / 100.0} for c in row}}
            with open(os.path.join(root, f'by_code_{v}.json'), 'w') as f:
                json.dump(payload, f)
        with open(os.path.join(root, 'baseline_by_code.json'), 'w') as f:
            json.dump({'by_code': {c: {'AP': baseline[c] / 100.0}
                                   for c in baseline}}, f)

    base = {'ODC': 51.43, 'LDC': 50.83, 'DDC': 49.99, 'GDC': 24.77,
            'PDC': 15.86}

    # ---- scenario A: everything healthy, clip wins, tier >= 35 ----
    tmp = tempfile.mkdtemp(prefix='sumwave_ok_')
    tbl_ok = {
        'id': dict(base),
        'hh0': {**base, 'GDC': 24.9, 'PDC': 17.0},
        'hh50': {**base, 'GDC': 23.0, 'PDC': 20.0},
        'det50': {**base, 'PDC': 19.0},
        'det0': {**base, 'PDC': 18.0},
        'clip1.0': {**base, 'ODC': 50.9, 'GDC': 26.0, 'PDC': 40.0},
        'clip2.0': {**base, 'GDC': 25.0, 'PDC': 36.0},
        'shrink1.0': {**base, 'PDC': 22.0},
        'shrink2.0': {**base, 'PDC': 24.0},
    }
    write_probe(tmp, tbl_ok, base)
    s = summarize(tmp, os.path.join(tmp, 'baseline_by_code.json'))
    check('A: variant order follows VARIANT_ORDER',
          s['variants'] == VARIANT_ORDER, str(s['variants']))
    check('A: check1 passes (id == baseline)',
          s['checks']['id_matches_baseline']['pass'])
    # ODC differs for clip1.0 -> check2 must FAIL and name the violation
    v2 = s['checks']['symlink_codes_bit_exact']
    check('A: check2 catches the modified ODC (clip1.0)',
          not v2['pass'] and any(x[0] == 'clip1.0' and x[1] == 'ODC'
                                 for x in v2['violations']))
    c3 = s['checks']['nonlinear_beats_linear']
    check('A: nonlinear best = clip1.0 40.0 > linear best 20.0',
          c3['nonlinear_best'] == {'variant': 'clip1.0', 'PDC': 40.0} and
          c3['linear_best']['PDC'] == 20.0 and c3['pass'])
    check('A: tier >= 35 -> path A',
          s['decision']['tier'] == '>= 35' and '方向 A' in s['decision']['action'],
          str(s['decision']))
    check('A: clip ODC cost warning fires (50.9-51.43 = -0.53 < 2? no)',
          not s['checks']['clip_odc_cost']['warning'])

    # ---- scenario B: healthy symlinks, codec-neutral, but clip < linear ----
    tmp2 = tempfile.mkdtemp(prefix='sumwave_bad_')
    tbl_bad = {
        'id': dict(base),
        'hh50': {**base, 'PDC': 30.0},
        'clip1.0': {**base, 'PDC': 20.0},
        'shrink1.0': {**base, 'PDC': 18.0},
    }
    write_probe(tmp2, tbl_bad, base)
    s2 = summarize(tmp2, os.path.join(tmp2, 'baseline_by_code.json'))
    check('B: check1+check2 pass',
          s2['checks']['id_matches_baseline']['pass'] and
          s2['checks']['symlink_codes_bit_exact']['pass'])
    check('B: prediction verdict is 与 §2 预测相反',
          not s2['checks']['nonlinear_beats_linear']['pass'] and
          '与 §2 预测相反' in s2['checks']['nonlinear_beats_linear']['prediction'])
    check('B: tier 25~35 (best clip 20 -> <25) -> path C',
          s2['decision']['best_clip_PDC'] == 20.0 and
          s2['decision']['tier'] == '< 25', str(s2['decision']))

    # ---- scenario C: codec not neutral -> check1 fails ----
    tmp3 = tempfile.mkdtemp(prefix='sumwave_codec_')
    tbl_c = {'id': {**base, 'PDC': 16.5}, 'clip1.0': {**base, 'PDC': 40.0}}
    write_probe(tmp3, tbl_c, base)
    s3 = summarize(tmp3, os.path.join(tmp3, 'baseline_by_code.json'))
    check('C: check1 fails when id deviates by 0.64',
          not s3['checks']['id_matches_baseline']['pass'])

    # ---- scenario D: ODC cost warning ----
    tmp4 = tempfile.mkdtemp(prefix='sumwave_odc_')
    tbl_d = {'id': dict(base), 'clip1.0': {**base, 'ODC': 48.0, 'PDC': 40.0}}
    write_probe(tmp4, tbl_d, base)
    s4 = summarize(tmp4, os.path.join(tmp4, 'baseline_by_code.json'))
    check('D: ODC drop 3.43 triggers the misfire warning',
          s4['checks']['clip_odc_cost']['warning'])

    # render must not crash on any scenario
    for s_ in (s, s2, s3, s4):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            render(s_)
    check('render runs on all scenarios', True)

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe-dir', default='prep/innov1/wavelet_probe',
                    help='dir containing by_code_<variant>.json')
    ap.add_argument('--baseline', default='prep/baseline/seed0/by_code.json',
                    help='baseline by_code json (same pipeline, seed0)')
    ap.add_argument('--out', default=None, help='write the summary json here')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()
    if args.self_test_only:
        sys.exit(self_test())
    s = summarize(args.probe_dir, args.baseline)
    render(s)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(f'[out] {args.out}')


if __name__ == '__main__':
    main()
