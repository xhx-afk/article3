"""SRFF-V1.2 fast36 adapter pilot 汇总（实现文档 §5.4 / 用户手册 §9,§10.3,§11）。

两阶段子命令：

* ``select``：读 epoch12/24/36 的 GDC/PDC ``auto`` 结果 + 冻结审计，按 RobustMean=(ΔGDC+ΔPDC)/2
  最大且两域均非负选择 checkpoint（不用 ODC/LDC/DDC 早停）；写 checkpoint_selection.{json,md}。
* ``final``：读 selected checkpoint 的 baseline/force_off/auto 的整体+五域+类别结果、冻结审计、机制诊断，
  计算 §11 验收量（ΔDomain/CleanMean/WorstClean/RobustMean/RobustAdvantage/TrainingEffect/AdapterEffect）
  并打印 §11.2/§11.3/§11.4 确定性标签；写 §9 结构的 results.{json,md}。

只读 JSON、仅依赖标准库；复用 summarize_srff_eval 的解析。缺文件/NaN/身份不符即非零退出，不补零。
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from summarize_srff_eval import DOMS, load, gv, per_category, fmt  # noqa: E402

CLEAN = ['ODC', 'LDC', 'DDC']
ROBUST = ['GDC', 'PDC']
EPOCHS = ['012', '024', '036']
EXIT_GATE = 2


def _sub(a, b):
    return None if (a is None or b is None) else a - b


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def _has_nan(o):
    if isinstance(o, bool):
        return False
    if isinstance(o, dict):
        return any(_has_nan(v) for v in o.values())
    if isinstance(o, list):
        return any(_has_nan(v) for v in o)
    if isinstance(o, float):
        return not math.isfinite(o)
    return False


def _die(msgs):
    print('[PILOT-GATE-FAIL] 完整性门禁未通过，非零退出：')
    for m in msgs:
        print('  -', m)
    raise SystemExit(EXIT_GATE)


def _load_ap(path, errs, need_cat=False):
    j = load(path)
    if not isinstance(j, dict) or '__missing__' in j:
        errs.append(f'缺失文件: {path}')
        return None, None
    if '__error__' in j:
        errs.append(f'损坏: {path}')
        return None, None
    if _has_nan(j):
        errs.append(f'含 NaN/Infinity: {path}')
    ap = gv(j, 'AP')
    if ap is None:
        errs.append(f'缺 coco_eval_bbox.AP: {path}')
    cats = per_category(j) if need_cat else {}
    return ap, cats


def _baseline_ap(baseline_dir, scope, errs, tpl='baseline_{scope}.json'):
    """baseline AP：优先 init_equivalence.json 的 subsets[scope].baseline_AP，其次 baseline_{scope}.json。"""
    ie = os.path.join(baseline_dir, 'init_equivalence.json')
    if os.path.isfile(ie):
        j = load(ie)
        sub = ((j.get('subsets', {}) or {}).get(scope) or {})
        if sub.get('baseline_AP') is not None:
            return float(sub['baseline_AP'])
    p = os.path.join(baseline_dir, tpl.format(scope=scope))
    if os.path.isfile(p):
        ap, _ = _load_ap(p, errs)
        return ap
    errs.append(f'baseline {scope} 不可得（既无 init_equivalence.json 子集，也无 {p}）')
    return None


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
def do_select(args):
    errs = []
    rows = {}
    for ep in EPOCHS:
        gdc_p = os.path.join(args.eval_dir, args.select_tpl.format(epoch=ep, dom='GDC'))
        pdc_p = os.path.join(args.eval_dir, args.select_tpl.format(epoch=ep, dom='PDC'))
        g_ap, _ = _load_ap(gdc_p, errs)
        p_ap, _ = _load_ap(pdc_p, errs)
        bg = _baseline_ap(args.baseline_dir, 'GDC', errs)
        bp = _baseline_ap(args.baseline_dir, 'PDC', errs)
        dg, dp = _sub(g_ap, bg), _sub(p_ap, bp)
        # 冻结审计
        audit_ok = None
        if args.audit_dir:
            ap_ = os.path.join(args.audit_dir, args.audit_tpl.format(epoch=ep))
            aj = load(ap_) if os.path.isfile(ap_) else {}
            audit_ok = bool(aj.get('ok')) if isinstance(aj, dict) and 'ok' in aj else None
            if audit_ok is None:
                errs.append(f'epoch{ep} 冻结审计缺失或无 ok 字段: {ap_}')
        rows[ep] = {'GDC_AP': g_ap, 'PDC_AP': p_ap, 'dGDC': dg, 'dPDC': dp,
                    'RobustMean': _mean([dg, dp]), 'audit_ok': audit_ok}
    if errs:
        _die(errs)
    # 选择规则（§9）
    cand = [ep for ep in EPOCHS if rows[ep]['audit_ok'] and rows[ep]['dGDC'] is not None
            and rows[ep]['dPDC'] is not None and rows[ep]['dGDC'] >= 0 and rows[ep]['dPDC'] >= 0]
    selected, reason = None, ''
    if not cand:
        reason = '无 checkpoint 同时满足 冻结审计通过 且 ΔGDC>=0 且 ΔPDC>=0 → 进入失败分流，不做完整五域评估'
    else:
        best = max(rows[ep]['RobustMean'] for ep in cand)
        tied = [ep for ep in cand if best - rows[ep]['RobustMean'] <= 0.02]
        selected = sorted(tied)[0]   # 差值 0.02 内选更早 epoch
        reason = f'RobustMean 最大={best:.4f}（0.02 内取更早 epoch）'
    out = {'epochs': rows, 'candidates': cand, 'selected_checkpoint':
           (args.checkpoint_tpl.format(epoch=selected) if selected else None),
           'selected_epoch': selected, 'reason': reason}
    _write(args.output, args.markdown, out, _select_md(out))
    print(f'[select] selected={selected} reason={reason}')
    if selected is None:
        raise SystemExit(EXIT_GATE)


def _select_md(out):
    L = ['# SRFF-V1.2 checkpoint 选择（仅 GDC/PDC，不看 clean 三域）', '',
         '| epoch | GDC AP | PDC AP | ΔGDC | ΔPDC | RobustMean | 冻结审计 |', '|---|---:|---:|---:|---:|---:|:--:|']
    for ep, r in out['epochs'].items():
        L.append(f"| {ep} | {fmt(r['GDC_AP'])} | {fmt(r['PDC_AP'])} | {_s(r['dGDC'])} | {_s(r['dPDC'])} "
                 f"| {_s(r['RobustMean'])} | {'PASS' if r['audit_ok'] else 'FAIL/NA'} |")
    L += ['', f"**selected_checkpoint**: `{out['selected_checkpoint']}`（epoch {out['selected_epoch']}）",
          f"理由：{out['reason']}"]
    return '\n'.join(L)


def _s(v):
    return 'NA' if v is None else f'{v:+.4f}'


# ---------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------
def do_final(args):
    errs = []
    states = {}
    for st in ('force_off', 'auto'):
        states[st] = {}
        for scope in ['overall'] + DOMS:
            p = os.path.join(args.eval_dir, args.state_tpl.format(state=st, scope=scope))
            ap, cats = _load_ap(p, errs, need_cat=(scope == 'overall'))
            states[st][scope] = {'AP': ap, 'cat': cats}
    base = {}
    for scope in ['overall'] + DOMS:
        if scope == 'overall':
            p = os.path.join(args.eval_dir, args.state_tpl.format(state='baseline', scope='overall'))
            ap, cats = (None, {})
            if os.path.isfile(p):
                ap, cats = _load_ap(p, errs, need_cat=True)
            else:
                ap = _baseline_ap(args.baseline_dir, 'overall', errs)
            base['overall'] = {'AP': ap, 'cat': cats}
        else:
            p = os.path.join(args.eval_dir, args.state_tpl.format(state='baseline', scope=scope))
            if os.path.isfile(p):
                ap, _ = _load_ap(p, errs)
            else:
                ap = _baseline_ap(args.baseline_dir, scope, errs)
            base[scope] = {'AP': ap}
    if errs:
        _die(errs)

    # §11 指标
    def d(state, scope):
        return _sub(states[state][scope]['AP'], base[scope]['AP'])
    delta = {scope: d('auto', scope) for scope in ['overall'] + DOMS}
    clean_mean = _mean([delta[s] for s in CLEAN])
    worst_clean = min([delta[s] for s in CLEAN if delta[s] is not None], default=None)
    robust_mean = _mean([delta[s] for s in ROBUST])
    robust_adv = None if robust_mean is None or clean_mean is None else robust_mean - max(0.0, -clean_mean)
    train_eff = {scope: _sub(states['force_off'][scope]['AP'], base[scope]['AP']) for scope in ['overall'] + DOMS}
    adapter_eff = {scope: _sub(states['auto'][scope]['AP'], states['force_off'][scope]['AP']) for scope in ['overall'] + DOMS}
    # overall scratch ΔAP
    sc_auto = states['auto']['overall']['cat'].get('scratch')
    sc_base = base['overall']['cat'].get('scratch')
    scratch_delta = _sub(sc_auto, sc_base)

    metrics = {'overall': {'baseline': base['overall']['AP'], 'force_off': states['force_off']['overall']['AP'],
                           'auto': states['auto']['overall']['AP'], 'delta': delta['overall']},
               'categories': {'auto': states['auto']['overall']['cat'], 'baseline': base['overall']['cat'],
                              'scratch_delta': scratch_delta}}
    for scope in DOMS:
        metrics[scope] = {'baseline': base[scope]['AP'], 'force_off': states['force_off'][scope]['AP'],
                          'auto': states['auto'][scope]['AP'], 'delta': delta[scope]}

    # 标签
    labels = _labels(delta, clean_mean, worst_clean, robust_mean, scratch_delta, train_eff)
    decision = _decision(delta, clean_mean, worst_clean, robust_mean, scratch_delta, labels, args)

    sel = load(args.selection) if args.selection and os.path.isfile(args.selection) else {}
    payload = {
        'identity': {'selection': sel.get('selected_checkpoint'), 'eval_dir': args.eval_dir},
        'baseline_load': _read_json(args.baseline_load) if args.baseline_load else {},
        'freeze_audit': _read_json(args.frozen_audit) if args.frozen_audit else {},
        'checkpoint_selection': sel,
        'metrics': metrics,
        'causal': {'training_effect': train_eff, 'adapter_effect': adapter_eff,
                   'total_effect': delta},
        'mechanism': {'clean_mean': clean_mean, 'worst_clean': worst_clean, 'robust_mean': robust_mean,
                      'robust_advantage': robust_adv, 'overall_scratch_delta': scratch_delta,
                      'diagnostics': _read_json(args.diagnostics) if args.diagnostics else {}},
        'decision': {'labels': labels, **decision},
    }
    _write(args.output, args.markdown, payload, _final_md(payload))
    print(f'[final] labels={labels} decision={decision.get("verdict")}')


def _read_json(p):
    j = load(p) if p and os.path.isfile(p) else {}
    return j if isinstance(j, dict) and '__missing__' not in j and '__error__' not in j else {}


def _labels(delta, clean_mean, worst_clean, robust_mean, scratch_delta, train_eff):
    L = []
    dg, dp = delta.get('GDC'), delta.get('PDC')
    dov = delta.get('overall')
    # §11.4 立即失败
    if dg is not None and dg < 0:
        L.append('FAIL_GDC_NEGATIVE')
    if dp is not None and dp < 0:
        L.append('FAIL_PDC_NEGATIVE')
    if robust_mean is not None and robust_mean < 0.10:
        L.append('FAIL_ROBUST_MEAN_LT_0.10')
    if dov is not None and dov < 0:
        L.append('FAIL_OVERALL_NEGATIVE')
    if scratch_delta is not None and scratch_delta < -0.30:
        L.append('FAIL_SCRATCH_LT_-0.30')
    # §11.3 强方向
    if (dg is not None and dg >= 0.30 and dp is not None and dp >= 0.50 and robust_mean is not None
            and robust_mean >= 0.40 and dov is not None and dov >= 0.20 and clean_mean is not None
            and clean_mean >= -0.10 and scratch_delta is not None and scratch_delta >= -0.20):
        L.append('STRONG_DIRECTIONAL_PASS')
    # §11.2 方向性
    if (dg is not None and dg > 0 and dp is not None and dp > 0 and robust_mean is not None
            and robust_mean >= 0.30 and dov is not None and dov >= 0 and clean_mean is not None
            and clean_mean >= -0.15 and worst_clean is not None and worst_clean >= -0.30
            and scratch_delta is not None and scratch_delta >= -0.30):
        L.append('DIRECTIONAL_PASS')
    if robust_mean is not None and 0.10 <= robust_mean < 0.30 and not any(x.startswith('FAIL') for x in L):
        L.append('BORDERLINE')
    return L


def _decision(delta, clean_mean, worst_clean, robust_mean, scratch_delta, labels, args):
    if any(x.startswith('FAIL') for x in labels):
        verdict = 'FAIL'
    elif 'STRONG_DIRECTIONAL_PASS' in labels:
        verdict = 'STRONG_PASS'
    elif 'DIRECTIONAL_PASS' in labels:
        verdict = 'DIRECTIONAL_PASS'
    elif 'BORDERLINE' in labels:
        verdict = 'BORDERLINE'
    else:
        verdict = 'INCONCLUSIVE'
    return {'verdict': verdict,
            'note': 'fast36 seed0 路线筛选，非论文统计显著性；可信度硬门见手册 §11.1（需 frozen audit/init-equiv/身份齐全）'}


def _final_md(p):
    m, c = p['metrics'], p['causal']
    L = ['# SRFF-V1.2 fast36 seed0 pilot 结果', '',
         f"- selected checkpoint: `{p['identity'].get('selection')}`",
         f"- 判定: **{p['decision']['verdict']}**  标签: {p['decision']['labels']}", '',
         '## §11 因果量（AP point）',
         '| scope | baseline | force_off | auto | ΔDomain(auto-base) | TrainingEffect(off-base) | AdapterEffect(auto-off) |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for scope in ['overall'] + DOMS:
        row = m[scope]
        L.append(f"| {scope} | {fmt(row['baseline'])} | {fmt(row['force_off'])} | {fmt(row['auto'])} "
                 f"| {_s(c['total_effect'][scope])} | {_s(c['training_effect'][scope])} | {_s(c['adapter_effect'][scope])} |")
    mm = p['mechanism']
    L += ['', f"- CleanMean={_s(mm['clean_mean'])} WorstClean={_s(mm['worst_clean'])} "
          f"RobustMean={_s(mm['robust_mean'])} RobustAdvantage={_s(mm['robust_advantage'])}",
          f"- overall scratch ΔAP={_s(mm['overall_scratch_delta'])}", '',
          '> 本文件为工具确定性输出；可信度硬门（frozen audit / init-equiv / 身份）须先全部通过再解读 AP。']
    return '\n'.join(L)


def _write(out_json, out_md, payload, md):
    if out_json:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        print(f'[saved] {out_json}')
    if out_md:
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md + '\n')
        print(f'[saved] {out_md}')


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.2 pilot 汇总（select/final）')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select')
    s.add_argument('--baseline-dir', required=True)
    s.add_argument('--eval-dir', required=True)
    s.add_argument('--audit-dir', default=None)
    s.add_argument('--output', required=True)
    s.add_argument('--markdown', default=None)
    s.add_argument('--select-tpl', default='e{epoch}_{dom}_auto.json')
    s.add_argument('--audit-tpl', default='adapter_epoch{epoch}.json')
    s.add_argument('--checkpoint-tpl', default='adapter_epoch{epoch}.pth')
    s.set_defaults(func=do_select)
    f = sub.add_parser('final')
    f.add_argument('--baseline-dir', required=True)
    f.add_argument('--eval-dir', required=True)
    f.add_argument('--selection', default=None)
    f.add_argument('--frozen-audit', default=None)
    f.add_argument('--diagnostics', default=None)
    f.add_argument('--baseline-load', default=None)
    f.add_argument('--output', required=True)
    f.add_argument('--markdown', default=None)
    f.add_argument('--state-tpl', default='{state}/{scope}.json')
    f.set_defaults(func=do_final)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
