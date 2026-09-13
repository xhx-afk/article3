"""SRFF-V1.1 局部门与双专家容量拆分汇总（11 状态 × GDC/PDC，α 响应曲线 + 分流标签）。

依据 ``SRFF-V1.1-局部门与双专家容量拆分-实现文档-Agent.md`` §7。同一 V1.1 checkpoint 上：

    ref_force_off : global force_off + learned local + learned router（= 上一轮 O）
    ref_learned   : global force_on  + learned local + learned router（= 上一轮 I）
    {mix,gaussian,trimmed}_a{002,005,010}
                  : global force_on + constant local α∈{0.02,0.05,0.10} + router∈{learned,gaussian_only,trimmed_only}

派生量（AP point）：
    GlobalGateOnEffect = ref_learned - ref_force_off       # 不再称 FullCapacity
    GaussianBestGDC    = max_α (gaussian_α_GDC - ref_force_off_GDC)
    TrimmedBestPDC     = max_α (trimmed_α_PDC  - ref_force_off_PDC)
    MixBestRobustMean  = max_α mean(ΔGDC, ΔPDC)（mix 族）

只读 JSON、仅依赖标准库；复用 summarize_srff_eval 的解析。任一 §7.6 完整性门禁不满足即非零退出。
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
from summarize_srff_eval import load, gv, per_category, fmt  # noqa: E402

DOMAINS = ['GDC', 'PDC']
CATS = ['blister', 'crack', 'hole', 'scratch']
METRICS = ['AP', 'AP50', 'AP75', 'AR100']
ALPHAS = [('a002', 0.02), ('a005', 0.05), ('a010', 0.10)]
FAMILIES = ['mix', 'gaussian', 'trimmed']
CAPACITY_STATES = [f'{fam}_{atag}' for fam in FAMILIES for atag, _ in ALPHAS]
REF_STATES = ['ref_force_off', 'ref_learned']
ALL_STATES = REF_STATES + CAPACITY_STATES
# 每个专家族的目标域（gaussian→GDC，trimmed→PDC，mix→两域）
TARGET_DOMS = {'gaussian': ['GDC'], 'trimmed': ['PDC'], 'mix': ['GDC', 'PDC']}
EXPECTED_CKPT_SHA = '0c6434ebca4f1887e831b924455a2201c6bfad0a3c227411184e4c3ea31802a0'
# §7.6 参照复现目标（AP point，容差 0.02）
REF_TARGETS = {('ref_force_off', 'GDC'): 24.44, ('ref_force_off', 'PDC'): 15.77,
               ('ref_learned', 'GDC'): 24.46, ('ref_learned', 'PDC'): 15.81}
REF_TOL = 0.02
EXIT_GATE = 2


def _sub(a, b):
    return None if (a is None or b is None) else a - b


def _has_nan(obj):
    if isinstance(obj, bool):
        return False
    if isinstance(obj, dict):
        return any(_has_nan(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nan(v) for v in obj)
    if isinstance(obj, float):
        return not math.isfinite(obj)
    return False


def _d4(v):
    return 'NA' if v is None else f'{v:+.4f}'


def load_metrics(path, gate_errors):
    j = load(path)
    if not isinstance(j, dict) or '__missing__' in j:
        gate_errors.append(f'缺失文件: {path}')
        return None
    if '__error__' in j:
        gate_errors.append(f'损坏/不可解析: {path} ({j.get("__error__")})')
        return None
    if _has_nan(j):
        gate_errors.append(f'含 NaN/Infinity: {path}')
    if gv(j, 'AP') is None:
        gate_errors.append(f'缺 coco_eval_bbox.AP: {path}')
    cats = per_category(j)
    for c in CATS:
        if cats.get(c) is None:
            gate_errors.append(f'缺每类别 AP[{c}]: {path}')
    m = {k: gv(j, k) for k in METRICS}
    m['cat'] = {c: cats.get(c) for c in CATS}
    return m


def check_manifest(path, gate_errors):
    if not path or not os.path.isfile(path):
        gate_errors.append(f'identity manifest 缺失: {path}')
        return None
    mf = load(path)
    if not isinstance(mf, dict) or '__error__' in mf:
        gate_errors.append(f'identity manifest 损坏: {path}')
        return None
    if mf.get('training_performed') is not False:
        gate_errors.append('manifest training_performed != false（疑似发生训练）')
    if mf.get('experiment_type') != 'eval_only':
        gate_errors.append(f"manifest experiment_type != eval_only（{mf.get('experiment_type')}）")
    b, a = mf.get('checkpoint_sha256_before'), mf.get('checkpoint_sha256_after')
    if not b:
        gate_errors.append('manifest 缺 checkpoint_sha256_before')
    if a is not None and b != a:
        gate_errors.append('manifest checkpoint SHA before/after 不一致')
    exp = mf.get('expect_checkpoint_sha256', EXPECTED_CKPT_SHA)
    if b and b != exp:
        gate_errors.append(f'checkpoint SHA 与预期身份不一致: {b} != {exp}')
    return mf


def check_reference_repro(M, gate_errors):
    for (st, dom), tgt in REF_TARGETS.items():
        ap = (M.get(st) or {}).get(dom, {}).get('AP')
        if ap is None:
            gate_errors.append(f'参照 {st}_{dom} 无 AP')
        elif abs(ap - tgt) > REF_TOL:
            gate_errors.append(f'参照未复现 {st}_{dom}: 实际 {ap:.4f} vs 目标 {tgt:.2f} (±{REF_TOL})')


def build(M):
    """M[state][dom] = metrics dict。返回差值、α 曲线、派生量。"""
    ref_off = M['ref_force_off']
    res = {'diff_vs_ref_off': {}, 'alpha_curves': {}, 'derived': {}}
    for st in ALL_STATES:
        res['diff_vs_ref_off'][st] = {}
        for dom in DOMAINS:
            d = {}
            for k in METRICS:
                d[k] = _sub(M[st][dom].get(k), ref_off[dom].get(k))
            d['cat'] = {c: _sub(M[st][dom]['cat'].get(c), ref_off[dom]['cat'].get(c)) for c in CATS}
            res['diff_vs_ref_off'][st][dom] = d
    # α 响应曲线（相对 ref_force_off 的 ΔAP）
    for fam in FAMILIES:
        res['alpha_curves'][fam] = {dom: {} for dom in DOMAINS}
        for atag, _ in ALPHAS:
            st = f'{fam}_{atag}'
            for dom in DOMAINS:
                res['alpha_curves'][fam][dom][atag] = res['diff_vs_ref_off'][st][dom]['AP']
    # 派生量
    ggoe = {dom: _sub(M['ref_learned'][dom]['AP'], ref_off[dom]['AP']) for dom in DOMAINS}
    gbest = max([v for v in res['alpha_curves']['gaussian']['GDC'].values() if v is not None], default=None)
    tbest = max([v for v in res['alpha_curves']['trimmed']['PDC'].values() if v is not None], default=None)
    mix_means = []
    for atag, _ in ALPHAS:
        g = res['alpha_curves']['mix']['GDC'][atag]
        p = res['alpha_curves']['mix']['PDC'][atag]
        if g is not None and p is not None:
            mix_means.append((g + p) / 2.0)
    res['derived'] = {
        'GlobalGateOnEffect': ggoe,
        'GaussianBestGDC': gbest,
        'TrimmedBestPDC': tbest,
        'MixBestRobustMean': max(mix_means) if mix_means else None,
    }
    return res


def build_labels(M, res):
    labels, vals = [], {}
    ref_learned = M['ref_learned']
    curves = res['alpha_curves']
    gbest = res['derived']['GaussianBestGDC']
    tbest = res['derived']['TrimmedBestPDC']
    vals['GaussianBestGDC'] = gbest
    vals['TrimmedBestPDC'] = tbest
    vals['MixBestRobustMean'] = res['derived']['MixBestRobustMean']
    vals['GlobalGateOnEffect'] = res['derived']['GlobalGateOnEffect']

    # LOCAL_GATE_SUPPRESSION：任一 constant 模式在目标域比 ref_learned 高 >= +0.20
    lgs = []
    for fam in FAMILIES:
        for atag, _ in ALPHAS:
            st = f'{fam}_{atag}'
            for dom in TARGET_DOMS[fam]:
                gap = _sub(M[st][dom]['AP'], ref_learned[dom]['AP'])
                vals.setdefault('local_gate_gap', {})[f'{st}_{dom}'] = gap
                if gap is not None and gap >= 0.20:
                    lgs.append(f'{st}@{dom}={gap:+.3f}')
    if lgs:
        labels.append('LOCAL_GATE_SUPPRESSION')
    vals['local_gate_suppression_hits'] = lgs

    # GAUSSIAN / TRIMMED basis
    if gbest is not None and gbest >= 0.20:
        labels.append('GAUSSIAN_BASIS_POSITIVE')
    if gbest is not None and gbest <= 0.10:
        labels.append('GAUSSIAN_BASIS_WEAK')
    if tbest is not None and tbest >= 0.20:
        labels.append('TRIMMED_BASIS_POSITIVE')
    if tbest is not None and tbest <= 0.10:
        labels.append('TRIMMED_BASIS_WEAK')
    if gbest is not None and 0.10 < gbest < 0.20:
        labels.append('BORDERLINE_GAUSSIAN')
    if tbest is not None and 0.10 < tbest < 0.20:
        labels.append('BORDERLINE_TRIMMED')
    if gbest is not None and tbest is not None and gbest <= 0.10 and tbest <= 0.10:
        labels.append('NO_FIXED_BASIS_SIGNAL')

    # ROUTER_INTERFERENCE：专家单路在目标域比同 α learned-mix 高 >= +0.20
    ri = []
    for atag, _ in ALPHAS:
        g_gap = _sub(M[f'gaussian_{atag}']['GDC']['AP'], M[f'mix_{atag}']['GDC']['AP'])
        t_gap = _sub(M[f'trimmed_{atag}']['PDC']['AP'], M[f'mix_{atag}']['PDC']['AP'])
        vals.setdefault('router_gap', {})[f'gaussian_{atag}_GDC'] = g_gap
        vals.setdefault('router_gap', {})[f'trimmed_{atag}_PDC'] = t_gap
        if g_gap is not None and g_gap >= 0.20:
            ri.append(f'gaussian_{atag}@GDC={g_gap:+.3f}')
        if t_gap is not None and t_gap >= 0.20:
            ri.append(f'trimmed_{atag}@PDC={t_gap:+.3f}')
    if ri:
        labels.append('ROUTER_INTERFERENCE')
    vals['router_interference_hits'] = ri

    # OVERCORRECTION：α=0.02 为正且 α=0.10 比其下降 >= 0.30
    oc = []
    for fam in FAMILIES:
        for dom in TARGET_DOMS[fam]:
            d002 = curves[fam][dom]['a002']
            d010 = curves[fam][dom]['a010']
            drop = _sub(d002, d010)
            vals.setdefault('overcorrection', {})[f'{fam}_{dom}'] = {'a002': d002, 'a010': d010, 'drop': drop}
            if d002 is not None and d002 > 0 and drop is not None and drop >= 0.30:
                oc.append(f'{fam}@{dom}: a002={d002:+.3f} drop={drop:+.3f}')
    if oc:
        labels.append('OVERCORRECTION')
    vals['overcorrection_hits'] = oc
    return {'labels': labels, 'values': vals}


def route_suggestions(labels):
    s = []
    if 'GAUSSIAN_BASIS_POSITIVE' in labels and 'LOCAL_GATE_SUPPRESSION' in labels:
        s.append('GAUSSIAN_BASIS_POSITIVE + LOCAL_GATE_SUPPRESSION → V1.2 保留 Gaussian 固定先验，重构局部门')
    if 'GAUSSIAN_BASIS_WEAK' in labels:
        s.append('GAUSSIAN_BASIS_WEAK → V1.2 使用 learnable Gaussian-like residual expert')
    if 'TRIMMED_BASIS_POSITIVE' in labels:
        s.append('TRIMMED_BASIS_POSITIVE → 保留 trimmed 专家')
    if 'TRIMMED_BASIS_WEAK' in labels:
        s.append('TRIMMED_BASIS_WEAK → PDC 专家也需要重构')
    if 'ROUTER_INTERFERENCE' in labels:
        s.append('ROUTER_INTERFERENCE → 使用显式证据路由或受约束路由')
    if 'OVERCORRECTION' in labels:
        s.append('OVERCORRECTION → V1.2 必须设置硬残差预算，不能无界放大')
    s.append('（不变原则）V1.2 必须从原 baseline checkpoint 开始并冻结共享网络')
    return s


def build_md(args, M, res, verdict, mf, gate_warnings):
    L = []
    P = L.append
    P('# SRFF-V1.1 局部门与双专家容量拆分结果（fast72 seed0，GDC/PDC）')
    P('')
    P('> 由 `tools/wood/summarize_srff_v11_expert_capacity.py` 生成。单位 AP point；ΔAP 均相对 `ref_force_off`。')
    P('> `ref_learned - ref_force_off` 统一记为 **GlobalGateOnEffect**（本轮不使用旧的容量命名）。')
    P('')
    P(f'- eval-dir: `{args.eval_dir}`')
    P(f'- identity-manifest: `{args.identity_manifest}`')
    if mf:
        P(f"- checkpoint_sha256_before: {mf.get('checkpoint_sha256_before')}")
        P(f"- training_performed={mf.get('training_performed')} experiment_type={mf.get('experiment_type')} "
          f"git_commit={mf.get('git_commit')}")
    if gate_warnings:
        P('')
        for w in gate_warnings:
            P(f'> 警告：{w}')

    # 参照复现
    P('')
    P('## 0. 参照复现（±0.02）')
    P('| state | domain | AP | 目标 | |Δ| | 复现 |')
    P('|---|---|---:|---:|---:|:--:|')
    for (st, dom), tgt in REF_TARGETS.items():
        ap = M[st][dom]['AP']
        P(f'| {st} | {dom} | {fmt(ap)} | {tgt:.2f} | {abs(ap - tgt):.4f} | '
          f'{"✓" if abs(ap - tgt) <= REF_TOL else "✗"} |')

    # 11 状态 AP + 每类别
    P('')
    P('## 1. 11 状态 GDC/PDC AP（及相对 ref_force_off 的 ΔAP）')
    P('| state | GDC AP | GDC Δ | PDC AP | PDC Δ |')
    P('|---|---:|---:|---:|---:|')
    for st in ALL_STATES:
        P(f"| {st} | {fmt(M[st]['GDC']['AP'])} | {_d4(res['diff_vs_ref_off'][st]['GDC']['AP'])} "
          f"| {fmt(M[st]['PDC']['AP'])} | {_d4(res['diff_vs_ref_off'][st]['PDC']['AP'])} |")

    # 每类别 AP 差值
    P('')
    P('## 2. 每类别 AP 差值（相对 ref_force_off）')
    for dom in DOMAINS:
        P('')
        P(f'### {dom}')
        P('| state | blister | crack | hole | scratch |')
        P('|---|---:|---:|---:|---:|')
        for st in ALL_STATES:
            cd = res['diff_vs_ref_off'][st][dom]['cat']
            P(f"| {st} | {_d4(cd['blister'])} | {_d4(cd['crack'])} | {_d4(cd['hole'])} | {_d4(cd['scratch'])} |")

    # α 响应曲线
    P('')
    P('## 3. 三条 α 响应曲线（ΔAP vs ref_force_off）')
    for fam, tdom in (('mix', 'GDC/PDC'), ('gaussian', 'GDC(目标)'), ('trimmed', 'PDC(目标)')):
        P('')
        P(f'### {fam}-family（{tdom}）')
        P('| α | GDC ΔAP | PDC ΔAP |')
        P('|---|---:|---:|')
        for atag, av in ALPHAS:
            P(f"| {av:.2f} | {_d4(res['alpha_curves'][fam]['GDC'][atag])} "
              f"| {_d4(res['alpha_curves'][fam]['PDC'][atag])} |")

    # 派生量
    P('')
    P('## 4. 派生量')
    dv = res['derived']
    P(f"- GlobalGateOnEffect(ref_learned-ref_force_off)：GDC {_d4(dv['GlobalGateOnEffect']['GDC'])}、"
      f"PDC {_d4(dv['GlobalGateOnEffect']['PDC'])}")
    P(f"- GaussianBestGDC = {_d4(dv['GaussianBestGDC'])}")
    P(f"- TrimmedBestPDC = {_d4(dv['TrimmedBestPDC'])}")
    P(f"- MixBestRobustMean = {_d4(dv['MixBestRobustMean'])}")

    # 标签
    P('')
    P('## 5. §7.4 分流标签（连续值 + 命中）')
    vv = verdict['values']
    P(f"- GaussianBestGDC={_d4(vv['GaussianBestGDC'])}（>=+0.20 POSITIVE / <=+0.10 WEAK / 其间 BORDERLINE）")
    P(f"- TrimmedBestPDC={_d4(vv['TrimmedBestPDC'])}（>=+0.20 POSITIVE / <=+0.10 WEAK / 其间 BORDERLINE）")
    P(f"- LOCAL_GATE_SUPPRESSION 命中：{vv['local_gate_suppression_hits'] or '无'}")
    P(f"- ROUTER_INTERFERENCE 命中：{vv['router_interference_hits'] or '无'}")
    P(f"- OVERCORRECTION 命中：{vv['overcorrection_hits'] or '无'}")
    P('')
    P(f"**命中标签**：{verdict['labels'] if verdict['labels'] else '（无）'}")

    # 路线建议
    P('')
    P('## 6. §7.5 条件式 V1.2 路线建议（仅打印，不改代码）')
    for s in verdict['route_suggestions']:
        P(f'- {s}')
    P('')
    P('> 说明：本文件为工具确定性输出，不含人工结论。')
    return '\n'.join(L)


def build_json(args, M, res, verdict, mf, gate_warnings):
    return {
        'provenance': {'eval_dir': args.eval_dir, 'identity_manifest': args.identity_manifest,
                       'manifest': mf},
        'warnings': gate_warnings,
        'states': {st: {dom: M[st][dom] for dom in DOMAINS} for st in ALL_STATES},
        'diff_vs_ref_off': res['diff_vs_ref_off'],
        'alpha_curves': res['alpha_curves'],
        'derived': res['derived'],
        'verdict': verdict,
    }


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.1 局部门/双专家容量拆分汇总')
    ap.add_argument('--eval-dir', required=True, help='含 22 份 {state}_{dom}.json 的目录')
    ap.add_argument('--identity-manifest', required=True)
    ap.add_argument('--out-md', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--file-tpl', default='{state}_{dom}.json', help='文件模板（可 CLI 覆盖）')
    args = ap.parse_args()

    gate_errors, gate_warnings = [], []
    M = {}
    for st in ALL_STATES:
        M[st] = {}
        for dom in DOMAINS:
            path = os.path.join(args.eval_dir, args.file_tpl.format(state=st, dom=dom))
            M[st][dom] = load_metrics(path, gate_errors) or {}

    mf = check_manifest(args.identity_manifest, gate_errors)

    # 参照复现需在有 AP 时检查
    if all(M[st][dom].get('AP') is not None for (st, dom) in REF_TARGETS):
        check_reference_repro(M, gate_errors)
    else:
        gate_errors.append('参照状态缺 AP，无法核验复现')

    if gate_errors:
        print('[CAPACITY-GATE-FAIL] 完整性门禁未通过，非零退出：')
        for e in gate_errors:
            print('  -', e)
        raise SystemExit(EXIT_GATE)

    res = build(M)
    verdict = build_labels(M, res)
    verdict['route_suggestions'] = route_suggestions(verdict['labels'])

    md = build_md(args, M, res, verdict, mf, gate_warnings)
    payload = build_json(args, M, res, verdict, mf, gate_warnings)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(md)
    print(f'\n[saved] {args.out_md}')
    print(f'[saved] {args.out_json}')


if __name__ == '__main__':
    main()
