"""SRFF-V1.1 三态门控因果诊断统一汇总（B/N/O/I 四态 + 因果分解 + 确定性分流判据）。

对**同一** V1.1 checkpoint 的四种推理状态做统一汇总（依据
``SRFF-V1.1-三态门控因果诊断-实现文档-Agent.md`` §6）：

    B = baseline（原 fast72 seed0，另一条训练轨迹，仅作参照）
    N = auto       （V1.1 正常门控，复现当前 V1.1）
    O = force_off  （严格关闭 SRFF 后的 V1.1 共享权重）
    I = force_on   （严格走 V1 校正路径的 V1.1 能力上限）

因果分解（统一 AP point 百分制）：

    JointTrainingDriftEstimate = O - B   # 仅“联合训练漂移估计”：B 与 V1.1 是两条训练轨迹，禁止称严格因果
    NormalGateContribution     = N - O   # 同一 checkpoint 推理干预差，可直接因果判断
    FullSRFFCapacity           = I - O
    GateSuppression            = I - N

只读 JSON、仅依赖标准库；复用 ``summarize_srff_eval`` 的解析。任一 §6.5 完整性门禁不满足
即以非零码退出（缺文件 / NaN / 无 AP / 五域不齐 / D 关身份或 mode 不符 / manifest 不一致）。

用法::
    python tools/wood/summarize_srff_v11_causal.py \
      --baseline-dir <旧验收目录> --causal-dir <本轮 eval 目录> \
      --diagnostic-json <auto 模式 D 关 JSON> \
      --out-md <结果.md> --out-json <结果.json>
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

# 复用 V1 汇总脚本的解析逻辑（load / gv / per_category 等），避免重复实现。
from summarize_srff_eval import DOMS, KEYS, load, gv, per_category, fmt, sfmt  # noqa: E402

CLEAN = ['ODC', 'LDC', 'DDC']
ROBUST = ['GDC', 'PDC']
DOM_METRICS = ['AP', 'AP50', 'AP75', 'AR100']
DECOMP_KEYS = ['JointTrainingDriftEstimate', 'NormalGateContribution',
               'FullSRFFCapacity', 'GateSuppression']
# 状态记号 -> mode 名（B 为 baseline，单独用 baseline 模板）
STATE_MODE = {'N': 'auto', 'O': 'force_off', 'I': 'force_on'}
AUTO_REPRO_AP_DEFAULT = 38.64
EXIT_GATE = 2


# ---------------------------------------------------------------------------
# 基础算术与完整性
# ---------------------------------------------------------------------------
def _sub(a, b):
    return None if (a is None or b is None) else a - b


def _decomp(b, n, o, i):
    """按 §6.3 返回四项因果分解（输入均为 AP point 标量）。"""
    return {
        'JointTrainingDriftEstimate': _sub(o, b),
        'NormalGateContribution': _sub(n, o),
        'FullSRFFCapacity': _sub(i, o),
        'GateSuppression': _sub(i, n),
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


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


def _load_checked(path, gate_errors):
    j = load(path)
    if not isinstance(j, dict) or '__missing__' in j:
        gate_errors.append(f'缺失文件: {path}')
        return None
    if '__error__' in j:
        gate_errors.append(f'损坏/不可解析文件: {path} ({j.get("__error__")})')
        return None
    if _has_nan(j):
        gate_errors.append(f'含 NaN/Infinity: {path}')
    if gv(j, 'AP') is None:
        gate_errors.append(f'缺 coco_eval_bbox.AP: {path}')
    return j


def load_state(dirpath, overall_tpl, dom_tpl, tag, gate_errors):
    """加载一个状态（overall + 五域），逐个做完整性检查。"""
    ov = _load_checked(os.path.join(dirpath, overall_tpl), gate_errors)
    doms = {}
    for d in DOMS:
        doms[d] = _load_checked(os.path.join(dirpath, dom_tpl.format(dom=d)), gate_errors)
    return {'tag': tag, 'overall': ov, 'domains': doms}


# ---------------------------------------------------------------------------
# 取值访问器（AP point）
# ---------------------------------------------------------------------------
def ov_metric(state, m):
    return gv(state['overall'], m) if state['overall'] else None


def dom_metric(state, d, m):
    j = state['domains'].get(d)
    return gv(j, m) if j else None


def ov_cat(state, c):
    return per_category(state['overall']).get(c) if state['overall'] else None


def dom_cat(state, d, c):
    j = state['domains'].get(d)
    return per_category(j).get(c) if j else None


# ---------------------------------------------------------------------------
# D 关诊断 JSON 提取 + 身份门禁
# ---------------------------------------------------------------------------
def check_diagnostic(diag, gate_errors):
    meta = diag.get('metadata', {}) if isinstance(diag, dict) else {}
    audit = meta.get('domain_audit', {}) or {}
    if meta.get('partial_run') is not False:
        gate_errors.append(f"D 关 partial_run != false（实际 {meta.get('partial_run')}）")
    if meta.get('num_images') != 2760:
        gate_errors.append(f"D 关 num_images != 2760（实际 {meta.get('num_images')}）")
    if audit.get('audit_ok') is not True:
        gate_errors.append(f"D 关 domain_audit.audit_ok != true（实际 {audit.get('audit_ok')}）")
    if meta.get('srff_active_block_count') != 1:
        gate_errors.append(f"D 关 active block 数 != 1（实际 {meta.get('srff_active_block_count')}）")
    if meta.get('srff_global_gate_mode') != 'auto':
        gate_errors.append(f"D 关 mode != auto（实际 {meta.get('srff_global_gate_mode')}）")


def extract_diagnostic(diag):
    """提取各域 global_score/global_gate/pre_global_gate/gate 均值 + comparisons（原值，不伪造）。"""
    out = {'metadata': {}, 'per_domain_gate': {}, 'comparisons': {}}
    if not isinstance(diag, dict):
        return out
    meta = diag.get('metadata', {}) or {}
    out['metadata'] = {
        'srff_version': meta.get('srff_version'),
        'srff_global_gate_mode': meta.get('srff_global_gate_mode'),
        'srff_global_threshold_low': meta.get('srff_global_threshold_low'),
        'srff_global_threshold_high': meta.get('srff_global_threshold_high'),
        'srff_active_block_count': meta.get('srff_active_block_count'),
        'checkpoint_source': meta.get('checkpoint_source'),
        'checkpoint_sha256': meta.get('checkpoint_sha256'),
        'num_images': meta.get('num_images'),
        'partial_run': meta.get('partial_run'),
    }
    blocks = meta.get('srff_blocks') or []
    bname = blocks[0]['diag_name'] if blocks and 'diag_name' in blocks[0] else 'block0_p5_to_p4'
    region = (((diag.get('groups', {}) or {}).get('by_block_domain_region', {}) or {})
              .get(bname, {}) or {})
    want = ['global_score', 'global_gate', 'pre_global_gate', 'gate']
    for d in DOMS:
        allq = ((region.get(d, {}) or {}).get('all', {}) or {}).get('quantities', {}) or {}
        out['per_domain_gate'][d] = {w: (allq.get(w, {}) or {}).get('mean') for w in want}
    out['comparisons'] = (diag.get('comparisons', {}) or {}).get(bname, {}) or {}
    return out


def check_identity_manifest(path, gate_errors, warnings):
    if not path:
        warnings.append('未提供 identity manifest；N/O/I checkpoint 身份一致性未经本工具核验'
                        '（请依赖手册的 sha256sum before/after 对比）')
        return None
    mf = load(path)
    if not isinstance(mf, dict) or '__missing__' in mf or '__error__' in mf:
        gate_errors.append(f'identity manifest 缺失或损坏: {path}')
        return None
    if mf.get('training_performed') is not False:
        gate_errors.append(f"identity manifest training_performed != false（实际 {mf.get('training_performed')}）")
    modes = mf.get('modes') or []
    for m in ('auto', 'force_off', 'force_on'):
        if m not in modes:
            gate_errors.append(f'identity manifest modes 缺 {m}')
    before = mf.get('checkpoint_sha256_before')
    after = mf.get('checkpoint_sha256_after')
    if before and after and before != after:
        gate_errors.append('identity manifest checkpoint SHA before/after 不一致（checkpoint 被改动）')
    return mf


# ---------------------------------------------------------------------------
# 分流判据（§6.4，仅 fast72 机制分流，非统计显著性）
# ---------------------------------------------------------------------------
def build_verdicts(B, N, O, I, auto_repro_ap):
    clean_scratch = {k: _mean([dom_cat(s, d, 'scratch') for d in CLEAN])
                     for k, s in (('B', B), ('N', N), ('O', O), ('I', I))}
    cs_drift = _sub(clean_scratch['O'], clean_scratch['B'])       # O-B
    cs_gate = _sub(clean_scratch['N'], clean_scratch['O'])        # N-O
    gdc_cap = _sub(dom_metric(I, 'GDC', 'AP'), dom_metric(O, 'GDC', 'AP'))   # I-O
    gdc_supp = _sub(dom_metric(I, 'GDC', 'AP'), dom_metric(N, 'GDC', 'AP'))  # I-N
    pdc_gate = _sub(dom_metric(N, 'PDC', 'AP'), dom_metric(O, 'PDC', 'AP'))  # N-O
    pdc_supp = _sub(dom_metric(I, 'PDC', 'AP'), dom_metric(N, 'PDC', 'AP'))  # I-N
    n_ap = ov_metric(N, 'AP')
    repro_delta = None if n_ap is None else abs(n_ap - auto_repro_ap)

    vals = {
        'clean_scratch_O_minus_B': cs_drift, 'clean_scratch_N_minus_O': cs_gate,
        'GDC_I_minus_N': gdc_supp, 'GDC_I_minus_O': gdc_cap,
        'PDC_N_minus_O': pdc_gate, 'PDC_I_minus_N': pdc_supp,
        'N_overall_AP': n_ap, 'auto_repro_AP_target': auto_repro_ap,
        'auto_repro_abs_delta': repro_delta,
    }
    labels = []
    if cs_drift is not None and cs_drift <= -0.50:
        labels.append('SHARED_DRIFT')
    if cs_gate is not None and cs_gate <= -0.50:
        labels.append('CLEAN_FALSE_ACTIVATION')
    if gdc_supp is not None and gdc_supp >= 0.30:
        labels.append('GDC_GATE_SUPPRESSION')
    if gdc_cap is not None and gdc_cap <= 0.20:
        labels.append('GDC_EXPERT_WEAK')
    if pdc_gate is not None and pdc_supp is not None and pdc_gate > 0 and pdc_supp <= 0.20:
        labels.append('PDC_PATH_RETAINED')
    if repro_delta is not None and repro_delta > 0.02:
        labels.append('AUTO_REPRO_FAIL')
    return {'values': vals, 'labels': labels}


# ---------------------------------------------------------------------------
# 汇总计算
# ---------------------------------------------------------------------------
def build_results(B, N, O, I):
    states = {'B': B, 'N': N, 'O': O, 'I': I}
    res = {}
    # overall 7 指标
    res['overall'] = {m: _decomp(ov_metric(B, m), ov_metric(N, m), ov_metric(O, m), ov_metric(I, m))
                      for m in KEYS}
    # 每域 4 指标
    res['per_domain'] = {d: {m: _decomp(dom_metric(B, d, m), dom_metric(N, d, m),
                                        dom_metric(O, d, m), dom_metric(I, d, m))
                             for m in DOM_METRICS} for d in DOMS}
    # overall 每类别
    cats = sorted({c for s in states.values() for c in per_category(s['overall'] or {})},
                  key=lambda x: str(x))
    res['overall_per_category'] = {c: _decomp(ov_cat(B, c), ov_cat(N, c), ov_cat(O, c), ov_cat(I, c))
                                   for c in cats}
    # clean 三域 scratch 平均
    cs = {k: _mean([dom_cat(s, d, 'scratch') for d in CLEAN]) for k, s in states.items()}
    res['clean_scratch_avg'] = {'states': cs, 'decomp': _decomp(cs['B'], cs['N'], cs['O'], cs['I'])}
    # GDC/PDC 的 capacity / suppression / gate contribution（域 AP）
    res['gdc_pdc'] = {}
    for d in ROBUST:
        res['gdc_pdc'][d] = {
            'FullSRFFCapacity_I_minus_O': _sub(dom_metric(I, d, 'AP'), dom_metric(O, d, 'AP')),
            'GateSuppression_I_minus_N': _sub(dom_metric(I, d, 'AP'), dom_metric(N, d, 'AP')),
            'NormalGateContribution_N_minus_O': _sub(dom_metric(N, d, 'AP'), dom_metric(O, d, 'AP')),
        }
    res['categories'] = cats
    return res


# ---------------------------------------------------------------------------
# MD / JSON 输出
# ---------------------------------------------------------------------------
def _d4(v):
    return 'NA' if v is None else f'{v:+.4f}'


def build_md(args, states, res, verdicts, diagx, gate_warnings, manifest):
    L = []
    P = L.append
    P('# SRFF-V1.1 三态门控因果诊断结果（fast72 seed0）')
    P('')
    P('> 由 `tools/wood/summarize_srff_v11_causal.py` 生成。B/N/O/I 见下；因果分解单位 AP point。')
    P('> `O-B` 仅为**联合训练漂移估计**（B 与 V1.1 是两条训练轨迹），非严格因果；'
      '`N-O`/`I-O`/`I-N` 为同一 checkpoint 推理干预差，可直接因果判断。')
    P('')
    P(f'- baseline-dir: `{args.baseline_dir}`')
    P(f'- causal-dir: `{args.causal_dir}`')
    P(f'- diagnostic-json: `{args.diagnostic_json}`')
    if manifest:
        P(f'- identity manifest: checkpoint_source={manifest.get("checkpoint_source")} '
          f'training_performed={manifest.get("training_performed")} '
          f'sha_before={manifest.get("checkpoint_sha256_before")}')
    if gate_warnings:
        P('')
        P('**警告**：')
        for w in gate_warnings:
            P(f'- {w}')

    # auto 复现门
    v = verdicts['values']
    P('')
    P('## 0. auto 复现门')
    P(f"- N(auto) overall AP = {fmt(v['N_overall_AP'])}，目标 {v['auto_repro_AP_target']:.2f}，"
      f"|Δ| = {fmt(v['auto_repro_abs_delta'])}（门限 0.02）")
    P(f"- {'AUTO_REPRO_FAIL' if 'AUTO_REPRO_FAIL' in verdicts['labels'] else 'auto 复现通过'}")

    # 四态 AP
    P('')
    P('## 1. 四态 AP（B/N/O/I）')
    P('| 指标 | B baseline | N auto | O force_off | I force_on |')
    P('|---|---:|---:|---:|---:|')
    for m in KEYS:
        P(f"| overall {m} | {fmt(ov_metric(states['B'], m))} | {fmt(ov_metric(states['N'], m))} "
          f"| {fmt(ov_metric(states['O'], m))} | {fmt(ov_metric(states['I'], m))} |")
    for d in DOMS:
        P(f"| {d} AP | {fmt(dom_metric(states['B'], d, 'AP'))} | {fmt(dom_metric(states['N'], d, 'AP'))} "
          f"| {fmt(dom_metric(states['O'], d, 'AP'))} | {fmt(dom_metric(states['I'], d, 'AP'))} |")

    # overall 因果分解
    P('')
    P('## 2. overall 因果分解（O-B / N-O / I-O / I-N）')
    P('| 指标 | JointTrainingDrift(O-B) | NormalGate(N-O) | FullCapacity(I-O) | GateSuppression(I-N) |')
    P('|---|---:|---:|---:|---:|')
    for m in KEYS:
        dc = res['overall'][m]
        P(f"| {m} | {_d4(dc['JointTrainingDriftEstimate'])} | {_d4(dc['NormalGateContribution'])} "
          f"| {_d4(dc['FullSRFFCapacity'])} | {_d4(dc['GateSuppression'])} |")

    # 每域因果分解（AP）
    P('')
    P('## 3. 每域 AP 因果分解')
    P('| domain | JointTrainingDrift(O-B) | NormalGate(N-O) | FullCapacity(I-O) | GateSuppression(I-N) |')
    P('|---|---:|---:|---:|---:|')
    for d in DOMS:
        dc = res['per_domain'][d]['AP']
        P(f"| {d} | {_d4(dc['JointTrainingDriftEstimate'])} | {_d4(dc['NormalGateContribution'])} "
          f"| {_d4(dc['FullSRFFCapacity'])} | {_d4(dc['GateSuppression'])} |")

    # overall 每类别因果分解
    P('')
    P('## 4. overall 每类别因果分解')
    P('| category | JointTrainingDrift(O-B) | NormalGate(N-O) | FullCapacity(I-O) | GateSuppression(I-N) |')
    P('|---|---:|---:|---:|---:|')
    for c in res['categories']:
        dc = res['overall_per_category'][c]
        P(f"| {c} | {_d4(dc['JointTrainingDriftEstimate'])} | {_d4(dc['NormalGateContribution'])} "
          f"| {_d4(dc['FullSRFFCapacity'])} | {_d4(dc['GateSuppression'])} |")

    # clean scratch 平均
    P('')
    P('## 5. clean 三域 scratch 平均因果分解')
    cs = res['clean_scratch_avg']
    P(f"- 四态 scratch 均值：B={fmt(cs['states']['B'])} N={fmt(cs['states']['N'])} "
      f"O={fmt(cs['states']['O'])} I={fmt(cs['states']['I'])}")
    dc = cs['decomp']
    P(f"- JointTrainingDrift(O-B)={_d4(dc['JointTrainingDriftEstimate'])}  "
      f"NormalGate(N-O)={_d4(dc['NormalGateContribution'])}  "
      f"FullCapacity(I-O)={_d4(dc['FullSRFFCapacity'])}  "
      f"GateSuppression(I-N)={_d4(dc['GateSuppression'])}")

    # GDC/PDC
    P('')
    P('## 6. GDC / PDC 能力与门压制')
    P('| domain | FullCapacity(I-O) | GateSuppression(I-N) | NormalGate(N-O) |')
    P('|---|---:|---:|---:|')
    for d in ROBUST:
        g = res['gdc_pdc'][d]
        P(f"| {d} | {_d4(g['FullSRFFCapacity_I_minus_O'])} | {_d4(g['GateSuppression_I_minus_N'])} "
          f"| {_d4(g['NormalGateContribution_N_minus_O'])} |")

    # 判据
    P('')
    P('## 7. §6.4 确定性分流判据（连续值 + 命中标签）')
    P(f"- clean scratch O-B = {_d4(v['clean_scratch_O_minus_B'])}（<=-0.50 → SHARED_DRIFT）")
    P(f"- clean scratch N-O = {_d4(v['clean_scratch_N_minus_O'])}（<=-0.50 → CLEAN_FALSE_ACTIVATION）")
    P(f"- GDC I-N = {_d4(v['GDC_I_minus_N'])}（>=+0.30 → GDC_GATE_SUPPRESSION）")
    P(f"- GDC I-O = {_d4(v['GDC_I_minus_O'])}（<=+0.20 → GDC_EXPERT_WEAK）")
    P(f"- PDC N-O = {_d4(v['PDC_N_minus_O'])} 且 PDC I-N = {_d4(v['PDC_I_minus_N'])}"
      f"（N-O>0 且 I-N<=+0.20 → PDC_PATH_RETAINED）")
    P(f"- |N_AP - {v['auto_repro_AP_target']:.2f}| = {fmt(v['auto_repro_abs_delta'])}（>0.02 → AUTO_REPRO_FAIL）")
    P('')
    P(f"**命中标签**：{verdicts['labels'] if verdicts['labels'] else '（无）'}")

    # D 关门控统计
    P('')
    P('## 8. D 关（auto）门控统计')
    P(f"- metadata: version={diagx['metadata'].get('srff_version')} "
      f"mode={diagx['metadata'].get('srff_global_gate_mode')} "
      f"active_blocks={diagx['metadata'].get('srff_active_block_count')} "
      f"tau=[{diagx['metadata'].get('srff_global_threshold_low')}, "
      f"{diagx['metadata'].get('srff_global_threshold_high')}]")
    P('')
    P('| domain | global_score_mean | global_gate_mean | pre_global_gate_mean | gate_mean(final) |')
    P('|---|---:|---:|---:|---:|')
    for d in DOMS:
        pg = diagx['per_domain_gate'].get(d, {})
        P(f"| {d} | {fmt(pg.get('global_score'), 4)} | {fmt(pg.get('global_gate'), 4)} "
          f"| {fmt(pg.get('pre_global_gate'), 5)} | {fmt(pg.get('gate'), 5)} |")
    P('')
    P('D 关 comparisons（原值）：')
    P('```json')
    P(json.dumps(diagx['comparisons'], ensure_ascii=False, indent=2))
    P('```')
    P('')
    P('> 说明：本文件为工具确定性输出，不含人工结论；下一版方向由 §6.4/§9 分流规则决定。')
    return '\n'.join(L)


def build_json(args, states, res, verdicts, diagx, gate_warnings, manifest):
    def state_aps(s):
        return {'overall': {m: ov_metric(s, m) for m in KEYS},
                'domains': {d: {m: dom_metric(s, d, m) for m in DOM_METRICS} for d in DOMS},
                'overall_per_category': per_category(s['overall'] or {})}
    return {
        'provenance': {'baseline_dir': args.baseline_dir, 'causal_dir': args.causal_dir,
                       'diagnostic_json': args.diagnostic_json,
                       'auto_repro_ap_target': args.auto_repro_ap,
                       'identity_manifest': manifest},
        'warnings': gate_warnings,
        'state_aps': {k: state_aps(s) for k, s in states.items()},
        'decomposition': res,
        'verdicts': verdicts,
        'diagnostic': diagx,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.1 三态门控因果诊断汇总（B/N/O/I）')
    ap.add_argument('--baseline-dir', required=True, help='含 baseline(B) overall+五域 JSON 的目录')
    ap.add_argument('--causal-dir', required=True, help='含 N/O/I（auto/force_off/force_on）18 份 JSON 的目录')
    ap.add_argument('--diagnostic-json', required=True, help='auto 模式 D 关 srff_checkpoint_diagnostics.json')
    ap.add_argument('--out-md', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--identity-manifest', default=None, help='identity_manifest.json（可选，用于 N/O/I 身份门禁）')
    ap.add_argument('--auto-repro-ap', type=float, default=AUTO_REPRO_AP_DEFAULT,
                    help='原 V1.1 fast72 seed0 整体 AP（auto 复现门目标，默认 38.64）')
    ap.add_argument('--base-overall-tpl', default='baseline_fast72_seed0_val_metrics.json')
    ap.add_argument('--base-dom-tpl', default='baseline_fast72_{dom}_val.json')
    ap.add_argument('--mode-overall-tpl', default='srff_v1_1_{mode}_overall.json')
    ap.add_argument('--mode-dom-tpl', default='srff_v1_1_{mode}_{dom}.json')
    args = ap.parse_args()

    gate_errors = []
    gate_warnings = []

    # B（baseline）
    B = load_state(args.baseline_dir, args.base_overall_tpl, args.base_dom_tpl, 'B', gate_errors)
    # N/O/I（同一 causal-dir，按 mode 模板）
    N = load_state(args.causal_dir, args.mode_overall_tpl.replace('{mode}', 'auto'),
                   args.mode_dom_tpl.replace('{mode}', 'auto'), 'N', gate_errors)
    O = load_state(args.causal_dir, args.mode_overall_tpl.replace('{mode}', 'force_off'),
                   args.mode_dom_tpl.replace('{mode}', 'force_off'), 'O', gate_errors)
    I = load_state(args.causal_dir, args.mode_overall_tpl.replace('{mode}', 'force_on'),
                   args.mode_dom_tpl.replace('{mode}', 'force_on'), 'I', gate_errors)

    # D 关诊断 JSON
    diag = load(args.diagnostic_json)
    if not isinstance(diag, dict) or '__missing__' in diag or '__error__' in diag:
        gate_errors.append(f'D 关诊断 JSON 缺失或损坏: {args.diagnostic_json}')
        diag = {}
    else:
        check_diagnostic(diag, gate_errors)

    # identity manifest（可选）
    mf_path = args.identity_manifest
    if mf_path is None:
        cand = os.path.join(args.causal_dir, 'identity_manifest.json')
        mf_path = cand if os.path.isfile(cand) else None
    manifest = check_identity_manifest(mf_path, gate_errors, gate_warnings)

    if gate_errors:
        print('[CAUSAL-GATE-FAIL] 完整性门禁未通过，非零退出：')
        for e in gate_errors:
            print('  -', e)
        raise SystemExit(EXIT_GATE)

    states = {'B': B, 'N': N, 'O': O, 'I': I}
    res = build_results(B, N, O, I)
    verdicts = build_verdicts(B, N, O, I, args.auto_repro_ap)
    diagx = extract_diagnostic(diag)

    md = build_md(args, states, res, verdicts, diagx, gate_warnings, manifest)
    payload = build_json(args, states, res, verdicts, diagx, gate_warnings, manifest)

    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

    print(md)
    print(f'\n[saved] {args.out_md}')
    print(f'[saved] {args.out_json}')
    if 'AUTO_REPRO_FAIL' in verdicts['labels']:
        print('[warn] AUTO_REPRO_FAIL：auto 未复现原 V1.1 AP，禁止据此解释 Force-OFF/ON（见手册 §8）。')


if __name__ == '__main__':
    main()
