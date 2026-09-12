"""汇总 SRFF-V1.1 验收评估结果为一份可读报告（整体 + 五域 + §5.8 生死量 + 每类别）。

复用 ``tools/wood/summarize_srff_eval.py`` 的解析逻辑（load/gv/per_category 等），
但**生死量定义按改进文档 §5.8 更新**（不再用 V1 的 -ΔODC 代表全部 clean loss）：

    CleanMean       = (ΔODC + ΔLDC + ΔDDC) / 3
    CleanLoss       = max(0, -CleanMean)
    RobustGain      = (ΔGDC + ΔPDC) / 2
    RobustAdvantage = RobustGain - CleanLoss
    WorstClean      = min(ΔODC, ΔLDC, ΔDDC)

方向性判读按验收手册 §7.1（通过，9 项全满足）/ §7.2（强通过）/ §7.3（失败或边界）给出，
并打印 §7.1 逐项核对表（含 Overall crack/scratch ΔAP >= -0.50 两项，与手册一致）。

只读 JSON、只依赖标准库；AP/AR 在 JSON 中为 [0,1]，本报告统一换算为百分制 AP point。
文件名可用参数覆盖，缺失文件会在报告顶部列出。

用法::
    python tools/wood/summarize_srff_v11_eval.py --accept-dir <dir> \
        --base-overall baseline_fast72_seed0_val_metrics.json \
        --v11-overall srff_v1_1_fast72_seed0_val_metrics.json \
        --base-dom-tpl 'baseline_fast72_{dom}_val.json' \
        --v11-dom-tpl 'srff_v1_1_fast72_{dom}_val.json'
"""

import argparse
import os
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

# 复用 V1 汇总脚本的解析逻辑（避免重复实现）
from summarize_srff_eval import (  # noqa: E402
    DOMS, KEYS, load, gv, fmt, sfmt, diff, per_category,
)

CLEAN = ["ODC", "LDC", "DDC"]
ROBUST = ["GDC", "PDC"]


def build_report(args):
    D = args.accept_dir
    lines = []
    P = lines.append

    base_all = load(os.path.join(D, args.base_overall))
    v11_all = load(os.path.join(D, args.v11_overall))
    dom = {}
    for dm in DOMS:
        bp = os.path.join(D, args.base_dom_tpl.format(dom=dm))
        vp = os.path.join(D, args.v11_dom_tpl.format(dom=dm))
        dom[dm] = (load(bp), load(vp))

    expected = [args.base_overall, args.v11_overall]
    for dm in DOMS:
        expected.append(args.base_dom_tpl.format(dom=dm))
        expected.append(args.v11_dom_tpl.format(dom=dm))
    missing = [f for f in expected if not os.path.isfile(os.path.join(D, f))]

    P("=" * 80)
    P("SRFF-V1.1 · fast72 · seed0 · 验收汇总（整体 + 五域 + §5.8 生死量 + 每类别）")
    P("=" * 80)
    P(f"accept_dir : {D}")
    P(f"缺失文件   : {missing if missing else '无（12 份齐全）'}")

    # ---- 整体 val ----
    P("")
    P("---- 整体 val（全部图） ----")
    P(f"{'metric':10s}{'baseline':>10s}{'v1.1':>10s}{'Δ(v1.1-base)':>14s}")
    for k in KEYS:
        b, s = gv(base_all, k), gv(v11_all, k)
        P(f"{k:10s}{fmt(b):>10s}{fmt(s):>10s}{sfmt(diff(b, s)):>14s}")

    # ---- 五域 val ----
    P("")
    P("---- 五域 val（每域子集） ----")
    P(f"{'domain':7s}{'bAP':>7s}{'vAP':>7s}{'ΔAP':>7s}{'bAP50':>7s}{'vAP50':>7s}"
      f"{'bAP75':>7s}{'vAP75':>7s}{'ΔAP75':>7s}{'bAR100':>8s}{'vAR100':>8s}")
    deltas = {}
    for dm in DOMS:
        bj, vj = dom[dm]
        bAP, vAP = gv(bj, "AP"), gv(vj, "AP")
        b50, v50 = gv(bj, "AP50"), gv(vj, "AP50")
        b75, v75 = gv(bj, "AP75"), gv(vj, "AP75")
        bAR, vAR = gv(bj, "AR100"), gv(vj, "AR100")
        deltas[dm] = diff(bAP, vAP)
        P(f"{dm:7s}{fmt(bAP):>7s}{fmt(vAP):>7s}{sfmt(deltas[dm]):>7s}"
          f"{fmt(b50):>7s}{fmt(v50):>7s}{fmt(b75):>7s}{fmt(v75):>7s}"
          f"{sfmt(diff(b75, v75)):>7s}{fmt(bAR):>8s}{fmt(vAR):>8s}")

    # ---- §5.8 生死量（V1.1 定义）----
    P("")
    P("---- §5.8 生死量（AP point，V1.1 定义：CleanMean/WorstClean） ----")
    if all(deltas.get(dm) is not None for dm in DOMS):
        clean_mean = sum(deltas[d] for d in CLEAN) / 3.0
        clean_loss = max(0.0, -clean_mean)
        robust_gain = sum(deltas[d] for d in ROBUST) / 2.0
        robust_adv = robust_gain - clean_loss
        worst_clean = min(deltas[d] for d in CLEAN)
        P(f"ΔODC={deltas['ODC']:+.2f}  ΔLDC={deltas['LDC']:+.2f}  ΔDDC={deltas['DDC']:+.2f}  "
          f"ΔGDC={deltas['GDC']:+.2f}  ΔPDC={deltas['PDC']:+.2f}")
        P(f"CleanMean={clean_mean:+.2f}   WorstClean={worst_clean:+.2f}")
        P(f"RobustGain={robust_gain:+.2f}   CleanLoss={clean_loss:.2f}   "
          f"RobustAdvantage={robust_adv:+.2f}")
        overall_d = diff(gv(base_all, "AP"), gv(v11_all, "AP"))
        P(f"整体 ΔAP={sfmt(overall_d)}")

        # 整体 crack/scratch ΔAP（§7.1/§7.3 需要）
        bc0, vc0 = per_category(base_all), per_category(v11_all)
        crack_d = diff(bc0.get("crack"), vc0.get("crack"))
        scratch_d = diff(bc0.get("scratch"), vc0.get("scratch"))
        P(f"整体 crack ΔAP={sfmt(crack_d)}   整体 scratch ΔAP={sfmt(scratch_d)}")

        # §7.1「通过」：9 项须全部满足
        crit71 = [
            ("ΔGDC > 0", deltas["GDC"] > 0, f"{deltas['GDC']:+.2f}"),
            ("ΔPDC > 0", deltas["PDC"] > 0, f"{deltas['PDC']:+.2f}"),
            ("RobustGain >= +0.50", robust_gain >= 0.50, f"{robust_gain:+.2f}"),
            ("CleanMean >= -0.30", clean_mean >= -0.30, f"{clean_mean:+.2f}"),
            ("WorstClean >= -0.50", worst_clean >= -0.50, f"{worst_clean:+.2f}"),
            ("RobustAdvantage > 0", robust_adv > 0, f"{robust_adv:+.2f}"),
            ("Overall ΔAP >= 0", overall_d is not None and overall_d >= 0, sfmt(overall_d)),
            ("Overall crack ΔAP >= -0.50", crack_d is not None and crack_d >= -0.50, sfmt(crack_d)),
            ("Overall scratch ΔAP >= -0.50", scratch_d is not None and scratch_d >= -0.50, sfmt(scratch_d)),
        ]
        P("")
        P("---- §7.1「通过」逐项核对（须全部 ✓）----")
        for _name, _met, _val in crit71:
            P(f"  [{'✓' if _met else '✗'}] {_name:30s} 值={_val}")
        pass_71 = all(_met for _, _met, _ in crit71)

        # §7.2「强通过」：在 §7.1 全过基础上再收紧
        strong_72 = pass_71 and robust_gain >= 0.70 and clean_mean >= -0.15 \
            and (overall_d is not None and overall_d >= 0.25)

        # §7.3「失败或边界」：任一触发（NaN/Inf、数据不齐、checkpoint 身份不清由评估日志/缺失清单把关）
        fail_73 = (deltas["GDC"] <= 0 or deltas["PDC"] <= 0) or clean_mean < -0.30 \
            or worst_clean < -0.50 or (overall_d is not None and overall_d < 0) \
            or robust_adv < 0 or (crack_d is not None and crack_d < -0.50) \
            or (scratch_d is not None and scratch_d < -0.50)

        if strong_72:
            verdict = "STRONG PASS（方向性，满足 §7.2 强通过）"
        elif pass_71:
            verdict = "PASS（方向性，满足 §7.1 全部 9 项）"
        elif fail_73:
            verdict = "FAIL（方向性，触发 §7.3 失败或边界）"
        else:
            verdict = "BORDERLINE（方向性）"
        P("")
        P(f"方向性判读: {verdict}")
        P("注: fast72 非最终门槛，仅方向性参考；论文结论需 132ep 多 seed + source-level bootstrap。")
        P("注: 训练后 checkpoint 的机制目标（clean/robust gate retention、robust/clean final gate ratio、")
        P("    active block=1）由 tools/wood/inspect_srff_checkpoint.py 测得，不在本 AP 汇总内伪造。")
    else:
        P("[warn] 部分域缺 AP，无法计算生死量；请检查上面的缺失文件 / NA。")

    # ---- 整体每类别 AP ----
    P("")
    P("---- 整体每类别 AP（baseline vs v1.1） ----")
    bc, vc = per_category(base_all), per_category(v11_all)
    P(f"{'category':12s}{'baseline':>10s}{'v1.1':>10s}{'Δ':>9s}")
    for c in sorted(set(bc) | set(vc), key=lambda x: str(x)):
        b, s = bc.get(c), vc.get(c)
        P(f"{str(c):12s}{fmt(b):>10s}{fmt(s):>10s}{sfmt(diff(b, s)):>9s}")

    # ---- 各域每类别 ΔAP ----
    P("")
    P("---- 各域每类别 ΔAP（v1.1 - baseline，定位收益/损失来自哪个缺陷类） ----")
    cats = sorted({c for dm in DOMS for c in per_category(dom[dm][0])}, key=lambda x: str(x))
    header = f"{'domain':8s}" + "".join(f"{str(c):>10s}" for c in cats)
    P(header)
    for dm in DOMS:
        bj, vj = dom[dm]
        bcat, vcat = per_category(bj), per_category(vj)
        row = f"{dm:8s}"
        for c in cats:
            row += f"{sfmt(diff(bcat.get(c), vcat.get(c))):>10s}"
        P(row)

    P("")
    P("=" * 80)
    P("汇总结束。完整诊断（阈值 P/R/F1、混淆矩阵、错误分解、hard images）见同目录原始 JSON。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="汇总 SRFF-V1.1 验收评估（整体+五域+§5.8生死量+每类别）")
    ap.add_argument("--accept-dir", default=os.environ.get("SRFF_ACCEPT_DIR"),
                    help="验收目录（默认取环境变量 SRFF_ACCEPT_DIR）")
    ap.add_argument("--base-overall", default="baseline_fast72_seed0_val_metrics.json")
    ap.add_argument("--v11-overall", "--srff-overall", dest="v11_overall",
                    default="srff_v1_1_fast72_seed0_val_metrics.json")
    ap.add_argument("--base-dom-tpl", default="baseline_fast72_{dom}_val.json")
    ap.add_argument("--v11-dom-tpl", "--srff-dom-tpl", dest="v11_dom_tpl",
                    default="srff_v1_1_fast72_{dom}_val.json")
    ap.add_argument("--out", default=None, help="另存汇总到该路径（默认仅打印到 stdout）")
    args = ap.parse_args()

    if not args.accept_dir:
        raise SystemExit("必须指定 --accept-dir 或设置环境变量 SRFF_ACCEPT_DIR")

    report = build_report(args)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
