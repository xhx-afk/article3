"""汇总 SRFF v1 验收评估结果为一份可读报告（整体 + 五域 + 生死量 + 每类别）。

读取 ``tools/wood/eval_metrics.py`` 产出的 JSON：
  * 整体 val：baseline / srff 各一份；
  * 五域 val：ODC/LDC/DDC/GDC/PDC × baseline/srff 各一份
    （由 ``tools/wood/split_val_domains.py`` 拆分子集后逐域评估得到）。

按 SRFF-V1-User-Acceptance-Guide.md §5.6 计算
ΔODC/ΔLDC/ΔDDC/ΔGDC/ΔPDC、RobustGain/CleanLoss/RobustAdvantage，
并按 §5.7 给出**方向性**判读（fast72 非最终门槛）。

只读 JSON、只依赖标准库；AP/AR 在 JSON 中为 [0,1]，本报告统一换算为百分制 AP point。
文件名可用参数覆盖，缺失文件会在报告顶部列出。
"""

import argparse
import json
import os

DOMS = ["ODC", "LDC", "DDC", "GDC", "PDC"]
SCALE = 100.0
KEYS = ["AP", "AP50", "AP75", "AR100", "AP_small", "AP_medium", "AP_large"]


def load(path):
    if not os.path.isfile(path):
        return {"__missing__": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 - 汇总脚本需容忍任意损坏文件
        return {"__error__": f"{type(exc).__name__}: {exc}", "__path__": path}


def bbox(j):
    if not isinstance(j, dict):
        return {}
    v = j.get("coco_eval_bbox")
    return v if isinstance(v, dict) else {}


def gv(j, k):
    """取 coco_eval_bbox[k] 并换算成百分制；无效返回 None。"""
    try:
        return float(bbox(j).get(k)) * SCALE
    except (TypeError, ValueError):
        return None


def fmt(v, nd=2):
    return "NA" if v is None else f"{v:.{nd}f}"


def sfmt(v, nd=2):
    return "NA" if v is None else f"{v:+.{nd}f}"


def diff(a, b):
    return None if (a is None or b is None) else b - a


def per_category(j):
    out = {}
    items = j.get("per_category_bbox", []) if isinstance(j, dict) else []
    for it in items:
        if isinstance(it, dict):
            try:
                out[it.get("category_name")] = float(it.get("AP")) * SCALE
            except (TypeError, ValueError):
                out[it.get("category_name")] = None
    return out


def build_report(args):
    D = args.accept_dir
    lines = []
    P = lines.append

    base_all = load(os.path.join(D, args.base_overall))
    srff_all = load(os.path.join(D, args.srff_overall))
    dom = {}
    for dm in DOMS:
        bp = os.path.join(D, args.base_dom_tpl.format(dom=dm))
        sp = os.path.join(D, args.srff_dom_tpl.format(dom=dm))
        dom[dm] = (load(bp), load(sp))

    expected = [args.base_overall, args.srff_overall]
    for dm in DOMS:
        expected.append(args.base_dom_tpl.format(dom=dm))
        expected.append(args.srff_dom_tpl.format(dom=dm))
    missing = [f for f in expected if not os.path.isfile(os.path.join(D, f))]

    P("=" * 80)
    P("SRFF v1 · fast72 · seed0 · 验收汇总（整体 + 五域 + §5.6 生死量 + 每类别）")
    P("=" * 80)
    P(f"accept_dir : {D}")
    P(f"缺失文件   : {missing if missing else '无（12 份齐全）'}")

    # ---- 整体 val ----
    P("")
    P("---- 整体 val（全部图） ----")
    P(f"{'metric':10s}{'baseline':>10s}{'srff':>10s}{'Δ(srff-base)':>14s}")
    for k in KEYS:
        b, s = gv(base_all, k), gv(srff_all, k)
        P(f"{k:10s}{fmt(b):>10s}{fmt(s):>10s}{sfmt(diff(b, s)):>14s}")

    # ---- 五域 val ----
    P("")
    P("---- 五域 val（每域子集） ----")
    P(f"{'domain':7s}{'bAP':>7s}{'sAP':>7s}{'ΔAP':>7s}{'bAP50':>7s}{'sAP50':>7s}"
      f"{'bAP75':>7s}{'sAP75':>7s}{'ΔAP75':>7s}{'bAR100':>8s}{'sAR100':>8s}")
    deltas = {}
    for dm in DOMS:
        bj, sj = dom[dm]
        bAP, sAP = gv(bj, "AP"), gv(sj, "AP")
        b50, s50 = gv(bj, "AP50"), gv(sj, "AP50")
        b75, s75 = gv(bj, "AP75"), gv(sj, "AP75")
        bAR, sAR = gv(bj, "AR100"), gv(sj, "AR100")
        deltas[dm] = diff(bAP, sAP)
        P(f"{dm:7s}{fmt(bAP):>7s}{fmt(sAP):>7s}{sfmt(deltas[dm]):>7s}"
          f"{fmt(b50):>7s}{fmt(s50):>7s}{fmt(b75):>7s}{fmt(s75):>7s}"
          f"{sfmt(diff(b75, s75)):>7s}{fmt(bAR):>8s}{fmt(sAR):>8s}")

    # ---- §5.6 生死量 ----
    P("")
    P("---- §5.6 生死量（AP point） ----")
    if all(deltas.get(dm) is not None for dm in DOMS):
        robust_gain = (deltas["GDC"] + deltas["PDC"]) / 2.0
        clean_loss = max(0.0, -deltas["ODC"])
        robust_adv = robust_gain - clean_loss
        P(f"ΔODC={deltas['ODC']:+.2f}  ΔLDC={deltas['LDC']:+.2f}  ΔDDC={deltas['DDC']:+.2f}  "
          f"ΔGDC={deltas['GDC']:+.2f}  ΔPDC={deltas['PDC']:+.2f}")
        P(f"RobustGain={robust_gain:+.2f}   CleanLoss={clean_loss:.2f}   "
          f"RobustAdvantage={robust_adv:+.2f}")
        overall_d = diff(gv(base_all, "AP"), gv(srff_all, "AP"))
        P(f"整体 ΔAP={sfmt(overall_d)}")
        if (deltas["GDC"] <= 0 and deltas["PDC"] <= 0) or robust_adv <= 0 \
                or deltas["ODC"] < -0.80 or (overall_d is not None and overall_d < -0.50):
            verdict = "FAIL（方向性）"
        elif deltas["GDC"] > 0 and deltas["PDC"] > 0 and robust_gain >= 0.50 \
                and deltas["ODC"] >= -0.50 and robust_adv > 0 \
                and (overall_d is None or overall_d >= -0.20):
            verdict = "PASS（方向性）"
        else:
            verdict = "BORDERLINE（方向性）"
        P(f"§5.7 判读: {verdict}")
        P("注: fast72 非最终门槛，仅方向性参考；论文结论需 132ep 多 seed + source-level bootstrap。")
    else:
        P("[warn] 部分域缺 AP，无法计算生死量；请检查上面的缺失文件 / NA。")

    # ---- 整体每类别 AP ----
    P("")
    P("---- 整体每类别 AP（baseline vs srff） ----")
    bc, sc = per_category(base_all), per_category(srff_all)
    P(f"{'category':12s}{'baseline':>10s}{'srff':>10s}{'Δ':>9s}")
    for c in sorted(set(bc) | set(sc), key=lambda x: str(x)):
        b, s = bc.get(c), sc.get(c)
        P(f"{str(c):12s}{fmt(b):>10s}{fmt(s):>10s}{sfmt(diff(b, s)):>9s}")

    # ---- 各域每类别 ΔAP ----
    P("")
    P("---- 各域每类别 ΔAP（srff - baseline，定位收益/损失来自哪个缺陷类） ----")
    cats = sorted({c for dm in DOMS for c in per_category(dom[dm][0])}, key=lambda x: str(x))
    header = f"{'domain':8s}" + "".join(f"{str(c):>10s}" for c in cats)
    P(header)
    for dm in DOMS:
        bj, sj = dom[dm]
        bcat, scat = per_category(bj), per_category(sj)
        row = f"{dm:8s}"
        for c in cats:
            row += f"{sfmt(diff(bcat.get(c), scat.get(c))):>10s}"
        P(row)

    P("")
    P("=" * 80)
    P("汇总结束。完整诊断（阈值 P/R/F1、混淆矩阵、错误分解、hard images）见同目录原始 JSON。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="汇总 SRFF 验收评估（整体+五域+生死量+每类别）")
    ap.add_argument("--accept-dir", default=os.environ.get("SRFF_ACCEPT_DIR"),
                    help="验收目录（默认取环境变量 SRFF_ACCEPT_DIR）")
    ap.add_argument("--base-overall", default="13_baseline_fast72_seed0_val_metrics.json")
    ap.add_argument("--srff-overall", default="14_srff_fast72_seed0_val_metrics.json")
    ap.add_argument("--base-dom-tpl", default="16_baseline_fast72_{dom}_val.json")
    ap.add_argument("--srff-dom-tpl", default="17_srff_fast72_{dom}_val.json")
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
