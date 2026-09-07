#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比度 × 尺寸 混杂检查（创新点 3 的命根子，10 分钟）

为什么要查：创新点 3 的【全部动机】是"对比度预测检出难度"，交接文档 §5.4 特意强调
"前者是假设，后者是数据（43.7% vs 83.0%，n=617）"。但同管线重算下这个单调性没复现：
    PDC 分层存活率 9.6 / 33.0 / 23.6  —— 中间最高，高对比反而回落。
一个自然的怀疑是【对比度与目标尺寸混杂】：§3.1 有 rho(sqrt(area), IoU) = 0.639，
小目标定位本来就差；而小的深色孔洞局部对比度天然高。若高对比层里小目标偏多，
存活率就会回落 —— 那么"对比度决定检出难度"这条因果主张就不成立，
创新点 3 必须重新设计（或改成"尺寸自适应"）。

本脚本读 degradation_paired.py 产出的 paired_instances.csv，回答一个问题：
    【控制住尺寸之后，对比度还预测检出吗？】

用法:
  python3 tools/analysis/contrast_size_confound.py \
      --csv prep/innov1/paired_instances.csv --codes PDC GDC LDC DDC \
      --out prep/innov1/contrast_size_confound.json
  python3 tools/analysis/contrast_size_confound.py --self-test-only
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np


def rankdata(x):
    """平均秩（处理并列）。"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    sx = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    out = np.empty(len(x), dtype=np.float64)
    out[order] = ranks
    return out


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return 0.0 if d <= 0 else float((a * b).sum() / d)


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def partial_spearman(a, b, c):
    """控制 c 之后 a 与 b 的偏相关（秩残差法）。"""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    def resid(y, x):
        x1 = np.stack([np.ones_like(x), x], 1)
        beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
        return y - x1 @ beta
    return pearson(resid(ra, rc), resid(rb, rc))


def tertile(v):
    """先按 first 秩再切三分位 —— 对比度大量并列，直接 qcut 会塌箱（§7.3 要点 3）。"""
    v = np.asarray(v, float)
    order = np.argsort(v, kind='mergesort')
    r = np.empty(len(v), dtype=np.int64)
    r[order] = np.arange(len(v))
    n = len(v)
    return np.where(r < n / 3, 0, np.where(r < 2 * n / 3, 1, 2))


def stratified_effect(det, ct, st, n_strata=3):
    """在每个尺寸层内部算"高对比 - 低对比"的存活率差，再按层大小加权平均。
    这是 Cochran-Mantel-Haenszel 式的、控制住尺寸的对比度效应。"""
    num = den = 0.0
    per = {}
    for s in range(n_strata):
        m = st == s
        if m.sum() < 6:
            continue
        lo, hi = det[m & (ct == 0)], det[m & (ct == 2)]
        if len(lo) == 0 or len(hi) == 0:
            continue
        d = float(hi.mean() - lo.mean())
        per[f'size{s}'] = dict(n_low=int(len(lo)), n_high=int(len(hi)),
                               surv_low=float(lo.mean()), surv_high=float(hi.mean()), diff=d)
        num += d * m.sum(); den += m.sum()
    return (num / den if den else float('nan')), per


def permutation_p(det, ct, st, n_perm=2000, seed=0):
    """零假设：尺寸层内部对比度与检出无关。层内打乱对比度分层标签。"""
    rng = np.random.default_rng(seed)
    obs, _ = stratified_effect(det, ct, st)
    if not np.isfinite(obs):
        return float('nan'), obs
    cnt = 0
    ct_p = ct.copy()
    idx_by_stratum = [np.where(st == s)[0] for s in range(3)]
    for _ in range(n_perm):
        for idx in idx_by_stratum:
            ct_p[idx] = ct[rng.permutation(idx)]
        e, _ = stratified_effect(det, ct_p, st)
        if np.isfinite(e) and abs(e) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n_perm + 1), obs


def analyze(rows, codes, n_perm, seed):
    contrast = np.array([float(r['contrast']) for r in rows])
    area = np.array([float(r['area']) for r in rows])
    size = np.sqrt(np.maximum(area, 0.0))
    cls = np.array([r.get('cls', '?') for r in rows])
    ct, st = tertile(contrast), tertile(size)

    out = {'n': len(rows)}
    out['spearman_contrast_vs_size'] = spearman(contrast, size)
    print(f"实例数 {len(rows)}")
    print(f"\nSpearman(对比度, sqrt(area)) = {out['spearman_contrast_vs_size']:+.3f}"
          f"  ——【{'两者显著混杂，下面的控制分析是必须的' if abs(out['spearman_contrast_vs_size'])>=0.2 else '混杂弱，对比度效应大体可以直读'}】")

    print(f"\n对比度三分位内部的尺寸构成（若高对比层里小目标扎堆，存活率回落就有平凡解释）")
    print(f"  {'层':6s} {'n':>5s} {'对比度中位':>10s} {'sqrt(area)中位':>13s} {'主要类别':>20s}")
    comp = {}
    for t, name in enumerate(('低对比', '中对比', '高对比')):
        m = ct == t
        cc = defaultdict(int)
        for c in cls[m]:
            cc[c] += 1
        top = ', '.join(f'{k} {v}' for k, v in sorted(cc.items(), key=lambda x: -x[1])[:2])
        comp[name] = dict(n=int(m.sum()), contrast_med=float(np.median(contrast[m])),
                          size_med=float(np.median(size[m])), classes=dict(cc))
        print(f"  {name:6s} {m.sum():5d} {np.median(contrast[m]):10.3f} "
              f"{np.median(size[m]):13.1f} {top:>20s}")
    out['tertile_composition'] = comp

    out['codes'] = {}
    for code in codes:
        key = f'det_{code}'
        if key not in rows[0]:
            print(f"\n[skip] csv 里没有 {key}")
            continue
        det = np.array([1.0 if str(r[key]).strip().lower() in ('1', 'true', 'yes') else 0.0
                        for r in rows])
        base = np.array([1.0 if str(r.get('det_ODC', 1)).strip().lower() in ('1', 'true', 'yes')
                         else 0.0 for r in rows])
        keep = base > 0                    # 只看 ODC 下的确信检出（与 §3.3 口径一致）
        if keep.sum() < 30:
            print(f"\n[skip] {code}: ODC 确信检出仅 {int(keep.sum())} 个，样本不足")
            continue
        d, c_, s_ = det[keep], ct[keep], st[keep]
        cn, sz = contrast[keep], size[keep]

        r_c = spearman(cn, d); r_s = spearman(sz, d)
        pr_c = partial_spearman(cn, d, sz); pr_s = partial_spearman(sz, d, cn)
        eff, per = stratified_effect(d, c_, s_)
        p, _ = permutation_p(d, c_, s_, n_perm, seed)

        print(f"\n=== {code}（ODC 确信检出 n={int(keep.sum())}，存活 {d.mean()*100:.1f}%）===")
        print(f"  边际  Spearman(对比度, 检出) = {r_c:+.3f}   Spearman(尺寸, 检出) = {r_s:+.3f}")
        print(f"  偏相关 对比度|控制尺寸      = {pr_c:+.3f}   尺寸|控制对比度      = {pr_s:+.3f}")
        print(f"  尺寸层内的对比度效应(高-低) = {eff:+.3f}   置换检验 p = {p:.4f}")
        print(f"  {'尺寸层':8s} {'n低/n高':>10s} {'存活低':>8s} {'存活高':>8s} {'差':>8s}")
        for k, v in per.items():
            print(f"  {k:8s} {str(v['n_low'])+'/'+str(v['n_high']):>10s} "
                  f"{v['surv_low']*100:7.1f}% {v['surv_high']*100:7.1f}% {v['diff']*100:+7.1f}pt")

        if p < 0.05 and abs(pr_c) > abs(pr_s):
            verd = 'CONTRAST_SURVIVES：控制尺寸后对比度仍显著且主导 → 创新点 3 的前提站得住'
        elif p < 0.05:
            verd = 'BOTH：对比度效应显著，但尺寸的偏相关更强 → 创新点 3 应改为对比度+尺寸联合条件化'
        elif abs(pr_s) > abs(pr_c):
            verd = 'SIZE_DOMINATES：控制尺寸后对比度不显著 → 【创新点 3 的动机不成立】，改做尺寸自适应'
        else:
            verd = 'NEITHER：两者都不显著 → 检出难度由别的因素决定，创新点 3 需重新找依据'
        print(f"  >>> {verd}")
        out['codes'][code] = dict(n=int(keep.sum()), survival=float(d.mean()),
                                  spearman_contrast=r_c, spearman_size=r_s,
                                  partial_contrast=pr_c, partial_size=pr_s,
                                  stratified_effect=eff, perm_p=p,
                                  per_size_stratum=per, verdict=verd)
    return out


def self_test():
    rng = np.random.default_rng(0); fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    chk('spearman 单调变换不变', abs(spearman(np.arange(50), np.exp(np.arange(50) / 10)) - 1) < 1e-9)
    chk('rankdata 处理并列', np.allclose(rankdata([1, 1, 2]), [1.5, 1.5, 3]))
    chk('tertile 三箱近似等大（全并列输入也不塌）',
        max(np.bincount(tertile(np.zeros(300)))) - min(np.bincount(tertile(np.zeros(300)))) <= 1)

    n = 900
    # 场景 A：对比度真因果，尺寸独立
    cn = rng.random(n); sz = rng.random(n)
    det = (rng.random(n) < np.clip(0.15 + 0.7 * cn, 0, 1)).astype(float)
    ct, st = tertile(cn), tertile(sz)
    effA, _ = stratified_effect(det, ct, st); pA, _ = permutation_p(det, ct, st, 500, 0)
    prA = partial_spearman(cn, det, sz)
    chk('场景A 对比度真因果：层内效应为正且 p<0.05', effA > 0.2 and pA < 0.05,
        f'eff {effA:+.3f}, p {pA:.4f}, 偏相关 {prA:+.3f}')

    # 场景 B：尺寸真因果，对比度只是尺寸的噪声代理（混杂陷阱）
    sz2 = rng.random(n)
    cn2 = np.clip(sz2 + rng.normal(0, 0.25, n), 0, 1)
    det2 = (rng.random(n) < np.clip(0.15 + 0.7 * sz2, 0, 1)).astype(float)
    ct2, st2 = tertile(cn2), tertile(sz2)
    effB, _ = stratified_effect(det2, ct2, st2); pB, _ = permutation_p(det2, ct2, st2, 500, 0)
    prB_c = partial_spearman(cn2, det2, sz2); prB_s = partial_spearman(sz2, det2, cn2)
    chk('场景B 尺寸真因果：边际相关仍为正（陷阱确实存在）', spearman(cn2, det2) > 0.15,
        f'边际 {spearman(cn2, det2):+.3f}')
    chk('场景B 控制尺寸后对比度被解释掉', abs(prB_c) < abs(prB_s) and effB < effA,
        f'偏相关 对比度 {prB_c:+.3f} < 尺寸 {prB_s:+.3f}; 层内效应 {effB:+.3f} < {effA:+.3f}')
    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv'); ap.add_argument('--out')
    ap.add_argument('--codes', nargs='+', default=['PDC', 'GDC', 'LDC', 'DDC'])
    ap.add_argument('--n-perm', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if not a.csv: ap.error('--csv 必需')
    rows = [r for r in csv.DictReader(open(a.csv, encoding='utf-8'))
            if r.get('contrast') not in (None, '', 'nan')]
    if not rows: sys.exit('csv 里没有可用行（contrast 全为空？）')
    out = analyze(rows, a.codes, a.n_perm, a.seed)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"\n写出 {a.out}")


if __name__ == '__main__':
    main()
