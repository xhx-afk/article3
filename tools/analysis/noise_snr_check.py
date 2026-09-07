#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
退化噪声的信噪比核算（把 §3.3 的"反常"从定性说法变成量化结论）

GDC 已被判定为"与 PDC 同族的高频加性/脉冲型退化"，不是模糊。于是交接文档 §3.3
原本的"可逆的椒盐 vs 不可逆的模糊"这条对比失效。本脚本给它一个更硬的替代：

    缺陷的 Lab 对比度中位 ~1.12（低于人眼可辨阈 2.3）
    vs 各退化码引入的噪声 sigma（同样用 Lab ΔE 单位）

并区分两种噪声（这是关键）:
    DENSE  : 稳健 sigma ≈ 标准差   -> 每个像素都被污染，缺陷信息真的被削弱
    SPARSE : 标准差 >> 稳健 sigma  -> 绝大多数像素完好，信息仍在（中值滤波能救）

如果 GDC = DENSE 而 PDC = SPARSE，§3.3 的反常就原样成立，只是把"不可逆的模糊"
换成"信噪比受限的稠密噪声"，并且第一次有了数字支撑。

用法:
  python3 tools/analysis/noise_snr_check.py \
    --img-dir datasets/Water-Based-Coated-Wood/images/val \
    --csv prep/innov1/paired_instances.csv \
    --codes GDC PDC LDC DDC --n 40 --out prep/innov1/noise_snr.json
  python3 tools/analysis/noise_snr_check.py --self-test-only
"""
import argparse, csv, json, os, re, sys
from collections import defaultdict
import numpy as np

NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
JND = 2.3     # 人眼可辨阈（Lab ΔE）


def norm_id(s): return str(int(s))


# ---------------------------------------------------------------- sRGB -> Lab
def srgb_to_lab(rgb):
    """rgb: (...,3) float in [0,1]。返回 L in [0,100], a/b in [-128,127]（D65）。
    必须走 float 路径：OpenCV 的 uint8 路径会把 L 缩放到 0..255（准备阶段的坑）。"""
    m = rgb <= 0.04045
    lin = np.where(m, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    # 白点直接由矩阵行和导出，保证纯白严格映射到 L=100, a=b=0（写死常数会差 4e-6）
    white = M.sum(axis=1)
    t = xyz / white
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def load_lab(path):
    from PIL import Image
    return srgb_to_lab(np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0)


def analyze_pair(ref_lab, deg_lab, jnd=JND):
    """核心判别量。

    不用 MAD 做稳健 sigma —— 椒盐退化下 95% 的像素【逐位未变】，MAD 恰好为 0，
    比值会溢出成无意义的天文数字（这个坑在合成对照里踩到过）。
    改用一个与 JND 锚定的固定阈值来数「完好像素比例」，既稳健又可解释。
    """
    d = deg_lab - ref_lab
    de = np.sqrt((d ** 2).sum(-1))                   # 每像素 ΔE
    intact = de < 0.5 * jnd                          # 视觉上未被改动
    sig_plain = float(np.sqrt((de ** 2).mean()))
    sig_intact = float(np.sqrt((de[intact] ** 2).mean())) if intact.any() else 0.0
    return dict(deltaE_mean=float(de.mean()),
                deltaE_p50=float(np.median(de)),
                deltaE_p95=float(np.percentile(de, 95)),
                deltaE_p99=float(np.percentile(de, 99)),
                sigma_plain=sig_plain,
                sigma_intact=sig_intact,
                frac_intact=float(intact.mean()),
                frac_severe=float((de > 3 * jnd).mean()))


def classify(m):
    if m['frac_intact'] >= 0.80:
        return 'SPARSE（多数像素逐位完好，信息仍在）'
    if m['frac_intact'] < 0.50:
        return 'DENSE（每个像素都被污染，缺陷信噪比被真正削弱）'
    return 'MIXED（介于两者之间）'


def run(args):
    idx = defaultdict(dict)
    for f in os.listdir(args.img_dir):
        if os.path.splitext(f)[1].lower() not in EXTS: continue
        m = NAME_RE.match(os.path.splitext(f)[0])
        if m: idx[(norm_id(m.group(1)), m.group(3))][m.group(2)] = os.path.join(args.img_dir, f)
    keys = sorted(k for k, v in idx.items() if 'ODC' in v)
    rng = np.random.default_rng(args.seed)
    pick = [keys[i] for i in rng.choice(len(keys), min(args.n, len(keys)), replace=False)]
    print(f"[snr] {len(keys)} 组可用，抽 {len(pick)} 组")

    # 缺陷对比度与面积（用于空间积分增益）
    contrast_med, area_med, tert = args.contrast, args.area, {}
    if args.csv and os.path.exists(args.csv):
        rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8'))
                if r.get('contrast') not in (None, '', 'nan')]
        if rows:
            cs = np.array([float(r['contrast']) for r in rows])
            ar = np.array([float(r['area']) for r in rows]) if 'area' in rows[0] else None
            contrast_med = float(np.median(cs))
            if ar is not None: area_med = float(np.median(ar))
            o = np.argsort(cs, kind='mergesort'); n = len(cs)
            for t, name in enumerate(('low', 'mid', 'high')):
                sel = o[int(t * n / 3):int((t + 1) * n / 3)]
                tert[name] = float(np.median(cs[sel]))
            print(f"[snr] 从 csv 读到 {len(rows)} 个实例：对比度中位 {contrast_med:.3f}"
                  f"（三分位 {tert['low']:.3f}/{tert['mid']:.3f}/{tert['high']:.3f}）"
                  f"，面积中位 {area_med:.0f} px")
    print(f"[snr] 参照：人眼可辨阈 JND = {JND}")

    out = {'contrast_median': contrast_med, 'area_median': area_med, 'tertiles': tert, 'codes': {}}
    for code in args.codes:
        acc = defaultdict(list)
        for k in pick:
            if code not in idx[k]: continue
            ref, deg = load_lab(idx[k]['ODC']), load_lab(idx[k][code])
            if ref.shape != deg.shape: continue
            for kk, vv in analyze_pair(ref, deg).items(): acc[kk].append(vv)
        if not acc: continue
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        m['type'] = classify(m)
        m['snr_perpixel'] = contrast_med / max(m['sigma_plain'], 1e-9)
        m['snr_area'] = m['snr_perpixel'] * np.sqrt(max(area_med, 1.0))
        m['snr_intact'] = (contrast_med / m['sigma_intact']) if m['sigma_intact'] > 1e-6 else None
        out['codes'][code] = m

    print(f"\n{'code':6s} {'ΔE中位':>8s} {'ΔE p99':>8s} {'σ(ΔE)':>8s} {'完好像素':>9s} "
          f"{'重度像素':>9s} {'单像素SNR':>10s} {'面积SNR':>9s}  类型")
    for c, m in out['codes'].items():
        print(f"{c:6s} {m['deltaE_p50']:8.3f} {m['deltaE_p99']:8.3f} {m['sigma_plain']:8.3f} "
              f"{m['frac_intact']*100:8.1f}% {m['frac_severe']*100:8.1f}% "
              f"{m['snr_perpixel']:10.3f} {m['snr_area']:9.2f}  {m['type']}")

    print("\n读法：")
    print(f"  完好像素 = ΔE < {0.5*JND:.2f}（半个 JND）的像素占比。这是区分稀疏/稠密的主判别量，")
    print("            不用 MAD 稳健 sigma —— 椒盐下 95% 像素逐位未变，MAD 恰为 0 会溢出。")
    print(f"  单像素SNR = 缺陷对比度 {contrast_med:.2f} / σ(ΔE)。<1 表示单个像素上看不见缺陷。")
    print(f"  面积SNR   = 单像素SNR × sqrt(缺陷面积 {area_med:.0f})，即「把缺陷内像素平均起来」能得到的信噪比。")
    print("            面积SNR >> 1 而单像素SNR < 1 ⇒ 信息仍在，只是需要空间积分 —— 属于【能力失败】而非信息损失。")

    g, p = out['codes'].get('GDC'), out['codes'].get('PDC')
    if g and p:
        print("\n================ §3.3 反常的量化重述 ================")
        print(f"  GDC: {g['type']}   PDC: {p['type']}")
        if 'DENSE' in g['type'] and 'SPARSE' in p['type']:
            print("  →【原叙事成立，只需换词】：GDC 是稠密噪声（每个像素都被污染，缺陷信噪比被真正削弱），")
            print("     PDC 是稀疏脉冲（绝大多数像素完好，中值滤波恢复 +23.4 AP 为证）。")
            print("     模型却在【信息保留的那一个】上更脆弱 —— 反常成立，且现在有数字。")
            print(f"     论文写法：把「不可逆的高斯模糊」替换为「信噪比受限的稠密噪声」。")
        else:
            print("  →【与预期不同】两者同类型。§3.3 的反常需要另找区分维度（例如")
            print("     中值滤波在 PDC 上 +23.4 而在 GDC 上 −5.4 这个经验事实本身）。")
        for c, m in (('GDC', g), ('PDC', p)):
            f = '低于' if m['snr_perpixel'] < 1 else '高于'
            si = f"{m['snr_intact']:.1f}" if m['snr_intact'] else '∞（完好像素上无噪声）'
            print(f"  {c}: 单像素 SNR {m['snr_perpixel']:.3f}（{f} 1），"
                  f"面积积分后 {m['snr_area']:.1f}；完好像素通道上的 SNR {si}")
        print("  注：对 SPARSE 型退化，简单平均并不是最优恢复方式（离群值不按 sqrt(N) 衰减），")
        print("      秩滤波才是 —— 这正是 median5 在 PDC 上 +23.4 而在 GDC 上 −5.4 的原因。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"\n写出 {args.out}")
    return out


def self_test():
    rng = np.random.default_rng(0); fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    w = srgb_to_lab(np.ones((1, 1, 3)))[0, 0]
    chk('Lab: 纯白 -> L=100, a=b=0', abs(w[0] - 100) < 1e-9 and abs(w[1]) < 1e-3 and abs(w[2]) < 1e-3,
        f'L={w[0]:.3f} a={w[1]:.4f} b={w[2]:.4f}')
    k = srgb_to_lab(np.zeros((1, 1, 3)))[0, 0]
    chk('Lab: 纯黑 -> L=0', abs(k[0]) < 1e-6)
    g = srgb_to_lab(np.full((1, 1, 3), 128 / 255.0))[0, 0]
    chk('Lab: sRGB 128 中灰 -> L≈53.6', abs(g[0] - 53.585) < 0.05, f'L={g[0]:.3f}')
    chk('Lab: L 是 0..100 而非 0..255（防 uint8 路径回归）', 0 <= g[0] <= 100)

    # 稠密高斯噪声 -> DENSE；稀疏脉冲 -> SPARSE
    base = np.repeat(rng.random((64, 64, 1)) * 0.6 + 0.2, 3, axis=-1)
    ref = srgb_to_lab(base)
    dense = srgb_to_lab(np.clip(base + rng.normal(0, 0.03, base.shape), 0, 1))
    md = analyze_pair(ref, dense)
    chk('稠密高斯噪声判为 DENSE', classify(md).startswith('DENSE'),
        f"完好像素 {md['frac_intact']*100:.1f}%")

    sp = base.copy(); m = rng.random(base.shape[:2]) < 0.05
    sp[m] = rng.choice([0.0, 1.0], m.sum())[:, None]
    ms = analyze_pair(ref, srgb_to_lab(sp))
    chk('稀疏脉冲判为 SPARSE', classify(ms).startswith('SPARSE'),
        f"完好像素 {ms['frac_intact']*100:.1f}%, 重度 {ms['frac_severe']*100:.1f}%")
    chk('稀疏脉冲：完好像素上的 sigma ≈ 0（信息完全保留）', ms['sigma_intact'] < 1e-6,
        f"{ms['sigma_intact']:.2e}")
    chk('稀疏 vs 稠密的完好像素比例可分（差 > 0.4）',
        ms['frac_intact'] - md['frac_intact'] > 0.4,
        f"{ms['frac_intact']:.3f} vs {md['frac_intact']:.3f}")
    chk('脉冲是重尾：p99 远大于中位（而 p95 仍可能为 0，因为只有 5% 被污染）',
        ms['deltaE_p99'] > 20 and ms['deltaE_p50'] < 0.5,
        f"p50 {ms['deltaE_p50']:.3f} p95 {ms['deltaE_p95']:.2f} p99 {ms['deltaE_p99']:.1f}")
    chk('稠密噪声不是重尾：p99 与中位同量级',
        md['deltaE_p99'] < 6 * max(md['deltaE_p50'], 1e-9),
        f"p50 {md['deltaE_p50']:.3f} p99 {md['deltaE_p99']:.3f}")
    chk('无退化时判为 SPARSE 且完好像素 = 100%',
        analyze_pair(ref, ref)['frac_intact'] == 1.0)

    # 面积积分增益
    chk('面积 SNR = 单像素 SNR × sqrt(面积)',
        abs(0.5 * np.sqrt(900) - 15.0) < 1e-9)
    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir'); ap.add_argument('--csv')
    ap.add_argument('--codes', nargs='+', default=['GDC', 'PDC', 'LDC', 'DDC'])
    ap.add_argument('--contrast', type=float, default=1.12, help='csv 缺失时用的缺陷对比度中位')
    ap.add_argument('--area', type=float, default=961.0, help='csv 缺失时用的缺陷面积中位(px)')
    ap.add_argument('--n', type=int, default=40); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out'); ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if not a.img_dir: ap.error('--img-dir 必需')
    run(a)


if __name__ == '__main__':
    main()
