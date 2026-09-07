#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDC 到底是"高斯模糊"还是"高斯噪声"？（5 分钟，纯 numpy + Pillow）

为什么要查：交接文档 §3.3 的核心论证是"椒盐（可逆）比高斯模糊（不可逆信息损失）
更致命"。但三条实测都指向 GDC 不是模糊：
  1. 子带探针：GDC 的 HH 内份额【升高】1.76~1.90x —— 模糊不会抬高 HH；
  2. 去噪探针：median 让 GDC 变【差】(24.10 -> 19.41) —— median 是脉冲工具，
     对模糊无害，但对加性噪声无效甚至有害；
  3. 代号 GaussianDirectionChange 从未说明是 blur。
若 GDC 是加性高斯噪声，§3.3 的"反常"要重写，创新点 1 的动机段落要跟着改，
论文里"不可逆 vs 可逆"这条对比也不能再用。

用法:
  python3 tools/analysis/gdc_nature_check.py \
      --img-dir datasets/Water-Based-Coated-Wood/images/val \
      --codes GDC PDC LDC DDC --n 40 --out prep/innov1/gdc_nature.json
  python3 tools/analysis/gdc_nature_check.py --self-test-only
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np

NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')


def gray(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def hf_energy(x):
    """一层 Haar 的三个细节子带总能量（衡量高频含量）。"""
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    a = x[:h, :w]
    x00, x01, x10, x11 = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
    lh = (x00 + x01 - x10 - x11) / 2; hl = (x00 - x01 + x10 - x11) / 2
    hh = (x00 - x01 - x10 + x11) / 2
    return float((lh ** 2 + hl ** 2 + hh ** 2).mean())


def grad_mag(x):
    gy = np.zeros_like(x); gx = np.zeros_like(x)
    gy[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2
    gx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
    return np.hypot(gx, gy)


def acf1(r):
    """残差在水平/垂直方向 lag-1 的自相关均值。白噪声 ~0，结构化残差显著非 0。"""
    r = r - r.mean()
    d = (r ** 2).mean()
    if d <= 0: return 0.0
    return float(((r[:, :-1] * r[:, 1:]).mean() + (r[:-1, :] * r[1:, :]).mean()) / (2 * d))


def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return 0.0 if den <= 0 else float((a * b).sum() / den)


def local_std(x, k=8):
    h, w = x.shape[0] // k * k, x.shape[1] // k * k
    b = x[:h, :w].reshape(h // k, k, w // k, k)
    return b.std(axis=(1, 3))


def analyze_pair(ref, deg):
    """ref = ODC（干净），deg = 退化图。返回一组判别量。"""
    r = deg - ref
    g = grad_mag(ref)
    ls_ref = local_std(ref)
    ls_res = local_std(r)
    smooth = ls_ref <= np.percentile(ls_ref, 25)
    textured = ls_ref >= np.percentile(ls_ref, 75)
    sm = float(ls_res[smooth].mean()) if smooth.any() else 0.0
    tx = float(ls_res[textured].mean()) if textured.any() else 0.0
    rc = r - r.mean()
    var = float((rc ** 2).mean())
    hf_ref = hf_energy(ref)
    return dict(
        hf_ref=hf_ref,
        hf_ratio=hf_energy(deg) / max(hf_ref, 1e-12),
        # 块级相关比逐像素相关稳健得多：模糊的残差强度正比于【局部高频含量】
        corr_blockstd=pearson(ls_res, ls_ref),
        res_std=float(r.std()),
        res_mean=float(r.mean()),
        acf1=acf1(r),
        corr_absres_grad=pearson(np.abs(r), g),
        sparse_frac=float((np.abs(r) > 0.2).mean()),
        kurtosis=float((rc ** 4).mean() / max(var ** 2, 1e-24)) if var > 0 else 0.0,
        std_textured_over_smooth=tx / max(sm, 1e-9),
    )


def verdict(m):
    """按判别量给出定性结论。

    主判别量是 hf_ratio（退化图 / 干净图 的细节子带能量比）—— 自检里四种已知退化
    分得非常开：模糊 0.18 / 提亮 1.44 / 加性高斯 3.51 / 椒盐 13.9。
    次判别量用于把"抬高高频"的三种情况分开。判定顺序不能调换。
    """
    # 1) 脉冲：稀疏 + 重尾，最好认
    if m['sparse_frac'] > 0.005 and m['kurtosis'] > 10:
        return 'IMPULSE（脉冲/椒盐：稀疏、大幅、重尾）'
    # 2) 亮度/对比度变换：残差平滑且与图像高度相关
    if abs(m['res_mean']) > 2 * max(m['res_std'], 1e-9) or \
            (m['kurtosis'] < 6 and abs(m['acf1']) > 0.7):
        return 'PHOTOMETRIC（亮度/对比度变换：残差平滑且高度相关）'
    # 3) 模糊：高频被压低。这是模糊唯一不可伪装的签名
    if m['hf_ratio'] < 0.9:
        return 'BLUR（模糊：高频被压低，残差正比于局部高频含量）'
    # 4) 加性噪声：高频被抬高，且残差与图像内容无关
    if m['hf_ratio'] > 1.1 and abs(m['acf1']) < 0.30 and m['corr_blockstd'] < 0.35:
        return 'ADDITIVE_NOISE（加性噪声：高频被抬高，残差与图像内容无关）'
    return 'UNCERTAIN（不落在任何模板内，看下面的数自行判断）'


def run(args):
    files = [f for f in os.listdir(args.img_dir)
             if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    groups = defaultdict(dict)
    for f in files:
        m = NAME_RE.match(os.path.splitext(f)[0])
        if m:
            groups[(str(int(m.group(1))), m.group(3))][m.group(2)] = f
    keys = sorted(k for k, v in groups.items() if 'ODC' in v)
    rng = np.random.default_rng(args.seed)
    pick = [keys[i] for i in rng.choice(len(keys), min(args.n, len(keys)), replace=False)]
    print(f"[gdc_nature] {len(keys)} 组可用，抽 {len(pick)} 组")

    out = {}
    for code in args.codes:
        acc = defaultdict(list)
        for k in pick:
            if code not in groups[k]:
                continue
            ref = gray(os.path.join(args.img_dir, groups[k]['ODC']))
            deg = gray(os.path.join(args.img_dir, groups[k][code]))
            if ref.shape != deg.shape:
                continue
            for kk, vv in analyze_pair(ref, deg).items():
                acc[kk].append(vv)
        if not acc:
            continue
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        m['verdict'] = verdict(m)
        out[code] = m

    print(f"\n{'code':6s} {'HF比':>8s} {'残差sd':>8s} {'acf1':>7s} {'块相关':>8s} "
          f"{'稀疏率':>8s} {'峰度':>8s} {'纹理/平滑':>9s}  判定")
    for code, m in out.items():
        print(f"{code:6s} {m['hf_ratio']:8.3f} {m['res_std']:8.4f} {m['acf1']:7.3f} "
              f"{m['corr_blockstd']:8.3f} {m['sparse_frac']:8.4f} {m['kurtosis']:8.2f} "
              f"{m['std_textured_over_smooth']:9.2f}  {m['verdict']}")

    print("\n读法：")
    print("  HF比  <1 = 高频被压低(模糊)；>1 = 高频被抬高(加性噪声/脉冲)")
    print("  块相关 高 = 残差强度正比于局部高频含量(模糊)；≈0 = 与内容无关(加性噪声)")
    print("  稀疏率+峰度 都高 = 脉冲")
    hr = float(np.mean([m['hf_ref'] for m in out.values()])) if out else 0.0
    if hr < 1e-5:
        print(f"\n  ⚠️ 干净图自身的高频能量极低 ({hr:.2e})，HF比 的分母很小、读数不稳。"
              "\n     真实木纹图不应出现这种情况；若出现，先确认没有拿错目录（比如拿到了缩略图）。")
    if 'GDC' in out:
        v = out['GDC']['verdict']
        print(f"\n>>> GDC 判定：{v}")
        if v.startswith('BLUR'):
            print("    与交接文档 §3.3 一致，'不可逆信息损失 vs 可去除噪声' 的对比可以继续用。")
        else:
            print("    【与交接文档 §3.3 相反】。§3.3 的'反常'需重写：")
            print("    - 不再是'椒盐 vs 不可逆模糊'，而是两种噪声之间的比较；")
            print("    - 创新点 1 的动机段落、鲁棒表的解释、V2 的'模糊压低高频'全部要改；")
            print("    - 去噪探针里 median 对 GDC 有害也就自洽了（median 是脉冲工具）。")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"\n写出 {args.out}")


def self_test():
    rng = np.random.default_rng(0); fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    base = np.cumsum(np.cumsum(rng.normal(0, 1, (256, 256)), 0), 1)
    base = (base - base.min()) / (base.max() - base.min()) * 0.6 + 0.2
    base += 0.03 * rng.normal(size=base.shape)          # 一点纹理，模拟木纹
    base = np.clip(base, 0, 1)

    k = np.array([1, 4, 6, 4, 1.]); k /= k.sum()
    blur = base.copy()
    for ax in (0, 1):
        blur = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), ax, blur)
    noise = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1)
    sp = base.copy(); m = rng.random(base.shape) < 0.05
    sp[m] = rng.choice([0.0, 1.0], m.sum())
    bright = np.clip(base * 1.2, 0, 1)

    for name, img, want in (('模糊', blur, 'BLUR'), ('加性高斯噪声', noise, 'ADDITIVE_NOISE'),
                            ('椒盐', sp, 'IMPULSE'), ('提亮', bright, 'PHOTOMETRIC')):
        mm = analyze_pair(base, img); v = verdict(mm)
        chk(f'判别 {name}', v.startswith(want),
            f"-> {v.split('（')[0]}  (HF比 {mm['hf_ratio']:.2f}, 块相关 {mm['corr_blockstd']:.2f}, "
            f"稀疏 {mm['sparse_frac']:.4f}, 峰度 {mm['kurtosis']:.1f}, acf1 {mm['acf1']:.2f})")

    chk('acf1 对白噪声 ≈ 0', abs(acf1(rng.normal(size=(200, 200)))) < 0.05)
    chk('acf1 对平滑场 > 0.8', acf1(np.cumsum(np.cumsum(rng.normal(size=(200, 200)), 0), 1)) > 0.8)
    chk('hf_energy 对模糊图更低', hf_energy(blur) < hf_energy(base))
    chk('hf_energy 对噪声图更高', hf_energy(noise) > hf_energy(base))
    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir'); ap.add_argument('--out')
    ap.add_argument('--codes', nargs='+', default=['GDC', 'PDC', 'LDC', 'DDC'])
    ap.add_argument('--n', type=int, default=40); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if not a.img_dir: ap.error('--img-dir 必需')
    run(a)


if __name__ == '__main__':
    main()
