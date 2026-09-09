#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强样本信噪比审计（解析判据，零训练）

依据：数据集的增强代码已公开
  https://github.com/CkFXXL123/The-data-augmentation-file-of-the-dataset
    GaussianNoise.m    imnoise(I,'gaussian',0,0.25)  -> sigma = 0.5（[0,1] 量纲 = 128 灰阶）
    SaltPepperBatch.m  imnoise(I,'salt & pepper',0.2) -> 密度 20%，逐通道独立
    BrightnessBatch.py HSV 的 V 通道 x1.3 / x0.7
    Rotation/Mirror    cv2.rotate / cv2.flip，无损像素置换

本脚本把这些参数换算成与缺陷对比度【同一个 Lab ΔE 量纲】的噪声 sigma，
再对每个 GT 实例算：
    单像素 SNR   = 缺陷对比度(ΔE) / sigma(ΔE)
    面积积分 SNR = 单像素 SNR × sqrt(缺陷面积)      （匹配滤波的理论上界）
低于阈值的实例，其标注在该增强副本上【没有图像证据支撑】—— 对网络而言是标签噪声。

用法:
  python3 tools/analysis/aug_snr_audit.py \
    --csv prep/innov1/paired_instances.csv \
    --img-dir datasets/Water-Based-Coated-Wood/images/train \
    --ann datasets/Water-Based-Coated-Wood/annotations/instances_train.json \
    --out prep/innov3/aug_snr_audit.json \
    --keep-table prep/innov3/train_keep.json
  python3 tools/analysis/aug_snr_audit.py --self-test-only

--img-dir 可选：给了就从真实 ODC 图采样底色分布来标定 sigma(ΔE)；
不给就用 --base-level（默认 0.45，高光泽涂层的典型中性中灰）。
"""
import argparse, csv, json, os, re, sys
import numpy as np

NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')
CODES = ('ODC', 'LDC', 'DDC', 'GDC', 'PDC')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
# 公开增强代码里的硬参数
AUG_PARAMS = dict(gauss_sigma=0.5, sp_density=0.2, bright_up=1.3, bright_down=0.7)


def norm_id(s): return str(int(s))


def srgb_to_lab(rgb):
    """float 路径，L in [0,100]。白点由矩阵行和导出，保证纯白严格 L=100。"""
    m = rgb <= 0.04045
    lin = np.where(m, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    t = xyz / M.sum(axis=1)
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def contrast_to_intensity(contrast_de, base=0.45):
    """把 Lab ΔE 的对比度换算成 sRGB [0,1] 强度差（局部线性化）。"""
    l0 = srgb_to_lab(np.array([[[base] * 3]]))[0, 0, 0]
    lo, hi = 0.0, min(1.0 - base, 0.5)
    for _ in range(60):
        mid = (lo + hi) / 2
        if srgb_to_lab(np.array([[[base + mid] * 3]]))[0, 0, 0] - l0 < contrast_de:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sigma_de(code, base_samples, params, rng, n=200000):
    """把增强参数换算成 Lab ΔE 量纲的噪声 sigma（含裁剪的非线性，蒙特卡洛）。"""
    b = rng.choice(base_samples, n) if len(base_samples) > 1 else np.full(n, base_samples[0])
    base = np.repeat(b[:, None], 3, 1)
    if code == 'ODC':
        deg = base.copy()
    elif code == 'GDC':
        deg = np.clip(base + rng.normal(0, params['gauss_sigma'], base.shape), 0, 1)
    elif code == 'PDC':
        deg = base.copy()
        m = rng.random(base.shape) < params['sp_density']
        deg[m] = (rng.random(int(m.sum())) < 0.5).astype(np.float64)
    elif code == 'LDC':
        deg = np.clip(base * params['bright_up'], 0, 1)
    elif code == 'DDC':
        deg = np.clip(base * params['bright_down'], 0, 1)
    else:
        raise ValueError(code)
    d = srgb_to_lab(deg[:, None, :]) - srgb_to_lab(base[:, None, :])
    de = np.sqrt((d ** 2).sum(-1)).ravel()
    intact = float((de < 0.5 * 2.3).mean())
    sat = float(((deg <= 0.001) | (deg >= 0.999)).mean())
    return dict(sigma_de=float(np.sqrt((de ** 2).mean())),
                de_p50=float(np.median(de)), de_p99=float(np.percentile(de, 99)),
                frac_intact=intact, frac_saturated=sat)


def sample_base_levels(img_dir, n_img=20, rng=None):
    """从 ODC 图里采底色分布（灰度均值层面），用于标定 sigma。"""
    from PIL import Image
    files = [f for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in EXTS and '_ODC_' in f]
    if not files:
        return None
    rng = rng or np.random.default_rng(0)
    pick = [files[i] for i in rng.choice(len(files), min(n_img, len(files)), replace=False)]
    vals = []
    for f in pick:
        a = np.asarray(Image.open(os.path.join(img_dir, f)).convert('RGB'),
                       dtype=np.float64) / 255.0
        g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        vals.append(g.ravel()[::97])
    return np.concatenate(vals)


# ---------------------------------------------------------------- 主流程
def run(args):
    rng = np.random.default_rng(args.seed)
    base_samples = None
    if args.img_dir and os.path.isdir(args.img_dir):
        base_samples = sample_base_levels(args.img_dir, args.n_img, rng)
    if base_samples is None or len(base_samples) == 0:
        base_samples = np.array([args.base_level])
        print(f"[audit] 未采到 ODC 底色，使用 --base-level {args.base_level}")
    else:
        print(f"[audit] 从 {args.img_dir} 采到 {len(base_samples)} 个底色样本，"
              f"中位 {np.median(base_samples):.3f}")

    params = dict(AUG_PARAMS)
    for k in params:
        v = getattr(args, k, None)
        if v is not None:
            params[k] = v
    print(f"[audit] 增强参数 {params}")

    noise = {c: sigma_de(c, base_samples, params, rng) for c in CODES}

    rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8'))
            if r.get('contrast') not in (None, '', 'nan')]
    if not rows:
        sys.exit('csv 里没有可用实例')
    contrast = np.array([float(r['contrast']) for r in rows])
    area = np.array([float(r['area']) for r in rows])
    print(f"[audit] {len(rows)} 个实例：对比度中位 {np.median(contrast):.3f} ΔE，"
          f"面积中位 {np.median(area):.0f} px")
    bl = float(np.median(base_samples))
    print(f"[audit] 对比度 {np.median(contrast):.3f} ΔE = "
          f"{contrast_to_intensity(float(np.median(contrast)), bl):.4f} 强度单位 = "
          f"{contrast_to_intensity(float(np.median(contrast)), bl)*255:.2f} 个灰阶")

    out = {'aug_params': params, 'noise': noise, 'n_instances': len(rows),
           'contrast_median': float(np.median(contrast)),
           'area_median': float(np.median(area)),
           'snr_thr': args.snr_thr, 'codes': {}}

    print(f"\n{'码':6s} {'σ(ΔE)':>8s} {'完好像素':>9s} {'饱和':>7s} "
          f"{'单像素SNR':>10s} {'中位面积SNR':>12s} {'SNR<1':>8s} {'SNR<%.0f' % args.snr_thr:>8s}")
    for c in CODES:
        s = noise[c]['sigma_de']
        if s <= 1e-9:
            pp = float('inf'); sa = np.full(len(rows), np.inf)
        else:
            pp = float(np.median(contrast)) / s
            sa = (contrast / s) * np.sqrt(np.maximum(area, 1.0))
        f1 = float((sa < 1.0).mean()); ft = float((sa < args.snr_thr).mean())
        out['codes'][c] = dict(sigma_de=s, snr_perpixel_median=pp,
                               snr_area_median=float(np.median(sa)),
                               frac_below_1=f1, frac_below_thr=ft,
                               **{k: v for k, v in noise[c].items() if k != 'sigma_de'})
        ppd = '   inf' if not np.isfinite(pp) else f"{pp:10.4f}"
        sam = '   inf' if not np.isfinite(np.median(sa)) else f"{np.median(sa):12.2f}"
        print(f"{c:6s} {s:8.2f} {noise[c]['frac_intact']*100:8.1f}% "
              f"{noise[c]['frac_saturated']*100:6.1f}% {ppd} {sam} "
              f"{f1*100:7.1f}% {ft*100:7.1f}%")

    # ---- 判据 ----
    print(f"\n================ 判据（阈值 SNR = {args.snr_thr}） ================")
    bad = [c for c in CODES if out['codes'][c]['frac_below_thr'] >= args.bad_frac]
    for c in CODES:
        d = out['codes'][c]
        tag = ('不可拟合（面积积分后仍低于阈值）' if d['frac_below_thr'] >= args.bad_frac
               else '可学')
        print(f"  {c}: {d['frac_below_thr']*100:.1f}% 的实例 SNR < {args.snr_thr} -> {tag}")
    out['bad_codes'] = bad
    print(f"\n  判为不可拟合的码：{bad or '（无）'}")

    # 训练集里这些码占多少
    if args.ann and os.path.exists(args.ann):
        ann = json.load(open(args.ann, encoding='utf-8'))
        per = {c: 0 for c in CODES}
        keep = {}
        for im in ann['images']:
            m = NAME_RE.match(os.path.splitext(os.path.basename(im['file_name']))[0])
            if not m: continue
            c = m.group(2); per[c] = per.get(c, 0) + 1
            keep[im['file_name']] = dict(code=c, keep=(c not in bad),
                                         weight=0.0 if c in bad else 1.0)
        tot = sum(per.values())
        drop = sum(per[c] for c in bad)
        print(f"\n  训练集 {tot} 张：{per}")
        print(f"  按判据应剔除 {drop} 张（{drop/max(tot,1)*100:.1f}%），保留 {tot-drop} 张")
        out['train_counts'] = per
        out['train_total'] = tot
        out['train_dropped'] = drop
        out['train_kept'] = tot - drop
        if args.keep_table:
            os.makedirs(os.path.dirname(args.keep_table) or '.', exist_ok=True)
            json.dump({'bad_codes': bad, 'snr_thr': args.snr_thr, 'images': keep},
                      open(args.keep_table, 'w', encoding='utf-8'),
                      indent=2, ensure_ascii=False)
            print(f"  写出保留表 {args.keep_table}")
        # 迭代对齐后的 epoches
        if drop:
            kept = tot - drop
            ep = int(round(132 * tot / max(kept, 1)))
            print(f"\n  迭代对齐：保留 {kept} 张时，--train-size {kept} --epoches {ep}"
                  f"（≈ 与现状 37884 iter 相同）")
            out['iter_matched_epoches'] = ep

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"\n写出 {args.out}")
    return out


# ---------------------------------------------------------------- 自检
def self_test():
    rng = np.random.default_rng(0); fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    w = srgb_to_lab(np.ones((1, 1, 3)))[0, 0]
    chk('Lab 纯白 L=100', abs(w[0] - 100) < 1e-9)
    chk('Lab L 在 0..100（防 uint8 路径）',
        0 <= srgb_to_lab(np.full((1, 1, 3), 0.5))[0, 0, 0] <= 100)

    # ΔE <-> 强度 换算（论文里要用的那个数）
    dv = contrast_to_intensity(1.12, 0.45)
    chk('对比度 1.12 ΔE ≈ 2.8 个灰阶', 2.4 < dv * 255 < 3.2, f'{dv*255:.2f} 灰阶')
    chk('换算单调', contrast_to_intensity(2.24, 0.45) > dv)

    base = np.array([0.45])
    n = {c: sigma_de(c, base, AUG_PARAMS, rng, 40000) for c in CODES}
    chk('ODC 无噪声', n['ODC']['sigma_de'] < 1e-9)
    chk('GDC σ(ΔE) 在 60~95（sigma=0.5 的必然后果）',
        60 < n['GDC']['sigma_de'] < 95, f"{n['GDC']['sigma_de']:.1f}")
    chk('GDC 完好像素 ≈ 0%', n['GDC']['frac_intact'] < 0.02,
        f"{n['GDC']['frac_intact']*100:.2f}%")
    chk('GDC 饱和像素 25~40%（裁剪造成的不可逆损失）',
        0.25 < n['GDC']['frac_saturated'] < 0.40, f"{n['GDC']['frac_saturated']*100:.1f}%")
    chk('PDC 逐通道 20% -> 每像素受影响 ≈ 48.8%，完好 ≈ 51%',
        0.46 < n['PDC']['frac_intact'] < 0.56, f"{n['PDC']['frac_intact']*100:.1f}%")
    chk('PDC 的 ΔE 中位 = 0（一半像素逐位未变）', n['PDC']['de_p50'] < 1e-9)
    chk('PDC 重尾：p99 >> 中位', n['PDC']['de_p99'] > 50)
    chk('LDC/DDC 是确定性变换：p99 ≈ 中位',
        n['LDC']['de_p99'] < 1.5 * max(n['LDC']['de_p50'], 1e-9)
        and n['DDC']['de_p99'] < 1.5 * max(n['DDC']['de_p50'], 1e-9))
    chk('LDC/DDC 的 σ 远小于 GDC/PDC',
        max(n['LDC']['sigma_de'], n['DDC']['sigma_de']) < 0.4 * n['GDC']['sigma_de'])

    # SNR 核心公式：中位缺陷在 GDC 下面积积分后仍 < 1
    c_med, a_med = 1.12, 31 * 62
    snr = c_med / n['GDC']['sigma_de'] * np.sqrt(a_med)
    chk('★ 中位缺陷在 GDC 下的面积积分 SNR < 1（信息论不可检出）',
        snr < 1.0, f'{snr:.3f}')
    snr_big = c_med / n['GDC']['sigma_de'] * np.sqrt(209 * 209)
    chk('大目标（95 分位）在 GDC 下 SNR > 1（所以 GDC 的 AP 不是 0）',
        snr_big > 1.0, f'{snr_big:.2f}')

    # 参数灵敏度：噪声减半，SNR 应当翻倍
    p2 = dict(AUG_PARAMS); p2['gauss_sigma'] = 0.25
    n2 = sigma_de('GDC', base, p2, rng, 40000)
    chk('sigma 减半 -> σ(ΔE) 明显下降，SNR 上升',
        n2['sigma_de'] < 0.75 * n['GDC']['sigma_de'],
        f"{n['GDC']['sigma_de']:.1f} -> {n2['sigma_de']:.1f}")

    # 底色分布不改变结论
    n3 = sigma_de('GDC', rng.uniform(0.2, 0.7, 5000), AUG_PARAMS, rng, 40000)
    chk('换一批底色，GDC 仍然 σ(ΔE) >> 缺陷对比度',
        n3['sigma_de'] > 20 * 1.12, f"{n3['sigma_de']:.1f}")

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', help='degradation_paired 的 paired_instances.csv')
    ap.add_argument('--img-dir'); ap.add_argument('--ann')
    ap.add_argument('--out'); ap.add_argument('--keep-table')
    ap.add_argument('--base-level', type=float, default=0.45)
    ap.add_argument('--n-img', type=int, default=20)
    ap.add_argument('--snr-thr', type=float, default=3.0,
                    help='面积积分 SNR 的可检测阈；1.0 是绝对信息下界，3.0 是常用可靠检出线')
    ap.add_argument('--bad-frac', type=float, default=0.5,
                    help='某码有多少比例实例低于阈值就判为不可拟合')
    ap.add_argument('--gauss-sigma', type=float); ap.add_argument('--sp-density', type=float)
    ap.add_argument('--bright-up', type=float); ap.add_argument('--bright-down', type=float)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if not a.csv: ap.error('--csv 必需')
    run(a)


if __name__ == '__main__':
    main()
