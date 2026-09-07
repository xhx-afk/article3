#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去噪质量 → 检测 AP 的关系（补上实验 O 结论的洞）

实验 O 得到"所有频带操作 ΔPDC 在 ±1.6 内"，据此落子 C（频带路径无效）。
但审稿人会问一句：**你的小波抑制到底把噪声去掉了没有？**
合成对照显示，被测的那批算子即使在理想条件下也比中值滤波弱 9~14 dB：

    退化图 18.49 | median3 +15.07 | median5 +14.59
    clip1.0(1层) +5.75 | clip1.0(3层) +10.32 | hh0 +1.21 | shrink1.0 +2.36

所以"频带路径无效"这句话，必须先量出真实数据上各变体的去噪强度才站得住。
本脚本把每个变体目录相对【配对 ODC 原图】的 PSNR/SSIM 算出来，与已有的 AP 并排，
给出「去噪质量 → ΔAP」的关系。

三种结局（预注册）:
  1. 变体的 ΔPSNR ≈ 0            -> 算子在真实数据上没去噪，落子 C 的措辞必须改成
                                    "所测算子无效"，不能推广到"频带路径无效"
  2. ΔPSNR 明显为正但 ΔAP ≈ 0，
     而 median 在同等/更高 PSNR 下 +23.4 AP
                                 -> **最强的负结果**：AP 与去噪质量脱钩，
                                    频率假设被机制性否定
  3. ΔPSNR 略正但远低于 median   -> C 成立，但必须写明"在 X dB 的抑制强度下"

用法:
  python3 tools/analysis/denoise_quality_vs_ap.py \
    --odc-dir datasets/Water-Based-Coated-Wood/images/val \
    --dirs orig=datasets/Water-Based-Coated-Wood/images/val \
           id=prep/innov1/wavelet_probe/id \
           hh0=prep/innov1/wavelet_probe/hh0 \
           clip1.0=prep/innov1/wavelet_probe/clip1.0 \
           clip1.0_L3=prep/innov1/wavelet_probe/clip1.0_L3 \
           median3=prep/denoise/median3 median5=prep/denoise/median5 \
    --code PDC --n 60 \
    --ap orig=15.86 id=16.27 hh0=16.66 clip1.0=15.42 median3=34.07 median5=38.44 \
    --out prep/innov1/denoise_quality_vs_ap.json
  python3 tools/analysis/denoise_quality_vs_ap.py --self-test-only
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
import numpy as np

NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')
CODES = ('ODC', 'LDC', 'DDC', 'GDC', 'PDC')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def norm_id(s): return str(int(s))


def gray(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def psnr(x, y, data_range=1.0):
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse <= 0 else float(10 * np.log10(data_range ** 2 / mse))


def _gauss1d(sigma=1.5, radius=5):
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(x, k):
    pad = len(k) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode='reflect')
    y = np.zeros_like(x)
    for i, w in enumerate(k):
        y += w * xp[i:i + x.shape[0], :]
    yp = np.pad(y, ((0, 0), (pad, pad)), mode='reflect')
    z = np.zeros_like(x)
    for i, w in enumerate(k):
        z += w * yp[:, i:i + x.shape[1]]
    return z


def ssim(x, y, data_range=1.0):
    """标准 SSIM（11x11 高斯窗, sigma 1.5），纯 numpy。"""
    k = _gauss1d(1.5, 5)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = _blur(x, k), _blur(y, k)
    mxx, myy, mxy = _blur(x * x, k), _blur(y * y, k), _blur(x * y, k)
    vx, vy, vxy = mxx - mx * mx, myy - my * my, mxy - mx * my
    s = ((2 * mx * my + C1) * (2 * vxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    return float(s.mean())


def rankdata(x):
    x = np.asarray(x, float); order = np.argsort(x, kind='mergesort'); sx = x[order]
    r = np.empty(len(x)); i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]: j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1; i = j + 1
    out = np.empty(len(x)); out[order] = r; return out


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return 0.0 if d <= 0 else float((a * b).sum() / d)


def spearman(a, b): return pearson(rankdata(a), rankdata(b))


def index_dir(d):
    out = {}
    if not os.path.isdir(d): return out
    for f in os.listdir(d):
        if os.path.splitext(f)[1].lower() not in EXTS: continue
        m = NAME_RE.match(os.path.splitext(f)[0])
        if m: out[(norm_id(m.group(1)), m.group(2), m.group(3))] = os.path.join(d, f)
    return out


def find_ap(obj, code):
    """在任意嵌套的 by_code json 里找到该增强码的 AP，schema 无关。"""
    if isinstance(obj, dict):
        if all(c in obj for c in CODES):
            v = obj[code]
            if isinstance(v, (int, float)): return float(v)
            if isinstance(v, dict):
                for k in ('AP', 'ap', 'AP50_95'):
                    if k in v and isinstance(v[k], (int, float)): return float(v[k])
        for v in obj.values():
            r = find_ap(v, code)
            if r is not None: return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_ap(v, code)
            if r is not None: return r
    return None


def run(args):
    ref_idx = index_dir(args.odc_dir)
    keys = sorted(k for k in ref_idx if k[1] == 'ODC')
    if not keys: sys.exit(f'{args.odc_dir} 里没有 ODC 图')
    rng = np.random.default_rng(args.seed)
    pick = [keys[i] for i in rng.choice(len(keys), min(args.n, len(keys)), replace=False)]
    print(f"[quality] ODC 参考 {len(keys)} 张，抽 {len(pick)} 组，退化码 = {args.code}")

    ap_map = {}
    for kv in (args.ap or []):
        n, v = kv.split('=', 1); ap_map[n] = float(v)
    for pat in (args.ap_json or []):
        for p in glob.glob(pat):
            name = re.sub(r'^by_code_|\.json$', '', os.path.basename(p))
            v = find_ap(json.load(open(p, encoding='utf-8')), args.code)
            # by_code_*.json stores AP as a FRACTION (effective_n convention)
            # while --ap manual values are PERCENT (e.g. orig=16.26);
            # normalize to percent so the two sources share one scale
            if v is not None:
                ap_map.setdefault(name, v * 100.0 if v <= 1.0 else v)

    rows = []
    for spec in args.dirs:
        name, d = spec.split('=', 1)
        idx = index_dir(d)
        ps, ss, n_ok = [], [], 0
        for (src, _, dirn) in pick:
            kd = (src, args.code, dirn)
            if kd not in idx: continue
            ref = gray(ref_idx[(src, 'ODC', dirn)]); cur = gray(idx[kd])
            if ref.shape != cur.shape: continue
            ps.append(psnr(cur, ref)); ss.append(ssim(cur, ref)); n_ok += 1
        if not n_ok:
            print(f"  [warn] {name}: 在 {d} 里没找到 {args.code} 的配对图，跳过"); continue
        rows.append(dict(name=name, dir=d, n=n_ok,
                         psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                         ap=ap_map.get(name)))

    base = next((r for r in rows if r['name'] == args.baseline), None)
    if base is None:
        print(f"  [warn] 没有名为 '{args.baseline}' 的行，ΔPSNR/ΔAP 无法计算")
    print(f"\n{'变体':14s} {'n':>4s} {'PSNR':>8s} {'ΔPSNR':>8s} {'SSIM':>7s} "
          f"{'ΔSSIM':>8s} {'AP':>7s} {'ΔAP':>7s}")
    for r in rows:
        dp = r['psnr'] - base['psnr'] if base else float('nan')
        ds = r['ssim'] - base['ssim'] if base else float('nan')
        da = (r['ap'] - base['ap']) if (base and r['ap'] is not None
                                        and base['ap'] is not None) else float('nan')
        r['d_psnr'], r['d_ssim'], r['d_ap'] = dp, ds, da
        aps = f"{r['ap']:7.2f}" if r['ap'] is not None else '      -'
        das = f"{da:+7.2f}" if np.isfinite(da) else '      -'
        print(f"{r['name']:14s} {r['n']:4d} {r['psnr']:8.2f} {dp:+8.2f} "
              f"{r['ssim']:7.4f} {ds:+8.4f} {aps} {das}")

    ok = [r for r in rows if np.isfinite(r.get('d_ap', np.nan)) and r['name'] != args.baseline]
    out = dict(code=args.code, n_pairs=len(pick), rows=rows)
    if len(ok) >= 3:
        rho = spearman([r['d_psnr'] for r in ok], [r['d_ap'] for r in ok])
        out['spearman_dpsnr_dap'] = rho
        tag = '符合假设：去噪越好 AP 越高' if rho >= 0.5 else \
              ('与假设相反：去噪质量与 AP 无关或反向' if rho < 0.3 else '关系微弱')
        print(f"\nSpearman(ΔPSNR, ΔAP) = {rho:+.3f}  ——【{tag}】")

    # ---- 预注册判据 ----
    wav = [r for r in ok if not r['name'].startswith('median') and
           not r['name'].startswith('bilateral')]
    med = [r for r in ok if r['name'].startswith('median')]
    print("\n================ 判据 ================")
    if wav:
        best = max(wav, key=lambda r: r['d_psnr'])
        print(f"  小波族最强去噪：{best['name']}  ΔPSNR {best['d_psnr']:+.2f} dB, "
              f"ΔAP {best['d_ap']:+.2f}")
        if best['d_psnr'] < 0.5:
            v = ('结局 1：小波变体在真实数据上几乎没有去噪 → '
                 '落子 C 的措辞必须收窄为「所测算子无效」，不能写成「频带路径无效」')
        elif med and best['d_psnr'] >= 0.6 * max(r['d_psnr'] for r in med) and abs(best['d_ap']) < 2.0:
            v = ('结局 2：去噪质量可比而 AP 脱钩 → **最强的负结果**，'
                 '频率假设被机制性否定，可直接写进论文')
        elif not med:
            v = (f"抑制强度 {best['d_psnr']:+.2f} dB，但本次没有 median 参照行 —— "
                 '结局 2 与 3 无法区分，请把 prep/denoise/median{3,5} 一并传进 --dirs')
        else:
            v = (f"结局 3：抑制强度仅 {best['d_psnr']:+.2f} dB（median 为 "
                 f"{max(r['d_psnr'] for r in med):+.2f} dB）→ "
                 'C 成立，但论文里必须写明抑制强度')
        out['verdict'] = v
        print(f"  >>> {v}")
    if med:
        b = max(med, key=lambda r: r['d_ap'])
        print(f"  中值参照：{b['name']}  ΔPSNR {b['d_psnr']:+.2f} dB, ΔAP {b['d_ap']:+.2f}")

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

    x = rng.random((64, 64))
    chk('PSNR(x,x) = 99（完全相同）', psnr(x, x) == 99.0)
    y = np.clip(x + 0.1, 0, 1)
    chk('PSNR 与解析式一致',
        abs(psnr(x, x + 0.1) - 10 * np.log10(1 / 0.01)) < 1e-9,
        f'{psnr(x, x+0.1):.4f} vs {10*np.log10(1/0.01):.4f}')
    chk('SSIM(x,x) = 1', abs(ssim(x, x) - 1.0) < 1e-9, f'{ssim(x,x):.6f}')
    n1 = np.clip(x + rng.normal(0, .02, x.shape), 0, 1)
    n2 = np.clip(x + rng.normal(0, .10, x.shape), 0, 1)
    chk('SSIM 随噪声单调下降', ssim(x, n1) > ssim(x, n2),
        f'{ssim(x,n1):.4f} > {ssim(x,n2):.4f}')
    chk('SSIM 在 [0,1] 内', 0 <= ssim(x, n2) <= 1)
    # SSIM 对结构变化敏感（PSNR 可能相同）
    shifted = np.roll(x, 3, axis=1)
    chk('SSIM 对结构位移敏感', ssim(x, shifted) < 0.9, f'{ssim(x, shifted):.4f}')

    chk('spearman 完全单调 = 1', abs(spearman([1, 2, 3, 4], [2, 4, 9, 20]) - 1) < 1e-9)
    chk('spearman 完全反向 = -1', abs(spearman([1, 2, 3, 4], [9, 5, 2, 1]) + 1) < 1e-9)

    chk('find_ap 能在嵌套 json 里定位',
        find_ap({'a': {'b': {'ODC': 1, 'LDC': 2, 'DDC': 3, 'GDC': 4, 'PDC': 5.5}}}, 'PDC') == 5.5)
    chk('find_ap 支持 {code: {AP: x}} 形式',
        find_ap({'x': {c: {'AP': i} for i, c in enumerate(CODES)}}, 'PDC') == 4.0)
    chk('find_ap 找不到时返回 None', find_ap({'q': 1}, 'PDC') is None)

    chk('文件名解析与归一化',
        NAME_RE.match('00001_PDC_3').groups() == ('00001', 'PDC', '3')
        and norm_id('00001') == norm_id('1') == '1')

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odc-dir', help='含 ODC 原图的目录（通常就是原始 val 图像目录）')
    ap.add_argument('--dirs', nargs='+', default=[], help='name=path，可给多个')
    ap.add_argument('--code', default='PDC')
    ap.add_argument('--baseline', default='orig', help='作为 Δ 参照的那一行名字')
    ap.add_argument('--ap', nargs='*', help='name=AP，手工给 AP')
    ap.add_argument('--ap-json', nargs='*', help='by_code_*.json 的 glob，自动取 AP')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if not a.odc_dir or not a.dirs: ap.error('--odc-dir 与 --dirs 必需')
    run(a)


if __name__ == '__main__':
    main()
