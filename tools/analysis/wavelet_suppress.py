#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 O · 小波域可恢复上限探针（图像侧，零训练）

问的问题：把"抑制被脉冲主导的子带"这件事在【输入图像上】做到极致，
PDC 最多能救回多少？这是任何网内子带门控的【上界】。
参照系：经典 median5 前端 = PDC 39.28（准备阶段实测）。

为什么必须做这个：A1 的空结果已经被证明是构造性的（线性子带融合 ≡ 2x2 步长卷积），
所以它没有回答"频带假设对不对"。这个探针不训练、不改模型，直接回答。

用法（与 denoise_probe.py 同构：给每个变体生成一个完整的 val 目录，未匹配的图软链）:
  python3 tools/analysis/wavelet_suppress.py \
      --img-dir datasets/Water-Based-Coated-Wood/images/val \
      --pattern '_(PDC|GDC)_' \
      --variants id hh0 hh50 det50 det0 shrink1.0 shrink2.0 \
      --out-root prep/innov1/wavelet_probe
  python3 tools/analysis/wavelet_suppress.py --self-test-only

随后对每个目录跑 dump_predictions + effective_n --by-code，同一份 ann.json 不变。

⚠️ `id` 变体是必须跑的对照：它做完整的正变换+逆变换但不改任何系数，
   用来把"重编码/浮点往返"的影响和"子带抑制"的影响分开。
   没有它，hh0 的 Δ 里混着一份说不清的编解码损失。
"""
import argparse, os, re, shutil, sys
import numpy as np

SUB = ('LL', 'LH', 'HL', 'HH')


# ---------------------------------------------------------------- Haar
def dwt2(a):
    """一层正交归一 Haar。a: (..., H, W)，H/W 必须为偶数。"""
    x00, x01 = a[..., 0::2, 0::2], a[..., 0::2, 1::2]
    x10, x11 = a[..., 1::2, 0::2], a[..., 1::2, 1::2]
    return ((x00 + x01 + x10 + x11) / 2.0,
            (x00 + x01 - x10 - x11) / 2.0,
            (x00 - x01 + x10 - x11) / 2.0,
            (x00 - x01 - x10 + x11) / 2.0)


def idwt2(LL, LH, HL, HH):
    """dwt2 的精确逆（正交基，逆 = 转置）。"""
    x00 = (LL + LH + HL + HH) / 2.0
    x01 = (LL + LH - HL - HH) / 2.0
    x10 = (LL - LH + HL - HH) / 2.0
    x11 = (LL - LH - HL + HH) / 2.0
    sh = list(LL.shape); sh[-2] *= 2; sh[-1] *= 2
    out = np.empty(sh, dtype=LL.dtype)
    out[..., 0::2, 0::2] = x00; out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10; out[..., 1::2, 1::2] = x11
    return out


def _pad_even(a):
    ph, pw = a.shape[-2] % 2, a.shape[-1] % 2
    if ph or pw:
        a = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)], mode='edge')
    return a, ph, pw


def mad_sigma(hh):
    """Donoho 的噪声尺度估计：sigma = median(|HH|) / 0.6745。"""
    return float(np.median(np.abs(hh)) / 0.6745)


def soft_threshold(c, tau):
    return np.sign(c) * np.maximum(np.abs(c) - tau, 0.0)


def apply_variant(img, variant, levels=1):
    """img: (H, W, 3) float in [0,1]。返回同形状。

    线性变体 (id/hh*/det*)：把子带乘常数 —— 注意这类操作【是线性的】，
      与 A1 失败的原因同类，所以它给出的是"线性子带缩放"的上界，会偏低。
    非线性变体 (shrink*)：对三个细节子带做软阈值（小波收缩），
      阈值 tau = k * sigma * sqrt(2 ln N)，sigma 由 HH 的 MAD 估计。
      这才是网内门控真正能逼近的那类操作 —— 【主要读这一组】。
    """
    a = np.moveaxis(img, -1, 0).astype(np.float64)      # (3, H, W)
    a, ph, pw = _pad_even(a)
    coeffs, cur = [], a
    for _ in range(levels):
        cur, ph2, pw2 = _pad_even(cur)
        LL, LH, HL, HH = dwt2(cur)
        coeffs.append((LH, HL, HH, ph2, pw2))
        cur = LL

    for lev in range(levels - 1, -1, -1):
        LH, HL, HH, ph2, pw2 = coeffs[lev]
        if variant == 'id':
            pass
        elif variant.startswith('hh'):
            LH, HL, HH = LH, HL, HH * (float(variant[2:]) / 100.0)
        elif variant.startswith('det'):
            f = float(variant[3:]) / 100.0
            LH, HL, HH = LH * f, HL * f, HH * f
        elif variant.startswith('clip'):
            # 系数限幅：把绝对值超过 k*sigma 的细节系数压回 ±k*sigma。
            # 这是【去脉冲】的正确算子（自检 T6 实测：椒盐 +5.7dB，而软阈值只有 +0.2dB）。
            # 注意它是【逐系数】的选择性操作 —— 一个"每子带一个标量"的门控
            # 在结构上无法表达它，这正是 A2 即使跑了也很可能失败的原因。
            k = float(variant[len('clip'):])
            for ci in range(a.shape[0]):
                tau = k * mad_sigma(HH[ci])
                LH[ci] = np.clip(LH[ci], -tau, tau)
                HL[ci] = np.clip(HL[ci], -tau, tau)
                HH[ci] = np.clip(HH[ci], -tau, tau)
        elif variant.startswith('shrink'):
            k = float(variant[len('shrink'):])
            for ci in range(a.shape[0]):
                sig = mad_sigma(HH[ci])
                tau = k * sig * np.sqrt(2.0 * np.log(max(HH[ci].size, 2)))
                LH[ci] = soft_threshold(LH[ci], tau)
                HL[ci] = soft_threshold(HL[ci], tau)
                HH[ci] = soft_threshold(HH[ci], tau)
        else:
            raise ValueError(f'未知变体 {variant}')
        cur = idwt2(cur, LH, HL, HH)
        if ph2: cur = cur[..., :-ph2, :]
        if pw2: cur = cur[..., :, :-pw2]

    if ph: cur = cur[..., :-ph, :]
    if pw: cur = cur[..., :, :-pw]
    return np.moveaxis(np.clip(cur, 0.0, 1.0), 0, -1)


VARIANT_HELP = {
    'id': '恒等（正逆变换但不改系数）—— 必跑对照，隔离编解码影响',
    'hh0': 'HH 置零', 'hh25': 'HH×0.25', 'hh50': 'HH×0.5',
    'det50': 'LH/HL/HH 全部 ×0.5', 'det0': '细节全部置零（= 2x2 均值下采样再放大，极端对照）',
    'shrink1.0': '软阈值收缩 k=1.0（非线性；高斯噪声的正确工具）',
    'shrink2.0': '软阈值收缩 k=2.0（更激进）',
    'clip1.0': '系数限幅 k=1.0（非线性、逐系数；脉冲噪声的正确工具，主读）',
    'clip2.0': '系数限幅 k=2.0',
}


# ---------------------------------------------------------------- 主流程
def run(args):
    from PIL import Image
    rex = re.compile(args.pattern)
    files = sorted(f for f in os.listdir(args.img_dir)
                   if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    hit = [f for f in files if rex.search(f)]
    print(f"[wavelet_suppress] {len(files)} 张图，匹配 '{args.pattern}' 的 {len(hit)} 张")
    if not hit:
        sys.exit("匹配到 0 张，检查 --pattern")

    for v in args.variants:
        out = os.path.join(args.out_root, v)
        os.makedirs(out, exist_ok=True)
        n_proc = 0
        for f in files:
            # 文件名与扩展名必须与原图【完全一致】：ann.json 的 file_name 指向 .jpg，
            # 改成 .png 会让 dataloader 找不到图（而且不会报有用的错）。
            dst = os.path.join(out, f)
            src = os.path.abspath(os.path.join(args.img_dir, f))
            if rex.search(f):
                im = np.asarray(Image.open(src).convert('RGB'), dtype=np.float64) / 255.0
                y = apply_variant(im, v, args.levels)
                pil = Image.fromarray(np.round(y * 255).astype(np.uint8))
                if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg'):
                    pil.save(dst, quality=100, subsampling=0)   # 最小化二次压缩
                else:
                    pil.save(dst)
                n_proc += 1
            else:
                # 未匹配的图一律软链（逐字节等于原图）——这样 ODC 的 AP 必须与 baseline
                # 完全相同，是一条免费的正确性校验。
                if os.path.lexists(dst): os.remove(dst)
                os.symlink(src, dst)
        # PARAMS 写在 out_root 下而不是图像目录里，避免给图像目录塞进非图片文件
        with open(os.path.join(args.out_root, f'{v}.params.json'), 'w', encoding='utf-8') as fh:
            fh.write(f'{{"variant": "{v}", "levels": {args.levels}, '
                     f'"pattern": "{args.pattern}", "processed": {n_proc}, '
                     f'"n_files": {len(files)}, "desc": "{VARIANT_HELP.get(v, "")}"}}\n')
        print(f"  [{v:10s}] 处理 {n_proc} 张，软链 {len(files)-n_proc} 张 -> {out}"
              f"   {VARIANT_HELP.get(v, '')}")

    print("\n自检提示：每个变体目录里未匹配的图都是软链，因此 ODC/LDC/DDC 的 AP"
          "\n        必须与 baseline 逐位相同；不同就说明脚本动了不该动的文件。")
    print("\n下一步：对每个目录跑 dump_predictions（-u val_dataloader.dataset.img_folder=<目录>）"
          "\n再跑 effective_n --by-code。ann.json 全程不变。"
          "\n判据见《创新点1-失败诊断与落子.md》§实验O。")


# ---------------------------------------------------------------- 自检
def self_test():
    rng = np.random.default_rng(0)
    fails = []
    def chk(n, c, extra=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {extra}")
        if not c: fails.append(n)

    # T1 正交性：正变换后逆变换精确还原
    a = rng.normal(size=(3, 64, 64))
    LL, LH, HL, HH = dwt2(a)
    chk('T1 Haar 逆变换精确还原', np.allclose(idwt2(LL, LH, HL, HH), a, atol=1e-12),
        f'max err {np.abs(idwt2(LL,LH,HL,HH)-a).max():.2e}')

    # T2 能量守恒
    chk('T2 能量守恒', np.isclose((LL**2+LH**2+HL**2+HH**2).sum(), (a**2).sum()))

    # T3 id 变体 = 恒等（float 级）
    img = rng.random((64, 64, 3))
    chk('T3 id 变体为恒等', np.allclose(apply_variant(img, 'id', 1), img, atol=1e-12),
        f'max err {np.abs(apply_variant(img,"id",1)-img).max():.2e}')
    chk('T3b id 变体多层也恒等', np.allclose(apply_variant(img, 'id', 3), img, atol=1e-12))

    # T4 奇数尺寸往返无损
    odd = rng.random((63, 65, 3))
    chk('T4 奇数尺寸 id 往返无损', np.allclose(apply_variant(odd, 'id', 2), odd, atol=1e-12))

    # 构造：平滑纹理 + 三种退化
    base = np.cumsum(np.cumsum(rng.normal(0, 1, (128, 128)), 0), 1)
    base = (base - base.min()) / (base.max() - base.min())
    base = np.repeat(base[..., None], 3, axis=-1) * 0.6 + 0.2
    sp = base.copy()
    m = rng.random(base.shape[:2]) < 0.05
    sp[m] = rng.choice([0.0, 1.0], m.sum())[:, None]
    gn = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1)
    k = np.array([1, 4, 6, 4, 1.]); k /= k.sum()
    bl = base.copy()
    for ax in (0, 1):
        bl = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), ax, bl)

    def psnr(x, y):
        mse = float(np.mean((x - y) ** 2))
        return 99.0 if mse <= 0 else 10 * np.log10(1.0 / mse)

    p_sp = psnr(sp, base)
    p_sp_hh = psnr(apply_variant(sp, 'hh0', 1), base)
    p_sp_sh = psnr(apply_variant(sp, 'shrink1.0', 1), base)
    p_sp_cl = psnr(apply_variant(sp, 'clip1.0', 1), base)
    chk('T5 椒盐：HH 置零使 PSNR 上升', p_sp_hh > p_sp, f'{p_sp:.2f} -> {p_sp_hh:.2f} dB')

    # T6/T8 是本脚本最重要的两条：两种退化需要【相反】的算子。
    # 这不是洁癖测试 —— 它编码了创新点 1 失败的第二个结构性原因。
    chk('T6 椒盐需要"限幅"而非"收缩"：clip 增益 > 3x shrink 增益',
        (p_sp_cl - p_sp) > 3 * max(p_sp_sh - p_sp, 1e-6),
        f'clip +{p_sp_cl-p_sp:.2f} dB vs shrink +{p_sp_sh-p_sp:.2f} dB vs hh0 +{p_sp_hh-p_sp:.2f} dB')

    p_bl = psnr(bl, base)
    p_bl_hh = psnr(apply_variant(bl, 'hh0', 1), base)
    chk('T7 高斯模糊：HH 置零几乎无改善（<0.5 dB）', abs(p_bl_hh - p_bl) < 0.5,
        f'{p_bl:.2f} -> {p_bl_hh:.2f} dB')

    p_gn = psnr(gn, base)
    p_gn_sh = psnr(apply_variant(gn, 'shrink1.0', 1), base)
    p_gn_cl = psnr(apply_variant(gn, 'clip1.0', 1), base)
    chk('T8 高斯噪声需要"收缩"而非"限幅"：shrink 增益 > clip 增益',
        (p_gn_sh - p_gn) > (p_gn_cl - p_gn),
        f'shrink +{p_gn_sh-p_gn:.2f} dB vs clip +{p_gn_cl-p_gn:.2f} dB')
    chk('T8b 两种退化的最优算子相反（交叉验证）',
        (p_sp_cl - p_sp) > (p_sp_sh - p_sp) and (p_gn_sh - p_gn) > (p_gn_cl - p_gn),
        '=> 单个"每子带一个标量"的门控无法同时覆盖两者，且无法表达逐系数选择')

    # T9 det0 == 2x2 均值下采样再最近邻放大
    d0 = apply_variant(img, 'det0', 1)
    mean2 = img.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))
    chk('T9 det0 等价于 2x2 均值池化后放大',
        np.allclose(d0, np.repeat(np.repeat(mean2, 2, 0), 2, 1), atol=1e-12))

    # T10 输出范围
    chk('T10 输出被裁到 [0,1]',
        apply_variant(sp, 'shrink2.0', 2).min() >= 0 and apply_variant(sp, 'shrink2.0', 2).max() <= 1)

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir')
    ap.add_argument('--pattern', default=r'_(PDC|GDC)_')
    ap.add_argument('--variants', nargs='+',
                    default=['id', 'hh0', 'hh50', 'det50', 'det0',
                             'clip1.0', 'clip2.0', 'shrink1.0', 'shrink2.0'])
    ap.add_argument('--levels', type=int, default=1, help='小波层数；1 = 只处理最细尺度')
    ap.add_argument('--out-root')
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only:
        sys.exit(self_test())
    if not a.img_dir or not a.out_root:
        ap.error('--img-dir 与 --out-root 必需')
    run(a)


if __name__ == '__main__':
    main()
