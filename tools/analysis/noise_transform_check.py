#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OfflineEquivalentNoise 的 N1–N6 自检（SNR 增强准则文档 §2.4）。

N6 是关键：它保证 online（transform）与 offline（数据集的 MATLAB 增强）两臂的
噪声强度在 Lab ΔE 量纲上真的相同（相差 < 5%），否则 D0 vs D3 的机制对照不成立。

需要在服务器环境跑（依赖 torchvision.transforms.v2）：
    python3 tools/analysis/noise_transform_check.py --self-test-only
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aug_snr_audit as asa  # noqa: E402  (srgb_to_lab / sigma_de)


def run_checks(base_level=0.45, sigma=0.5, density=0.2, img_dir=None):
    from engine.data.transforms import OfflineEquivalentNoise
    from engine.data._misc import Image, BoundingBoxes

    print('[noise-transform-check] N1–N6')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    H = W = 64
    x0 = torch.full((1, 3, H, W), base_level, dtype=torch.float32)

    # N1 p=0 -> 逐位相同（且返回原对象）
    tf0 = OfflineEquivalentNoise(sigma=sigma, density=density,
                                 p_gaussian=0.0, p_salt_pepper=0.0)
    y = tf0(Image(x0.clone()))
    check('N1 p_gaussian=p_salt_pepper=0 -> 输出与输入逐位相同',
          torch.equal(torch.as_tensor(y), x0))

    # N2 p_gaussian=1 -> std 增加、值域仍在 [0,1]
    tfg = OfflineEquivalentNoise(sigma=sigma, p_gaussian=1.0, p_salt_pepper=0.0)
    yg = torch.as_tensor(tfg(Image(x0.clone()))).float()
    check('N2 p_gaussian=1: std 增加且值域 [0,1]',
          float(yg.std()) > 0.1 and float(yg.min()) >= 0.0
          and float(yg.max()) <= 1.0,
          f'std={float(yg.std()):.3f} min={float(yg.min()):.3f} '
          f'max={float(yg.max()):.3f}')

    # N3 p_salt_pepper=1, density=0.2 -> 改动比例 ≈ 20%±2%，改后值只有 0/1
    tfs = OfflineEquivalentNoise(density=density, p_gaussian=0.0,
                                 p_salt_pepper=1.0)
    ys = torch.as_tensor(tfs(Image(x0.clone()))).float()
    changed = (ys != x0)
    frac = float(changed.float().mean())
    vals = ys[changed]
    only01 = bool(((vals == 0.0) | (vals == 1.0)).all()) if vals.numel() else False
    check('N3 椒盐: 改动比例 ≈ density 且改后值只有 0/1',
          abs(frac - density) < 0.02 and only01,
          f'frac={frac:.4f} only01={only01}')

    # N4 同一张图连续两次前向输出不同（重采样而非固定实现）
    tfr = OfflineEquivalentNoise(sigma=sigma, p_gaussian=1.0, p_salt_pepper=0.0)
    ya = torch.as_tensor(tfr(Image(x0.clone()))).float()
    yb = torch.as_tensor(tfr(Image(x0.clone()))).float()
    check('N4 连续两次前向输出不同（在线重采样）', not torch.equal(ya, yb))

    # N5 box 不被改动
    boxes = BoundingBoxes(
        torch.tensor([[10.0, 10.0, 40.0, 50.0], [5.0, 5.0, 20.0, 20.0]]),
        format='xyxy', canvas_size=(H, W))
    tfmix = OfflineEquivalentNoise(sigma=sigma, density=density,
                                   p_gaussian=1.0, p_salt_pepper=0.0)
    out_b = tfmix(boxes)
    check('N5 BoundingBoxes 前后逐位相同',
          torch.equal(torch.as_tensor(out_b).float(),
                      torch.as_tensor(boxes).float()))

    # N6 在线 GDC 的 σ(ΔE) 与 aug_snr_audit 的离线换算相差 < 5%
    base_samples = np.array([base_level])
    if img_dir:
        real = asa.sample_base_levels(img_dir)
        if real is not None and len(real):
            base_samples = real
    ref = asa.sigma_de('GDC', base_samples, {'gauss_sigma': sigma},
                       np.random.default_rng(0), n=200000)
    px = []
    tf6 = OfflineEquivalentNoise(sigma=sigma, p_gaussian=1.0, p_salt_pepper=0.0)
    big = torch.full((1, 3, 256, 256), float(np.median(base_samples)),
                     dtype=torch.float32)
    for _ in range(4):
        y6 = torch.as_tensor(tf6(Image(big.clone()))).float()
        px.append(y6.numpy().reshape(-1, 3))
    px = np.concatenate(px, 0).astype(np.float64)
    lab_d = asa.srgb_to_lab(px[:, None, :])
    lab_0 = asa.srgb_to_lab(np.full((1, 1, 3), float(np.median(base_samples))))
    de = np.sqrt(((lab_d - lab_0) ** 2).sum(-1)).ravel()
    sig_online = float(np.sqrt((de ** 2).mean()))
    rel = abs(sig_online - ref['sigma_de']) / max(ref['sigma_de'], 1e-9)
    check('N6 在线 σ(ΔE) 与离线换算相差 < 5%', rel < 0.05,
          f'online={sig_online:.2f} offline={ref["sigma_de"]:.2f} '
          f'rel={rel * 100:.2f}%')
    print(f'  [info] N6: online σ(ΔE)={sig_online:.2f}, '
          f'aug_snr_audit σ(ΔE)={ref["sigma_de"]:.2f} '
          f'(文档参照值 ≈78.7，随底色分布变化)')

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-level', type=float, default=0.45)
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--density', type=float, default=0.2)
    ap.add_argument('--img-dir', default=None,
                    help='可选：真实 ODC 图目录（N6 用真实底色分布）')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()
    sys.exit(run_checks(args.base_level, args.sigma, args.density, args.img_dir))


if __name__ == '__main__':
    main()
