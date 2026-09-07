#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两条等价性定理的可复现演示（负结果章节的核心材料，论文附录可直接引用）

定理 1（杀死创新点 1 的 A1）
  Haar 一层分解 + 每通道对四个子带做线性 1x1 融合
    == 一个 2x2 depthwise stride-2 卷积
    ⊂ 一个 3x3 depthwise stride-2 (pad=1) 卷积（即 HGNetv2 原本的下采样算子）
  ⇒ "小波下采样保留了全部信息"在线性融合下是空话：信息丢在融合那一步，
    换基不改变函数。A1 的空结果是构造性的，不含关于小波假设的信息。

定理 2（挡住创新点 2 的一个错误前置实验）
  2x 双线性上采样 == 2x 最近邻上采样 + 一个固定 3x3 卷积（核 = [[1,2,1],[2,4,2],[1,2,1]]/16）
  而 DEIM 的 CCFF 在上采样之后紧跟 fpn_blocks（含 3x3 卷积），该卷积可以吸收这个核
  ⇒ bilinear-CCFF 的函数类 ⊂ nearest-CCFF 的函数类，"把最近邻换成双线性"
    同样是 null-by-construction，不能作为创新点 2 的前置判决。

用法:
  python3 tools/analysis/equivalence_demo.py            # 打印演示表
  python3 tools/analysis/equivalence_demo.py --self-test-only
"""
import argparse, sys
import numpy as np

# 正交归一 Haar（四个子带平方和 == 原 2x2 块平方和）
HAAR = 0.5 * np.array([[[1, 1], [1, 1]],      # LL
                       [[1, 1], [-1, -1]],    # LH
                       [[1, -1], [1, -1]],    # HL
                       [[1, -1], [-1, 1]]], dtype=np.float64)
BILINEAR_K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0


# ---------------------------------------------------------------- 定理 1
def haar_then_linear(x, a):
    """A1 的算子：一层 Haar -> 每通道对四个子带线性加权求和。"""
    H, W = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    blk = x[:H, :W].reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3)
    return np.einsum('ij,yxij->yx', np.einsum('k,kij->ij', a, HAAR), blk)


def conv2x2_stride2(x, k):
    H, W = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    blk = x[:H, :W].reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3)
    return np.einsum('ij,yxij->yx', k, blk)


def conv3x3_stride2_pad1(x, k):
    """与 nn.Conv2d(C, C, 3, stride=2, padding=1, groups=C) 的单通道行为一致。"""
    xp = np.pad(x, 1)
    H, W = x.shape[0] // 2, x.shape[1] // 2
    return np.stack([[float((k * xp[2 * i:2 * i + 3, 2 * j:2 * j + 3]).sum())
                      for j in range(W)] for i in range(H)])


def project_kernel_to_haar(k2x2):
    """任意 2x2 核在 Haar 基下的坐标（Haar 是 R^4 的完备正交基）。"""
    return np.einsum('kij,ij->k', HAAR, k2x2)


# ---------------------------------------------------------------- 定理 2
def up_nearest(x):
    return np.repeat(np.repeat(x, 2, 0), 2, 1)


def up_bilinear(x):
    """align_corners=False, scale=2 —— 与 F.interpolate(mode='bilinear') 一致。"""
    H, W = x.shape
    yy = np.clip((np.arange(2 * H) + 0.5) / 2 - 0.5, 0, H - 1)
    xx = np.clip((np.arange(2 * W) + 0.5) / 2 - 0.5, 0, W - 1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy, wx = (yy - y0)[:, None], (xx - x0)[None, :]
    return (x[np.ix_(y0, x0)] * (1 - wy) * (1 - wx) + x[np.ix_(y1, x0)] * wy * (1 - wx)
            + x[np.ix_(y0, x1)] * (1 - wy) * wx + x[np.ix_(y1, x1)] * wy * wx)


def conv3x3_same(z, k):
    zp = np.pad(z, 1, mode='edge')
    out = np.zeros_like(z)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * zp[i:i + z.shape[0], j:j + z.shape[1]]
    return out


# ---------------------------------------------------------------- 演示
def demo(seed=0):
    rng = np.random.default_rng(seed)
    print("=" * 74)
    print("定理 1  Haar + 线性子带融合  ==  2x2 步长卷积  ⊂  3x3 步长卷积")
    print("=" * 74)
    x = rng.normal(size=(32, 32))
    a = rng.normal(size=4)
    y_a1 = haar_then_linear(x, a)
    k_eq = np.einsum('k,kij->ij', a, HAAR)
    e1 = np.abs(y_a1 - conv2x2_stride2(x, k_eq)).max()
    print(f"  (1a) A1 的输出 == 2x2 核 {np.round(k_eq.ravel(), 4)} 的步长卷积"
          f"        max|Δ| = {e1:.2e}")

    k_any = rng.normal(size=(2, 2))
    a_rec = project_kernel_to_haar(k_any)
    e2 = np.abs(np.einsum('k,kij->ij', a_rec, HAAR) - k_any).max()
    print(f"  (1b) 反向：任意 2x2 核都能由四个子带的线性组合复现          max|Δ| = {e2:.2e}")

    k3 = np.zeros((3, 3)); k3[1:, 1:] = k_eq
    e3 = np.abs(conv3x3_stride2_pad1(x, k3) - conv2x2_stride2(x, k_eq)).max()
    print(f"  (1c) 3x3 步长卷积把右下 2x2 设为该核即可精确复现             max|Δ| = {e3:.2e}")
    print(f"  ⇒ A1 的自由度 4 ⊂ baseline 的自由度 9；A1 的空结果是构造性的。\n")

    print("=" * 74)
    print("定理 2  双线性上采样  ==  最近邻上采样 + 固定 3x3 卷积")
    print("=" * 74)
    z = rng.normal(size=(16, 16))
    e4 = np.abs(conv3x3_same(up_nearest(z), BILINEAR_K) - up_bilinear(z)).max()
    z2 = rng.normal(size=(24, 20))
    e5 = np.abs(conv3x3_same(up_nearest(z2), BILINEAR_K) - up_bilinear(z2)).max()
    print(f"  核 = [[1,2,1],[2,4,2],[1,2,1]]/16")
    print(f"  (2a) 复现双线性（16x16）                                     max|Δ| = {e4:.2e}")
    print(f"  (2b) 同一个核换一张不同尺寸的图（24x20）                     max|Δ| = {e5:.2e}")
    print(f"  ⇒ CCFF 里上采样后紧跟的 3x3 卷积可以吸收这个核，")
    print(f"    bilinear-CCFF ⊂ nearest-CCFF —— 该对照同样 null-by-construction。\n")

    print("对论文的意义：这两条都不是经验观察，是可验证的代数事实。")
    print("凡是「能被所替换算子表示」的模块，其空结果不含信息，不能作为主实验臂。")
    return dict(t1a=e1, t1b=e2, t1c=e3, t2a=e4, t2b=e5)


def self_test():
    fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    chk('Haar 行正交归一', np.allclose(HAAR.reshape(4, 4) @ HAAR.reshape(4, 4).T, np.eye(4)))
    rng = np.random.default_rng(3)
    for s in range(5):
        r = np.random.default_rng(s)
        x, a = r.normal(size=(24, 24)), r.normal(size=4)
        k = np.einsum('k,kij->ij', a, HAAR)
        if not np.allclose(haar_then_linear(x, a), conv2x2_stride2(x, k), atol=1e-12):
            fails.append(f'定理1a seed{s}')
    chk('定理 1a（5 个随机种子）', not any(f.startswith('定理1a') for f in fails))

    k_any = rng.normal(size=(2, 2))
    chk('定理 1b 任意 2x2 核可由 Haar 基复现',
        np.allclose(np.einsum('k,kij->ij', project_kernel_to_haar(k_any), HAAR), k_any, atol=1e-12))

    x = rng.normal(size=(32, 32)); a = rng.normal(size=4)
    k_eq = np.einsum('k,kij->ij', a, HAAR)
    k3 = np.zeros((3, 3)); k3[1:, 1:] = k_eq
    chk('定理 1c 3x3 严格包含 2x2',
        np.allclose(conv3x3_stride2_pad1(x, k3), conv2x2_stride2(x, k_eq), atol=1e-12))

    # 反例守卫：一个真正用到 3x3 全部权重的核，不可能被 A1 表示
    k3full = rng.normal(size=(3, 3))
    best = min(np.abs(conv3x3_stride2_pad1(x, k3full) - haar_then_linear(x, aa)).max()
               for aa in (project_kernel_to_haar(k3full[1:, 1:]), np.zeros(4)))
    chk('反例：一般 3x3 核无法被 A1 表示（包含关系是真子集）', best > 1e-3,
        f'最小残差 {best:.3f}')

    for sh in ((16, 16), (24, 20), (8, 32)):
        z = rng.normal(size=sh)
        if not np.allclose(conv3x3_same(up_nearest(z), BILINEAR_K), up_bilinear(z), atol=1e-12):
            fails.append(f'定理2 {sh}')
    chk('定理 2（三种尺寸）', not any(f.startswith('定理2') for f in fails))

    # 反例守卫：最近邻本身不等于双线性（否则定理 2 是平凡的）
    z = rng.normal(size=(16, 16))
    chk('反例：最近邻本身 != 双线性（定理 2 非平凡）',
        np.abs(up_nearest(z) - up_bilinear(z)).max() > 0.1,
        f'max|Δ| {np.abs(up_nearest(z)-up_bilinear(z)).max():.3f}')

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {sorted(set(fails))}')
    return 1 if fails else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    sys.exit(self_test() if a.self_test_only else (demo(a.seed) and 0))
