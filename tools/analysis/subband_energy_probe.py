#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新点 1 · 第 0 步：子带能量探针（图像空间快筛）

回答一个花 1.5 周之前必须先回答的问题：
  「脉冲噪声在 HH 子带的能量占比异常高」—— 这条子带能量门控赖以成立的前提，
  在这份数据的真实图像上到底成不成立？在哪个尺度上最明显？
  低对比度缺陷的能量是不是真的集中在 LL/LH/HL、不会被门控误抑？

只依赖 numpy + Pillow。不需要 torch，不需要 checkpoint。
真正决定性的读数在特征空间（见 --feature-space 的说明），这里是图像空间的快筛：
若图像空间就没有信号，特征空间更不会有。

用法:
  python3 tools/analysis/subband_energy_probe.py \
      --ann datasets/Water-Based-Coated-Wood/annotations/instances_val.json \
      --img-dir datasets/Water-Based-Coated-Wood/images/val \
      --per-code 60 --out prep/innov1/subband_probe.json
  python3 tools/analysis/subband_energy_probe.py --self-test-only
"""
import argparse, json, os, re, sys
from collections import defaultdict

import numpy as np

NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')
CODES = ['ODC', 'LDC', 'DDC', 'GDC', 'PDC']
SUB = ['LL', 'LH', 'HL', 'HH']
# 正交归一 Haar（能量守恒：四个子带平方和 == 原 2x2 块平方和）
HAAR = 0.5 * np.array([[[1, 1], [1, 1]],
                       [[1, 1], [-1, -1]],
                       [[1, -1], [1, -1]],
                       [[1, -1], [-1, 1]]], dtype=np.float64)
# 三个改造落点（stage2/3/4 的 downsample）的输入分辨率 = 原图的 1/4, 1/8, 1/16
STRIDES = [1, 2, 4, 8, 16]
TARGET_STRIDES = [4, 8, 16]


def norm_id(s):
    return str(int(s))


def avg_pool(img, k):
    if k == 1:
        return img
    h, w = img.shape
    h, w = h // k * k, w // k * k
    return img[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def subbands(img):
    """一层 Haar 分解。返回 (4, H/2, W/2)。"""
    h, w = img.shape
    h, w = h // 2 * 2, w // 2 * 2
    blk = img[:h, :w].reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3)
    return np.einsum('kij,yxij->kyx', HAAR, blk)


def energy_fraction(img, mask=None):
    """返回 (LL, LH, HL, HH) 的能量占比。mask 为原尺度布尔图时，
    先降到子带尺度（2x2 块内任一为真即计入）。"""
    co = subbands(img)
    e = co ** 2
    if mask is not None:
        h, w = mask.shape
        h, w = h // 2 * 2, w // 2 * 2
        m = mask[:h, :w].reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))
        if m.sum() < 16:
            return None
        e = e[:, m]
        tot = e.sum()
        return None if tot <= 0 else e.sum(axis=1) / tot
    tot = e.sum()
    return None if tot <= 0 else e.sum(axis=(1, 2)) / tot


def to_gray(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def auc(pos, neg):
    """Mann-Whitney AUC：HH 占比能否把 PDC 与 ODC 分开。"""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = np.empty(allv.shape[0], dtype=np.float64)
    order = np.argsort(allv, kind='mergesort')
    sv = allv[order]
    i = 0
    ranks = np.empty(len(allv))
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    r[order] = ranks
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


# ------------------------------------------------------------------ 主流程
def run(args):
    ann = json.load(open(args.ann, encoding='utf-8'))
    by_img = defaultdict(list)
    for a in ann['annotations']:
        by_img[a['image_id']].append(a)
    catname = {c['id']: c['name'] for c in ann['categories']}

    # 按 (原图, 方向) 收齐 5 个码
    groups = defaultdict(dict)
    meta = {}
    for im in ann['images']:
        m = NAME_RE.match(os.path.splitext(os.path.basename(im['file_name']))[0])
        if not m:
            continue
        s, c, d = norm_id(m.group(1)), m.group(2), m.group(3)
        groups[(s, d)][c] = im
        meta[im['id']] = im
    full = [k for k, v in groups.items() if len(v) == 5]
    full.sort()
    rng = np.random.default_rng(args.seed)
    pick = [full[i] for i in rng.choice(len(full), min(args.per_code, len(full)), replace=False)]
    print(f"[probe] 可用 (原图, 方向) 组 {len(full)}，抽样 {len(pick)} 组 × 5 码 = {len(pick)*5} 张图")

    # frac[stride][code] -> list of (4,)
    frac = {s: defaultdict(list) for s in STRIDES}
    # ODC 上的 GT 框内 / 框外
    inside, outside = {s: [] for s in STRIDES}, {s: [] for s in STRIDES}
    per_cls_inside = {s: defaultdict(list) for s in STRIDES}

    for gi, key in enumerate(pick):
        for code, im in groups[key].items():
            p = os.path.join(args.img_dir, os.path.basename(im['file_name']))
            if not os.path.exists(p):
                print(f"[warn] 缺图 {p}"); continue
            g = to_gray(p)
            for s in STRIDES:
                gs = avg_pool(g, s)
                f = energy_fraction(gs)
                if f is not None:
                    frac[s][code].append(f)
            if code == 'ODC':
                H, W = g.shape
                gt = np.zeros((H, W), dtype=bool)
                boxes = []
                for a in by_img.get(im['id'], []):
                    x, y, w, h = [float(v) for v in a['bbox']]
                    x0, y0 = max(0, int(x)), max(0, int(y))
                    x1, y1 = min(W, int(np.ceil(x + w))), min(H, int(np.ceil(y + h)))
                    if x1 > x0 and y1 > y0:
                        gt[y0:y1, x0:x1] = True
                        boxes.append((x0, y0, x1, y1, a['category_id']))
                if not boxes:
                    continue
                for s in STRIDES:
                    gs, ms = avg_pool(g, s), avg_pool(gt.astype(np.float64), s) > 0.5
                    fi = energy_fraction(gs, ms)
                    fo = energy_fraction(gs, ~ms)
                    if fi is not None: inside[s].append(fi)
                    if fo is not None: outside[s].append(fo)
                    for (x0, y0, x1, y1, cid) in boxes:
                        bm = np.zeros_like(ms); 
                        bm[max(0, y0 // s):max(1, y1 // s), max(0, x0 // s):max(1, x1 // s)] = True
                        fb = energy_fraction(gs, bm)
                        if fb is not None:
                            per_cls_inside[s][catname.get(cid, str(cid))].append(fb)
        if (gi + 1) % 20 == 0:
            print(f"  ... {gi+1}/{len(pick)} 组")

    out = {'n_groups': len(pick), 'strides': {}, 'verdict': {}}
    print("\n================ 分增强码 · 子带能量占比（均值） ================")
    for s in STRIDES:
        tag = f"stride{s}" + ("  <-- 落点" if s in TARGET_STRIDES else "")
        print(f"\n-- 输入分辨率 1/{s} {tag}")
        print(f"   {'code':6s} {'LL':>9s} {'LH':>10s} {'HL':>10s} {'HH':>10s} "
              f"{'HH/(LH+HL+HH)':>14s} {'HH/ODC':>9s}")
        rec = {}
        odc_hh = np.mean([f[3] for f in frac[s]['ODC']]) if frac[s]['ODC'] else np.nan
        for c in CODES:
            if not frac[s][c]:
                continue
            m = np.mean(np.stack(frac[s][c]), axis=0)
            rec[c] = dict(zip(SUB, [float(v) for v in m]))
            rec[c]['HH_ratio_vs_ODC'] = float(m[3] / odc_hh) if odc_hh > 0 else float('nan')
            hf = float(m[3] / max(m[1] + m[2] + m[3], 1e-30))
            rec[c]['HH_share_of_highfreq'] = hf
            print(f"   {c:6s} {m[0]:9.5f} {m[1]:10.3e} {m[2]:10.3e} {m[3]:10.3e} "
                  f"{hf:14.4f} {m[3]/odc_hh:8.2f}x")
        def hh_share(fs):
            return [float(f[3] / max(f[1] + f[2] + f[3], 1e-30)) for f in fs]
        if frac[s]['PDC'] and frac[s]['ODC']:
            a = auc(hh_share(frac[s]['PDC']), hh_share(frac[s]['ODC']))
            rec['AUC_HHshare_PDC_vs_ODC'] = float(a)
            o = float(np.mean(hh_share(frac[s]['ODC'])))
            for c in CODES:
                if frac[s][c]:
                    rec.setdefault(c, {})['HHshare_ratio_vs_ODC'] = \
                        float(np.mean(hh_share(frac[s][c])) / max(o, 1e-30))
            print(f"   HH/(LH+HL+HH) 区分 PDC vs ODC 的 AUC = {a:.4f}")
        if inside[s] and outside[s]:
            mi = np.mean(np.stack(inside[s]), axis=0)
            mo = np.mean(np.stack(outside[s]), axis=0)
            rec['ODC_inside_gt'] = dict(zip(SUB, [float(v) for v in mi]))
            rec['ODC_outside_gt'] = dict(zip(SUB, [float(v) for v in mo]))
            print(f"   ODC 框内 {mi[0]:9.5f} {mi[1]:10.3e} {mi[2]:10.3e} {mi[3]:10.3e} "
                  f"{mi[3]/max(mi[1]+mi[2]+mi[3],1e-30):14.4f}")
            print(f"   ODC 框外 {mo[0]:9.5f} {mo[1]:10.3e} {mo[2]:10.3e} {mo[3]:10.3e} "
                  f"{mo[3]/max(mo[1]+mo[2]+mo[3],1e-30):14.4f}")
            rec['ODC_inside_per_class'] = {
                k: dict(zip(SUB, [float(v) for v in np.mean(np.stack(v), axis=0)]))
                for k, v in per_cls_inside[s].items() if v}
        out['strides'][f"1/{s}"] = rec

    # ---- 预注册判据 ----
    print("\n================ 判据 ================")
    verdict = {}
    for s in TARGET_STRIDES:
        r = out['strides'][f"1/{s}"]
        ratio = r.get('PDC', {}).get('HHshare_ratio_vs_ODC', float('nan'))
        a = r.get('AUC_HHshare_PDC_vs_ODC', float('nan'))
        if ratio >= 3.0 and a >= 0.90:
            lv = 'STRONG'
        elif ratio >= 1.5 and a >= 0.70:
            lv = 'WEAK'
        else:
            lv = 'ABSENT'
        verdict[f"1/{s}"] = dict(pdc_hhshare_ratio=float(ratio), auc=float(a), level=lv)
        print(f"  1/{s}: PDC 的 HH/(LH+HL+HH) = ODC 的 {ratio:.2f}x, AUC {a:.3f} -> {lv}")
    gdc = [out['strides'][f"1/{s}"].get('GDC', {}).get('HHshare_ratio_vs_ODC', float('nan'))
           for s in TARGET_STRIDES]
    print(f"  GDC 的 HH 占比比值 {['%.2f' % g for g in gdc]}"
          f"  ——【{'符合假设：模糊压低高频，门控无法靠抑制救 GDC' if np.nanmean(gdc) < 1.0 else '与假设相反：GDC 的 HH 也升高，需重新解释'}】")
    for s in TARGET_STRIDES:
        r = out['strides'][f"1/{s}"]
        if 'ODC_inside_gt' in r:
            i_, o_ = r['ODC_inside_gt'], r['ODC_outside_gt']
            osc = (i_['LH'] + i_['HL']) / max(i_['HH'], 1e-12)
            print(f"  1/{s}: 缺陷框内 (LH+HL)/HH = {osc:.1f}  "
                  f"——【{'符合假设：缺陷能量在有向子带，不会被 HH 抑制误伤' if osc >= 3 else '与假设相反：缺陷能量有相当部分落在 HH，门控有误抑风险'}】")
    out['verdict'] = verdict
    levels = [v['level'] for v in verdict.values()]
    out['overall'] = 'STRONG' if 'STRONG' in levels else ('WEAK' if 'WEAK' in levels else 'ABSENT')
    print(f"\n  总判定: {out['overall']}"
          f"   (STRONG=门控有信号可学，按计划推进 | WEAK=先做无门控的 A1 | ABSENT=停下重议动机)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"\n  写出 {args.out}")
    return out


# ------------------------------------------------------------------ 自检
def self_test():
    fails = []
    def chk(name, cond, extra=''):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        if not cond: fails.append(name)

    # 1) 能量守恒 / 正交
    X = np.random.default_rng(0).normal(size=(64, 64))
    co = subbands(X)
    chk('Haar 能量守恒', np.isclose((co ** 2).sum(), (X[:64, :64] ** 2).sum()))
    chk('Haar 行正交归一', np.allclose(HAAR.reshape(4, 4) @ HAAR.reshape(4, 4).T, np.eye(4)))

    # 2) 常数图：能量全在 LL
    f = energy_fraction(np.full((64, 64), 0.5))
    chk('常数图能量全在 LL', f[0] > 0.999, f'LL={f[0]:.4f}')

    # 3) 椒盐 vs 干净：HH 占比必须显著升高
    rng = np.random.default_rng(1)
    smooth = np.cumsum(np.cumsum(rng.normal(0, 1, (256, 256)), 0), 1); smooth /= smooth.std()
    sp = smooth.copy(); m = rng.random(smooth.shape) < 0.05
    sp[m] = rng.choice([-6.0, 6.0], m.sum())
    f_clean, f_sp = energy_fraction(smooth), energy_fraction(sp)
    chk('椒盐使 HH 占比升高 >= 10x', f_sp[3] > 10 * max(f_clean[3], 1e-9),
        f'{f_clean[3]:.3e} -> {f_sp[3]:.3e}')

    # 4) 高斯模糊：HH 占比必须下降（对应 GDC 无法靠抑制解决）
    k = np.array([1, 4, 6, 4, 1], dtype=float); k /= k.sum()
    bl = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 0, smooth)
    bl = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, bl)
    f_bl = energy_fraction(bl)
    chk('高斯模糊使 HH 占比下降', f_bl[3] < f_clean[3], f'{f_clean[3]:.3e} -> {f_bl[3]:.3e}')

    # 5) 有向边缘：能量进 LH/HL 而不是 HH（缺陷不会被误抑的依据）
    # 边缘必须落在 2x2 块内部（列 65），否则每个块都是常数、能量全进 LL
    edge = np.zeros((128, 128)); edge[:, 65:] = 1.0
    f_e = energy_fraction(edge)
    chk('竖直阶跃边缘的能量进 HL 而非 HH', f_e[2] > 20 * max(f_e[3], 1e-12),
        f'HL={f_e[2]:.4f} HH={f_e[3]:.6f}')
    # 对齐到块边界的同一条边缘：能量全进 LL（说明上面那条测的确实是"块内边缘"）
    edge2 = np.zeros((128, 128)); edge2[:, 64:] = 1.0
    chk('对齐块边界的边缘能量全在 LL', energy_fraction(edge2)[0] > 0.999)

    # 6) mask 版与全图版在全 True mask 下一致
    f_all = energy_fraction(smooth, np.ones_like(smooth, dtype=bool))
    chk('mask=全True 与无 mask 一致', np.allclose(f_all, f_clean))

    # 7) AUC 自检
    chk('AUC 完全可分 = 1.0', abs(auc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9)
    chk('AUC 完全重合 = 0.5', abs(auc([1, 2, 3], [1, 2, 3]) - 0.5) < 1e-9)

    # 8) avg_pool 尺度正确
    chk('avg_pool 缩放正确', avg_pool(np.ones((64, 64)), 8).shape == (8, 8))

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann')
    ap.add_argument('--img-dir')
    ap.add_argument('--per-code', type=int, default=60, help='抽多少个 (原图, 方向) 组')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()
    if args.self_test_only:
        sys.exit(self_test())
    if not args.ann or not args.img_dir:
        ap.error('--ann 与 --img-dir 必需（或用 --self-test-only）')
    run(args)


if __name__ == '__main__':
    main()
