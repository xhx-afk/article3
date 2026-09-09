#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D4 测试时增强 + 多种子融合（零训练，直接产出改进后的 AP）

为什么是这个：
  §3.1 实测「定位误差是纯方差、零偏置」（bias^2 仅占 MSE 3.4%）。
  方差主导的误差，标准解法就是【平均】—— 而这份数据天然给了 8 个 D4 视图，
  模型也在全部 8 个方向上训练过，所以 8 个视图的预测是同一个量的 8 次近独立采样。
  平均它们，方差最多降 8 倍 => IoU 上升 => 【AP75 的涨幅应大于 AP50】。
  这条预测找了六轮，这里第一次可以零训练地检验。

  同理，三个种子的预测也可以融合（种子间 sd 0.65）。

两个模式：
  --make-views  生成 8 个 D4 视图的图像目录（文件名不变，沿用同一份 ann.json，
                因为 orig_size 取自实际图像 w,h = image.size，不读 ann 的 width/height）
  --fuse        把各视图/各种子的 pred.json 映射回原始坐标系并做 WBF 融合，可直接评测

用法:
  python3 tools/analysis/tta_fuse.py --make-views \
      --img-dir datasets/Water-Based-Coated-Wood/images/val \
      --out-root prep/innov5/views
  # 对每个视图目录跑一次 dump_predictions（ann.json 不变），然后：
  python3 tools/analysis/tta_fuse.py --fuse --ann <ODC 子集 ann.json> \
      --inputs identity:pred_identity.json rot90:pred_rot90.json ... \
      --out prep/innov5/pred_tta.json --eval
  # 多种子融合（视图都填 identity）：
  python3 tools/analysis/tta_fuse.py --fuse --ann <ann> \
      --inputs identity:s0.json identity:s1.json identity:s2.json --out e.json --eval
  python3 tools/analysis/tta_fuse.py --self-test-only
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIEWS = ('identity', 'rot90', 'rot180', 'rot270',
         'flipH', 'flipV', 'transpose', 'anti_transpose')
# g 的逆：把视图坐标映射回原图坐标时用
INV = {'identity': 'identity', 'rot90': 'rot270', 'rot180': 'rot180',
       'rot270': 'rot90', 'flipH': 'flipH', 'flipV': 'flipV',
       'transpose': 'transpose', 'anti_transpose': 'anti_transpose'}


# ---------------------------------------------------------------- 图像变换
def img_transform(a, name):
    """a: (H, W, 3) numpy。与下面的 box_transform 必须严格对应（自检 T4 逐像素验）。"""
    if name == 'identity':       return a
    if name == 'rot90':          return np.rot90(a, k=-1)          # 顺时针
    if name == 'rot180':         return np.rot90(a, k=2)
    if name == 'rot270':         return np.rot90(a, k=1)           # 逆时针
    if name == 'flipH':          return a[:, ::-1]
    if name == 'flipV':          return a[::-1, :]
    if name == 'transpose':      return a.transpose(1, 0, 2) if a.ndim == 3 else a.T
    if name == 'anti_transpose':
        t = a.transpose(1, 0, 2) if a.ndim == 3 else a.T
        return t[::-1, ::-1]
    raise ValueError(name)


def box_transform(b, name, W, H):
    """b = [x, y, w, h]（xywh），(W, H) 是【变换前】图像的宽高。返回 (新框, 新尺寸)。"""
    x, y, w, h = b
    if name == 'identity':       return [x, y, w, h], (W, H)
    if name == 'rot90':          return [H - y - h, x, h, w], (H, W)
    if name == 'rot180':         return [W - x - w, H - y - h, w, h], (W, H)
    if name == 'rot270':         return [y, W - x - w, h, w], (H, W)
    if name == 'flipH':          return [W - x - w, y, w, h], (W, H)
    if name == 'flipV':          return [x, H - y - h, w, h], (W, H)
    if name == 'transpose':      return [y, x, h, w], (H, W)
    if name == 'anti_transpose': return [H - y - h, W - x - w, h, w], (H, W)
    raise ValueError(name)


def view_size(W, H, name):
    return (H, W) if name in ('rot90', 'rot270', 'transpose', 'anti_transpose') else (W, H)


# ---------------------------------------------------------------- WBF
def _iou_1_to_n(box, arr):
    if len(arr) == 0:
        return np.zeros(0)
    bx2, by2 = box[0] + box[2], box[1] + box[3]
    ax2, ay2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    iw = np.maximum(0, np.minimum(bx2, ax2) - np.maximum(box[0], arr[:, 0]))
    ih = np.maximum(0, np.minimum(by2, ay2) - np.maximum(box[1], arr[:, 1]))
    inter = iw * ih
    ua = box[2] * box[3] + arr[:, 2] * arr[:, 3] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def wbf(boxes, scores, n_models, iou_thr=0.55, score_mode='avg_scaled', midx=None):
    """Weighted Box Fusion（原始 WBF 语义：**每路只贡献一个代表框**）。

    boxes: (N,4) xywh；scores: (N,)；midx: (N,) 每个框来自哪一路（0..n_models-1）。
    midx=None 时按输入顺序轮流分配路（等价于“每框一路”，供自检的简单调用）。

    簇坐标 = 各路代表框（该路在簇内分数最高者）的分数加权平均
             —— 这就是「对方差主导的误差做平均」。
    簇分数 = Σ_路 rep_r / n_models  （avg_scaled：缺席路 rep=0 → 只在少数视图
             出现的框被降权，TTA 抑制假阳的机制）
           = Σ_路 rep_r / 出现路数   （avg：种子融合用，不惩罚缺席）

    为何必须按路取代表而不是对簇内所有框取算术均值：DEIM 每图输出 300 个
    query，同一目标周围会有多个低分冗余框挤进同一簇；算术均值会把 0.9 的 TP
    稀释到 0.1 量级，PR 曲线排序错乱，AP 崩塌（实测 −15 AP）。
    """
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    nm = max(int(n_models), 1)
    if midx is None:
        midx = np.arange(len(boxes)) % nm
    midx = np.asarray(midx, dtype=np.int64)
    order = np.argsort(-scores, kind='mergesort')
    boxes, scores, midx = boxes[order], scores[order], midx[order]

    cl_box = np.zeros((0, 4))          # 各簇当前的融合框
    members = []                       # 各簇：{model_idx: (score, box)} 只留该路最高分
    for b, s, m in zip(boxes, scores, midx):
        if len(cl_box):
            ious = _iou_1_to_n(b, cl_box)
            k = int(np.argmax(ious))
            if ious[k] >= iou_thr:
                d = members[k]
                m = int(m)
                if m not in d or s > d[m][0]:
                    d[m] = (float(s), b)
                mb = np.array([v[1] for v in d.values()], dtype=np.float64)
                ms = np.array([v[0] for v in d.values()], dtype=np.float64)
                cl_box[k] = (mb * ms[:, None]).sum(0) / max(ms.sum(), 1e-12)
                continue
        members.append({int(m): (float(s), b)})
        cl_box = np.concatenate([cl_box, b[None]], 0)

    out_b, out_s = [], []
    for k, d in enumerate(members):
        rep = np.zeros(nm, dtype=np.float64)     # 每路的代表分（缺席 = 0）
        for m, (s, _) in d.items():
            if 0 <= m < nm:
                rep[m] = max(rep[m], s)
        if score_mode == 'avg_scaled':
            s_out = float(rep.sum() / nm)
        else:
            present = int((rep > 0).sum())
            s_out = float(rep.sum() / present) if present else 0.0
        out_b.append(cl_box[k]); out_s.append(s_out)
    return np.asarray(out_b), np.asarray(out_s)


# ---------------------------------------------------------------- 模式
def make_views(args):
    from PIL import Image
    files = [f for f in os.listdir(args.img_dir)
             if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    print(f"[tta] {len(files)} 张图 -> {len(args.views)} 个视图")
    for v in args.views:
        out = os.path.join(args.out_root, v)
        os.makedirs(out, exist_ok=True)
        for f in files:
            src = os.path.abspath(os.path.join(args.img_dir, f))
            dst = os.path.join(out, f)          # 文件名不变，沿用同一份 ann.json
            if v == 'identity':
                if os.path.lexists(dst): os.remove(dst)
                os.symlink(src, dst)            # 恒等视图直接软链，逐字节相同
                continue
            a = np.asarray(Image.open(src).convert('RGB'))
            t = np.ascontiguousarray(img_transform(a, v))
            pil = Image.fromarray(t)
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg'):
                pil.save(dst, quality=100, subsampling=0)
            else:
                pil.save(dst)
        print(f"  [{v:14s}] -> {out}")
    print("\n对每个目录跑一次 dump_predictions（-u val_dataloader.dataset.img_folder=<目录>，"
          "\nann_file 不用改：orig_size 取自实际图像），再用 --fuse 融合。")
    print("恒等视图是软链 -> 它的 AP 必须与 baseline 逐位相同，这是一条免费的正确性校验。")


def load_pred(p):
    d = json.load(open(p, encoding='utf-8'))
    return d if isinstance(d, list) else (d.get('annotations') or d.get('detections') or [])


def fuse(args):
    ann = json.load(open(args.ann, encoding='utf-8'))
    size = {im['id']: (im['width'], im['height']) for im in ann['images']}
    keep_ids = set(size)

    per = defaultdict(lambda: defaultdict(list))     # image_id -> cat -> [(box, score, model_idx)]
    n_models = len(args.inputs)
    for mi, spec in enumerate(args.inputs):
        v, path = spec.split(':', 1)
        assert v in VIEWS, f'未知视图 {v}'
        dets = load_pred(path)
        # 每个 (图, 类) 只保留前 topk，控制融合的复杂度（COCO AP 的 maxDets 是 100）
        bucket = defaultdict(list)
        for d in dets:
            iid = int(d['image_id'])
            if iid in keep_ids:
                bucket[(iid, int(d['category_id']))].append(d)
        n_in = 0
        for (iid, c), ds in bucket.items():
            ds.sort(key=lambda z: -float(z['score']))
            W, H = size[iid]
            vw, vh = view_size(W, H, v)          # 该视图下图像的实际尺寸
            for d in ds[:args.topk]:
                # 把视图坐标映射【回】原图坐标：用 g 的逆，尺寸用视图的尺寸
                nb, _ = box_transform(list(d['bbox']), INV[v], vw, vh)
                per[iid][c].append((nb, float(d['score']), mi))
                n_in += 1
        print(f"  [{v:14s}] {path}  纳入 {n_in} 个框")

    out = []
    for iid, cats in per.items():
        for c, lst in cats.items():
            b, s = wbf([x[0] for x in lst], [x[1] for x in lst],
                       n_models, args.iou_thr, args.score_mode,
                       midx=[x[2] for x in lst])
            for bb, ss in zip(b, s):
                out.append({'image_id': int(iid), 'category_id': int(c),
                            'bbox': [round(float(v_), 2) for v_ in bb],
                            'score': round(float(ss), 5)})
    print(f"[tta] 融合后 {len(out)} 个框（{n_models} 路输入，IoU 阈 {args.iou_thr}）")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'))
        print(f"[tta] 写出 {args.out}")

    if args.eval:
        from resolution_sweep import coco_ap
        r = coco_ap(ann, out)
        print(f"\n融合后：AP {r['AP']:.2f}  AP50 {r['AP50']:.2f}  AP75 {r['AP75']:.2f}")
        if args.baseline_pred:
            b = coco_ap(ann, load_pred(args.baseline_pred))
            dA, d50, d75 = r['AP'] - b['AP'], r['AP50'] - b['AP50'], r['AP75'] - b['AP75']
            print(f"基准    ：AP {b['AP']:.2f}  AP50 {b['AP50']:.2f}  AP75 {b['AP75']:.2f}")
            print(f"Δ       ：AP {dA:+.2f}  AP50 {d50:+.2f}  AP75 {d75:+.2f}")
            tag = ('符合假设：方差主导的误差被平均掉，高 IoU 段受益更多'
                   if d75 > d50 else '与假设相反：AP75 涨幅未超过 AP50')
            print(f"\n【{tag}】")
            print("  找了六轮的那条可证伪预测（ΔAP75 > ΔAP50），这里第一次零训练地检验。")
    return out


# ---------------------------------------------------------------- 自检
def self_test():
    rng = np.random.default_rng(0); fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    # T1 逆元正确：g 后接 g^-1 = 恒等（框）
    W, H, b = 200, 120, [30.0, 20.0, 40.0, 25.0]
    ok = True
    for v in VIEWS:
        nb, sz = box_transform(list(b), v, W, H)
        bb, sz2 = box_transform(nb, INV[v], sz[0], sz[1])
        if not (np.allclose(bb, b) and sz2 == (W, H)): ok = False
    chk('T1 框：g 后接 g^-1 回到原位（8/8）', ok)

    # T2 图像变换的逆同样正确
    a = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    ok = all(np.array_equal(img_transform(img_transform(a, v), INV[v]), a) for v in VIEWS)
    chk('T2 图像：g 后接 g⁻¹ 逐像素回到原图（8/8）', ok)

    # ★ T3 图像变换与框变换严格对应（最关键的一条）
    ok, detail = True, []
    for v in VIEWS:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        x, y, w, h = 30, 20, 40, 25
        canvas[y:y + h, x:x + w] = 255
        t = img_transform(canvas, v)
        ys, xs = np.where(t[..., 0] > 0)
        got = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
        exp, sz = box_transform([x, y, w, h], v, W, H)
        if not (np.allclose(got, exp) and t.shape[:2] == (sz[1], sz[0])):
            ok = False; detail.append(f'{v}: got {got} exp {exp}')
    chk('★ T3 图像变换与框变换逐像素一致（8/8）', ok, '; '.join(detail))

    # T4 WBF：完全相同的 N 路输入 -> 框不变、分数不变
    boxes = [[10., 10., 40., 30.]] * 8
    fb, fs = wbf(boxes, [0.8] * 8, 8)
    chk('T4 WBF：8 路完全一致 -> 框与分数都不变',
        len(fb) == 1 and np.allclose(fb[0], boxes[0]) and abs(fs[0] - 0.8) < 1e-9)

    # T5 只在 1/8 视图里出现 -> 分数降到 1/8
    fb, fs = wbf([[10., 10., 40., 30.]], [0.8], 8)
    chk('T5 WBF：只出现在 1/8 路 -> 分数 ×1/8（TTA 抑制假阳的机制）',
        abs(fs[0] - 0.8 / 8) < 1e-9, f'{fs[0]:.4f}')

    # ★ T6 坐标平均：对称抖动的框融合后回到中心（方差被平均掉）
    base = np.array([100., 100., 50., 40.])
    jit = [base + np.array([d, 0, 0, 0]) for d in (-4, -2, 0, 2, 4)]
    fb, fs = wbf([list(j) for j in jit], [0.9] * 5, 5, iou_thr=0.5)
    chk('★ T6 WBF：对称抖动被平均回中心（这就是「方差主导 -> 平均有效」）',
        len(fb) == 1 and abs(fb[0][0] - 100.0) < 1e-6, f'x={fb[0][0]:.4f}')

    # T7 两个远离的框不会被并簇
    fb, fs = wbf([[0., 0., 20., 20.], [500., 500., 20., 20.]], [0.9, 0.8], 2)
    chk('T7 WBF：不相交的框保持为两个簇', len(fb) == 2)

    # ★ T8 端到端：抖动的多视图预测融合后 IoU 提升
    gt = np.array([100., 100., 50., 40.])
    def iou(a, b):
        ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
        iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
        it = iw * ih
        return it / (a[2]*a[3] + b[2]*b[3] - it)
    r = np.random.default_rng(1)
    noisy = [gt + r.normal(0, 3.0, 4) for _ in range(8)]
    fb, _ = wbf([list(n) for n in noisy], [0.9] * 8, 8, iou_thr=0.5)
    single = float(np.mean([iou(n, gt) for n in noisy]))
    fused = iou(fb[0], gt)
    chk('★ T8 端到端：8 视图融合后的 IoU 高于单视图均值',
        fused > single, f'单视图均值 {single:.4f} -> 融合 {fused:.4f}')

    # T9 topk 截断不改变高分框
    chk('T9 视图尺寸推导正确',
        view_size(200, 120, 'rot90') == (120, 200) and
        view_size(200, 120, 'flipH') == (200, 120))

    # T10 恒等视图是真恒等
    nb, sz = box_transform(list(b), 'identity', W, H)
    chk('T10 identity 是真恒等', np.allclose(nb, b) and sz == (W, H))

    # ★ T11 回归锁：同一路的多个低分冗余框挤进同簇时，高分 TP 不得被稀释
    #   （旧实现用簇内算术均值 + min(簇大小, n_models) 截断，在这里把 0.9
    #     稀释到 0.1 量级，实测导致 AP 崩塌 −15）
    tb = [[10., 10., 40., 30.]] + [[10.5, 10.5, 40., 30.]] * 5 + \
         [[10., 10., 40., 30.], [10., 10., 40., 30.]]
    ts = [0.90] + [0.001] * 5 + [0.85, 0.88]
    tm = [0] * 6 + [1, 2]                      # 路0 贡献 6 个框，路1/2 各 1 个
    fb, fs = wbf(tb, ts, 3, iou_thr=0.55, score_mode='avg_scaled', midx=tm)
    exp = (0.90 + 0.85 + 0.88) / 3.0
    chk('★ T11 同路低分冗余框不稀释融合分数（回归锁）',
        len(fb) == 1 and abs(fs[0] - exp) < 1e-9,
        f'got {fs[0]:.4f}, expect {exp:.4f} (旧实现会给 ~0.17)')

    # ★ T12 单路融合等价：n_models=1 时高分框分数保持原值（不被自己的冗余框拉低）
    fb1, fs1 = wbf([[10., 10., 40., 30.], [10.4, 10.4, 40., 30.], [10.2, 10.2, 40., 30.]],
                   [0.9, 0.02, 0.01], 1, iou_thr=0.5, score_mode='avg_scaled',
                   midx=[0, 0, 0])
    chk('★ T12 单路融合：高分框分数不被同路冗余框稀释',
        len(fb1) == 1 and abs(fs1[0] - 0.9) < 1e-9, f'{fs1}')

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-views', action='store_true')
    ap.add_argument('--img-dir'); ap.add_argument('--out-root')
    ap.add_argument('--views', nargs='+', default=list(VIEWS))
    ap.add_argument('--fuse', action='store_true')
    ap.add_argument('--ann'); ap.add_argument('--inputs', nargs='+',
                                              help='<view>:<pred.json>，可给多个')
    ap.add_argument('--iou-thr', type=float, default=0.55)
    ap.add_argument('--score-mode', default='avg_scaled', choices=['avg_scaled', 'avg'])
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--out'); ap.add_argument('--eval', action='store_true')
    ap.add_argument('--baseline-pred', help='用于算 Δ 的单视图基准 pred.json')
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if a.make_views:
        if not (a.img_dir and a.out_root): ap.error('--make-views 需要 --img-dir --out-root')
        make_views(a); return
    if a.fuse:
        if not (a.ann and a.inputs): ap.error('--fuse 需要 --ann --inputs')
        fuse(a); return
    ap.error('给 --make-views 或 --fuse 之一')


if __name__ == '__main__':
    main()
