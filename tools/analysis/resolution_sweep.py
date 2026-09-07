#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分辨率扫描：创新点 2 的零训练前置判决

为什么需要它：TIDE@0.75 显示 ODC 口径 90.3% 的可恢复误差是"框不准"（Loc 29.34，
与 AP50−AP75 = 28.84 双口径互证）。创新点 2 的主张是"CCFF 的最近邻上采样丢高频，
小目标边界受损"。在投入 1.5 周之前，先用零训练的方式问一句：
    **特征细节到底是不是绑定约束？**

做法：用训练好的 baseline，在【多尺度训练覆盖过的】分辨率上评测。
generate_scales(640, 20) 给出 480…800 全是 32 的倍数，全部分布内，无 train/test 失配。
若 AP75 与 AP_small 随分辨率显著上升 → 细节是绑定约束，创新点 2 的路是活的。
若基本不动 → 分辨率不是约束，FreqFusion 也危险，必须重新找机制依据。

⚠️ 两处耦合必须同时改，改一处会静默出错：
    eval_spatial_size        （HybridEncoder 的 pos_embed 依赖它）
    val_dataloader ... Resize（实际送进网络的尺寸）
本脚本的 --dry-run 会把正确的覆盖串打出来，直接复制。

用法:
  # 1) 打印每个分辨率要跑的命令（不执行）
  python3 tools/analysis/resolution_sweep.py --dry-run \
      --cfg configs/deim_dfine/custom/coated_wood_s.yml \
      --ckpt ./deim_outputs/coated_wood/s_640_e132_seed0/best_stg2.pth \
      --out-dir prep/innov2/ressweep
  # 2) 跑完后汇总（只读 pred json，不需要 GPU）
  python3 tools/analysis/resolution_sweep.py --from-preds prep/innov2/ressweep \
      --ann prep/baseline/seed0/odc/ann.json --out prep/innov2/ressweep/summary.json
  python3 tools/analysis/resolution_sweep.py --self-test-only
"""
import argparse, glob, json, os, re, sys
import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100
AREA_RNG = {'all': (0.0, 1e10), 'small': (0.0, 32 ** 2),
            'medium': (32 ** 2, 96 ** 2), 'large': (96 ** 2, 1e10)}
# generate_scales(640, base_size_repeat=20) 的原生集合（engine/data/dataloader.py:86）
DEFAULT_SIZES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]


# ---------------------------------------------------------------- AP（含面积区间）
def iou_matrix(a, b):
    """a,b: (N,4)/(M,4) xywh -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0], a[:, 1]; ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]; bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(0, np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]))
    ih = np.maximum(0, np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]))
    inter = iw * ih
    ua = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def _match(gt_boxes, gt_ig, dt_boxes, dt_area, rng_lo, rng_hi):
    """pycocotools evaluateImg 的语义：
       - GT 排序把非忽略放前面；
       - 已匹配到非忽略 GT 后遇到忽略 GT 立即停止（不退化去匹配忽略项）；
       - 匹配到忽略 GT 的检测记 dtIg（既不算 TP 也不算 FP）；
       - 未匹配且自身面积不在区间内的检测也记 dtIg。
    """
    T = len(IOU_THRS)
    n_dt = min(len(dt_boxes), MAX_DETS)
    dt_boxes, dt_area = dt_boxes[:n_dt], dt_area[:n_dt]
    tp = np.zeros((n_dt, T), dtype=np.int64)
    dtig = np.zeros((n_dt, T), dtype=np.int64)
    if n_dt == 0:
        return tp, dtig
    order = np.argsort(gt_ig, kind='mergesort')
    gt_boxes, gt_ig = gt_boxes[order], gt_ig[order]
    ious = iou_matrix(dt_boxes, gt_boxes)
    dt_out = ~((dt_area >= rng_lo) & (dt_area <= rng_hi))
    for t, thr in enumerate(IOU_THRS):
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for d in range(n_dt):
            best, best_iou = -1, min(thr, 1 - 1e-10)
            for g in range(len(gt_boxes)):
                if matched[g]:
                    continue
                if best > -1 and gt_ig[best] == 0 and gt_ig[g] == 1:
                    break
                if ious[d, g] < best_iou:
                    continue
                best_iou, best = ious[d, g], g
            if best == -1:
                dtig[d, t] = 1 if dt_out[d] else 0
                continue
            matched[best] = True
            dtig[d, t] = gt_ig[best]
            tp[d, t] = 1 - gt_ig[best]
    return tp, dtig


def coco_ap(ann, dt, area='all', gt_subset=None):
    lo, hi = AREA_RNG[area]
    gt_by_ic, dt_by_ic = {}, {}
    for a in ann['annotations']:
        w, h = float(a['bbox'][2]), float(a['bbox'][3])
        ig = int(a.get('iscrowd', 0)) or int(a.get('ignore', 0))
        if not (lo <= w * h <= hi):
            ig = 1
        if gt_subset is not None and a['id'] not in gt_subset:
            ig = 1
        gt_by_ic.setdefault((int(a['image_id']), int(a['category_id'])), []).append(
            ([a['bbox'][0], a['bbox'][1], w, h], ig))
    for d in dt:
        dt_by_ic.setdefault((int(d['image_id']), int(d['category_id'])), []).append(d)

    cats = sorted({c for _, c in gt_by_ic} | {c for _, c in dt_by_ic})
    ap_thr = np.zeros((len(cats), len(IOU_THRS)))
    valid = np.zeros(len(cats), dtype=bool)
    for ci, cat in enumerate(cats):
        tps, dtigs, scores, npig = [], [], [], 0
        keys = {k for k in gt_by_ic if k[1] == cat} | {k for k in dt_by_ic if k[1] == cat}
        for k in sorted(keys):
            g = gt_by_ic.get(k, [])
            gb = np.array([x[0] for x in g], dtype=np.float64).reshape(-1, 4)
            gi = np.array([x[1] for x in g], dtype=np.int64)
            npig += int((gi == 0).sum())
            ds = sorted(dt_by_ic.get(k, []), key=lambda z: -float(z['score']))
            db = np.array([d['bbox'] for d in ds], dtype=np.float64).reshape(-1, 4)
            da = db[:, 2] * db[:, 3] if len(db) else np.zeros(0)
            t, ig = _match(gb, gi, db, da, lo, hi)
            n = t.shape[0]
            tps.append(t); dtigs.append(ig)
            scores.append(np.array([float(d['score']) for d in ds[:n]]))
        if npig == 0:
            continue
        valid[ci] = True
        if not tps or sum(len(s) for s in scores) == 0:
            continue
        tp = np.concatenate(tps); ig = np.concatenate(dtigs); sc = np.concatenate(scores)
        o = np.argsort(-sc, kind='mergesort')
        tp, ig = tp[o], ig[o]
        keep_tp = np.cumsum(np.logical_and(tp, 1 - ig), axis=0).astype(np.float64)
        keep_fp = np.cumsum(np.logical_and(1 - tp, 1 - ig), axis=0).astype(np.float64)
        for t in range(len(IOU_THRS)):
            rec = keep_tp[:, t] / npig
            prec = keep_tp[:, t] / np.maximum(keep_tp[:, t] + keep_fp[:, t], 1e-12)
            for i in range(len(prec) - 1, 0, -1):
                if prec[i] > prec[i - 1]:
                    prec[i - 1] = prec[i]
            idx = np.searchsorted(rec, REC_THRS, side='left')
            q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
            q[idx >= len(prec)] = 0.0
            ap_thr[ci, t] = q.mean()
    if not valid.any():
        return dict(AP=float('nan'), AP50=float('nan'), AP75=float('nan'))
    m = ap_thr[valid]
    return dict(AP=float(m.mean()) * 100, AP50=float(m[:, 0].mean()) * 100,
                AP75=float(m[:, 5].mean()) * 100)


# ---------------------------------------------------------------- 命令生成
def commands(args):
    print("# 每个分辨率两处耦合覆盖必须同时给；只改一处会静默出错。")
    print(f"mkdir -p {args.out_dir}")
    for s in args.sizes:
        ov = (f"eval_spatial_size=[{s},{s}] "
              f"\"val_dataloader.dataset.transforms.ops=[{{type: Resize, size: [{s},{s}]}}, "
              f"{{type: ConvertPILImage, dtype: float32, scale: True}}]\"")
        print(f"python3 tools/wood/dump_predictions.py -c {args.cfg} -r {args.ckpt} \\\n"
              f"  -u {ov} \\\n  -o {args.out_dir}/pred_{s}.json")
    print(f"\npython3 tools/analysis/resolution_sweep.py --from-preds {args.out_dir} \\\n"
          f"  --ann <ODC 子集的 ann.json> --out {args.out_dir}/summary.json")


# ---------------------------------------------------------------- 汇总
def summarize(args):
    ann = json.load(open(args.ann, encoding='utf-8'))
    files = sorted(glob.glob(os.path.join(args.from_preds, 'pred_*.json')),
                   key=lambda p: int(re.search(r'pred_(\d+)\.json', p).group(1)))
    if not files:
        sys.exit(f'{args.from_preds} 下没有 pred_<size>.json')
    rows = []
    for p in files:
        size = int(re.search(r'pred_(\d+)\.json', p).group(1))
        dt = json.load(open(p, encoding='utf-8'))
        if isinstance(dt, dict):
            dt = dt.get('annotations') or dt.get('detections') or []
        r = dict(size=size)
        r.update(coco_ap(ann, dt, 'all'))
        for a in ('small', 'medium', 'large'):
            r[f'AP_{a}'] = coco_ap(ann, dt, a)['AP']
        rows.append(r)
        print(f"  size {size}: AP {r['AP']:.2f}  AP50 {r['AP50']:.2f}  AP75 {r['AP75']:.2f}  "
              f"AP_s {r['AP_small']:.2f}  AP_m {r['AP_medium']:.2f}  AP_l {r['AP_large']:.2f}")

    base = next((r for r in rows if r['size'] == args.ref_size), rows[len(rows) // 2])
    top = max(rows, key=lambda r: r['size'])
    d75 = top['AP75'] - base['AP75']
    ds = top['AP_small'] - base['AP_small']
    print(f"\n参照 {base['size']} → 最高 {top['size']}：ΔAP75 = {d75:+.2f}，ΔAP_small = {ds:+.2f}")
    if d75 >= 2.0:
        v = ('通过：特征细节是绑定约束 → 创新点 2（融合路径）的路是活的，'
             'FreqFusion 值得 1.5 周；同时跑 P2 参照定标尺')
    elif d75 >= 0.5:
        v = ('边缘：先跑 P2 参照定出上界，再决定是否投入 FreqFusion')
    else:
        v = ('未通过：分辨率不是绑定约束 → 创新点 2 也危险，'
             '投入 FreqFusion 之前必须重新找机制依据')
    tag = '符合假设' if ds > d75 * 0.8 else '与假设相反：小目标不是主要受益者'
    print(f"AP_small 的涨幅 {'大于' if ds > d75 else '不大于'} 总体 AP75 涨幅 ——【{tag}】")
    print(f">>> {v}")
    out = dict(rows=rows, ref_size=base['size'], d_ap75=d75, d_ap_small=ds, verdict=v)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"写出 {args.out}")
    return out


# ---------------------------------------------------------------- 自检
def _toy(n_img=6, box=(10, 10, 40, 40), cat=0):
    ann = {'images': [{'id': i, 'file_name': f'{i}.jpg', 'width': 200, 'height': 200}
                      for i in range(n_img)],
           'annotations': [{'id': i, 'image_id': i, 'category_id': cat,
                            'bbox': list(box), 'iscrowd': 0} for i in range(n_img)],
           'categories': [{'id': cat, 'name': 'x'}]}
    dt = [{'image_id': i, 'category_id': cat, 'bbox': list(box), 'score': 0.9}
          for i in range(n_img)]
    return ann, dt


def self_test():
    fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    ann, dt = _toy()
    r = coco_ap(ann, dt)
    chk('完美预测 AP = 100', abs(r['AP'] - 100) < 1e-9, f"{r['AP']:.4f}")

    # IoU 恰为 0.62：AP=30, AP50=100, AP75=0（162/38 的精确构造）
    ann2, dt2 = _toy(box=(0, 0, 162, 162))
    for d in dt2: d['bbox'] = [38, 0, 162, 162]
    r2 = coco_ap(ann2, dt2)
    chk('IoU=0.62 -> AP 30 / AP50 100 / AP75 0',
        abs(r2['AP'] - 30) < 1e-6 and abs(r2['AP50'] - 100) < 1e-6 and r2['AP75'] < 1e-9,
        f"{r2['AP']:.3f}/{r2['AP50']:.1f}/{r2['AP75']:.1f}")

    # 漏检一半 -> AP50 = 51/101
    ann3, dt3 = _toy(n_img=8)
    r3 = coco_ap(ann3, dt3[:4])
    chk('漏检一半 -> AP50 = 51/101', abs(r3['AP50'] - 100 * 51 / 101) < 1e-6, f"{r3['AP50']:.4f}")

    # ★ COCO 的正确行为：全部 GT 已检出之后的尾部重复框【不会】拉低 AP
    #   （101 点插值下，recall 达到 1.0 时 precision 仍为 1）。
    #   《准备阶段-Agent实现文档》§5.4 自检 #4 把这条写反了 —— 见交付文档 §1.4。
    dup = dt + [dict(d, score=0.5) for d in dt]
    chk('尾部重复框不改变 AP（COCO 的正确行为）',
        abs(coco_ap(ann, dup)['AP'] - coco_ap(ann, dt)['AP']) < 1e-9,
        f"{coco_ap(ann, dup)['AP']:.4f} vs {coco_ap(ann, dt)['AP']:.4f}")
    # 排在 TP 之前的高分背景 FP 才会拉低 AP
    fp_first = [{'image_id': i, 'category_id': 0, 'bbox': [150, 150, 20, 20], 'score': 0.99}
                for i in range(3)] + dt
    chk('高分背景 FP（排在 TP 之前）拉低 AP',
        coco_ap(ann, fp_first)['AP'] < coco_ap(ann, dt)['AP'] - 1.0,
        f"{coco_ap(ann, fp_first)['AP']:.2f} < {coco_ap(ann, dt)['AP']:.2f}")

    # 面积区间：40x40=1600 属于 medium
    chk('面积区间：small 无 GT -> nan', np.isnan(coco_ap(ann, dt, 'small')['AP']))
    chk('面积区间：medium == all', abs(coco_ap(ann, dt, 'medium')['AP'] - 100) < 1e-9)

    # ★ ignore 语义：区间外的 GT 上的正确检测，不能被算成 FP
    ann4 = json.loads(json.dumps(ann))
    ann4['annotations'].append({'id': 999, 'image_id': 0, 'category_id': 0,
                                'bbox': [100, 100, 150, 150], 'iscrowd': 0})   # large
    dt4 = dt + [{'image_id': 0, 'category_id': 0, 'bbox': [100, 100, 150, 150], 'score': 0.95}]
    chk('★ 区间外 GT 上的检测既不算 TP 也不算 FP（medium 口径 AP 仍为 100）',
        abs(coco_ap(ann4, dt4, 'medium')['AP'] - 100) < 1e-9,
        f"{coco_ap(ann4, dt4, 'medium')['AP']:.4f}  （错误实现会掉到 ~83）")
    chk('★ 同一份数据在 large 口径下也是 100',
        abs(coco_ap(ann4, dt4, 'large')['AP'] - 100) < 1e-9)

    # gt_subset 与 area 正交
    chk('gt_subset 全选 == 不给 subset',
        abs(coco_ap(ann, dt, 'all', gt_subset={a['id'] for a in ann['annotations']})['AP']
            - coco_ap(ann, dt, 'all')['AP']) < 1e-12)
    sub = {0, 1, 2}
    chk('gt_subset 子集：被排除的 GT 上的检测不计 FP',
        abs(coco_ap(ann, dt, 'all', gt_subset=sub)['AP'] - 100) < 1e-9,
        f"{coco_ap(ann, dt, 'all', gt_subset=sub)['AP']:.4f}")

    # 命令生成里的两处耦合
    class A: pass
    a = A(); a.sizes = [640]; a.cfg = 'c.yml'; a.ckpt = 'k.pth'; a.out_dir = 'o'
    import io as _io, contextlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        commands(a)
    txt = buf.getvalue()
    chk('命令里同时含 eval_spatial_size 与 Resize（两处耦合）',
        'eval_spatial_size=[640,640]' in txt and 'Resize, size: [640,640]' in txt)
    chk('默认尺寸集合全是 32 的倍数且落在多尺度训练范围内',
        all(s % 32 == 0 for s in DEFAULT_SIZES) and min(DEFAULT_SIZES) == 480
        and max(DEFAULT_SIZES) == 800)

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--cfg'); ap.add_argument('--ckpt'); ap.add_argument('--out-dir')
    ap.add_argument('--sizes', type=int, nargs='+', default=DEFAULT_SIZES)
    ap.add_argument('--from-preds'); ap.add_argument('--ann')
    ap.add_argument('--ref-size', type=int, default=640)
    ap.add_argument('--out'); ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if a.dry_run:
        if not (a.cfg and a.ckpt and a.out_dir): ap.error('--dry-run 需要 --cfg --ckpt --out-dir')
        commands(a); return
    if a.from_preds:
        if not a.ann: ap.error('--from-preds 需要 --ann')
        summarize(a); return
    ap.error('给 --dry-run 或 --from-preds 之一')


if __name__ == '__main__':
    main()
