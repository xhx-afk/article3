#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 TTA 集成的框精化【训练集】标注（Step 2，只改 ann.json，零代码改动）

动机：§3.1 实测定位误差是【纯方差、零偏置】（bias² 仅占 MSE 3.4%），
把观测到的 IoU 换算成像素约 2.3~2.7 px/边。而 D4-TTA 的融合框是同一个量的
8 次采样的加权平均 —— 方差更低。用它去精化训练标注，等于给回归头一个
噪声更小的目标。

三条硬保护（都在代码里，不许绕过）：
  1. 路径含 val/test 直接拒绝执行 —— 评测标注一个字节不改；
  2. GT 实例【只替换坐标，不增不减】—— 数量、类别、image_id 全部保持；
  3. 只有当同类教师框与原 GT 的 IoU ≥ --iou-thr 且分数 ≥ --score-thr 才替换，
     否则保留原 GT（防止把模型的错误写进标注）。

用法:
  python3 tools/analysis/refine_labels.py \
    --ann datasets/Water-Based-Coated-Wood/annotations/instances_train.json \
    --pred prep/innov5/train/pred_tta.json \
    --out  datasets/CoatedWood_Refined/annotations/instances_train.json
  python3 tools/analysis/refine_labels.py --self-test-only
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np


def guard_train_only(*paths):
    for p in paths:
        if not p: continue
        low = os.path.normpath(p).lower().replace('\\', '/')
        for bad in ('/val', '/test', 'instances_val', 'instances_test', '_val', '_test'):
            if bad in low:
                sys.exit(f"拒绝执行：路径 {p} 看起来指向评测集。本脚本只允许改训练标注。")


def iou_1_to_n(box, arr):
    if len(arr) == 0: return np.zeros(0)
    b = np.asarray(box, dtype=np.float64); a = np.asarray(arr, dtype=np.float64).reshape(-1, 4)
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    iw = np.maximum(0, np.minimum(bx2, ax2) - np.maximum(b[0], a[:, 0]))
    ih = np.maximum(0, np.minimum(by2, ay2) - np.maximum(b[1], a[:, 1]))
    inter = iw * ih
    ua = b[2] * b[3] + a[:, 2] * a[:, 3] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def refine(ann, dets, iou_thr=0.7, score_thr=0.5, alpha=1.0):
    """alpha: 1.0 = 完全用教师框；0.5 = 与原 GT 取中点（更保守）。"""
    by_ic = defaultdict(list)
    for d in dets:
        if float(d['score']) >= score_thr:
            by_ic[(int(d['image_id']), int(d['category_id']))].append(d)
    used = defaultdict(set)
    out = json.loads(json.dumps({k: v for k, v in ann.items() if k != 'annotations'}))
    new_anns, n_rep, deltas = [], 0, []
    for a in ann['annotations']:
        key = (int(a['image_id']), int(a['category_id']))
        cand = by_ic.get(key, [])
        na = dict(a)
        if cand:
            boxes = [c['bbox'] for c in cand]
            ious = iou_1_to_n(a['bbox'], boxes)
            order = np.argsort(-ious)
            for k in order:
                if int(k) in used[key]:      # 一个教师框只用于一个 GT
                    continue
                if ious[k] >= iou_thr:
                    used[key].add(int(k))
                    tb = np.asarray(boxes[k], dtype=np.float64)
                    ob = np.asarray(a['bbox'], dtype=np.float64)
                    nb = alpha * tb + (1 - alpha) * ob
                    deltas.append(float(np.abs(nb - ob).mean()))
                    na['bbox'] = [round(float(v), 2) for v in nb]
                    na['area'] = float(nb[2] * nb[3])
                    n_rep += 1
                break
        new_anns.append(na)
    out['annotations'] = new_anns
    stats = dict(n_gt=len(ann['annotations']), n_replaced=n_rep,
                 frac_replaced=n_rep / max(len(ann['annotations']), 1),
                 mean_shift_px=float(np.mean(deltas)) if deltas else 0.0,
                 p95_shift_px=float(np.percentile(deltas, 95)) if deltas else 0.0)
    return out, stats


def run(args):
    guard_train_only(args.ann, args.out)
    ann = json.load(open(args.ann, encoding='utf-8'))
    dets = json.load(open(args.pred, encoding='utf-8'))
    if isinstance(dets, dict):
        dets = dets.get('annotations') or dets.get('detections') or []
    out, st = refine(ann, dets, args.iou_thr, args.score_thr, args.alpha)

    assert len(out['annotations']) == len(ann['annotations']), 'GT 数量被改变了'
    for a, b in zip(ann['annotations'], out['annotations']):
        assert a['id'] == b['id'] and a['image_id'] == b['image_id'] \
            and a['category_id'] == b['category_id'], 'GT 身份被改变了'
    print(f"[refine] GT {st['n_gt']} 个，替换 {st['n_replaced']} 个"
          f"（{st['frac_replaced']*100:.1f}%）")
    print(f"[refine] 每边平均移动 {st['mean_shift_px']:.2f} px，p95 {st['p95_shift_px']:.2f} px")
    print(f"[refine] ✓ GT 数量 / id / 类别 / image_id 全部未变 —— 只改了坐标")
    if st['frac_replaced'] < 0.3:
        print("  ⚠️ 替换率偏低（<30%）：多数 GT 没有匹配上高分教师框，"
              "精化的作用面很小，预期收益有限")
    if st['mean_shift_px'] > 6:
        print("  ⚠️ 平均移动 >6px：教师框与标注差得远，可能引入系统偏差，"
              "建议用 --alpha 0.5 取中点，或提高 --iou-thr")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f"[refine] 写出 {args.out}")
        print("  图像目录不用动：新 ann.json 与原图像目录配套使用即可。")
    if args.stats:
        json.dump(st, open(args.stats, 'w', encoding='utf-8'), indent=2)
    return st


def self_test():
    fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    ann = {'images': [{'id': i, 'file_name': f'{i}.jpg', 'width': 200, 'height': 200}
                      for i in range(1, 4)],
           'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0,
                            'bbox': [10., 10., 40., 30.], 'area': 1200., 'iscrowd': 0},
                           {'id': 2, 'image_id': 2, 'category_id': 1,
                            'bbox': [50., 50., 20., 20.], 'area': 400., 'iscrowd': 0},
                           {'id': 3, 'image_id': 3, 'category_id': 0,
                            'bbox': [80., 80., 30., 30.], 'area': 900., 'iscrowd': 0}],
           'categories': [{'id': 0, 'name': 'a'}, {'id': 1, 'name': 'b'}]}

    # 高 IoU 教师框 -> 替换
    dets = [{'image_id': 1, 'category_id': 0, 'bbox': [12., 11., 39., 30.], 'score': 0.9}]
    out, st = refine(ann, dets)
    chk('高 IoU 教师框被采用', st['n_replaced'] == 1 and out['annotations'][0]['bbox'][0] == 12.0)
    chk('GT 数量不变', len(out['annotations']) == 3)
    chk('未匹配的 GT 原样保留', out['annotations'][1]['bbox'] == [50., 50., 20., 20.])
    chk('area 被同步更新', abs(out['annotations'][0]['area'] - 39 * 30) < 1e-6)

    # 低 IoU -> 不替换（防止把错误写进标注）
    out2, st2 = refine(ann, [{'image_id': 1, 'category_id': 0,
                              'bbox': [100., 100., 40., 30.], 'score': 0.9}])
    chk('★ 低 IoU 的教师框被拒绝（不把模型的错误写进标注）', st2['n_replaced'] == 0)

    # 低分 -> 不替换
    out3, st3 = refine(ann, [dict(dets[0], score=0.2)])
    chk('低分教师框被拒绝', st3['n_replaced'] == 0)

    # 类别必须一致
    out4, st4 = refine(ann, [dict(dets[0], category_id=1)])
    chk('类别不同的教师框不参与', st4['n_replaced'] == 0)

    # alpha 取中点
    out5, st5 = refine(ann, dets, alpha=0.5)
    chk('alpha=0.5 取中点', abs(out5['annotations'][0]['bbox'][0] - 11.0) < 1e-9,
        f"{out5['annotations'][0]['bbox'][0]}")

    # 一个教师框不会被两个 GT 共用
    ann2 = json.loads(json.dumps(ann))
    ann2['annotations'].append({'id': 4, 'image_id': 1, 'category_id': 0,
                                'bbox': [11., 11., 40., 30.], 'area': 1200., 'iscrowd': 0})
    out6, st6 = refine(ann2, dets)
    chk('★ 一个教师框只用于一个 GT（不重复消费）', st6['n_replaced'] == 1)

    # 身份字段不被改
    chk('id / image_id / category_id 全部不变',
        all(a['id'] == b['id'] and a['image_id'] == b['image_id']
            and a['category_id'] == b['category_id']
            for a, b in zip(ann['annotations'], out['annotations'])))

    # 移动量统计
    chk('平均移动量计算正确', abs(st['mean_shift_px'] - np.mean([2, 1, 1, 0])) < 1e-9,
        f"{st['mean_shift_px']:.4f}")

    # val 保护
    import subprocess
    r = subprocess.run([sys.executable, __file__, '--ann', '/x/instances_val.json',
                        '--pred', '/x/p.json', '--out', '/y/o.json'],
                       capture_output=True, text=True)
    chk('★ 指向 val 的路径被拒绝执行',
        r.returncode != 0 and '拒绝执行' in r.stdout + r.stderr)

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann'); ap.add_argument('--pred'); ap.add_argument('--out')
    ap.add_argument('--stats')
    ap.add_argument('--iou-thr', type=float, default=0.7)
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=1.0,
                    help='1.0=完全用教师框，0.5=与原GT取中点（更保守）')
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if not (a.ann and a.pred): ap.error('--ann 与 --pred 必需')
    run(a)


if __name__ == '__main__':
    main()
