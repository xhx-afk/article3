#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪声记忆检查（零训练的前置判决）

问题：GDC 的中位缺陷在图像上信噪比 < 1（面积积分后仍不可检出），PDC 需要 CNN
表达不了的秩统计量。这两类样本网络【拟合不了】，但离线增强让每张图只有【一个
固定的噪声实现】，在 132 个 epoch 里被看了 132 遍 —— 这正是可以被记忆的设置。

判据：比较同一 checkpoint 在【train 划分】与【val 划分】上的分码 AP。
用 ODC 的 train-val 落差作为"普通过拟合"的基准，看 GDC/PDC 的落差超出多少：

    excess_gap(code) = [AP_train(code) - AP_val(code)] - [AP_train(ODC) - AP_val(ODC)]

excess_gap 大 => 网络对这些码有【特异性的记忆】，容量被噪声实现吃掉了。

用法:
  # 1) 打印要跑的推理命令（train 划分只需跑一次）
  python3 tools/analysis/memorization_check.py --dry-run \
      --cfg configs/deim_dfine/custom/coated_wood_s.yml \
      --ckpt ./deim_outputs/coated_wood/deim_s_640_e132_seed0/best_stg2.pth \
      --root datasets/Water-Based-Coated-Wood --out-dir prep/innov3/memcheck
  # 2) 汇总（零 GPU）
  python3 tools/analysis/memorization_check.py \
      --ann-train datasets/Water-Based-Coated-Wood/annotations/instances_train.json \
      --pred-train prep/innov3/memcheck/pred_train.json \
      --ann-val   datasets/Water-Based-Coated-Wood/annotations/instances_val.json \
      --pred-val  prep/baseline/seed0/pred_val.json \
      --out prep/innov3/memcheck/summary.json
  python3 tools/analysis/memorization_check.py --self-test-only
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from resolution_sweep import coco_ap          # 已与 effective_n 交叉验证到小数点后 14 位
except ImportError:                                # pragma: no cover
    sys.exit('需要 tools/analysis/resolution_sweep.py（提供交叉验证过的 coco_ap）')

NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')
CODES = ('ODC', 'LDC', 'DDC', 'GDC', 'PDC')


def code_of(fn):
    m = NAME_RE.match(os.path.splitext(os.path.basename(fn))[0])
    return m.group(2) if m else None


def subset(ann, dt, code):
    ids = {im['id'] for im in ann['images'] if code_of(im['file_name']) == code}
    a = {k: v for k, v in ann.items() if k not in ('images', 'annotations')}
    a['images'] = [im for im in ann['images'] if im['id'] in ids]
    a['annotations'] = [x for x in ann['annotations'] if x['image_id'] in ids]
    d = [x for x in dt if int(x['image_id']) in ids]
    return a, d


def load_pred(p):
    d = json.load(open(p, encoding='utf-8'))
    return d if isinstance(d, list) else (d.get('annotations') or d.get('detections') or [])


def commands(args):
    print("# train 划分只需跑一次全量推理（9160 张），分码由汇总脚本内部切分。")
    print(f"mkdir -p {args.out_dir}")
    print(f"python3 tools/wood/dump_predictions.py -c {args.cfg} -r {args.ckpt} \\\n"
          f"  -u val_dataloader.dataset.img_folder={args.root}/images/train \\\n"
          f"     val_dataloader.dataset.ann_file={args.root}/annotations/instances_train.json \\\n"
          f"  -o {args.out_dir}/pred_train.json")
    print(f"\n# val 侧直接复用 baseline 已有的 pred_val.json，不用重跑。")
    print(f"\npython3 tools/analysis/memorization_check.py \\\n"
          f"  --ann-train {args.root}/annotations/instances_train.json \\\n"
          f"  --pred-train {args.out_dir}/pred_train.json \\\n"
          f"  --ann-val {args.root}/annotations/instances_val.json \\\n"
          f"  --pred-val <baseline 的 pred_val.json> --out {args.out_dir}/summary.json")


def summarize(args):
    at, av = (json.load(open(p, encoding='utf-8')) for p in (args.ann_train, args.ann_val))
    dt, dv = load_pred(args.pred_train), load_pred(args.pred_val)
    rows = {}
    for c in CODES:
        a1, d1 = subset(at, dt, c)
        a2, d2 = subset(av, dv, c)
        if not a1['images'] or not a2['images']:
            continue
        r1, r2 = coco_ap(a1, d1), coco_ap(a2, d2)
        rows[c] = dict(ap_train=r1['AP'], ap_val=r2['AP'], gap=r1['AP'] - r2['AP'],
                       ap50_train=r1['AP50'], ap50_val=r2['AP50'],
                       n_train=len(a1['images']), n_val=len(a2['images']))
    if 'ODC' not in rows:
        sys.exit('缺少 ODC 子集，无法建立过拟合基准')
    base_gap = rows['ODC']['gap']
    for c, r in rows.items():
        r['excess_gap'] = r['gap'] - base_gap

    print(f"\n{'码':6s} {'train图':>8s} {'AP_train':>9s} {'AP_val':>8s} {'落差':>8s} "
          f"{'超出ODC':>9s}")
    for c in CODES:
        if c not in rows: continue
        r = rows[c]
        print(f"{c:6s} {r['n_train']:8d} {r['ap_train']:9.2f} {r['ap_val']:8.2f} "
              f"{r['gap']:+8.2f} {r['excess_gap']:+9.2f}")
    print(f"\nODC 的 train-val 落差 {base_gap:+.2f} = 普通过拟合基准；"
          f"各码的『超出ODC』才是【特异性噪声记忆】。")

    print("\n================ 判据 ================")
    verdict = {}
    for c in ('GDC', 'PDC'):
        if c not in rows: continue
        e = rows[c]['excess_gap']
        if e >= args.strong:
            lv, txt = 'STRONG', ('网络记住了该码的固定噪声实现 → 容量被浪费，'
                                 '训练消融（D1）直接开跑')
        elif e >= args.weak:
            lv, txt = 'MODERATE', '有一定记忆 → D1 仍值得跑，但预期收益较小'
        else:
            lv, txt = 'ABSENT', ('没有特异性记忆 → 损害机制要改成「梯度噪声」而非「记忆」，'
                                 'D1 照跑但论证要调整')
        verdict[c] = dict(excess_gap=e, level=lv)
        print(f"  {c}: 超出 ODC {e:+.2f} AP -> {lv}  —— {txt}")
    mx = max((verdict[c]['excess_gap'] for c in verdict), default=float('nan'))
    overall = ('STRONG' if any(v['level'] == 'STRONG' for v in verdict.values())
               else ('MODERATE' if any(v['level'] == 'MODERATE' for v in verdict.values())
                     else 'ABSENT'))
    print(f"\n  总判定：{overall}（最大超出 {mx:+.2f} AP）")
    print("  提示：这张表本身就是论文里的一张图 —— 横轴增强码，两条柱 train/val。")

    out = dict(rows=rows, base_gap=base_gap, verdict=verdict, overall=overall)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"\n写出 {args.out}")
    return out


# ---------------------------------------------------------------- 自检
def _mk(n_src, codes, box=(10, 10, 40, 40)):
    imgs, anns, iid, aid = [], [], 0, 0
    for s in range(1, n_src + 1):
        for c in codes:
            for d in range(2):
                iid += 1; aid += 1
                imgs.append({'id': iid, 'file_name': f'{s:05d}_{c}_{d}.jpg',
                             'width': 200, 'height': 200})
                anns.append({'id': aid, 'image_id': iid, 'category_id': 0,
                             'bbox': list(box), 'iscrowd': 0})
    return {'images': imgs, 'annotations': anns, 'categories': [{'id': 0, 'name': 'x'}]}


def _pred(ann, per_code_quality):
    """per_code_quality[code] in [0,1]: 1 = 完美框, 0 = 完全错的框。"""
    out = []
    for im in ann['images']:
        c = code_of(im['file_name'])
        q = per_code_quality.get(c, 1.0)
        if q <= 0:
            out.append({'image_id': im['id'], 'category_id': 0,
                        'bbox': [150, 150, 20, 20], 'score': 0.9})
        else:
            shift = (1 - q) * 40
            out.append({'image_id': im['id'], 'category_id': 0,
                        'bbox': [10 + shift, 10, 40, 40], 'score': 0.9})
    return out


def self_test():
    fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    chk('code_of 解析正确', code_of('00001_GDC_3.jpg') == 'GDC' and code_of('x.jpg') is None)

    ann = _mk(6, CODES)
    a1, d1 = subset(ann, _pred(ann, {}), 'GDC')
    chk('subset 只留该码', all(code_of(im['file_name']) == 'GDC' for im in a1['images']))
    chk('subset 图数 = 6 板 × 2 方向', len(a1['images']) == 12)
    chk('subset 的 GT 与 pred 同步', len(a1['annotations']) == 12 and len(d1) == 12)

    class A: pass
    import io, contextlib, tempfile

    def run_case(qual_train, qual_val, tag):
        at, av = _mk(6, CODES), _mk(6, CODES)
        a = A()
        tmp = tempfile.mkdtemp()
        for nm, obj in (('at', at), ('av', av)):
            json.dump(obj, open(os.path.join(tmp, nm + '.json'), 'w'))
        json.dump(_pred(at, qual_train), open(os.path.join(tmp, 'pt.json'), 'w'))
        json.dump(_pred(av, qual_val), open(os.path.join(tmp, 'pv.json'), 'w'))
        a.ann_train = os.path.join(tmp, 'at.json'); a.ann_val = os.path.join(tmp, 'av.json')
        a.pred_train = os.path.join(tmp, 'pt.json'); a.pred_val = os.path.join(tmp, 'pv.json')
        a.out = None; a.strong = 10.0; a.weak = 5.0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = summarize(a)
        return r

    # 情形 1：GDC 在 train 上完美、val 上很差 -> 特异性记忆
    r = run_case({'GDC': 1.0}, {'GDC': 0.35}, 'mem')
    chk('★ 构造出记忆时判为 STRONG', r['verdict']['GDC']['level'] == 'STRONG',
        f"excess {r['verdict']['GDC']['excess_gap']:+.1f}")

    # 情形 2：所有码 train/val 一致 -> 无记忆
    r2 = run_case({}, {}, 'nomem')
    chk('★ 无记忆时判为 ABSENT 且 excess ≈ 0',
        r2['overall'] == 'ABSENT' and abs(r2['verdict']['GDC']['excess_gap']) < 1e-6,
        f"excess {r2['verdict']['GDC']['excess_gap']:+.3f}")

    # 情形 3：所有码都过拟合同样多 -> excess 仍为 0（ODC 基准起作用）
    r3 = run_case({c: 1.0 for c in CODES}, {c: 0.35 for c in CODES}, 'uniform')
    chk('★ 全码同等过拟合时 excess ≈ 0（ODC 基准正确扣除了普通过拟合）',
        abs(r3['verdict']['GDC']['excess_gap']) < 1e-6 and r3['overall'] == 'ABSENT',
        f"ODC 落差 {r3['base_gap']:+.1f}, GDC excess {r3['verdict']['GDC']['excess_gap']:+.3f}")

    # 命令生成
    a = A(); a.cfg = 'c.yml'; a.ckpt = 'k.pth'; a.root = 'datasets/X'; a.out_dir = 'o'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        commands(a)
    t = buf.getvalue()
    chk('命令同时覆盖 img_folder 与 ann_file（两处都要改，否则读的还是 val）',
        'img_folder=datasets/X/images/train' in t and
        'ann_file=datasets/X/annotations/instances_train.json' in t)

    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--cfg'); ap.add_argument('--ckpt'); ap.add_argument('--root')
    ap.add_argument('--out-dir')
    ap.add_argument('--ann-train'); ap.add_argument('--pred-train')
    ap.add_argument('--ann-val'); ap.add_argument('--pred-val')
    ap.add_argument('--strong', type=float, default=10.0)
    ap.add_argument('--weak', type=float, default=5.0)
    ap.add_argument('--out'); ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if a.dry_run:
        if not (a.cfg and a.ckpt and a.root and a.out_dir):
            ap.error('--dry-run 需要 --cfg --ckpt --root --out-dir')
        commands(a); return
    if not all([a.ann_train, a.pred_train, a.ann_val, a.pred_val]):
        ap.error('汇总模式需要 --ann-train --pred-train --ann-val --pred-val')
    summarize(a)


if __name__ == '__main__':
    main()
