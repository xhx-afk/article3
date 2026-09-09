#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 SNR 判据重构【训练集】（val / test 一动不动）

用途：aug_snr_audit.py 判出哪些增强码的样本在图像上没有证据支撑（标签噪声），
本脚本据此生成一个只含保留样本的训练目录 + ann.json，并打印迭代对齐后的配置生成命令。

⚠️ 这【不是】改数据集划分：
   - 哪些板子在 train / val / test —— 一块不改；
   - val 与 test 的 ann.json 与图像目录 —— 一个字节不碰；
   - 只改训练集内部用哪些增强副本，且判据是可解析计算的，不是按 AP 搜出来的。
   脚本内置硬性保护：输入路径里出现 val/test 直接拒绝执行。

用法:
  python3 tools/analysis/build_snr_subset.py \
    --ann  datasets/Water-Based-Coated-Wood/annotations/instances_train.json \
    --img-dir datasets/Water-Based-Coated-Wood/images/train \
    --keep-codes ODC LDC DDC \
    --out-root datasets/CoatedWood_SNR/D1
  # 或按 aug_snr_audit 的保留表
  python3 tools/analysis/build_snr_subset.py ... --keep-table prep/innov3/train_keep.json
  python3 tools/analysis/build_snr_subset.py --self-test-only
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

NAME_RE = re.compile(r'^(\d+)_([A-Z]{3})_(\d+)$')
CODES = ('ODC', 'LDC', 'DDC', 'GDC', 'PDC')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def norm_id(s): return str(int(s))


def guard_train_only(*paths):
    """防止误改评测集。"""
    for p in paths:
        if not p: continue
        low = os.path.normpath(p).lower().replace('\\', '/')
        for bad in ('/val', '/test', 'instances_val', 'instances_test', '_val', '_test'):
            if bad in low:
                sys.exit(f"拒绝执行：路径 {p} 看起来指向评测集。本脚本只允许操作训练集。")


def parse(fn):
    return NAME_RE.match(os.path.splitext(os.path.basename(fn))[0])


def build(ann, keep_fn):
    imgs = [im for im in ann['images'] if keep_fn(im['file_name'])]
    ids = {im['id'] for im in imgs}
    anns = [a for a in ann['annotations'] if a['image_id'] in ids]
    out = {k: v for k, v in ann.items() if k not in ('images', 'annotations')}
    out['images'], out['annotations'] = imgs, anns
    return out


def summarize(ann):
    per, srcs = Counter(), defaultdict(set)
    for im in ann['images']:
        m = parse(im['file_name'])
        if not m: continue
        per[m.group(2)] += 1
        srcs[m.group(2)].add(norm_id(m.group(1)))
    allsrc = set().union(*srcs.values()) if srcs else set()
    return per, allsrc


def run(args):
    guard_train_only(args.ann, args.img_dir, args.out_root)
    ann = json.load(open(args.ann, encoding='utf-8'))
    per0, src0 = summarize(ann)
    print(f"[build] 输入训练集 {len(ann['images'])} 图 / {len(ann['annotations'])} GT / "
          f"{len(src0)} 块原图")
    print(f"        分码 {dict(per0)}")

    if args.keep_table:
        tbl = json.load(open(args.keep_table, encoding='utf-8'))
        keepset = {k for k, v in tbl['images'].items() if v.get('keep')}
        keep_fn = lambda fn: fn in keepset or os.path.basename(fn) in keepset
        rule = f"keep-table {args.keep_table}（bad_codes={tbl.get('bad_codes')}）"
    else:
        keep = set(args.keep_codes) if args.keep_codes else set(CODES) - set(args.drop_codes or [])
        keep_fn = lambda fn: (parse(fn) is not None and parse(fn).group(2) in keep)
        rule = f"保留码 {sorted(keep)}"
    print(f"[build] 规则：{rule}")

    sub = build(ann, keep_fn)
    per1, src1 = summarize(sub)
    print(f"[build] 输出 {len(sub['images'])} 图 / {len(sub['annotations'])} GT / "
          f"{len(src1)} 块原图")
    print(f"        分码 {dict(per1)}")

    if not sub['images']:
        sys.exit('保留后为空，检查规则')
    missing = src0 - src1
    if missing:
        sys.exit(f"拒绝执行：有 {len(missing)} 块原图被整块剔除（示例 {sorted(missing)[:5]}）。"
                 f"\n这会改变有效样本量，等同于改划分。规则必须对每块板保留至少一个副本。")
    print(f"[build] ✓ 原图集合未变（{len(src1)} 块），有效样本量不变 —— 这不是改划分")

    img_out = os.path.join(args.out_root, 'images', 'train')
    ann_out = os.path.join(args.out_root, 'annotations', 'instances_train.json')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(os.path.dirname(ann_out), exist_ok=True)
    n_link = 0
    for im in sub['images']:
        b = os.path.basename(im['file_name'])
        src = os.path.abspath(os.path.join(args.img_dir, b))
        dst = os.path.join(img_out, b)
        if not os.path.exists(src):
            print(f"[warn] 缺图 {src}"); continue
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(src, dst); n_link += 1
    json.dump(sub, open(ann_out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"[build] 软链 {n_link} 张 -> {img_out}")
    print(f"[build] 写出 {ann_out}")

    # 评测侧：直接软链原始的 val/test，确保逐字节相同
    root_in = os.path.dirname(os.path.dirname(os.path.abspath(args.img_dir)))
    for split in ('val', 'test'):
        s_img = os.path.join(root_in, 'images', split)
        s_ann = os.path.join(root_in, 'annotations', f'instances_{split}.json')
        if os.path.isdir(s_img):
            d = os.path.join(args.out_root, 'images', split)
            if os.path.lexists(d): os.remove(d) if os.path.islink(d) else None
            if not os.path.exists(d): os.symlink(os.path.abspath(s_img), d)
        if os.path.exists(s_ann):
            d = os.path.join(args.out_root, 'annotations', f'instances_{split}.json')
            if os.path.lexists(d): os.remove(d) if os.path.islink(d) else None
            if not os.path.exists(d): os.symlink(os.path.abspath(s_ann), d)
    print(f"[build] val / test 直接软链原始文件（逐字节相同，评测口径不变）")

    n_keep, n_full = len(sub['images']), len(ann['images'])
    ep = int(round(args.ref_epoches * n_full / max(n_keep, 1)))
    print(f"\n迭代对齐（与现状 {args.ref_epoches} epoch × {n_full} 张 同样的梯度步数）：")
    print(f"  python3 tools/make_dataset_config.py --model s --name {args.name} \\\n"
          f"    --num-classes 4 --train-size {n_keep} --epoches {ep} \\\n"
          f"    --data-root {os.path.dirname(os.path.abspath(args.out_root))} "
          f"--subdir {os.path.basename(args.out_root)} --seed 0")
    print(f"\n同 epoch 对照（次要读数，样本少 -> 梯度步数也少）：--epoches {args.ref_epoches}")
    return sub


# ---------------------------------------------------------------- 自检
def self_test():
    import shutil, tempfile
    fails = []
    def chk(n, c, e=''):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        if not c: fails.append(n)

    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, 'images', 'train'); os.makedirs(img_dir)
    imgs, anns, iid, aid = [], [], 0, 0
    for s in range(1, 6):
        for c in CODES:
            for d in range(8):
                iid += 1
                fn = f'{s:05d}_{c}_{d}.jpg'
                open(os.path.join(img_dir, fn), 'wb').write(b'x' * (10 + iid))
                imgs.append({'id': iid, 'file_name': fn, 'width': 100, 'height': 100})
                for _ in range(2):
                    aid += 1
                    anns.append({'id': aid, 'image_id': iid, 'category_id': 0,
                                 'bbox': [1, 2, 3, 4], 'iscrowd': 0})
    ann = {'images': imgs, 'annotations': anns,
           'categories': [{'id': i, 'name': n} for i, n in
                          enumerate(['blister', 'crack', 'hole', 'scratch'])]}
    ann_p = os.path.join(tmp, 'annotations', 'instances_train.json')
    os.makedirs(os.path.dirname(ann_p)); json.dump(ann, open(ann_p, 'w'))

    sub = build(ann, lambda fn: parse(fn).group(2) in {'ODC', 'LDC', 'DDC'})
    chk('过滤后图数 = 5 板 × 3 码 × 8 方向 = 120', len(sub['images']) == 120,
        str(len(sub['images'])))
    chk('GT 同步过滤（每图 2 个）', len(sub['annotations']) == 240)
    kept_ids = {im['id'] for im in sub['images']}
    chk('没有孤儿 GT', all(a['image_id'] in kept_ids for a in sub['annotations']))
    chk('categories 原样保留', sub['categories'] == ann['categories'])
    per1, src1 = summarize(sub)
    chk('原图集合未变（5 块）', len(src1) == 5)
    chk('保留的码正确', set(per1) == {'ODC', 'LDC', 'DDC'})

    chk('只保留 ODC 也不丢板子', len(summarize(build(ann, lambda fn: parse(fn).group(2) == 'ODC'))[1]) == 5)

    # 会整块丢板子的规则必须被拒
    bad = build(ann, lambda fn: norm_id(parse(fn).group(1)) in {'1', '2'})
    _, src_bad = summarize(bad)
    chk('★ 会整块剔除板子的规则可被检出（src 集合缩小）', len(src_bad) < 5,
        f'{len(src_bad)} < 5')

    # val/test 保护
    import subprocess
    r = subprocess.run([sys.executable, __file__, '--ann',
                        os.path.join(tmp, 'annotations', 'instances_val.json'),
                        '--img-dir', os.path.join(tmp, 'images', 'val'),
                        '--out-root', os.path.join(tmp, 'o'), '--keep-codes', 'ODC'],
                       capture_output=True, text=True)
    chk('★ 指向 val 的路径被拒绝执行', r.returncode != 0 and '拒绝执行' in r.stdout + r.stderr)

    # 端到端
    out_root = os.path.join(tmp, 'D1')
    class A: pass
    a = A(); a.ann = ann_p; a.img_dir = img_dir; a.out_root = out_root
    a.keep_codes = ['ODC', 'LDC', 'DDC']; a.drop_codes = None; a.keep_table = None
    a.ref_epoches = 132; a.name = 'coated_wood_d1'
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(a)
    txt = buf.getvalue()
    o_img = os.path.join(out_root, 'images', 'train')
    chk('端到端：输出目录图数正确', len(os.listdir(o_img)) == 120, str(len(os.listdir(o_img))))
    one = os.listdir(o_img)[0]
    chk('端到端：输出是软链且指回原图',
        os.path.islink(os.path.join(o_img, one)) and
        os.path.realpath(os.path.join(o_img, one)) == os.path.realpath(os.path.join(img_dir, one)))
    chk('端到端：ann 与目录一致',
        len(json.load(open(os.path.join(out_root, 'annotations', 'instances_train.json')))['images'])
        == len(os.listdir(o_img)))
    chk('端到端：打印了迭代对齐的 epoches',
        '--epoches 220' in txt, '5 板×3码×8向=120，132×200/120=220')
    chk('端到端：提示了原图集合未变', '这不是改划分' in txt)

    # 幂等
    with contextlib.redirect_stdout(io.StringIO()):
        run(a)
    chk('重复执行幂等（软链被覆盖而不是报错）', len(os.listdir(o_img)) == 120)

    shutil.rmtree(tmp, ignore_errors=True)
    print('\nALL SELF-TESTS PASSED' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann'); ap.add_argument('--img-dir'); ap.add_argument('--out-root')
    ap.add_argument('--keep-codes', nargs='+'); ap.add_argument('--drop-codes', nargs='+')
    ap.add_argument('--keep-table')
    ap.add_argument('--ref-epoches', type=int, default=132)
    ap.add_argument('--name', default='coated_wood_snr')
    ap.add_argument('--self-test-only', action='store_true')
    a = ap.parse_args()
    if a.self_test_only: sys.exit(self_test())
    if not (a.ann and a.img_dir and a.out_root):
        ap.error('--ann --img-dir --out-root 必需')
    run(a)


if __name__ == '__main__':
    main()
