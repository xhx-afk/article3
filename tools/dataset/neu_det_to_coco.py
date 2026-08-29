#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NEU-DET (VOC 格式) → COCO 格式转换（M2 用户手册第 2 节调用）。

职责：
- 递归扫描解压目录下所有 *.xml（VOC 格式）与同名图片，不假设任何固定层级；
- 类别表固定为 ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']，
  顺序写死并落进 json 的 categories，保证多次生成、多个方法之间类别 id 一致；
- 按类别分层划分（NEU-DET 每类恰好 300 张、每张图只含单一缺陷类），默认 8:2 → 1440 / 360，
  与 tools/make_dataset_config.py 的 neu_det 预设 train_size=1440 对齐；
- 划分用 --seed 固定，且把划分结果同时写一份 split.json（图片文件名列表）以便复现与审计；
- 输出目录结构：
    <dst>/
    ├── raw/                                 解压出的原始文件（脚本只读）
    ├── images/train/*.jpg                   （复制或软链，--link 控制）
    ├── images/val/*.jpg
    ├── annotations/instances_train.json
    ├── annotations/instances_val.json
    └── split.json
- 结尾打印自检：train/val 张数、每类 bbox 数、bbox 宽高分位数、train∩val 文件名交集
  必须为空；另外统计并打印 area < 1024 px²（COCO small 口径）的实例数与占比。
  不要预设"NEU-DET 没有 small 目标"——这个数字决定主表里 APs 一列是报 -1 还是真值。

用法：
  python tools/dataset/neu_det_to_coco.py \
    --src /data/defect/NEU-DET/raw \
    --dst /data/defect/NEU-DET \
    --val-ratio 0.2 --seed 0 --link
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

# 顺序写死：类别 id = 索引 + 1，跨方法跨实验绝不漂移
CLASSES = ['crazing', 'inclusion', 'patches',
           'pitted_surface', 'rolled-in_scale', 'scratches']

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def find_image(stem: str, candidates: list):
    """在候选里找与 xml 同名的图片。"""
    for p in candidates:
        if p.stem == stem and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def _parse_xml_hardened(xml_path: Path):
    """拒绝带 DTD/实体的 XML，防实体膨胀与外部实体注入。"""
    with xml_path.open('rb') as f:
        data = f.read()
    low = data[:4096].lower()
    if b'<!doctype' in low or b'<!entity' in low:
        raise ValueError(f'{xml_path}: 含 DOCTYPE/ENTITY 声明，拒绝解析（不可信 XML）')
    return ET.fromstring(data)


def parse_voc_xml(xml_path: Path):
    """返回 (width, height, [(name, xmin, ymin, xmax, ymax), ...])"""
    root = _parse_xml_hardened(xml_path)
    size = root.find('size')
    w = int(float(size.find('width').text))
    h = int(float(size.find('height').text))
    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text.strip()
        b = obj.find('bndbox')
        boxes.append((name,
                      float(b.find('xmin').text), float(b.find('ymin').text),
                      float(b.find('xmax').text), float(b.find('ymax').text)))
    return w, h, boxes


def build_coco(records, image_id_map, out_path: Path):
    """records: [(filename, width, height, boxes), ...]，id 从 1 连续编号。"""
    images, annotations = [], []
    ann_id = 1
    for img_id, (filename, width, height, boxes) in enumerate(records, start=1):
        image_id_map[filename] = img_id
        images.append({'id': img_id, 'file_name': filename,
                       'width': width, 'height': height})
        for name, xmin, ymin, xmax, ymax in boxes:
            xmin, ymin = max(0.0, xmin), max(0.0, ymin)
            xmax, ymax = min(float(width), xmax), min(float(height), ymax)
            bw, bh = xmax - xmin, ymax - ymin
            if bw <= 0 or bh <= 0:
                print(f'    [warn] {filename}: 非法 bbox 被跳过 ({bw:.1f}x{bh:.1f})')
                continue
            annotations.append({
                'id': ann_id, 'image_id': img_id,
                'category_id': CLASSES.index(name) + 1,
                'bbox': [xmin, ymin, bw, bh], 'area': bw * bh,
                'iscrowd': 0,
            })
            ann_id += 1
    coco = {
        'images': images, 'annotations': annotations,
        'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(CLASSES)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(coco, f)
    return coco


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, help='解压出的原始目录（只读）')
    ap.add_argument('--dst', required=True, help='输出根目录')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--link', action='store_true',
                    help='用软链代替复制（Windows 无权限时自动退回复制）')
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    assert src.is_dir(), f'{src} 不存在'
    assert 0.0 < args.val_ratio < 1.0, '--val-ratio 必须在 (0,1)'

    # ---- 扫描：xml 与同名图片配对 ----
    xmls = sorted(src.rglob('*.xml'))
    assert xmls, f'{src} 下没找到任何 *.xml'
    img_candidates = sorted(p for p in src.rglob('*')
                            if p.suffix.lower() in IMG_EXTS)
    img_by_stem = defaultdict(list)
    for p in img_candidates:
        img_by_stem[p.stem].append(p)

    records = []            # (filename, w, h, boxes)
    img_class = {}          # filename -> 主类别（分层划分用）
    skipped = []
    for xml in xmls:
        stem = xml.stem
        img = find_image(stem, img_by_stem.get(stem, []))
        if img is None:
            skipped.append(xml.name)
            continue
        w, h, boxes = parse_voc_xml(xml)
        if not boxes:
            skipped.append(xml.name)
            continue
        filename = stem + img.suffix.lower()
        records.append((filename, w, h, boxes))
        # NEU-DET 每张图只含单一缺陷类；万一不是，取第一个类做分层键
        img_class[filename] = boxes[0][0]
    assert records, '没有配对成功的样本，检查图片与 xml 是否同名同目录'

    # ---- 按类别分层划分（固定 seed）----
    rng = random.Random(args.seed)
    by_class = defaultdict(list)
    for rec in records:
        by_class[img_class[rec[0]]].append(rec)

    train_recs, val_recs = [], []
    for name in sorted(by_class):
        items = sorted(by_class[name])          # 先排序保证 seed 可复现
        rng.shuffle(items)
        n_val = int(round(len(items) * args.val_ratio))
        val_recs.extend(items[:n_val])
        train_recs.extend(items[n_val:])

    # ---- 落盘 ----
    for sub in ('images/train', 'images/val'):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    def materialize(recs, subset):
        for filename, *_ in recs:
            src_img = find_image(Path(filename).stem, img_by_stem[Path(filename).stem])
            target = dst / 'images' / subset / filename
            if args.link:
                try:
                    if not target.exists():
                        os.symlink(src_img.resolve(), target)
                    continue
                except OSError:
                    pass                        # 无软链权限（常见于 Windows）→ 复制
            shutil.copy2(src_img, target)

    materialize(train_recs, 'train')
    materialize(val_recs, 'val')

    id_map = {}
    train_coco = build_coco(train_recs, id_map, dst / 'annotations' / 'instances_train.json')
    val_coco = build_coco(val_recs, id_map, dst / 'annotations' / 'instances_val.json')

    split = {
        'seed': args.seed,
        'val_ratio': args.val_ratio,
        'train': [r[0] for r in train_recs],
        'val': [r[0] for r in val_recs],
    }
    with (dst / 'split.json').open('w', encoding='utf-8') as f:
        json.dump(split, f, indent=2)

    # ---- 自检 ----
    print('=' * 62)
    print(f'[self-check] train={len(train_recs)} val={len(val_recs)} '
          f'(ratio={len(val_recs) / (len(train_recs) + len(val_recs)):.3f})')
    if skipped:
        print(f'[self-check] 未配对/空标注跳过 {len(skipped)} 个 xml: '
              f'{skipped[:5]}{"..." if len(skipped) > 5 else ""}')

    for split_name, coco in (('train', train_coco), ('val', val_coco)):
        cnt = Counter(CLASSES[a['category_id'] - 1] for a in coco['annotations'])
        print(f'[self-check] {split_name} 每类 bbox 数: '
              + ', '.join(f'{c}={cnt.get(c, 0)}' for c in CLASSES))

    def report_sizes(coco, split_name):
        ws = [a['bbox'][2] for a in coco['annotations']]
        hs = [a['bbox'][3] for a in coco['annotations']]
        areas = np.asarray([a['area'] for a in coco['annotations']])
        if not areas.size:
            return
        q = lambda v: [round(float(np.percentile(v, p)), 1) for p in (5, 25, 50, 75, 95)]
        n_small = int((areas < 1024).sum())
        print(f'[self-check] {split_name} bbox 宽分位数[5/25/50/75/95]: {q(ws)} px')
        print(f'[self-check] {split_name} bbox 高分位数[5/25/50/75/95]: {q(hs)} px')
        print(f'[self-check] {split_name} area<1024px² (COCO small) 实例: '
              f'{n_small}/{areas.size} ({n_small / areas.size * 100:.2f}%) —— '
              f'该数字决定主表 APs 列报真值还是 -1，以实测为准')

    report_sizes(train_coco, 'train')
    report_sizes(val_coco, 'val')

    overlap = set(split['train']) & set(split['val'])
    print(f'[self-check] train∩val 文件名交集: {len(overlap)} 个')
    assert not overlap, 'train/val 泄漏！划分逻辑有 bug，禁止使用本次输出'
    print('[self-check] OK —— 类别顺序已写入 json categories，split 已固化到 split.json')


if __name__ == '__main__':
    main()
