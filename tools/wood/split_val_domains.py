"""按文件名把 COCO val/test JSON 拆成五域(ODC/LDC/DDC/GDC/PDC)子集，用于分组评估。

背景
----
``tools/wood/eval_metrics.py`` 只输出整体 COCO / 类别 / 尺寸 / 阈值 / 混淆 / 错误分解，
没有 GDC/PDC/ODC 分组与 source-level bootstrap 接口；且 ``annotations/`` 下只有
``instances_{train,val,test}.json``，没有五域子集 JSON。因此 SRFF 验收 §5.5 无法直接执行。

本工具**只读**原始 ``instances_{split}.json``，按 ``file_name`` 中的域标记拆出若干子集
COCO JSON（保留原始 image id / annotation id / categories），供 ``eval_metrics.py``
逐域评估。绝不修改原始 JSON、不改训练 split、不重标注、不改类别映射。

文件名约定（经数据核验）::

    {source_id}_{DOMAIN}_{index}.jpg
      source_id : 原始来源 ID（同源衍生样本共享，供 source-level bootstrap 以来源为单位重采样）
      DOMAIN    : 域标记（预期 ODC/LDC/DDC/GDC/PDC，工具会从数据中自动发现并报告）
      index     : 该 (来源, 域) 下的衍生序号（1..N，对应方向/衍生）

用法
----
::

    # 1) 只核验结构、打印统计，不写任何文件
    python tools/wood/split_val_domains.py \
        --ann datasets/Water-Based-Coated-Wood/annotations/instances_val.json --analyze

    # 2) 生成五域子集 JSON（写到独立目录，原始 JSON 不动）
    python tools/wood/split_val_domains.py \
        --ann datasets/Water-Based-Coated-Wood/annotations/instances_val.json \
        --out-dir datasets/Water-Based-Coated-Wood/annotations/domains

仅依赖标准库（json/os/re/argparse/collections），本地与远程都可运行。
"""

import argparse
import json
import os
import re
from collections import defaultdict

# {source}_{DOMAIN}_{index}.{ext}
FILENAME_RE = re.compile(
    r'^(?P<source>\d+)_(?P<domain>[A-Za-z0-9]+)_(?P<index>\d+)\.(?P<ext>[A-Za-z0-9]+)$'
)
EXPECTED_DOMAINS = ('ODC', 'LDC', 'DDC', 'GDC', 'PDC')


def parse_file_name(file_name):
    """从 file_name 解析 (source, domain, index)。不匹配返回 None。"""
    stem = os.path.basename(file_name)
    m = FILENAME_RE.match(stem)
    if m is None:
        return None
    return m.group('source'), m.group('domain').upper(), int(m.group('index'))


def load_coco(ann_path):
    with open(ann_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze(payload):
    """解析全部 image 的 file_name，返回结构统计与分组索引。"""
    images = payload.get('images', [])
    per_domain_images = defaultdict(list)      # domain -> [image dict]
    per_domain_sources = defaultdict(set)      # domain -> {source}
    per_domain_index = defaultdict(set)        # domain -> {index}
    per_source_domains = defaultdict(set)      # source -> {domain}
    unmatched = []
    all_sources = set()

    for img in images:
        parsed = parse_file_name(img.get('file_name', ''))
        if parsed is None:
            unmatched.append(img.get('file_name'))
            continue
        source, domain, index = parsed
        per_domain_images[domain].append(img)
        per_domain_sources[domain].add(source)
        per_domain_index[domain].add(index)
        per_source_domains[source].add(domain)
        all_sources.add(source)

    return {
        'total_images': len(images),
        'matched_images': len(images) - len(unmatched),
        'unmatched': unmatched,
        'domains': sorted(per_domain_images.keys()),
        'per_domain_images': per_domain_images,
        'per_domain_sources': per_domain_sources,
        'per_domain_index': per_domain_index,
        'per_source_domains': per_source_domains,
        'all_sources': all_sources,
    }


def print_report(stats, payload, split_name):
    categories = payload.get('categories', [])
    annotations = payload.get('annotations', [])
    print('=' * 72)
    print(f'域拆分校验报告  split={split_name}')
    print('=' * 72)
    print(f'categories      : {[(c.get("id"), c.get("name")) for c in categories]}')
    print(f'total images    : {stats["total_images"]}')
    print(f'total annotations: {len(annotations)}')
    print(f'matched images  : {stats["matched_images"]}')
    print(f'unmatched images: {len(stats["unmatched"])}')
    if stats['unmatched']:
        print('  示例(最多10个):', stats['unmatched'][:10])
    print(f'distinct sources: {len(stats["all_sources"])}')
    print(f'domains found   : {stats["domains"]}')
    missing = [d for d in EXPECTED_DOMAINS if d not in stats['domains']]
    extra = [d for d in stats['domains'] if d not in EXPECTED_DOMAINS]
    print(f'expected domains: {list(EXPECTED_DOMAINS)}  missing={missing}  extra={extra}')
    print('-' * 72)
    print(f'{"domain":8s}{"images":>8s}{"sources":>9s}{"idx_min":>9s}{"idx_max":>9s}{"img/source":>12s}')
    for d in stats['domains']:
        imgs = stats['per_domain_images'][d]
        srcs = stats['per_domain_sources'][d]
        idxs = stats['per_domain_index'][d]
        ratio = len(imgs) / len(srcs) if srcs else 0
        print(f'{d:8s}{len(imgs):>8d}{len(srcs):>9d}{min(idxs):>9d}{max(idxs):>9d}{ratio:>12.2f}')
    # 每个来源是否五域齐全
    domain_count_hist = defaultdict(int)
    for s, ds in stats['per_source_domains'].items():
        domain_count_hist[len(ds)] += 1
    print('-' * 72)
    print('每来源覆盖的域数量分布 (域数 -> 来源数):', dict(sorted(domain_count_hist.items())))


def write_subsets(payload, stats, out_dir, split_name):
    """为每个域写一个子集 COCO JSON，保留原始 id/categories；返回 {domain: path}。"""
    os.makedirs(out_dir, exist_ok=True)
    annotations = payload.get('annotations', [])
    categories = payload.get('categories', [])
    licenses = payload.get('licenses', [])
    info = payload.get('info', {})

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann.get('image_id')].append(ann)

    written = {}
    coverage_total = 0
    for d in stats['domains']:
        imgs = stats['per_domain_images'][d]
        img_ids = {im['id'] for im in imgs}
        sub_anns = [a for a in annotations if a.get('image_id') in img_ids]
        sub_info = dict(info)
        sub_info['description'] = (
            f'{info.get("description", "")} | domain-subset={d} split={split_name} '
            f'(auto-split by split_val_domains.py, read-only derived)'
        ).strip(' |')
        sub = {
            'info': sub_info,
            'licenses': licenses,
            'images': imgs,
            'annotations': sub_anns,
            'categories': categories,
        }
        out_path = os.path.join(out_dir, f'{d}_{split_name}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(sub, f, ensure_ascii=False)
        written[d] = out_path
        coverage_total += len(imgs)
        print(f'[write] {d:6s} images={len(imgs):5d} annotations={len(sub_anns):6d} -> {out_path}')

    # 覆盖性校验：所有已匹配图像必须恰好落入一个域子集
    assert coverage_total == stats['matched_images'], \
        f'覆盖不完整: 子集合计 {coverage_total} != 匹配图像 {stats["matched_images"]}'
    if stats['unmatched']:
        print(f'[warn] {len(stats["unmatched"])} 张图像 file_name 不符合约定，未写入任何子集。')

    manifest = {
        'split': split_name,
        'total_images': stats['total_images'],
        'matched_images': stats['matched_images'],
        'unmatched_count': len(stats['unmatched']),
        'distinct_sources': len(stats['all_sources']),
        'domains': {
            d: {
                'images': len(stats['per_domain_images'][d]),
                'sources': len(stats['per_domain_sources'][d]),
                'index_min': min(stats['per_domain_index'][d]),
                'index_max': max(stats['per_domain_index'][d]),
                'path': written[d],
            } for d in stats['domains']
        },
    }
    manifest_path = os.path.join(out_dir, f'domain_manifest_{split_name}.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'[write] manifest -> {manifest_path}')
    return written


def infer_split_name(ann_path):
    base = os.path.basename(ann_path)
    m = re.search(r'instances_(\w+)\.json', base)
    return m.group(1) if m else os.path.splitext(base)[0]


def main():
    parser = argparse.ArgumentParser(description='按文件名域标记拆分 COCO JSON（只读派生，不改原文件）')
    parser.add_argument('--ann', required=True, help='原始 instances_{split}.json 路径')
    parser.add_argument('--out-dir', default=None, help='子集 JSON 输出目录（不指定则只校验）')
    parser.add_argument('--analyze', action='store_true', help='只打印校验报告，不写文件')
    parser.add_argument('--split-name', default=None, help='子集命名用的 split 名，默认从文件名推断')
    args = parser.parse_args()

    split_name = args.split_name or infer_split_name(args.ann)
    payload = load_coco(args.ann)
    stats = analyze(payload)
    print_report(stats, payload, split_name)

    if args.analyze or args.out_dir is None:
        # 只校验：结构不合规时以非零码退出，便于门禁
        ok = (not stats['unmatched']) and (set(stats['domains']) >= set(EXPECTED_DOMAINS))
        print('=' * 72)
        print('ANALYZE_ONLY  structural_ok=', ok)
        raise SystemExit(0 if ok else 1)

    write_subsets(payload, stats, args.out_dir, split_name)
    print('DONE')


if __name__ == '__main__':
    main()
