#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 eval_metrics.py 产出的 json，输出 markdown 表与 go/no-go 读数。

输入：若干个 tools/wood/eval_metrics.py 产出的 metrics json（位置参数）。
  - 文件名（去扩展名）作为 run 名，可用 `run名=path` 显式指定；
  - 含 `run0` / `baseline`（或第一个文件）作为参照系 Run-0；
  - Params(M)/GFLOPs/FPS 若 json 里有对应键（n_parameters / gflops / fps）则取，
    否则留 `-`（跑 tools/benchmark/trt_benchmark.py 补）。

用法：
  python tools/distill/collect_results.py \
    run0=./deim_outputs/neu_det/deim_s_640_e200_seed0/eval_val/metrics.json \
    runA=./deim_outputs/neu_det/deim_s_640_e200_distill_runA/eval_val/metrics.json \
    runB=./deim_outputs/neu_det/deim_s_640_e200_distill_runB/eval_val/metrics.json
"""

import argparse
import json
from pathlib import Path


COCO_KEYS = {
    'mAP': 'AP', 'AP50': 'AP50', 'AP75': 'AP75',
    'APs': 'AP_small', 'APm': 'AP_medium', 'APl': 'AP_large',
}


def load(path: str):
    if '=' in path and not Path(path).exists():
        name, _, file = path.partition('=')
    else:
        name, file = Path(path).stem, path
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
    bbox = data.get('coco_eval_bbox', {})
    row = {k: bbox.get(v) for k, v in COCO_KEYS.items()}
    row['Params(M)'] = data.get('n_parameters')
    if row['Params(M)'] is not None:
        row['Params(M)'] = row['Params(M)'] / 1e6
    row['GFLOPs'] = data.get('gflops')
    row['FPS'] = data.get('fps')
    row['_name'] = name
    return row


def fmt(v, scale=1.0, nd=1):
    return '-' if v is None else f'{v * scale:.{nd}f}'


def delta(cur, ref):
    return None if cur is None or ref is None else cur - ref


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('jsons', nargs='+', help='run名=json路径 或 json路径')
    args = ap.parse_args()
    assert args.jsons, '至少给一个 metrics json'

    rows = [load(p) for p in args.jsons]

    # 参照系：名字含 run0/baseline 优先，否则取第一个
    ref = next((r for r in rows if r['_name'].lower() in ('run0', 'baseline')), rows[0])

    header = ['run', 'mAP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
              'Params(M)', 'GFLOPs', 'FPS']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')

    def emit(row, dmap=None):
        cells = [row['_name']]
        cells += [fmt(row[k], 100 if k == 'mAP' else 1, 1 if k != 'mAP' else 2)
                  for k in COCO_KEYS]
        cells += [fmt(row['Params(M)'], 1, 2), fmt(row['GFLOPs'], 1, 1),
                  fmt(row['FPS'], 1, 1)]
        print('| ' + ' | '.join(cells) + ' |')
        if dmap:
            cells = [f"Δ({row['_name']}−{ref['_name']})"]
            cells += [fmt(dmap[k], 100, 2) for k in COCO_KEYS]
            cells += ['-', '-', '-']
            print('| ' + ' | '.join(cells) + ' |')

    for row in rows:
        if row is ref:
            emit(row)
        else:
            dmap = {k: delta(row[k], ref[k]) for k in COCO_KEYS}
            emit(row, dmap)

    # go/no-go 读数（§3.2）：Run-A 涨点 = 蒸馏这条路成立
    run_a = next((r for r in rows if 'runa' in r['_name'].lower().replace('-', '')), None)
    print('\n[go/no-go]')
    if run_a is not None:
        d = delta(run_a['mAP'], ref['mAP'])
        verdict = 'GO（蒸馏有效）' if d is not None and d > 0 else 'NO-GO / 无效'
        print(f"  Run-A mAP Δ = {fmt(d, 100, 2)} → {verdict}")
    else:
        print('  未识别 Run-A（run 名含 runA），请检查文件名')
    run_b = next((r for r in rows if 'runb' in r['_name'].lower().replace('-', '')), None)
    if run_b is not None:
        d = delta(run_b['mAP'], ref['mAP'])
        print(f"  Run-B mAP Δ = {fmt(d, 100, 2)}")
        print('  （Run-B − Run-A 的差距 = 创新点 2 频域对齐的动机，Run-C 见 M3）')


if __name__ == '__main__':
    main()
