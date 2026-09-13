"""SRFF-V1.2 机制诊断（实现文档 §5.3 / 用户手册 §10.2）。

对全量五域 val，hook V1.2 adapter 的 ``_core``，按 domain/class/source 汇总：
stationarity score 与 global gate、cv/w_g/w_p、delta_g_rms/delta_p_rms/混合前后 RMS、
budget_scale 与 clamp_fraction、adapter effect 的相对特征变化、非有限计数与样本数审计。

输出::
    srff_v12_checkpoint_diagnostics.json
    domain_summary.csv / class_summary.csv / source_summary.csv

单进程运行；gate_mode 默认 auto。torch.load 走 solver 既有 resume 路径（用户自己的可信 checkpoint）。
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.solver import TASKS  # noqa: E402

DOMAINS = ['ODC', 'LDC', 'DDC', 'GDC', 'PDC']
ADAPTER_CLS = 'FrozenBaseEvidenceConditionedResidualAdapter'
QUANTS = ['global_score', 'global_gate', 'cv', 'w_g', 'w_p', 'delta_g_rms', 'delta_p_rms',
          'delta_rms_pre', 'delta_rms_post', 'budget_scale', 'low_rms', 'adapter_effect']


def _per_image_rms(t):
    tf = t.float()
    return (tf.square().mean(dim=(1, 2, 3)) + 1e-12).sqrt()   # (B,)


def _find_adapter(model):
    for m in model.modules():
        if type(m).__name__ == ADAPTER_CLS:
            return m
    return None


def _source_of(file_name):
    """source 启发式：优先父目录名，否则文件名主干去掉最后一段数字/后缀。可按实际数据调整。"""
    p = Path(file_name)
    if p.parent.name and p.parent.name not in ('.', 'val', 'images'):
        return p.parent.name
    stem = p.stem
    for sep in ('_', '-'):
        if sep in stem:
            return stem.rsplit(sep, 1)[0]
    return stem


def _load_domain_map(domain_dir, tpl):
    dom_of = {}
    for d in DOMAINS:
        p = os.path.join(domain_dir, tpl.format(dom=d))
        if not os.path.isfile(p):
            continue
        j = json.load(open(p, encoding='utf-8'))
        for im in j.get('images', []):
            dom_of[int(im['id'])] = d
    return dom_of


def _load_val_meta(ann_file):
    j = json.load(open(ann_file, encoding='utf-8'))
    cats = {int(c['id']): c.get('name', str(c['id'])) for c in j.get('categories', [])}
    img_file = {int(im['id']): im.get('file_name', '') for im in j.get('images', [])}
    img_classes = defaultdict(set)
    for a in j.get('annotations', []):
        cid = int(a.get('category_id', -1))
        if cid in cats:
            img_classes[int(a['image_id'])].add(cats[cid])
    return img_file, img_classes, len(j.get('images', []))


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.2 机制诊断')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--gate-mode', default='auto', choices=['auto', 'force_off', 'force_on'])
    ap.add_argument('--full-val', action='store_true')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--domain-ann-dir', default=None)
    ap.add_argument('--domain-tpl', default='{dom}_val.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    cfg = YAMLConfig(args.config, resume=args.checkpoint, device=str(device))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    adapter = _find_adapter(model)
    if adapter is None:
        raise SystemExit('[inspect-v12] 未找到 V1.2 adapter')
    adapter.gate_mode = args.gate_mode

    val_ann = cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']
    domain_dir = args.domain_ann_dir or str(Path(val_ann).parent / 'domains')
    dom_of = _load_domain_map(domain_dir, args.domain_tpl)
    img_file, img_classes, n_val = _load_val_meta(val_ann)

    # hook adapter._core 采集每图量
    captured = {}
    orig_core = adapter._core

    def patched(high, low):
        out = orig_core(high, low)
        if out.get('ran_experts', True) and 'global_gate' in out:
            b = low.shape[0]
            gg = out['global_gate'].reshape(-1)[:b].detach().float().cpu()
            rec = {
                'global_score': out['global_score'].reshape(-1)[:b].detach().float().cpu(),
                'global_gate': gg,
                'cv': out['cv'].reshape(-1)[:b].detach().float().cpu(),
                'w_g': out['w_g'].reshape(-1)[:b].detach().float().cpu(),
                'w_p': out['w_p'].reshape(-1)[:b].detach().float().cpu(),
                'delta_g_rms': _per_image_rms(out['delta_g']).cpu(),
                'delta_p_rms': _per_image_rms(out['delta_p']).cpu(),
                'delta_rms_pre': out['delta_rms_pre'].reshape(-1)[:b].detach().float().cpu(),
                'delta_rms_post': out['delta_rms_post'].reshape(-1)[:b].detach().float().cpu(),
                'budget_scale': out['budget_scale'].reshape(-1)[:b].detach().float().cpu(),
                'low_rms': out['low_rms'].reshape(-1)[:b].detach().float().cpu(),
                'adapter_effect': ((out['low_out'] - low).abs().mean(dim=(1, 2, 3)) /
                                   (low.abs().mean(dim=(1, 2, 3)) + 1e-6)).detach().float().cpu(),
            }
            rec['clamp'] = (out['budget_scale'].reshape(-1)[:b] < 1.0).detach().float().cpu()
            rec['nonfinite'] = (~torch.isfinite(out['low_out']).flatten(1).all(dim=1)).detach().float().cpu()
            captured['rec'] = rec
        else:
            captured['rec'] = None
        return out

    adapter._core = patched

    acc = {'domain': defaultdict(lambda: defaultdict(list)),
           'class': defaultdict(lambda: defaultdict(list)),
           'source': defaultdict(lambda: defaultdict(list))}
    nonfinite_total = 0
    n_seen = 0
    loader = solver.val_dataloader
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            model(samples)
            rec = captured.get('rec')
            ids = [int(t['image_id']) for t in targets]
            if rec is None:
                n_seen += len(ids)
                continue
            for i, iid in enumerate(ids[:len(rec['global_gate'])]):
                n_seen += 1
                nonfinite_total += int(rec['nonfinite'][i])
                dom = dom_of.get(iid, 'UNK')
                src = _source_of(img_file.get(iid, str(iid)))
                classes = img_classes.get(iid, set()) or {'__bg__'}
                for q in QUANTS:
                    v = float(rec[q][i])
                    acc['domain'][dom][q].append(v)
                    acc['source'][src][q].append(v)
                    for c in classes:
                        acc['class'][c][q].append(v)
                acc['domain'][dom]['clamp_fraction'].append(float(rec['clamp'][i]))
                acc['source'][src]['clamp_fraction'].append(float(rec['clamp'][i]))
                for c in classes:
                    acc['class'][c]['clamp_fraction'].append(float(rec['clamp'][i]))

    adapter._core = orig_core

    def summarize(group):
        out = {}
        for key, qd in group.items():
            out[key] = {'n': len(qd.get('global_gate', [])),
                        **{q: (sum(qd[q]) / len(qd[q]) if qd.get(q) else None) for q in QUANTS + ['clamp_fraction']}}
        return out

    dom_sum, cls_sum, src_sum = summarize(acc['domain']), summarize(acc['class']), summarize(acc['source'])
    per_dom_n = {d: dom_sum.get(d, {}).get('n', 0) for d in DOMAINS}
    payload = {
        'metadata': {'config': args.config, 'checkpoint': args.checkpoint, 'gate_mode': args.gate_mode,
                     'srff_version': 'v1_2', 'num_val_images': n_val, 'num_seen': n_seen,
                     'non_finite_count': nonfinite_total, 'domain_ann_dir': domain_dir,
                     'per_domain_images': per_dom_n, 'n_sources': len(src_sum), 'n_classes': len(cls_sum),
                     'audit_ok': nonfinite_total == 0 and n_seen == n_val},
        'by_domain': dom_sum, 'by_class': cls_sum, 'by_source': src_sum,
    }
    with open(os.path.join(args.output_dir, 'srff_v12_checkpoint_diagnostics.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

    cols = ['n'] + QUANTS + ['clamp_fraction']
    for fname, group in (('domain_summary.csv', dom_sum), ('class_summary.csv', cls_sum),
                         ('source_summary.csv', src_sum)):
        with open(os.path.join(args.output_dir, fname), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['key'] + cols)
            for key in sorted(group):
                row = group[key]
                w.writerow([key] + [row.get(c) for c in cols])

    print(f'[inspect-v12] num_seen={n_seen}/{n_val} non_finite={nonfinite_total} '
          f'domains={per_dom_n} sources={len(src_sum)} audit_ok={payload["metadata"]["audit_ok"]}')
    print(f'[inspect-v12] 输出 -> {args.output_dir}')


if __name__ == '__main__':
    main()
