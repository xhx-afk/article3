"""SRFF-V1.2 初始化等价检查（实现文档 §5.1 / 用户手册 §6）。

用同一 baseline checkpoint 分别构建 baseline(use_srff=False) 与 V1.2(zero-init adapter)，在同一 val
子集上比较：

* 张量级：两模型对同一批图的检测输出（pred_logits/pred_boxes）max/mean abs diff（允许浮点可解释误差）；
* AP 级：官方 evaluator 复评 overall + 指定域的 baseline / V1.2 force_off / V1.2 auto，
  要求 V1.2 与 baseline 的原始 AP 差 <= 1e-6 AP point（force_off 与 zero-init auto 都应严格等价）。

输出 ``CHECK=PASS/FAIL`` + ``init_equivalence.json``；任一不满足非零退出。域标注默认从配置 val ann_file
推导到同级 ``domains/{dom}_val.json``，可用 --domain-ann-dir / --domain-tpl 覆盖。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from engine.solver.det_engine import evaluate  # noqa: E402

TOL_AP = 1e-6
ADAPTER_CLS = 'FrozenBaseEvidenceConditionedResidualAdapter'
EXIT_FAIL = 2


def _mk_solver(config, tuning, device, use_srff=None, ann=None):
    upd = {'tuning': tuning}
    if device:
        upd['device'] = device
    if use_srff is not None:
        upd['HybridEncoder'] = {'use_srff': use_srff}
    if ann:
        upd['val_dataloader'] = {'dataset': {'ann_file': ann}}
    cfg = YAMLConfig(config, **upd)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()   # _setup: 建模 + tuning(strict=False) 加载 baseline + val_dataloader + evaluator
    return solver, cfg


def _ap(solver):
    module = solver.ema.module if solver.ema else solver.model
    stats, _ = evaluate(module, solver.criterion, solver.postprocessor,
                        solver.val_dataloader, solver.evaluator, solver.device)
    return float(stats['coco_eval_bbox'][0]) * 100.0


def _set_gate(solver, mode):
    n = 0
    for m in solver.model.modules():
        if type(m).__name__ == ADAPTER_CLS:
            m.gate_mode = mode
            n += 1
    return n


def _derive_domain_dir(cfg):
    ann = (((cfg.yaml_cfg.get('val_dataloader', {}) or {}).get('dataset', {}) or {}).get('ann_file'))
    if not ann:
        return None
    return str(Path(ann).parent / 'domains')


def _tensor_diff(base_solver, v12_solver, device, img=640):
    x = torch.randn(2, 3, img, img, device=device)
    _set_gate(v12_solver, 'force_off')
    base_solver.model.eval()
    v12_solver.model.eval()
    with torch.no_grad():
        ob = base_solver.model(x)
        ov = v12_solver.model(x)
    dl = (ob['pred_logits'].float() - ov['pred_logits'].float()).abs()
    db = (ob['pred_boxes'].float() - ov['pred_boxes'].float()).abs()
    return {'pred_logits_max': float(dl.max()), 'pred_logits_mean': float(dl.mean()),
            'pred_boxes_max': float(db.max()), 'pred_boxes_mean': float(db.mean())}


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.2 初始化等价检查')
    ap.add_argument('--config', required=True)
    ap.add_argument('--baseline-checkpoint', required=True)
    ap.add_argument('--domains', nargs='+', default=['ODC', 'GDC', 'PDC'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--domain-ann-dir', default=None)
    ap.add_argument('--domain-tpl', default='{dom}_val.json')
    ap.add_argument('--img-size', type=int, default=640)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 先建一次以推导域目录（用 overall）
    _, cfg0 = _mk_solver(args.config, args.baseline_checkpoint, device, use_srff=False)
    domain_dir = args.domain_ann_dir or _derive_domain_dir(cfg0)

    subsets = [('overall', None)] + [(d, os.path.join(domain_dir, args.domain_tpl.format(dom=d)))
                                     for d in args.domains] if domain_dir else [('overall', None)]
    if not domain_dir:
        print('[init-equiv][warn] 未能推导域标注目录，仅评估 overall；请用 --domain-ann-dir 指定')

    results, fails = {}, []
    tdiff = None
    for name, ann in subsets:
        if ann and not os.path.isfile(ann):
            fails.append(f'{name}: 域标注缺失 {ann}')
            continue
        base_solver, _ = _mk_solver(args.config, args.baseline_checkpoint, device, use_srff=False, ann=ann)
        v12_solver, _ = _mk_solver(args.config, args.baseline_checkpoint, device, use_srff=True, ann=ann)
        n_ad = _set_gate(v12_solver, 'force_off')
        if n_ad == 0:
            fails.append(f'{name}: V1.2 模型未找到 adapter')
        ap_base = _ap(base_solver)
        ap_off = _ap(v12_solver)
        _set_gate(v12_solver, 'auto')
        ap_auto = _ap(v12_solver)
        if tdiff is None and name == 'overall':
            tdiff = _tensor_diff(base_solver, v12_solver, base_solver.device, args.img_size)
        d_off, d_auto = abs(ap_off - ap_base), abs(ap_auto - ap_base)
        results[name] = {'baseline_AP': ap_base, 'v12_force_off_AP': ap_off, 'v12_auto_AP': ap_auto,
                         'abs_diff_force_off': d_off, 'abs_diff_auto': d_auto,
                         'num_images': len(base_solver.val_dataloader.dataset),
                         'pass': d_off <= TOL_AP and d_auto <= TOL_AP}
        print(f'[init-equiv] {name}: baseline={ap_base:.6f} force_off={ap_off:.6f}(Δ{d_off:.2e}) '
              f'auto={ap_auto:.6f}(Δ{d_auto:.2e}) -> {"OK" if results[name]["pass"] else "FAIL"}')
        if not results[name]['pass']:
            fails.append(f'{name}: AP 差超过 {TOL_AP}（force_off Δ={d_off:.2e}, auto Δ={d_auto:.2e}）')
        del base_solver, v12_solver

    payload = {'config': args.config, 'baseline_checkpoint': args.baseline_checkpoint,
               'tolerance_ap_point': TOL_AP, 'tensor_diff': tdiff, 'subsets': results,
               'failures': fails, 'CHECK': 'PASS' if not fails else 'FAIL'}
    outp = os.path.join(args.output_dir, 'init_equivalence.json')
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    if tdiff:
        print(f'[init-equiv] tensor diff: {tdiff}')
    for x in fails:
        print('[init-equiv][fail]', x)
    print(f'CHECK={"PASS" if not fails else "FAIL"}  (详情 {outp})')
    if fails:
        raise SystemExit(EXIT_FAIL)


if __name__ == '__main__':
    main()
