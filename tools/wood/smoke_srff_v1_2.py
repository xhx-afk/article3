"""SRFF-V1.2 smoke（实现文档 §5.5 / 用户手册 §5）。

从真实 config 构建模型、用 -t 语义加载 baseline、冻结、建 optimizer，再做 --steps 步最小检测训练，
核验：baseline 加载、optimizer 参数集合、冻结一致性、adapter 更新、RMS 预算、非有限计数、
checkpoint 保存再加载、stage restart 未触发、EMA 未创建。输出机器可读 JSON。

用法::
    python tools/wood/smoke_srff_v1_2.py --config <v1.2.yml> --tuning <baseline.pth> --device cpu --steps 2 --output smoke.json
    CUDA_VISIBLE_DEVICES=0 python tools/wood/smoke_srff_v1_2.py --config ... --tuning ... --device cuda --amp --steps 2 --output ...
"""

import argparse
import json
import sys
import tempfile
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.misc import adapter_freeze as AF  # noqa: E402

PATTERNS = [r'^encoder\.srff_blocks\.0\.']
R = {}


def _strip(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}


def _find_adapter(model):
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'FrozenBaseEvidenceConditionedResidualAdapter':
            return mod
    return None


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.2 smoke')
    ap.add_argument('--config', required=True)
    ap.add_argument('--tuning', required=True, help='原始 baseline best_stg2.pth')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--img-size', type=int, default=320)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    use_amp = bool(args.amp and device.type == 'cuda')
    torch.manual_seed(args.seed)
    print(f'== SRFF-V1.2 smoke | device={device} amp={use_amp} steps={args.steps} ==')

    # 构建模型（不建 dataloader/criterion，避免重依赖）
    cfg = YAMLConfig(args.config, device=str(device))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model.to(device)

    adapter_cfg = cfg.yaml_cfg.get('adapter_only_training', {}) or {}
    patterns = list(adapter_cfg.get('trainable_param_patterns', PATTERNS) or PATTERNS)

    # 加载 baseline（-t 语义：优先 ema.module）
    state = torch.load(args.tuning, map_location='cpu')
    pre = state['ema']['module'] if ('ema' in state and 'module' in state.get('ema', {})) else state['model']
    model.load_state_dict(_strip(pre), strict=False)
    try:
        audit = AF.audit_baseline_load(model, args.tuning, patterns)
        R['baseline_load_ok'] = bool(audit['ok'])
        R['baseline_source'] = audit['source']
    except Exception as exc:  # noqa: BLE001
        R['baseline_load_ok'] = False
        R['baseline_load_error'] = str(exc)

    # 冻结 + optimizer 集合
    tr, fz = AF.freeze_non_adapter(model, patterns)
    AF.verify_trainable_set(model, patterns)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    adapter_named = {n for n, p in model.named_parameters() if AF.is_adapter_name(n, patterns)}
    opt = torch.optim.AdamW(opt_params, lr=1e-3)
    n_opt = sum(len(g['params']) for g in opt.param_groups)
    R['optimizer_param_set_ok'] = (n_opt == len(adapter_named) == len(tr))
    R['trainable_params'] = len(tr)
    R['frozen_params'] = fz

    # EMA 未创建 / stage restart 未触发（配置层）
    R['ema_disabled'] = bool(cfg.ema is None) and bool(cfg.yaml_cfg.get('use_ema') is False)
    R['skip_stage_restart'] = bool(adapter_cfg.get('skip_stage_restart', False))

    adapter = _find_adapter(model)
    R['adapter_found'] = adapter is not None
    # smoke 专用：压低全局门 tau，确保随机合成输入下 gate≈1、adapter 能拿到梯度（不改配置/正式训练）
    adapter.tau_low, adapter.tau_high = 0.0, 1e-3

    # frozen_eval 语义：apply 后 adapter=train、非 adapter BN/Dropout=eval
    AF.apply_adapter_train_mode(model, patterns)
    R['frozen_eval_bn_ok'] = bool(adapter.training
                                  and AF.count_non_adapter_train_mode(model, patterns) == 0)

    # 冻结前快照
    frozen_p = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    frozen_b = {n: b.detach().clone() for n, b in model.named_buffers() if not AF.is_adapter_name(n, patterns)}
    adapter_p = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # 两步最小训练：在 adapter 子模块上前反向（避开 decoder/pos_embed；全模型等价由 C 关 init-equiv 验证）
    non_finite = 0
    budget_ok = True
    S = args.img_size
    Cad = adapter.channels
    h5, h4 = max(S // 32, 4), max(S // 16, 8)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    for step in range(args.steps):
        opt.zero_grad()
        AF.apply_adapter_train_mode(model, patterns)
        hi = torch.randn(2, Cad, h5, h5, device=device)
        lo = torch.randn(2, Cad, h4, h4, device=device)
        if use_amp:
            with torch.autocast(device_type='cuda'):
                oc = adapter._core(hi, lo)
                loss = oc['low_out'].float().square().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            oc = adapter._core(hi, lo)
            loss = oc['low_out'].square().mean()
            loss.backward()
            opt.step()
        if not bool(torch.isfinite(loss)):
            non_finite += 1
        if 'delta_rms_post' in oc:
            if not bool(torch.all(oc['delta_rms_post'] <= adapter.alpha_max * oc['low_rms'] + 1e-4)):
                budget_ok = False
            if not bool(torch.isfinite(oc['low_out']).all()):
                non_finite += 1

    # 非 adapter grad 应为 None（item 12 语义）
    grad_leak = [n for n, p in model.named_parameters()
                 if not AF.is_adapter_name(n, patterns) and p.grad is not None]
    R['non_adapter_grad_leak'] = grad_leak[:5]

    R['frozen_params_equal'] = all(torch.equal(p.detach(), frozen_p[n])
                                   for n, p in model.named_parameters() if not p.requires_grad)
    R['frozen_buffers_equal'] = all(torch.equal(b.detach(), frozen_b[n])
                                    for n, b in model.named_buffers() if not AF.is_adapter_name(n, patterns))
    R['adapter_changed'] = any(not torch.equal(p.detach(), adapter_p[n])
                               for n, p in model.named_parameters() if p.requires_grad)
    R['budget_ok'] = bool(budget_ok)
    R['non_finite_count'] = int(non_finite)

    # checkpoint 保存再加载
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'adapter.pth')
        ck = AF.build_adapter_checkpoint(model, opt, 0, patterns, baseline_sha256='b' * 64,
                                         config_sha256='c' * 64, git=None, extra={'ap_overall': 0.0})
        AF.save_adapter_checkpoint(p, ck)
        loaded = torch.load(p, map_location='cpu')
        R['checkpoint_save_reload_ok'] = (
            loaded['use_ema'] is False and 'ema' not in loaded and len(loaded['adapter']) > 0
            and set(loaded['model'].keys()) == set(_strip(model.state_dict()).keys()))

    required = ['baseline_load_ok', 'optimizer_param_set_ok', 'frozen_params_equal',
                'frozen_buffers_equal', 'adapter_changed', 'budget_ok', 'checkpoint_save_reload_ok',
                'ema_disabled', 'skip_stage_restart', 'frozen_eval_bn_ok']
    R['overall_pass'] = bool(all(R.get(k) for k in required) and R['non_finite_count'] == 0
                             and not R['non_adapter_grad_leak'])
    print(json.dumps(R, ensure_ascii=False, indent=2))
    print(f'\n== SRFF-V1.2 smoke 总体: {"PASS" if R["overall_pass"] else "FAIL"} ==')
    if args.output:
        Path(os.path.dirname(os.path.abspath(args.output)) or '.').mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(R, f, ensure_ascii=False, indent=2)
        print(f'[saved] {args.output}')
    sys.exit(0 if R['overall_pass'] else 1)


if __name__ == '__main__':
    main()
