"""SRFF v1 烟测与开销脚本（编码 Agent 自检用）。

用途：在不启动完整训练、不下载权重、不写入数据集的前提下，快速验证
``HybridEncoder`` 在 ``use_srff=False/True`` 两种路径下的前后向、SRFF 新增可训练
参数量、门控诊断标量，以及（CUDA 可用时）640 输入 batch=1 的 encoder 前向延迟增幅。

示例：
    python tools/wood/smoke_srff.py --device cpu
    python tools/wood/smoke_srff.py --device auto --profile --warmup 20 --iters 100

设计约束：
* CPU-only 环境不得报错，只跳过 AMP 与 CUDA 延迟项。
* 延迟门槛（<10%）以 CUDA event 计时为准；环境波动无法稳定判断时报告原始样本，
  不伪造结论。
* SRFF 结构参数与 ``configs/deim_dfine/deim_hgnetv2_s_custom.yml`` 的 HybridEncoder
  保持一致，以便参数量/延迟具备参考意义。
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.deim.hybrid_encoder import HybridEncoder


# 与 DEIM-S 自定义配方一致的 encoder 结构（不含 SRFF 开关）。
ENCODER_KWARGS = dict(
    in_channels=[256, 512, 1024],
    feat_strides=[8, 16, 32],
    hidden_dim=256,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.0,
    enc_act='gelu',
    use_encoder_idx=[2],
    num_encoder_layers=1,
    pe_temperature=10000,
    expansion=0.5,
    depth_mult=0.34,
    act='silu',
    version='dfine',
)

# SRFF 开关与超参（与 coated_wood_s_srff_v1.yml 一致）。
SRFF_KWARGS = dict(
    use_srff=True,
    srff_gaussian_kernel=5,
    srff_trim_kernel=3,
    srff_gate_hidden=16,
    srff_gate_init_bias=-4.0,
    srff_eps=1e-6,
)

PARAM_BUDGET = int(0.01 * 1e6)  # 两个 SRFF block 新增可训练参数上限 0.01M
LATENCY_BUDGET_PCT = 10.0       # 端到端延迟增幅目标 <10%


def build_encoder(use_srff):
    kwargs = dict(ENCODER_KWARGS)
    if use_srff:
        kwargs.update(SRFF_KWARGS)
    else:
        kwargs['use_srff'] = False
    return HybridEncoder(**kwargs)


def make_feats(batch, spatial_p3, device):
    """构造 P3/P4/P5 随机特征（stride 8/16/32），通道 [256,512,1024]。"""
    s = spatial_p3
    return [
        torch.randn(batch, 256, s, s, device=device),
        torch.randn(batch, 512, s // 2, s // 2, device=device),
        torch.randn(batch, 1024, s // 4, s // 4, device=device),
    ]


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def resolve_device(choice):
    if choice == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('--device cuda 指定了 CUDA，但当前环境 torch.cuda.is_available()=False')
        return torch.device('cuda')
    if choice == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def run_fwd_bwd(enc, feats, tag):
    """FP32 前后向，返回 (loss_finite, shapes, grad_finite)。"""
    enc.train()
    enc.zero_grad(set_to_none=True)
    outs = enc(feats)
    loss = sum(o.pow(2).mean() for o in outs)
    loss.backward()

    loss_finite = bool(torch.isfinite(loss).item())
    shapes = [tuple(o.shape) for o in outs]

    grad_finite = True
    for _, p in enc.named_parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
            grad_finite = False
            break

    print(f'[{tag}] input shapes : {[tuple(f.shape) for f in feats]}')
    print(f'[{tag}] output shapes: {shapes}')
    print(f'[{tag}] loss={loss.item():.6f} finite={loss_finite} grad_finite={grad_finite}')
    return loss_finite, shapes, grad_finite


def run_amp_fwd_bwd(enc, feats, tag):
    """CUDA FP16 autocast 前后向，检查 finite。"""
    enc.train()
    enc.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outs = enc(feats)
        loss = sum(o.float().pow(2).mean() for o in outs)
    loss.backward()

    out_finite = all(bool(torch.isfinite(o.float()).all().item()) for o in outs)
    grad_finite = True
    for _, p in enc.named_parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
            grad_finite = False
            break
    print(f'[{tag}] AMP autocast: loss={loss.item():.6f} out_finite={out_finite} grad_finite={grad_finite}')
    return out_finite and grad_finite and bool(torch.isfinite(loss).item())


def report_diagnostics(enc, feats):
    """用 forward_pre_hook 捕获每个 SRFF block 的真实 (high, low)，调用 analyze()。"""
    captured = {}

    def make_hook(index):
        def hook(_module, inputs):
            captured[index] = (inputs[0].detach(), inputs[1].detach())
        return hook

    handles = [
        block.register_forward_pre_hook(make_hook(i))
        for i, block in enumerate(enc.srff_blocks)
    ]
    try:
        enc.eval()
        with torch.no_grad():
            _ = enc(feats)
    finally:
        for handle in handles:
            handle.remove()

    for i, block in enumerate(enc.srff_blocks):
        high, low = captured[i]
        stats = block.analyze(high, low)
        scalar = {k: float(v.item()) for k, v in stats.items()}
        print(f'\n[SRFF block {i}] high={tuple(high.shape)} low={tuple(low.shape)}')
        for key in (
            'gate_mean', 'gate_p95', 'gate_max', 'structure_mean',
            'gaussian_weight_mean', 'trimmed_weight_mean', 'relative_delta',
        ):
            print(f'    {key:22s}= {scalar[key]:.6f}')


def measure_latency(enc, feats, warmup, iters):
    """CUDA event 计时，返回每次 forward 的毫秒列表。"""
    enc.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = enc(feats)
        torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = enc(feats)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    return samples


def summarize(samples):
    return {
        'median': statistics.median(samples),
        'min': min(samples),
        'max': max(samples),
        'mean': statistics.fmean(samples),
    }


def main():
    parser = argparse.ArgumentParser(description='SRFF v1 烟测与开销脚本')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--profile', action='store_true', help='测量 640 输入 batch=1 的 encoder 前向延迟')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    is_cuda = device.type == 'cuda'
    print(f'torch={torch.__version__} device={device} cuda_available={torch.cuda.is_available()}')

    ok = True

    # ---- 1. 构建两个 encoder，对比参数量 ----
    print_header('1. 构建 encoder 与 SRFF 新增可训练参数量')
    enc_off = build_encoder(use_srff=False).to(device)
    enc_on = build_encoder(use_srff=True).to(device)

    n_off = count_trainable(enc_off)
    n_on = count_trainable(enc_on)
    n_srff_direct = sum(
        p.numel() for block in enc_on.srff_blocks for p in block.parameters() if p.requires_grad
    )
    added = n_on - n_off

    print(f'use_srff=False trainable params : {n_off:,}')
    print(f'use_srff=True  trainable params : {n_on:,}')
    print(f'SRFF blocks 直接统计新增参数    : {n_srff_direct:,}  ({n_srff_direct/1e6:.6f} M)')
    print(f'encoder 差值新增参数            : {added:,}  ({added/1e6:.6f} M)')
    print(f'关闭路径不含 srff_blocks 参数   : '
          f'{not any("srff_blocks" in n for n, _ in enc_off.named_parameters())}')

    param_gate_pass = (added == n_srff_direct) and added < PARAM_BUDGET
    print(f'参数门槛 (<{PARAM_BUDGET/1e6:.3f}M 且差值==直接统计) : {"PASS" if param_gate_pass else "FAIL"}')
    ok = ok and param_gate_pass

    # ---- 2. 小尺寸前后向（FP32）----
    print_header('2. 小尺寸随机 P3/P4/P5 前后向 (FP32, batch=2, P3=40)')
    small_feats = make_feats(batch=2, spatial_p3=40, device=device)
    off_finite, _, off_grad = run_fwd_bwd(enc_off, small_feats, 'use_srff=False')
    on_finite, on_shapes, on_grad = run_fwd_bwd(enc_on, small_feats, 'use_srff=True ')
    ok = ok and off_finite and on_finite and off_grad and on_grad

    # ---- 3. AMP（仅 CUDA）----
    print_header('3. FP16 autocast 前后向')
    if is_cuda:
        amp_ok = run_amp_fwd_bwd(enc_on, small_feats, 'use_srff=True ')
        ok = ok and amp_ok
    else:
        print('CPU-only：跳过 AMP（autocast FP16 需要 CUDA）。')

    # ---- 4. 门控诊断标量 ----
    print_header('4. SRFF block analyze() 诊断标量（真实内部特征）')
    report_diagnostics(enc_on, small_feats)

    # ---- 5. 延迟 profile（仅 CUDA，640 输入 batch=1）----
    print_header('5. encoder 前向延迟 (batch=1, 640 输入 -> P3=80)')
    if args.profile:
        if is_cuda:
            profile_feats = make_feats(batch=1, spatial_p3=80, device=device)
            off_samples = measure_latency(enc_off, profile_feats, args.warmup, args.iters)
            on_samples = measure_latency(enc_on, profile_feats, args.warmup, args.iters)
            off_stat = summarize(off_samples)
            on_stat = summarize(on_samples)
            delta_pct = (on_stat['median'] - off_stat['median']) / off_stat['median'] * 100.0

            print(f'warmup={args.warmup} iters={args.iters}')
            print(f'use_srff=False median={off_stat["median"]:.3f} ms '
                  f'(min={off_stat["min"]:.3f}, max={off_stat["max"]:.3f}, mean={off_stat["mean"]:.3f})')
            print(f'use_srff=True  median={on_stat["median"]:.3f} ms '
                  f'(min={on_stat["min"]:.3f}, max={on_stat["max"]:.3f}, mean={on_stat["mean"]:.3f})')
            print(f'延迟增幅 (median)                = {delta_pct:+.2f}%')
            print(f'目标门槛 <{LATENCY_BUDGET_PCT:.0f}%                 : '
                  f'{"PASS" if delta_pct < LATENCY_BUDGET_PCT else "超出目标(见原始样本)"}')
            # 报告原始样本，避免用主观描述代替数字。
            print('原始样本 use_srff=False (ms): '
                  + ', '.join(f'{v:.3f}' for v in off_samples[:20])
                  + (' ...' if len(off_samples) > 20 else ''))
            print('原始样本 use_srff=True  (ms): '
                  + ', '.join(f'{v:.3f}' for v in on_samples[:20])
                  + (' ...' if len(on_samples) > 20 else ''))
        else:
            print('CPU-only：安全跳过 GPU(CUDA event) 延迟 profile。'
                  '延迟门槛需在 CUDA 环境用 --profile 测量。')
    else:
        print('未指定 --profile：跳过延迟测量。')

    print_header('烟测结果')
    print('总体状态:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
