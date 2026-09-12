"""SRFF-V1.1 烟测与机制验收脚本（文档 §5.6 / §7）。

在不训练、不下载权重、不写数据集的前提下验证：
* baseline / V1 / V1.1 三种开关下的参数量与每个 level 的实际类型；
* 全局退化触发门 m 的边界（q<=tau_low→0、中点→0.5、q>=tau_high→1）；
* clean-like（m=0）时 low_out 严格等于 low、gate 为 0；degraded-like（m=1）时保留 V1；
* 全局门本身无可训练参数（V1.1 单 block 参数 == V1 单 block 参数）；
* V1.1 相对 V1 少一个 block 的参数与计算；forward/backward、shape、dtype、finite；
* baseline off 路径无 SRFF 参数；V1 旧配置仍可运行；CUDA 可用时 AMP 与延迟对比。

用法::
    python tools/wood/smoke_srff_v1_1.py --device cpu
    CUDA_VISIBLE_DEVICES=0 python tools/wood/smoke_srff_v1_1.py --device cuda --profile
"""

import argparse
import inspect
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.srff import SelectiveRobustFrequencyFusion
from engine.deim.srff_v1_1 import SelectiveRobustFrequencyFusionV11

# 与 DEIM-S 自定义配方一致的 encoder 结构。
ENCODER_KWARGS = dict(
    in_channels=[256, 512, 1024], feat_strides=[8, 16, 32], hidden_dim=256, nhead=8,
    dim_feedforward=1024, dropout=0.0, enc_act='gelu', use_encoder_idx=[2],
    num_encoder_layers=1, pe_temperature=10000, expansion=0.5, depth_mult=0.34,
    act='silu', version='dfine',
)
LATENCY_BUDGET_PCT = 10.0


def build_encoder(use_srff, version='v1', active_levels=None, device='cpu'):
    kw = dict(ENCODER_KWARGS)
    kw['use_srff'] = use_srff
    if use_srff:
        kw['srff_version'] = version
        if active_levels is not None:
            kw['srff_active_levels'] = active_levels
    return HybridEncoder(**kw).to(device)


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def srff_param_count(enc):
    if not enc.use_srff:
        return 0
    return sum(p.numel() for b in enc.srff_blocks for p in b.parameters() if p.requires_grad)


def make_feats(batch, spatial_p3, device):
    s = spatial_p3
    return [torch.randn(batch, 256, s, s, device=device),
            torch.randn(batch, 512, s // 2, s // 2, device=device),
            torch.randn(batch, 1024, s // 4, s // 4, device=device)]


def resolve_device(choice):
    if choice == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('--device cuda 指定了 CUDA，但 torch.cuda.is_available()=False')
        return torch.device('cuda')
    if choice == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def header(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


def rel_change(low_out, low):
    num = (low_out - low).abs().mean()
    den = low.abs().mean() + 1e-6
    return float(num / den)


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.1 烟测与机制验收')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    is_cuda = device.type == 'cuda'
    print(f'torch={torch.__version__} device={device} cuda_available={torch.cuda.is_available()}')
    ok = True

    # ---- 1. 三种开关的参数量与 block 类型 ----
    header('1. baseline / V1 / V1.1 参数量与每个 level 类型')
    enc_off = build_encoder(False, device=device)
    enc_v1 = build_encoder(True, 'v1', None, device)
    enc_v11 = build_encoder(True, 'v1_1', [0], device)

    p_off, p_v1, p_v11 = count_trainable(enc_off), count_trainable(enc_v1), count_trainable(enc_v11)
    s_off, s_v1, s_v11 = srff_param_count(enc_off), srff_param_count(enc_v1), srff_param_count(enc_v11)
    print(f'use_srff=False : total={p_off:,}  srff={s_off}')
    print(f'V1  (2 levels) : total={p_v1:,}  srff={s_v1:,}  types='
          f'{[type(b).__name__ for b in enc_v1.srff_blocks]}')
    print(f'V1.1(level[0]) : total={p_v11:,}  srff={s_v11:,}  types='
          f'{[type(b).__name__ for b in enc_v11.srff_blocks]}')
    print(f'V1.1 active_levels={enc_v11.srff_active_levels}  '
          f'block1 is Identity={isinstance(enc_v11.srff_blocks[1], torch.nn.Identity)}')
    one_v1_block = s_v1 // 2
    checks = {
        'off 无 SRFF 参数': s_off == 0,
        'V1 有两个 SRFF block': s_v1 > 0 and all(type(b).__name__ == 'SelectiveRobustFrequencyFusion' for b in enc_v1.srff_blocks),
        'V1.1 block0 为 V11': type(enc_v11.srff_blocks[0]).__name__ == 'SelectiveRobustFrequencyFusionV11',
        'V1.1 block1 为无参数 Identity': isinstance(enc_v11.srff_blocks[1], torch.nn.Identity) and len(list(enc_v11.srff_blocks[1].parameters())) == 0,
        'V1.1 只有一个有效 SRFF block': s_v11 == one_v1_block,
        '全局门不新增可训练参数(V1.1单block==V1单block)': s_v11 == one_v1_block,
        'V1.1 参数少于 V1': s_v11 < s_v1,
        'state dict 无 srff_blocks.1.*': not any(k.startswith('srff_blocks.1.') for k in enc_v11.state_dict()),
    }
    for k, v in checks.items():
        print(f'  [{"OK" if v else "FAIL"}] {k}')
        ok = ok and v

    # ---- 2. 全局门边界 + clean/degraded 行为（standalone V11 block）----
    header('2. 全局退化触发门 m：边界与 clean/degraded 行为')
    blk = SelectiveRobustFrequencyFusionV11(channels=16).to(device)
    for score, expect in [(0.76, 0.0), (0.78, 0.0), (0.79, 0.5), (0.80, 1.0), (0.85, 1.0)]:
        g = float(blk._smooth_global_gate(torch.tensor([[[[score]]]], device=device)))
        # float32 下 smoothstep 中点约 0.500004（非精确 0.5），容差放宽到 1e-4；0/1 端为 clamp 精确值。
        good = abs(g - expect) < 1e-4
        print(f'  [{"OK" if good else "FAIL"}] m(q={score})={g:.6f} 期望≈{expect}')
        ok = ok and good

    high = torch.randn(2, 16, 12, 12, device=device)
    low = torch.randn(2, 16, 24, 24, device=device)

    # clean-like：阈值拉高使真实 q 落在 m=0 → 严格恒等
    blk.tau_low, blk.tau_high = 0.9999, 1.0
    oc = blk._core(high, low)
    clean_gate0 = float(oc['global_gate'].max()) == 0.0
    clean_identity = torch.equal(oc['low_out'], low)
    clean_rel = rel_change(oc['low_out'], low)
    print(f'  [{"OK" if clean_gate0 else "FAIL"}] clean-like: global_gate.max=0')
    print(f'  [{"OK" if clean_identity else "FAIL"}] clean-like: low_out 严格等于 low (relative_delta={clean_rel:.3e})')
    ok = ok and clean_gate0 and clean_identity

    # degraded-like：阈值拉低使 m=1 → 保留 V1
    v1_blk = SelectiveRobustFrequencyFusion(channels=16).to(device)
    v1_blk.load_state_dict(blk.state_dict())
    blk.tau_low, blk.tau_high = 0.0, 1e-6
    od = blk._core(high, low)
    ov1 = v1_blk._core(high, low)
    deg_gate1 = float(od['global_gate'].min()) == 1.0
    deg_match = torch.allclose(od['low_out'], ov1['low_out'], atol=1e-5)
    deg_rel = rel_change(od['low_out'], low)
    print(f'  [{"OK" if deg_gate1 else "FAIL"}] degraded-like: global_gate.min=1')
    print(f'  [{"OK" if deg_match else "FAIL"}] degraded-like: low_out≈同权重 V1 (relative_delta={deg_rel:.5f})')
    ok = ok and deg_gate1 and deg_match

    # 默认阈值下 transition：final gate == pre_global_gate * m
    blk.tau_low, blk.tau_high = 0.78, 0.80
    ot = blk._core(high, low)
    trans = torch.allclose(ot['gate'], ot['pre_global_gate'] * ot['global_gate'], atol=1e-6)
    print(f'  [{"OK" if trans else "FAIL"}] transition: gate == pre_global_gate * global_gate')
    ok = ok and trans

    # 不读取 filename/domain/clean target：forward 与 _core 只接受 (high, low)
    sig_fwd = list(inspect.signature(blk.forward).parameters)
    sig_core = list(inspect.signature(blk._core).parameters)
    no_meta = sig_fwd == ['high', 'low'] and sig_core == ['high', 'low']
    print(f'  [{"OK" if no_meta else "FAIL"}] forward/_core 仅接受 (high, low)：{sig_fwd} / {sig_core}')
    ok = ok and no_meta

    # ---- 3. 前后向 + shape/dtype/finite（FP32）----
    header('3. encoder 前后向（FP32, batch=2, P3=40）：shape/dtype/finite')
    feats = make_feats(2, 40, device)
    for tag, enc in [('off', enc_off), ('v1', enc_v1), ('v1_1', enc_v11)]:
        enc.train(); enc.zero_grad(set_to_none=True)
        outs = enc(feats)
        loss = sum(o.pow(2).mean() for o in outs)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            (p.grad is None) or bool(torch.isfinite(p.grad).all()) for _, p in enc.named_parameters())
        shapes = [tuple(o.shape) for o in outs]
        print(f'  [{tag}] loss={loss.item():.4f} finite={finite} out_shapes={shapes}')
        ok = ok and finite

    # ---- 4. AMP（仅 CUDA）----
    header('4. FP16 autocast 前后向')
    if is_cuda:
        enc_v11.train(); enc_v11.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outs = enc_v11(feats)
            loss = sum(o.float().pow(2).mean() for o in outs)
        loss.backward()
        amp_ok = all(bool(torch.isfinite(o.float()).all()) for o in outs) and bool(torch.isfinite(loss))
        amp_ok = amp_ok and all((p.grad is None) or bool(torch.isfinite(p.grad).all())
                                for _, p in enc_v11.named_parameters())
        print(f'  [{"OK" if amp_ok else "FAIL"}] V1.1 AMP finite={amp_ok}')
        ok = ok and amp_ok
    else:
        print('  CPU-only：跳过 AMP。')

    # ---- 5. 延迟 profile（仅 CUDA，640 输入 batch=1）----
    header('5. encoder 前向延迟 (batch=1, 640 输入 -> P3=80)')
    if args.profile and is_cuda:
        pf = make_feats(1, 80, device)

        def measure(enc):
            enc.eval()
            with torch.no_grad():
                for _ in range(args.warmup):
                    enc(pf)
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.iters):
                    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                    a.record(); enc(pf); b.record(); torch.cuda.synchronize()
                    ts.append(a.elapsed_time(b))
            return ts

        t_off = statistics.median(measure(enc_off))
        t_v1 = statistics.median(measure(enc_v1))
        t_v11 = statistics.median(measure(enc_v11))
        print(f'  off median={t_off:.3f}ms  V1 median={t_v1:.3f}ms  V1.1 median={t_v11:.3f}ms')
        d_v1 = (t_v1 - t_off) / t_off * 100.0
        d_v11 = (t_v11 - t_off) / t_off * 100.0
        print(f'  V1 vs off={d_v1:+.2f}%   V1.1 vs off={d_v11:+.2f}%   (目标 <{LATENCY_BUDGET_PCT:.0f}%)')
        print(f'  V1.1 应不慢于 V1（少一个 block）: {t_v11 <= t_v1 * 1.02}')
    else:
        print('  未启用 --profile 或 CPU-only：跳过延迟测量。')

    header('烟测结果')
    print('总体状态:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
