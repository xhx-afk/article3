"""Wavelet-domain downsampling for HGNetv2 (innovation 1).

HaarDWT replaces the learnable stride-2 depthwise conv with a fixed,
orthonormal Haar transform (stride 2). The four subbands (LL/LH/HL/HH) are
then mixed per channel by a grouped 1x1 projection, optionally modulated by
SubbandEnergyGate, which reads the per-channel ENERGY SHARE of the four
subbands -- not the feature mean, which is exactly what separates this from a
plain SE gate (ablation A3).

Channel order is CHANNEL-MAJOR: [c0_LL, c0_LH, c0_HL, c0_HH, c1_LL, ...] so
that Conv2d(4C, C, 1, groups=C) group c consumes exactly channel c's four
subbands. Do NOT switch to subband-major.

The gate MLP is shared across channels (~40 params): "impulse noise raises
the HH share" is a channel-independent physical fact.

Self-test (W1-W9):
    python -m engine.backbone.wavelet --self-test-only
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['HaarDWT', 'SubbandEnergyGate', 'HaarSubbandDownsample',
           'SUBBANDS', 'HAAR_KERNELS']

# Orthonormal Haar: the four subband squares sum EXACTLY to the original 2x2
# block's square sum (energy conservation) -- the gate reads energy SHARES, so
# this property is load-bearing. Rows are orthonormal.
_HAAR = 0.5 * torch.tensor([
    [[ 1.,  1.], [ 1.,  1.]],   # LL  low-pass
    [[ 1.,  1.], [-1., -1.]],   # LH  horizontal edges
    [[ 1., -1.], [ 1., -1.]],   # HL  vertical edges
    [[ 1., -1.], [-1.,  1.]],   # HH  diagonal / impulse noise
])
SUBBANDS = ('LL', 'LH', 'HL', 'HH')
HAAR_KERNELS = _HAAR


class HaarDWT(nn.Module):
    """One level of 2D Haar transform, stride 2. Channel-major output order:
    [c0_LL, c0_LH, c0_HL, c0_HH, c1_LL, ...] -- aligns with groups=C 1x1."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        w = _HAAR.unsqueeze(1).repeat(channels, 1, 1, 1)   # (4C, 1, 2, 2)
        # persistent=False: the fixed kernel must NOT enter the state_dict
        # (would show up as unexpected keys when loading -t weights)
        self.register_buffer('weight', w, persistent=False)

    def forward(self, x):
        assert x.shape[-2] % 2 == 0 and x.shape[-1] % 2 == 0, \
            f'Haar needs even spatial dims, got {tuple(x.shape[-2:])}'
        return F.conv2d(x, self.weight.to(x.dtype), stride=2,
                        groups=self.channels)


class SubbandEnergyGate(nn.Module):
    """Subband energy-share gate.

    p = per-channel energy share of the 4 subbands (B, C, 4), normalized
    across subbands. g = 1 + tanh(MLP(p)); the MLP's last layer is
    zero-initialized so AT CONSTRUCTION g == 1 and the module is exactly a
    pure Haar downsample (checked by W3).

    The energy is computed in fp32: under AMP, fp16 squares overflow on deep
    features.
    """

    def __init__(self, reduction=4, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.fc1 = nn.Linear(4, reduction)
        self.fc2 = nn.Linear(reduction, 4)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)   # initial state == identity
        self.last_p = None              # gate_probe.py reads this

    def forward(self, x_sb):
        e = x_sb.float().pow(2).mean(dim=(-2, -1))          # (B, C, 4) fp32
        p = e / (e.sum(dim=-1, keepdim=True) + self.eps)    # energy share
        self.last_p = p.detach()
        g = 1.0 + torch.tanh(self.fc2(F.relu(self.fc1(p))))  # (B, C, 4)
        return x_sb * g.to(x_sb.dtype)[..., None, None], g


class HaarSubbandDownsample(nn.Module):
    """Drop-in replacement for HG_Stage.downsample
    (was ConvBNAct(C, C, k3, s2, groups=C, use_act=False)).

    mode='dw'   grouped 1x1 (4->1 per channel), 4C params  -- mainline; fewer
                params than the original depthwise 9C. Staying depthwise is
                deliberate: the original unit does not mix channels either;
                mixing is the job of the following HG_Block.
    mode='full' dense 1x1 (4C->C), 4C^2 params            -- ablation A2-full

    se_baseline=True (A3): gate driven by the subband MEAN instead of the
    energy share -- the control that proves the energy share is what matters.

    last_gate / (via SubbandEnergyGate.last_p) are read by gate_probe.py.
    """

    def __init__(self, channels, gate=True, mode='dw', reduction=4,
                 se_baseline=False):
        super().__init__()
        if mode not in ('dw', 'full'):
            raise ValueError(f"mode must be 'dw' or 'full', got {mode!r}")
        self.channels, self.mode = channels, mode
        self.dwt = HaarDWT(channels)
        if mode == 'dw':
            self.proj = nn.Conv2d(4 * channels, channels, 1, groups=channels,
                                  bias=False)
        else:
            self.proj = nn.Conv2d(4 * channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)   # aligns with ConvBNAct's BN
        self.gate = SubbandEnergyGate(reduction) if gate else None
        self.se_baseline = se_baseline       # A3 control
        self.last_gate = None                # gate_probe.py reads this

    def forward(self, x):
        B, C = x.shape[:2]
        y = self.dwt(x)                                  # (B, 4C, H/2, W/2)
        if self.gate is not None:
            y5 = y.view(B, C, 4, *y.shape[-2:])
            if self.se_baseline:                          # A3 control
                d = y5.float().mean(dim=(-2, -1))
                g = 1.0 + torch.tanh(self.gate.fc2(F.relu(self.gate.fc1(d))))
                y5 = y5 * g.to(y5.dtype)[..., None, None]
            else:
                y5, g = self.gate(y5)
            self.last_gate = g.detach()
            y = y5.reshape(B, 4 * C, *y.shape[-2:])
        return self.bn(self.proj(y))


# -----------------------------------------------------------------------------
# Self-test (W1-W9)
# -----------------------------------------------------------------------------

def self_test():
    print('[self-test] engine.backbone.wavelet')
    failures = []
    torch.manual_seed(0)

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    C, H, W = 6, 16, 16
    x = torch.randn(2, C, H, W)

    # W1. energy conservation: sum of subband squares == input square sum
    dwt = HaarDWT(C)
    y = dwt(x).view(2, C, 4, H // 2, W // 2)
    e_in = x.pow(2).view(2, C, H // 2, 2, W // 2, 2).sum(dim=(3, 5))
    e_out = y.pow(2).sum(dim=2)
    check('W1 Haar energy conservation (rtol 1e-6)',
          torch.allclose(e_in, e_out, rtol=1e-6, atol=1e-7),
          f'max abs diff {(e_in - e_out).abs().max().item():.2e}')

    # W2. rows orthonormal: K K^T == I
    K = _HAAR.reshape(4, 4)
    eye = K @ K.t()
    check('W2 Haar rows orthonormal (K K^T == I)',
          torch.allclose(eye, torch.eye(4), atol=1e-6),
          f'max abs diff {(eye - torch.eye(4)).abs().max().item():.2e}')

    # W3. gate is identity at construction; gated == ungated bit-exactly
    g_sb = torch.randn(2, C, 4, H // 2, W // 2)
    gate = SubbandEnergyGate(reduction=4)
    _, g = gate(g_sb)
    check('W3a constructed gate output == 1 (<1e-6)',
          (g - 1.0).abs().max().item() < 1e-6,
          f'max |g-1| = {(g - 1.0).abs().max().item():.2e}')
    m_gate = HaarSubbandDownsample(C, gate=True, mode='dw')
    m_no = HaarSubbandDownsample(C, gate=False, mode='dw')
    m_no.proj.load_state_dict(m_gate.proj.state_dict())
    m_no.bn.load_state_dict(m_gate.bn.state_dict())
    m_gate.eval()
    m_no.eval()
    with torch.no_grad():
        y1 = m_gate(x)
        y2 = m_no(x)
    check('W3b gated == ungated bit-exact at construction',
          torch.equal(y1, y2))

    # W4. channel independence (mode='dw'): perturbing input channel k only
    #     changes output channel k
    m_g = HaarSubbandDownsample(C, gate=True, mode='dw')
    m_g.eval()
    with torch.no_grad():
        out = m_g(x)
        x2 = x.clone()
        x2[:, 3] += 10.0
        out2 = m_g(x2)
    changed = (out2 - out).abs().sum(dim=(0, 2, 3))
    check('W4 channel independence (mode=dw)',
          changed[3] > 0 and bool((changed[:3] == 0).all()) and
          bool((changed[4:] == 0).all()),
          f'changed = {changed.tolist()}')

    # W5. odd spatial dims -> AssertionError
    raised = False
    try:
        dwt(torch.randn(1, C, 15, 16))
    except AssertionError:
        raised = True
    check('W5 odd spatial size raises AssertionError', raised)

    # W6. parameter counts (proj conv only, BN excluded as in the doc's bench)
    def n_proj(m):
        return sum(p.numel() for n_, p in m.named_parameters()
                   if n_.startswith('proj.'))
    def n_gate(m):
        return sum(p.numel() for n_, p in m.named_parameters()
                   if n_.startswith('gate.'))
    check('W6a mode=dw proj params == 4C',
          n_proj(HaarSubbandDownsample(C, gate=False, mode='dw')) == 4 * C)
    check('W6b gate params == 40 (reduction=4)',
          n_gate(HaarSubbandDownsample(C, gate=True, mode='dw')) == 40)
    check('W6c mode=full proj params == 4C^2',
          n_proj(HaarSubbandDownsample(C, gate=False, mode='full')) == 4 * C * C)
    # no Haar buffer leaks into parameters / state_dict
    m_chk = HaarSubbandDownsample(C, gate=True, mode='dw')
    check('W6d Haar kernel not in state_dict (persistent=False)',
          all('.weight' not in k or 'proj' in k or 'bn' in k or 'fc' in k
              for k in m_chk.state_dict()) and
          all(not k.endswith('dwt.weight') for k in m_chk.state_dict()))

    # W7. salt-and-pepper input -> HH share >= 10x the clean-input share
    def hh_share(inp):
        y = HaarDWT(C)(inp)
        x_sb = y.view(1, C, 4, *y.shape[-2:])
        g2 = SubbandEnergyGate(reduction=4)
        with torch.no_grad():
            g2(x_sb)
        return float(g2.last_p[..., 3].mean())
    clean = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W) \
        .repeat(1, C, 1, 1).contiguous()
    noisy = clean.clone()
    spike = torch.rand(1, C, H, W) < 0.05
    noisy[spike] = torch.where(torch.rand_like(noisy)[spike] < 0.5,
                               torch.zeros(1), torch.ones(1)).to(noisy.dtype)
    p_clean, p_noisy = hh_share(clean), hh_share(noisy)
    check('W7 salt-and-pepper HH share >= 10x clean',
          p_noisy >= 10.0 * max(p_clean, 1e-12),
          f'clean={p_clean:.2e} noisy={p_noisy:.2e}')

    # W8. AMP: forward must stay finite; output dtype follows conv autocast
    amp_ok, amp_detail = False, ''
    try:
        if not hasattr(torch, 'autocast'):
            # torch <= 1.9 on this host: no autocast capability at all
            print('  [SKIP] W8 AMP forward finite (no autocast in this '
                  'torch build; verified on the server torch)')
        elif torch.cuda.is_available():
            dev = torch.device('cuda')
            m_amp = HaarSubbandDownsample(C, gate=True, mode='dw').to(dev)
            xa = torch.randn(2, C, H, W, device=dev)
            with torch.autocast('cuda', dtype=torch.float16):
                out = m_amp(xa)
            amp_ok = bool(torch.isfinite(out.float()).all())
            amp_detail = f'out dtype {out.dtype}'
            check('W8 AMP forward finite (no NaN/Inf)', amp_ok, amp_detail)
        else:
            m_amp = HaarSubbandDownsample(C, gate=True, mode='dw')
            xa = torch.randn(2, C, H, W)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                out = m_amp(xa)
            amp_ok = bool(torch.isfinite(out.float()).all())
            amp_detail = f'out dtype {out.dtype}'
            check('W8 AMP forward finite (no NaN/Inf)', amp_ok, amp_detail)
    except Exception as e:  # older torch without cpu autocast etc.
        amp_detail = f'skipped/failed: {e}'
        check('W8 AMP forward finite (no NaN/Inf)', amp_ok, amp_detail)

    # W9. output spatial size == ConvBNAct(C,C,3,stride=2) on even input.
    # (A plain conv+BN is used instead of importing ConvBNAct so the self-test
    # also runs as a plain script without pulling in the whole engine package;
    # ConvBNAct's conv is exactly Conv2d(C, C, 3, stride=2, padding=1).)
    conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, groups=C,
                     bias=False)
    m_dw = HaarSubbandDownsample(C, gate=True, mode='dw')
    m_dw.eval()
    with torch.no_grad():
        s_wave = m_dw(x).shape[-2:]
        s_conv = conv(x).shape[-2:]
    check('W9 output size == conv stride-2 baseline',
          s_wave == s_conv == (H // 2, W // 2),
          f'{s_wave} vs {s_conv}')

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()
    if args.self_test_only:
        sys.exit(self_test())
    ap.error('nothing to do: pass --self-test-only')


if __name__ == '__main__':
    main()
