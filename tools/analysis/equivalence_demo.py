#!/usr/bin/env python3
"""Empirical demo of the A1 <=> 2x2-strided-conv equivalence (paper appendix).

Failure-diagnosis doc §1 proved on paper (1e-16 numerics) that:
  - the four Haar subbands form a complete orthonormal basis of R^4 (the 2x2
    block space);
  - a per-channel 1x1 over the four subbands == a 2x2 depthwise stride-2 conv;
  - the baseline's 3x3 depthwise stride-2 (pad=1) conv reproduces A1 EXACTLY by
    putting that 2x2 kernel in its bottom-right corner and zeroing the rest.
=> A1's function class is a strict SUBSET of the operator it replaced
   (4 degrees of freedom vs 9).

This script demonstrates all of it end-to-end with real conv2d calls (no
engine imports; the Haar matrix is inlined so the demo is self-contained and
directly citable):

    python tools/analysis/equivalence_demo.py --self-test-only

Also prints the operator-class upper-bound table (doc §5.3 material) reminder.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

# orthonormal Haar, rows = subband kernels flattened row-major over the 2x2
# block: [p00, p01, p10, p11]. H is orthogonal: H @ H.T == I.
HAAR = 0.5 * torch.tensor([
    [1.,  1.,  1.,  1.],   # LL
    [1.,  1., -1., -1.],   # LH
    [1., -1.,  1., -1.],   # HL
    [1., -1., -1.,  1.],   # HH
], dtype=torch.float64)
SUBBANDS = ('LL', 'LH', 'HL', 'HH')


def haar_dwt(x):
    """Channel-major Haar DWT, stride 2: (B,C,H,W) -> (B,4C,H/2,W/2)."""
    C = x.shape[1]
    w = HAAR.to(x.dtype).view(4, 1, 2, 2).repeat(C, 1, 1, 1)  # (4C,1,2,2)
    return F.conv2d(x, w, stride=2, groups=C)


def a1_forward(x, a):
    """A1 = HaarDWT + grouped 1x1 (4->1 per channel). a: (C, 4) coefficients."""
    C = x.shape[1]
    y = haar_dwt(x)
    proj = a.to(x.dtype).view(C, 4, 1, 1)   # groups=C: group c eats [4c:4c+4]
    return F.conv2d(y, proj, groups=C)       # 1x1 weight, stride 1


def conv3x3_from_2x2(w2x2):
    """3x3 depthwise stride-2 pad-1 kernel whose BOTTOM-RIGHT 2x2 equals w2x2.

    For even input sizes the padded border taps multiply zeros of the kernel,
    so this conv is exactly the 2x2 strided conv on every aligned window.
    """
    C = w2x2.shape[0]
    K = torch.zeros(C, 1, 3, 3, dtype=w2x2.dtype)
    K[:, 0, 1:, 1:] = w2x2.view(C, 2, 2)
    return K


def demo():
    print('[equivalence] Haar basis and A1 <=> 2x2-strided-conv demo')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    torch.manual_seed(0)
    C, B, H, W = 5, 2, 16, 16
    x = torch.randn(B, C, H, W, dtype=torch.float64)

    # 0. H is orthonormal and complete (rank 4) -- the basis claim
    check('H @ H.T == I (orthonormal)',
          torch.allclose(HAAR @ HAAR.T, torch.eye(4, dtype=torch.float64),
                         atol=1e-12))
    check('rank(H) == 4 (complete basis of the 2x2 block space)',
          int(torch.linalg.matrix_rank(HAAR)) == 4)

    # 1. direction "conv -> A1": ANY random 3x3-depthwise-s2 kernel restricted
    #    to its bottom-right 2x2 is reproduced EXACTLY by A1 coefficients
    #    a = w_flat @ H^T   (since w = a @ H  and  H^{-1} = H^T)
    w = torch.randn(C, 4, dtype=torch.float64)          # flat 2x2 kernels
    K = conv3x3_from_2x2(w)
    y_conv = F.conv2d(x, K, stride=2, padding=1, groups=C)
    a = w @ HAAR.T                                       # (C,4)
    y_a1 = a1_forward(x, a)
    err1 = float((y_conv - y_a1).abs().max())
    check('conv3x3(bottom-right 2x2=w) == A1(a=w H^T) exactly',
          err1 < 1e-10, f'max abs err {err1:.2e}')

    # 2. direction "A1 -> conv": ANY random subband coefficient set is a 2x2
    #    strided conv with kernel w = a @ H
    a2 = torch.randn(C, 4, dtype=torch.float64)
    w2 = a2 @ HAAR
    y_a1b = a1_forward(x, a2)
    y_convb = F.conv2d(x, conv3x3_from_2x2(w2), stride=2, padding=1, groups=C)
    err2 = float((y_a1b - y_convb).abs().max())
    check('A1(a) == conv3x3(w = a H) exactly', err2 < 1e-10,
          f'max abs err {err2:.2e}')

    # 3. strict subset: a 3x3 kernel with energy OUTSIDE the bottom-right 2x2
    #    (e.g. top-left tap) CANNOT be represented by A1 -- least-squares
    #    residual over the aligned windows is strictly positive
    K3 = torch.zeros(C, 1, 3, 3, dtype=torch.float64)
    K3[:, 0, 0, 0] = 1.0                                 # top-left tap only
    y3 = F.conv2d(x, K3, stride=2, padding=1, groups=C)
    # best A1 approximation: solve for a on the Haar coefficients of the output?
    # simplest rigorous residual: project K3's 3x3 taps -- A1 can only express
    # the bottom-right 2x2 block, so residual energy = the other 5 taps
    representable = torch.zeros_like(K3)
    representable[:, 0, 1:, 1:] = K3[:, 0, 1:, 1:]
    resid = float((K3 - representable).pow(2).sum())
    y_best = a1_forward(x, (representable[:, 0, 1:, 1:].reshape(C, 4)) @ HAAR.T)
    err3 = float((y3 - y_best).abs().max())
    check('3x3 kernel with top-left tap is NOT representable (residual > 0)',
          resid > 0 and err3 > 1e-6, f'residual {resid:.3e}, out err {err3:.2e}')

    # 4. degrees-of-freedom statement, printed for the appendix
    print(f'  [info] DoF: baseline 3x3 depthwise = 9 per channel; '
          f'A1 (Haar + grouped 1x1) = 4 per channel -> strict subset')
    print(f'  [info] operator-class upper bounds (failure-diagnosis §2, '
          f'wavelet_suppress self-test): salt-pepper clip +5.74 dB vs '
          f'per-subband scale +1.23 dB; gaussian shrink +5.11 dB vs '
          f'clip +2.07 dB')

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()
    # the demo IS the self-test (doc §5.1: "20 lines, cite in the appendix")
    sys.exit(demo())


if __name__ == '__main__':
    main()
