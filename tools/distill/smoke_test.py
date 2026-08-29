#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 蒸馏落地自检（S1–S8）。

单卡、不需要数据集，跑完 2 分钟内给结论：

  python tools/distill/smoke_test.py -c configs/deim_dfine/custom/neu_det_s_distill.yml
  # 覆盖项与训练一致，例如 Run-A：
  python tools/distill/smoke_test.py -c configs/deim_dfine/custom/neu_det_s_distill.yml \
      -u 'distiller.layers=[1]'

断言清单：
  S1 教师输出网格形状（480/640/800）
  S2 学生编码器三层输出形状
  S3 教师网格 == 学生 P4 网格（多尺度各测一次，§3.1 不变量）
  S4 loss_distill 有限、requires_grad、可 backward
  S5 教师零可训练参数、教师不进 distiller.state_dict()（C2/C6）
  S6 model.state_dict() 键集合与蒸馏前完全相同（C2）
  S7 打印 aligner 参数量 / 教师参数量 / 教师前向显存增量（供手册记录）
  S8 loss_distill.backward() 后 HybridEncoder 梯度范数 > 0（最重要的一条）
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig, yaml_utils
from engine.misc import dist_utils
from engine.deim.distill import EncoderFeatureHook


def fmt(v):
    return f'{v / 1e6:.1f}M' if v >= 1e6 else f'{v:,}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-u', '--update', nargs='+', default=[], help='配置覆盖，如 distiller.layers=[1]')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'smoke_test on device: {device}')

    updates = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(args.config, **updates)
    # 离线自检不下载 backbone 预训练权重
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # ---------- 基线键集（S6 参照）：distiller 构建前抓 model 的 state_dict 键 ----------
    model = cfg.model.to(device)
    baseline_keys = set(model.state_dict().keys())
    n_baseline_params = sum(p.numel() for p in model.parameters())

    distiller = cfg.distiller.to(device)
    teacher = distiller.teacher
    # CPU 上 fp16 autocast 不可用，退到 fp32（自检只验逻辑，不验精度）
    if device.type == 'cpu':
        teacher.autocast_dtype = torch.float32
        print('  [note] CPU 环境：教师 autocast 退为 fp32')
    if teacher.patch_size != 16:
        print(f'  [warn] 教师 patch_size={teacher.patch_size} != 16，'
              '§3.1 的 P4 精确对齐不变量按实际 patch_size 判定')

    hook = EncoderFeatureHook(dist_utils.de_parallel(model).encoder)

    # ---------- S1: 教师输出网格 ----------
    print('\n[S1] teacher output grids')
    teacher.eval()
    for size in (480, 640, 800):
        imgs = torch.rand(2, 3, size, size, device=device)
        feat = teacher(imgs)
        expect = (teacher.embed_dim, size // teacher.patch_size, size // teacher.patch_size)
        assert tuple(feat.shape[1:]) == expect, f'S1 fail: {tuple(feat.shape)} != {expect}'
        print(f'  size={size} -> {tuple(feat.shape)} OK')

    # ---------- S2 + S3: 学生编码器形状 & P4 精确对齐 ----------
    print('\n[S2][S3] student encoder outputs & P4 exact alignment')
    was_training = model.training
    model.eval()                     # 见 S8 注释：eval 避开 denoising 组对 targets 的要求
    p4_idx = min(range(len(distiller.student_strides)),
                 key=lambda i: abs(distiller.student_strides[i] - teacher.patch_size))
    for size in (480, 640, 800):
        imgs = torch.rand(2, 3, size, size, device=device)
        hook.enabled = True
        _ = model(imgs)
        feats = hook.pop()
        for i, f in enumerate(feats):
            s = size // distiller.student_strides[i]
            assert tuple(f.shape) == (2, distiller.student_channels[i], s, s), \
                f'S2 fail: layer{i} {tuple(f.shape)} != (2, {distiller.student_channels[i]}, {s}, {s})'
        # §3.1 不变量：stride == patch_size 的层必须逐像素精确对齐
        if distiller.exact.get(p4_idx, False):
            t_grid = (size // teacher.patch_size, size // teacher.patch_size)
            s_grid = tuple(feats[p4_idx].shape[-2:])
            assert s_grid == t_grid, \
                f'S3 fail: P4 网格 {s_grid} != 教师网格 {t_grid}（输入 {size}）'
        print(f'  size={size} -> encoder ' +
              '/'.join(str(tuple(f.shape[-2:])) for f in feats) +
              f' | teacher grid={size // teacher.patch_size} OK (exact layer={p4_idx})')

    # ---------- S4: 损失有限、可反传 ----------
    print('\n[S4] loss_distill forward/backward')
    imgs = torch.rand(2, 3, 640, 640, device=device)
    hook.enabled = True
    _ = model(imgs)
    out = distiller(imgs, hook.pop(), epoch=0)
    loss = out['loss_distill']
    assert torch.isfinite(loss).all(), f'S4 fail: loss 非有限 {loss}'
    assert loss.requires_grad, 'S4 fail: loss_distill.requires_grad is False（计算图断了）'
    loss.backward()
    print(f'  loss_distill={loss.item():.4f}  finite & backward OK')

    # ---------- S5: 教师零可训练参数、不进 state_dict（C2/C6） ----------
    print('\n[S5] teacher frozen & excluded from state_dict')
    n_teacher_train = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    assert n_teacher_train == 0, f'S5 fail: 教师有 {n_teacher_train} 个可训练参数'
    sd_keys = list(distiller.state_dict().keys())
    assert all(k.startswith('aligners.') for k in sd_keys), \
        f'S5 fail: distiller.state_dict 出现非 aligner 键: {sd_keys[:8]}'
    assert all('teacher' not in k and 'model' not in k for k in sd_keys), \
        f'S5 fail: 教师权重进了 distiller.state_dict: {sd_keys[:8]}'
    print(f'  distiller.state_dict keys: {sd_keys} OK')

    # ---------- S6: model 键集合不变（C2） ----------
    print('\n[S6] model.state_dict unchanged')
    keys_now = set(model.state_dict().keys())
    assert keys_now == baseline_keys, \
        f'S6 fail: model 键集合变化: 新增 {sorted(keys_now - baseline_keys)[:8]}'
    n_model_params = sum(p.numel() for p in model.parameters())
    assert n_model_params == n_baseline_params, 'S6 fail: model 参数量变化'
    print(f'  {len(baseline_keys)} keys, {fmt(n_model_params)} params OK')

    # ---------- S7: 记录量级（供手册填写） ----------
    print('\n[S7] magnitude report')
    n_aligner = sum(p.numel() for p in distiller.parameters() if p.requires_grad)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f'  aligner trainable params : {n_aligner} ({n_aligner / 1e6:.3f}M)')
    print(f'  teacher params           : {n_teacher} ({n_teacher / 1e6:.1f}M), frozen')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _ = teacher(imgs)
        delta = torch.cuda.max_memory_allocated() - base
        print(f'  teacher fwd mem delta    : {delta / 1024 ** 2:.0f} MiB @ 640x640 bs=2')
    else:
        print('  teacher fwd mem delta    : N/A (CPU)')

    # ---------- S8: 蒸馏梯度必须真的到达编码器（最重要） ----------
    print('\n[S8] distill gradient reaches HybridEncoder')
    model.zero_grad(set_to_none=True)
    distiller.zero_grad(set_to_none=True)

    # 必须 eval：train 模式下 num_denoising>0 会走
    # get_contrastive_denoising_training_group(targets, ...)，targets=None 直接抛异常，
    # 报错信息还很难读。eval 只影响 BN/dropout，不关闭 autograd，
    # 梯度连通性照验；forward hook 在 eval 下同样触发。
    model.eval()
    hook.enabled = True
    _ = model(imgs)
    out = distiller(imgs, hook.pop(), epoch=0)
    out['loss_distill'].backward()
    if was_training:
        model.train()

    enc = dist_utils.de_parallel(model).encoder
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    gnorm = sum(g.norm().item() for g in grads)
    n_nonzero = sum(1 for g in grads if g.abs().sum() > 0)

    assert n_nonzero > 0 and gnorm > 0, (
        'loss_distill 没有把梯度传到 HybridEncoder。'
        '几乎可以肯定是 hook 抓到的张量被 detach 了，或者 aligner 输入用了 .data。'
        '此时 loss_distill 仍会正常下降（aligner 自己就能拟合），但蒸馏完全没有发生。')
    print(f'  S8 OK | encoder grad norm from loss_distill = {gnorm:.4e} | 非零梯度张量 {n_nonzero} 个')

    hook.remove()
    print('\n' + '=' * 62)
    print('ALL SMOKE TESTS PASSED (S1–S8)')


if __name__ == '__main__':
    main()
