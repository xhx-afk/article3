"""SRFF-V1.2 冻结审计（实现文档 §5.2 / 用户手册 §8）。

对比原始 baseline checkpoint（取 ``ema.module``）与 V1.2 adapter checkpoint（取 ``model``）：

* 除 adapter key 外，所有共享 parameter / persistent buffer 必须 ``torch.equal``（逐位相等）；
* 分别统计 frozen_parameter_mismatch_count / frozen_buffer_mismatch_count，并输出每个 mismatch 的
  key、reason、dtype、shape、max_abs_diff；
* adapter_changed_count：零初始化的 ``out_proj`` 变为非零的 adapter 张量数（>0 表明确已训练）；
* non_finite_count：非有限张量数（须为 0）；
* ``git_commit`` 缺失或为 None → 失败（非 git 工作区加 --allow-missing-git 显式放行并告警）；
* 输出 ``frozen_audit.json`` 并打印 ``CHECK=PASS/FAIL``；任一失败返回非零退出码。

buffer/parameter 分类用 BatchNorm running-stats 命名启发式（DEIM 的持久 buffer 主要是 BN running stats）。
torch.load 与仓库既有加载路径一致（加载用户自己的可信 checkpoint）。
"""

import argparse
import json
import re
import sys

import torch

DEFAULT_PATTERNS = [r'^encoder\.srff_blocks\.0\.']
BUFFER_SUFFIXES = ('running_mean', 'running_var', 'num_batches_tracked')
EXIT_FAIL = 2


def _strip(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}


def _is_adapter(k, patterns):
    return any(re.search(p, k) for p in patterns)


def _is_buffer_key(k):
    return k.endswith(BUFFER_SUFFIXES)


def _pick_state(ck, prefer):
    if prefer == 'ema.module' and 'ema' in ck and isinstance(ck['ema'], dict) and 'module' in ck['ema']:
        return _strip(ck['ema']['module']), 'ema.module'
    if 'model' in ck:
        return _strip(ck['model']), 'model'
    raise SystemExit(f'[frozen-audit] checkpoint 无期望 state 字段（prefer={prefer}）')


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.2 冻结审计')
    ap.add_argument('--baseline-checkpoint', required=True, help='原始 baseline best_stg2.pth（取 ema.module）')
    ap.add_argument('--adapter-checkpoint', required=True, help='V1.2 adapter checkpoint（取 model）')
    ap.add_argument('--output', required=True, help='frozen_audit.json 输出路径')
    ap.add_argument('--patterns', nargs='+', default=DEFAULT_PATTERNS)
    ap.add_argument('--allow-missing-git', action='store_true',
                    help='非 git 工作区：git_commit=None 时仅告警不失败（身份以 SHA256 为准）')
    args = ap.parse_args()

    base_ck = torch.load(args.baseline_checkpoint, map_location='cpu')
    v12_ck = torch.load(args.adapter_checkpoint, map_location='cpu')
    base_sd, base_src = _pick_state(base_ck, 'ema.module')
    v12_sd, v12_src = _pick_state(v12_ck, 'model')

    fails, warns = [], []
    param_mismatch, buffer_mismatch = [], []
    adapter_keys, adapter_changed, adapter_nonfinite, preexisting_nonfinite = [], [], [], []

    for k, v in v12_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        finite = bool(torch.isfinite(v).all())
        if _is_adapter(k, args.patterns):
            adapter_keys.append(k)
            if 'out_proj' in k and float(v.abs().max()) > 0.0:
                adapter_changed.append(k)
            if not finite:
                adapter_nonfinite.append(k)          # adapter 张量必须有限
            continue
        if k not in base_sd:
            (buffer_mismatch if _is_buffer_key(k) else param_mismatch).append(
                {'key': k, 'reason': 'missing_in_baseline'})
            continue
        b = base_sd[k]
        if tuple(b.shape) != tuple(v.shape) or b.dtype != v.dtype:
            rec = {'key': k, 'reason': 'shape_or_dtype', 'baseline_shape': list(b.shape),
                   'v12_shape': list(v.shape), 'baseline_dtype': str(b.dtype), 'v12_dtype': str(v.dtype)}
            (buffer_mismatch if _is_buffer_key(k) else param_mismatch).append(rec)
            continue
        if not torch.equal(b, v):
            rec = {'key': k, 'reason': 'value', 'dtype': str(v.dtype), 'shape': list(v.shape),
                   'max_abs_diff': float((b.float() - v.float()).abs().max())}
            (buffer_mismatch if _is_buffer_key(k) else param_mismatch).append(rec)
        elif not finite:
            # 与 baseline 逐位相等的非有限 frozen 张量 = baseline 自带占位（如 decoder.anchors 的 ±inf），非本轮引入
            preexisting_nonfinite.append(k)

    for k in base_sd:
        if not _is_adapter(k, args.patterns) and k not in v12_sd:
            (buffer_mismatch if _is_buffer_key(k) else param_mismatch).append(
                {'key': k, 'reason': 'missing_in_v12'})

    git = v12_ck.get('git_commit')
    if git is None:
        if args.allow_missing_git:
            warns.append('git_commit=None（非 git 工作区，--allow-missing-git 放行；身份以 SHA256 为准）')
        else:
            fails.append('git_commit 缺失或为 None（§5.2；非 git 环境请加 --allow-missing-git）')
    if param_mismatch or buffer_mismatch:
        fails.append(f'冻结被破坏：param mismatch={len(param_mismatch)} buffer mismatch={len(buffer_mismatch)}')
    if adapter_nonfinite:
        fails.append(f'{len(adapter_nonfinite)} 个 adapter 张量非有限：{adapter_nonfinite[:5]}')
    if preexisting_nonfinite:
        warns.append(f'{len(preexisting_nonfinite)} 个 frozen 张量非有限但与 baseline 逐位相等'
                     f'（baseline 自带占位如 decoder.anchors，非本轮引入）：{preexisting_nonfinite[:5]}')
    if not adapter_keys:
        fails.append('未找到任何 adapter 参数')
    if not adapter_changed:
        fails.append('adapter 零初始化 out_proj 仍全为 0：未见训练更新（§5.2 要求 adapter_changed_count>0）')

    audit = {
        'baseline': {'path': args.baseline_checkpoint, 'source': base_src},
        'adapter_checkpoint': {'path': args.adapter_checkpoint, 'source': v12_src, 'git_commit': git,
                               'baseline_sha256': v12_ck.get('baseline_sha256'),
                               'config_sha256': v12_ck.get('config_sha256'), 'epoch': v12_ck.get('epoch')},
        'frozen_parameter_mismatch_count': len(param_mismatch),
        'frozen_buffer_mismatch_count': len(buffer_mismatch),
        'frozen_parameter_mismatches': param_mismatch,
        'frozen_buffer_mismatches': buffer_mismatch,
        'adapter_key_count': len(adapter_keys),
        'adapter_changed_count': len(adapter_changed),
        'adapter_changed_keys': adapter_changed[:20],
        'non_finite_count': len(adapter_nonfinite),
        'adapter_non_finite_keys': adapter_nonfinite[:20],
        'preexisting_nonfinite_frozen_keys': preexisting_nonfinite[:20],
        'warnings': warns,
        'failures': fails,
        'ok': not fails,
        'CHECK': 'PASS' if not fails else 'FAIL',
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

    print(f'[frozen-audit] baseline={base_src} adapter={v12_src} git_commit={git}')
    print(f'[frozen-audit] frozen_parameter_mismatch_count={len(param_mismatch)} '
          f'frozen_buffer_mismatch_count={len(buffer_mismatch)} '
          f'adapter_changed_count={len(adapter_changed)} non_finite_count={len(adapter_nonfinite)} '
          f'preexisting_nonfinite_frozen={len(preexisting_nonfinite)}')
    for w in warns:
        print(f'[frozen-audit][warn] {w}')
    for x in fails:
        print(f'[frozen-audit][fail] {x}')
    print(f'CHECK={"PASS" if not fails else "FAIL"}  (详情 {args.output})')
    if fails:
        raise SystemExit(EXIT_FAIL)


if __name__ == '__main__':
    main()
