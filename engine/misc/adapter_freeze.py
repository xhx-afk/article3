"""SRFF-V1.2 adapter-only 训练的冻结 / 审计 / eval 模式 / checkpoint 辅助。

依据 ``SRFF-V1.2-冻结基线可学习双专家适配器-实现文档-Agent.md`` §3。集中实现，最小化对
核心训练引擎（_solver / det_solver / det_engine）的侵入：这些引擎仅在 adapter 模式下调用本模块。

约定：adapter 白名单 pattern 为正则，匹配**去掉 DDP ``module.`` 前缀后**的参数/模块名
（如 ``^encoder\\.srff_blocks\\.0\\.``）。冻结在 DDP 包装前执行，故彼时名字本就无前缀；
frozen_eval 在 DDP 后执行，故统一用 ``_strip_module`` 归一化（满足 §6 item19）。
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.dropout import _DropoutNd

# frozen_eval 需置 eval 的“有状态/随机”叶层：BN（冻结 running stats）、Dropout（关闭随机）。
_FROZEN_EVAL_TYPES = (_BatchNorm, _DropoutNd)


def _strip_module(name: str) -> str:
    return name[7:] if name.startswith('module.') else name


def is_adapter_name(name: str, patterns) -> bool:
    n = _strip_module(name)
    return any(re.search(p, n) for p in patterns)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def git_commit(repo=None):
    try:
        c = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo or os.getcwd(),
                           capture_output=True, text=True, timeout=10)
        return c.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 冻结与校验
# ---------------------------------------------------------------------------
def freeze_non_adapter(model: nn.Module, patterns):
    """非 adapter 参数 requires_grad=False；adapter 参数 True。返回 (trainable_names, frozen_count)。

    在 DDP 包装前调用（此时名字无 module. 前缀）。空 pattern 或匹配不到参数直接报错（§3.2）。
    """
    if not patterns:
        raise RuntimeError('adapter_only_training.trainable_param_patterns 不得为空')
    trainable, frozen = [], 0
    for name, p in model.named_parameters():
        if is_adapter_name(name, patterns):
            p.requires_grad_(True)
            trainable.append(_strip_module(name))
        else:
            p.requires_grad_(False)
            frozen += 1
    if not trainable:
        raise RuntimeError(f'adapter pattern {patterns} 未匹配到任何参数，拒绝训练（避免空 optimizer）')
    # BN running stats 是 buffer，无 requires_grad；其“冻结”由 frozen_eval（非 adapter 恒 eval）保证。
    return sorted(trainable), frozen


def verify_trainable_set(model: nn.Module, patterns):
    """校验 requires_grad=True 集合 == adapter 白名单集合；任何越界即报错（§3.2）。返回可训练名列表。"""
    bad = [_strip_module(n) for n, p in model.named_parameters()
           if p.requires_grad and not is_adapter_name(n, patterns)]
    if bad:
        raise RuntimeError(f'以下非 adapter 参数仍 requires_grad=True（冻结失败）：{bad[:8]}{" ..." if len(bad) > 8 else ""}')
    adapter = {_strip_module(n) for n, p in model.named_parameters() if is_adapter_name(n, patterns)}
    trainable = {_strip_module(n) for n, p in model.named_parameters() if p.requires_grad}
    if adapter != trainable:
        raise RuntimeError(f'adapter 白名单与 requires_grad 集合不一致：仅白名单{sorted(adapter - trainable)[:4]} '
                           f'仅可训练{sorted(trainable - adapter)[:4]}')
    return sorted(trainable)


def apply_adapter_train_mode(model: nn.Module, patterns):
    """frozen_eval（§3.2）：整体 train()，再把**非 adapter 的 BN/Dropout 叶层**置 eval()。

    不能整体 eval()：HybridEncoder 在 eval 下改用 eval_spatial_size 预计算的 pos_embed，
    与多尺度/任意尺寸的训练输入不匹配（tensor size mismatch）。整体 train() 保留 encoder
    的动态 pos_embed；仅把 BN/Dropout 叶层 eval() 即可在数学上冻结基线（BN 用 running
    stats、dropout 关闭），adapter 子树保持 train()。
    """
    model.train()
    for name, mod in model.named_modules():
        clean = _strip_module(name)
        in_adapter = bool(clean) and any(re.search(p, clean) or re.search(p, clean + '.') for p in patterns)
        if (not in_adapter) and isinstance(mod, _FROZEN_EVAL_TYPES):
            mod.eval()


def count_non_adapter_train_mode(model: nn.Module, patterns):
    """诊断用：统计非 adapter 子树里仍处于 training=True 的 BN/Dropout 数（应为 0）。"""
    bad = 0
    for name, mod in model.named_modules():
        clean = _strip_module(name)
        in_adapter = clean and any(re.search(p, clean) or re.search(p, clean + '.') for p in patterns)
        if (not in_adapter) and isinstance(mod, _FROZEN_EVAL_TYPES):
            if mod.training:
                bad += 1
    return bad


def adapter_state_dict(model: nn.Module, patterns):
    """抽取 adapter-only state（参数 + persistent buffer），key 去 module. 前缀。"""
    out = {}
    for name, p in model.named_parameters():
        if is_adapter_name(name, patterns):
            out[_strip_module(name)] = p.detach().cpu()
    for name, b in model.named_buffers():
        if is_adapter_name(name, patterns):
            out[_strip_module(name)] = b.detach().cpu()
    return out


# ---------------------------------------------------------------------------
# baseline 加载审计（§3.3）
# ---------------------------------------------------------------------------
def audit_baseline_load(model: nn.Module, ckpt_path, patterns, expected_source='ema.module'):
    """核对 baseline 加载：非 adapter 的模型 key 必须全部命中 baseline、无 shape mismatch；
    adapter key 允许 missing（新模块）。返回 audit dict（供写 baseline_load_audit.json）。

    任一 hard-fail 条件成立即抛 RuntimeError（§3.3：baseline key 缺失/shape 不同/继续训练均禁止）。
    """
    state = torch.load(ckpt_path, map_location='cpu')
    if 'ema' in state and isinstance(state['ema'], dict) and 'module' in state['ema']:
        pre, source = state['ema']['module'], 'ema.module'
    elif 'model' in state:
        pre, source = state['model'], 'model'
    else:
        raise RuntimeError(f'baseline checkpoint 无 ema.module / model 字段: {ckpt_path}')
    pre = {_strip_module(k): v for k, v in pre.items()}
    model_state = {_strip_module(k): v for k, v in model.state_dict().items()}

    matched, missing, shape_mismatch = [], [], []
    for k, v in model_state.items():
        if k in pre:
            if tuple(v.shape) == tuple(pre[k].shape):
                matched.append(k)
            else:
                shape_mismatch.append(k)
        else:
            missing.append(k)
    unexpected = [k for k in pre if k not in model_state]

    bad_missing = [k for k in missing if not is_adapter_name(k, patterns)]
    adapter_missing = [k for k in missing if is_adapter_name(k, patterns)]
    bad_unexpected = [k for k in unexpected if not is_adapter_name(k, patterns)]

    audit = {
        'checkpoint_path': str(ckpt_path),
        'checkpoint_sha256': sha256_file(ckpt_path),
        'source': source, 'expected_source': expected_source,
        'matched_count': len(matched),
        'missing_non_adapter': bad_missing,
        'missing_adapter_expected': adapter_missing,
        'unexpected': unexpected,
        'shape_mismatch': shape_mismatch,
    }
    hard_fail = []
    if bad_missing:
        hard_fail.append(f'baseline 缺失非 adapter key {len(bad_missing)} 个，例如 {bad_missing[:5]}')
    if shape_mismatch:
        hard_fail.append(f'shape 不一致 {len(shape_mismatch)} 个，例如 {shape_mismatch[:5]}')
    if source != expected_source:
        hard_fail.append(f'checkpoint 来源应为 {expected_source}，实际 {source}')
    audit['ok'] = not hard_fail
    audit['hard_fail'] = hard_fail
    if hard_fail:
        raise RuntimeError('baseline 加载审计失败：\n  - ' + '\n  - '.join(hard_fail))
    return audit


# ---------------------------------------------------------------------------
# adapter checkpoint（§3.4）
# ---------------------------------------------------------------------------
def build_adapter_checkpoint(model, optimizer, epoch, patterns, baseline_sha256=None,
                             config_sha256=None, git=None, extra=None):
    """构造 adapter checkpoint：完整 model + adapter-only state + optimizer + 身份元数据；无伪 ema 字段。"""
    de = model.module if hasattr(model, 'module') else model
    ck = {
        'model': de.state_dict(),
        'adapter': adapter_state_dict(model, patterns),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
        'baseline_sha256': baseline_sha256,
        'config_sha256': config_sha256,
        'git_commit': git,
        'use_ema': False,
    }
    if extra:
        ck.update(extra)
    return ck


def save_adapter_checkpoint(path, ckpt):
    Path(os.path.dirname(str(path)) or '.').mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))
