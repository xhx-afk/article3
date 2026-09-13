"""SRFF-V1.1 局部门/双专家容量拆分的身份清单工具（仅依赖标准库）。

依据 ``SRFF-V1.1-局部门与双专家容量拆分-实现文档-Agent.md`` §7.3：

* ``--phase before``：评估前创建。若清单已存在则拒绝覆盖；读取实际文件计算
  git/config/checkpoint/代码 SHA256，记录 experiment_type=eval_only、training_performed=false、
  checkpoint_source=ema、seed=0、11 个状态与两个域；校验 checkpoint SHA256 与预期身份一致。
* ``--phase after``：评估后重新计算 git/config/checkpoint/代码身份，逐项与 before 比对，
  任一不一致即非零退出（不得补写成功状态）；一致才写入 checkpoint_sha256_after 与结束 UTC。

所有 SHA 均由本工具读取真实文件计算，禁止从命令行接收并盲信任意 SHA。
非 git 工作区：git 字段记为 null 并告警（身份追溯以 SHA256 为准），不因此拒绝。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# §1 预期 checkpoint 身份（上一轮 V1.1 fast72 seed0 best_stg2.pth）
EXPECTED_CKPT_SHA = '0c6434ebca4f1887e831b924455a2201c6bfad0a3c227411184e4c3ea31802a0'

# §7.3 需记录 SHA256 的代码文件（相对仓库根）
CODE_FILES = [
    'engine/deim/srff.py',
    'engine/deim/srff_v1_1.py',
    'engine/deim/hybrid_encoder.py',
    'tools/wood/eval_metrics.py',
    'tools/wood/summarize_srff_v11_expert_capacity.py',
]

# §2 的 11 个状态与 2 个域
STATES = ['ref_force_off', 'ref_learned',
          'mix_a002', 'mix_a005', 'mix_a010',
          'gaussian_a002', 'gaussian_a005', 'gaussian_a010',
          'trimmed_a002', 'trimmed_a005', 'trimmed_a010']
DOMAINS = ['GDC', 'PDC']

EXIT_REFUSE = 2
EXIT_MISMATCH = 3


def _die(code, msg):
    print(f'[capacity-manifest] {msg}', file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path):
    if not os.path.isfile(path):
        _die(EXIT_REFUSE, f'文件不存在，拒绝写入占位 SHA: {path}')
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def file_meta(path):
    return {'path': os.path.abspath(path), 'size': os.path.getsize(path), 'sha256': sha256_file(path)}


def git_info(repo):
    """返回 (commit, branch, dirty)；非 git 工作区返回 (None, None, None)。"""
    try:
        c = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, timeout=10)
        if c.returncode != 0:
            return None, None, None
        b = subprocess.run(['git', 'branch', '--show-current'], cwd=repo, capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', 'status', '--porcelain'], cwd=repo, capture_output=True, text=True, timeout=10)
        commit = c.stdout.strip() or None
        branch = b.stdout.strip() or None
        dirty = bool(s.stdout.strip())
        return commit, branch, dirty
    except Exception:
        return None, None, None


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def collect_identity(args, repo):
    commit, branch, dirty = git_info(repo)
    code = {}
    for rel in CODE_FILES:
        code[rel] = file_meta(os.path.join(repo, rel))
    return {
        'git_commit': commit, 'git_branch': branch, 'git_dirty': dirty,
        'config': file_meta(args.config),
        'checkpoint': file_meta(args.checkpoint),
        'code_files': code,
    }


def do_before(args, repo):
    if os.path.exists(args.out):
        _die(EXIT_REFUSE, f'清单已存在，拒绝覆盖（如需重跑请换新目录）: {args.out}')
    ident = collect_identity(args, repo)
    ckpt_sha = ident['checkpoint']['sha256']
    if ckpt_sha != args.expect_checkpoint_sha:
        _die(EXIT_REFUSE, f'checkpoint SHA256 与预期身份不一致：\n  实际={ckpt_sha}\n  预期={args.expect_checkpoint_sha}')
    if ident['git_dirty'] is True:
        _die(EXIT_REFUSE, 'git 工作区 dirty（有未提交修改），拒绝继续')
    if ident['git_commit'] is None:
        print('[capacity-manifest] 注意：非 git 工作区，git 字段记为 null（身份以 SHA256 为准）', file=sys.stderr)
    mf = {
        'phase': 'before',
        'utc_before': utcnow(),
        'experiment_type': 'eval_only',
        'training_performed': False,
        'checkpoint_source': 'ema',
        'seed': 0,
        'expect_checkpoint_sha256': args.expect_checkpoint_sha,
        'git_commit': ident['git_commit'],
        'git_branch': ident['git_branch'],
        'git_dirty': ident['git_dirty'],
        'config_path': ident['config']['path'],
        'config_size': ident['config']['size'],
        'config_sha256': ident['config']['sha256'],
        'checkpoint_path': ident['checkpoint']['path'],
        'checkpoint_size': ident['checkpoint']['size'],
        'checkpoint_sha256_before': ckpt_sha,
        'checkpoint_sha256_after': None,
        'code_files': ident['code_files'],
        'states': STATES,
        'domains': DOMAINS,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(mf, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(f'[capacity-manifest] before 写入 {args.out}')
    print(f'[capacity-manifest] git_commit={mf["git_commit"]} git_dirty={mf["git_dirty"]}')
    print(f'[capacity-manifest] checkpoint_sha256_before={ckpt_sha}')
    print(f'[capacity-manifest] config_sha256={mf["config_sha256"]}')


def do_after(args, repo):
    if not os.path.isfile(args.out):
        _die(EXIT_REFUSE, f'--phase after 需要已存在的 before 清单: {args.out}')
    with open(args.out, 'r', encoding='utf-8') as f:
        mf = json.load(f)
    ident = collect_identity(args, repo)
    mismatches = []
    if mf.get('checkpoint_sha256_before') != ident['checkpoint']['sha256']:
        mismatches.append(f"checkpoint: before={mf.get('checkpoint_sha256_before')} after={ident['checkpoint']['sha256']}")
    if mf.get('config_sha256') != ident['config']['sha256']:
        mismatches.append('config sha256 changed')
    if mf.get('git_commit') != ident['git_commit']:
        mismatches.append(f"git_commit: before={mf.get('git_commit')} after={ident['git_commit']}")
    before_code = mf.get('code_files', {})
    for rel in CODE_FILES:
        b = (before_code.get(rel) or {}).get('sha256')
        a = ident['code_files'][rel]['sha256']
        if b != a:
            mismatches.append(f'code changed: {rel}')
    if mf.get('training_performed') is not False:
        mismatches.append('training_performed != false')
    if mismatches:
        print('[capacity-manifest] 身份不一致，拒绝写入 after（非零退出）：', file=sys.stderr)
        for m in mismatches:
            print('  -', m, file=sys.stderr)
        raise SystemExit(EXIT_MISMATCH)
    mf['phase'] = 'after'
    mf['checkpoint_sha256_after'] = ident['checkpoint']['sha256']
    mf['utc_after'] = utcnow()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(mf, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(f'[capacity-manifest] after 校验通过并写入 {args.out}')
    print(f'[capacity-manifest] checkpoint 前后一致: {mf["checkpoint_sha256_before"]}')


def main():
    ap = argparse.ArgumentParser(description='SRFF-V1.1 容量拆分身份清单（before/after）')
    ap.add_argument('--phase', required=True, choices=['before', 'after'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo', default=None, help='仓库根（默认当前工作目录）')
    ap.add_argument('--expect-checkpoint-sha', default=EXPECTED_CKPT_SHA,
                    help='预期 checkpoint SHA256（默认上一轮 V1.1 身份）')
    args = ap.parse_args()
    repo = args.repo or os.getcwd()
    if args.phase == 'before':
        do_before(args, repo)
    else:
        do_after(args, repo)


if __name__ == '__main__':
    main()
