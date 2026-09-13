"""生成 / 终结 SRFF-V1.1 三态因果诊断的身份清单 identity_manifest.json（仅依赖标准库）。

依据 ``SRFF-V1.1-三态门控因果诊断-实现文档-Agent.md`` §7：记录 git 身份、config/checkpoint
的 SHA256、checkpoint 来源、seed、三态 mode，并显式声明 training_performed=false。

两阶段用法::
    # eval 前：写入 before（config/checkpoint 当前 SHA）
    python tools/wood/make_identity_manifest.py \
      --config <cfg.yml> --checkpoint <best_stg2.pth> \
      --out <causal_dir>/identity_manifest.json

    # eval 后：写入 after 并核对 checkpoint 未被改动
    python tools/wood/make_identity_manifest.py \
      --checkpoint <best_stg2.pth> --out <causal_dir>/identity_manifest.json --finalize

严禁在文件缺失时写入占位 SHA：任一必需文件不存在即非零退出。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys


def sha256_file(path, chunk=1 << 20):
    if not os.path.isfile(path):
        raise SystemExit(f'[manifest] 文件不存在，拒绝写入占位 SHA: {path}')
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def git_info(repo):
    try:
        c = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
                           capture_output=True, text=True, timeout=10)
        s = subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                           capture_output=True, text=True, timeout=10)
        commit = c.stdout.strip() if c.returncode == 0 else None
        dirty = (bool(s.stdout.strip()) if s.returncode == 0 else None)
        return commit, dirty
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description='生成/终结三态因果诊断身份清单')
    ap.add_argument('--out', required=True, help='identity_manifest.json 路径')
    ap.add_argument('--checkpoint', required=True, help='V1.1 best_stg2.pth（N/O/I 共用同一份）')
    ap.add_argument('--config', default=None, help='V1.1 配置 YAML（before 阶段必填）')
    ap.add_argument('--repo', default=None, help='git 仓库根（默认当前工作目录）')
    ap.add_argument('--modes', default='auto,force_off,force_on')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--source', default='ema', help='checkpoint 来源（ema/model）')
    ap.add_argument('--finalize', action='store_true', help='eval 后写入 checkpoint_sha256_after 并核对')
    args = ap.parse_args()

    repo = args.repo or os.getcwd()
    ckpt_after = sha256_file(args.checkpoint)  # 当前 checkpoint SHA（拒绝占位）

    if args.finalize:
        if not os.path.isfile(args.out):
            raise SystemExit(f'[manifest] --finalize 需要已存在的清单: {args.out}')
        with open(args.out, 'r', encoding='utf-8') as f:
            mf = json.load(f)
        mf['checkpoint_sha256_after'] = ckpt_after
        before = mf.get('checkpoint_sha256_before')
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(mf, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        same = (before == ckpt_after)
        print(f'[manifest] checkpoint_sha256_before={before}')
        print(f'[manifest] checkpoint_sha256_after ={ckpt_after}')
        print(f'[manifest] checkpoint 未改动: {same}')
        if not same:
            print('[manifest] 严重：checkpoint SHA 前后不一致，本轮身份不可信', file=sys.stderr)
            raise SystemExit(3)
        return

    # before 阶段：需要 config
    if not args.config:
        raise SystemExit('[manifest] before 阶段必须提供 --config')
    commit, dirty = git_info(repo)
    mf = {
        'git_commit': commit,
        'git_dirty': dirty,
        'config_path': os.path.abspath(args.config),
        'config_sha256': sha256_file(args.config),
        'checkpoint_path': os.path.abspath(args.checkpoint),
        'checkpoint_sha256_before': ckpt_after,
        'checkpoint_sha256_after': None,
        'checkpoint_source': args.source,
        'seed': args.seed,
        'modes': [m.strip() for m in args.modes.split(',') if m.strip()],
        'training_performed': False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(mf, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(f'[manifest] 写入 {args.out}')
    print(f'[manifest] git_commit={commit} git_dirty={dirty}')
    print(f'[manifest] config_sha256={mf["config_sha256"]}')
    print(f'[manifest] checkpoint_sha256_before={ckpt_after}')
    if commit is None:
        print('[manifest] 注意：非 git 仓库，git_commit=null（身份追溯依赖 SHA256）', file=sys.stderr)


if __name__ == '__main__':
    main()
