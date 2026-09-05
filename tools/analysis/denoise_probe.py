#!/usr/bin/env python3
"""Coarse denoising probe: filter the degraded val images and produce a
complete, drop-in val image directory per filter.

- images matching --pattern (default '_(PDC|GDC)_', the noise-dominated codes)
  are filtered and written out;
- non-matching images become relative symlinks (os.symlink) so the directory
  is complete without duplicating data; if the filesystem refuses symlinks a
  plain copy is used with a loud warning;
- filter parameters are fixed and recorded in <out-root>/<filter>/PARAMS.json
  (this is a coarse probe -- no parameter tuning, per §9);
- output pixels are PNG-encoded (under the ORIGINAL file name so the existing
  ann_file keeps working). If the source image was JPEG this avoids mixing a
  "denoising gain" with a second JPEG compression loss;
- no retraining, no ann.json change: the user only swaps
  val_dataloader.dataset.img_folder.

Self-test:
    python tools/analysis/denoise_probe.py --self-test-only
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # only degradation_paired / denoise_probe need cv2
    cv2 = None

FILTERS = {
    'median3': {
        'fn': lambda img: cv2.medianBlur(img, 3),
        'params': {'op': 'cv2.medianBlur', 'ksize': 3},
    },
    'median5': {
        'fn': lambda img: cv2.medianBlur(img, 5),
        'params': {'op': 'cv2.medianBlur', 'ksize': 5},
    },
    'bilateral': {
        'fn': lambda img: cv2.bilateralFilter(img, d=5, sigmaColor=75,
                                              sigmaSpace=75),
        'params': {'op': 'cv2.bilateralFilter', 'd': 5, 'sigmaColor': 75,
                   'sigmaSpace': 75},
    },
}


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _remove_existing(path):
    """Remove an existing entry BEFORE writing.

    CRITICAL: opening an existing symlink with mode 'wb' truncates THROUGH
    the link, i.e. it would destroy the source image in the original val
    directory. Removing first makes reruns safe and idempotent.
    """
    if os.path.lexists(path):
        os.remove(path)


def _link_or_copy(src, dst):
    """Relative symlink preferred; fall back to an explicit copy."""
    _remove_existing(dst)
    try:
        rel = os.path.relpath(src, start=os.path.dirname(dst) or '.')
        os.symlink(rel, dst)
        return 'symlink'
    except OSError:
        shutil.copyfile(src, dst)
        return 'copy'


def process_directory(img_dir, out_root, pattern, filters):
    """Build <out_root>/<filter>/ for each filter; returns per-filter stats."""
    compiled = re.compile(pattern)
    files = sorted(os.listdir(img_dir))
    stats = {}
    for filt in filters:
        if filt not in FILTERS:
            sys.exit(f'[denoise_probe] unknown filter {filt!r}; '
                     f'choices: {sorted(FILTERS)}')
        out_dir = Path(out_root) / filt
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / 'PARAMS.json').open('w', encoding='utf-8') as f:
            json.dump({'filter': filt, **FILTERS[filt]['params'],
                       'output_codec': 'png (under original file name)'},
                      f, indent=2)
        n_filtered = n_linked = n_copied = 0
        for name in files:
            src = os.path.join(img_dir, name)
            if not os.path.isfile(src):
                continue
            dst = str(out_dir / name)
            if compiled.search(name):
                img = cv2.imread(src, cv2.IMREAD_COLOR)
                if img is None:
                    sys.exit(f'[denoise_probe] failed to read {src}')
                # ACTUALLY APPLY THE FILTER (this was once missing, which made
                # all three filter dirs identical re-encodes of the input)
                filtered = FILTERS[filt]['fn'](img)
                # encode as PNG with RGB channel order so PIL/torchvision
                # decode the same colors the model was trained on
                ok, buf = cv2.imencode(
                    '.png', cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
                if not ok:
                    sys.exit(f'[denoise_probe] PNG encode failed for {src}')
                _remove_existing(dst)
                with open(dst, 'wb') as f:
                    f.write(buf.tobytes())
                n_filtered += 1
            else:
                mode = _link_or_copy(os.path.abspath(src),
                                     os.path.abspath(dst))
                if mode == 'symlink':
                    n_linked += 1
                else:
                    n_copied += 1
        stats[filt] = {'filtered': n_filtered, 'symlinked': n_linked,
                       'copied': n_copied}
        print(f'[denoise_probe] {filt}: filtered {n_filtered}, symlinked '
              f'{n_linked}, copied {n_copied} -> {out_dir}')
        # read-back verification: every output file must decode as an image
        bad = verify_directory(out_dir, files)
        if bad:
            sys.exit(f'[denoise_probe] {filt}: {len(bad)} unreadable output '
                     f'file(s), e.g. {bad[:5]} -- aborted')
    return stats


def verify_directory(out_dir, names):
    """Return the list of output files that do not decode as images."""
    bad = []
    for name in names:
        p = str(Path(out_dir) / name)
        if not os.path.isfile(p):
            bad.append(name)
            continue
        if cv2.imread(p, cv2.IMREAD_COLOR) is None:
            bad.append(name)
    return bad


def run(args):
    img_abs = os.path.abspath(args.img_dir)
    out_abs = os.path.abspath(args.out_root)
    if img_abs == out_abs or img_abs.startswith(out_abs + os.sep) \
            or out_abs.startswith(img_abs + os.sep):
        sys.exit('[denoise_probe] --img-dir and --out-root must be disjoint '
                 'directories (refusing to write into the source data)')
    print('[denoise_probe] output pixels are PNG-encoded under the original '
          'file name: no second JPEG compression, ann_file stays valid.')
    stats = process_directory(args.img_dir, args.out_root, args.pattern,
                              args.filters)
    n_in = len([f for f in os.listdir(args.img_dir)
                if os.path.isfile(os.path.join(args.img_dir, f))])
    for filt, s in stats.items():
        assert s['filtered'] + s['symlinked'] + s['copied'] == n_in, \
            f'file count mismatch for {filt}'
    print('[denoise_probe] done. Point val_dataloader.dataset.img_folder at '
          f'{args.out_root}/<filter> to evaluate a filter.')


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _salt_pepper(img, amount, seed):
    rng = np.random.RandomState(seed)
    out = img.copy()
    m = rng.random_sample(img.shape[:2]) < amount
    out[m] = np.where(rng.random_sample(m.sum()).reshape(-1, 1) < 0.5,
                      0, 255).astype(img.dtype)
    return out


def self_test():
    print('[self-test] denoise_probe')
    failures = []

    def check(name, cond, detail=''):
        print(f'  [{"ok" if cond else "FAIL"}] {name}'
              + (f' ({detail})' if detail and not cond else ''))
        if not cond:
            failures.append(name)

    rng = np.random.RandomState(0)
    clean = np.full((120, 160, 3), 128, dtype=np.uint8)
    clean[20:60, 40:110] = (60, 140, 200)
    clean = rng.randint(0, 30, clean.shape).astype(np.uint8) * 0 + clean
    # light texture so images are not perfectly flat
    clean = clean + rng.randint(-3, 4, clean.shape).astype(np.int16)
    clean = np.clip(clean, 0, 255).astype(np.uint8)

    # 1. salt & pepper -> median3 gains >= 8 dB
    noisy = _salt_pepper(clean, 0.05, seed=1)
    den = cv2.medianBlur(noisy, 3)
    gain_sp = psnr(den, clean) - psnr(noisy, clean)
    check('salt&pepper 5%: median3 PSNR gain >= 8 dB', gain_sp >= 8.0,
          f'gain={gain_sp:.2f} dB')

    # 2. gaussian blur -> median3 gains < 1 dB (median cannot undo blur)
    blurred = cv2.GaussianBlur(clean, (5, 5), 1.2)
    den_b = cv2.medianBlur(blurred, 3)
    gain_blur = psnr(den_b, clean) - psnr(blurred, clean)
    check('gaussian blur: median3 PSNR gain < 1 dB', gain_blur < 1.0,
          f'gain={gain_blur:.2f} dB')

    # 3/4. end-to-end directory processing with symlinks
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix='denoise_selftest_'))
    img_dir = tmp / 'val'
    img_dir.mkdir()
    names = ['000001_PDC_00001.jpg', '000002_GDC_00002.jpg',
             '000003_ODC_00003.jpg']
    for i, name in enumerate(names):
        cv2.imwrite(str(img_dir / name), clean + i)

    out_root = tmp / 'probe'
    process_directory(str(img_dir), str(out_root), '_(PDC|GDC)_',
                      ['median3', 'median5', 'bilateral'])

    out_dir = out_root / 'median3'
    in_files = sorted(os.listdir(img_dir))
    out_files = sorted(f for f in os.listdir(out_dir) if f != 'PARAMS.json')
    check('output file count == input file count',
          out_files == in_files, f'{out_files} vs {in_files}')

    with (out_dir / 'PARAMS.json').open(encoding='utf-8') as f:
        params = json.load(f)
    check('PARAMS.json records the filter parameters',
          params['filter'] == 'median3' and params['ksize'] == 3)

    linked = out_dir / names[2]  # ODC -> must NOT be filtered
    is_link = os.path.islink(str(linked))
    if is_link:
        check('unmatched file is a symlink to the original',
              os.path.realpath(str(linked)) ==
              os.path.realpath(str(img_dir / names[2])))
    else:
        print('  [SKIP] symlink unsupported on this filesystem '
              '(copy fallback used); verifying bytes instead')
    if not is_link or True:
        with open(linked, 'rb') as fa, open(img_dir / names[2], 'rb') as fb:
            same = fa.read() == fb.read()
        check('unmatched file bytes identical to original', same)
    # matched file must differ from the original (it was actually filtered)
    with open(out_dir / names[0], 'rb') as fa, \
            open(img_dir / names[0], 'rb') as fb:
        check('matched file content changed (filter applied)',
              fa.read() != fb.read())
    # regression lock for "filter never applied": the three filters MUST
    # produce pairwise-different outputs (a re-encode-only bug makes them
    # identical and silently turns the probe into a no-op)
    h = [hashlib.md5(open(out_root / f / names[0], 'rb').read()).hexdigest()
         for f in ('median3', 'median5', 'bilateral')]
    check('median3 / median5 / bilateral outputs pairwise different',
          len(set(h)) == 3, str(h))

    # 5. regression lock: a RERUN on the same out-root must never modify the
    #    input files (an old version truncated the originals through stale
    #    symlinks) and all outputs must stay readable
    before = {n: open(img_dir / n, 'rb').read() for n in names}
    process_directory(str(img_dir), str(out_root), '_(PDC|GDC)_', ['median3'])
    after = {n: open(img_dir / n, 'rb').read() for n in names}
    check('rerun does not touch input files', before == after)
    check('all outputs readable after rerun',
          not verify_directory(out_dir, names))

    if failures:
        print(f'SELF-TESTS FAILED: {failures}')
        return 1
    print('ALL SELF-TESTS PASSED')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--img-dir', default=None,
                    help='original val image directory')
    ap.add_argument('--pattern', default='_(PDC|GDC)_',
                    help="regex selecting images to filter (default '_"
                         "(PDC|GDC)_')")
    ap.add_argument('--filters', nargs='+', default=['median3', 'median5',
                                                     'bilateral'],
                    choices=sorted(FILTERS))
    ap.add_argument('--out-root', default=None,
                    help='one complete val directory per filter is created '
                         'under this root')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args()

    if args.self_test_only:
        sys.exit(self_test())
    if not args.img_dir or not args.out_root:
        ap.error('--img-dir and --out-root are required '
                 '(unless --self-test-only)')
    run(args)


if __name__ == '__main__':
    main()
