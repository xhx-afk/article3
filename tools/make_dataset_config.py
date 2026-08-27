#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按数据集生成 DEIM 训练配置，消除 T0 里"四个锚点手改易错"的风险。

用法（放到仓库根目录的 tools/ 下运行）：

  python tools/make_dataset_config.py \
      --model s --name neu_det --num-classes 6 \
      --train-size 1440 --epoches 200 \
      --data-root /data/defect/NEU-DET

  # 一次生成五个数据集 × 两个尺寸
  python tools/make_dataset_config.py --preset all --data-root /data/defect

生成的文件 __include__ 上层的 deim_hgnetv2_{s,m}_custom.yml（模型配方，永不改动），
只覆盖与数据集相关的字段，并把互相耦合的四个增强锚点一次性算好写全，
避免"改了 epoches 却忘了改 policy.epoch"这类静默错误。

输出：configs/deim_dfine/custom/<name>_<model>.yml
"""

import argparse
import math
import os

# 官方 DEIM 配方里固定不变的两个数
STAGE_START = 4
NO_AUG_EPOCH = 12

# 官方配方对应的 total_batch_size；偏离它时 lr 按线性缩放
REF_BATCH = 32
REF_LR = {
    # model: (主 lr, backbone lr, backbone 正则写法)
    "s": (0.0004, 0.0002, "norm|bn"),
    "m": (0.0004, 0.00004, "bn"),
}

# 五个数据集的默认参数。train_size 请按你实际划分后的训练集张数改。
PRESETS = {
    # name           model_default  num_classes  train_size  epoches  subdir
    "neu_det":       dict(num_classes=6,  train_size=1440,  epoches=200, subdir="NEU-DET"),
    "gc10_det":      dict(num_classes=10, train_size=1840,  epoches=200, subdir="GC10-DET"),
    "coated_wood":   dict(num_classes=4,  train_size=10720, epoches=120, subdir="CoatedWood"),
    "vn_woodknot":   dict(num_classes=1,  train_size=4000,  epoches=120, subdir="VNWoodKnot"),
    "kodytek_wood":  dict(num_classes=7,  train_size=16220, epoches=80,  subdir="KodytekWood"),
}


def derive_schedule(epoches: int, train_size: int, batch: int):
    """从 epoches 与数据规模推导全部耦合字段。"""
    no_aug = NO_AUG_EPOCH if epoches > 40 else max(2, epoches // 10)
    aug_stop = epoches - no_aug
    stage_middle = STAGE_START + (epoches - no_aug) // 2

    iters_per_epoch = math.ceil(train_size / batch)
    total_iters = iters_per_epoch * epoches
    # warmup 取总 iter 的 4%，夹在 [100, 2000]
    warmup_iter = int(min(2000, max(100, round(total_iters * 0.04 / 50) * 50)))
    ema_warmups = max(50, warmup_iter // 2)

    return dict(
        no_aug_epoch=no_aug,
        aug_stop_epoch=aug_stop,
        stage_middle_epoch=stage_middle,
        iters_per_epoch=iters_per_epoch,
        total_iters=total_iters,
        warmup_iter=warmup_iter,
        ema_warmups=ema_warmups,
    )


def train_ops_block(imgsz: int, mosaic_out: int) -> str:
    return f"""      ops:
        - {{ type: Mosaic, output_size: {mosaic_out}, rotation_range: 10,
            translation_range: [0.1, 0.1], scaling_range: [0.5, 1.5],
            probability: 1.0, fill_value: 0, use_cache: False,
            max_cached_images: 50, random_pop: True }}
        - {{ type: RandomPhotometricDistort, p: 0.5 }}
        - {{ type: RandomZoomOut, fill: 0 }}
        - {{ type: RandomIoUCrop, p: 0.8 }}
        - {{ type: SanitizeBoundingBoxes, min_size: 1 }}
        - {{ type: RandomHorizontalFlip }}
        - {{ type: Resize, size: [{imgsz}, {imgsz}] }}
        - {{ type: SanitizeBoundingBoxes, min_size: 1 }}
        - {{ type: ConvertPILImage, dtype: "float32", scale: True }}
        - {{ type: ConvertBoxes, fmt: "cxcywh", normalize: True }}
"""


def build(model, name, num_classes, train_size, epoches, batch, imgsz,
          data_root, subdir, seed):
    sch = derive_schedule(epoches, train_size, batch)
    lr, bb_lr, bb_pat = REF_LR[model]
    scale = batch / REF_BATCH
    lr_s, bb_lr_s = lr * scale, bb_lr * scale

    root = os.path.join(data_root, subdir) if subdir else data_root
    mosaic_out = imgsz // 2

    L = []
    a = L.append
    a("# 本文件由 tools/make_dataset_config.py 自动生成，请勿手改。")
    a("# 需要改动请改生成参数后重新生成，否则四个增强锚点很容易改漏。")
    a(f"#")
    a(f"# 数据集 {name} | 模型 DEIM-{model.upper()} | 类别 {num_classes}")
    a(f"# 训练集 {train_size} 张 | batch {batch} | {sch['iters_per_epoch']} it/epoch")
    a(f"# epoches {epoches} → 总 {sch['total_iters']} iter | warmup {sch['warmup_iter']} "
      f"({sch['warmup_iter']/sch['total_iters']*100:.1f}%)")
    a("#")
    a("# 训练：")
    a(f"#   torchrun --master_port=7777 --nproc_per_node=2 train.py \\")
    a(f"#     -c configs/deim_dfine/custom/{name}_{model}.yml \\")
    a(f"#     -t /path/to/deim_dfine_hgnetv2_{model}_coco.pth --use-amp --seed {seed}")
    a("")
    a(f'__include__: ["../deim_hgnetv2_{model}_custom.yml"]')
    a("")
    a(f"output_dir: ./deim_outputs/{name}/deim_{model}_{imgsz}_e{epoches}_seed{seed}")
    a(f"num_classes: {num_classes}")
    a("remap_mscoco_category: False")
    a("")
    a("# ---- 训练轮数与调度：由 epoches 推导，四处已保持一致 ----")
    a(f"epoches: {epoches}")
    a(f"flat_epoch: {sch['stage_middle_epoch']}")
    a(f"no_aug_epoch: {sch['no_aug_epoch']}")
    a("lr_gamma: 0.5")
    a(f"warmup_iter: {sch['warmup_iter']}")
    a("")
    a("ema:")
    a(f"  warmups: {sch['ema_warmups']}")

    if imgsz != 640:
        a("")
        a("# ---- 输入尺寸 ----")
        a(f"eval_spatial_size: [{imgsz}, {imgsz}]")

    if abs(scale - 1.0) > 1e-6:
        a("")
        a(f"# ---- batch {batch} ≠ 官方 {REF_BATCH}，lr 按线性缩放 ×{scale:g} ----")
        a("optimizer:")
        a("  type: AdamW")
        a("  params:")
        if model == "s":
            a(f'    - params: "^(?=.*backbone)(?!.*{bb_pat}).*$"')
            a(f"      lr: {bb_lr_s:.8g}")
            a(f'    - params: "^(?=.*backbone)(?=.*{bb_pat}).*$"')
            a(f"      lr: {bb_lr_s:.8g}")
            a("      weight_decay: 0.")
            a('    - params: "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$"')
            a("      weight_decay: 0.")
        else:
            a(f'    - params: "^(?=.*backbone)(?!.*{bb_pat}).*$"')
            a(f"      lr: {bb_lr_s:.8g}")
            a('    - params: "^(?=.*(?:norm|bn)).*$"')
            a("      weight_decay: 0.")
        a(f"  lr: {lr_s:.8g}")
        a("  betas: [0.9, 0.999]")
        a("  weight_decay: 0.0001")

    a("")
    a("# ---- 数据与增强调度 ----")
    a("train_dataloader:")
    a(f"  total_batch_size: {batch}")
    a("  dataset:")
    a(f"    img_folder: {root}/images/train")
    a(f"    ann_file: {root}/annotations/instances_train.json")
    a("    transforms:")
    if imgsz != 640:
        a(train_ops_block(imgsz, mosaic_out).rstrip())
    a("      policy:")
    a("        name: stop_epoch")
    a(f"        epoch: [{STAGE_START}, {sch['stage_middle_epoch']}, {sch['aug_stop_epoch']}]")
    a('        ops: ["Mosaic", "RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"]')
    a("      mosaic_prob: 0.5")
    a("  collate_fn:")
    a("    type: BatchImageCollateFunction")
    a(f"    base_size: {imgsz}")
    a("    mixup_prob: 0.5")
    a(f"    mixup_epochs: [{STAGE_START}, {sch['stage_middle_epoch']}]")
    a(f"    stop_epoch: {sch['aug_stop_epoch']}")
    a("    ema_restart_decay: 0.9999")
    a("")
    a("val_dataloader:")
    a(f"  total_batch_size: {batch}")
    a("  dataset:")
    a(f"    img_folder: {root}/images/val")
    a(f"    ann_file: {root}/annotations/instances_val.json")
    if imgsz != 640:
        a("    transforms:")
        a("      ops:")
        a(f"        - {{ type: Resize, size: [{imgsz}, {imgsz}] }}")
        a('        - { type: ConvertPILImage, dtype: "float32", scale: True }')
    a("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["s", "m"], default="s")
    ap.add_argument("--name", help="数据集名，也用作输出文件名")
    ap.add_argument("--preset", help="使用内置预设；'all' 生成全部五个数据集 × s/m")
    ap.add_argument("--num-classes", type=int)
    ap.add_argument("--train-size", type=int, help="划分后训练集张数（不是总张数）")
    ap.add_argument("--epoches", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data-root", default="/data/defect")
    ap.add_argument("--subdir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="configs/deim_dfine/custom")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = []

    if args.preset == "all":
        for name, cfg in PRESETS.items():
            for model in ("s", "m"):
                jobs.append((model, name, cfg["num_classes"], cfg["train_size"],
                             cfg["epoches"], cfg["subdir"]))
    elif args.preset:
        cfg = PRESETS[args.preset]
        jobs.append((args.model, args.preset, cfg["num_classes"], cfg["train_size"],
                     cfg["epoches"], cfg["subdir"]))
    else:
        assert args.name and args.num_classes and args.train_size and args.epoches, \
            "不用 --preset 时必须给 --name --num-classes --train-size --epoches"
        jobs.append((args.model, args.name, args.num_classes, args.train_size,
                     args.epoches, args.subdir))

    for model, name, nc, ts, ep, subdir in jobs:
        text = build(model, name, nc, ts, ep, args.batch, args.imgsz,
                     args.data_root, subdir, args.seed)
        path = os.path.join(args.out_dir, f"{name}_{model}.yml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        sch = derive_schedule(ep, ts, args.batch)
        print(f"[ok] {path}  epoches={ep} flat={sch['stage_middle_epoch']} "
              f"aug_stop={sch['aug_stop_epoch']} warmup={sch['warmup_iter']}")


if __name__ == "__main__":
    main()
