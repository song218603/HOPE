# -*- coding: utf-8 -*-
"""
Training entry point for HOPE.

Usage:
    python train.py --data-dir D:/BUAA/dex-ycb
    python train.py --data-dir D:/BUAA/dex-ycb --resume outputs/checkpoints/last.ckpt
    python train.py --data-dir D:/BUAA/dex-ycb --devices 2 --strategy ddp
"""
import sys
import os
import argparse

sys.path.insert(0, 'model')

# chumpy compatibility patch for newer numpy
import numpy as np
np.bool    = np.bool_
np.int     = np.int_
np.float   = np.float64
np.complex = np.complex128
np.object  = object
np.str     = np.str_
np.unicode = np.str_

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train HOPE')
    p.add_argument('--cfg',        default='config/train.yaml',
                   help='path to YACS config YAML')
    p.add_argument('--data-dir',   default=None,
                   help='path to DexYCB root (overrides DEX_YCB_DIR env var)')
    p.add_argument('--resume',     default=None,
                   help='resume training from a .ckpt checkpoint file')
    p.add_argument('--devices',    default='auto',
                   help='number of GPUs (int) or "auto"')
    p.add_argument('--strategy',   default='auto',
                   help='PL strategy: "auto", "ddp", "ddp_find_unused_parameters_false", ...')
    p.add_argument('--precision',  default='16-mixed',
                   help='"16-mixed" (default), "32", "bf16-mixed"')
    p.add_argument('--accum-grad', type=int, default=1,
                   help='gradient accumulation steps (effective batch = BATCH_SIZE * accum)')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--workers',    type=int, default=4,
                   help='DataLoader num_workers per GPU')
    return p.parse_args()


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> CN:
    with open(path) as f:
        cfg = CN.load_cfg(f)
    cfg.freeze()
    return cfg


# ── Data ──────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: CN, num_workers: int):
    from data.DexYCBVideoDataset import DexYCBVideoDataset

    T = cfg.MODEL.T_CONTEXT

    train_ds = DexYCBVideoDataset(split='train', T=T, stride=T // 2)
    val_ds   = DexYCBVideoDataset(split='val',   T=T, stride=T)

    common = dict(
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True,  **common)
    val_dl   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)
    return train_dl, val_dl


# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(cfg: CN):
    ckpt = ModelCheckpoint(
        dirpath='outputs/checkpoints',
        filename='hope-{epoch:03d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    lr_mon = LearningRateMonitor(logging_interval='step')
    early  = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min',
        verbose=True,
    )
    return [ckpt, lr_mon, early]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Data directory
    if args.data_dir:
        os.environ['DEX_YCB_DIR'] = args.data_dir
    if 'DEX_YCB_DIR' not in os.environ:
        raise RuntimeError(
            "DexYCB data directory not specified. "
            "Use --data-dir or set the DEX_YCB_DIR environment variable."
        )

    pl.seed_everything(args.seed, workers=True)

    cfg = load_cfg(args.cfg)
    print(f"Config:\n{cfg}\n")

    # Model
    from hope import HOPE
    model = HOPE(cfg)

    # Data
    train_dl, val_dl = build_dataloaders(cfg, num_workers=args.workers)

    # Outputs
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs',        exist_ok=True)

    # Convert devices arg (argparse gives a string)
    devices = int(args.devices) if str(args.devices).isdigit() else args.devices

    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        accelerator='auto',
        devices=devices,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=build_callbacks(cfg),
        logger=TensorBoardLogger('outputs/logs', name='hope'),
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accum_grad,
        deterministic=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
