# -*- coding: utf-8 -*-
"""
Dry-run script: verifies dataset loading and model forward pass
without any training.  Run from project root:
    python test.py
"""
import sys
import os
sys.path.insert(0, 'model')

import numpy as np
# chumpy compatibility patch for newer numpy
np.bool    = np.bool_
np.int     = np.int_
np.float   = np.float64
np.complex = np.complex128
np.object  = object
np.str     = np.str_
np.unicode = np.str_

import torch
from yacs.config import CfgNode as CN


# ── Config ────────────────────────────────────────────────────────────────────

def build_cfg() -> CN:
    cfg = CN()

    cfg.MANO = CN()
    cfg.MANO.ROOT           = 'MANO'
    cfg.MANO.USE_PCA        = False
    cfg.MANO.FLAT_HAND_MEAN = True
    cfg.MANO.SIDE           = 'right'

    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE       = 'dinov2_vitl14_reg'
    cfg.MODEL.DECODER_LAYERS = 6
    cfg.MODEL.DECODER_HEADS  = 8
    cfg.MODEL.DROPOUT        = 0.0
    cfg.MODEL.AGG_DIM        = 256
    cfg.MODEL.AGG_LAYERS     = 2
    cfg.MODEL.TS_LAYERS      = 2
    cfg.MODEL.TS_WINDOW      = 4
    cfg.MODEL.T_CONTEXT      = 8

    cfg.TRAIN = CN()
    cfg.TRAIN.LR           = 1e-4
    cfg.TRAIN.WEIGHT_DECAY = 1e-4
    cfg.TRAIN.LOSS_JOINTS3D = 300.0
    cfg.TRAIN.LOSS_JOINTS2D = 300.0
    cfg.TRAIN.LOSS_POSE     = 60.0
    cfg.TRAIN.LOSS_TRANS    = 60.0
    cfg.TRAIN.LOSS_OBJ_POS  = 10.0
    cfg.TRAIN.LOSS_OBJ_ROT  = 10.0
    cfg.TRAIN.LOSS_COARSE   = 0.5

    return cfg


# ── Test 1: Dataset ───────────────────────────────────────────────────────────

def test_dataset():
    print("── Test 1: DexYCB dataset ────────────────────────────────────")
    os.environ['DEX_YCB_DIR'] = r'D:\BUAA\dex-ycb'
    from data.DexYCBDataset import DexYCBDataset

    ds = DexYCBDataset(split='train')
    img, depth, target = ds[0]

    print(f"  total samples : {len(ds)}")
    print(f"  img           : {tuple(img.shape)}   dtype={img.dtype}")
    print(f"  depth         : {tuple(depth.shape)} dtype={depth.dtype}")
    print(f"  joint_3d      : {tuple(target['joint_3d'].shape)}")
    print(f"  joint_2d      : {tuple(target['joint_2d'].shape)}")
    print(f"  mano_pca      : {tuple(target['mano_pca'].shape)}")
    print(f"  has_hand      : {target['has_hand'].item()}")
    print(f"  K             :\n{target['K'].numpy()}")
    print("  PASS\n")
    return img.shape[-2], img.shape[-1]   # H, W


# ── Test 2: Model forward pass ────────────────────────────────────────────────

def test_model(cfg: CN, H: int, W: int, device: torch.device):
    print("── Test 2: Model forward pass ────────────────────────────────")
    from hope import HOPE

    print("  Building model... (DINOv2 may download on first run)")
    model = HOPE(cfg).to(device)
    model.eval()

    n_trainable = sum(p.numel() for p in model.get_parameters())
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Total params     : {n_total:>12,}")
    print(f"  Trainable params : {n_trainable:>12,}  (backbone frozen)")

    B = 2
    T = cfg.MODEL.T_CONTEXT      # 8
    batch = {'image': torch.randn(B, T, 3, H, W, device=device)}
    print(f"\n  Input  : (B={B}, T={T}, C=3, H={H}, W={W})")

    with torch.no_grad():
        out = model.forward_step(batch)

    print("\n  Outputs:")
    for k, v in out.items():
        print(f"    {k:<22} {tuple(v.shape)}")

    # Shape assertions
    assert out['hand_params_coarse'].shape == (B, T, 61),    "hand_params_coarse"
    assert out['obj_params_coarse'].shape  == (B, T, 6),     "obj_params_coarse"
    assert out['joints_3d_coarse'].shape   == (B, T, 21, 3), "joints_3d_coarse"
    assert out['hand_params'].shape        == (B, T, 61),    "hand_params"
    assert out['obj_params'].shape         == (B, T, 6),     "obj_params"
    assert out['joints_3d'].shape          == (B, T, 21, 3), "joints_3d"

    if device.type == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"\n  Peak GPU memory  : {mem_mb:.1f} MB")

    print("  All shape assertions passed")
    print("  PASS\n")


# ── Test 3: training_step (loss computation) ─────────────────────────────────

def test_training_step(cfg: CN, H: int, W: int, device: torch.device):
    print("── Test 3: training_step (loss computation) ──────────────────")
    from hope import HOPE

    model = HOPE(cfg).to(device)
    model.train()

    B = 2
    T = cfg.MODEL.T_CONTEXT   # 8

    batch = {
        'image'     : torch.randn(B, T, 3, H, W,    device=device),
        'joint_3d'  : torch.randn(B, T, 21, 3,      device=device),
        'joint_2d'  : torch.rand (B, T, 21, 2,      device=device) * 640,
        'has_hand'  : torch.ones (B, T,              device=device),
        'mano_pose' : torch.randn(B, T, 48,          device=device),
        'mano_trans': torch.randn(B, T, 3,           device=device),
        'obj_trans' : torch.randn(B, T, 3,           device=device),
        'obj_rot'   : torch.eye(3, device=device)
                          .unsqueeze(0).unsqueeze(0)
                          .expand(B, T, 3, 3).contiguous(),
        'K'         : torch.tensor(
                          [[525., 0., 320.],
                           [0., 525., 240.],
                           [0.,   0.,   1.]], device=device
                      ).unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3).contiguous(),
    }

    loss = model.training_step(batch, batch_idx=0)
    print(f"  loss = {loss.item():.4f}")
    assert loss.isfinite(), "Loss is not finite!"
    print("  PASS\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}\n")

    cfg = build_cfg()

    H, W = test_dataset()
    test_model(cfg, H, W, device)
    test_training_step(cfg, H, W, device)

    print("OK  Dry run completed successfully")
