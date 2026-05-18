# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from data.DexYCBDataset import DexYCBDataset


class DexYCBVideoDataset(Dataset):
    """
    Wraps DexYCBDataset to return T-frame clips from the same camera sequence.
    Each __getitem__ returns a dict of stacked tensors (T, ...) ready for HOPE.

    batch layout expected by training_step:
        image      : (T, 3, H, W)
        joint_3d   : (T, 21, 3)
        joint_2d   : (T, 21, 2)
        has_hand   : (T,)
        mano_pose  : (T, 48)
        mano_trans : (T, 3)
        obj_trans  : (T, 3)    first object translation (zeros if none)
        obj_rot    : (T, 3, 3) first object rotation matrix (identity if none)
        K          : (T, 3, 3)
    """

    def __init__(self, split: str = 'train', T: int = 8, stride: int = 4,
                 transform=None):
        self._base = DexYCBDataset(split=split, transform=transform)
        self.T      = T
        self.stride = stride
        self.clips  = self._build_clips()
        print(f"[{split}] VideoDataset: {len(self.clips)} clips "
              f"(T={T}, stride={stride})")

    def _build_clips(self):
        seq_map = defaultdict(list)
        for i, s in enumerate(self._base.samples):
            seq_map[s['seq_key']].append((s['frame_idx'], i))

        clips = []
        for frames in seq_map.values():
            frames.sort(key=lambda x: x[0])
            indices = [idx for _, idx in frames]
            for start in range(0, len(indices) - self.T + 1, self.stride):
                clips.append(indices[start: start + self.T])
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_indices = self.clips[idx]
        frames = [self._get_frame(i) for i in frame_indices]
        return {k: torch.stack([f[k] for f in frames], dim=0)
                for k in frames[0]}

    def _get_frame(self, sample_idx: int) -> dict:
        """Return a single-frame dict with all tensors needed for loss computation."""
        import cv2
        info = self._base.samples[sample_idx]

        img   = cv2.imread(info['img_path'])
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(info['depth_path'], cv2.IMREAD_ANYDEPTH)
        if depth is None:
            depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        else:
            depth = depth.astype(np.float32) / 1000.0

        data     = np.load(info['label_path'])
        pose_m   = data['pose_m'].flatten()          # (51,)
        has_hand = not np.allclose(pose_m, 0)
        j3d      = data['joint_3d'][0].astype(np.float32)   # (21, 3)
        j2d      = data['joint_2d'][0].astype(np.float32)   # (21, 2)

        # Object pose: use first object; fall back to identity/zero
        pose_y   = data['pose_y']                     # (N_obj, 4, 4)
        if pose_y.shape[0] > 0:
            obj_rot   = pose_y[0, :3, :3].astype(np.float32)
            obj_trans = pose_y[0, :3,  3].astype(np.float32)
        else:
            obj_rot   = np.eye(3, dtype=np.float32)
            obj_trans = np.zeros(3, dtype=np.float32)

        K = self._base.intrinsics_cache.get(
            info['cam_serial'], np.eye(3, dtype=np.float32)
        )

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        return {
            'image'     : img_tensor,                                         # (3,H,W)
            'joint_3d'  : torch.from_numpy(j3d),                              # (21,3)
            'joint_2d'  : torch.from_numpy(j2d),                              # (21,2)
            'has_hand'  : torch.tensor(1.0 if has_hand else 0.0),             # scalar
            'mano_pose' : torch.from_numpy(pose_m[:48].astype(np.float32)),   # (48,)
            'mano_trans': torch.from_numpy(pose_m[48:51].astype(np.float32)), # (3,)
            'obj_rot'   : torch.from_numpy(obj_rot),                          # (3,3)
            'obj_trans' : torch.from_numpy(obj_trans),                        # (3,)
            'K'         : torch.from_numpy(K).float(),                        # (3,3)
        }
