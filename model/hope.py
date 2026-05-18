import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict
from yacs.config import CfgNode

from manopth.manolayer import ManoLayer
from losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss
from component.Backbone import DINOv2Backbone
from component.Decoder import HandPoseDecoder, ObjPoseDecoder
from component.Aggregator import HandAggregator, ObjAggregator
from component.TemporalFilter import TemporalSplitter

# MANO parameter layout: [pose(48) | shape(10) | trans(3)]
HAND_DIM = 61
OBJ_DIM  = 6   # position(3) + rotation(3)


class HOPE(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # ── Backbone: DINOv2, frozen ──────────────────────────────────────────
        self.backbone = DINOv2Backbone(
            model_name=cfg.MODEL.BACKBONE,
            freeze=True,
        )
        feat_dim = self.backbone.feat_dim   # 1024 for ViT-L/14
        agg_dim  = cfg.MODEL.AGG_DIM        # 256

        # ── MANO model (axis-angle, no PCA) ───────────────────────────────────
        self.mano = ManoLayer(
            mano_root=cfg.MANO.ROOT,
            use_pca=cfg.MANO.USE_PCA,
            flat_hand_mean=cfg.MANO.FLAT_HAND_MEAN,
            side=cfg.MANO.SIDE,
        )

        # ── Stage 1: coarse per-frame decoders ───────────────────────────────
        # Query: mean MANO joints / zero obj pose
        # Context: DINOv2 patch tokens
        self.hand_decoder = HandPoseDecoder(
            dim=feat_dim,
            num_layers=cfg.MODEL.DECODER_LAYERS,
            num_heads=cfg.MODEL.DECODER_HEADS,
            dropout=cfg.MODEL.DROPOUT,
        )
        self.obj_decoder = ObjPoseDecoder(
            dim=feat_dim,
            num_layers=cfg.MODEL.DECODER_LAYERS,
            num_heads=cfg.MODEL.DECODER_HEADS,
            dropout=cfg.MODEL.DROPOUT,
        )

        # ── Stage 2: cross-modal aggregation (intra-frame hand ↔ obj) ─────────
        # Jointly optimizes pose using the other modality's coarse estimate.
        # Outputs HandPoseFeature and ObjPoseFeature in agg_dim space.
        self.hand_aggregator = HandAggregator(
            hand_dim=HAND_DIM,
            obj_dim=OBJ_DIM,
            dim=agg_dim,
            num_layers=cfg.MODEL.AGG_LAYERS,
            dropout=cfg.MODEL.DROPOUT,
        )
        self.obj_aggregator = ObjAggregator(
            hand_dim=HAND_DIM,
            obj_dim=OBJ_DIM,
            dim=agg_dim,
            num_layers=cfg.MODEL.AGG_LAYERS,
            dropout=cfg.MODEL.DROPOUT,
        )

        # ── Stage 3: temporal refinement (cross-frame, per branch) ───────────
        # Each branch independently gathers context from ±window_size frames.
        # hand_ts uses kinematic embedding; obj_ts does not.
        self.hand_ts = TemporalSplitter(
            dim=agg_dim,
            num_slots=1,
            num_layers=cfg.MODEL.TS_LAYERS,
            max_frames=cfg.MODEL.T_CONTEXT,
            dropout=cfg.MODEL.DROPOUT,
            use_kinematic=True,
            window_size=cfg.MODEL.TS_WINDOW,
        )
        self.obj_ts = TemporalSplitter(
            dim=agg_dim,
            num_slots=1,
            num_layers=cfg.MODEL.TS_LAYERS,
            max_frames=cfg.MODEL.T_CONTEXT,
            dropout=cfg.MODEL.DROPOUT,
            use_kinematic=False,
            window_size=cfg.MODEL.TS_WINDOW,
        )

        # ── Stage 4: refined decoders ─────────────────────────────────────────
        # Same class as Stage 1 but with query_dim=agg_dim override so the
        # TS feature is used as query instead of joint coords / Fourier pose.
        # Context is still original DINOv2 patch tokens for visual grounding.
        # Output dicts are identical to Stage 1: {'pose','shape','trans'} / {'position','rotation'}.
        self.hand_refine_decoder = HandPoseDecoder(
            dim=feat_dim,
            num_layers=cfg.MODEL.DECODER_LAYERS,
            num_heads=cfg.MODEL.DECODER_HEADS,
            dropout=cfg.MODEL.DROPOUT,
            query_dim=agg_dim,
        )
        self.obj_refine_decoder = ObjPoseDecoder(
            dim=feat_dim,
            num_layers=cfg.MODEL.DECODER_LAYERS,
            num_heads=cfg.MODEL.DECODER_HEADS,
            dropout=cfg.MODEL.DROPOUT,
            query_dim=agg_dim,
        )

        # ── Mean-pose query buffers (not trained) ─────────────────────────────
        self.register_buffer('init_pose',     torch.zeros(1, 48))
        self.register_buffer('init_betas',    torch.zeros(1, 10))
        self.register_buffer('init_trans',    torch.zeros(1, 3))
        self.register_buffer('init_obj_pose', torch.zeros(1, OBJ_DIM))

        # ── Losses ────────────────────────────────────────────────────────────
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_mean_joints(self, BT: int) -> torch.Tensor:
        """Run MANO with zero pose → mean-hand joint positions (BT, 21, 3)."""
        with torch.no_grad():
            _, th_jtr = self.mano(
                self.init_pose.expand(BT, -1),
                self.init_betas.expand(BT, -1),
                self.init_trans.expand(BT, -1),
            )
        return th_jtr

    def _mano_forward(self, hand_params: torch.Tensor) -> torch.Tensor:
        """
        hand_params: (..., 61)  [pose(48) | shape(10) | trans(3)]
        Returns th_jtr with same leading dims + (21, 3).
        """
        leading = hand_params.shape[:-1]
        flat = hand_params.view(-1, HAND_DIM)
        _, th_jtr = self.mano(flat[:, :48], flat[:, 48:58], flat[:, 58:])
        return th_jtr.view(*leading, 21, 3)

    # ── Core forward ──────────────────────────────────────────────────────────

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Four-stage pipeline:

          Stage 1 — Coarse decoding (per frame, parallel)
            DINOv2 features  →  HandPoseDecoder / ObjPoseDecoder
            → hand_params_coarse (B, T, 61),  obj_params_coarse (B, T, 6)

          Stage 2 — Cross-modal aggregation (intra-frame hand ↔ obj)
            [hand_params, obj_params]  →  HandAggregator / ObjAggregator
            → HandPoseFeature (B, T, agg_dim),  ObjPoseFeature (B, T, agg_dim)

          Stage 3 — Temporal refinement (cross-frame, per branch, ±window frames)
            HandPoseFeature  →  hand_ts  →  hand_feat_ts (B, T, agg_dim)
            ObjPoseFeature   →  obj_ts   →  obj_feat_ts  (B, T, agg_dim)

          Stage 4 — Refined decoding (per frame, parallel)
            hand_feat_ts + DINOv2  →  hand_refine_decoder  →  hand_params_ref
            obj_feat_ts  + DINOv2  →  obj_refine_decoder   →  obj_params_ref

        Args:
            batch['image']: (B, T, C, H, W)  T == cfg.MODEL.T_CONTEXT
        """
        x = batch['image']          # (B, T, C, H, W)
        B, T = x.shape[:2]

        # ── Stage 1: per-frame DINOv2 features + coarse decoding ─────────────
        dino_feats = self.backbone(x)                       # (B, T, N, D)
        N, D = dino_feats.shape[2], dino_feats.shape[3]
        dino_flat = dino_feats.view(B * T, N, D)            # (B*T, N, D)

        th_jtr_init   = self._get_mean_joints(B * T)              # (B*T, 21, 3)
        obj_pose_init = self.init_obj_pose.expand(B * T, -1)      # (B*T, 6)

        hand_out = self.hand_decoder(th_jtr_init, dino_flat)
        # {'pose': (B*T,48), 'shape': (B*T,10), 'trans': (B*T,3)}
        obj_out  = self.obj_decoder(obj_pose_init, dino_flat)
        # {'position': (B*T,3), 'rotation': (B*T,3)}

        hand_params_coarse = torch.cat(
            [hand_out['pose'], hand_out['shape'], hand_out['trans']], dim=-1
        ).view(B, T, HAND_DIM)                                     # (B, T, 61)

        obj_params_coarse = torch.cat(
            [obj_out['position'], obj_out['rotation']], dim=-1
        ).view(B, T, OBJ_DIM)                                      # (B, T, 6)

        joints_3d_coarse = self._mano_forward(hand_params_coarse)  # (B, T, 21, 3)

        # ── Stage 2: cross-modal aggregation (intra-frame) ───────────────────
        hand_feat = self.hand_aggregator(hand_params_coarse, obj_params_coarse)
        # HandPoseFeature: (B, T, agg_dim)
        obj_feat  = self.obj_aggregator(hand_params_coarse, obj_params_coarse)
        # ObjPoseFeature:  (B, T, agg_dim)

        # ── Stage 3: temporal refinement (cross-frame, per branch) ───────────
        hand_feat_ts, = self.hand_ts([hand_feat])           # (B, T, agg_dim)
        obj_feat_ts,  = self.obj_ts([obj_feat])             # (B, T, agg_dim)

        # ── Stage 4: refined decoding (TS feature → re-attend DINOv2) ────────
        # The TS feature carries cross-modal + temporal context as query.
        # DINOv2 patch tokens provide fine-grained visual grounding as context.
        # Output dicts mirror Stage 1 for symmetric loss computation.
        hand_out_ref = self.hand_refine_decoder(
            hand_feat_ts.view(B * T, -1),   # query:   (B*T, agg_dim)
            dino_flat,                       # context: (B*T, N, D)
        )
        # {'pose': (B*T,48), 'shape': (B*T,10), 'trans': (B*T,3)}

        obj_out_ref = self.obj_refine_decoder(
            obj_feat_ts.view(B * T, -1),    # query:   (B*T, agg_dim)
            dino_flat,                       # context: (B*T, N, D)
        )
        # {'position': (B*T,3), 'rotation': (B*T,3)}

        hand_params_ref = torch.cat(
            [hand_out_ref['pose'], hand_out_ref['shape'], hand_out_ref['trans']], dim=-1
        ).view(B, T, HAND_DIM)              # (B, T, 61)

        obj_params_ref = torch.cat(
            [obj_out_ref['position'], obj_out_ref['rotation']], dim=-1
        ).view(B, T, OBJ_DIM)              # (B, T, 6)

        joints_3d_ref = self._mano_forward(hand_params_ref)        # (B, T, 21, 3)

        return {
            # coarse (for auxiliary supervision)
            'hand_params_coarse': hand_params_coarse,   # (B, T, 61)
            'obj_params_coarse':  obj_params_coarse,    # (B, T, 6)
            'joints_3d_coarse':   joints_3d_coarse,     # (B, T, 21, 3)
            # refined (primary output)
            'hand_params': hand_params_ref,             # (B, T, 61)
            'obj_params':  obj_params_ref,              # (B, T, 6)
            'joints_3d':   joints_3d_ref,               # (B, T, 21, 3)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quick test: (B, T, C, H, W) → (B, T, N, D)."""
        return self.backbone(x)

    # ── Optimization ──────────────────────────────────────────────────────────

    def get_parameters(self):
        return (
            list(self.hand_decoder.parameters())
            + list(self.obj_decoder.parameters())
            + list(self.hand_aggregator.parameters())
            + list(self.obj_aggregator.parameters())
            + list(self.hand_ts.parameters())
            + list(self.obj_ts.parameters())
            + list(self.hand_refine_decoder.parameters())
            + list(self.obj_refine_decoder.parameters())
        )

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.get_parameters())
        return torch.optim.AdamW(
            params,
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _rotmat_to_axisangle(R: torch.Tensor) -> torch.Tensor:
        """R: (..., 3, 3) → axis-angle (..., 3)."""
        shape  = R.shape[:-2]
        R_flat = R.reshape(-1, 3, 3)
        trace  = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
        theta  = torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6))
        sin_t  = theta.sin().clamp(min=1e-6)
        rx = R_flat[:, 2, 1] - R_flat[:, 1, 2]
        ry = R_flat[:, 0, 2] - R_flat[:, 2, 0]
        rz = R_flat[:, 1, 0] - R_flat[:, 0, 1]
        axis = torch.stack([rx, ry, rz], dim=-1) / (2 * sin_t.unsqueeze(-1))
        return (axis * theta.unsqueeze(-1)).reshape(*shape, 3)

    def _project_joints(self, joints_3d: torch.Tensor,
                        K: torch.Tensor) -> torch.Tensor:
        """Perspective projection.  joints_3d: (B,T,21,3), K: (B,T,3,3) → (B,T,21,2)."""
        B, T, N, _ = joints_3d.shape
        j    = joints_3d.reshape(B * T, N, 3)
        k    = K.reshape(B * T, 3, 3)
        proj = (k @ j.permute(0, 2, 1)).permute(0, 2, 1)   # (B*T, 21, 3)
        j2d  = proj[..., :2] / proj[..., 2:3].clamp(min=1e-6)
        return j2d.reshape(B, T, N, 2)

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(self, out: Dict, batch: Dict) -> Dict:
        B, T = batch['image'].shape[:2]
        has_hand = batch['has_hand'].reshape(B * T)            # (BT,)

        j3d_gt   = batch['joint_3d'].reshape(B * T, 21, 3)
        j2d_gt   = batch['joint_2d'].reshape(B * T, 21, 2)
        pose_gt  = batch['mano_pose'].reshape(B * T, 48)
        trans_gt = batch['mano_trans'].reshape(B * T, 3)
        opos_gt  = batch['obj_trans'].reshape(B * T, 3)
        orot_gt  = self._rotmat_to_axisangle(
                       batch['obj_rot'].reshape(B * T, 3, 3))  # (BT, 3)
        K        = batch['K']                                   # (B, T, 3, 3)

        # per-joint confidence = has_hand scalar broadcast to all 21 joints
        conf     = has_hand[:, None, None].expand(B * T, 21, 1)
        j3d_conf = torch.cat([j3d_gt, conf], dim=-1)           # (BT, 21, 4)
        j2d_conf = torch.cat([j2d_gt, conf], dim=-1)           # (BT, 21, 3)

        def _losses(joints_3d, hand_params, obj_params):
            l_j3d = self.keypoint_3d_loss(
                joints_3d.reshape(B * T, 21, 3), j3d_conf)

            j2d_pred = self._project_joints(joints_3d, K)
            l_j2d = self.keypoint_2d_loss(
                j2d_pred.reshape(B * T, 21, 2), j2d_conf)

            hp     = hand_params.reshape(B * T, HAND_DIM)
            l_pose  = self.mano_parameter_loss(hp[:, :48],  pose_gt,  has_hand)
            l_trans = self.mano_parameter_loss(hp[:, 58:],  trans_gt, has_hand)

            op     = obj_params.reshape(B * T, OBJ_DIM)
            l_opos = F.l1_loss(op[:, :3], opos_gt)
            l_orot = F.l1_loss(op[:, 3:], orot_gt)
            return l_j3d, l_j2d, l_pose, l_trans, l_opos, l_orot

        L  = _losses(out['joints_3d'],        out['hand_params'],        out['obj_params'])
        Lc = _losses(out['joints_3d_coarse'], out['hand_params_coarse'], out['obj_params_coarse'])

        w = self.cfg.TRAIN
        c = w.LOSS_COARSE
        total = (
            w.LOSS_JOINTS3D * (L[0] + c * Lc[0])
            + w.LOSS_JOINTS2D * (L[1] + c * Lc[1])
            + w.LOSS_POSE     * (L[2] + c * Lc[2])
            + w.LOSS_TRANS    * (L[3] + c * Lc[3])
            + w.LOSS_OBJ_POS  * (L[4] + c * Lc[4])
            + w.LOSS_OBJ_ROT  * (L[5] + c * Lc[5])
        )
        return {
            'loss':         total,
            'loss_j3d':     L[0],
            'loss_j2d':     L[1],
            'loss_pose':    L[2],
            'loss_trans':   L[3],
            'loss_obj_pos': L[4],
            'loss_obj_rot': L[5],
        }

    # ── Training / Validation steps ───────────────────────────────────────────

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        out    = self.forward_step(batch, train=True)
        losses = self._compute_loss(out, batch)
        for k, v in losses.items():
            self.log(f'train/{k}', v, on_step=True, on_epoch=True,
                     prog_bar=(k == 'loss'), sync_dist=True)
        return losses['loss']

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        out    = self.forward_step(batch, train=False)
        losses = self._compute_loss(out, batch)
        for k, v in losses.items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True,
                     prog_bar=(k == 'loss'), sync_dist=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.backbone.backbone.eval()
