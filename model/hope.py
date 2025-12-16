import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple
from yacs.config import CfgNode

from mano.mano import MANO
from losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss
from component.Aggreator import Aggreator
from component.HandAwareDecoder import HandAwareDecoder
from component.ObjectAwareDecoder import ObjectAwareDecoder
from component.TemporalFilter import TemporalFilter


class HOPE(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        return all_params
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.backbone.eval()
    
    def forward_step(self, batch : Dict, train : bool = False) -> Dict:
        x = batch["image"]

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out
    


if __name__ == '__main__':
    net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    net.to('cuda')
    x = torch.randn(1, 3, 490, 644).cuda()

    print(net(x).shape)