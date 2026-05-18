import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2Backbone(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitl14_reg", freeze: bool = True):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.feat_dim   = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]  # 14
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)

        # DINOv2 requires H and W to be multiples of patch_size (14).
        # Resize down to the nearest valid size when needed.
        pH, pW = self.patch_size, self.patch_size
        tH = (H // pH) * pH
        tW = (W // pW) * pW
        if tH != H or tW != W:
            frames = F.interpolate(frames, size=(tH, tW),
                                   mode='bilinear', align_corners=False)

        feats = self.backbone.forward_features(frames)
        patch_tokens = feats["x_norm_patchtokens"]          # (B*T, N, D)
        N, D = patch_tokens.shape[1], patch_tokens.shape[2]
        return patch_tokens.view(B, T, N, D)


if __name__ == "__main__":
    model = DINOv2Backbone(freeze=True)
    x = torch.randn(2, 4, 3, 490, 644)   # 2 videos, 4 frames each
    out = model(x)
    print(f"[Backbone] Input: {x.shape} → Output: {out.shape}")