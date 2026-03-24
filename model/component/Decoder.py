import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, query, context):
        """
        query:   (B, Nq, D)
        context: (B, Nk, D)  ← DINOv2 output
        """
        B, Nq, D = query.shape

        query = self.norm_q(query)
        context = self.norm_kv(context)

        Q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)
        return self.out_proj(out)
    

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, N, D)"""
        B, N, D = x.shape
        x_norm = self.norm(x)

        QKV = self.qkv_proj(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        Q, K, V = QKV.unbind(dim=2)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
    

class FFN(nn.Module):
    """Feed-Forward Network (MLP block inside decoder layer)"""
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(self.norm(x))
    

class PoseDecoderLayer(nn.Module):
    """Single [CA → SA → MLP] block"""
    def __init__(self, dim, num_heads=8, ffn_dim=None, dropout=0.0):
        super().__init__()
        self.ca = CrossAttention(dim, num_heads, dropout)
        self.sa = SelfAttention(dim, num_heads, dropout)
        self.ffn = FFN(dim, ffn_dim, dropout)

    def forward(self, query, context):
        """
        query:   (B, Nq, D)
        context: (B, Nk, D)
        """
        query = query + self.ca(query, context)
        query = query + self.sa(query)
        query = query + self.ffn(query)
        return query
    

class PoseDecoder(nn.Module):
    """
    General Pose Decoder (shared structure for Hand and Obj).
    
    Args:
        num_queries:  number of learnable queries
        dim:          feature dimension
        num_layers:   number of [CA→SA→MLP] blocks
        num_heads:    attention heads
        output_dim:   final output dimension (e.g. 61 for MANO, 6 for obj)
        ffn_dim:      hidden dim of FFN (default: 4*dim)
        dropout:      dropout rate
    """
    def __init__(
        self,
        num_queries: int,
        dim: int,
        num_layers: int,
        output_dim: int,
        num_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Learnable MANO/Obj queries
        self.queries = nn.Embedding(num_queries, dim)

        # Stacked decoder layers
        self.layers = nn.ModuleList([
            PoseDecoderLayer(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final MLP head: F' → F
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, output_dim),
        )

    def forward(self, dino_features):
        """
        Args:
            dino_features: (B, Nk, D)  ← DINOv2 patch tokens (frozen)
        Returns:
            F: (B, num_queries, output_dim)
        """
        B = dino_features.shape[0]

        query = self.queries.weight.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            query = layer(query, dino_features)

        F = self.head(query)
        return F
    

class HandPoseDecoder(PoseDecoder):
    """
    Output per query:
        pose:  48  (15 joints + 1 global) × 3 axis-angle
        shape: 10  MANO shape betas
        trans:  3  root translation
        total: 61
    """
    POSE_DIM = 48
    SHAPE_DIM = 10
    TRANS_DIM = 3
    OUTPUT_DIM = POSE_DIM + SHAPE_DIM + TRANS_DIM  # 61

    def __init__(self, dim, num_layers=6, num_heads=8, ffn_dim=None, dropout=0.0):
        super().__init__(
            num_queries=1,       # one hand per frame
            dim=dim,
            num_layers=num_layers,
            output_dim=self.OUTPUT_DIM,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, dino_features):
        """
        Returns dict with parsed MANO params.
        dino_features: (B, Nk, D)
        """
        F = super().forward(dino_features)  # (B, 1, 61)
        F = F.squeeze(1)                    # (B, 61)

        return {
            "pose":  F[:, :self.POSE_DIM],                               # (B, 48)
            "shape": F[:, self.POSE_DIM:self.POSE_DIM+self.SHAPE_DIM],   # (B, 10)
            "trans": F[:, self.POSE_DIM+self.SHAPE_DIM:],                # (B, 3)
        }


class ObjPoseDecoder(PoseDecoder):
    """
    Output per query:
        position: 3   (x, y, z)
        rotation: 3   axis-angle
        total:    6
    """
    OUTPUT_DIM = 6

    def __init__(self, dim, num_layers=6, num_heads=8, ffn_dim=None, dropout=0.0):
        super().__init__(
            num_queries=1,
            dim=dim,
            num_layers=num_layers,
            output_dim=self.OUTPUT_DIM,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, dino_features):
        """
        Returns dict with parsed obj params.
        dino_features: (B, Nk, D)
        """
        F = super().forward(dino_features)  # (B, 1, 6)
        F = F.squeeze(1)                    # (B, 6)

        return {
            "position": F[:, :3],  # (B, 3)
            "rotation": F[:, 3:],  # (B, 3) axis-angle
        }