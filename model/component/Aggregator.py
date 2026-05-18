import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmbedding(nn.Module):
    """
    Standard sinusoidal positional encoding along the time axis.
    Supports variable-length sequences.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x : (B, T, D)
        returns x + pos_emb of shape (1, T, D)
        """
        B, T, D = x.shape
        device = x.device

        position = torch.arange(T, device=device).unsqueeze(1).float()   # (T, 1)
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D)
        )                                                                  # (D/2,)

        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D // 2])

        return x + pe.unsqueeze(0)
    

class AggregatorLayer(nn.Module):
    """
    One aggregator block:
        1. Conv1d(input) + Pos.Embedding
        2. Conv1d → Q, K, V
        3. Scaled dot-product attention: Softmax(Q·Kᵀ / sqrt(d)) · V  → F'
        4. FFN: Conv → ReLU → Conv  (with residual)

    All Conv1d use kernel_size=1 (i.e. pointwise), acting as linear projections
    along the feature dimension while treating T as the sequence axis.

    Args:
        in_dim   : input feature dimension (after concat & first Conv)
        dim      : internal attention dimension
        ffn_dim  : hidden dim of FFN  (default: 2 * dim)
        dropout  : dropout on attention weights
    """

    def __init__(self, in_dim: int, dim: int, ffn_dim: int = None, dropout: float = 0.0):
        super().__init__()
        ffn_dim = ffn_dim or dim * 2

        # ── input projection ─────────────────────────────────────────────────
        self.input_conv  = nn.Conv1d(in_dim, dim, kernel_size=1)
        self.input_norm  = nn.LayerNorm(dim)

        # ── positional embedding ──────────────────────────────────────────────
        self.pos_emb = SinusoidalPosEmbedding(dim)

        # ── Q / K / V projections (Conv1d kernel=1) ───────────────────────────
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1)

        self.scale   = dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # ── FFN: Conv → ReLU → Conv ───────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ffn_dim, dim, kernel_size=1),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x : (B, T, in_dim)   ← concat of hand & obj features
        returns : (B, T, dim)
        """
        B, T, _ = x.shape

        # ── Input projection + LayerNorm ──────────────────────────────────────
        # Conv1d expects (B, C, L)
        h = self.input_conv(x.transpose(1, 2))   # (B, dim, T)
        h = self.input_norm(h.transpose(1, 2))    # (B, T, dim)

        # ── Positional embedding ──────────────────────────────────────────────
        h = self.pos_emb(h)                        # (B, T, dim)

        # ── Q / K / V  via Conv1d ─────────────────────────────────────────────
        h_t = h.transpose(1, 2)                    # (B, dim, T)
        Q = self.q_conv(h_t)                       # (B, dim, T)
        K = self.k_conv(h_t)                       # (B, dim, T)
        V = self.v_conv(h_t)                       # (B, dim, T)

        # ── Scaled dot-product attention  (T×T) ──────────────────────────────
        # Q,K,V : (B, dim, T) → treat T as sequence, dim as features
        Q = Q.transpose(1, 2)   # (B, T, dim)
        K = K.transpose(1, 2)   # (B, T, dim)
        V = V.transpose(1, 2)   # (B, T, dim)

        attn = (Q @ K.transpose(-2, -1)) * self.scale   # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        F_prime = attn @ V                              # (B, T, dim)

        # ── FFN with residual ─────────────────────────────────────────────────
        F_prime_t = F_prime.transpose(1, 2)            # (B, dim, T)
        ffn_out   = self.ffn(F_prime_t)                # (B, dim, T)
        ffn_out   = ffn_out.transpose(1, 2)            # (B, T, dim)

        # residual + LayerNorm
        out = self.ffn_norm(F_prime + ffn_out)         # (B, T, dim)
        return out
    

class Aggregator(nn.Module):
    """
    Aggregates Hand and Object coarse pose features across T frames.

    Pipeline:
        [Hand Pose || Obj Pose]  →  AggregatorLayer  →  split → hand_feat / obj_feat

    Args:
        hand_dim  : dim of hand pose input  (e.g. 61 for MANO params)
        obj_dim   : dim of obj  pose input  (e.g.  6 for pos+rot)
        dim       : internal & output feature dim
        num_layers: number of stacked AggregatorLayers
        ffn_dim   : FFN hidden dim (default: 2*dim)
        dropout   : attention dropout
    """

    def __init__(
        self,
        hand_dim: int,
        obj_dim: int,
        dim: int,
        num_layers: int = 2,
        ffn_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        in_dim = hand_dim + obj_dim   # concat dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                in_dim  = in_dim if i == 0 else dim,
                dim     = dim,
                ffn_dim = ffn_dim,
                dropout = dropout,
            )
            for i in range(num_layers)
        ])

    def forward(self, hand_pose, obj_pose):
        """
        Args:
            hand_pose : (B, T, hand_dim)
            obj_pose  : (B, T, obj_dim)
        Returns:
            x : (B, T, dim)  融合特征
        """
        x = torch.cat([hand_pose, obj_pose], dim=-1)   # (B, T, hand_dim+obj_dim)
        for layer in self.layers:
            x = layer(x)                               # (B, T, dim)
        return x


class HandAggregator(Aggregator):
    """
    输出 hand 分支特征: (B, T, dim)
    """
    def __init__(self, hand_dim, obj_dim, dim, num_layers=2, ffn_dim=None, dropout=0.0):
        super().__init__(hand_dim, obj_dim, dim, num_layers, ffn_dim, dropout)
        self.out_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, hand_pose, obj_pose):
        x = super().forward(hand_pose, obj_pose)   # (B, T, dim)
        return self.out_proj(x)


class ObjAggregator(Aggregator):
    """
    输出 obj 分支特征: (B, T, dim)
    """
    def __init__(self, hand_dim, obj_dim, dim, num_layers=2, ffn_dim=None, dropout=0.0):
        super().__init__(hand_dim, obj_dim, dim, num_layers, ffn_dim, dropout)
        self.out_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, hand_pose, obj_pose):
        x = super().forward(hand_pose, obj_pose)   # (B, T, dim)
        return self.out_proj(x)


if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    BATCH_SIZE = 2
    AGG_DIM    = 256
    NUM_LAYERS = 2
    HAND_DIM   = 61   # 48 pose + 10 shape + 3 trans
    OBJ_DIM    = 6    # 3 position + 3 rotation
    T_LIST     = [8, 16, 32]

    hand_agg = HandAggregator(hand_dim=HAND_DIM, obj_dim=OBJ_DIM, dim=AGG_DIM, num_layers=NUM_LAYERS, dropout=0.1).to(device)
    obj_agg  = ObjAggregator( hand_dim=HAND_DIM, obj_dim=OBJ_DIM, dim=AGG_DIM, num_layers=NUM_LAYERS, dropout=0.1).to(device)

    print("=" * 55)
    print(f"  HandAggregator params : {sum(p.numel() for p in hand_agg.parameters()):>10,}")
    print(f"  ObjAggregator  params : {sum(p.numel() for p in obj_agg.parameters()):>10,}")
    print("=" * 55)

    # ── Shape test ────────────────────────────────────────────────────────────
    print("\n  Shape test (eval mode)")
    print("-" * 55)
    hand_agg.eval()
    obj_agg.eval()
    with torch.no_grad():
        for T in T_LIST:
            hand_pose = torch.randn(BATCH_SIZE, T, HAND_DIM, device=device)
            obj_pose  = torch.randn(BATCH_SIZE, T, OBJ_DIM,  device=device)

            hand_feat = hand_agg(hand_pose, obj_pose)
            obj_feat  = obj_agg(hand_pose, obj_pose)

            print(f"  T={T:>3d} | hand_feat={tuple(hand_feat.shape)}  obj_feat={tuple(obj_feat.shape)}")
            assert hand_feat.shape == (BATCH_SIZE, T, AGG_DIM)
            assert obj_feat.shape  == (BATCH_SIZE, T, AGG_DIM)

    print("\n  All shape assertions passed OK")

    # ── Gradient flow check ───────────────────────────────────────────────────
    print("\n" + "-" * 55)
    print("  Gradient Flow Check (training mode)")
    print("-" * 55)
    hand_agg.train()
    obj_agg.train()
    T = 16
    hand_pose = torch.randn(BATCH_SIZE, T, HAND_DIM, device=device, requires_grad=True)
    obj_pose  = torch.randn(BATCH_SIZE, T, OBJ_DIM,  device=device, requires_grad=True)

    hand_feat = hand_agg(hand_pose, obj_pose)
    obj_feat  = obj_agg(hand_pose, obj_pose)
    loss = hand_feat.mean() + obj_feat.mean()
    loss.backward()

    has_grad = all(p.grad is not None for p in list(hand_agg.parameters()) + list(obj_agg.parameters()) if p.requires_grad)
    print(f"  All parameters received gradients: {has_grad}  " + ("OK" if has_grad else "WARNING!"))
    print(f"  hand_pose.grad shape : {hand_pose.grad.shape}")
    print(f"  obj_pose.grad  shape : {obj_pose.grad.shape}")
    print("\n  Demo completed successfully.\n")