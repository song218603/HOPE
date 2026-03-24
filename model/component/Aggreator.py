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
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_dim, dim, kernel_size=1),
            nn.LayerNorm(dim),          # applied after transpose
        )
        # We'll handle LayerNorm manually (needs (B,T,D) layout)
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

        # Output heads: project aggregated feature to hand / obj branches
        self.hand_out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.obj_out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, hand_pose, obj_pose):
        """
        Args:
            hand_pose : (B, T, hand_dim)
            obj_pose  : (B, T, obj_dim)
        Returns:
            hand_feat : (B, T, dim)
            obj_feat  : (B, T, dim)
        """
        # ── Concat hand & obj along feature dim ──────────────────────────────
        x = torch.cat([hand_pose, obj_pose], dim=-1)   # (B, T, hand_dim+obj_dim)

        # ── Stacked aggregator layers ─────────────────────────────────────────
        for layer in self.layers:
            x = layer(x)                               # (B, T, dim)

        # ── Split into hand / obj branches ───────────────────────────────────
        hand_feat = self.hand_out_proj(x)              # (B, T, dim)
        obj_feat  = self.obj_out_proj(x)               # (B, T, dim)

        return hand_feat, obj_feat
    

if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # ── Config ────────────────────────────────────────────────────────────────
    BATCH_SIZE = 2
    AGG_DIM    = 256   # aggregator internal dim
    NUM_LAYERS = 2

    # Hand pose: 61 (48 pose + 10 shape + 3 trans)
    # Obj  pose:  6 (3 position + 3 rotation)
    HAND_DIM = 61
    OBJ_DIM  = 6

    # Variable-length sequence test
    T_LIST = [8, 16, 32]

    # ── Build model ───────────────────────────────────────────────────────────
    model = Aggregator(
        hand_dim=HAND_DIM,
        obj_dim=OBJ_DIM,
        dim=AGG_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 55)
    print(f"  Aggregator params : {total_params:>10,}")
    print("=" * 55)

    # ── Variable-length forward pass ──────────────────────────────────────────
    print("\n  Variable-length sequence test (eval mode)")
    print("-" * 55)
    model.eval()
    with torch.no_grad():
        for T in T_LIST:
            hand_pose = torch.randn(BATCH_SIZE, T, HAND_DIM).to(device)
            obj_pose  = torch.randn(BATCH_SIZE, T, OBJ_DIM).to(device)

            t0 = time.time()
            hand_feat, obj_feat = model(hand_pose, obj_pose)
            elapsed = (time.time() - t0) * 1000

            print(f"  T={T:>3d} | "
                  f"hand_feat={tuple(hand_feat.shape)}  "
                  f"obj_feat={tuple(obj_feat.shape)}  "
                  f"| {elapsed:.2f} ms")

            assert hand_feat.shape == (BATCH_SIZE, T, AGG_DIM), "hand_feat shape mismatch"
            assert obj_feat.shape  == (BATCH_SIZE, T, AGG_DIM), "obj_feat shape mismatch"

    print("\n  All shape assertions passed OK")

    # ── Gradient flow check ───────────────────────────────────────────────────
    print("\n" + "-" * 55)
    print("  Gradient Flow Check (training mode)")
    print("-" * 55)
    model.train()
    T = 16
    hand_pose = torch.randn(BATCH_SIZE, T, HAND_DIM, requires_grad=True).to(device)
    obj_pose  = torch.randn(BATCH_SIZE, T, OBJ_DIM,  requires_grad=True).to(device)

    hand_feat, obj_feat = model(hand_pose, obj_pose)
    loss = hand_feat.mean() + obj_feat.mean()
    loss.backward()

    has_grad = all(
        p.grad is not None
        for p in model.parameters() if p.requires_grad
    )
    print(f"  All parameters received gradients: {has_grad}  "
          + ("OK" if has_grad else "WARNING!"))
    print(f"  hand_pose.grad shape : {hand_pose.grad.shape}")
    print(f"  obj_pose.grad  shape : {obj_pose.grad.shape}")

    print("\n  Demo completed successfully.\n")