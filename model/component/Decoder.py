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
    

def fourier_encode(x: torch.Tensor, num_freqs: int = 5) -> torch.Tensor:
    """
    Fourier 位置编码（纯计算，无可学习参数）。
    x:       (B, C)
    return:  (B, C * 2 * num_freqs)
    """
    freqs = (2 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype))  # (L,)
    x = x.unsqueeze(-1) * freqs          # (B, C, L)
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, C, 2L)
    return x.flatten(1)                  # (B, C * 2L)


class PoseDecoder(nn.Module):
    """
    General Pose Decoder (shared structure for Hand and Obj).
    query_dim: input query 维度（由子类决定）
    """
    def __init__(
        self,
        query_dim: int,
        dim: int,
        num_layers: int,
        output_dim: int,
        num_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 将外部 query 投影到 dim
        self.query_proj = nn.Linear(query_dim, dim)

        # Stacked decoder layers
        self.layers = nn.ModuleList([
            PoseDecoderLayer(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, output_dim),
        )

    def forward(self, query_vec: torch.Tensor, dino_features: torch.Tensor):
        """
        query_vec:     (B, query_dim)   ← 由子类构造的 pose 向量
        dino_features: (B, Nk, D)       ← DINOv2 输出
        Returns:       (B, output_dim)
        """
        query = self.query_proj(query_vec).unsqueeze(1)  # (B, 1, D)

        for layer in self.layers:
            query = layer(query, dino_features)

        return self.head(query).squeeze(1)               # (B, output_dim)


class HandPoseDecoder(PoseDecoder):
    """
    Stage 1 (default): query = th_jtr (B, 21, 3) flattened to (B, 63)
    Stage 4 (query_dim override): query = feature vector (B, query_dim)
    Output dict: {'pose': (B,48), 'shape': (B,10), 'trans': (B,3)}
    """
    POSE_DIM   = 48
    SHAPE_DIM  = 10
    TRANS_DIM  = 3
    OUTPUT_DIM = POSE_DIM + SHAPE_DIM + TRANS_DIM  # 61
    DEFAULT_QUERY_DIM = 21 * 3                      # 63

    def __init__(self, dim, num_layers=6, num_heads=8, ffn_dim=None, dropout=0.0,
                 query_dim: int = None):
        actual_query_dim = query_dim if query_dim is not None else self.DEFAULT_QUERY_DIM
        super().__init__(
            query_dim=actual_query_dim,
            dim=dim,
            num_layers=num_layers,
            output_dim=self.OUTPUT_DIM,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self._custom_query = query_dim is not None

    def forward(self, query: torch.Tensor, dino_features: torch.Tensor):
        """
        query:         (B, 21, 3) joint coords  — Stage 1 (default)
                    or (B, query_dim) feature   — Stage 4 (query_dim override)
        dino_features: (B, Nk, D)
        """
        query_vec = query if self._custom_query else query.flatten(1)
        F = super().forward(query_vec, dino_features)    # (B, 61)
        return {
            "pose":  F[:, :self.POSE_DIM],
            "shape": F[:, self.POSE_DIM:self.POSE_DIM + self.SHAPE_DIM],
            "trans": F[:, self.POSE_DIM + self.SHAPE_DIM:],
        }


class ObjPoseDecoder(PoseDecoder):
    """
    Stage 1 (default): query = obj_pose (B, 6), Fourier-encoded to (B, 60)
    Stage 4 (query_dim override): query = feature vector (B, query_dim)
    Output dict: {'position': (B,3), 'rotation': (B,3)}
    """
    OUTPUT_DIM   = 6
    OBJ_POSE_DIM = 6
    NUM_FREQS    = 5    # default Fourier freqs → query_dim = 6 * 2 * 5 = 60

    def __init__(self, dim, num_layers=6, num_heads=8, ffn_dim=None, dropout=0.0,
                 query_dim: int = None):
        actual_query_dim = query_dim if query_dim is not None \
            else self.OBJ_POSE_DIM * 2 * self.NUM_FREQS  # 60
        super().__init__(
            query_dim=actual_query_dim,
            dim=dim,
            num_layers=num_layers,
            output_dim=self.OUTPUT_DIM,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self._custom_query = query_dim is not None

    def forward(self, query: torch.Tensor, dino_features: torch.Tensor):
        """
        query:         (B, 6) obj pose  — Stage 1 (default, Fourier-encoded internally)
                    or (B, query_dim) feature — Stage 4 (query_dim override)
        dino_features: (B, Nk, D)
        """
        query_vec = query if self._custom_query \
            else fourier_encode(query, self.NUM_FREQS)  # (B, 60)
        F = super().forward(query_vec, dino_features)   # (B, 6)
        return {
            "position": F[:, :3],
            "rotation": F[:, 3:],
        }


if __name__ == "__main__":
    B, D = 2, 1024
    dino_features = torch.randn(B, 1, D)   # DINOv2 CLS token → (B, 1, D)

    # Hand: th_jtr 来自 ManoLayer，shape (B, 21, 3)
    th_jtr   = torch.randn(B, 21, 3)
    # Obj: position(3) + rotation(3)
    obj_pose = torch.randn(B, 6)

    hand_decoder = HandPoseDecoder(dim=D, num_layers=4, num_heads=8)
    obj_decoder  = ObjPoseDecoder(dim=D,  num_layers=4, num_heads=8)

    hand_out = hand_decoder(th_jtr, dino_features)
    obj_out  = obj_decoder(obj_pose, dino_features)

    print("=== HandPoseDecoder ===")
    print(f"  query input:  th_jtr {th_jtr.shape} → flatten → (B, {21*3})")
    for k, v in hand_out.items():
        print(f"  {k}: {v.shape}")

    print("=== ObjPoseDecoder ===")
    print(f"  query input:  obj_pose {obj_pose.shape} → Fourier(L=5) → (B, {6*2*5})")
    for k, v in obj_out.items():
        print(f"  {k}: {v.shape}")