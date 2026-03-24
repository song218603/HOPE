import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KinematicTemporalPosEmbedding(nn.Module):
    def __init__(self, num_slots, max_frames, dim, use_kinematic=True):
        ...
        # 所有 branch 都有
        self.temporal_emb = nn.Parameter(torch.zeros(num_slots, max_frames, dim))
        self.slot_emb     = nn.Parameter(torch.zeros(num_slots, 1, dim))  # 替代 kinematic

        # 仅 Hand branch 额外叠加
        self.use_kinematic = use_kinematic
        if use_kinematic:
            self.kinematic_emb = nn.Parameter(torch.zeros(num_slots, 1, dim))

        nn.init.trunc_normal_(self.temporal_emb,  std=0.02)
        nn.init.trunc_normal_(self.slot_emb,      std=0.02)
        if use_kinematic:
            nn.init.trunc_normal_(self.kinematic_emb, std=0.02)

    def forward(self, x, T):
        S = self.num_slots
        t_emb = self.temporal_emb[:, :T, :]        # (S, T, C)
        s_emb = self.slot_emb.expand(S, T, -1)     # (S, T, C)
        emb = t_emb + s_emb

        if self.use_kinematic:
            k_emb = self.kinematic_emb.expand(S, T, -1)
            emb = emb + k_emb                      # 叠加运动学信息

        return x + emb.reshape(1, S * T, -1)
    

class TemporalSplitterLayer(nn.Module):
    """[LayerNorm → Q/K/V Conv → ReLU Attention + residual → FFN + residual]"""
    def __init__(self, dim: int, ffn_dim: int = None, dropout: float = 0.0):
        super().__init__()
        ffn_dim = ffn_dim or dim * 2
        self.norm    = nn.LayerNorm(dim)
        self.q_conv  = nn.Conv1d(dim, dim, kernel_size=1)
        self.k_conv  = nn.Conv1d(dim, dim, kernel_size=1)
        self.v_conv  = nn.Conv1d(dim, dim, kernel_size=1)
        self.scale   = dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ffn_dim, dim, kernel_size=1),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, C)
        h   = self.norm(x)
        h_t = h.transpose(1, 2)                        # (B, C, N)
        Q   = self.q_conv(h_t).transpose(1, 2)        # (B, N, C)
        K   = self.k_conv(h_t).transpose(1, 2)
        V   = self.v_conv(h_t).transpose(1, 2)

        attn = F.relu((Q @ K.transpose(-2, -1)) * self.scale)  # ReLU!
        attn = self.dropout(attn)
        R    = x + attn @ V                            # residual 1

        ffn_out = self.ffn(R.transpose(1, 2)).transpose(1, 2)
        return self.ffn_norm(R + ffn_out)
    

class TemporalSplitter(nn.Module):
    """
    手或物体 branch 的跨帧时序细化模块。
    输入: S 个 (B, T, C) 特征流
    输出: S 个 (B, T, C) 细化特征流
    """
    def __init__(
            self, 
            dim: int,
            num_slots: int = 3,
            num_layers: int = 2,
            max_frames: int = 512,
            ffn_dim: int = None,
            dropout: float = 0.0,
            use_kinematic=True
            ):
        super().__init__()
        self.dim       = dim
        self.num_slots = num_slots
        self.pos_emb   = KinematicTemporalPosEmbedding(num_slots, max_frames, dim, use_kinematic)
        self.layers    = nn.ModuleList([
            TemporalSplitterLayer(dim, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.psi = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
            for _ in range(num_slots)
        ])

    def forward(self, F_list):
        # F_list: list of S tensors (B, T, C)
        B, T, C = F_list[0].shape
        x = torch.stack(F_list, dim=1).reshape(B, self.num_slots * T, C)
        x = self.pos_emb(x, T)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(B, self.num_slots, T, C)
        return [self.psi[s](x[:, s]) for s in range(self.num_slots)]
    

if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    BATCH_SIZE = 2
    DIM        = 256
    NUM_SLOTS  = 3
    NUM_LAYERS = 2

    model = TemporalSplitter(
        dim=DIM, num_slots=NUM_SLOTS, num_layers=NUM_LAYERS,
        max_frames=512, dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print(f"  TemporalSplitter params : {total_params:>10,}")
    print("=" * 60)

    # -- Variable-length forward pass -----------------------------------------
    print("\n  Variable-length sequence test (eval mode)")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        for T in [8, 16, 32]:
            F_list = [torch.randn(BATCH_SIZE, T, DIM).to(device) for _ in range(NUM_SLOTS)]
            t0 = time.time()
            F_out = model(F_list)
            elapsed = (time.time() - t0) * 1000
            shapes = [tuple(f.shape) for f in F_out]
            print(f"  T={T:>3d} | outputs={shapes} | {elapsed:.2f} ms")
            for f in F_out:
                assert f.shape == (BATCH_SIZE, T, DIM)

    print("\n  All shape assertions passed OK")

    # -- Pos embedding shapes -------------------------------------------------
    print("\n" + "-" * 60)
    print("  Positional Embedding Shapes")
    print("-" * 60)
    print(f"  temporal_emb  : {tuple(model.pos_emb.temporal_emb.shape)}"
          f"  (num_slots={NUM_SLOTS}, max_frames, dim)")
    print(f"  kinematic_emb : {tuple(model.pos_emb.kinematic_emb.shape)}"
          f"  (num_slots={NUM_SLOTS}, 1, dim)")

    # -- Gradient check -------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Gradient Flow Check (training mode)")
    print("-" * 60)
    model.train()
    T = 16
    F_list = [torch.randn(BATCH_SIZE, T, DIM, requires_grad=True).to(device)
              for _ in range(NUM_SLOTS)]
    loss = sum(f.mean() for f in model(F_list))
    loss.backward()

    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  All parameters received gradients : {has_grad}  {'OK' if has_grad else 'WARNING!'}")
    for i, f in enumerate(F_list):
        print(f"  F{i+1}.grad shape : {f.grad.shape}")

    print("\n  Demo completed successfully.\n")