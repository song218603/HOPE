import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    """
    将标量时间 t 编码成高维向量
    """

    def __init__(self, dim_out, max_freq=10):
        super().__init__()
        self.freqs = torch.pow(2, torch.arange(max_freq)) 
        self.dim_out = dim_out

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1) # 变为 (B, 1)
            
        B = t.shape[0]
        freqs = self.freqs.to(t.device).unsqueeze(0).repeat(B, 1) # (B, max_freq)
        t_scaled = t * freqs 
        emb = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1) # (B, 2 * max_freq)
        
        if self.dim_out != emb.shape[-1]:
             print("Warning: Temporal Embedding output dim might not match C. Add Linear layer if needed.")
             pass
             
        return emb

class TemporalFilter(nn.Module):
    def __init__(self):
        super().__init__()