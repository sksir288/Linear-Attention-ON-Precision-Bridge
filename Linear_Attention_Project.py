import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, proj_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj, self.k_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.v_proj, self.out_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.phi_q = nn.Sequential(nn.Linear(self.head_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.phi_k = nn.Sequential(nn.Linear(self.head_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, x):
        B, N, D = x.shape
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        phi_Q = F.softplus(self.phi_q(Q), beta=2.0) + 1e-5
        phi_K = F.softplus(self.phi_k(K), beta=2.0) + 1e-5
        
        KV_cum = torch.cumsum(torch.einsum('bhni,bhnj->bhnij', phi_K, V), dim=2)
        Z_cum = torch.cumsum(phi_K, dim=2)
        
        num = torch.einsum('bhni,bhnij->bhnj', phi_Q, KV_cum)
        den = torch.einsum('bhni,bhni->bhn', phi_Q, Z_cum).unsqueeze(-1) + 1e-6
        return self.out_proj((num / den).transpose(1, 2).contiguous().view(B, N, D))

print("Linear Attention Architecture Ready!")
