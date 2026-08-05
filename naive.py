import torch
import torch.nn.functional as F

def naive_decay_gated_attention(q, k, v, g):
    """
    Canonical Pure PyTorch Reference for Decay-Gated O(N) Causal Linear Attention.
    q, k, v: [B, H, T, D]
    g: [B, H, T, D] or [B, H, T, 1]
    """
    B, H, T, D = q.shape
    
    # Broadcast scalar gate to D dimension if needed
    if g.shape[-1] == 1:
        g = g.expand(-1, -1, -1, D)
        
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0
    
    output = torch.zeros_like(v)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    
    for t in range(T):
        decay = torch.exp(-torch.abs(g[:, :, t:t+1, :])) # [B, H, 1, D]
        S = S * decay
        
        kt = k[:, :, t, :].unsqueeze(-1)  # [B, H, D, 1]
        vt = v[:, :, t, :].unsqueeze(-2)  # Fixed Axis: [B, H, 1, D]
        
        S = S + torch.matmul(kt, vt)
        
        qt = q[:, :, t, :].unsqueeze(-2)  # Fixed Axis: [B, H, 1, D]
        ot = torch.matmul(qt, S).squeeze(-2) # Fixed Axis: [B, H, D]
        
        output[:, :, t, :] = ot
        
    return output
