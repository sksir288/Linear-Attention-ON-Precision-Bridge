import torch
import torch.nn.functional as F

def naive_decay_gated_attention(q, k, v, g):
    """
    Canonical Pure PyTorch Reference for Decay-Gated O(N) Causal Linear Attention.
    
    q: [B, H, T, D]
    k: [B, H, T, D]
    v: [B, H, T, D]
    g: [B, H, T, D] or [B, H, T, 1] - Decay gate values in log-space or raw gate
    """
    B, H, T, D = q.shape
    
    # Ensure FP32 precision for ground truth accumulator
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    
    # Feature Map (ELU + 1 for non-negativity)
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0
    
    output = torch.zeros_like(v)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    
    for t in range(T):
        # Current token gate decay factor
        decay = torch.exp(-torch.abs(g[:, :, t:t+1, :])) # shape [B, H, 1, D]
        
        # Decay past state: S_t = S_{t-1} * decay + (K_t^T x V_t)
        # S is [B, H, D_k, D_v]
        S = S * decay
        
        # Outer product K_t^T * V_t
        kt = k[:, :, t, :].unsqueeze(-1)  # [B, H, D, 1]
        vt = v[:, :, t, :].unsqueeze(1)   # [B, H, 1, D]
        
        S = S + torch.matmul(kt, vt)
        
        # Compute output: O_t = Q_t * S_t
        qt = q[:, :, t, :].unsqueeze(1)   # [B, H, 1, D]
        ot = torch.matmul(qt, S).squeeze(1) # [B, H, D]
        
        output[:, :, t, :] = ot
        
    return output
