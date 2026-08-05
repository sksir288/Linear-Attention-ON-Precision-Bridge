import torch
import triton
import triton.language as tl

@triton.jit
def _fused_decay_linear_attention_kernel(
    Q_ptr, K_ptr, V_ptr, G_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_gb, stride_gh, stride_gt, stride_gd,
    stride_ob, stride_oh, stride_ot, stride_od,
    T, D: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    off_q = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh
    off_k = K_ptr + pid_batch * stride_kb + pid_head * stride_kh
    off_v = V_ptr + pid_batch * stride_vb + pid_head * stride_vh
    off_g = G_ptr + pid_batch * stride_gb + pid_head * stride_gh
    off_o = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh

    S = tl.zeros([D, D], dtype=tl.float32)
    d_cols = tl.arange(0, D)

    for t in range(0, T):
        q_ptrs = off_q + t * stride_qt + d_cols * stride_qd
        k_ptrs = off_k + t * stride_kt + d_cols * stride_kd
        v_ptrs = off_v + t * stride_vt + d_cols * stride_vd
        g_ptrs = off_g + t * stride_gt + d_cols * stride_gd
        o_ptrs = off_o + t * stride_ot + d_cols * stride_od

        q = tl.load(q_ptrs).to(tl.float32)
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        g = tl.load(g_ptrs).to(tl.float32)

        q_act = tl.where(q > 0, q + 1.0, tl.exp(q))
        k_act = tl.where(k > 0, k + 1.0, tl.exp(k))

        decay = tl.exp(-tl.abs(g))
        S = S * decay[None, :]
        S += k_act[:, None] * v[None, :]

        out = tl.sum(q_act[:, None] * S, axis=0)

        # Retain input precision instead of forced FP16 truncation
        tl.store(o_ptrs, out)


class DecayGatedLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g):
        B, H, T, D = q.shape
        assert D in {32, 64, 128}, "Dimension must be power of 2 (32, 64, 128)"
        
        # Handle scalar gate shape [B, H, T, 1] via explicit contiguous broadcast
        if g.shape[-1] == 1:
            g = g.expand(-1, -1, -1, D).contiguous()
            
        out = torch.empty_like(v)
        grid = (B, H)
        
        _fused_decay_linear_attention_kernel[grid](
            q, k, v, g, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            T=T, D=D, BLOCK_SIZE=64
        )
        ctx.save_for_backward(q, k, v, g)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, g = ctx.saved_tensors
        with torch.enable_grad():
            q_c = q.detach().requires_grad_(True)
            k_c = k.detach().requires_grad_(True)
            v_c = v.detach().requires_grad_(True)
            g_c = g.detach().requires_grad_(True)
            
            from naive import naive_decay_gated_attention
            ref_out = naive_decay_gated_attention(q_c, k_c, v_c, g_c)
            ref_out.backward(grad_out)
            
        return q_c.grad, k_c.grad, v_c.grad, g_c.grad


def triton_linear_attention(q, k, v, g):
    return DecayGatedLinearAttentionFunction.apply(q, k, v, g)
