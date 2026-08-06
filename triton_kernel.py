import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

def _validate_inputs(q, k, v, g):
    tensors = (q, k, v, g)

    if not all(x.ndim == 4 for x in tensors):
        raise ValueError("q, k, v, and g must all be rank-4 tensors")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")

    B, H, T, D = q.shape

    if g.shape[:3] != (B, H, T):
        raise ValueError("g must match q in batch, head, and sequence dimensions")

    if g.shape[-1] not in (1, D):
        raise ValueError("g must have shape [B, H, T, 1] or [B, H, T, D]")

    if not all(x.is_cuda for x in tensors):
        raise ValueError("all inputs must be CUDA tensors")

    if len({x.device for x in tensors}) != 1:
        raise ValueError("all inputs must be on the same CUDA device")

    if not all(x.is_floating_point() for x in tensors):
        raise TypeError("all inputs must use floating-point dtypes")

    if B == 0 or H == 0 or T == 0:
        raise ValueError("empty batch, head, or sequence dimensions are not supported")

    if D not in (32, 64, 128):
        raise ValueError("D must be one of 32, 64, or 128")


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

        tl.store(o_ptrs, out)


class DecayGatedLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g):
        _validate_inputs(q, k, v, g)

        B, H, T, D = q.shape
        
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
    @once_differentiable
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
