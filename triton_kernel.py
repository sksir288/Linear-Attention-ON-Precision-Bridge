import torch
import triton
import triton.language as tl

@triton.jit
def _linear_attn_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, D: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh

    kv_state = tl.zeros([D, D], dtype=tl.float32)
    k_state = tl.zeros([D], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        mask = offs_n[:, None] < N

        k_blk = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask, other=0.0)
        v_blk = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask, other=0.0)
        q_blk = tl.load(q_ptr + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd, mask=mask, other=0.0)

        q_blk = tl.where(q_blk > 0, q_blk + 1.0, tl.exp(q_blk))
        k_blk = tl.where(k_blk > 0, k_blk + 1.0, tl.exp(k_blk))

        kv_state += tl.dot(tl.trans(k_blk), v_blk)
        k_state += tl.sum(k_blk, axis=0)

        num = tl.dot(q_blk, kv_state)
        den = tl.sum(q_blk * k_state[None, :], axis=1, keep_dims=True) + 1e-6
        out_blk = num / den

        tl.store(out_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od, out_blk, mask=mask)

def triton_linear_attention(q, k, v):
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    BLOCK_N = 64
    grid = (B, H)

    _linear_attn_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N=N, D=D, BLOCK_N=BLOCK_N
    )
    return out
