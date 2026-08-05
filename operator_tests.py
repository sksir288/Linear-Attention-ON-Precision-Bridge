import torch
from naive import naive_decay_gated_attention

def test_causality():
    B, H, T, D = 1, 2, 128, 64
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    g = torch.randn(B, H, T, D)

    out1 = naive_decay_gated_attention(q, k, v, g)

    v_mod = v.clone()
    v_mod[:, :, -1, :] += 10.0

    out2 = naive_decay_gated_attention(q, k, v_mod, g)

    diff = torch.abs(out1[:, :, :-1, :] - out2[:, :, :-1, :]).max().item()
    assert diff == 0.0, f"Causality Violation Detected! Max diff: {diff}"
    print("✅ Causality Test Passed!")

def test_triton_parity_and_scalar_gate():
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping Triton parity check.")
        return

    from triton_kernel import triton_linear_attention

    B, H, T, D = 2, 4, 128, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    
    # Test Scalar Gate Broadcasting [B, H, T, 1]
    g_scalar = torch.randn(B, H, T, 1, device="cuda", dtype=torch.float32)

    out_naive = naive_decay_gated_attention(q, k, v, g_scalar)
    out_triton = triton_linear_attention(q, k, v, g_scalar)

    torch.testing.assert_close(out_naive, out_triton, rtol=1e-3, atol=1e-3)
    print("✅ Scalar Gate Broadcasting & Relative Parity Test Passed!")

if __name__ == "__main__":
    test_causality()
    test_triton_parity_and_scalar_gate()
