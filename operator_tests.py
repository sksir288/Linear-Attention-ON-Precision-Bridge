import torch
import pytest
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

def test_validation_layer():
    if not torch.cuda.is_available():
        return

    from triton_kernel import triton_linear_attention

    q = torch.randn(1, 1, 7, 32, device="cuda")
    
    # Mismatched sequence length check
    k_short_T = torch.randn(1, 1, 6, 32, device="cuda")
    v = torch.randn(1, 1, 7, 32, device="cuda")
    g = torch.randn(1, 1, 7, 32, device="cuda")

    try:
        triton_linear_attention(q, k_short_T, v, g)
        assert False, "Failed to reject mismatched T shape"
    except ValueError:
        pass

    # Non-floating dtype check
    q_int = torch.randint(0, 10, (1, 1, 7, 32), device="cuda")
    try:
        triton_linear_attention(q_int, q_int, q_int, q_int)
        assert False, "Failed to reject integer inputs"
    except TypeError:
        pass

    print("✅ Host-side Input Validation Layer Passed!")

if __name__ == "__main__":
    test_causality()
    test_validation_layer()
