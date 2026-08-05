import torch
from naive import naive_decay_gated_attention

def test_causality():
    """ Verify future tokens do not alter past token outputs """
    B, H, T, D = 1, 2, 128, 64
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    g = torch.randn(B, H, T, D)

    out1 = naive_decay_gated_attention(q, k, v, g)

    # Mutate last token V[T-1]
    v_mod = v.clone()
    v_mod[:, :, -1, :] += 10.0

    out2 = naive_decay_gated_attention(q, k, v_mod, g)

    # Outputs for tokens 0..T-2 must remain IDENTICAL
    diff = torch.abs(out1[:, :, :-1, :] - out2[:, :, :-1, :]).max().item()
    assert diff == 0.0, f"Causality Violation Detected! Max diff: {diff}"
    print("✅ Causality Test Passed: Future tokens do not affect past outputs.")

if __name__ == "__main__":
    test_causality()
