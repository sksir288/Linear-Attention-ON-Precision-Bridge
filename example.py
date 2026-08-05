import torch
from naive import naive_decay_gated_attention

# Parameters
batch_size = 1
num_heads = 4
seq_len = 512
dim = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup sample inputs
q = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
k = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
v = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
g = torch.randn(batch_size, num_heads, seq_len, dim, device=device)

# Execute Naive Canonical Operator
output = naive_decay_gated_attention(q, k, v, g)

print(f"✅ Canonical Reference Execution Successful!")
print(f"Output Tensor Shape: {output.shape}")
