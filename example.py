import torch
from triton_kernel import FusedLinearAttention

batch_size = 1
seq_len = 131072  # 128K Context
dim = 64
num_heads = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

q = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=torch.float16)
k = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=torch.float16)
v = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=torch.float16)

attention_layer = FusedLinearAttention(dim=dim, num_heads=num_heads).to(device)

with torch.no_grad():
    output = attention_layer(q, k, v)

print(f"Execution Successful! Output shape: {output.shape}")
