import torch
from flash_attn import flash_attn_func

# A100 loves BF16
device = "cuda"
dtype = torch.bfloat16

# Shapes: [Batch, SeqLen, NumHeads, HeadDim]
# Note: HeadDim must be a multiple of 8, typically up to 256
q = torch.randn(2, 2048, 8, 128, device=device, dtype=dtype)
k = torch.randn(2, 2048, 8, 128, device=device, dtype=dtype)
v = torch.randn(2, 2048, 8, 128, device=device, dtype=dtype)

# FlashAttention-2 execution
# softmax_scale is 1/sqrt(head_dim) by default
output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)