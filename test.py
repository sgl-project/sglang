import torch
from sglang.jit_kernel import fused_moe

i = torch.randn(1, device='cuda')

fused_moe.moe_wna16_marlin_gemm(i, None, i, None, i, None, None, None, None,i, i, i, i, i, 8, 1, True, True, 1, 1, 1, True,  True, True, True)