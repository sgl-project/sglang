import torch
from sgl_kernel import int8_scaled_mm
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm

M, N, K = 1024, 4096, 8192


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


a = to_int8(torch.randn((M, K), device="cuda") * 5)
b = to_int8(torch.randn((N, K), device="cuda").t() * 5)
o = torch.empty((M, N), device="cuda", dtype=torch.float16)
scale_a = torch.ones((M,), device="cuda", dtype=torch.float32)
scale_b = torch.ones((N,), device="cuda", dtype=torch.float32)
bias = torch.zeros((N,), device="cuda", dtype=torch.bfloat16)

o1 = vllm_scaled_mm(a, b, scale_a, scale_b, torch.float16)
print(o1.shape)
print(o1)
int8_scaled_mm(o, a, b, scale_a, scale_b, None)
print(o.shape)
print(o)
