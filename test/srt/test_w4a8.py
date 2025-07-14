import sgl_kernel
import torch

x = torch.randn(10, 10, device="cuda")
qweight = torch.randn(10, 10, device="cuda")
s1_scales = torch.randn(10, device="cuda")
input_scales = torch.randn(10, device="cuda")
s1_szeros = torch.randn(10, device="cuda")
input_sum = torch.randn(10, device="cuda")
output_buffer = torch.randn(10, device="cuda")

torch.ops.sgl_kernel.gemm_forward_cuda.default(
    x, qweight, s1_scales, input_scales, s1_szeros, input_sum, output_buffer
)
