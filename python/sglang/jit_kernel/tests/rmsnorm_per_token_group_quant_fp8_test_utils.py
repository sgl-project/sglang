import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-5
GROUP_SIZE = 128
TRACE_HIDDEN_SIZE = 2048
TRACE_ROW_STRIDE = 2624


def make_input(
    num_tokens: int,
    hidden_size: int = TRACE_HIDDEN_SIZE,
    *,
    trace_stride: bool = False,
) -> torch.Tensor:
    if trace_stride:
        assert hidden_size == TRACE_HIDDEN_SIZE
        backing = torch.randn(num_tokens, TRACE_ROW_STRIDE, device=DEVICE, dtype=DTYPE)
        x = backing[:, :hidden_size]
        assert x.stride() == (TRACE_ROW_STRIDE, 1)
        return x
    return torch.randn(num_tokens, hidden_size, device=DEVICE, dtype=DTYPE)


def alloc_outputs(
    num_tokens: int, hidden_size: int = TRACE_HIDDEN_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_q = torch.empty(num_tokens, hidden_size, device=DEVICE, dtype=fp8_dtype)
    output_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=(num_tokens, hidden_size),
        device=DEVICE,
        group_size=GROUP_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    output_norm = torch.empty(num_tokens, hidden_size, device=DEVICE, dtype=DTYPE)
    return output_q, output_s, output_norm
