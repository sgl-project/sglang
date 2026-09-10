"""Fused SiLU-and-mul with static per-tensor FP8 quantization on SM89."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.activation.activation import _fast_math_flags
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def silu_and_mul_static_fp8_module(dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "silu_and_mul_static_fp8",
        *args,
        cuda_files=["elementwise/silu_and_mul_static_fp8.cuh"],
        extra_cuda_cflags=_fast_math_flags(),
        cuda_wrappers=[("run", f"SiluAndMulStaticFP8<{args}>::run")],
    )


@register_custom_op(mutates_args=["q", "rows"])
def silu_and_mul_static_fp8_out(
    x: torch.Tensor, scale: torch.Tensor, q: torch.Tensor, rows: torch.Tensor
) -> None:
    if torch.cuda.get_device_capability(x.device) != (8, 9):
        raise ValueError("SiLU static-FP8 fusion is qualified only on SM89")
    silu_and_mul_static_fp8_module(x.dtype).run(x, q, scale.reshape(1), rows)


def silu_and_mul_static_fp8(x, scale):
    """Quantize SiLU(gate) * up for x=[M, 2K] using a scalar FP32 scale.

    Return FP8 [M, K] activations and FP32 [M, 1] copies of the input scale.
    """
    if (
        not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.ndim != 2
        or not x.is_contiguous()
        or x.storage_offset() % 8
        or x.shape[1] == 0
        or x.shape[1] % 16
        or x.numel() > 2**32 - 1
        or scale.numel() != 1
        or scale.dtype != torch.float32
        or scale.device != x.device
        or not scale.is_contiguous()
    ):
        raise ValueError("unsupported SiLU static-FP8 fusion input")
    m, k2 = x.shape
    q = torch.empty((m, k2 // 2), device=x.device, dtype=torch.uint8)
    rows = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    silu_and_mul_static_fp8_out(x, scale, q, rows)
    return q.view(torch.float8_e4m3fn), rows
