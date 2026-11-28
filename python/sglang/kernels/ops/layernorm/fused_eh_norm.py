from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def is_supported_fused_eh_norm_hidden_size(hidden_size: int) -> bool:
    return hidden_size > 256 and hidden_size <= 8192 and hidden_size % 256 == 0


@cache_once
def _jit_fused_eh_norm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "fused_eh_norm",
        *args,
        cuda_files=["elementwise/fused_eh_norm.cuh"],
        cuda_wrappers=[("fused_eh_norm", f"FusedEHNormKernel<{args}>::run")],
    )


def fused_eh_norm(
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return fused EH norm + cat for contiguous CUDA fp16/bf16 tensors."""
    if inputs_embeds.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"fused_eh_norm: unsupported dtype {inputs_embeds.dtype}; "
            "expected torch.float16 or torch.bfloat16"
        )
    if inputs_embeds.dim() != 2:
        raise RuntimeError(
            f"fused_eh_norm: inputs_embeds must be 2D, got {inputs_embeds.dim()}D"
        )
    hidden_size = inputs_embeds.shape[1]
    if not is_supported_fused_eh_norm_hidden_size(hidden_size):
        raise RuntimeError(
            f"fused_eh_norm: unsupported hidden_size={hidden_size} "
            "(must be in (256, 8192] and a multiple of 256)"
        )
    output = torch.empty(
        (inputs_embeds.shape[0], hidden_size * 2),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    if inputs_embeds.shape[0] == 0:
        return output
    module = _jit_fused_eh_norm_module(hidden_size, inputs_embeds.dtype)
    module.fused_eh_norm(
        inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, output, eps
    )
    return output


@triton.jit
def _fused_eh_norm_kernel(
    inputs_embeds,
    previous_hidden,
    enorm_weight,
    hnorm_weight,
    output,
    embeds_row_stride,
    hidden_row_stride,
    hidden_size,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    half = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < hidden_size
    if half == 0:
        x = tl.load(inputs_embeds + row * embeds_row_stride + offsets, mask, other=0.0)
        w = tl.load(enorm_weight + offsets, mask, other=0.0)
    else:
        x = tl.load(
            previous_hidden + row * hidden_row_stride + offsets, mask, other=0.0
        )
        w = tl.load(hnorm_weight + offsets, mask, other=0.0)
    x = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_size + eps)
    y = x * rstd * w.to(tl.float32)
    tl.store(
        output + row * (2 * hidden_size) + half * hidden_size + offsets,
        y.to(output.dtype.element_ty),
        mask,
    )


def fused_eh_norm_triton(
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``cat(enorm(inputs_embeds), hnorm(previous_hidden))`` in one launch (the
    two RMSNorms and the cat are three otherwise). Rows need unit column stride."""
    rows, hidden_size = inputs_embeds.shape
    output = torch.empty(
        (rows, hidden_size * 2), dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    if rows:
        _fused_eh_norm_kernel[(rows, 2)](
            inputs_embeds,
            previous_hidden,
            enorm_weight,
            hnorm_weight,
            output,
            inputs_embeds.stride(0),
            previous_hidden.stride(0),
            hidden_size,
            eps,
            BLOCK=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return output
