from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_GROUP_SIZE = 128


@cache_once
def _jit_module(in_dtype: torch.dtype, use_pdl: bool) -> Module:
    args = make_cpp_args(in_dtype, use_pdl)
    return load_jit(
        "dsv4_fp8_wo_a_group_major_quant_ue8m0",
        *args,
        cuda_files=["deepseek_v4/fp8_wo_a_group_major_quant.cuh"],
        cuda_wrappers=[
            (
                "fp8_wo_a_group_major_quant_ue8m0",
                f"FP8WoAGroupMajorQuantUE8M0Kernel<{args}>::run",
            )
        ],
        # Match the AOT/JIT v2 quant path's fast-math build so FP8 rounding stays
        # bit-identical for the DSV4 wo_a replacement.
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="fp8_wo_a_group_major_quant_ue8m0",
    mutates_args=["output_q", "output_s"],
)
def _fp8_wo_a_group_major_quant_ue8m0_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    """Opaque custom-op boundary for the DeepSeek-V4 wo_a quant JIT kernel."""
    assert input.dim() == 3, "input must have shape [T, G, D]"
    assert output_q.shape == input.shape
    assert input.stride(-1) == 1, "input hidden dimension must be contiguous"
    assert output_q.is_contiguous(), "output_q must be contiguous"
    assert input.dtype in (torch.bfloat16, torch.float16)
    assert input.shape[-1] % _GROUP_SIZE == 0
    element_alignment = 16 // input.element_size()
    assert input.data_ptr() % 16 == 0, "input base pointer must be 16-byte aligned"
    assert (
        input.stride(0) % element_alignment == 0
    ), "input token stride must preserve 16-byte vector-load alignment"
    assert (
        input.stride(1) % element_alignment == 0
    ), "input group stride must preserve 16-byte vector-load alignment"

    if input.numel() == 0:
        return

    num_tokens, num_groups, hidden = input.shape
    hidden_groups = hidden // _GROUP_SIZE
    assert output_s.shape == (num_tokens, num_groups, hidden_groups)
    assert output_s.dtype == torch.float32
    assert output_s.stride() == (
        hidden_groups,
        num_tokens * hidden_groups,
        1,
    ), "output_s must be the DSV4 group-major [T, G, D/128] view"

    module = _jit_module(input.dtype, is_arch_support_pdl())
    module.fp8_wo_a_group_major_quant_ue8m0(input, output_q, output_s)


@debug_kernel_api
def fp8_wo_a_group_major_quant_ue8m0(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    _fp8_wo_a_group_major_quant_ue8m0_custom_op(input, output_q, output_s)


def sglang_per_token_group_quant_fp8_dsv4_woa(
    x: torch.Tensor,
    group_size: int = _GROUP_SIZE,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3, "DSV4 wo_a quant expects [tokens, groups, hidden]"
    assert group_size == _GROUP_SIZE, "DSV4 wo_a quant uses group_size=128"
    assert eps == 1e-10, "DSV4 wo_a quant uses eps=1e-10"
    assert x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1, "`x` hidden dimension is not contiguous"
    assert x.is_cuda, "DSV4 wo_a quant is only supported on CUDA"

    num_tokens, num_groups, hidden = x.shape
    hidden_groups = hidden // group_size
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(
        (num_groups, num_tokens, hidden_groups),
        device=x.device,
        dtype=torch.float32,
    ).transpose(0, 1)

    if x.numel() > 0:
        fp8_wo_a_group_major_quant_ue8m0(x, x_q, x_s)

    return x_q, x_s
