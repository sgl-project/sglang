from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_GROUP_SIZE = 128


@cache_once
def _jit_module(in_dtype: torch.dtype, use_pdl: bool) -> Module:
    args = make_cpp_args(in_dtype, use_pdl)
    return load_jit(
        make_name("fp8_wo_a_group_major_quant_ue8m0"),
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
    assert input.dtype in (torch.bfloat16, torch.float16)

    module = _jit_module(input.dtype, is_arch_support_pdl())
    module.fp8_wo_a_group_major_quant_ue8m0(input, output_q, output_s)


@debug_kernel_api
def fp8_wo_a_group_major_quant_ue8m0(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    _fp8_wo_a_group_major_quant_ue8m0_custom_op(input, output_q, output_s)


def sglang_per_token_group_quant_fp8_dsv4_wo_a(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize DSV4 wo_a activations for DeepGEMM fp8_einsum.

    The input is a [T, G, D] bf16/fp16 tensor whose hidden dimension is
    contiguous. The output codes are contiguous [T, G, D] fp8 values. The scale
    tensor is returned as logical [T, G, D/128] fp32 UE8M0 values backed by
    contiguous [G, T, D/128] storage, so each group/head [T, S] panel is
    contiguous for the DeepGEMM recipe=(1, 1, 128) consumer. Group size is fixed
    to 128 and the absmax floor is fixed to 1e-10.
    """
    num_tokens, num_groups, hidden = x.shape
    hidden_groups = hidden // _GROUP_SIZE
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    x_s_storage = torch.empty(
        (num_groups, num_tokens, hidden_groups),
        device=x.device,
        dtype=torch.float32,
    )

    if x.numel() > 0:
        fp8_wo_a_group_major_quant_ue8m0(x, x_q, x_s_storage)

    # DeepGEMM fp8_einsum consumes each group/head [T, S] scale panel contiguously.
    return x_q, x_s_storage.transpose(0, 1)
