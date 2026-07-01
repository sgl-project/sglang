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
from sglang.srt.utils import ceil_align
from sglang.srt.utils.custom_op import register_custom_op

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(group_size: int) -> Module:
    args = make_cpp_args(group_size, is_arch_support_pdl())
    return load_jit(
        make_name("fp8_wo_a_group_major_quant_ue8m0"),
        *args,
        cuda_files=["deepseek_v4/fp8_wo_a_group_major_quant.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"Fp8WoAGroupMajorQuantUe8m0Kernel<{args}>::run",
            ),
        ],
        # Match the per-token-group quant v2 math used by the previous helper.
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="fp8_wo_a_group_major_quant_ue8m0",
    mutates_args=["o_fp8", "o_s"],
)
def _fp8_wo_a_group_major_quant_ue8m0_custom_op(
    o_group_major: torch.Tensor,
    o_fp8: torch.Tensor,
    o_s: torch.Tensor,
    group_size: int,
) -> None:
    module = _jit_module(group_size)
    module.run(
        o_group_major,
        o_fp8,
        o_s,
        int(o_s.stride(0)),
        int(o_s.stride(1)),
        int(o_s.stride(2)),
    )


@debug_kernel_api
def fp8_wo_a_group_major_quant_ue8m0(
    o_group_major: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize DeepSeek-V4 wo_a activations in one fused group-major launch.

    ``o_group_major`` is logical ``[G, T, D]``. The returned scale tensor is
    logical ``[G, T, ceil(D / group_size, 4) / 4]`` with four UE8M0 scale bytes
    packed per int32 and backed by the TMA-aligned ``[G, scale_inner, aligned_T]``
    layout expected by DeepGEMM.
    """
    assert o_group_major.is_cuda
    assert o_group_major.dtype == torch.bfloat16
    assert o_group_major.dim() == 3
    assert o_group_major.is_contiguous()

    G, T, D = o_group_major.shape
    assert D % group_size == 0, "D must be divisible by group_size"
    num_groups = D // group_size
    scale_inner = ceil_align(num_groups, 4) // 4
    aligned_t = ceil_align(T, 4)

    o_fp8 = torch.empty_like(o_group_major, dtype=torch.float8_e4m3fn)
    o_s = torch.empty(
        (G, scale_inner, aligned_t),
        device=o_group_major.device,
        dtype=torch.int32,
    ).transpose(-1, -2)[:, :T, :]
    _fp8_wo_a_group_major_quant_ue8m0_custom_op(
        o_group_major,
        o_fp8,
        o_s,
        int(group_size),
    )
    return o_fp8, o_s
