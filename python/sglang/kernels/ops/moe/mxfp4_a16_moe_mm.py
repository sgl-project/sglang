from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import is_sm90_supported
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Group size and K tile of the instantiated collective; the caller's scale layout has to agree
# with both (see mxfp4_a16_grouped_mm_sm90.cuh).
MXFP4_A16_SCALE_GROUP_SIZE = 32
MXFP4_A16_TILE_K = 128
MXFP4_A16_PACKED_SCALES_NUM = MXFP4_A16_TILE_K // MXFP4_A16_SCALE_GROUP_SIZE


def _mxfp4_a16_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]


@cache_once
def _jit_mxfp4_a16_moe_mm_module() -> Module:
    """Compile and cache the SM90 MXFP4-weight bf16-activation grouped MoE GEMM."""
    if not is_sm90_supported():
        raise RuntimeError("mxfp4_a16_moe_mm requires SM90 (Hopper).")
    return load_jit(
        "mxfp4_a16_moe_mm_sm90",
        cuda_files=["gemm/cutlass_moe/mxfp4_a16_grouped_mm_sm90.cuh"],
        cuda_wrappers=[("mxfp4_a16_moe_mm_sm90", "mxfp4_a16_moe_mm_sm90")],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_mxfp4_a16_cuda_flags(),
    )


@register_custom_op(
    op_name="mxfp4_a16_moe_mm",
    mutates_args=["out"],
)
def _mxfp4_a16_moe_mm_custom_op(
    out: torch.Tensor,
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> None:
    module = _jit_mxfp4_a16_moe_mm_module()
    module.mxfp4_a16_moe_mm_sm90(out, a, b_q, b_scales, expert_offsets, problem_sizes)


@debug_kernel_api
def mxfp4_a16_moe_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> None:
    """Grouped GEMM over MoE experts: bf16 activations x MXFP4 weights, bf16 output.

    :param out: [m_total, n] bf16, rows sorted by expert.
    :param a: [m_total, k] bf16, rows sorted by expert.
    :param b_q: [num_experts, n, k // 2] uint8, two E2M1 codes per byte, low nibble is the even k.
    :param b_scales: [num_experts, k // 128, n * 4] uint8 E8M0 with the +126 exponent bias baked in,
                     interleaved so one row holds 4 consecutive groups per output element.
    :param expert_offsets: [num_experts] int32, first row of each expert in `a` / `out`.
    :param problem_sizes: [num_experts, 3] int32 per-expert (n, tokens, k).
    """
    _mxfp4_a16_moe_mm_custom_op(out, a, b_q, b_scales, expert_offsets, problem_sizes)
