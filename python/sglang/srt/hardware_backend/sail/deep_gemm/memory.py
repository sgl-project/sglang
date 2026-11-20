"""Memory budget for the kernel types only the SAIL DeepGEMM build serves.

``layers/deep_gemm_wrapper/compile_utils.py`` prices the kernel types every
DeepGEMM build serves. For a type its own table does not cover,
``_BaseWarmupExecutor.get_memory_requirement`` imports this module and asks here,
the same on-demand import ``create`` uses to reach ``warmup``.

A budget has to bound the buffers its executor holds live at the widest M of the
warmup list: the compile path halves ``max_m`` until the estimate fits the free
device memory, then walks the whole list. An under-estimate is an OOM at the
widest M; an over-estimate needlessly shrinks the list. ``test_warmup_memory``
asserts every estimate is within 64 bytes of what the executor actually
allocates, which is what pins these formulas to the executors in ``warmup``.

Scale-layout conversion is priced through the shared
``_get_fp8_scale_workspace`` rather than reimplemented: the estimate has to
follow ``DEEPGEMM_NEED_TMA_ALIGNED_SCALES`` / ``DEEPGEMM_SCALE_UE8M0``, which
are read in that module's scope and are patched there by the test.
"""

from enum import Enum
from typing import Callable, Dict

from sglang.srt.hardware_backend.sail.deep_gemm.kernel_types import (
    SailDeepGemmKernelType,
)
from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
    _BLOCK_SIZE,
    _get_fp8_scale_workspace,
)
from sglang.srt.utils import ceil_div

_GB = 1 << 30


def _fused_requirement(name: str, max_m: int, n: int, k: int, num_groups: int) -> int:
    """GEMM1 reads the routing indices instead of a scattered lhs.

    Warmup runs with topk=1; the metadata terms bound the vendor's block_m
    16..256 schedule.
    """
    n_align = max_m + 1 - num_groups
    max_blocks = num_groups - 1 + ceil_div(n_align, 16)
    sorted_rows = (num_groups - 1 + ceil_div(n_align, 256)) * 256
    block_size = 1 << max(0, num_groups - 1).bit_length()
    padded_blocks = ceil_div(sorted_rows, block_size)
    alignment_bytes = (
        40
        + num_groups * 4
        + max_blocks * 16
        + sorted_rows * 4
        + 4
        + max_m * 8
        + (2 * padded_blocks + 2) * num_groups * 4
    )
    if "BF16I4BF16" in name:
        scale_bytes = 1 if "MMA" in name or "VALU" in name else 2
        tensor_bytes = (
            max_m * k * 2
            + num_groups * n * (k // 2)
            + num_groups * n * (k // 32) * scale_bytes
            + max_m * n * 2
        )
        return tensor_bytes + alignment_bytes
    bf16 = name == "GROUPED_GEMM_NT_BF16_FUSED"
    tensor_bytes = (max_m * k + num_groups * n * k) * (2 if bf16 else 1) + max_m * n * 2
    if not bf16:
        if "F4F4" in name:
            # K is the packed byte width; a uint16 holds the scales of two g32
            # groups. During RHS preprocessing the raw scales and the output are
            # live at the same time, while the LHS is only a view.
            tensor_bytes += (max_m + 2 * num_groups * n) * ceil_div(k, 32) * 2
        elif "CHANNEL" in name or "I8I8" in name:
            tensor_bytes += (max_m + num_groups * n) * 4
        else:
            tensor_bytes += (
                max_m * ceil_div(k, _BLOCK_SIZE)
                + num_groups * ceil_div(n, _BLOCK_SIZE) * ceil_div(k, _BLOCK_SIZE)
            ) * 4
    return tensor_bytes + alignment_bytes


def _channel_like_requirement(
    name: str, max_m: int, n: int, k: int, num_groups: int
) -> int:
    """Per-channel fp8, int8 and fp4 layouts, together with their bias variants."""
    masked = "MASKED" in name
    grouped = name.startswith("GROUPED_")
    lhs_groups = num_groups if masked else 1
    rhs_groups = num_groups if grouped else 1
    # For FP4 scales the input and the preprocessed output are live at the same
    # time; the packed K is already a byte width.
    scale_bytes = 2 * (k // 16) if "F4F4" in name else 4
    return (
        lhs_groups * max_m * k
        + rhs_groups * n * k
        + (lhs_groups * max_m + rhs_groups * n) * scale_bytes
        + (num_groups if masked else max_m) * 4 * grouped
        + lhs_groups * max_m * n * 2
        + (rhs_groups * n * 4 if name.endswith("_BIAS") else 0)
    )


def _fp8_nopad_requirement(
    _name: str, max_m: int, n: int, k: int, num_groups: int
) -> int:
    """Blockwise fp8 with the per-expert row counts handed in by the caller."""
    lhs_scales = max_m * ceil_div(k, _BLOCK_SIZE) * 4
    rhs_scales = ceil_div(n, _BLOCK_SIZE) * ceil_div(k, _BLOCK_SIZE) * 4
    return (
        max_m * k
        + num_groups * n * k
        + max_m * 4
        + max_m * n * 2
        + lhs_scales
        + _get_fp8_scale_workspace(
            max_m, n, k, lhs_groups=1, rhs_groups=num_groups, block_size=_BLOCK_SIZE
        )
        + num_groups * rhs_scales
    )


def _bf16_nopad_requirement(
    _name: str, max_m: int, n: int, k: int, num_groups: int
) -> int:
    """bf16 with the per-expert row counts handed in by the caller."""
    return max_m * k * 2 + num_groups * n * k * 2 + max_m * 4 + max_m * n * 2


def _int4_requirement(name: str, max_m: int, n: int, k: int, num_groups: int) -> int:
    """W4A16 layouts: the MMA encoding keeps the packed 4-bit K, the VALU one
    pairs fp4 weights with a bf16 scale."""
    masked = "MASKED" in name
    if "MMA" in name:
        lhs_groups = num_groups if masked else 1
        return (
            lhs_groups * max_m * k * 2
            + num_groups * n * (k // 2)
            + num_groups * n * (k // 32)
            + (num_groups if masked else max_m) * 4
            + lhs_groups * max_m * n * 2
        )
    if masked:
        return (
            num_groups * max_m * k * 2
            + num_groups * (k // 16) * (n * 2) * 4
            + num_groups * (k // 32) * n * 2
            + num_groups * 4
            + num_groups * max_m * n * 2
        )
    return (
        max_m * k * 2
        + num_groups * (k // 16) * (n * 2) * 4
        + num_groups * (k // 32) * n * 2
        + max_m * 4
        + max_m * n * 2
    )


# One entry per member of SailDeepGemmKernelType: an unpriced type raises below
# instead of estimating zero, and test_warmup_memory walks the whole enum.
_SAIL_MEMORY_REQUIREMENTS: Dict[Enum, Callable[..., int]] = {
    # Grouped row-count (nopad) and per-channel variants of the fp8 layouts.
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD: _fp8_nopad_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL: _channel_like_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL: _channel_like_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL: _channel_like_requirement,
    SailDeepGemmKernelType.GEMM_NT_F8F8BF16_CHANNEL: _channel_like_requirement,
    # int8.
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_MASKED: _channel_like_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG: _channel_like_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_NOPAD: _channel_like_requirement,
    SailDeepGemmKernelType.GEMM_NT_I8I8BF16: _channel_like_requirement,
    # bf16.
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16_NOPAD: _bf16_nopad_requirement,
    # fp4, including the bias variants.
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED: _channel_like_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD: _channel_like_requirement,
    SailDeepGemmKernelType.GEMM_NT_F4F4BF16: _channel_like_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS: _channel_like_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS: _channel_like_requirement,
    SailDeepGemmKernelType.GEMM_NT_F4F4BF16_BIAS: _channel_like_requirement,
    # int4 weight encodings.
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MASKED: _int4_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_NOPAD: _int4_requirement,
    # Gather-read fused GEMMs.
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED: _fused_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL: _fused_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_FUSED: _fused_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16_FUSED: _fused_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED: _fused_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED: _int4_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD: _int4_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED: _fused_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED: _fused_requirement,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED: _fused_requirement,
}


def sail_memory_requirement(
    kernel_type: Enum, max_m: int, n: int, k: int, num_groups: int
) -> float:
    """Price one build-specific kernel type, in GiB."""
    price = _SAIL_MEMORY_REQUIREMENTS.get(kernel_type)
    if price is None:
        raise ValueError(f"Invalid kernel type: {kernel_type.name}")
    return price(kernel_type.name, max_m, n, k, num_groups) / _GB
