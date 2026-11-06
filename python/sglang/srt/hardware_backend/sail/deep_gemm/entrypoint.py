"""SAIL-only DeepGEMM entry points.

The vendor ``deep_gemm`` build on SAIL adds nopad grouped GEMMs (the caller
passes the per-expert row counts explicitly), the INT8/FP4 entry points and the
gather-read fused MoE API. None of them exist in the CUDA build that
``layers/deep_gemm_wrapper`` targets, so they live here instead of growing the
shared wrapper.

The module also hosts the masked FP8/BF16 grouped GEMMs whose *signature* the
vendor build diverges on (per-channel kernel types, ``enable_sbo_overlap`` and
the optional overlap arguments), so the shared wrappers stay upstream-faithful
and SAIL callers import the variant from here.

Both the sibling custom-op wrapper ``sail_fused_deepgemm_ops`` and the SAIL MoE
runners import from this module directly. Routing them through
``deep_gemm_wrapper`` instead would make that package depend on
``srt.hardware_backend``, and the wrapper's ``from .entrypoint import *``
re-export cannot host these functions without a package-level import cycle.
"""

from typing import Any, Optional, Tuple

import torch

from sglang.srt.hardware_backend.sail.deep_gemm.kernel_types import (
    SailDeepGemmKernelType,
)
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
    _ensure_cuda,
    _sanity_check_input,
    configure_deep_gemm_num_sms,
)
from sglang.srt.utils import is_sail

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

_is_sail = is_sail()


def grouped_gemm_nt_f8f8bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
):
    assert _is_sail, "grouped_gemm_nt_f8f8bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD
    if _is_sail and lhs[1].shape[-1] == 1 and rhs[1].shape[-1] == 1:
        kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(lhs, rhs, out, m_indices, m_rows)


def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    """Masked FP8 grouped GEMM, per-channel aware.

    Differs from the shared wrapper in two ways: a per-channel scale selects the
    ``_CHANNEL`` kernel type, and SBO is requested as ``enable_sbo_overlap``
    (the CUDA build names the same switch ``enable_overlap``).
    """
    assert _is_sail, (
        "grouped_gemm_nt_f8f8bf16_masked is only supported by SAIL DeepGEMM"
    )

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED
    if lhs[1].shape[-1] == 1 and rhs[1].shape[-1] == 1:
        kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    lhs = _ensure_cuda(lhs)
    rhs = _ensure_cuda(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            fp4_kwargs = {}
            if recipe_a is not None:
                fp4_kwargs["recipe_a"] = recipe_a
            if recipe_b is not None:
                fp4_kwargs["recipe_b"] = recipe_b

            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **fp4_kwargs,
                **(
                    dict(
                        enable_sbo_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def grouped_gemm_nt_bf16_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    """BF16 masked grouped GEMM with the SBO switch this build expects."""
    assert _is_sail, "grouped_gemm_nt_bf16_masked is only supported by SAIL DeepGEMM"

    num_groups, _, k = a.shape
    _, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return deep_gemm.m_grouped_bf16_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            expected_m,
            **(
                dict(
                    enable_sbo_overlap=True,
                    max_block_n=max_block_n,
                    signal=overlap_args.signal,
                )
                if overlap_args is not None
                else {}
            ),
        )


# int8 implementation
def gemm_nt_i8i8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    assert _is_sail, "gemm_nt_i8i8bf16 is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = SailDeepGemmKernelType.GEMM_NT_I8I8BF16

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.gemm_int8_int8_bf16_nt(lhs, rhs, out)


def grouped_gemm_nt_i8i8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    assert _is_sail, (
        "grouped_gemm_nt_i8i8bf16_contig is only supported by SAIL DeepGEMM"
    )

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(lhs, rhs, out, m_indices)


def grouped_gemm_nt_i8i8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    assert _is_sail, (
        "grouped_gemm_nt_i8i8bf16_masked is only supported by SAIL DeepGEMM"
    )

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            **(
                dict(
                    enable_sbo_overlap=True,
                    max_block_n=max_block_n,
                    signal=overlap_args.signal,
                )
                if overlap_args is not None
                else {}
            ),
        )


def grouped_gemm_nt_i8i8bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
):
    assert _is_sail, "grouped_gemm_nt_i8i8bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_NOPAD

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(
            lhs, rhs, out, m_indices, m_rows
        )


def grouped_gemm_nt_bf16_nopad(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
):
    assert _is_sail, "grouped_gemm_nt_bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs.shape
    num_groups, n, _ = rhs.shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16_NOPAD

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
            lhs, rhs, out, m_indices, m_rows
        )


# mxfp4 implementation
def gemm_nt_f4f4bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
):
    assert _is_sail, "gemm_nt_f4f4bf16 is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1

    kernel_type = (
        SailDeepGemmKernelType.GEMM_NT_F4F4BF16
        if bias is None
        else SailDeepGemmKernelType.GEMM_NT_F4F4BF16_BIAS
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.gemm_fp4_fp4_bf16_nt(lhs, rhs, bias, out)


def grouped_gemm_nt_f4f4bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    assert _is_sail, (
        "grouped_gemm_nt_f4f4bf16_masked is only supported by SAIL DeepGEMM"
    )

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = (
        SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED
        if bias is None
        else SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS
    )

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            return deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
                lhs,
                rhs,
                bias,
                out,
                masked_m,
                expected_m,
                **(
                    dict(
                        enable_sbo_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def grouped_gemm_nt_f4f4bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
):
    assert _is_sail, "grouped_gemm_nt_f4f4bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape

    kernel_type = (
        SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD
        if bias is None
        else SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
            lhs, rhs, bias, out, m_indices, m_rows
        )


def _get_w4a16_weight_shape(weight: torch.Tensor) -> Tuple[int, int, int]:
    """Return the logical E/N/K; MMA consumes FP4 directly while VALU/INT4 use the
    Marlin layout."""
    if weight.dtype == torch.uint8:
        num_groups, n, packed_k = weight.shape
        return num_groups, n, packed_k * 2
    return weight.shape[0], weight.shape[2] // 2, weight.shape[1] * 16


def grouped_gemm_nt_bf16i4bf16_masked(
    lhs: torch.Tensor,
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    # lhs: shape [e, m, k], dtype bf16
    # VALU/INT4: int32 weights [e, k//16, n*2], scales [e, k//32, n].
    # Scales are uint8 E8M0 (VALU) or BF16 (INT4).
    # MMA: uint8 FP4 weights [e, n, k//2], uint8 tiled scales [e, n//64, k*2].

    assert _is_sail, (
        "grouped_gemm_nt_bf16i4bf16_masked is only supported by SAIL DeepGEMM"
    )

    num_groups, _, k = lhs.shape
    _, n, _ = _get_w4a16_weight_shape(rhs[0])

    kernel_type = (
        SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED
        if rhs[0].dtype == torch.uint8
        else SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MASKED
    )

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        # W4A16 masked has no SBO entry point, so there is nothing to forward.
        deep_gemm.m_grouped_gemm_w4a16_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
        )


def grouped_gemm_nt_bf16i4bf16_nopad(
    lhs: torch.Tensor,
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: Optional[torch.Tensor] = None,
):
    # lhs: shape [m, k], dtype bf16
    # VALU/INT4: int32 weights [e, k//16, n*2], scales [e, k//32, n].
    # Scales are uint8 E8M0 (VALU) or BF16 (INT4).
    # MMA: uint8 FP4 weights [e, n, k//2], uint8 tiled scales [e, n//64, k*2].

    assert _is_sail, (
        "grouped_gemm_nt_bf16i4bf16_nopad is only supported by SAIL DeepGEMM"
    )

    m, k = lhs.shape
    num_groups, n, _ = _get_w4a16_weight_shape(rhs[0])

    kernel_type = (
        SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD
        if rhs[0].dtype == torch.uint8
        else SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_NOPAD
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_w4a16_nopad(
            lhs,
            rhs,
            out,
            m_indices,
            m_rows,
        )


def _require_sail_fused_api(gemm_name: str) -> None:
    if not _is_sail:
        raise ValueError("gather-read fused MoE is only supported by SAIL DeepGEMM.")
    if not ENABLE_JIT_DEEPGEMM:
        raise RuntimeError("SAIL fused MoE requires DeepGEMM JIT to be enabled.")
    for name in ("moe_align_block_size", gemm_name):
        if not callable(getattr(deep_gemm, name, None)):
            raise RuntimeError(
                f"SAIL fused MoE is missing the DeepGEMM API {name}; please install "
                "a version that includes the fused MoE API refactor (53ea8ff)."
            )


def moe_align_block_size(
    A: torch.Tensor,
    B: torch.Tensor,
    topk_ids: torch.Tensor,
    perchannel_quant: bool = False,
):
    gemm_names = {
        torch.float8_e4m3fn: "m_grouped_gemm_fp8_fp8_bf16_nt_fused",
        torch.int8: "m_grouped_gemm_int8_int8_bf16_nt_fused",
        torch.bfloat16: "m_grouped_gemm_bf16_bf16_bf16_nt_fused",
        torch.uint8: "m_grouped_gemm_fp4_fp4_bf16_nt_fused",
    }
    if A.dtype not in gemm_names:
        raise ValueError(f"SAIL fused MoE does not support activation dtype {A.dtype}.")
    gemm_name = gemm_names[A.dtype]
    if A.dtype == torch.bfloat16 and B.dtype in (torch.int32, torch.uint8):
        gemm_name = "m_grouped_gemm_w4a16_fused"
    _require_sail_fused_api(gemm_name)
    return deep_gemm.moe_align_block_size(
        A, B, topk_ids, perchannel_quant=perchannel_quant
    )


def grouped_gemm_nt_bf16i4bf16_fused(
    lhs: torch.Tensor,
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_w4a16_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs.shape
    num_groups, n, _ = _get_w4a16_weight_shape(rhs[0])
    if rhs[0].dtype == torch.uint8:
        kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED
    elif rhs[1].dtype == torch.uint8:
        kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED
    else:
        kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_w4a16_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_bf16bf16bf16_fused(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_bf16_bf16_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs.shape
    num_groups, n, _ = rhs.shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_i8i8bf16_fused(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_int8_int8_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_f8f8bf16_fused(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
    perchannel_quant: bool = False,
):
    gemm_name = "m_grouped_gemm_fp8_fp8_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    # At K=128 the last dim of block scales is also 1, so it cannot be used to
    # infer channel mode.
    kernel_type = (
        SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL
        if perchannel_quant
        else SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED
    )
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_f4f4bf16_fused(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_fp4_fp4_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    if lhs[1].dtype != torch.uint16 or not lhs[1].is_contiguous():
        raise ValueError(
            "SAIL FP4 gather-read requires uint16 K-major contiguous activation scales."
        )
    # Consistent with the existing FP4 wrapper, warmup and tuning use the packed
    # byte width for K.
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )
