import logging
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.asym_gemm_wrapper import compile_utils
from sglang.srt.layers.asym_gemm_wrapper.configurer import (  # noqa: F401
    ASYMGEMM_BLACKWELL,
    ASYMGEMM_SCALE_UE8M0,
    ENABLE_JIT_ASYMGEMM,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

if ENABLE_JIT_ASYMGEMM:
    import asym_gemm
    from asym_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = get_bool_env_var("SGLANG_ASYMGEMM_SANITY_CHECK")
 

def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.asym_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_asym_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            return asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                disable_ue8m0_cast=False,
            )
        
def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    list_size: torch.Tensor,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.asym_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(lhs, rhs, out, offsets, experts, list_size,
            disable_ue8m0_cast=False,
        )


# NVFP4 uses per-row scales for activations and per-row scales for weights
# (granularity_mn=1, granularity_k=16). Scales are stored as E4M3 payload bytes;
# the kernel internally re-packs them into the TMA-aligned uint32 layout required
# by block-scaled NVFP4 MMA.
_FP4_RECIPE = (1, 1, 16)


def grouped_gemm_nt_fp4fp4bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe: Optional[Tuple[int, int, int]] = None,
):
    num_groups, _, k_packed = lhs[0].shape
    _, n, _ = rhs[0].shape
    k = k_packed * 2
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_FP4_MASKED

    with compile_utils.asym_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_asym_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            return asym_gemm.m_grouped_fp4_asym_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                recipe=recipe if recipe is not None else _FP4_RECIPE,
                disable_ue8m0_cast=True,
            )


def grouped_gemm_nt_fp4fp4bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    list_size: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
):

    m, k_packed = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    k = k_packed * 2
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_FP4_CONTIG

    with compile_utils.asym_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        asym_gemm.m_grouped_fp4_asym_gemm_nt_contiguous(
            lhs,
            rhs,
            out,
            offsets,
            experts,
            list_size,
            recipe=recipe if recipe is not None else _FP4_RECIPE,
            disable_ue8m0_cast=True,
        )


def grouped_gemm_nt_bf16bf16bf16_masked(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    num_groups: int,
):
    num_groups, m_max, k = lhs.shape
    _, n, _ = rhs.shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    with compile_utils.asym_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            compiled_dims="nk",
        )


def grouped_gemm_nt_bf16bf16bf16_contig(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    list_size: torch.Tensor,
):
    m, k = lhs.shape
    num_groups, n, _ = rhs.shape
    kernel_type = compile_utils.AsymGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG

    with compile_utils.asym_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            lhs, rhs, out, offsets, experts, list_size
        )


def update_asym_gemm_config(gpu_id: int, server_args: ServerArgs):
    compile_utils.update_asym_gemm_config(gpu_id, server_args)


@contextmanager
def configure_asym_gemm_num_sms(num_sms):
    if num_sms is None:
        yield
    else:
        original_num_sms = asym_gemm.get_num_sms()
        asym_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            asym_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
