from contextlib import contextmanager
from typing import Tuple

import torch

from sglang.srt.server_args import ServerArgs

TODO_handle_no_deepgemm
try:
    from deep_gemm import (
        fp8_m_grouped_gemm_nt_masked as _grouped_gemm_nt_f8f8bf16_masked_raw,
        m_grouped_fp8_gemm_nt_contiguous as _grouped_gemm_nt_f8f8bf16_contig_raw,
        fp8_gemm_nt as _gemm_nt_f8f8bf16_raw,
    )

    DEEPGEMM_REQUIRE_UE8M0 = True
except ImportError:
    from deep_gemm import (
        m_grouped_gemm_fp8_fp8_bf16_nt_masked as _grouped_gemm_nt_f8f8bf16_masked_raw,
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous as _grouped_gemm_nt_f8f8bf16_contig_raw,
        gemm_fp8_fp8_bf16_nt as _gemm_nt_f8f8bf16_raw,
    )

    DEEPGEMM_REQUIRE_UE8M0 = False

ENABLE_JIT_DEEPGEMM = TODO


def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape

    kernel_type = DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED
    _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)

    with _log_jit_build(expected_m, n, k, kernel_type):
        _grouped_gemm_nt_f8f8bf16_masked_raw(
            lhs, rhs, out, masked_m, expected_m
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape

    kernel_type = DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG
    _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)

    with _log_jit_build(m, n, k, kernel_type):
        _grouped_gemm_nt_f8f8bf16_contig_raw(lhs, rhs, out, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape

    kernel_type = DeepGemmKernelType.GEMM_NT_F8F8BF16
    _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, 1)

    with _log_jit_build(m, n, k, kernel_type):
        _gemm_nt_f8f8bf16_raw(
            lhs,
            rhs,
            out,
            # NOTE HACK
            recipe=(1, 128, 128),
        )


# TODO improve?
def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    TODO


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)
