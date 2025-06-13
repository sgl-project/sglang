import logging
from contextlib import contextmanager
from typing import Tuple

import torch

from sglang.srt.layers.quantization.deep_gemm_wrapper import compile_utils
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var, get_device_sm

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        logger.warning("Failed to import deep_gemm, disable ENABLE_JIT_DEEPGEMM.")
        return False

    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    return get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true")


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

    try:
        from deep_gemm import fp8_gemm_nt as _gemm_nt_f8f8bf16_raw
        from deep_gemm import (
            fp8_m_grouped_gemm_nt_masked as _grouped_gemm_nt_f8f8bf16_masked_raw,
        )
        from deep_gemm import (
            m_grouped_fp8_gemm_nt_contiguous as _grouped_gemm_nt_f8f8bf16_contig_raw,
        )

        DEEPGEMM_REQUIRE_UE8M0 = True
    except ImportError:
        from deep_gemm import gemm_fp8_fp8_bf16_nt as _gemm_nt_f8f8bf16_raw
        from deep_gemm import (
            m_grouped_gemm_fp8_fp8_bf16_nt_contiguous as _grouped_gemm_nt_f8f8bf16_contig_raw,
        )
        from deep_gemm import (
            m_grouped_gemm_fp8_fp8_bf16_nt_masked as _grouped_gemm_nt_f8f8bf16_masked_raw,
        )

        DEEPGEMM_REQUIRE_UE8M0 = False


def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        _grouped_gemm_nt_f8f8bf16_masked_raw(lhs, rhs, out, masked_m, expected_m)


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        _grouped_gemm_nt_f8f8bf16_contig_raw(lhs, rhs, out, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        _gemm_nt_f8f8bf16_raw(
            lhs,
            rhs,
            out,
            recipe=TODO_handle_recipe,
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
