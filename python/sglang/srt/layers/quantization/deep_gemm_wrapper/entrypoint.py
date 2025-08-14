import logging
from contextlib import contextmanager
from typing import Tuple

import torch

from sglang.srt.layers.quantization.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.quantization.deep_gemm_wrapper.configurer import (
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
    )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(lhs, rhs, out, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    deep_gemm.fp8_gemm_nt(
        lhs,
        rhs,
        out,
    )


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    compile_utils.update_deep_gemm_config(gpu_id, server_args)


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
