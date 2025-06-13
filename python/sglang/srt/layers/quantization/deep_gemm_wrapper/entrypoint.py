import torch
from typing import Tuple


# TODO from layer.py
# try:
#     from deep_gemm import fp8_m_grouped_gemm_nt_masked, m_grouped_fp8_gemm_nt_contiguous
#
#     print("hi layer.py use deep_gemm new version")
# except ImportError:
#     from deep_gemm import (
#         m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
#         m_grouped_gemm_fp8_fp8_bf16_nt_masked,
#     )
#
#     m_grouped_fp8_gemm_nt_contiguous = m_grouped_gemm_fp8_fp8_bf16_nt_contiguous
#     fp8_m_grouped_gemm_nt_masked = m_grouped_gemm_fp8_fp8_bf16_nt_masked
#     print("hi layer.py use deep_gemm old version")


def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    TODO_recipe_arg
    TODO

def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    TODO

def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    TODO

ENABLE_DEEPGEMM = TODO
