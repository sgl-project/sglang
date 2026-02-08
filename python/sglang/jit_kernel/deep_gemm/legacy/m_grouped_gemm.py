import torch
import triton
import triton.language as tl
from typing import Tuple

from .tune_options import *
from .._C import get_mk_alignment_for_contiguous_layout


@triton.autotune(configs=get_m_grouped_gemm_configs(), key=[])
@triton.jit
def m_grouped_bf16_gemm_contiguous_tl_impl(a_ptr, b_ptr, d_ptr,
                                           m_indices_ptr,
                                           M,
                                           N: tl.constexpr,
                                           K: tl.constexpr,
                                           BLOCK_SIZE_M: tl.constexpr,
                                           BLOCK_SIZE_N: tl.constexpr,
                                           BLOCK_SIZE_K: tl.constexpr,
                                           GROUP_SIZE_M: tl.constexpr,
                                           IS_B_K_MAJOR: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = (n_range < N)[None, :]

    # Empty tokens
    batch_id = tl.load(m_indices_ptr + pid_m * BLOCK_SIZE_M).to(tl.int64)
    if batch_id < 0:
        d_ptrs = d_ptr + m_range[:, None].to(tl.int64) * N + n_range[None, :]
        tl.store(d_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=d_ptr.dtype.element_ty), mask=n_mask)
        return

    # Compute
    a_ptrs = a_ptr + m_range[:, None].to(tl.int64) * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    b_ptrs = b_ptr + batch_id * K * N + \
             tl.arange(0, BLOCK_SIZE_K)[:, None].to(tl.int64) * (1 if IS_B_K_MAJOR else N) + \
             n_range[None, :].to(tl.int64) * (K if IS_B_K_MAJOR else 1)
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = (k + tl.arange(0, BLOCK_SIZE_K)) < K
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * (1 if IS_B_K_MAJOR else N)

    # Write back
    d_ptrs = d_ptr + m_range[:, None].to(tl.int64) * N + n_range[None, :]
    tl.store(d_ptrs, accumulator.to(d_ptr.dtype.element_ty), mask=n_mask)


def m_grouped_bf16_gemm_nt_contiguous_tl(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                                         m_indices: torch.Tensor):
    r0, r1, r2 = b.shape

    assert a.is_contiguous() and (b.is_contiguous or b.mT.is_contiguous())
    assert m_indices.is_contiguous() and d.is_contiguous()
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32 and d.dtype == torch.bfloat16
    assert a.dim() == 2 and b.dim() == 3 and d.dim() == 2
    assert a.size(1) == r2 and a.size(0) == d.size(0) and r1 == d.size(1)
    assert m_indices.numel() == a.size(0)
    assert a.size(0) % get_mk_alignment_for_contiguous_layout() == 0
    M, K = a.shape
    B, N, K_ = r0, r1, r2

    # For Triton 2.0, persistent kernel will lead to errors
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    m_grouped_bf16_gemm_contiguous_tl_impl[grid](
        a, b, d, m_indices, M, N, K, IS_B_K_MAJOR=b.is_contiguous())


def m_grouped_bf16_gemm_nn_contiguous_tl(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                                         m_indices: torch.Tensor):
    m_grouped_bf16_gemm_nt_contiguous_tl(a, b.mT, d, m_indices)
