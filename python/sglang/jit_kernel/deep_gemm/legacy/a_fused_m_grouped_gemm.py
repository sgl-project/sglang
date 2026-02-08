import torch
import triton
import triton.language as tl
from typing import Tuple

from .tune_options import *
from .._C import get_mk_alignment_for_contiguous_layout


@triton.autotune(configs=get_m_grouped_gemm_configs(), key=[])
@triton.jit
def a_fused_m_grouped_bf16_gemm_contiguous_tl_impl(a_ptr, b_ptr, d_ptr,
                                                   m_indices_ptr, m_row_indices_ptr,
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

    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_range = tl.max_contiguous(tl.multiple_of(m_range, BLOCK_SIZE_M), BLOCK_SIZE_M)
    n_range = tl.max_contiguous(tl.multiple_of(n_range, BLOCK_SIZE_N), BLOCK_SIZE_N)
    n_mask = (n_range < N)[None, :]

    batch_id = tl.load(m_indices_ptr + pid_m * BLOCK_SIZE_M).to(tl.int64)
    if batch_id < 0:
        d_ptrs = d_ptr + m_range[:, None].to(tl.int64) * N + n_range[None, :]
        tl.store(d_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=d_ptr.dtype.element_ty), mask=n_mask)
        return

    # b block
    rows = tl.load(m_row_indices_ptr + m_range).to(tl.int64)

    # Compute
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_range = (k + tl.arange(0, BLOCK_SIZE_K)).to(tl.int64)
        k_mask = k_range < K
        a_ptrs = a_ptr + rows[:, None] * K + k_range[None, :]
        b_ptrs = b_ptr + batch_id * K * N + k_range[:, None] * (1 if IS_B_K_MAJOR else N) + n_range[None, :].to(tl.int64) * (K if IS_B_K_MAJOR else 1)
        a = tl.load(a_ptrs, mask=(rows >= 0)[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
        acc = tl.dot(a, b, acc)
    d = acc.to(d_ptr.dtype.element_ty)

    # Write back
    d_ptrs = d_ptr + m_range[:, None].to(tl.int64) * N + n_range[None, :]
    tl.store(d_ptrs, d, mask=n_mask)


def a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                                                 mappings: Tuple[torch.Tensor, torch.Tensor]):
    m_indices, m_row_indices = mappings
    r0, r1, r2 = b.shape

    assert a.is_contiguous() and (b.is_contiguous() or b.mT.is_contiguous()) and d.is_contiguous()
    assert m_indices.is_contiguous() and m_row_indices.is_contiguous()
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and d.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32 and m_row_indices.dtype == torch.int32
    assert a.dim() == 2 and b.dim() == 3 and d.dim() == 2
    assert a.size(1) == r2 and d.size(0) == m_indices.numel() and d.size(1) == r1
    assert m_indices.numel() == m_row_indices.numel()
    assert m_indices.numel() % get_mk_alignment_for_contiguous_layout() == 0

    if d.size(0) == 0:
        return d

    M_, K = a.shape
    B, K, N = r0, r2, r1
    M = m_indices.numel()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    a_fused_m_grouped_bf16_gemm_contiguous_tl_impl[grid](a, b, d, m_indices, m_row_indices,
                                                         M, N, K, IS_B_K_MAJOR=b.is_contiguous())


def a_fused_m_grouped_bf16_gemm_nn_contiguous_tl(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                                                   mappings: Tuple[torch.Tensor, torch.Tensor]):
    a_fused_m_grouped_bf16_gemm_nt_contiguous_tl(a, b.mT, d, mappings)
