import torch
import triton
import triton.language as tl
from typing import Tuple

from .tune_options import *
from .._C import get_mk_alignment_for_contiguous_layout


@triton.autotune(configs=get_k_grouped_gemm_configs(), key=[], restore_value=['d_ptr'])
@triton.jit
def a_fused_k_grouped_bf16_gemm_contiguous_tl_impl(a_ptr, b_ptr, d_ptr,
                                                   k_indices_ptr, k_start_ptr, k_end_ptr,
                                                   M: tl.constexpr,
                                                   N: tl.constexpr,
                                                   K,
                                                   ACC: tl.constexpr,
                                                   BLOCK_SIZE_M: tl.constexpr,
                                                   BLOCK_SIZE_N: tl.constexpr,
                                                   BLOCK_SIZE_K: tl.constexpr,
                                                   GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_b = (pid // (num_pid_m * num_pid_n)).to(tl.int64)
    pid = pid % (num_pid_m * num_pid_n)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_range = tl.max_contiguous(tl.multiple_of(m_range, BLOCK_SIZE_M), BLOCK_SIZE_M)
    n_range = tl.max_contiguous(tl.multiple_of(n_range, BLOCK_SIZE_N), BLOCK_SIZE_N)
    m_mask = (m_range < M)[:, None]
    n_mask = (n_range < N)[None, :]

    k_start = tl.load(k_start_ptr + pid_b)
    k_end = tl.load(k_end_ptr + pid_b)
    if k_start >= k_end:
        if not ACC:
            d_ptrs = d_ptr + pid_b * M * N + m_range[:, None].to(tl.int64) * N + n_range[None, :]
            tl.store(d_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=d_ptr.dtype.element_ty), mask=m_mask & n_mask)
        return

    # Compute
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        rows = tl.load(k_indices_ptr + k_range).to(tl.int64)
        a_ptrs = a_ptr + m_range[:, None] + rows[None, :] * M
        
        b_ptrs = b_ptr + k_range[:, None].to(tl.int64) * N + n_range[None, :]
        a = tl.load(a_ptrs, mask=(rows >= 0)[None, :] & m_mask, other=0)
        b = tl.load(b_ptrs, mask=n_mask, other=0)
        acc = tl.dot(a, b, acc)

    # Write back
    d_ptrs = d_ptr + pid_b * M * N + m_range[:, None].to(tl.int64) * N + n_range[None, :]
    if ACC:
        acc += tl.load(d_ptrs, mask=m_mask & n_mask)
    acc = acc.to(d_ptr.dtype.element_ty)
    tl.store(d_ptrs, acc, mask=m_mask & n_mask)


def a_fused_k_grouped_bf16_gemm_tn_contiguous_tl(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                                                 handle: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], acc: bool):
    k_indices, k_start, k_end = handle

    assert a.is_contiguous() and b.is_contiguous() and d.is_contiguous()
    assert k_indices.is_contiguous() and k_start.is_contiguous() and k_end.is_contiguous()
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert k_indices.dtype == torch.int32 and k_start.dtype == torch.int32 and k_end.dtype == torch.int32
    assert a.dim() == 2 and b.dim() == 2 and d.dim() == 3
    assert k_start.numel() == k_end.numel() and k_indices.size(0) == b.size(0)
    assert d.size(0) == k_start.numel() and d.size(1) == a.size(1) and d.size(2) == b.size(1)
    assert b.size(0) % get_mk_alignment_for_contiguous_layout() == 0

    K_, M = a.shape
    K, N = b.shape
    B = k_start.numel()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']) * B,)
    a_fused_k_grouped_bf16_gemm_contiguous_tl_impl[grid](
        a, b, d, k_indices, k_start, k_end, M, N, K, ACC=acc)
