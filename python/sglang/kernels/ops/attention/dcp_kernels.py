# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Triton kernels for decode context parallel (DCP).

Consolidated from the two merged DCP implementations:
  - create_triton_kv_indices_for_dcp_triton  (PR #25090, Triton/MHA path)
  - create_dcp_kv_indices / update_kv_lens_and_indices  (PR #14194, MLA path)
  - _correct_attn_cp_out_kernel / correct_attn_out / CPTritonContext  (PR #14194)
"""

from typing import Optional, Tuple

import torch
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    _blockwise_barrier,
    _local_ld_128,
    _multimem_st_128,
    _sync_threads,
)


# ---------------------------------------------------------------------------
# KV-index build (PR #25090, Triton/MHA): per-rank local KV indices.
# ---------------------------------------------------------------------------
@triton.jit
def create_triton_kv_indices_for_dcp_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    dcp_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)

    # First absolute token position in this range owned by dcp_rank.
    # Triton follows C-style remainder for negative values, so avoid
    # computing the offset as a negative remainder when kv_start > dcp_rank.
    kv_start_mod = kv_start % dcp_size
    first = kv_start + ((dcp_rank + dcp_size - kv_start_mod) % dcp_size)
    local_len = tl.load(dcp_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(local_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < local_len
        abs_pos = first + offset * dcp_size
        data = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + abs_pos,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + kv_indices_offset + offset, data // dcp_size, mask=mask
        )


# ---------------------------------------------------------------------------
# KV-index build (PR #14194, MLA): global prefix+extend layout for the
# all-gathered dcp_kv_buffer, plus the per-rank shard/compact kernel.
# ---------------------------------------------------------------------------
@triton.jit
def create_mla_kv_page_table_for_dcp(
    req_to_token_ptr,
    req_pool_indices_ptr,
    local_seq_lens_ptr,
    block_kv_indices_ptr,
    v2p_ptr,  # in: [num_pages + 1] int64 -- virtual->physical page table
    req_to_token_stride: tl.constexpr,
    block_table_stride: tl.constexpr,
    PHYSICAL_PAGE_SIZE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    HAS_V2P: tl.constexpr,
):
    """This rank's cyclic slice of each request, as a page table.

    ``HAS_V2P`` picks the id space the emitted page number is in: the
    DCP-collapsed page IS physical on a static pool, and still VIRTUAL under
    the unified memory pool, where it takes one more gather through ``v2p_ptr``
    to the physical page the per-layer views are indexed by.
    """
    req = tl.program_id(0)
    page_block = tl.program_id(1)
    page_offsets = page_block * PAGES_PER_BLOCK + tl.arange(0, PAGES_PER_BLOCK)
    local_len = tl.load(local_seq_lens_ptr + req)
    local_pages = tl.cdiv(local_len, PHYSICAL_PAGE_SIZE)
    mask = page_offsets < local_pages
    global_positions = DCP_RANK + page_offsets * PHYSICAL_PAGE_SIZE * DCP_SIZE
    req_pool_index = tl.load(req_pool_indices_ptr + req)
    virtual_locs = tl.load(
        req_to_token_ptr + req_pool_index * req_to_token_stride + global_positions,
        mask=mask,
        other=0,
    )
    pages = virtual_locs // DCP_SIZE // PHYSICAL_PAGE_SIZE
    if HAS_V2P:
        # A `-1` in req_to_token and a freed (`-1`) v2p row both clamp to entry
        # 0, the reserved padding page.
        pages = tl.where(virtual_locs < 0, 0, pages)
        physical = tl.load(v2p_ptr + pages, mask=mask, other=0)
        pages = tl.maximum(physical, 0)
    tl.store(
        block_kv_indices_ptr + req * block_table_stride + page_offsets,
        pages.to(tl.int32),
        mask=mask,
    )


@triton.jit
def compact_dcp_verify_token_table_to_ragged(
    token_table_ptr,  # in: [bs * q_len, table_stride] from create_mla_kv_page_table_for_dcp
    local_kv_lens_ptr,  # in: [bs] this rank's shard length per request, contiguous
    kv_indptr_ptr,  # in: [bs + 1] cumsum of the lengths, each clamped to >= 1
    kv_indices_ptr,  # out: ragged ids, one run per request
    table_stride: tl.constexpr,
    Q_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Rectangular token table -> ragged (kv_indices, kv_indptr).

    Every row of a request repeats that request's shard, so this reads row 0.
    """
    # The grid is (bs,) and the token loop is inside the kernel, so a cuda
    # graph can capture this launch.
    req = tl.program_id(0)
    out_offset = tl.load(kv_indptr_ptr + req)
    local_len = tl.load(local_kv_lens_ptr + req).to(tl.int32)
    # The table must be wide enough for the whole shard. A narrow table makes
    # this kernel read the next query row and return the wrong ids.
    tl.device_assert(local_len <= table_stride)

    if local_len == 0:
        # Give an empty shard one slot, because the attention kernel fails
        # on a shard of length zero. The caller removes that row later.
        tl.store(kv_indices_ptr + out_offset, 0)
    else:
        row = token_table_ptr + req * Q_LEN * table_stride
        for i in range(tl.cdiv(local_len, BLOCK_SIZE)):
            offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
            mask = offset < local_len
            tl.store(
                kv_indices_ptr + out_offset + offset,
                tl.load(row + offset, mask=mask),
                mask=mask,
            )


@triton.jit
def create_dcp_kv_indices(
    kv_indptr,
    extend_lens_ptr,
    extend_cu_lens_ptr,
    extend_prefix_lens_ptr,
    extend_cu_prefix_lens_ptr,
    kv_indices_ptr,
    extend_prefix_lens_sum,
    dcp_world_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    prefix_len = tl.load(extend_prefix_lens_ptr + pid)
    prefix_start = tl.load(extend_cu_prefix_lens_ptr + pid)
    kv_ind_start = tl.load(kv_indptr + pid)
    num_loop = tl.cdiv(prefix_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < prefix_len
        data = prefix_start + offset
        tl.store(kv_indices_ptr + kv_ind_start + offset, data, mask=mask)
    extend_len = tl.load(extend_lens_ptr + pid)
    extend_start = tl.load(extend_cu_lens_ptr + pid)
    num_loop = tl.cdiv(extend_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < extend_len
        data = extend_prefix_lens_sum + extend_start + offset
        tl.store(
            kv_indices_ptr + kv_ind_start + prefix_len + offset,
            data,
            mask=mask,
        )


@triton.jit
def update_kv_lens_and_indices(
    kv_lens: torch.Tensor,
    kv_lens_cumsum: torch.Tensor,
    kv_indices: torch.Tensor,
    local_kv_lens: torch.Tensor,
    local_kv_lens_cumsum: torch.Tensor,
    local_kv_indices: torch.Tensor,
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bs_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    local_kv_len = tl.load(local_kv_lens + bs_idx)
    local_kv_indices_start = tl.load(local_kv_lens_cumsum + bs_idx)
    kv_indices_start = tl.load(kv_lens_cumsum + bs_idx)

    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < local_kv_len

    kv_indice_offsets = offsets * dcp_world_size + dcp_rank + kv_indices_start
    local_kv_indices_offsets = local_kv_indices_start + offsets

    kv_values = tl.load(kv_indices + kv_indice_offsets, mask=mask)
    tl.store(local_kv_indices + local_kv_indices_offsets, kv_values, mask=mask)


# ---------------------------------------------------------------------------
# Partial-attention LSE correction (PR #14194, MLA path).
# ---------------------------------------------------------------------------
@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    new_outputs_stride_H,
    new_outputs_stride_B,
    new_outputs_stride_D,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_LSE_BASE_ON_E: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ H, B, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)

    # Use int32 for offsets where possible to reduce register pressure
    b_i32 = batch_idx.to(tl.int32)
    h_i32 = head_idx.to(tl.int32)

    # Vectorized load of LSE values: shape = [N]
    num_n_offsets = tl.arange(0, N_ROUNDED)
    lse_offsets = (
        num_n_offsets * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    )

    # Compute final LSE using online softmax algorithm (more numerically stable)
    lse = tl.load(lses_ptr + lse_offsets)

    # Replace NaN and inf with -inf for numerical stability
    neg_inf = float("-inf")
    lse = tl.where((lse != lse) | (lse == float("inf")), neg_inf, lse)

    # Online softmax: find max, subtract, exp, sum, log
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == neg_inf, 0.0, lse_max)
    lse = lse - lse_max
    lse_exp = tl.exp(lse) if IS_LSE_BASE_ON_E else tl.exp2(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    final_lse = (tl.log(lse_acc) if IS_LSE_BASE_ON_E else tl.log2(lse_acc)) + lse_max

    # Compute correction factor
    lse_offset = lse_idx * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    local_lse = tl.load(lses_ptr + lse_offset)
    lse_diff = local_lse - final_lse
    lse_diff = tl.where(
        (lse_diff != lse_diff) | (lse_diff == float("inf")),
        neg_inf,
        lse_diff,
    )
    factor = tl.exp(lse_diff) if IS_LSE_BASE_ON_E else tl.exp2(lse_diff)

    # Store final LSE
    tl.store(vlse_ptr + b_i32 * lses_stride_B + h_i32 * lses_stride_H, final_lse)

    # Load output with vectorized access: shape = [D]
    d_offsets = tl.arange(0, HEAD_DIM)
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    new_output_offsets = (
        head_idx * new_outputs_stride_H
        + batch_idx * new_outputs_stride_B
        + d_offsets * new_outputs_stride_D
    )
    # Apply correction and store
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + new_output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: Optional[CPTritonContext],
    new_output: torch.Tensor = None,
    is_lse_base_on_e: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    no_sH, no_sB, no_sD = new_output.stride()
    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        new_output,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        no_sH,
        no_sB,
        no_sD,
        cp_rank,
    )
    const_args = {
        "HEAD_DIM": D,
        "N_ROUNDED": N,
        "IS_LSE_BASE_ON_E": is_lse_base_on_e,
    }

    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return new_output, lse


# A2A DCP reduce: LSE-weighted combine of N partial attention outputs
# (used by the a2a / fi_a2a communication backends, see comm.py).


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    """Number of output-dtype elements needed to store one fp32 LSE value."""
    return torch.finfo(torch.float32).bits // torch.finfo(output_dtype).bits


@triton.jit
def _dcp_pack_a2a_send_kernel(
    out_ptr,
    lse_ptr,
    dst_o_ptr,
    dst_lse_ptr,
    out_stride_B,
    out_stride_H,
    lse_stride_B,
    lse_stride_H,
    dst_o_stride_N,
    dst_o_stride_B,
    dst_o_stride_H,
    dst_lse_stride_N,
    dst_lse_stride_B,
    dst_lse_stride_H,
    H_PER_RANK: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter one (batch, head) partial into its peer's send slot."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    peer = h // H_PER_RANK
    h_local = h % H_PER_RANK

    src = out_ptr + b * out_stride_B + h * out_stride_H
    dst = (
        dst_o_ptr
        + peer * dst_o_stride_N
        + b * dst_o_stride_B
        + h_local * dst_o_stride_H
    )

    for start in tl.range(0, WORDS, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < WORDS
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    tl.store(
        dst_lse_ptr
        + peer * dst_lse_stride_N
        + b * dst_lse_stride_B
        + h_local * dst_lse_stride_H,
        tl.load(lse_ptr + b * lse_stride_B + h * lse_stride_H),
    )


def dcp_pack_a2a_send(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    dst_o: torch.Tensor,
    dst_lse: torch.Tensor,
) -> None:
    """Scatter ``[B, H, D]`` partials + ``[B, H]`` LSE into a transport's send slots.

    ``dst_o`` is ``[N, B_max, H // N, D]`` and ``dst_lse`` ``[N, B_max, H // N]``,
    in any stride order. Rows beyond ``B`` are left untouched.
    """
    B, H, D = cp_attn_out.shape
    N, B_max, H_per_rank = dst_lse.shape
    lpd = _lse_pack_dim(cp_attn_out.dtype)

    if cp_attn_lse.dtype != torch.float32 or dst_lse.dtype != torch.float32:
        raise ValueError("LSE tensors must be float32")
    if D % lpd:
        raise ValueError(f"head dim {D} must be a multiple of the LSE pack dim {lpd}")
    if dst_o.shape != (N, B_max, H_per_rank, D) or H_per_rank * N != H or B > B_max:
        raise ValueError(
            f"destination {tuple(dst_o.shape)} / {tuple(dst_lse.shape)} does not "
            f"match out {tuple(cp_attn_out.shape)}"
        )

    out_words = cp_attn_out.view(torch.float32)
    dst_o_words = dst_o.view(torch.float32)
    words = D // lpd

    _dcp_pack_a2a_send_kernel[(B, H)](
        out_words,
        cp_attn_lse,
        dst_o_words,
        dst_lse,
        out_words.stride(0),
        out_words.stride(1),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        dst_o_words.stride(0),
        dst_o_words.stride(1),
        dst_o_words.stride(2),
        dst_lse.stride(0),
        dst_lse.stride(1),
        dst_lse.stride(2),
        H_PER_RANK=H_per_rank,
        WORDS=words,
        BLOCK=min(1024, triton.next_power_of_2(words)),
    )


@triton.jit
def _dcp_pack_q_kernel(
    q_nope_ptr,
    q_pe_ptr,
    out_ptr,
    nope_stride_B,
    nope_stride_H,
    pe_stride_B,
    pe_stride_H,
    out_stride_H,
    out_stride_B,
    D_NOPE: tl.constexpr,
    D_PE: tl.constexpr,
):
    """One (batch, head) row of ``[q_pe | q_nope]``, cast to the output dtype."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    out_ty = out_ptr.dtype.element_ty
    dst = out_ptr + h * out_stride_H + b * out_stride_B
    pe = tl.arange(0, D_PE)
    q_pe = tl.load(q_pe_ptr + b * pe_stride_B + h * pe_stride_H + pe)
    tl.store(dst + pe, q_pe.to(out_ty))
    nope = tl.arange(0, D_NOPE)
    q_nope = tl.load(q_nope_ptr + b * nope_stride_B + h * nope_stride_H + nope)
    tl.store(dst + D_PE + nope, q_nope.to(out_ty))


def dcp_pack_q(q_nope: torch.Tensor, q_pe: torch.Tensor, out: torch.Tensor) -> None:
    """Pack ``[B, H, D_nope]`` and ``[B, H, D_pe]`` into ``out`` ``[H, B, D_pe + D_nope]``.

    The cast is Triton's, the same one the sparse-MLA kernels apply when they
    load a bf16 q against an FP8 KV cache, so an FP8 ``out`` matches their
    in-kernel conversion bit for bit.
    """
    B, H, d_nope = q_nope.shape
    d_pe = q_pe.shape[-1]
    if q_nope.stride(-1) != 1 or q_pe.stride(-1) != 1:
        raise ValueError("q_nope and q_pe need a unit-stride last dim")
    if out.shape != (H, B, d_pe + d_nope) or out.stride(-1) != 1:
        raise ValueError(f"out {tuple(out.shape)} does not match q {(B, H)}")
    if B == 0:
        return
    _dcp_pack_q_kernel[(B, H)](
        q_nope,
        q_pe,
        out,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        out.stride(0),
        out.stride(1),
        D_NOPE=d_nope,
        D_PE=d_pe,
    )


@triton.jit
def _dcp_lse_combine_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    recv_output_stride_N,
    recv_output_stride_B,
    recv_output_stride_H,
    recv_output_stride_D,
    recv_lse_stride_N,
    recv_lse_stride_B,
    recv_lse_stride_H,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Combine N partial attention outputs weighted by their LSE values.

    Grid: (B, H_local).
    Each program handles one (batch, head) position across all N shards.

    Two-pass approach:
    Pass 1: find max LSE and weight sum across shards
    Pass 2: accumulate weighted outputs
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_base = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H

    # Pass 1: find max LSE across N shards
    lse_max = tl.load(recv_lse_ptr + lse_base).to(tl.float32)
    lse_max = tl.where(
        (lse_max != lse_max) | (lse_max == float("inf")), -float("inf"), lse_max
    )
    for i in tl.static_range(1, N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        lse_max = tl.where(lse_i > lse_max, lse_i, lse_max)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Pass 2: accumulate weighted outputs
    weight_sum = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in tl.static_range(N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        centered = lse_i - lse_max
        if IS_BASE_E:
            w = tl.exp(centered)
        else:
            w = tl.exp2(centered)
        weight_sum += w

        o_offsets = (
            i * recv_output_stride_N
            + batch_idx * recv_output_stride_B
            + head_idx * recv_output_stride_H
            + d_offsets * recv_output_stride_D
        )
        partial_out = tl.load(recv_output_ptr + o_offsets).to(tl.float32)
        acc += partial_out * w

    acc = acc / tl.maximum(weight_sum, 1.0)

    out_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty))

    if RETURN_LSE:
        if IS_BASE_E:
            global_lse = tl.log(weight_sum) + lse_max
        else:
            global_lse = tl.log2(weight_sum) + lse_max
        out_lse_offset = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    is_lse_base_on_e: bool = True,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the Triton LSE-combine kernel.

    Args:
        recv_output: [N, B, H_local, D] partial outputs from each DCP rank.
        recv_lse:    [N, B, H_local]    log-sum-exp from each DCP rank.
        is_lse_base_on_e: True if LSE uses base-e (FlashAttention),
                          False if base-2 (FlashInfer).
        return_lse: If True, also return the combined global LSE.

    Returns:
        (combined_output [B, H_local, D], combined_lse [B, H_local] or None)
    """
    N, B, H_local, D = recv_output.shape
    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )
    out_lse = (
        torch.empty((B, H_local), device=recv_lse.device, dtype=recv_lse.dtype)
        if return_lse
        else recv_lse.new_empty(0)
    )

    grid = (B, H_local)
    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        recv_output.stride(0),
        recv_output.stride(1),
        recv_output.stride(2),
        recv_output.stride(3),
        recv_lse.stride(0),
        recv_lse.stride(1),
        recv_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )
    return out, (out_lse if return_lse else None)


@triton.jit
def _peer_store_128(ptr, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """{
            .reg .pred p;
            setp.ne.s32 p, $6, 0;
            @p st.relaxed.sys.global.v4.b32 [$1], {$2, $3, $4, $5};
        }""",
        "=r,l,r,r,r,r,r",
        [ptr, x, y, z, w, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _dcp_pack_exchange_kernel(
    src,
    lse,
    peer_ptrs,
    signals,
    B: tl.constexpr,
    SRC_B: tl.constexpr,
    SRC_H: tl.constexpr,
    LSE_B: tl.constexpr,
    LSE_H: tl.constexpr,
    MAX_B: tl.constexpr,
    RANK: tl.constexpr,
    SRC_ALIGNED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Head-sharded BF16 partials; four extra words preserve peer-store alignment.
    CHUNKS: tl.constexpr = HEAD_DIM // 8
    WORDS: tl.constexpr = HEAD_DIM // 2
    PITCH: tl.constexpr = WORDS + 4
    _blockwise_barrier(signals, RANK, 8, sem="relaxed")
    _sync_threads()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for start in range(0, tl.cdiv(B * 96 * CHUNKS, tl.num_programs(0) * BLOCK)):
        index = offsets + start * tl.num_programs(0) * BLOCK
        valid = index < B * 96 * CHUNKS
        chunk = index % CHUNKS
        head = index // CHUNKS % 96
        row = index // (96 * CHUNKS)
        peer = head // 12
        head_local = head % 12
        source = src + row.to(tl.int64) * SRC_B + head * SRC_H + chunk * 4
        address = tl.load(peer_ptrs + peer).to(tl.pointer_type(tl.int32))
        base = ((RANK * MAX_B + row.to(tl.int64)) * 12 + head_local) * PITCH
        if SRC_ALIGNED:
            x, y, z, w = _local_ld_128(source, valid)
        else:
            x = tl.load(source, valid, other=0)
            y = tl.load(source + 1, valid, other=0)
            z = tl.load(source + 2, valid, other=0)
            w = tl.load(source + 3, valid, other=0)
        _peer_store_128(address + base + chunk * 4, x, y, z, w, valid)
        copy_lse = valid & (chunk == 0)
        lse_value = tl.load(lse + row * LSE_B + head * LSE_H, copy_lse, other=0)
        tl.store(address + base + WORDS, lse_value.to(tl.int32, bitcast=True), copy_lse)
    _sync_threads()
    _blockwise_barrier(signals, RANK, 8, sem="acq_rel")


class DcpDirectAllToAll:
    """Persistent receive storage; unsupported shapes use FlashInfer's exchange."""

    _MAX_ROWS = 1024

    def __init__(self, group):
        self.group = group
        self.enabled = (
            group.world_size == 8
            and torch.cuda.get_device_name(group.device) == "NVIDIA B300 SXM6 AC"
        )
        self.buffers = {}

    def __call__(self, output: torch.Tensor, lse: torch.Tensor):
        if not (
            self.enabled
            and output.dtype == torch.bfloat16
            and output.ndim == 3
            and output.shape[1] == 96
            and output.shape[2] in (128, 512)
            and output.stride(2) == 1
            and output.stride(0) % 2 == 0
            and output.stride(1) % 2 == 0
            and lse.dtype == torch.float32
            and lse.shape == output.shape[:2]
            and 0 < output.shape[0] <= self._MAX_ROWS
        ):
            return None
        dim = output.shape[2]
        storage = self.buffers.get(dim)
        if storage is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            # Build during eager warmup, even at a smaller batch, so the verify
            # graph can reuse it. Pad each head to keep 16-byte peer stores aligned.
            symm_mem.set_signal_pad_size(
                max(symm_mem.get_signal_pad_size(), 128 * 8 * 4)
            )
            with torch.inference_mode(False):
                buffer = symm_mem.empty(
                    (8, self._MAX_ROWS, 12, dim // 2 + 4),
                    dtype=torch.int32,
                    device=output.device,
                )
            handle = symm_mem.rendezvous(buffer, self.group.device_group)
            assert handle.rank == self.group.rank_in_group
            peers = torch.tensor(
                handle.buffer_ptrs, dtype=torch.int64, device=output.device
            )
            storage = self.buffers[dim] = (buffer, handle, peers)
        buffer, handle, peers = storage
        rows = output.shape[0]
        if rows < 512:
            return None
        words = output.view(torch.int32)
        _dcp_pack_exchange_kernel[(128,)](
            words,
            lse,
            peers,
            handle.signal_pad_ptrs_dev,
            rows,
            words.stride(0),
            words.stride(1),
            lse.stride(0),
            lse.stride(1),
            self._MAX_ROWS,
            handle.rank,
            words.data_ptr() % 16 == 0
            and words.stride(0) % 4 == 0
            and words.stride(1) % 4 == 0,
            dim,
            1024,
            num_warps=32,
        )
        # Borrowed until the next exchange; the caller immediately combines
        # the partials into an owned result with the existing LSE kernel.
        return (
            buffer.view(torch.bfloat16)[:, :rows, :, :dim],
            buffer.view(torch.float32)[:, :rows, :, dim // 2],
        )


@triton.jit
def _gather_candidate_prefix_kernel(
    src,
    lengths,
    multicast_ptr,
    signal_ptrs,
    rows,
    TOPK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPACT_PREFIX: tl.constexpr = False,
):
    _blockwise_barrier(signal_ptrs, RANK, WORLD, sem="relaxed")
    _sync_threads()
    lane = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        length = tl.load(lengths + row)
        n = tl.minimum(length, TOPK)
        dst = multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
        base = (RANK * tl.cast(rows, tl.int64) + row) * TOPK * 2
        if COMPACT_PREFIX and length <= TOPK:
            # Complete prefixes are emitted in local position order.
            # The receiver reconstructs positions as column * WORLD + RANK.
            for start in range(0, tl.cdiv(n, 4), BLOCK):
                quad = start + lane
                mask = quad * 4 < n
                offset = (row.to(tl.int64) * TOPK + quad * 4) * 2
                s0, _, s1, _ = _local_ld_128(src + offset, mask)
                s2, _, s3, _ = _local_ld_128(src + offset + 4, mask)
                _multimem_st_128(dst + base + quad * 4, s0, s1, s2, s3, mask)
        else:
            # Each store holds two pairs; the consumer masks an odd tail.
            for start in range(0, tl.cdiv(n, 2), BLOCK):
                pair = start + lane
                mask = pair * 2 < n
                offset = (row.to(tl.int64) * TOPK + pair * 2) * 2
                v0, v1, v2, v3 = _local_ld_128(src + offset, mask)
                _multimem_st_128(dst + base + pair * 4, v0, v1, v2, v3, mask)
    _sync_threads()
    _blockwise_barrier(signal_ptrs, RANK, WORLD, sem="acq_rel")
