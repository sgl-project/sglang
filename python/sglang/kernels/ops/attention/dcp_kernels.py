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
import triton
import triton.language as tl

from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
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
    req_to_token_stride: tl.constexpr,
    block_table_stride: tl.constexpr,
    PHYSICAL_PAGE_SIZE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
):
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
    physical_pages = virtual_locs // DCP_SIZE // PHYSICAL_PAGE_SIZE
    tl.store(
        block_kv_indices_ptr + req * block_table_stride + page_offsets,
        physical_pages,
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
    tl.store(
        local_kv_indices + local_kv_indices_offsets,
        kv_values // dcp_world_size,
        mask=mask,
    )


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

    acc = acc / weight_sum

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


def _lse_weighted_combine_cpu(
    partial_outputs: torch.Tensor,
    partial_lses: torch.Tensor,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """CPU reference: combine N partial attention outputs using LSE weights.

    Args:
        partial_outputs: [N, B, H_local, D]
        partial_lses:    [N, B, H_local]
        is_lse_base_on_e: base-e (True) or base-2 (False)

    Returns:
        [B, H_local, D] combined output
    """
    N, B, H_local, D = partial_outputs.shape
    partial_outputs = partial_outputs.float()
    partial_lses = partial_lses.float()

    # Sanitize
    partial_lses = torch.where(
        torch.isnan(partial_lses) | torch.isinf(partial_lses),
        torch.full_like(partial_lses, float("-inf")),
        partial_lses,
    )

    # max LSE for numerical stability
    lse_max, _ = partial_lses.max(dim=0)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    centered = partial_lses - lse_max.unsqueeze(0)
    if is_lse_base_on_e:
        weights = torch.exp(centered)
    else:
        weights = torch.pow(2.0, centered)

    weight_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weight_sum

    combined = (partial_outputs * weights.unsqueeze(-1)).sum(dim=0)
    return combined


# ---------------------------------------------------------------------------
# aiter MLA DCP: page-table build, KV page packing, and the split-KV reduce.
#
# aiter's ``mla_decode_fwd(..., skip_reduce=True)`` returns the per-segment
# partials ``(segm_output, segm_max, segm_expsum)`` but its own reduce kernel
# only writes the merged ``output`` -- no LSE. DCP needs the LSE of each rank's
# KV shard to merge partial attention across ranks (``cp_lse_ag_out_rs_mla`` ->
# ``correct_attn_out``, base-2), so ``dcp_mla_reduce`` below replicates aiter's
# reduce math and additionally writes
# ``lse = overall_max + log2(overall_expsum)``.
#
# Note these are an INTRA-rank reduce over aiter's split-KV segments, distinct
# from ``correct_attn_out`` / ``dcp_lse_combine_triton`` above, which merge
# across ranks.
# ---------------------------------------------------------------------------


def build_dcp_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    local_kv_lens: torch.Tensor,
    bs: int,
    max_pages: int,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    out: Optional[torch.Tensor] = None,
):
    """Build this rank's PAGE table for aiter's ``mla_decode_fwd``.

    mla_decode_fwd derives its KV tile straight from the paged block size
    (``TILE_SIZE == block_size``; the MLA kernel asserts
    ``NUM_BLOCKS_GATHER_PER_TILE == 1``), so a block size of 1 collapses every
    tile to a single token: measured on gfx950 that is ~19x slower than a block
    size of 16 at the same KV volume (27 vs 511 GB/s), and it dominated the DCP
    decode step (~96% of a 128k-context ITL).

    Under DCP the allocator's page is ``page_size * dcp_size`` (see
    kv_cache_builder), so each rank holds ``page_size`` CONTIGUOUS physical
    slots per virtual page and the shard can be addressed by page. The
    per-token owner rule is unchanged (``pos % dcp_size == rank``, physical
    ``pos // dcp_size``); paging only guarantees the contiguity that lets the
    tile match the page.

    ``local_kv_lens`` is this rank's shard length per request, in TOKENS
    (the kernel's ``seqused_k``); the table itself is indexed in pages.
    """
    if out is None:
        out = torch.zeros(bs, max_pages, dtype=torch.int32, device=req_to_token.device)
    if max_pages == 0:
        return out
    pages_per_block = get_num_page_per_block_flashmla(page_size)
    create_mla_kv_page_table_for_dcp[
        (bs, get_num_kv_index_blocks_flashmla(max_pages, page_size))
    ](
        req_to_token,
        req_pool_indices,
        local_kv_lens,
        out,
        req_to_token.stride(0),
        out.stride(0),
        PHYSICAL_PAGE_SIZE=page_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        PAGES_PER_BLOCK=pages_per_block,
    )
    return out


@triton.jit
def _pack_dcp_kv_pages_kernel(
    src_ptr,  # [n_src, 1, D] assembled dcp_kv_buffer
    dst_ptr,  # [n_pages * PAGE, 1, D] paged staging buffer
    src_idx_ptr,  # [total] dcp_kv_indices: sequence position -> src row
    dst_idx_ptr,  # [total] sequence position -> padded/paged dst row
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    src = tl.load(src_idx_ptr + row).to(tl.int64)
    dst = tl.load(dst_idx_ptr + row).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    tl.store(
        dst_ptr + dst * D + offs,
        tl.load(src_ptr + src * D + offs, mask=mask),
        mask=mask,
    )


def pack_dcp_kv_into_pages(
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
    page_size: int,
):
    """Repack the per-forward ``dcp_kv_buffer`` into a paged staging buffer for
    aiter's ``mla_prefill_fwd``.

    The assembled buffer is laid out as [all requests' gathered prefixes | all
    requests' extend tokens], so a request's sequence is split across two
    regions and cannot be addressed by page (the kernel requires every page but the
    sequence's last to be full). Repacking into per-request page-aligned regions
    lets the KV tile match the page: the kernel takes TILE_SIZE straight from
    the block size, and at block size 1 a 16384x16384 chunk measured 5156 ms vs
    39 ms at 64 on gfx950 (~131x).

    The layout of ``dcp_kv_buffer`` itself is left alone -- flashinfer_mla reads
    the same buffer. The copy is one pass over the batch's KV (a prefill batch
    holds only the few requests of one chunk), which the attention saving dwarfs.

    Returns (paged_kv [n_pages, page_size, 1, D], block_tables [bs, max_pages]).
    """
    total, _, dim = kv_buffer.shape[0], kv_buffer.shape[1], kv_buffer.shape[-1]
    device = kv_buffer.device
    lens = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(torch.int64)
    pages = (lens + page_size - 1) // page_size
    page_start = torch.cumsum(pages, dim=0) - pages
    # sequence position -> row in the padded buffer
    shift = page_start * page_size - kv_indptr[:bs].to(torch.int64)
    n_tokens = int(kv_indptr[bs].item())
    dst_idx = torch.repeat_interleave(shift, lens) + torch.arange(
        n_tokens, device=device, dtype=torch.int64
    )

    n_pages = int(pages.sum().item())
    # zeros, not empty: the kernel reads the whole last page of a sequence and masks
    # by seqused_k, so uninitialized tail rows could feed NaNs into the QK GEMM.
    paged = torch.zeros(
        (n_pages * page_size, 1, dim), dtype=kv_buffer.dtype, device=device
    )
    if n_tokens > 0:
        _pack_dcp_kv_pages_kernel[(n_tokens,)](
            kv_buffer,
            paged,
            kv_indices,
            dst_idx,
            D=dim,
            BLOCK=triton.next_power_of_2(dim),
        )
    max_pages = int(pages.max().item()) if bs > 0 else 0
    block_tables = page_start[:, None] + torch.arange(
        max_pages, device=device, dtype=torch.int64
    )
    return paged.view(-1, page_size, 1, dim), block_tables.to(torch.int32)


@triton.jit
def _dcp_mla_reduce_kernel(
    out_ptr,  # [num_tokens, num_query_heads, KV_LORA_RANK]
    lse_ptr,  # [num_tokens, num_query_heads] (base-2)
    segm_output_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS, KV_LORA_RANK]
    segm_max_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS]
    seq_lens_ptr,  # [num_tokens] local (this-rank shard) kv length per token
    num_query_heads: tl.constexpr,
    out_stride0: tl.int64,
    out_stride1: tl.int64,
    lse_stride0: tl.int64,
    TILE_SIZE: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + tok)

    # A rank owns no committed KV for a request whose prefix is shorter than the
    # rank index, so an all-zero shard length is a normal input here (the planner
    # clamps local_kv_lens to min 0, not min 1). Emit the identity element of the
    # LSE merge -- out = 0 with lse = -inf, which lse_combine_base2 and
    # cp_lse_ag_out_rs_mla both weight to zero.
    #
    # This has to be an early return, not a mask on the result: with seq_len 0
    # tiles_per_segment below is 0, so act_num_segments divides by zero, and the
    # NaNs that follow cannot be cleaned up downstream -- the merge zeroes the
    # WEIGHT via nan_to_num, but NaN * 0 is still NaN, so a single empty shard
    # poisons the merged output of an otherwise healthy batch.
    if seq_len == 0:
        tl.store(
            out_ptr
            + tok * out_stride0
            + head * out_stride1
            + tl.arange(0, KV_LORA_RANK),
            tl.zeros([KV_LORA_RANK], dtype=out_ptr.type.element_ty),
        )
        tl.store(lse_ptr + tok * lse_stride0 + head, float("-inf"))
        return

    # aiter picks the same segment count regardless of seq_len; only the first
    # act_num_segments hold valid data (the rest of the empty() buffer is garbage).
    tiles_per_segment = tl.cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = tl.cdiv(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < act_num_segments

    seg_off = (
        tok.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + head * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + seg_off, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + seg_off, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    out_off = (
        tok.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + head * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    segm_output = tl.load(segm_output_ptr + out_off, mask=segm_mask[:, None], other=0.0)
    segm_output = segm_output * tl.math.exp2(segm_max - overall_max)[:, None]
    acc = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc / overall_expsum)

    # base-2 LSE, matching correct_attn_out / cp_lse_ag_out_rs_mla.
    lse = tl.where(
        overall_expsum == 0.0, float("-inf"), overall_max + tl.log2(overall_expsum)
    )

    tl.store(
        out_ptr + tok * out_stride0 + head * out_stride1 + tl.arange(0, KV_LORA_RANK),
        acc.to(out_ptr.type.element_ty),
    )
    tl.store(lse_ptr + tok * lse_stride0 + head, lse)


def dcp_mla_reduce(
    segm_output: torch.Tensor,
    segm_max: torch.Tensor,
    segm_expsum: torch.Tensor,
    seq_lens: torch.Tensor,
    tile_size: int,
    out_dtype: torch.dtype,
):
    """Reduce mla_decode_fwd skip_reduce partials to (out, lse2).

    Args:
        segm_output: [num_tokens, H, NUM_SEGMENTS, KV_LORA_RANK]
        segm_max / segm_expsum: [num_tokens, H, NUM_SEGMENTS]
        seq_lens: [num_tokens] local (this-rank) kv length per token
        tile_size: kernel TILE_SIZE (== paged block_size passed to mla_decode_fwd)
    Returns:
        out: [num_tokens, H, KV_LORA_RANK] (out_dtype)
        lse: [num_tokens, H] float32, base-2
    """
    num_tokens, num_heads, num_segments, kv_lora_rank = segm_output.shape
    out = torch.empty(
        num_tokens, num_heads, kv_lora_rank, dtype=out_dtype, device=segm_output.device
    )
    lse = torch.empty(
        num_tokens, num_heads, dtype=torch.float32, device=segm_output.device
    )
    _dcp_mla_reduce_kernel[(num_tokens, num_heads)](
        out,
        lse,
        segm_output,
        segm_max,
        segm_expsum,
        seq_lens,
        num_query_heads=num_heads,
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        lse_stride0=lse.stride(0),
        TILE_SIZE=tile_size,
        KV_LORA_RANK=kv_lora_rank,
        NUM_SEGMENTS_PER_SEQ=num_segments,
    )
    return out, lse
