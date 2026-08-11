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

"""Split-KV reduce for aiter's MLA decode, emitting per-(token, head) LSE.

aiter's ``mla_decode_fwd(..., skip_reduce=True)`` returns the per-segment
partials ``(segm_output, segm_max, segm_expsum)`` but its own reduce kernel only
writes the merged ``output`` — no LSE. Decode context parallel (DCP) needs the
LSE of each rank's KV shard to merge partial attention across ranks
(``cp_lse_ag_out_rs_mla`` -> ``correct_attn_out``, which is base-2).

This kernel replicates aiter's reduce math (base-2, unnormalized ``segm_output``,
``act_num_segments`` masking) and additionally writes ``lse = overall_max +
log2(overall_expsum)`` so the DCP cross-rank merge can consume it directly.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dcp_kernels import create_mla_kv_page_table_for_dcp
from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)


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
