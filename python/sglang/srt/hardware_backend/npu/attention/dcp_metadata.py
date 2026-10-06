"""Ascend-specific metadata builders for MLA decode context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton

from sglang.kernels.ops.attention.dcp_kernels import create_mla_kv_page_table_for_dcp
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


def init_mla_dcp_metadata(
    metadata: ForwardMetadata,
    forward_batch: ForwardBatch,
    req_to_token: torch.Tensor,
    page_size: int,
    speculative_step_id: int,
) -> torch.Tensor:
    """Build token-interleaved FIA metadata for eager decode/verify.

    The caller selects target dense MLA only; DSA's page-interleaved metadata
    and replicated draft KV do not use these builders.
    """
    parallel = get_parallel()
    if forward_batch.forward_mode.is_target_verify():
        query_len = int(forward_batch.spec_info.draft_token_num)
        effective_seq_lens = forward_batch.seq_lens.to(torch.int64) + query_len
        metadata.dcp_mtp_attn_mask = build_mla_dcp_mtp_mask(
            forward_batch.seq_lens, query_len, parallel.dcp_size, parallel.dcp_rank
        )
    else:
        effective_seq_lens = forward_batch.seq_lens
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            effective_seq_lens = effective_seq_lens + int(speculative_step_id + 1)
    metadata.block_tables, local_seq_lens = build_mla_dcp_local_block_tables(
        req_to_token,
        forward_batch.req_pool_indices,
        effective_seq_lens,
        page_size,
        parallel.dcp_size,
        parallel.dcp_rank,
    )
    if metadata.dcp_mtp_attn_mask is not None:
        # FIA checks S2 against the full paged capacity, not just valid KV.
        required_mask_width = metadata.block_tables.shape[1] * page_size
        mask = metadata.dcp_mtp_attn_mask
        if mask.shape[-1] < required_mask_width:
            mask = torch.cat(
                [
                    mask,
                    mask.new_ones(
                        *mask.shape[:-1], required_mask_width - mask.shape[-1]
                    ),
                ],
                dim=-1,
            )
        metadata.dcp_mtp_attn_mask = mask.contiguous()
    return local_seq_lens


def init_mla_dcp_graph_state(
    max_bs: int,
    graph_context_len: int,
    page_size: int,
    speculative_num_draft_tokens: int | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate local FIA page tables, lengths and mask at fixed graph addresses."""
    parallel = get_parallel()
    max_local_context_len = graph_context_len // parallel.dcp_size + int(
        parallel.dcp_rank < graph_context_len % parallel.dcp_size
    )
    graph_num_pages = max(1, (max_local_context_len + page_size - 1) // page_size)
    buffers = {
        "block_tables": torch.empty(
            (max_bs, graph_num_pages), dtype=torch.int32, device=device
        ),
        "dcp_seq_lens": torch.zeros(max_bs, dtype=torch.int32, device=device),
    }
    if speculative_num_draft_tokens is not None:
        buffers["dcp_mtp_attn_mask"] = torch.ones(
            (max_bs, speculative_num_draft_tokens, graph_num_pages * page_size),
            dtype=torch.bool,
            device=device,
        )
    return buffers


def update_mla_dcp_graph_metadata(
    metadata: ForwardMetadata,
    req_to_token: torch.Tensor,
    *,
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    forward_mode: ForwardMode,
    spec_info: SpecInput | None,
    max_len: int,
    page_size: int,
    speculative_num_draft_tokens: int | None,
    speculative_step_id: int,
    in_capture: bool,
) -> None:
    """Update captured FIA buffers in place without replay-time CPU length reads."""
    parallel = get_parallel()
    if forward_mode.is_target_verify():
        query_len = int(speculative_num_draft_tokens)
        effective_seq_lens = seq_lens[:bs].to(torch.int64) + query_len
    elif forward_mode.is_decode_or_idle() and spec_info is not None:
        effective_seq_lens = seq_lens[:bs].to(torch.int64) + int(
            speculative_step_id + 1
        )
    else:
        effective_seq_lens = seq_lens[:bs].to(torch.int64)

    max_local_len = max_len // parallel.dcp_size + int(
        parallel.dcp_rank < max_len % parallel.dcp_size
    )
    active_num_pages = min(
        metadata.block_tables.shape[1],
        max(1, (max_local_len + page_size - 1) // page_size),
    )
    dcp_block_tables, local_seq_lens = build_mla_dcp_local_block_tables(
        req_to_token,
        req_pool_indices[:bs],
        effective_seq_lens,
        page_size,
        parallel.dcp_size,
        parallel.dcp_rank,
        num_pages=active_num_pages,
    )
    metadata.block_tables[:, :active_num_pages].copy_(dcp_block_tables)
    metadata.block_tables[:, active_num_pages:].zero_()
    metadata.seq_lens.copy_(local_seq_lens)
    # Capture consumes the list; NPUGraphRunner patches FIA's lengths on replay.
    if in_capture:
        metadata.seq_lens_cpu_list = local_seq_lens.cpu().int().tolist()

    if forward_mode.is_target_verify():
        dcp_mask = build_mla_dcp_mtp_mask(
            seq_lens[:bs],
            query_len,
            parallel.dcp_size,
            parallel.dcp_rank,
            max_local_kv_len=active_num_pages * page_size,
        )
        active_mask_width = active_num_pages * page_size
        metadata.dcp_mtp_attn_mask[:, :, :active_mask_width].copy_(dcp_mask)
        metadata.dcp_mtp_attn_mask[:, :, active_mask_width:].fill_(True)


def build_mla_dcp_local_block_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    physical_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    *,
    num_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the FIA view and fill it with the common DCP page-table kernel.

    Lengths must fit req_to_token; an explicit num_pages must cover every
    local sequence. The graph caller sizes it from the CPU length upper bound.
    """
    local_seq_lens = get_dcp_lens(seq_lens, dcp_size, dcp_rank).to(torch.int32)
    if num_pages is None:
        max_len = int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        num_pages = max(1, (max_len + physical_page_size - 1) // physical_page_size)
    block_tables = torch.zeros(
        (req_pool_indices.numel(), num_pages),
        dtype=torch.int32,
        device=req_to_token.device,
    )
    if not req_pool_indices.numel():
        return block_tables, local_seq_lens
    create_mla_kv_page_table_for_dcp[
        (req_pool_indices.numel(), triton.cdiv(num_pages, 128))
    ](
        req_to_token,
        req_pool_indices,
        local_seq_lens,
        block_tables,
        None,
        req_to_token.stride(0),
        block_tables.stride(0),
        PHYSICAL_PAGE_SIZE=physical_page_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        PAGES_PER_BLOCK=128,
        HAS_V2P=False,
    )
    return block_tables, local_seq_lens


def build_mla_dcp_mtp_mask(
    prefix_lens: torch.Tensor,
    query_len: int,
    dcp_size: int,
    dcp_rank: int,
    *,
    max_local_kv_len: int | None = None,
) -> torch.Tensor:
    """Build FIA's rank-local causal mask for a fixed DSPARK verify window."""
    prefix_lens = prefix_lens.to(torch.int64)
    total_lens = prefix_lens + query_len
    local_seq_lens = get_dcp_lens(total_lens, dcp_size, dcp_rank).to(torch.int32)

    if max_local_kv_len is None:
        max_local_kv_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        )

    # FIA rejects a zero-width mask, including for an empty local shard.
    mask = torch.ones(
        (prefix_lens.numel(), max(1, query_len), max(1, max_local_kv_len)),
        dtype=torch.bool,
        device=prefix_lens.device,
    )
    if query_len == 0 or max_local_kv_len == 0:
        return mask.contiguous()

    q_idx = torch.arange(query_len, device=prefix_lens.device, dtype=torch.int64)
    k_idx = torch.arange(max_local_kv_len, device=prefix_lens.device, dtype=torch.int64)
    last_visible = torch.div(
        prefix_lens[:, None] + q_idx[None, :] - dcp_rank,
        dcp_size,
        rounding_mode="floor",
    )
    mask = k_idx[None, None, :] > last_visible[:, :, None]
    mask |= k_idx[None, None, :] >= local_seq_lens.to(torch.int64)[:, None, None]
    return mask.contiguous()
