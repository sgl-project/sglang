"""Ascend-specific metadata builders for MLA decode context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.dcp.layout import get_dcp_chain_spec_lens, get_dcp_lens

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
    *,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Build page-interleaved FIA metadata for eager decode/verify.

    The caller selects target dense MLA only; DSA's page-interleaved metadata
    and replicated draft KV do not use these builders.
    """
    if forward_batch.forward_mode.is_target_verify():
        query_len = int(forward_batch.spec_info.draft_token_num)
        effective_seq_lens = forward_batch.seq_lens.to(torch.int64) + query_len
        metadata.dcp_spec_seq_lens = get_dcp_chain_spec_lens(
            effective_seq_lens,
            query_len,
            dcp_size,
            dcp_rank,
            interleave_size=page_size,
        )
    else:
        metadata.dcp_spec_seq_lens = None
        metadata.dcp_mtp_attn_mask = None
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
        dcp_size,
        dcp_rank,
    )
    metadata.seq_lens = local_seq_lens
    if metadata.dcp_spec_seq_lens is not None:
        # FIA checks S2 against the full paged capacity, not just valid KV.
        required_mask_width = metadata.block_tables.shape[1] * page_size
        metadata.dcp_mtp_attn_mask = build_mla_dcp_mtp_mask(
            forward_batch.seq_lens,
            query_len,
            dcp_size,
            dcp_rank,
            page_size=page_size,
            max_local_kv_len=required_mask_width,
            local_query_lens=metadata.dcp_spec_seq_lens,
        )
    # Derive FIA's Host lengths from the CPU mirror without a device readback.
    effective_seq_lens_cpu = forward_batch.seq_lens_cpu.to(torch.int64)
    if forward_batch.forward_mode.is_target_verify():
        spec_algorithm = forward_batch.spec_algorithm
        if spec_algorithm is not None and spec_algorithm.is_dflash_family():
            # Active Host lengths already include verify tokens; Ascend's
            # zero-filled DP padding rows still need the verify block.
            effective_seq_lens_cpu = torch.where(
                effective_seq_lens_cpu == 0, query_len, effective_seq_lens_cpu
            )
        else:
            effective_seq_lens_cpu = effective_seq_lens_cpu + query_len
    elif (
        forward_batch.forward_mode.is_decode_or_idle()
        and forward_batch.spec_info is not None
    ):
        effective_seq_lens_cpu = effective_seq_lens_cpu + int(speculative_step_id + 1)
    metadata.seq_lens_cpu_int = get_dcp_lens(
        effective_seq_lens_cpu, dcp_size, dcp_rank, interleave_size=page_size
    ).int()
    return local_seq_lens


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
    dcp_size: int,
    dcp_rank: int,
) -> None:
    """Update captured FIA buffers in place without replay-time CPU length reads."""
    if forward_mode.is_target_verify():
        query_len = int(speculative_num_draft_tokens)
        effective_seq_lens = seq_lens[:bs].to(torch.int64) + query_len
    elif forward_mode.is_decode_or_idle() and spec_info is not None:
        effective_seq_lens = seq_lens[:bs].to(torch.int64) + int(
            speculative_step_id + 1
        )
    else:
        effective_seq_lens = seq_lens[:bs].to(torch.int64)

    cycle_size = page_size * dcp_size
    max_local_len = max_len // cycle_size * page_size + min(
        page_size, max(0, max_len % cycle_size - dcp_rank * page_size)
    )
    active_num_pages = min(
        metadata.block_tables.shape[1],
        max(1, (max_local_len + page_size - 1) // page_size),
    )
    metadata.block_tables.zero_()
    _, local_seq_lens = build_mla_dcp_local_block_tables(
        req_to_token,
        req_pool_indices[:bs],
        effective_seq_lens,
        page_size,
        dcp_size,
        dcp_rank,
        num_pages=active_num_pages,
        block_tables=metadata.block_tables[:, :active_num_pages],
    )
    metadata.seq_lens.copy_(local_seq_lens)
    # Capture consumes the list; NPUGraphRunner patches FIA's lengths on replay.
    if in_capture:
        metadata.seq_lens_cpu_list = local_seq_lens.cpu().int().tolist()

    if forward_mode.is_target_verify():
        metadata.dcp_spec_seq_lens.copy_(
            get_dcp_chain_spec_lens(
                effective_seq_lens,
                query_len,
                dcp_size,
                dcp_rank,
                interleave_size=page_size,
            )
        )
        dcp_mask = build_mla_dcp_mtp_mask(
            seq_lens[:bs],
            query_len,
            dcp_size,
            dcp_rank,
            page_size=page_size,
            max_local_kv_len=active_num_pages * page_size,
            local_query_lens=metadata.dcp_spec_seq_lens,
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
    block_tables: torch.Tensor | None = None,
    query_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate or fill FIA's page-interleaved rank-local block table.

    Lengths must fit req_to_token; an explicit num_pages must cover every
    local sequence. The graph caller sizes it from the CPU length upper bound.
    """
    if query_len is None:
        local_seq_lens = get_dcp_lens(
            seq_lens, dcp_size, dcp_rank, interleave_size=physical_page_size
        ).to(torch.int32)
    else:
        local_seq_lens = get_dcp_chain_spec_lens(
            seq_lens, query_len, dcp_size, dcp_rank, interleave_size=physical_page_size
        )
    if num_pages is None:
        max_len = int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        num_pages = max(1, (max_len + physical_page_size - 1) // physical_page_size)
    if block_tables is None:
        block_tables = torch.zeros(
            (req_pool_indices.numel(), num_pages),
            dtype=torch.int32,
            device=req_to_token.device,
        )
    if not req_pool_indices.numel():
        return block_tables, local_seq_lens
    page_stride = physical_page_size * dcp_size
    torch.div(
        req_to_token[req_pool_indices, : num_pages * page_stride : page_stride],
        page_stride,
        rounding_mode="floor",
        out=block_tables,
    )
    if query_len is not None:
        block_tables = block_tables.repeat_interleave(query_len, dim=0)
    return block_tables, local_seq_lens


def build_mla_dcp_mtp_mask(
    prefix_lens: torch.Tensor,
    query_len: int,
    dcp_size: int,
    dcp_rank: int,
    *,
    page_size: int = 1,
    max_local_kv_len: int | None = None,
    local_query_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adapt common per-query DCP frontiers to FIA's fixed verify mask."""
    prefix_lens = prefix_lens.to(torch.int64)
    if max_local_kv_len is None:
        local_seq_lens = get_dcp_lens(
            prefix_lens + query_len,
            dcp_size,
            dcp_rank,
            interleave_size=page_size,
        ).to(torch.int32)
        max_local_kv_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        )

    # FIA rejects a zero-width mask, including for an empty local shard.
    if query_len == 0 or max_local_kv_len == 0:
        return torch.ones(
            (prefix_lens.numel(), max(1, query_len), max(1, max_local_kv_len)),
            dtype=torch.bool,
            device=prefix_lens.device,
        ).contiguous()

    if local_query_lens is None:
        local_query_lens = get_dcp_chain_spec_lens(
            prefix_lens + query_len,
            query_len,
            dcp_size,
            dcp_rank,
            interleave_size=page_size,
        )
    k_idx = torch.arange(max_local_kv_len, device=prefix_lens.device, dtype=torch.int64)
    mask = k_idx[None, None, :] >= local_query_lens.view(
        prefix_lens.numel(), query_len, 1
    )
    return mask.contiguous()
