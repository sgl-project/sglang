from __future__ import annotations

from typing import NamedTuple

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens


class DSV4DCPIndices(NamedTuple):
    local: torch.Tensor
    owned: torch.Tensor


class DSV4C4TopKResult(NamedTuple):
    page_indices: torch.Tensor
    local_raw_indices: torch.Tensor
    local_lens: torch.Tensor
    global_indices: torch.Tensor


def validate_dsv4_dcp_topology(
    *,
    dcp_size: int,
    dcp_rank: int,
    attn_tp_size: int,
    attn_tp_rank: int,
    attn_dp_size: int,
    disaggregation_mode: str = "null",
) -> None:
    if dcp_size <= 1:
        return
    if disaggregation_mode != "null":
        raise NotImplementedError(
            "DeepSeek V4 DCP does not support disaggregated serving yet; "
            "the PD transfer path is not owner-local aware."
        )
    if attn_dp_size != 1:
        raise NotImplementedError(
            "DeepSeek V4 DCP currently requires attention data parallel size 1; "
            "otherwise a DCP group can gather Q across different requests."
        )
    if attn_tp_size % dcp_size != 0:
        raise NotImplementedError(
            "DeepSeek V4 DCP requires attention TP size divisible by DCP size, "
            f"got attn_tp_size={attn_tp_size}, dcp_size={dcp_size}."
        )
    if attn_tp_rank % dcp_size != dcp_rank:
        raise RuntimeError(
            "DeepSeek V4 DCP rank mapping is inconsistent: "
            f"attn_tp_rank={attn_tp_rank}, dcp_rank={dcp_rank}, "
            f"dcp_size={dcp_size}."
        )


def select_dcp_attn_sink(
    attn_sink: torch.Tensor,
    local_num_heads: int,
    attn_tp_rank: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    if dcp_size < 1 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(
            f"Invalid DCP geometry: dcp_size={dcp_size}, dcp_rank={dcp_rank}"
        )
    group_start_rank = attn_tp_rank - dcp_rank
    if group_start_rank < 0:
        raise ValueError(
            f"Invalid DCP subgroup: attn_tp_rank={attn_tp_rank}, "
            f"dcp_rank={dcp_rank}"
        )
    head_start = group_start_rank * local_num_heads
    head_end = head_start + dcp_size * local_num_heads
    if head_end > attn_sink.numel():
        raise ValueError(
            f"DCP sink slice [{head_start}:{head_end}] exceeds "
            f"{attn_sink.numel()} global heads"
        )
    return attn_sink[head_start:head_end].contiguous()


def select_dsv4_attn_sink_input(
    global_attn_sink: torch.Tensor,
    local_attn_sink: torch.Tensor,
    local_num_heads: int,
    dcp_size: int,
) -> torch.Tensor:
    if dcp_size < 1:
        raise ValueError(f"Invalid DCP size: {dcp_size}")
    if dcp_size == 1:
        if local_num_heads > local_attn_sink.numel():
            raise ValueError(
                f"Local sink has {local_attn_sink.numel()} heads, "
                f"need {local_num_heads}"
            )
        return local_attn_sink[:local_num_heads].contiguous()
    return global_attn_sink


def localize_full_indices(
    global_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
) -> DSV4DCPIndices:
    if dcp_size == 1:
        return DSV4DCPIndices(global_indices, global_indices >= 0)
    owned = (global_indices >= 0) & (global_indices % dcp_size == dcp_rank)
    local = torch.where(owned, global_indices // dcp_size, -1)
    return DSV4DCPIndices(local, owned)


def localize_compressed_indices(
    global_full_indices: torch.Tensor,
    compress_ratio: int,
    dcp_size: int,
    dcp_rank: int,
) -> DSV4DCPIndices:
    if compress_ratio not in (4, 128):
        raise ValueError(f"Unsupported DSV4 compression ratio: {compress_ratio}")
    global_compressed = global_full_indices // compress_ratio
    owned = (global_full_indices >= 0) & (global_compressed % dcp_size == dcp_rank)
    local = torch.where(owned, global_compressed // dcp_size, -1)
    return DSV4DCPIndices(local, owned)


def local_compressed_lens(
    seq_lens: torch.Tensor,
    compress_ratio: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    if compress_ratio not in (4, 128):
        raise ValueError(f"Unsupported DSV4 compression ratio: {compress_ratio}")
    return get_dcp_lens(seq_lens // compress_ratio, dcp_size, dcp_rank)


def local_swa_lens(
    seq_lens: torch.Tensor,
    window_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    starts = torch.clamp(seq_lens - window_size, min=0)
    lengths = seq_lens - starts
    return get_dcp_lens(lengths, dcp_size, dcp_rank, start=starts)


def build_local_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    max_seq_len: int,
    physical_page_size: int,
    dcp_size: int,
) -> torch.Tensor:
    logical_page_size = physical_page_size * dcp_size
    global_page_starts = req_to_token[
        req_pool_indices.to(torch.int64), :max_seq_len:logical_page_size
    ]
    return (global_page_starts // logical_page_size).to(torch.int32)


def local_c4_topk_candidates(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    topk: int,
    dcp_size: int,
    dcp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    width = logits.shape[1]
    positions = torch.arange(width, device=logits.device, dtype=torch.int64)
    valid = positions.unsqueeze(0) < local_lens.to(torch.int64).unsqueeze(1)
    masked = logits.float().masked_fill(~valid, -float("inf"))
    actual_k = min(topk, width)
    order = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :actual_k]
    scores = torch.gather(masked, 1, order)
    selected_valid = torch.gather(valid, 1, order)
    global_indices = order * dcp_size + dcp_rank
    global_indices = global_indices.masked_fill(~selected_valid, -1)
    scores = scores.masked_fill(~selected_valid, -float("inf"))
    if actual_k < topk:
        scores = torch.nn.functional.pad(
            scores, (0, topk - actual_k), value=-float("inf")
        )
        global_indices = torch.nn.functional.pad(
            global_indices, (0, topk - actual_k), value=-1
        )
    return scores, global_indices


def merge_c4_topk_candidates(
    candidate_scores: torch.Tensor,
    candidate_global_indices: torch.Tensor,
    topk: int,
    dcp_size: int,
    dcp_rank: int,
    page_table: torch.Tensor,
    c4_page_size: int,
) -> DSV4C4TopKResult:
    valid = candidate_global_indices >= 0
    sentinel = torch.iinfo(candidate_global_indices.dtype).max
    ids_for_sort = torch.where(valid, candidate_global_indices, sentinel)
    id_order = torch.argsort(ids_for_sort, dim=1, stable=True)
    scores_by_id = torch.gather(candidate_scores, 1, id_order)
    ids_by_id = torch.gather(candidate_global_indices, 1, id_order)
    score_order = torch.argsort(scores_by_id, dim=1, descending=True, stable=True)
    score_order = score_order[:, :topk]
    global_indices = torch.gather(ids_by_id, 1, score_order)

    owned = (global_indices >= 0) & (global_indices % dcp_size == dcp_rank)
    owner_order = torch.argsort((~owned).to(torch.int32), dim=1, stable=True)
    global_by_owner = torch.gather(global_indices, 1, owner_order)
    owned_by_owner = torch.gather(owned, 1, owner_order)
    local_raw = torch.where(
        owned_by_owner,
        global_by_owner // dcp_size,
        -1,
    )
    local_lens = owned.sum(dim=1).to(torch.int32)

    page_ids = torch.clamp(local_raw // c4_page_size, min=0)
    offsets = torch.remainder(torch.clamp(local_raw, min=0), c4_page_size)
    physical_pages = torch.gather(page_table, 1, page_ids.to(torch.int64))
    page_indices = physical_pages * c4_page_size + offsets
    page_indices = page_indices.to(torch.int32).masked_fill(~owned_by_owner, -1)
    return DSV4C4TopKResult(
        page_indices=page_indices,
        local_raw_indices=local_raw.to(torch.int32),
        local_lens=local_lens,
        global_indices=global_indices.to(torch.int32),
    )
