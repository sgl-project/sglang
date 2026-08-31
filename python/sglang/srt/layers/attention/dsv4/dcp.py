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


class DSV4DCPCombinedQTopKWorkspace(NamedTuple):
    local_combined: torch.Tensor
    gathered_combined: torch.Tensor


def combined_q_topk_candidate_view(
    combined: torch.Tensor,
    *,
    local_heads: int,
    topk: int,
) -> torch.Tensor:
    """Return the zero-copy int64 candidate rows inside a bf16 combined buffer."""
    if combined.ndim != 3 or combined.dtype != torch.bfloat16:
        raise ValueError(
            "combined Q/top-k storage must be a 3D bfloat16 tensor, "
            f"got shape={tuple(combined.shape)}, dtype={combined.dtype}"
        )
    if not combined.is_contiguous():
        raise ValueError("combined Q/top-k storage must be contiguous")
    batch_size, combined_heads, head_dim = combined.shape
    if local_heads <= 0 or local_heads >= combined_heads:
        raise ValueError(
            f"local_heads must be in (0, {combined_heads}), got {local_heads}"
        )
    if topk <= 0 or head_dim % 4 != 0 or (topk * 4) % head_dim != 0:
        raise ValueError(
            f"topk={topk} cannot be represented as whole bf16 heads of D={head_dim}"
        )
    candidate_heads = topk * 4 // head_dim
    if combined_heads != local_heads + candidate_heads:
        raise ValueError(
            "combined Q/top-k shape mismatch: expected "
            f"{local_heads + candidate_heads} heads, got {combined_heads}"
        )

    # Reinterpreting bf16 as int64 shrinks the final dimension by four.
    # Preserve the physical combined row stride explicitly, including B=1
    # where Tensor.view would canonicalize the singleton row stride to K.
    words = combined.view(torch.int64)
    words_per_row = combined_heads * head_dim // 4
    candidate_offset = local_heads * head_dim // 4
    return torch.as_strided(
        words,
        size=(batch_size, topk),
        stride=(words_per_row, 1),
        storage_offset=words.storage_offset() + candidate_offset,
    )


def combined_q_topk_rank_major_q_view(
    gathered_combined: torch.Tensor,
    *,
    batch_size: int,
    local_heads: int,
    topk: int,
    dcp_size: int,
) -> torch.Tensor:
    """Return gathered Q as [B, DCP, Hlocal, D] without materializing it."""
    combined_q_topk_candidate_view(
        gathered_combined,
        local_heads=local_heads,
        topk=topk,
    )
    if gathered_combined.shape[0] != dcp_size * batch_size:
        raise ValueError(
            "gathered combined row count must equal DCP*B, got "
            f"{gathered_combined.shape[0]} != {dcp_size}*{batch_size}"
        )
    combined_heads, head_dim = gathered_combined.shape[1:]
    rank_rows = gathered_combined.view(dcp_size, batch_size, combined_heads, head_dim)
    return rank_rows[:, :, :local_heads, :].permute(1, 0, 2, 3)


def validate_dsv4_dcp_topology(
    *,
    dcp_size: int,
    dcp_rank: int,
    attn_tp_size: int,
    attn_tp_rank: int,
    attn_dp_size: int,
    comm_backend: str = "ag_rs",
    disaggregation_mode: str = "null",
) -> None:
    if dcp_size <= 1:
        return
    if comm_backend not in ("ag_rs", "a2a"):
        raise NotImplementedError(
            "DeepSeek V4 DCP currently supports only the ag_rs and a2a communication "
            f"backend; got dcp_comm_backend={comm_backend!r}."
        )
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


def run_packed_c4_topk(
    *,
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    local_page_table: torch.Tensor,
    local_candidates: torch.Tensor,
    gathered_candidates: torch.Tensor,
    out_page_indices: torch.Tensor,
    out_local_lens: torch.Tensor,
    c4_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_group,
    out_local_raw_indices: torch.Tensor | None = None,
) -> None:
    """Run owner-local C4 candidate selection and one packed all-gather."""
    from sglang.kernels.ops.attention.dsv4 import (
        dcp_topk_candidates,
        dcp_topk_merge,
    )

    local_lens = local_lens.reshape(-1).to(torch.int32).contiguous()
    dcp_topk_candidates(
        logits,
        local_lens,
        local_candidates,
        dcp_size,
        dcp_rank,
    )
    dcp_group.all_gather_into_tensor(gathered_candidates, local_candidates)
    dcp_topk_merge(
        gathered_candidates,
        local_page_table,
        out_page_indices,
        out_local_lens,
        c4_page_size,
        dcp_size,
        dcp_rank,
        out_local_raw_indices,
    )


def run_combined_q_c4_topk(
    *,
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    local_page_table: torch.Tensor,
    workspace: DSV4DCPCombinedQTopKWorkspace,
    local_heads: int,
    out_page_indices: torch.Tensor,
    out_local_lens: torch.Tensor,
    c4_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_group,
    out_local_raw_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select candidates and gather candidates+Q in one contiguous bf16 AG."""
    from sglang.kernels.ops.attention.dsv4 import (
        dcp_topk_candidates,
        dcp_topk_merge,
    )

    topk = out_page_indices.shape[1]
    local_combined, gathered_combined = workspace
    local_candidates = combined_q_topk_candidate_view(
        local_combined,
        local_heads=local_heads,
        topk=topk,
    )
    gathered_candidates = combined_q_topk_candidate_view(
        gathered_combined,
        local_heads=local_heads,
        topk=topk,
    )
    if local_combined.shape[0] != logits.shape[0]:
        raise ValueError(
            "combined Q/top-k workspace must use the exact decode batch size, "
            f"got {local_combined.shape[0]} rows for {logits.shape[0]} queries"
        )
    if gathered_combined.shape[0] != dcp_size * local_combined.shape[0]:
        raise ValueError(
            "gathered combined workspace must have DCP*B rows, got "
            f"{gathered_combined.shape[0]}"
        )

    local_lens = local_lens.reshape(-1).to(torch.int32).contiguous()
    dcp_topk_candidates(
        logits,
        local_lens,
        local_candidates,
        dcp_size,
        dcp_rank,
    )
    dcp_group.all_gather_into_tensor(gathered_combined, local_combined)
    dcp_topk_merge(
        gathered_candidates,
        local_page_table,
        out_page_indices,
        out_local_lens,
        c4_page_size,
        dcp_size,
        dcp_rank,
        out_local_raw_indices,
    )
    return combined_q_topk_rank_major_q_view(
        gathered_combined,
        batch_size=local_combined.shape[0],
        local_heads=local_heads,
        topk=topk,
        dcp_size=dcp_size,
    )


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
