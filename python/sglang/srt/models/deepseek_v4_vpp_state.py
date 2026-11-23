"""Batch-owned shared-attention state at eager V4.1 virtual-stage boundaries."""

from __future__ import annotations

import torch


def stage_hash_ids(stage_id, hasher, input_ids, batch, tensors):
    """Hash and commit Engram history once per batch, at the first stage."""
    if hasher is None:
        return None
    if stage_id == 0:
        return hasher(input_ids, batch)
    hashes = tensors.get("vpp_engram_hash_ids")
    if hashes is None:
        raise ValueError("An Engram VPP handoff requires this batch's hash ids")
    return hashes


def request_rows(batch):
    requests = batch.req_pool_indices.to(torch.int64)
    if batch.forward_mode.is_decode():
        return requests
    return torch.repeat_interleave(
        requests,
        batch.extend_seq_lens.to(torch.int64),
        output_size=batch.input_ids.numel(),
    )


def full_page_ids(batch, req_to_token, page_size):
    """Logical page order, including a partially written last page."""
    pages = [
        req_to_token[request, : int(length) : page_size] // page_size
        for request, length in zip(batch.req_pool_indices, batch.seq_lens_cpu)
        if int(length)
    ]
    return (
        torch.cat(pages).to(torch.int64)
        if pages
        else torch.empty(0, dtype=torch.int64, device=req_to_token.device)
    )


def logical_selection(slots, batch, req_to_token, page_size, ratio):
    """Convert pool slots to request-local positions before crossing ranks."""
    width = (max(int(n) for n in batch.seq_lens_cpu) + page_size - 1) // page_size
    table = (
        req_to_token[
            batch.req_pool_indices.to(torch.int64), : width * page_size : page_size
        ].to(torch.int64)
        // page_size
    )
    counts = (batch.seq_lens.to(torch.int64) + page_size - 1) // page_size
    table = table.masked_fill(
        torch.arange(width, device=table.device)[None, :] >= counts[:, None],
        torch.iinfo(torch.int64).max,
    )
    physical, logical = table.sort(dim=1)
    rows = torch.arange(table.shape[0], device=table.device)
    if not batch.forward_mode.is_decode():
        rows = torch.repeat_interleave(
            rows, batch.extend_seq_lens.to(torch.int64), output_size=slots.shape[0]
        )
    compressed_page = page_size // ratio
    selected_pages = slots.to(torch.int64).clamp_min(0) // compressed_page
    indices = torch.searchsorted(
        physical[rows].contiguous(), selected_pages.contiguous()
    ).clamp_max(width - 1)
    positions = logical[rows].gather(1, indices) * compressed_page
    positions += slots.clamp_min(0) % compressed_page
    return positions.masked_fill(slots < 0, -1).to(torch.int32)


def physical_selection(positions, batch, req_to_token, ratio):
    rows = request_rows(batch)
    full_positions = positions.to(torch.int64).clamp_min(0) * ratio
    slots = req_to_token[rows[:, None], full_positions] // ratio
    return slots.masked_fill(positions < 0, -1).to(torch.int32)


def export_state(tensors, *, config, next_layer, batch, pool, req_to_token, metadata):
    ratio = config.compress_ratios[next_layer]
    if ratio in (1, 2):
        source = max(
            layer
            for layer in config.kv_source_layer_ids
            if layer <= next_layer and config.compress_ratios[layer] == ratio
        )
        if source < next_layer:
            tensors["vpp_source_layer"] = source
            tensors.update(
                {
                    f"vpp_source_{name}": value
                    for name, value in pool.export_source_pages(
                        source, full_page_ids(batch, req_to_token, pool.page_size)
                    ).items()
                }
            )
        # Index-source layers recompute their own selection. Other layers reuse
        # the latest selection of their ratio, which metadata init would discard.
        if next_layer not in config.index_source_layer_ids:
            core = metadata.core_metadata
            raw = core.sparse_raw_indices(ratio)
            if raw is None:
                raw = logical_selection(
                    core.sparse_page_indices(ratio),
                    batch,
                    req_to_token,
                    pool.page_size,
                    ratio,
                )
            tensors["vpp_index_ratio"] = ratio
            tensors["vpp_index_positions"] = raw
            tensors["vpp_index_lengths"] = core.sparse_topk_lengths(ratio)
    if 0 <= config.candidate_source_layer_id < next_layer:
        _export_candidates(tensors, metadata.candidate_metadata)


def install_state(tensors, *, batch, pool, req_to_token, metadata):
    source = tensors.get("vpp_source_layer")
    if source is not None:
        pool.install_source_pages(
            source,
            full_page_ids(batch, req_to_token, pool.page_size),
            {
                name: tensors[f"vpp_source_{name}"]
                for name in ("kv", "index", "index_payload", "index_scale")
                if f"vpp_source_{name}" in tensors
            },
        )
    ratio = tensors.get("vpp_index_ratio")
    if ratio is not None:
        raw = tensors["vpp_index_positions"]
        metadata.core_metadata.set_sparse_topk(
            ratio,
            page_indices=physical_selection(raw, batch, req_to_token, ratio),
            topk_lengths=tensors["vpp_index_lengths"],
            raw_indices=raw if not batch.forward_mode.is_decode() else None,
        )
    if "vpp_candidate_kind" in tensors:
        metadata.candidate_metadata = _install_candidates(
            tensors, batch, pool, req_to_token
        )


def _export_candidates(tensors, candidate):
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
    from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import (
        _SparsePrefillTable,
        _SparseTable,
    )

    if candidate is None:
        return
    if isinstance(candidate, BlockIds):
        tensors["vpp_candidate_kind"] = "blocks"
        tensors["vpp_candidate_blocks"] = candidate.blocks
        tensors["vpp_candidate_rows"] = candidate.rows_per_request
        if candidate.decode_mask is not None:
            tensors["vpp_candidate_mask"] = candidate.decode_mask
    elif isinstance(candidate, _SparseTable):
        # The source publishes on a side stream. Join it before the scheduler
        # snapshots these tensors and records the stage's producer-ready event.
        torch.cuda.current_stream().wait_event(candidate.ready)
        tensors["vpp_candidate_kind"] = "sparse"
        tensors["vpp_candidate_blocks"] = candidate.blocks
        tensors["vpp_candidate_lengths"] = candidate.valid_lens
        if isinstance(candidate, _SparsePrefillTable):
            tensors["vpp_candidate_compress_lens"] = candidate.compress_lens
            tensors["vpp_candidate_request_ids"] = candidate.request_ids
            tensors["vpp_candidate_rows"] = candidate.rows_per_request
            tensors["vpp_candidate_q_dtype"] = str(candidate.q_dtype).split(".")[-1]
    else:
        raise TypeError(f"Unsupported VPP candidate metadata: {type(candidate)}")


def _install_candidates(tensors, batch, pool, req_to_token):
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds

    blocks = tensors["vpp_candidate_blocks"]
    if tensors["vpp_candidate_kind"] == "blocks":
        return BlockIds(
            blocks=blocks,
            rows_per_request=tensors["vpp_candidate_rows"],
            decode_mask=tensors.get("vpp_candidate_mask"),
        )
    from sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table import (
        _build_prefill_table,
    )

    # Candidate filtering is ratio-1. Rebuild the physical block ids and the
    # DeepGEMM schedule against the receiver's index-page allocation.
    index_page_size = pool.get_index_k_page_size(1)
    width = (
        max(int(n) for n in batch.seq_lens_cpu) + index_page_size - 1
    ) // index_page_size
    table = (
        req_to_token[
            request_rows(batch), : width * index_page_size : index_page_size
        ].to(torch.int32)
        // index_page_size
    )
    prefill = not batch.forward_mode.is_decode()
    return _build_prefill_table(
        blocks=blocks,
        compress_lens=tensors["vpp_candidate_compress_lens"]
        if prefill
        else batch.seq_lens.to(torch.int32),
        page_table=table.contiguous(),
        page_size=index_page_size,
        request_ids=tensors["vpp_candidate_request_ids"]
        if prefill
        else torch.arange(blocks.shape[0], device=blocks.device, dtype=torch.int32),
        rows_per_request=tensors.get("vpp_candidate_rows"),
        q_dtype=getattr(torch, tensors.get("vpp_candidate_q_dtype", "uint8")),
        valid_lens=tensors["vpp_candidate_lengths"],
    )
