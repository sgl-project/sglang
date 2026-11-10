"""Graph-stable row metadata for chain MTP over host-offloaded KV."""

from __future__ import annotations

import torch


def sparse_query_width(forward_batch):
    """Read the actual chain width; decode always has one query per row."""
    if forward_batch.forward_mode.is_decode():
        return 1
    width = getattr(forward_batch.spec_info, "num_tokens_per_req", None)
    if not isinstance(width, int) or width <= 0:
        raise ValueError("Sparse target verify requires its actual chain query width.")
    return width


def mtp_row_geometry(forward_batch, num_tokens):
    """Validate the fixed request window independently of TP suffix padding."""
    batch = forward_batch.req_pool_indices.numel()
    query_width = getattr(forward_batch.spec_info, "num_tokens_per_req", -1)
    if not (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    ):
        query_width = 1
    if query_width <= 0:
        if batch == 0 or num_tokens % batch:
            raise ValueError("Sparse MTP requires a fixed query width per request")
        query_width = num_tokens // batch
    logical_tokens = batch * query_width
    if batch == 0 or logical_tokens > num_tokens:
        raise ValueError("Sparse MTP query window exceeds the padded token rows")
    if forward_batch.positions.numel() != num_tokens:
        raise ValueError("Sparse MTP requires one absolute position per query")
    return batch, query_width, logical_tokens


def prepare_mtp_rows(forward_batch, num_tokens, request_rows, context_len):
    """Return [B,Q] request IDs, absolute positions and live-query masks.

    seq_lens is the prefix length in draft/verify, whereas positions already
    includes the draft step. Draft-extend has a fixed Q stride but only its
    num_accept_tokens leading columns contain KV to publish.
    """
    batch, query_width, logical_tokens = mtp_row_geometry(forward_batch, num_tokens)
    token_to_slots = forward_batch.req_pool_indices
    if token_to_slots.device.type == "npu":
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata import (
            mtp_rows,
        )

        return mtp_rows(forward_batch, query_width, request_rows, context_len)
    token_to_slots = token_to_slots.long()
    token_to_slots = token_to_slots[:, None].expand(batch, query_width)
    # TP may append token rows independently of request padding (e.g. 2*3
    # queries padded to 8). Those rows are not another column of each request.
    positions = forward_batch.positions[:logical_tokens].long().view(batch, query_width)
    valid = (
        (token_to_slots > 0)
        & (token_to_slots < request_rows)
        & (forward_batch.seq_lens[:, None] > 0)
        & (positions >= 0)
        & (positions < context_len)
    )
    if forward_batch.forward_mode.is_draft_extend_v2():
        counts = forward_batch.spec_info.num_accept_tokens
        valid = valid & (
            torch.arange(query_width, device=positions.device)[None] < counts[:, None]
        )
    if forward_batch.out_cache_loc is not None:
        valid = valid & (
            forward_batch.out_cache_loc[:logical_tokens].view(batch, query_width) >= 0
        )
    return token_to_slots, positions, valid
