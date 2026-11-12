"""Graph-stable row metadata for chain MTP over host-offloaded KV."""

from __future__ import annotations


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
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata import (
        mtp_rows,
    )

    _, query_width, _ = mtp_row_geometry(forward_batch, num_tokens)
    return mtp_rows(forward_batch, query_width, request_rows, context_len)
