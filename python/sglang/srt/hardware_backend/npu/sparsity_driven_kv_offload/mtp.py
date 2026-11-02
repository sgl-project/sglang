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


def prepare_mtp_rows(forward_batch, num_tokens, request_rows, context_len):
    """Return [B,Q] request IDs, absolute positions and live-query masks.

    seq_lens is the prefix length in draft/verify, whereas positions already
    includes the draft step. Draft-extend has a fixed Q stride but only its
    num_accept_tokens leading columns contain KV to publish.
    """
    requests = forward_batch.req_pool_indices.long()
    batch = requests.numel()
    width = getattr(forward_batch.spec_info, "num_tokens_per_req", -1)
    if not (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    ):
        width = 1
    if width <= 0:
        if batch == 0 or num_tokens % batch:
            raise ValueError("Sparse MTP requires a fixed query width per request")
        width = num_tokens // batch
    logical_tokens = batch * width
    if batch == 0 or logical_tokens > num_tokens:
        raise ValueError("Sparse MTP query window exceeds the padded token rows")
    if forward_batch.positions.numel() != num_tokens:
        raise ValueError("Sparse MTP requires one absolute position per query")
    requests = requests[:, None].expand(batch, width)
    # TP may append token rows independently of request padding (e.g. 2*3
    # queries padded to 8). Those rows are not another column of each request.
    positions = forward_batch.positions[:logical_tokens].long().view(batch, width)
    valid = (
        (requests > 0)
        & (requests < request_rows)
        & (forward_batch.seq_lens[:, None] > 0)
        & (positions >= 0)
        & (positions < context_len)
    )
    if forward_batch.forward_mode.is_draft_extend_v2():
        counts = forward_batch.spec_info.num_accept_tokens
        valid = valid & (
            torch.arange(width, device=positions.device)[None] < counts[:, None]
        )
    if forward_batch.out_cache_loc is not None:
        valid = valid & (
            forward_batch.out_cache_loc[:logical_tokens].view(batch, width) >= 0
        )
    return requests, positions, valid


def validate_mtp_chain(forward_batch, positions, live, request_rows):
    """Optionally assert the chain and KV-location contract on device."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if positions.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError(
            "Native MTP async validation is unsupported during NPU graph capture; "
            "disable graph capture for validation."
        )
    from sglang.srt.utils.async_probe import maybe_assert_async

    requests = forward_batch.req_pool_indices
    active = forward_batch.seq_lens > 0
    valid_request = (requests > 0) & (requests < request_rows)
    contiguous = (positions[:, 1:] == positions[:, :-1] + 1).all(dim=1)
    maybe_assert_async(
        ((~active) | (valid_request & live.all(dim=1) & contiguous)).all(),
        "Native MTP requires complete contiguous live query rows.",
    )
