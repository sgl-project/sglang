"""Graph-stable row metadata for chain MTP over host-offloaded KV."""

from __future__ import annotations

import torch


def is_sparse_mtp_batch(forward_batch) -> bool:
    mode = forward_batch.forward_mode
    return (
        mode.is_target_verify()
        or mode.is_draft_extend_v2()
        or (mode.is_decode() and getattr(forward_batch, "spec_info", None) is not None)
    )


def use_native_mtp_verify(backend, forward_batch, manager):
    """Select the opt-in long Q4 verify path; short rows use the reference."""
    if not (
        getattr(manager.lim_adapter, "mtp_full_visible", False)
        and forward_batch.forward_mode.is_target_verify()
        and getattr(forward_batch.spec_info, "num_tokens_per_req", -1) == 4
    ):
        return False
    if backend.graph_mode:
        # Graph replay eligibility is checked against CPU lengths by the runner.
        return True
    lengths = forward_batch.seq_lens_cpu
    return lengths is not None and all(
        int(length) == 0 or int(length) >= manager.cache_capacity
        for length in lengths.tolist()
    )


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
    """Optionally assert the Q4 chain and KV-location contract on device."""
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
        "Native MTP requires complete contiguous Q4 live rows.",
    )


def mtp_topk_rows(sources, num_tokens):
    """Pad token rows without reinterpreting missing rows as a smaller K."""
    if sources.shape[0] > num_tokens:
        raise ValueError("Sparse MTP indexer has more rows than the query")
    return torch.nn.functional.pad(
        sources, (0, 0, 0, num_tokens - sources.shape[0]), value=-1
    )


def mtp_topk_mask(sources, positions, valid_rows):
    """IndexShare and padding must never expose a future/rejected KV row."""
    return valid_rows[:, None] & (sources >= 0) & (sources <= positions[:, None])


def mtp_compact_topk(sources, positions, valid_rows):
    """Move invalid entries to the suffix, preserving vendor reduction order."""
    valid = mtp_topk_mask(sources, positions, valid_rows)
    sources = sources.masked_fill(~valid, -1)
    columns = torch.arange(
        sources.shape[-1], device=sources.device, dtype=torch.float32
    )
    # Unique column keys preserve the original order without relying on a
    # stable-sort implementation. Reordering valid KV changes BF16 reductions.
    # K<=2048 makes these keys exact in FP32 and keeps sorting on AiCore.
    order = torch.where(valid, columns, columns + sources.shape[-1]).argsort(dim=-1)
    return sources.gather(-1, order)


def mtp_selected_kv_view(buffer, batch, top_k):
    """Pack short TopK rows into the contiguous prefix of shared scratch.

    Slicing the capacity dimension retains its full row stride, while native
    KV copy requires packed rows even when a short prompt selects fewer keys.
    This view aliases the same allocation without reserving more HBM.
    """
    tail_shape = buffer.shape[2:]
    return buffer.view(-1, *tail_shape)[: batch * top_k].view(batch, top_k, *tail_shape)
