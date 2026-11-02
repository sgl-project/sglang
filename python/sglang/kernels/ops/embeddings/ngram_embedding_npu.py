"""Adapt the existing Ascend OE operators to the shared ngram interface."""

import torch


def _npu_ops():
    import sgl_kernel_npu  # noqa: F401

    return torch.ops.npu


def _compute_windows(
    ne_n,
    ne_k,
    ne_weights,
    ne_mods,
    embedder_offsets,
    token_table,
    rows,
    positions,
    valid,
    output,
    eos_token_id,
):
    # The existing operator stops at negative tokens but has no EOS argument.
    # Give each query its short context, masking only *previous* EOS tokens.
    # Its request metadata is staged in UB, so bound the synthetic batch size.
    ops = _npu_ops()
    lookback = torch.arange(1 - ne_n, 1, device=token_table.device)
    for start in range(0, output.shape[0], 1024):
        end = min(start + 1024, output.shape[0])
        batch_size = end - start
        # Active rows/positions follow the shared token-table contract. Only
        # graph padding and lookback before the request start need adjustment.
        chunk_positions = torch.where(valid[start:end], positions[start:end], 0)
        cols = chunk_positions[:, None] + lookback
        context = token_table[rows[start:end, None], cols.clamp_min(0)]
        context = context.masked_fill(
            (cols < 0) | ((context == eos_token_id) & (lookback != 0)), -1
        ).contiguous()
        result = ops.compute_n_gram_ids(
            ne_weights,
            ne_mods,
            embedder_offsets,
            context[:, -1].contiguous(),
            torch.arange(1, batch_size + 1, dtype=torch.int32, device=context.device),
            context,
            torch.arange(batch_size, dtype=torch.int64, device=context.device),
            torch.full(
                (batch_size,), ne_n - 1, dtype=torch.int32, device=context.device
            ),
            batch_size=batch_size,
            oe_n=ne_n,
            oe_k=ne_k,
            max_context_len=ne_n,
        )
        # Supply a deterministic embedding index for shared graph-padding rows
        # beyond the valid query ends.
        output[start:end].copy_(result.masked_fill(~valid[start:end, None], 0))


def compute_n_gram_ids(
    ne_n,
    ne_k,
    ne_weights,
    ne_mods,
    exclusive_ne_embedder_size_sums,
    tokens,
    exclusive_req_len_sums,
    ne_token_table,
    row_indices,
    column_starts,
    n_gram_ids,
    eos_token_id,
):
    if tokens.numel() == 0:
        return
    batch_size = row_indices.numel()
    if batch_size == 0:
        n_gram_ids.zero_()
        return
    offsets = torch.arange(tokens.numel(), device=tokens.device)
    requests = torch.searchsorted(exclusive_req_len_sums[1:], offsets, right=True)
    valid = requests < batch_size
    requests = requests.clamp_max(batch_size - 1)
    positions = column_starts[requests] + offsets - exclusive_req_len_sums[requests]
    _compute_windows(
        ne_n,
        ne_k,
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        ne_token_table,
        row_indices[requests],
        positions,
        valid,
        n_gram_ids,
        eos_token_id,
    )


def compute_n_gram_ids_decode(
    ne_n,
    ne_k,
    ne_weights,
    ne_mods,
    exclusive_ne_embedder_size_sums,
    ne_token_table,
    row_indices,
    column_starts,
    n_gram_ids,
    eos_token_id,
):
    if row_indices.numel() == 0:
        return
    _compute_windows(
        ne_n,
        ne_k,
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        ne_token_table,
        row_indices,
        column_starts,
        torch.ones_like(row_indices, dtype=torch.bool),
        n_gram_ids,
        eos_token_id,
    )


def update_token_table(
    tokens, ne_token_table, row_indices, column_starts, req_lens, ignore_tokens
):
    if row_indices.numel() == 0 or tokens.numel() == 0:
        return
    # CUDA stores -token for ignored tokens (including -0 == 0); the old NPU
    # operator writes -1. Encode before calling it to preserve the shared table.
    encoded = tokens.to(torch.int32)
    for index in range(ignore_tokens.numel()):
        encoded = torch.where(tokens == ignore_tokens[index], -tokens, encoded).to(
            torch.int32
        )
    _npu_ops().update_oe_token_table(
        encoded.contiguous(),
        req_lens.to(torch.int32).contiguous(),
        row_indices.to(torch.int64).contiguous(),
        column_starts.to(torch.int32).contiguous(),
        encoded.new_empty(0),
        batch_size=row_indices.numel(),
        max_context_len=ne_token_table.shape[1],
        oe_token_table=ne_token_table,
    )


def update_token_table_decode(tokens, ne_token_table, row_indices, column_starts):
    update_token_table(
        tokens,
        ne_token_table,
        row_indices,
        column_starts,
        torch.ones_like(column_starts, dtype=torch.int32),
        tokens.new_empty(0),
    )
