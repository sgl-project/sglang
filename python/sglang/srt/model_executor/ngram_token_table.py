"""Utilities for updating LongCat ngram embedding token tables."""

from __future__ import annotations

import torch

from sglang.jit_kernel.ngram_embedding import update_token_table


def update_ngram_token_table_after_sampling(
    *,
    ngram_embedding_info,
    next_token_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
) -> bool:
    """Update the ngram token table with sampled tokens.

    Returns whether the token table was updated.
    """
    skip_token_table_update = ngram_embedding_info.skip_token_table_update
    if skip_token_table_update is not None:
        # Skip chunked (not-yet-finished) prefill requests: their sampled token
        # is a pseudo prediction and must not pollute the token table.
        indices = (~skip_token_table_update).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return False
        update_token_table(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids[indices].to(torch.int32),
            row_indices=req_pool_indices[indices],
            column_starts=seq_lens[indices].to(torch.int32),
            req_lens=torch.ones(
                indices.numel(), dtype=torch.int32, device=next_token_ids.device
            ),
            ignore_tokens=None,
        )
        return True

    ngram_embedding_info.out_column_starts[:batch_size] = seq_lens
    ngram_embedding_info.out_req_lens[:batch_size] = 1
    update_token_table(
        ne_token_table=ngram_embedding_info.token_table,
        tokens=next_token_ids.to(torch.int32),
        row_indices=req_pool_indices,
        column_starts=ngram_embedding_info.out_column_starts,
        req_lens=ngram_embedding_info.out_req_lens,
        ignore_tokens=None,
    )
    return True
