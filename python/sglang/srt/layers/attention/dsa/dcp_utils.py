from __future__ import annotations

import torch


def dcp_topk_candidates(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk: int,
    dcp_size: int,
    dcp_rank: int,
    num_init_tokens: int = 0,
    num_local_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select fixed-size candidates with global positions for an exact merge.

    Paged MQA leaves the logits beyond each sequence uninitialized. Mask them
    before selection, including ranks with no visible tokens. Forced sink and
    recent tokens are defined in global sequence coordinates.
    """
    positions = (
        torch.arange(logits.shape[1], device=logits.device) * dcp_size + dcp_rank
    )
    valid = positions[None, :] < seq_lens[:, None]
    forced = (positions[None, :] < num_init_tokens) | (
        positions[None, :] >= seq_lens[:, None] - num_local_tokens
    )
    scores = logits.masked_fill(forced & valid, float("inf"))
    scores = scores.masked_fill(~valid, float("-inf"))
    # All ranks exchange the same number of candidates even when their local
    # context widths differ or are smaller than topk.
    if scores.shape[1] < topk:
        scores = torch.nn.functional.pad(
            scores, (0, topk - scores.shape[1]), value=float("-inf")
        )
    values, indices = scores.topk(topk, dim=-1)
    global_indices = indices * dcp_size + dcp_rank
    global_indices.masked_fill_(values == float("-inf"), -1)
    return values, global_indices


def merge_dcp_topk_candidates(
    scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Return this rank's part of the global top-k, padded with -1.

    Inputs concatenate every rank's local top-k along the last dimension.
    Selecting from this union is exact: a discarded local candidate already
    has at least k better candidates on its own rank.
    """
    winners = scores.topk(topk, dim=-1).indices
    selected = positions.gather(1, winners)
    owned = (selected >= 0) & (selected % dcp_size == dcp_rank)
    local = torch.where(owned, selected // dcp_size, -1)
    # Keep valid entries before padding; attention kernels may use a length
    # bound as well as masking negative indices.
    return local.sort(dim=-1, descending=True).values.to(torch.int32)


def localize_dcp_indexer_write_loc(
    loc: torch.Tensor, *, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    """Map widened DCP token ids to this rank's dense indexer cache rows.

    Non-owned writes are redirected to row zero, the allocator's reserved sink.
    Keeping the output shape unchanged makes this safe for CUDA graph replay.
    """
    if dcp_size == 1:
        return loc
    owned = loc % dcp_size == dcp_rank
    return torch.where(owned, loc // dcp_size, torch.zeros_like(loc))


def localize_dcp_indexer_seq_lens(
    seq_lens: torch.Tensor, *, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    """Count sequence tokens owned by one interleaved DCP rank."""
    if dcp_size == 1:
        return seq_lens
    return torch.clamp((seq_lens + dcp_size - 1 - dcp_rank) // dcp_size, min=0)


def localize_dcp_indexer_page_table(
    page_table: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    max_local_len: int | None = None,
) -> torch.Tensor:
    """Build the rank-local page-size-1 table consumed by the DSA indexer.

    The allocator publishes widened ids and assigns token ``t`` to rank
    ``t % dcp_size``.  DSA index-K and MLA KV both store that rank's rows densely,
    so select the owned sequence positions and collapse their ids by dcp_size.
    """
    if dcp_size == 1:
        return page_table if max_local_len is None else page_table[:, :max_local_len]
    if max_local_len is None:
        max_local_len = max(1, (page_table.shape[1] + dcp_size - 1) // dcp_size)
    stop = dcp_rank + max_local_len * dcp_size
    local = page_table[:, dcp_rank:stop:dcp_size] // dcp_size
    # Keep a valid page-table column even when this rank owns no tokens.
    # The sequence lengths mask padding; paged MQA still needs a nonempty
    # table geometry. Equal widths also keep rank-local capture shapes aligned.
    return torch.nn.functional.pad(local, (0, max_local_len - local.shape[1]))
