"""Deterministic DSA selection in logical slot order (FP8 GLM)."""

import torch


def ordered_dsa_topk(logits, lengths, topk=2048, row_starts=None):
    import flashinfer

    rows, cols = logits.shape
    if topk <= 0:
        raise ValueError("topk must be positive")
    if rows == 0 or cols == 0:
        return torch.full((rows, topk), -1, device=logits.device, dtype=torch.int32)
    # The ragged API reads only each row's valid window and directly emits
    # [0..length) for short rows. Avoid copying/masking the cache-capacity-wide
    # score buffer just to select all of a short request's tokens.
    logical = flashinfer.top_k_ragged_transform(
        logits.contiguous(),
        torch.zeros_like(lengths),
        lengths,
        topk,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=True,
        row_starts=row_starts,
    )
    return _order_indices(logical, topk)


@torch.compile(dynamic=True)
def _order_indices(logical, topk):
    """Fuse invalid-slot masking, logical order and output padding."""
    result = torch.full(
        (logical.shape[0], topk), -1, device=logical.device, dtype=torch.int32
    )
    pad = torch.iinfo(torch.int32).max
    keys = torch.where(logical >= 0, logical, pad).sort(dim=-1).values
    result[:, : keys.shape[1]] = torch.where(keys == pad, -1, keys)
    return result
