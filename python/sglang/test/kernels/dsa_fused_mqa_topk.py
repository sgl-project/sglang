"""Inputs and a tie-aware fp32 oracle for the fused DSA indexer top-k."""

from typing import Dict, List, Optional

import torch

NUM_HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64


def make_ragged_batch(
    seq_lens: List[int], extend_lens: List[int], *, seed: int, device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """A packed prefill batch laid out like the DSA indexer's gathered KV.

    Request ``i`` owns ``seq_lens[i]`` consecutive keys of the flattened KV and
    its last ``extend_lens[i]`` tokens are query rows; row ``j`` of it sees
    ``[base_i, base_i + seq_lens[i] - extend_lens[i] + j + 1)``, the indexer's
    ``ks`` / ``ke``. Keys carry power-of-two (UE8M0) scales; head gates are signed.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, keys = sum(extend_lens), sum(seq_lens)
    starts, ends, base = [], [], 0
    for seq_len, extend_len in zip(seq_lens, extend_lens, strict=True):
        starts += [base] * extend_len
        ends += range(base + seq_len - extend_len + 1, base + seq_len + 1)
        base += seq_len
    i32 = dict(dtype=torch.int32, device=device)
    return dict(
        q_fp8=torch.randn(rows, NUM_HEADS, HEAD_DIM, generator=gen, device=device).to(
            torch.float8_e4m3fn
        ),
        k_fp8=torch.randn(keys, HEAD_DIM, generator=gen, device=device).to(
            torch.float8_e4m3fn
        ),
        k_scale=torch.exp2(
            torch.randint(-4, 2, (keys,), generator=gen, device=device).float()
        ),
        weights=torch.randn(rows, NUM_HEADS, generator=gen, device=device) / NUM_HEADS,
        starts=torch.tensor(starts, **i32),
        ends=torch.tensor(ends, **i32),
    )


def make_page_table(
    seq_lens: List[int], *, seed: int, device: str = "cuda"
) -> torch.Tensor:
    """A ``req_to_token``-like page_size=1 table for a packed batch.

    Token ``p`` of request ``i`` sits in KV slot ``page * 64 + p % 64`` of a pool
    page drawn without replacement, so requests own disjoint slots. Columns past
    ``seq_lens[i]`` hold slots of a page no request owns, and the table is a
    column slice of a wider pool, so its row stride exceeds its width.
    """
    gen = torch.Generator().manual_seed(seed)
    num_pages = [(seq_len + PAGE_SIZE - 1) // PAGE_SIZE for seq_len in seq_lens]
    pages = torch.randperm(sum(num_pages) + 1, generator=gen).to(torch.int32)
    width = max(seq_lens)
    cols = torch.arange(width + PAGE_SIZE, dtype=torch.int32)
    pool = (pages[-1] * PAGE_SIZE + cols % PAGE_SIZE).repeat(len(seq_lens), 1)
    first = 0
    for i, (seq_len, n) in enumerate(zip(seq_lens, num_pages, strict=True)):
        own, pos = pages[first : first + n], cols[:seq_len]
        pool[i, :seq_len] = own[pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE
        first += n
    return pool.to(device)[:, :width]


def row_requests(extend_lens: List[int], device: str = "cuda") -> torch.Tensor:
    """``[rows]`` int32 request of each query row, the indexer's token_to_batch_idx."""
    return torch.repeat_interleave(
        torch.arange(len(extend_lens), dtype=torch.int32),
        torch.tensor(extend_lens),
    ).to(device)


def paged_to_relative(
    indices: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: List[int],
    row_batch: torch.Tensor,
) -> torch.Tensor:
    """Row-relative key indices of PAGED ``indices`` (KV slots, negative entries
    kept), asserting every slot holds a token of the row's own request."""
    num_slots = int(page_table.max()) + 1
    slot_request = torch.full((num_slots,), -1, dtype=torch.long, device=indices.device)
    slot_position = torch.full_like(slot_request, -1)
    for i, seq_len in enumerate(seq_lens):
        slots = page_table[i, :seq_len].long()
        slot_request[slots] = i
        slot_position[slots] = torch.arange(seq_len, device=indices.device)
    valid = indices >= 0
    assert int(indices.max()) < num_slots, "slot outside the KV pool"
    slots = indices.long().clamp(min=0)
    own = slot_request[slots] == row_batch.long()[:, None]
    assert torch.all(own | ~valid), "slot of another request or of no request"
    return torch.where(valid, slot_position[slots], indices.long()).to(torch.int32)


def reference_scores(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """fp32 ``[rows, keys]`` indexer scores, ``-inf`` outside each row's window."""
    q = batch["q_fp8"].float()
    k = batch["k_fp8"].float()
    scores = torch.zeros(q.shape[0], k.shape[0], device=q.device)
    for h in range(q.shape[1]):
        scores += batch["weights"][:, h, None] * torch.relu(q[:, h] @ k.T)
    scores *= batch["k_scale"][None]
    cols = torch.arange(k.shape[0], device=q.device)[None]
    visible = (cols >= batch["starts"][:, None]) & (cols < batch["ends"][:, None])
    return scores.masked_fill(~visible, float("-inf"))


def assert_topk_matches_reference(
    indices: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    scores: torch.Tensor,
    *,
    out_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Check ``indices`` (``key - start + offset``, ``-1`` padded) per row.

    Each row holds exactly ``min(topk, visible keys)`` distinct in-window keys,
    pads the rest with ``-1``, and every selected key scores at least the exact
    k-th score up to a 2e-4 relative plus 1e-4 absolute tolerance, so key sets
    may differ only among (near-)ties at the boundary.
    """
    topk = indices.shape[1]
    starts, ends = batch["starts"].long(), batch["ends"].long()
    valid = indices >= 0
    expected = (ends - starts).clamp(max=topk)
    assert torch.equal(valid.sum(1), expected), "selected count != min(topk, visible)"
    assert torch.all(indices[~valid] == -1), "padding must be -1"
    offsets = 0 if out_offsets is None else out_offsets.long()[:, None]
    keys = indices.long() - offsets + starts[:, None]
    in_window = (keys >= starts[:, None]) & (keys < ends[:, None])
    assert torch.all(in_window | ~valid), "selected key outside the row's window"
    ordered = keys.masked_fill(~valid, -1).sort(dim=1).values
    assert torch.all((ordered[:, 1:] < 0) | (ordered.diff(dim=1) > 0)), "duplicates"
    exact = scores.topk(min(topk, scores.shape[1]), dim=1).values
    kth = exact.gather(1, (expected - 1).clamp(min=0)[:, None])
    picked = scores.gather(1, keys.clamp(min=0)).masked_fill(~valid, float("inf"))
    tol = kth.abs() * 2e-4 + 1e-4
    short = (picked < kth - tol).any(dim=1)
    assert not short.any(), f"rows below the k-th score: {short.nonzero().flatten()}"
