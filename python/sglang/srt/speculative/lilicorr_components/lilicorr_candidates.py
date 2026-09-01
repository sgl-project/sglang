"""The candidate lattice: top-k plus normalized log-probs over the target head.

One implementation, shared by the eager draft seam and the graph-folded sampler.
They differ only in how the logits buffer is supplied -- the folded path passes a
preallocated ``logits_out`` so that no large allocation lands in a CUDA graph's
private pool, while the eager path chunks to cap the same buffer -- and every
operation here is per-row, so neither choice can change a value.

The head scores log-probs normalized over the **full** vocabulary, so the
log-partition is part of the contract, not an afterthought: returning raw top-k
logits would score a different function.

Alongside the lattice itself, this module resolves the target-side state the
lattice is built from -- the vocabulary shard, the embedding table, and the
anchor row -- so that the DFLASH worker holds no LiLiCorr logic beyond
dispatching to it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.lilicorr import lilicorr_topk_lse


def resolve_vocab_shard(lm_head) -> Tuple[int, int]:
    """``(num_org, org_vocab_start)`` for this rank's slice of the target head.

    A head without ``shard_indices`` is unsharded, so it owns the whole vocabulary
    starting at 0.
    """
    shard = getattr(lm_head, "shard_indices", None)
    if shard is None:
        return int(lm_head.weight.shape[0]), 0
    if int(shard.num_added_elements) != 0:
        raise NotImplementedError(
            "LiLiCorr's candidate head does not support added vocabulary: the "
            "added rows sit past the padded base shard, so a single contiguous "
            "top-k would silently skip them."
        )
    return int(shard.num_org_elements), int(shard.org_vocab_start_index)


def target_input_embeddings(worker):
    """The *target* model's input embedding table.

    Deliberately not the worker's ``_resolve_dflash_embedding_module``: that
    helper returns the **draft's** own table for Nemotron-3.5 drafts and the
    target's otherwise, whereas the head must always embed candidate ids with the
    target's table, because that is the table it was trained against. Taking the
    draft's would load and run, and score the wrong function.
    """
    model = worker.target_worker.model_runner.model
    embed = model.get_input_embeddings()
    if embed is None:
        raise RuntimeError("DFLASH target model exposes no input embeddings.")
    return embed


def lilicorr_candidates(
    *,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    num_org: int,
    org_vocab_start: int,
    topk: int,
    logits_out: Optional[torch.Tensor] = None,
    tp_group=None,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row top-k over the target LM head, as normalized log-probs and global ids.

    Returns ``(log_probs [N, topk] fp32, ids [N, topk] int64)``. Equivalent to
    ``log_softmax(logits).topk(topk)`` without materializing the full vocabulary.

    ``tp_group`` is required when the head is vocabulary-sharded: the per-rank
    top-k and per-rank partition are combined into a global top-k and a global
    partition. The three per-step collectives that needs are fused into one
    all-gather by packing ``[vals | ids as fp32 | lse]`` per row; ids below 2**24
    are exact in fp32, which every vocabulary in use satisfies.
    """
    topk = int(topk)
    if topk > int(num_org):
        raise ValueError(
            f"LiLiCorr candidate topk={topk} exceeds this rank's vocabulary slice "
            f"({num_org} rows), so the lattice cannot be filled."
        )
    tp_size = 1 if tp_group is None else int(tp_group.world_size)
    num_rows = int(hidden_states.shape[0])

    def one_span(rows: torch.Tensor, logits_buf: Optional[torch.Tensor]):
        if rows.dtype != weight.dtype:
            rows = rows.to(weight.dtype)
        if logits_buf is None:
            logits = torch.matmul(rows, weight[:num_org].T)
        else:
            logits = logits_buf
            torch.matmul(rows, weight[:num_org].T, out=logits)
        vals, ids, lse = lilicorr_topk_lse(logits, topk)
        ids = ids + org_vocab_start
        if tp_size > 1:
            vals, ids, lse = _combine_across_ranks(
                vals=vals, ids=ids, lse=lse, topk=topk, tp_group=tp_group
            )
        return vals - lse.unsqueeze(-1), ids

    # The folded path is one span by construction, and it is the captured body,
    # so it returns directly: staging through a second pair of buffers would put
    # two more allocations and two more copies inside the graph for nothing.
    if logits_out is not None:
        return one_span(hidden_states, logits_out)

    device = hidden_states.device
    out_vals = torch.empty((num_rows, topk), dtype=torch.float32, device=device)
    out_ids = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for start in range(0, num_rows, int(chunk_size)):
        end = min(num_rows, start + int(chunk_size))
        vals, ids = one_span(hidden_states[start:end], None)
        out_vals[start:end] = vals
        out_ids[start:end] = ids
    return out_vals, out_ids


def _combine_across_ranks(
    *,
    vals: torch.Tensor,
    ids: torch.Tensor,
    lse: torch.Tensor,
    topk: int,
    tp_group,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp_size = int(tp_group.world_size)
    rows = int(vals.shape[0])
    width = 2 * topk + 1
    packed = torch.empty((rows, width), dtype=torch.float32, device=vals.device)
    packed[:, :topk] = vals.float()
    packed[:, topk : 2 * topk] = ids.to(torch.float32)
    packed[:, 2 * topk] = lse.float()

    gathered = torch.empty(
        tp_size * rows * width, dtype=torch.float32, device=vals.device
    )
    tp_group.all_gather_into_tensor(gathered, packed.contiguous().view(-1))
    gathered = gathered.view(tp_size, rows, width)

    all_vals = gathered[:, :, :topk].permute(1, 0, 2).reshape(rows, tp_size * topk)
    all_ids = (
        gathered[:, :, topk : 2 * topk]
        .permute(1, 0, 2)
        .reshape(rows, tp_size * topk)
        .round()
        .to(torch.int64)
    )
    top_vals, top_idx = torch.topk(all_vals, topk, dim=-1)
    return (
        top_vals,
        torch.gather(all_ids, 1, top_idx),
        torch.logsumexp(gathered[:, :, 2 * topk], dim=0),
    )


def per_request_last_row(
    *,
    num_rows: int,
    positions: Optional[torch.Tensor],
    commit_lens: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Index of each request's last committed row, or None if not recoverable.

    The DFLASH context append is called from two places and only one supplies the
    per-request lengths: decode and verify pass ``commit_lens``, while
    prefill/extend does not. For prefill the boundaries are still exactly
    recoverable from ``positions``, which is always passed: rows are request-major
    and each request's positions increase strictly, so a request ends wherever the
    next position fails to increase, plus the final row.

    Returning None leaves the anchor unset rather than fabricating one, because a
    wrong anchor is a silent acceptance regression rather than a failure.
    """
    if commit_lens is not None:
        return (torch.cumsum(commit_lens.to(torch.int64), dim=0) - 1).clamp_min(0)
    if positions is None or positions.numel() == 0:
        return None
    pos = positions.to(torch.int64).flatten()
    if int(pos.numel()) != int(num_rows):
        return None
    if int(pos.numel()) == 1:
        return torch.zeros(1, dtype=torch.int64, device=pos.device)
    resets = (pos[1:] <= pos[:-1]).nonzero(as_tuple=False).flatten()
    last = torch.tensor([int(pos.numel()) - 1], dtype=torch.int64, device=pos.device)
    return torch.cat([resets.to(torch.int64), last])


def publish_anchor(
    *,
    draft_sampler,
    ctx_hidden: torch.Tensor,
    positions: Optional[torch.Tensor],
    commit_lens: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Each request's last committed context row, as the head's anchor.

    ``ctx_hidden`` is the fc-projected target context the caller already computed
    for the KV write, so the anchor costs one index_select rather than a second
    projection. When the row boundaries are not recoverable the anchor is None,
    which the head scores as "no anchor"; fabricating one would be a silent
    acceptance regression instead.

    Returned for the eager path and, when a folded sampler exists, also written
    into the fixed address its captured graph reads. A None anchor is published as
    zero rows rather than skipped, so a padded replay cannot read the previous
    step's anchor off that address.
    """
    ends = per_request_last_row(
        num_rows=int(ctx_hidden.shape[0]),
        positions=positions,
        commit_lens=commit_lens,
    )
    anchor = None if ends is None else ctx_hidden.index_select(0, ends)
    if draft_sampler is not None:
        draft_sampler.set_anchor(anchor, 0 if anchor is None else int(anchor.shape[0]))
    return anchor
