"""The candidate lattice: top-k plus normalized log-probs over the target head.

One implementation, shared by the eager draft seam and the graph-folded sampler.
They differ only in how the logits buffer is supplied -- the folded path passes a
preallocated ``logits_out`` so that no large allocation lands in a CUDA graph's
private pool, while the eager path chunks to cap the same buffer -- and every
operation here is per-row, so neither choice can change a value.

The head scores log-probs normalized over the full vocabulary, so the
log-partition is part of the contract: returning raw top-k logits would score a
different function.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.lilicorr import lilicorr_topk_lse
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding


def resolve_vocab_shard(lm_head) -> Tuple[int, int]:
    """``(num_org, org_vocab_start)`` for this rank's slice of the target head."""
    if not isinstance(lm_head, VocabParallelEmbedding):
        # Not vocab-parallel, so this rank owns the whole vocabulary from 0.
        return int(lm_head.weight.shape[0]), 0
    shard = lm_head.shard_indices
    if int(shard.num_added_elements) != 0:
        raise NotImplementedError(
            "LiLiCorr's candidate head does not support added vocabulary: the "
            "added rows sit past the padded base shard, so a single contiguous "
            "top-k would silently skip them."
        )
    return int(shard.num_org_elements), int(shard.org_vocab_start_index)


def target_input_embeddings(target_model):
    """The target model's input embedding table.

    Deliberately not the worker's ``_resolve_dflash_embedding_module``, which
    returns the draft's own table for Nemotron-3.5 drafts: the head must embed
    candidate ids with the table it was trained against, and the draft's would
    load, run, and score the wrong function.
    """
    embed = target_model.get_input_embeddings()
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

    Returns ``(log_probs [N, topk] fp32, tokens [N, topk] int64)``. Equivalent to
    ``log_softmax(logits).topk(topk)`` without materializing the full vocabulary.

    ``tp_group`` is required when the head is vocabulary-sharded, to combine the
    per-rank top-k and partition into global ones.
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
        vals, tokens, lse = lilicorr_topk_lse(logits, topk)
        tokens = tokens + org_vocab_start
        if tp_size > 1:
            vals, tokens, lse = _combine_across_ranks(
                vals=vals, tokens=tokens, lse=lse, topk=topk, tp_group=tp_group
            )
        return vals - lse.unsqueeze(-1), tokens

    # The folded path is one span by construction and is the captured body, so it
    # returns directly: staging through a second pair of buffers would put two
    # more allocations and two more copies inside the graph for nothing.
    if logits_out is not None:
        return one_span(hidden_states, logits_out)

    device = hidden_states.device
    out_vals = torch.empty((num_rows, topk), dtype=torch.float32, device=device)
    out_tokens = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for start in range(0, num_rows, int(chunk_size)):
        end = min(num_rows, start + int(chunk_size))
        vals, tokens = one_span(hidden_states[start:end], None)
        out_vals[start:end] = vals
        out_tokens[start:end] = tokens
    return out_vals, out_tokens


def _combine_across_ranks(
    *,
    vals: torch.Tensor,
    tokens: torch.Tensor,
    lse: torch.Tensor,
    topk: int,
    tp_group,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Three per-step collectives fused into one all-gather by packing
    # [vals | tokens as fp32 | lse] per row; ids below 2**24 are exact in fp32,
    # which every vocabulary in use satisfies.
    tp_size = int(tp_group.world_size)
    rows = int(vals.shape[0])
    width = 2 * topk + 1
    packed = torch.empty((rows, width), dtype=torch.float32, device=vals.device)
    packed[:, :topk] = vals.float()
    packed[:, topk : 2 * topk] = tokens.to(torch.float32)
    packed[:, 2 * topk] = lse.float()

    gathered = torch.empty(
        tp_size * rows * width, dtype=torch.float32, device=vals.device
    )
    tp_group.all_gather_into_tensor(gathered, packed.contiguous().view(-1))
    gathered = gathered.view(tp_size, rows, width)

    all_vals = gathered[:, :, :topk].permute(1, 0, 2).reshape(rows, tp_size * topk)
    all_tokens = (
        gathered[:, :, topk : 2 * topk]
        .permute(1, 0, 2)
        .reshape(rows, tp_size * topk)
        .round()
        .to(torch.int64)
    )
    top_vals, top_idx = torch.topk(all_vals, topk, dim=-1)
    return (
        top_vals,
        torch.gather(all_tokens, 1, top_idx),
        torch.logsumexp(gathered[:, :, 2 * topk], dim=0),
    )


def per_request_last_row(
    *,
    num_rows: int,
    extend_lens: Optional[torch.Tensor],
    commit_lens: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Index of each request's last committed row, or None if not recoverable.

    The two callers hand over two different layouts, and the distinction is the
    whole content of this function:

    * **verify** passes ``commit_lens`` against a buffer padded to ``[bs,
      block_size]`` and flattened, so requests sit at a constant stride and only
      the first ``commit_lens[i]`` rows of each are live;
    * **prefill/extend** passes ``extend_lens`` against a packed, request-major
      buffer, so requests are ragged and back to back.
    """
    if commit_lens is not None:
        bs = int(commit_lens.shape[0])
        if bs == 0 or num_rows % bs != 0:
            return None
        # Padded: request i occupies rows [i*stride, i*stride + commit_lens[i]).
        stride = num_rows // bs
        base = torch.arange(bs, device=commit_lens.device, dtype=torch.int64) * stride
        return (base + commit_lens.to(torch.int64) - 1).clamp_min(0)
    if extend_lens is None or extend_lens.numel() == 0:
        return None
    lens = extend_lens.to(torch.int64).flatten()
    if int(lens.sum()) != int(num_rows):
        return None
    # Packed: cumulative lengths land on each request's last row.
    return (torch.cumsum(lens, dim=0) - 1).clamp_min(0)


def publish_anchor(
    *,
    draft_sampler,
    ctx_hidden: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    commit_lens: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Each request's last committed context row, as the head's anchor.

    ``ctx_hidden`` is the fc-projected target context the caller already computed
    for the KV write, so the anchor costs one index_select rather than a second
    projection. Unrecoverable boundaries return None, which the head scores as
    "no anchor"; fabricating one would be a silent acceptance regression.
    """
    ends = per_request_last_row(
        num_rows=int(ctx_hidden.shape[0]),
        extend_lens=extend_lens,
        commit_lens=commit_lens,
    )
    anchor = None if ends is None else ctx_hidden.index_select(0, ends)
    if draft_sampler is not None:
        draft_sampler.set_anchor(anchor, 0 if anchor is None else int(anchor.shape[0]))
    return anchor
