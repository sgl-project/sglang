"""Score reduction + top-K selection for Double Sparsity.

The per-(batch, token) selection scores are produced directly off the
resident MLA latent by the absorbed-latent score kernels
(:mod:`absorbed_latent_kernel`). This module owns the stages that follow,
both capture-safe (no host syncs, no dynamic shapes):

1. **Reduce**. The per-rank scores are all-reduced across the attention TP
   group, so per-rank top-K agrees by construction.

2. **Select**. ``select_topk_sequence_order`` consumes the all-reduced scores
   plus the per-slot ``written`` validity mask. Returns ``(selected_indices,
   valid_lengths)`` with ``selected_indices`` in **sequence-order ascending**
   (logical token position order) with ``-1`` padding, per the selector ABI
   contract. The top-K step uses ``torch.topk`` + ``torch.sort`` (both
   CUDA-graph capture-safe with static shapes).

``project_query_onto_channels`` projects a query onto each head's channel
mask; it is the canonical projection the absorbed-latent build mirrors.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


SELECTED_PAD_VALUE = -1
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def project_query_onto_channels(
    queries: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Project ``queries[bs, H, head_dim]`` onto each head's channel mask.

    Returns ``query_projected[bs, H, label_dim]``.
    """

    if queries.dim() != 3:
        raise ValueError(
            f"queries must be 3-D [bs, H, head_dim], got shape {tuple(queries.shape)}."
        )
    if channel_selection.dim() != 2 or channel_weights.dim() != 2:
        raise ValueError(
            "channel_selection/channel_weights must be 2-D [H, label_dim]."
        )
    bs, num_heads, head_dim = queries.shape
    if channel_selection.shape[0] != num_heads:
        raise ValueError(
            f"channel_selection head dim {int(channel_selection.shape[0])} does not match "
            f"queries num_heads {num_heads}."
        )

    selection_idx = channel_selection.long().unsqueeze(0).expand(bs, -1, -1)
    gathered = torch.gather(queries, dim=-1, index=selection_idx)
    return gathered * channel_weights.unsqueeze(0)


def ds_scorer_is_graph_safe(config) -> bool:
    """``True`` iff the configured selector variants are all on the graph-safe
    path, so the selector can run under CUDA-graph capture.

    All non-learned variants are graph-safe: ``head_agg`` (mean) lives in the
    absorbed paged score kernel, and ``anchor_mode`` (recency/global/strided) is
    a tensorized fixed-shape post-topK force-include in
    ``retrieve_topk_graph_safe``. None require ``--disable-cuda-graph``. (The
    ``recall_oracle`` diagnostic is gated separately by
    ``ds_recall_oracle_enabled``.) Retained as the single guard predicate so a
    future non-graph-safe variant can re-introduce a gate here.
    """
    return True


def ds_recall_oracle_enabled(config) -> bool:
    """``True`` iff the config-borne recall-oracle diagnostic is on.

    Config-borne (not env) so it reaches TP worker subprocesses
    (BL-20260602-ds-flag-must-be-config-borne-not-env). Like a non-default
    scorer it forces the eager selector path so the host-syncing oracle hook
    actually re-runs every decode step (under CUDA-graph replay the Python does
    not re-run); the validator additionally requires ``--disable-cuda-graph``.
    """
    if config is None:
        return False
    return bool(getattr(config, "recall_oracle", False))


def ds_lifted_budget_decode_available() -> bool:
    """``True`` iff the opt-in adjustable-budget (lifted) decode backend path is
    implemented and wired into selection/decode, so ``enable_lifted_budget_decode``
    can actually be honored.

    Returns ``True``: the selector widens its budget to ``lifted_budget_top_k`` and
    the decode routes the selected slots through the request-local compact remap →
    ``dequantize_k_cache_paged`` (eager) / the fixed-shape graph-safe
    ``build_lifted_compact_kv_fixed`` + ``dequantize_k_cache_paged_out`` into a
    preallocated ``DSGraphState`` scratch (under CUDA-graph capture) →
    ``flash_mla_sparse_fwd``. The path is **CUDA-graph-safe** (proven zero-alloc
    replay; the validator no longer requires ``--disable-cuda-graph``). The
    validator's lifted-budget shape checks govern enablement now that this seam is
    open (mirroring :func:`ds_scorer_is_graph_safe`).
    """
    return True


_score_reduce_fallback_logged = False

# Transport evidence: one log line per distinct (shape, dtype, path, algorithm)
# score-reduce bucket, emitted from the host-side reduce call (capture/eager —
# graph replay re-runs the captured kernels, not this Python).
_score_reduce_buckets_logged: set = set()


def _log_score_reduce_bucket(
    view: torch.Tensor, custom_ar: bool, algorithm: str
) -> None:
    key = (tuple(view.shape), str(view.dtype), custom_ar, algorithm)
    if key in _score_reduce_buckets_logged:
        return
    _score_reduce_buckets_logged.add(key)
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_weak_contiguous,
    )

    logger.info(
        "double_sparsity score reduce bucket: shape=%s dtype=%s bytes=%d "
        "weak_contiguous=%s custom_ar=%s algorithm=%s",
        tuple(view.shape),
        view.dtype,
        view.numel() * view.element_size(),
        bool(is_weak_contiguous(view)),
        custom_ar,
        algorithm,
    )


class PinnedDSScoreReduceCA:
    """Custom-AR wrapper that pins the DS score reduce to one algorithm.

    Floating-point summation order is part of the DS selection exactness
    contract: CustomAllReduceV2's size-based algorithm selection would
    silently flip small compact score buffers (<=160 KB on 8 ranks) to
    one-shot, changing the summation order relative to the served two-shot
    path. This wrapper passes a per-call override so ONLY the DS score reduce
    is pinned — the wrapped communicator object and every default model
    collective keep their size-based behavior.

    ``should_custom_ar`` additionally REFUSES (raises on) a non-weak-contiguous
    input instead of letting the eligibility check route it to NCCL: a strided
    view handed to the reduce means a compact scratch buffer was sliced out of
    a wider allocation, and a silent transport change is forbidden while the
    pin is in force.
    """

    pinned_algo_name = "TWO_SHOT_PULL"

    def __init__(self, base_ca):
        from sglang.jit_kernel.all_reduce import AllReduceAlgo

        self.base_ca = base_ca
        self.pinned_algo = AllReduceAlgo.TWO_SHOT_PULL

    @property
    def disabled(self) -> bool:
        return bool(getattr(self.base_ca, "disabled", False))

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
            is_weak_contiguous,
        )

        if not is_weak_contiguous(inp):
            raise AssertionError(
                "double_sparsity score reduce: bf16 tensor "
                f"shape={tuple(inp.shape)} strides={tuple(inp.stride())} is not "
                "weak-contiguous. Compact selector scratch must be a real "
                "allocation, not a strided view of a wider buffer — a strided "
                "input would silently fall back to NCCL, which the pinned "
                "transport contract forbids."
            )
        return self.base_ca.should_custom_ar(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self.base_ca.custom_all_reduce(inp, override_algo=self.pinned_algo)


def reduce_token_scores(
    token_scores: torch.Tensor,
    *,
    process_group=None,
    reduce_ca=None,
    bf16_scratch: Optional[torch.Tensor] = None,
    use_bf16: bool = False,
    copy_back: bool = True,
) -> torch.Tensor:
    """SUM-reduce per-rank partial token scores across the attention TP group.

    The ONE reduce shared by the eager and graph-safe selection paths. Token
    label signatures stay TP/head-sharded, so per-rank scores are partial and
    the SUM makes every rank's selection identical by construction. Operates
    in place on ``token_scores`` and returns it.

    ``use_bf16`` (score_reduce_dtype="bf16", the served default): the fp32
    scores are cast into a bf16 view (the preallocated ``bf16_scratch`` on the
    graph-safe path; a dynamic cast on the eager path), reduced over half the
    bytes — through ``reduce_ca`` (custom all-reduce) when the byte size
    passes its eligibility check, so the reduce is a named custom-AR kernel
    instead of an NCCL ring — and cast back in place. Scoring and top-k stay
    fp32; the transport quantization is gated by the selection-recall bound.
    Every rank receives the same reduced bytes, so cross-rank selection
    agreement is preserved. An eligibility miss (e.g. bs × width × 2 bytes
    over the custom-AR cap) falls back to an NCCL bf16 reduce and is logged
    loudly once — never a silent backend change.

    ``use_bf16=False`` keeps the original in-place fp32 reduce. No process
    group / distributed not initialized → no-op.

    ``copy_back=False`` (bf16 path only) skips the bf16→fp32 copy-back and
    returns the REDUCED BF16 tensor as the authoritative result — for
    consumers that upcast in-register (the radix top-k), whose compared
    values are then bit-identical to the copy-back fp32 values (bf16→fp32
    is exact) while the copy-back kernel disappears.
    """
    global _score_reduce_fallback_logged

    if process_group is None:
        return token_scores
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return token_scores
    if not use_bf16:
        torch.distributed.all_reduce(
            token_scores,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )
        return token_scores

    if bf16_scratch is not None:
        bf16_view = bf16_scratch[: token_scores.shape[0], : token_scores.shape[1]]
        bf16_view.copy_(token_scores)
    else:
        bf16_view = token_scores.to(torch.bfloat16)
    if reduce_ca is not None and reduce_ca.should_custom_ar(bf16_view):
        _log_score_reduce_bucket(
            bf16_view,
            custom_ar=True,
            algorithm=getattr(reduce_ca, "pinned_algo_name", "size_based"),
        )
        reduced = reduce_ca.custom_all_reduce(bf16_view)
        if not copy_back:
            return reduced
        token_scores.copy_(reduced)
        return token_scores
    if reduce_ca is not None and not _score_reduce_fallback_logged:
        logger.warning(
            "double_sparsity score reduce: bf16 tensor %s (%d bytes) is not "
            "custom-AR eligible; falling back to NCCL bf16 all-reduce. "
            "This is a documented per-shape fallback, not the named-kernel path.",
            tuple(bf16_view.shape),
            bf16_view.numel() * bf16_view.element_size(),
        )
        _score_reduce_fallback_logged = True
    _log_score_reduce_bucket(bf16_view, custom_ar=False, algorithm="NCCL_BF16")
    torch.distributed.all_reduce(
        bf16_view,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )
    if not copy_back:
        return bf16_view
    token_scores.copy_(bf16_view)
    return token_scores


def all_reduce_token_scores(
    token_scores: torch.Tensor,
    *,
    process_group=None,
) -> torch.Tensor:
    """Original in-place fp32 SUM reduce (compat wrapper over
    :func:`reduce_token_scores`)."""

    return reduce_token_scores(token_scores, process_group=process_group)


def _topk_by_score_then_pos(
    vals: torch.Tensor,
    pos: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` candidates by (value DESCENDING, then position ASCENDING) — the one
    deterministic tie-break all DS top-k selectors share.

    ``vals``/``pos`` are ``[bs, m]`` (``pos`` are the distinct logical positions of the
    candidates). Done as two stable passes: (1) order candidates by position ascending,
    then (2) stable argsort by value descending — so equal values resolve toward the
    lower position. Returns ``(top_positions [bs, k] int64, top_vals [bs, k])``. Uses
    fresh argsort outputs (no in/out aliasing — BL-20260527-torch-topk-aliasing).
    """
    pos_order = torch.argsort(pos, dim=-1, stable=True)            # ascending position
    pos_a = torch.gather(pos, 1, pos_order)
    vals_a = torch.gather(vals, 1, pos_order)
    val_order = torch.argsort(vals_a, dim=-1, descending=True, stable=True)[:, :k]
    return torch.gather(pos_a, 1, val_order), torch.gather(vals_a, 1, val_order)


def select_topk_sequence_order(
    token_scores: torch.Tensor,
    max_top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K selection returning sequence-order-ascending logical token positions.

    token_scores: [bs, max_tokens] fp32. Caller must have already applied any
                  per-request validity mask (unwritten / out-of-range tokens = -inf).

    Returns:
        selected_indices: int32 [bs, max_top_k], ascending, -1 padded.
        valid_lengths:    int32 [bs].
    """

    if token_scores.dim() != 2:
        raise ValueError(
            f"token_scores must be 2-D [bs, max_tokens], got {tuple(token_scores.shape)}."
        )
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    bs, max_tokens = token_scores.shape
    device = token_scores.device

    effective_top_k = min(max_top_k, max_tokens)
    # Deterministic tie-break: select by (score DESCENDING, then logical position
    # ASCENDING). token_scores columns are already in position order, so the shared
    # helper's stable score-descending argsort breaks score ties toward the lower
    # position -- the single ordering all DS top-k selectors honor (see
    # _topk_by_score_then_pos / blocked_topk_sequence_order).
    positions = torch.arange(max_tokens, device=device, dtype=torch.int64).unsqueeze(0).expand(bs, -1)
    topk_indices, topk_scores = _topk_by_score_then_pos(token_scores, positions, effective_top_k)

    invalid_entries = torch.isneginf(topk_scores)
    topk_indices = torch.where(
        invalid_entries,
        torch.full_like(topk_indices, max_tokens),
        topk_indices,
    )

    sorted_indices, _ = torch.sort(topk_indices, dim=-1)

    selected = torch.full(
        (bs, max_top_k),
        SELECTED_PAD_VALUE,
        dtype=torch.int32,
        device=device,
    )
    valid_mask_real = sorted_indices < max_tokens
    valid_lengths = valid_mask_real.to(torch.int32).sum(dim=-1)

    if effective_top_k > 0:
        position_grid = torch.arange(effective_top_k, device=device)
        keep_positions = position_grid.unsqueeze(0) < valid_lengths.unsqueeze(1)
        real_slice = torch.where(
            keep_positions,
            sorted_indices.to(torch.int32),
            torch.full_like(sorted_indices, SELECTED_PAD_VALUE, dtype=torch.int32),
        )
        selected[:, :effective_top_k] = real_slice

    return selected, valid_lengths.to(torch.int32)


def blocked_topk_sequence_order(
    token_scores: torch.Tensor,
    max_top_k: int,
    block_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact blocked top-K — identical output to :func:`select_topk_sequence_order`.

    Partitions the ``max_tokens`` score axis into fixed ``block_width`` blocks,
    keeps each block's top ``min(max_top_k, block_width)`` candidates, then merges
    them and takes the global top-K. The result (ascending logical positions,
    ``-1`` padded, + valid_lengths) is identical to the monolithic selection.

    Exactness: a token in the global top-K has within-block rank <= its global
    rank <= K, so it is in its block's top-min(K, block_width); the union of the
    per-block candidates therefore contains the global top-K, and merging them under
    the SHARED deterministic tie-break (score descending, then logical position
    ascending) reproduces ``select_topk_sequence_order`` EXACTLY -- including on
    finite ties (equal scores resolve to the lower position in both). This is the
    exactness ORACLE and the
    eager fallback for the graph-safe blocked top-k, whose value is that it can
    SKIP blocks entirely past each request's ``seq_len`` (every such block is all
    ``-inf`` and contributes no candidate), shrinking the per-decode-step work
    versus a monolithic ``torch.topk`` over the full KV-index width.

    token_scores: [bs, max_tokens] fp32 (unwritten/out-of-range tokens = -inf).
    Returns: selected_indices int32 [bs, max_top_k] ascending -1-padded; valid_lengths int32 [bs].
    """
    if token_scores.dim() != 2:
        raise ValueError(f"token_scores must be 2-D, got {tuple(token_scores.shape)}.")
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    if block_width <= 0:
        raise ValueError(f"block_width must be positive, got {block_width}.")
    bs, max_tokens = token_scores.shape
    device = token_scores.device
    K = min(max_top_k, max_tokens)
    bw = block_width
    nb = (max_tokens + bw - 1) // bw
    pad = nb * bw - max_tokens
    if pad:
        sc = token_scores.new_full((bs, nb * bw), float("-inf"))
        sc[:, :max_tokens] = token_scores
    else:
        sc = token_scores
    blk = sc.view(bs, nb, bw)
    kb = min(K, bw)
    # per-block top-kb candidate LOCAL positions by (score desc, local-pos asc) -- the
    # argsort indices ARE the local positions; stable keeps ascending pos on ties.
    blk_order = torch.argsort(blk, dim=-1, descending=True, stable=True)[:, :, :kb]
    block_base = (torch.arange(nb, device=device, dtype=torch.int64) * bw).view(1, nb, 1)
    cand_pos = (block_base + blk_order).reshape(bs, nb * kb)
    cand_vals = torch.gather(blk, 2, blk_order).reshape(bs, nb * kb)
    # global top-K over the union, by the SHARED (score desc, position asc) contract
    # -- identical to select_topk_sequence_order, including on finite ties.
    eff = min(K, nb * kb)
    sel_pos, merge_vals = _topk_by_score_then_pos(cand_vals, cand_pos, eff)
    invalid = torch.isneginf(merge_vals)
    sel_pos = torch.where(invalid, torch.full_like(sel_pos, max_tokens), sel_pos)
    sorted_pos, _ = torch.sort(sel_pos, dim=-1)
    selected = torch.full((bs, max_top_k), SELECTED_PAD_VALUE, dtype=torch.int32, device=device)
    valid_lengths = (sorted_pos < max_tokens).to(torch.int32).sum(dim=-1)
    grid = torch.arange(eff, device=device)
    keep = grid.unsqueeze(0) < valid_lengths.unsqueeze(1)
    real = torch.where(
        keep, sorted_pos.to(torch.int32),
        torch.full_like(sorted_pos, SELECTED_PAD_VALUE, dtype=torch.int32),
    )
    selected[:, :eff] = real
    return selected, valid_lengths.to(torch.int32)


def _anchor_positions(n: int, budget: int, mode: str) -> list:
    """Deterministic anchor logical positions in ``[0, n)`` for one request.

    - ``recency``: the ``budget`` most-recent positions ``[n-budget, n)``.
    - ``global``: the ``budget`` earliest stable positions ``[0, budget)``.
    - ``strided``: ``budget`` distinct evenly-spaced positions over ``[0, n)``.
    Clamps ``budget`` to ``n``; returns ``[]`` for ``off`` / empty.
    """
    if budget <= 0 or n <= 0 or mode == "off":
        return []
    b = min(budget, n)
    if mode == "recency":
        return list(range(n - b, n))
    if mode == "global":
        return list(range(0, b))
    if mode == "strided":
        if b == 1:
            return [0]
        step = (n - 1) / (b - 1)
        return sorted({int(round(i * step)) for i in range(b)})
    return []


def _anchor_positions_tensor(
    seq_lens: torch.Tensor, eb: torch.Tensor, A: int, mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensorized ``_anchor_positions``: ``[bs, A]`` anchor logical positions +
    ``[bs, A]`` validity (slot ``i`` valid iff ``i < eb`` and, for strided, it is
    the first occurrence of its value). Graph-safe (no host sync, fixed shape)."""
    device = seq_lens.device
    bs = seq_lens.shape[0]
    i = torch.arange(A, device=device).view(1, A)
    n = seq_lens.view(bs, 1).to(torch.int64)
    ebv = eb.view(bs, 1)
    valid = i < ebv
    if mode == "recency":
        pos = n - ebv + i
    elif mode == "global":
        pos = i.expand(bs, A).clone()
    elif mode == "strided":
        denom = (ebv - 1).clamp(min=1).to(torch.float64)
        step = (n - 1).to(torch.float64) / denom
        pos = torch.round(i.to(torch.float64) * step).to(torch.int64)
        pos = torch.where(ebv == 1, torch.zeros_like(pos), pos)
        # strided's set-dedup: values are ascending in i, so a duplicate is == prev.
        prev = torch.cat(
            [torch.full((bs, 1), -1, dtype=torch.int64, device=device), pos[:, :-1]],
            dim=1,
        )
        valid = valid & (pos != prev)
    else:
        pos = torch.zeros(bs, A, dtype=torch.int64, device=device)
        valid = torch.zeros(bs, A, dtype=torch.bool, device=device)
    pos = torch.where(valid, pos, torch.full_like(pos, -1))
    return pos, valid


def _stable_argsort_ascending(
    key: torch.Tensor, tiebreak_pos: torch.Tensor
) -> torch.Tensor:
    """Argsort ``key`` ascending with ``tiebreak_pos`` ascending as the stable
    tie-break (two stable passes). Mirrors the eager list ``.sort(key=score)``,
    which keeps the original position-ascending order among equal scores."""
    order_p = torch.argsort(tiebreak_pos, dim=1, stable=True)
    key_p = torch.gather(key, 1, order_p)
    order_k = torch.argsort(key_p, dim=1, stable=True)
    return torch.gather(order_p, 1, order_k)


def _force_include_anchor(
    indices: torch.Tensor,
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    anchor_budget: int,
    anchor_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Force each request's deterministic anchor positions (per ``anchor_mode``)
    into the selection, evicting the lowest-scoring non-anchor selected positions
    (stable position-ascending tie-break) and preserving the per-row selected
    count.

    Fully tensorized (no per-row Python loop, no ``.item()`` host sync, fixed
    shapes), so it is graph-safe and is used by BOTH the eager logical path and
    the graph-safe Triton path — guaranteeing identical selection. Bit-identical
    to the former per-row reference (fuzz-verified) including the R3 over-budget
    clamp (``effective_budget = min(anchor_budget, valid_count, seq_len)``) and
    strided set-dedup.
    """
    if anchor_mode == "off" or anchor_budget <= 0:
        return indices.to(torch.int32), (indices >= 0).to(torch.int32).sum(-1)
    device = indices.device
    bs, K = indices.shape
    max_seq = scores.shape[1]
    # Bound the [bs, A] temporaries: the effective budget is
    # min(anchor_budget, valid_count, seq_len) and valid_count <= K, so anchor
    # slots beyond K (or max_seq) can never be valid. Clamping A here is
    # bit-identical (clamped-out slots would be invalid anyway) but stops a
    # pathological opt-in anchor_budget from over-allocating scratch.
    A = min(int(anchor_budget), K, max_seq)
    pos = indices.to(torch.int64)
    real_mask = pos >= 0
    real_count = real_mask.sum(1)
    n = seq_lens.to(torch.int64)
    eb = torch.minimum(torch.full_like(real_count, A), real_count)
    eb = torch.minimum(eb, n)  # _anchor_positions further clamps the budget to n

    apos, avalid = _anchor_positions_tensor(n, eb, A, anchor_mode)
    psafe = pos.clamp(min=0)

    # max_seq-wide membership masks (an extra sentinel column absorbs -1 pads).
    sel_mask = torch.zeros(bs, max_seq + 1, dtype=torch.bool, device=device)
    sel_mask.scatter_(1, torch.where(real_mask, pos, torch.full_like(pos, max_seq)), True)
    sel_mask = sel_mask[:, :max_seq]
    anc_mask = torch.zeros(bs, max_seq + 1, dtype=torch.bool, device=device)
    anc_mask.scatter_(1, torch.where(avalid, apos, torch.full_like(apos, max_seq)), True)
    anc_mask = anc_mask[:, :max_seq]

    missing = avalid & ~torch.gather(sel_mask, 1, apos.clamp(min=0))   # [bs,A]
    evictable = real_mask & ~torch.gather(anc_mask, 1, psafe)          # [bs,K]
    k = torch.minimum(missing.sum(1), evictable.sum(1))               # [bs]

    # Evict the k lowest-score evictables (score asc, position asc tie-break).
    big_score = torch.finfo(torch.float32).max
    evict_score = torch.where(
        evictable, torch.gather(scores, 1, psafe),
        torch.full((bs, K), big_score, dtype=scores.dtype, device=device),
    )
    order = _stable_argsort_ascending(evict_score, pos)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(K, device=device).view(1, K).expand(bs, K))
    drop = evictable & (rank < k.view(bs, 1))
    keep = real_mask & ~drop

    # Insert the first k missing anchors (ascending position).
    miss_rank = torch.cumsum(missing.to(torch.int64), dim=1) - 1
    insert = missing & (miss_rank < k.view(bs, 1))

    # Combine keep + inserted positions, sort ascending, pad to K with -1.
    big = max_seq + 10
    keep_pos = torch.where(keep, psafe, torch.full_like(psafe, big))
    ins_pos = torch.where(insert, apos.clamp(min=0), torch.full((bs, A), big, dtype=torch.int64, device=device))
    combined, _ = torch.sort(torch.cat([keep_pos, ins_pos], dim=1), dim=1)
    out = combined[:, :K]
    out = torch.where(out >= big, torch.full_like(out, -1), out).to(torch.int32)
    return out, (out >= 0).to(torch.int32).sum(-1)


def _maybe_record_recall_oracle(
    scores: torch.Tensor,
    selected_indices: torch.Tensor,
    layer_id: int,
    max_top_k: int,
    process_group=None,
    recall_oracle: bool = False,
    absorbed_scores: Optional[torch.Tensor] = None,
    absorbed_indices: Optional[torch.Tensor] = None,
) -> None:
    """Record one recall-oracle sample for the active NIAH trial, if enabled.

    Pure no-op (immediate return) when the oracle is off — so production
    selection is byte-for-byte unchanged. Enabled either by the config-borne
    ``recall_oracle`` flag (the path that reaches TP workers) or the env flag
    (harness / unit tests).

    **Fail-closed when enabled** — a diagnostic must never silently guess or
    silently drop a sample. With no active trial, an out-of-range harness needle
    position, or a payload-build exception, we emit an explicit ``failure``
    record keyed by ``(request_id, trial_id, layer_id, decode_step)`` instead of
    returning quietly; the sweep asserts on these + on missing successes. We do
    NOT filter out-of-range positions (that silently masked the absent 64K
    records) and we do NOT swallow exceptions.

    Records ONLY on the primary TP rank: the scores are identical across ranks
    after ``all_reduce_token_scores``, so rank-0-only recording avoids 8×
    duplicate writes + cross-process file contention on the sink.
    """
    from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as _sink

    if recall_oracle:
        # Latch the config-borne enable so the sink + trial-file paths resolve
        # to the fixed cross-process defaults (env does not reach TP workers).
        _sink.enable_via_config()
    if not _sink.oracle_enabled():
        return
    # Primary-rank guard (scores are all-reduce-identical across TP ranks).
    try:
        _rk = -1
        if (
            process_group is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            _rk = torch.distributed.get_rank(group=process_group)
        if _rk not in (-1, 0):
            return  # non-primary rank (silent — every rank but 0)
    except Exception:
        pass

    sample_idx = _sink.next_sample_index()
    trial = _sink.get_active_trial()
    if trial is None:
        # Fail-closed: enabled but no trial registered for this decode. Emit a
        # marker (the harness clears the sink before measured trials, so warmup
        # markers do not pollute the measured run).
        _sink.record_oracle_failure(
            reason="no_active_trial",
            request_id=None,
            trial_id=None,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
        )
        return

    max_tokens = int(scores.shape[-1])
    out_of_range = [p for p in trial.needle_positions if not (0 <= p < max_tokens)]
    if out_of_range:
        # Fail-closed: reject (do NOT filter) — a partial/empty span would
        # silently mis-measure recall.
        _sink.record_oracle_failure(
            reason="span_out_of_range",
            request_id=trial.request_id,
            trial_id=trial.trial_id,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
            extra={
                "needle_positions": list(trial.needle_positions),
                "out_of_range": out_of_range,
                "max_tokens": max_tokens,
            },
        )
        return

    try:
        from sglang.srt.layers.attention.double_sparsity.selection_recall_oracle import (
            oracle_payload_for_row,
        )

        needle = torch.as_tensor(
            trial.needle_positions, dtype=torch.int64, device=scores.device
        )
        payload = oracle_payload_for_row(
            scores[0],
            needle,
            selected_indices_row=selected_indices[0],
            stride=1,
            index_topk=int(max_top_k),
        )
        # Side-by-side absorbed-latent payload (score-only diagnostic). Shares the
        # one record + sample index — do NOT advance next_sample_index again — so
        # the table and absorbed rows compare at the same (layer, decode_step).
        if absorbed_scores is not None and absorbed_indices is not None:
            payload["absorbed"] = oracle_payload_for_row(
                absorbed_scores[0],
                needle,
                selected_indices_row=absorbed_indices[0],
                stride=1,
                index_topk=int(max_top_k),
            )
        _sink.record_oracle_sample(
            request_id=trial.request_id,
            trial_id=trial.trial_id,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
            payload=payload,
        )
    except Exception as _e:
        # Fail-closed: surface the failure as a record rather than swallowing it.
        _sink.record_oracle_failure(
            reason=f"exception:{type(_e).__name__}:{_e}",
            request_id=trial.request_id,
            trial_id=trial.trial_id,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
        )
        return


def absorbed_topk_select(
    *,
    queries: torch.Tensor,
    absorbed_w_sel: torch.Tensor,
    channel_selection_layer: torch.Tensor,
    channel_weights_layer: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    max_top_k: int,
    written_layer: Optional[torch.Tensor] = None,
    absorbed_latent_fp8: Optional[torch.Tensor] = None,
    absorbed_latent_scales: Optional[torch.Tensor] = None,
    absorbed_latent: Optional[torch.Tensor] = None,
    per_request_valid: Optional[torch.Tensor] = None,
    process_group=None,
    reduce_ca=None,
    score_reduce_bf16: bool = False,
    head_agg: str = "max",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Selection: score → all-reduce → per-request mask → top-K → ascend,
    from the resident MLA latent.

    ``score[b, t] = agg_h ( v_h[b] · c_kv[t] )`` for ``scorer_norm="off"`` — the
    absorbed-latent identity the recall oracle already validated. The rope dims are
    excluded by construction: ``absorbed_w_sel`` is the K-noPE ``W_UK`` rows (built
    from ``kv_b_proj`` sliced to ``[:qk_nope_head_dim]``) and ``queries`` is the
    no-PE query, so the score never touches a positional channel.

    Reads the paged fp8 latent (``absorbed_latent_fp8`` + ``absorbed_latent_scales``,
    the resident pool bytes) on CUDA, or a dequantized ``absorbed_latent`` ``[T, lora]``
    on the CPU reference path. Returns ``(selected_indices, valid_lengths)`` —
    sequence-ascending int32, ``-1`` padded.
    """
    if absorbed_latent_fp8 is not None and absorbed_latent_scales is not None:
        from sglang.srt.layers.attention.double_sparsity.absorbed_latent_kernel import (
            absorbed_latent_score_logical_paged,
        )

        scores = absorbed_latent_score_logical_paged(
            queries,
            absorbed_latent_fp8,
            absorbed_latent_scales,
            absorbed_w_sel,
            channel_selection_layer,
            channel_weights_layer,
            req_pool_indices,
            req_to_token,
            seq_lens,
            max_seq_len,
            written=written_layer,
            head_agg=head_agg,
        )
    elif absorbed_latent is not None:
        from sglang.srt.layers.attention.double_sparsity.absorbed_latent import (
            absorbed_latent_score_logical,
        )

        scores = absorbed_latent_score_logical(
            queries,
            absorbed_latent,
            absorbed_w_sel,
            channel_selection_layer,
            channel_weights_layer,
            req_pool_indices,
            req_to_token,
            seq_lens,
            max_seq_len,
            written=written_layer,
            head_agg=head_agg,
        )
    else:
        raise ValueError(
            "absorbed_topk_select requires either (absorbed_latent_fp8, "
            "absorbed_latent_scales) for the paged path or absorbed_latent for the "
            "dequantized reference path."
        )

    scores = reduce_token_scores(
        scores,
        process_group=process_group,
        reduce_ca=reduce_ca,
        use_bf16=score_reduce_bf16,
    )
    if per_request_valid is not None:
        if per_request_valid.shape != scores.shape:
            raise ValueError(
                f"per_request_valid shape {tuple(per_request_valid.shape)} must "
                f"match absorbed score shape {tuple(scores.shape)}."
            )
        scores = scores.masked_fill(~per_request_valid.to(torch.bool), float("-inf"))
    return select_topk_sequence_order(scores, max_top_k)


def retrieve_topk_graph_safe(
    *,
    queries: torch.Tensor,
    written: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    max_top_k: int,
    out_indices: torch.Tensor,
    out_lengths: torch.Tensor,
    # Pre-allocated scratch tensors (required for CUDA / Triton fast path)
    scratch_scores: Optional[torch.Tensor] = None,         # fp32 [max_bs, max_seq_len]
    scratch_topk_values: Optional[torch.Tensor] = None,    # fp32 [max_bs, max_top_k]
    scratch_topk_indices: Optional[torch.Tensor] = None,   # int64 [max_bs, max_top_k]
    scratch_invalid_mask: Optional[torch.Tensor] = None,   # bool [max_bs, max_top_k]
    scratch_sorted_vals: Optional[torch.Tensor] = None,    # int64 [max_bs, max_top_k]
    scratch_boundary: Optional[torch.Tensor] = None,       # int64 [max_bs, 1] = max_seq_len
    scratch_valid_i64: Optional[torch.Tensor] = None,      # int64 [max_bs, 1]
    per_request_valid: Optional[torch.Tensor] = None,      # bool [bs, max_seq_len]
    scratch_pv_mask: Optional[torch.Tensor] = None,        # bool [max_bs, max_seq_len]
    scratch_throwaway_idx: Optional[torch.Tensor] = None,  # int64 [max_bs, max_top_k]
    scratch_scores_bf16: Optional[torch.Tensor] = None,    # bf16 [max_bs, max_seq_len]
    radix_topk_scratch: Optional[dict] = None,  # topk_kernel scratch bundle
    topk_block: int = 1024,
    process_group=None,
    reduce_ca=None,
    score_reduce_bf16: bool = False,
    recall_oracle: bool = False,
    score_capture: bool = False,
    scorer_norm: str = "off",
    head_agg: str = "max",
    anchor_mode: str = "off",
    anchor_budget: int = 0,
    absorbed_latent_fp8: Optional[torch.Tensor] = None,
    absorbed_latent_scales: Optional[torch.Tensor] = None,
    absorbed_w_sel: Optional[torch.Tensor] = None,
    absorbed_latent: Optional[torch.Tensor] = None,
    scratch_absorbed_v: Optional[torch.Tensor] = None,  # fp32 [max_bs, H, kv_lora_rank]
    scratch_absorbed_qsel: Optional[torch.Tensor] = None,  # fp32 [max_bs, H, label_dim]
    scratch_absorbed_sel_i64: Optional[torch.Tensor] = None,  # int64 [H, label_dim]
    scratch_absorbed_q: Optional[torch.Tensor] = None,  # fp32 [max_bs, H, nope_dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture-safe selection that writes results into caller-owned buffers.

    The returned top-K comes from the absorbed-latent score read straight off
    the resident MLA latent — no labels gather. The absorbed identity holds only
    for scorer_norm="off", and the rope dims are excluded by construction (no-PE
    queries + K-noPE W_UK rows in ``absorbed_w_sel``). ``written`` is the
    slot-validity bitmap ``[L, T]`` so a reused physical slot's stale latent is
    masked to ``-inf`` until its fresh KV write lands.

    On CUDA with Triton + all scratch buffers provided: uses an allocation-free
    pipeline.  After a single warmup call, subsequent calls perform zero new
    CUDA allocations:

        1. the paged absorbed kernel fills ``scratch_scores`` directly.
        2. (optional) ``per_request_valid`` is applied via in-place masked_fill_.
        3. ``topk`` with ``out=(values, indices)`` (allocation-free after warmup).
        4. ``isneginf`` + ``masked_fill_`` to sentinel-out invalid entries.
        5. ``topk(largest=False, sorted=True)`` for an allocation-free ascending sort.
        6. ``ge`` + ``masked_fill_`` to convert sentinels to ``-1`` in output.
        7. ``searchsorted`` with ``out=`` for valid_lengths.

    Fallback path (CPU, or scratch tensors missing): calls the eager
    :func:`absorbed_topk_select`.  This branch is intended for unit tests;
    do NOT route production graph capture through it.
    """
    bs = req_pool_indices.shape[0]
    device = queries.device

    assert scorer_norm == "off", (
        "Double Sparsity selection requires scorer_norm='off' (the absorbed-latent "
        f"identity only holds there); got {scorer_norm!r}."
    )
    assert absorbed_w_sel is not None, (
        "Double Sparsity selection requires absorbed_w_sel (the bind-time K-noPE "
        "W_UK projection)."
    )

    use_triton_fast = (
        _TRITON_AVAILABLE
        and device.type == "cuda"
        and scratch_scores is not None
        and scratch_topk_values is not None
        and scratch_topk_indices is not None
        and scratch_invalid_mask is not None
        and scratch_sorted_vals is not None
        and scratch_boundary is not None
        and scratch_valid_i64 is not None
        and scratch_throwaway_idx is not None
    )

    # CPU / no-scratch fallback (unit tests): the eager absorbed_topk_select
    # scores the resident latent without the in-place graph-state scratch. The
    # graph-safe path below fills scratch_scores in place and shares the same
    # reduce + radix top-k.
    if not use_triton_fast:
        indices, valid = absorbed_topk_select(
            queries=queries,
            absorbed_w_sel=absorbed_w_sel,
            channel_selection_layer=channel_selection[layer_id],
            channel_weights_layer=channel_weights[layer_id],
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            max_top_k=max_top_k,
            written_layer=written[layer_id] if written is not None else None,
            absorbed_latent_fp8=absorbed_latent_fp8,
            absorbed_latent_scales=absorbed_latent_scales,
            absorbed_latent=absorbed_latent,
            per_request_valid=per_request_valid,
            process_group=process_group,
            reduce_ca=reduce_ca,
            score_reduce_bf16=score_reduce_bf16,
            head_agg=head_agg,
        )
        mtk = indices.shape[1]
        out_indices[:bs, :mtk].copy_(indices)
        if mtk < out_indices.shape[1]:
            out_indices[:bs, mtk:].fill_(-1)
        out_lengths[:bs].copy_(valid)
        return out_indices, out_lengths

    # Triton fast path — zero-allocation after warmup.
    # Contract (caller responsibility — bind_runtime_data enforces it for the
    # channel-mask tensors): channel_selection int32, channel_weights fp32,
    # req_pool_indices / req_to_token / seq_lens int32. queries may be
    # fp32 / fp16 / bf16 — the kernel casts via tl.load(...).to(tl.float32).
    sel_layer = channel_selection[layer_id]
    w_layer = channel_weights[layer_id]
    assert sel_layer.dtype == torch.int32, (
        f"channel_selection must be int32, got {sel_layer.dtype}"
    )
    assert w_layer.dtype == torch.float32, (
        f"channel_weights must be float32, got {w_layer.dtype}"
    )
    assert req_pool_indices.dtype == torch.int32, (
        f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
    )
    assert req_to_token.dtype == torch.int32, (
        f"req_to_token must be int32, got {req_to_token.dtype}"
    )
    assert seq_lens.dtype == torch.int32, (
        f"seq_lens must be int32, got {seq_lens.dtype}"
    )

    # NVTX ranges name the three DS-specific cost buckets (logical score /
    # score all-reduce / top-k select) so profiles can attribute them without
    # kernel-name matching. Host-side annotations: they mark eager decode and
    # the capture-time launches; CUDA-graph replay does not re-emit them.
    scores_view = scratch_scores[:bs, :max_seq_len]
    # Dead positions (past seq_len) only need -inf when a full-width consumer
    # reads them: the legacy torch.topk pipeline scans the whole scratch, the
    # recall oracle ranks the full score row, and the anchor force-include is
    # defensive-listed. The sequence-bounded radix selector reads none of them.
    _store_dead = (
        radix_topk_scratch is None or recall_oracle or anchor_mode != "off"
    )
    # Score the logical positions straight from the resident fp8 latent into
    # scratch_scores IN PLACE. v_h is built into scratch_absorbed_v
    # allocation-free, then the paged absorbed kernel writes the score; from here
    # the path is reduce + radix top-k. The absorbed paged kernel always stores
    # -inf over dead positions, matching _store_dead's full-width consumers.
    assert absorbed_latent_fp8 is not None and absorbed_latent_scales is not None, (
        "Double Sparsity graph-safe selection requires the resident fp8 latent "
        "(absorbed_latent_fp8, absorbed_latent_scales)."
    )
    # Fail closed: the absorbed scratch MUST be present before the CUDA fast
    # path runs. A None here would make absorbed_latent_score_logical_paged
    # fall back to the ALLOCATING absorbed_latent_v (breaking the graph-safe
    # zero-alloc contract) instead of building v_h in place.
    assert (
        scratch_absorbed_v is not None
        and scratch_absorbed_qsel is not None
        and scratch_absorbed_sel_i64 is not None
        and scratch_absorbed_q is not None
    ), (
        "Double Sparsity graph-safe selection requires the preallocated absorbed "
        "scratch (scratch_absorbed_v, scratch_absorbed_qsel, "
        "scratch_absorbed_sel_i64, scratch_absorbed_q); one is None, which "
        "would silently route through the allocating fallback."
    )
    from sglang.srt.layers.attention.double_sparsity.absorbed_latent_kernel import (
        absorbed_latent_score_logical_paged,
    )

    scratch_absorbed_sel_i64.copy_(sel_layer)
    sel_i64 = scratch_absorbed_sel_i64
    torch.cuda.nvtx.range_push("ds_absorbed_score")
    absorbed_latent_score_logical_paged(
        queries,
        absorbed_latent_fp8,
        absorbed_latent_scales,
        absorbed_w_sel,
        sel_layer,
        w_layer,
        req_pool_indices,
        req_to_token,
        seq_lens,
        max_seq_len,
        written=written[layer_id] if written is not None else None,
        head_agg=head_agg,
        out=scores_view,
        scratch_v=scratch_absorbed_v,
        scratch_qsel=scratch_absorbed_qsel,
        channel_selection_i64=sel_i64,
        scratch_q=scratch_absorbed_q,
    )
    torch.cuda.nvtx.range_pop()

    # The radix selector upcasts score loads in-register, so the reduced bf16
    # buffer can be its authoritative input: the compared values are
    # bit-identical to the fp32 copy-back (bf16→fp32 is exact) and the
    # copy-back kernel disappears. The full-row fp32 consumers — the legacy
    # torch.topk pipeline, the recall oracle's ranking, and the anchor
    # force-include — keep the copy-back.
    bf16_used = score_reduce_bf16 and scratch_scores_bf16 is not None
    bf16_authoritative = (
        radix_topk_scratch is not None
        and bf16_used
        and not recall_oracle
        and anchor_mode == "off"
    )
    topk_scores = scores_view
    # DEC-9 scorepath exp-1: snapshot the PRE-reduce per-rank score (scores_view
    # straight off the absorbed paged kernel, before the cross-TP all-reduce).
    # The reduce mutates scores_view in place (copy_back) or returns a separate
    # bf16 view, so the pre-reduce values must be cloned NOW. Eager-only,
    # capture-guarded, off by default — one bool check on the hot path when off.
    pre_reduce_snapshot = None
    if score_capture and not torch.cuda.is_current_stream_capturing():
        pre_reduce_snapshot = scores_view.detach().clone()
    if process_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.cuda.nvtx.range_push("ds_score_allreduce")
        reduced = reduce_token_scores(
            scores_view,
            process_group=process_group,
            reduce_ca=reduce_ca,
            bf16_scratch=scratch_scores_bf16,
            use_bf16=bf16_used,
            copy_back=not bf16_authoritative,
        )
        if bf16_authoritative:
            topk_scores = reduced
        torch.cuda.nvtx.range_pop()

    if per_request_valid is not None:
        assert scratch_pv_mask is not None, (
            "per_request_valid requires scratch_pv_mask in graph-safe path"
        )
        pv_view = scratch_pv_mask[:bs, :max_seq_len]
        # copy_ handles dtype conversion in-place (no allocation when shapes match).
        pv_view.copy_(per_request_valid)
        # In-place flip: True = valid → True = invalid; then masked_fill_(invalid, -inf).
        torch.logical_not(pv_view, out=pv_view)
        # Masks the AUTHORITATIVE buffer: bf16(-inf) upcasts to fp32(-inf),
        # so the masked selection is identical on either dtype.
        topk_scores.masked_fill_(pv_view, float("-inf"))

    torch.cuda.nvtx.range_push("ds_topk_select")
    if radix_topk_scratch is not None:
        # Sequence-aware deterministic radix top-k: work proportional to each
        # row's live window, exact (score desc, pos asc) selection emitted in
        # ascending order directly (replaces the two full-width torch.topk
        # passes below). Fixed grids; allocation-free with the scratch bundle.
        from sglang.srt.layers.attention.double_sparsity.topk_kernel import (
            select_topk_sequence_order_triton,
        )

        select_topk_sequence_order_triton(
            topk_scores,
            seq_lens,
            max_top_k,
            out_indices=out_indices,
            out_lengths=out_lengths,
            block=topk_block,
            **radix_topk_scratch,
        )
        if max_top_k < out_indices.shape[1]:
            out_indices[:bs, max_top_k:].fill_(-1)
    else:
        effective_k = min(max_top_k, max_seq_len)
        topk_vals_view = scratch_topk_values[:bs, :effective_k]
        topk_idx_view = scratch_topk_indices[:bs, :effective_k]
        invalid_view = scratch_invalid_mask[:bs, :effective_k]
        sorted_vals_view = scratch_sorted_vals[:bs, :effective_k]
        boundary_view = scratch_boundary[:bs]
        valid_i64_view = scratch_valid_i64[:bs]

        # Step 1: top-K by score (unsorted, largest).
        torch.topk(
            scores_view,
            effective_k,
            dim=-1,
            largest=True,
            sorted=False,
            out=(topk_vals_view, topk_idx_view),
        )

        # Step 2: sentinel-out invalid (-inf) entries; replace their position with max_seq_len.
        torch.isneginf(topk_vals_view, out=invalid_view)
        topk_idx_view.masked_fill_(invalid_view, max_seq_len)

        # Step 3: ascending sort using topk(largest=False, sorted=True).
        # PyTorch's topk requires output indices NOT to alias input — aliasing
        # corrupts the read (observed: input [3, 1] → output values [0, 1]).
        # Route throwaway gather indices into a dedicated scratch.
        assert scratch_throwaway_idx is not None, (
            "scratch_throwaway_idx is required for the graph-safe topk pipeline"
        )
        throwaway_view = scratch_throwaway_idx[:bs, :effective_k]
        torch.topk(
            topk_idx_view,
            effective_k,
            dim=-1,
            largest=False,
            sorted=True,
            out=(sorted_vals_view, throwaway_view),
        )

        # Step 4: copy sorted positions to int32 output, then sentinel → -1.
        out_indices[:bs, :effective_k].copy_(sorted_vals_view)
        torch.ge(sorted_vals_view, max_seq_len, out=invalid_view)
        out_indices[:bs, :effective_k].masked_fill_(invalid_view, -1)
        if effective_k < out_indices.shape[1]:
            out_indices[:bs, effective_k:].fill_(-1)

        # Step 5: count valid (< max_seq_len) entries via searchsorted on the sorted vector.
        boundary_view.fill_(max_seq_len)
        torch.searchsorted(
            sorted_vals_view, boundary_view, right=False, out=valid_i64_view
        )
        out_lengths[:bs].copy_(valid_i64_view.squeeze(-1))
    torch.cuda.nvtx.range_pop()

    # Graph-safe anchor-budget force-include (R9): tensorized, fixed-shape, no
    # host sync — bit-identical to the eager path (same _force_include_anchor).
    # Off by default; under CUDA-graph capture the extra ops are captured once and
    # replay reuses their memory (alloc-free on replay).
    if anchor_mode != "off" and anchor_budget > 0:
        a_idx, a_len = _force_include_anchor(
            out_indices[:bs, :max_top_k], scores_view, seq_lens,
            int(anchor_budget), anchor_mode,
        )
        out_indices[:bs, :max_top_k].copy_(a_idx)
        out_lengths[:bs].copy_(a_len)

    # Flag-gated SCORE capture (DEC-9 Q2 instrument): dump the absorbed score row
    # the top-k just consumed. ``topk_scores`` is the AUTHORITATIVE input to the
    # selection top-k — the bf16 reduced view when bf16 is authoritative, else the
    # fp32 ``scores_view``; the dump upcasts to fp32 (bf16->fp32 is exact). Column
    # t == logical position t (same domain as selection_capture). Eager decode
    # only (the host copy is illegal under capture); off by default — one getattr
    # here when off. Capture-guarded; this Python does not re-run on graph replay.
    if score_capture and not torch.cuda.is_current_stream_capturing():
        from sglang.srt.layers.attention.double_sparsity.score_capture import (
            maybe_dump_score_capture,
        )

        maybe_dump_score_capture(
            scores=topk_scores[:bs],
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            layer_id=layer_id,
            pre_reduce_scores=(
                pre_reduce_snapshot[:bs] if pre_reduce_snapshot is not None else None
            ),
        )

    # Flag-gated recall oracle on the production GPU decode path. ``scores_view``
    # is the all-reduced + per-request-masked score tensor (after the all_reduce
    # above, the same tensor the top-K consumed); ``out_indices[:bs]`` is the
    # selection. Capture-guarded (host syncs illegal during graph capture) and
    # off by default, so production decode is unaffected. Records only in eager
    # decode (under graph replay this Python does not re-run).
    if not torch.cuda.is_current_stream_capturing():
        # ``scores_view`` is the all-reduced + per-request-masked absorbed-latent
        # score the top-K above consumed; ``out_indices[:bs]`` is the selection.
        _maybe_record_recall_oracle(
            scores_view,
            out_indices[:bs],
            layer_id,
            max_top_k,
            process_group=process_group,
            recall_oracle=recall_oracle,
        )

    return out_indices, out_lengths


# Public alias for the end-to-end selector pipeline.
retrieve_topk = absorbed_topk_select
