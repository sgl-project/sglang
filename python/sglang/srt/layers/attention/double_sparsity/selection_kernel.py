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
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

SELECTED_PAD_VALUE = -1
_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401  # availability probe for _TRITON_AVAILABLE
    import triton.language as tl  # noqa: F401  # availability probe

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def assert_rope_selection_supported(
    *,
    is_nextn: bool,
    dcp_world_size: int,
    forward_mode_is_decode: bool,
    q_pe,
    positions,
    rotary_emb,
) -> None:
    """Fail closed for rope-aware DS selection on non-validated runtimes.

    The caller invokes this only when ``rope_aware_score`` is on. rope-aware
    selection is validated ONLY for single-token MLA decode with q_pe/positions/
    rotary_emb threaded; every other runtime raises here
    rather than silently scoring no-PE. (fp8-vs-bf16 resident KV is checked
    separately by :func:`assert_rope_fp8_resident` once the pool dtype is known.)
    """
    if is_nextn:
        raise RuntimeError(
            "Double Sparsity 'rope_aware_score' is not supported on MTP/nextn "
            "layers (not validated); set rope_aware_score=false or disable MTP."
        )
    if dcp_world_size > 1:
        raise RuntimeError(
            "Double Sparsity 'rope_aware_score' is not supported with decode "
            "context parallel (DCP world size > 1; not validated); set "
            "rope_aware_score=false."
        )
    if not forward_mode_is_decode:
        raise RuntimeError(
            "Double Sparsity 'rope_aware_score' is validated only for single-token "
            "decode (speculative/extend not supported). Set rope_aware_score=false."
        )
    if q_pe is None or positions is None or rotary_emb is None:
        raise RuntimeError(
            "Double Sparsity 'rope_aware_score' requires q_pe, positions, and "
            "rotary_emb at the selection site; one is None (the rope query is not "
            "threaded on this path)."
        )


def assert_rope_fp8_resident(resident_dtype) -> None:
    """Fail closed when rope-aware selection sees a non-fp8 resident KV latent.

    The resident post-RoPE k_pe slice + the fp8 absorbed identity are validated
    only for the fp8 KV layout; bf16 resident KV has a different byte layout
    (fp8-only first ship; bf16 KV fails closed).
    """
    if resident_dtype == torch.bfloat16:
        raise RuntimeError(
            "Double Sparsity 'rope_aware_score' is validated only for fp8 KV "
            "cache; the resident KV here is bf16. Pin --kv-cache-dtype fp8_e4m3 "
            "or set rope_aware_score=false."
        )


def ds_scorer_is_graph_safe(config) -> bool:
    """``True`` iff every configured selector variant runs under CUDA-graph
    capture. Both supported scorers (cosine, raw-dot "off") are graph-safe;
    retained as the single guard so a future non-graph-safe variant can gate here.
    """
    if config is None:
        return True
    return getattr(config, "scorer_norm", "off") in ("off", "cosine")


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
    pos_order = torch.argsort(pos, dim=-1, stable=True)  # ascending position
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
    # _topk_by_score_then_pos).
    positions = (
        torch.arange(max_tokens, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(bs, -1)
    )
    topk_indices, topk_scores = _topk_by_score_then_pos(
        token_scores, positions, effective_top_k
    )

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
    scorer_norm: str = "off",
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
    if scorer_norm != "off":
        # Only the raw channel-dot ("off") absorbed identity is implemented on this
        # eager path. Cosine is accepted by config but its key-norm division is
        # built only on the graph-safe retrieve_topk_graph_safe path, so fail
        # loudly rather than silently returning a raw-dot selection under cosine.
        raise NotImplementedError(
            f"Double Sparsity eager absorbed_topk_select supports scorer_norm="
            f"'off' only; got {scorer_norm!r} (the cosine key-norm path is the "
            f"graph-safe retrieve_topk_graph_safe)."
        )
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


if _TRITON_AVAILABLE:

    @triton.jit
    def _current_slot_force_include_kernel(
        scores_ptr,  # [bs, max_seq_len] authoritative topk_scores (bf16 or fp32)
        seq_lens_ptr,  # [bs] int32
        qnorm_ptr,  # [bs, H] fp32 — per-row query selection-channel norms (cosine)
        scores_row_stride,
        qnorm_row_stride,
        max_seq_len,
        H,
        BLOCK_H: tl.constexpr,
        COSINE: tl.constexpr,
    ):
        # One program per row. Force-include the current decode slot (logical
        # position seq_len-1) by writing +inf. FAIL-CLOSED: write nothing when
        # seq_len-1 is outside the scored width [0, max_seq_len). For cosine,
        # additionally require the row's q-norm to be finite. In place — zero
        # allocation, graph-safe. The current slot's own key-norm is NOT gated:
        # the current decode token's KV is valid by construction, and its cache
        # entry can lag one step (gating on it regressed the gate 0.970 -> 0.640).
        b = tl.program_id(0)
        cur = tl.load(seq_lens_ptr + b).to(tl.int32) - 1
        in_range = (cur >= 0) & (cur < max_seq_len)
        do_write = in_range
        if COSINE:
            offs = tl.arange(0, BLOCK_H)
            hmask = offs < H
            qn = tl.load(qnorm_ptr + b * qnorm_row_stride + offs, mask=hmask, other=0.0)
            q_finite = (tl.sum((qn != qn).to(tl.int32)) == 0) & (
                tl.max(qn) < float("inf")
            )
            do_write = in_range & q_finite
        # Masked store: when do_write is False nothing is written; when True,
        # in_range holds so `cur` is a safe column index.
        store_col = tl.where(do_write, cur, 0)
        tl.store(
            scores_ptr + b * scores_row_stride + store_col,
            float("inf"),
            mask=do_write,
        )


def _force_include_current_slot(
    topk_scores: torch.Tensor,  # [bs, max_seq_len] authoritative post-reduce buffer
    seq_lens: torch.Tensor,  # int32 [bs]
    max_seq_len: int,
    bs: int,
    *,
    cosine: bool,
    scratch_qnorm: Optional[torch.Tensor],  # fp32 [max_bs, H] (cosine)
) -> None:
    """Fail-closed current-slot +inf force-include.

    0-alloc / graph-safe: a single Triton program per row writes +inf in place
    only for a width-covered current slot whose row q-norm is finite (cosine).
    The current slot's own key-norm is NOT gated — the current decode token's KV
    is valid by construction.
    """
    if bs <= 0 or max_seq_len <= 0:
        return
    if cosine:
        # Cosine needs the q-norm scratch to verify finiteness; absent it, fail
        # closed (do not force-include without the finite-norm guarantee).
        if scratch_qnorm is None:
            return
        H = int(scratch_qnorm.shape[1])
        _current_slot_force_include_kernel[(bs,)](
            topk_scores,
            seq_lens,
            scratch_qnorm,
            topk_scores.stride(0),
            scratch_qnorm.stride(0),
            max_seq_len,
            H,
            BLOCK_H=triton.next_power_of_2(H),
            COSINE=True,
        )
    else:
        # raw-dot: no norm gate; width fail-closed only. qnorm_ptr is never
        # dereferenced (COSINE=False is constexpr) so pass the scores buffer.
        _current_slot_force_include_kernel[(bs,)](
            topk_scores,
            seq_lens,
            topk_scores,
            topk_scores.stride(0),
            0,
            max_seq_len,
            1,
            BLOCK_H=1,
            COSINE=False,
        )


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
    scratch_scores: Optional[torch.Tensor] = None,  # fp32 [max_bs, max_seq_len]
    per_request_valid: Optional[torch.Tensor] = None,  # bool [bs, max_seq_len]
    scratch_pv_mask: Optional[torch.Tensor] = None,  # bool [max_bs, max_seq_len]
    scratch_scores_bf16: Optional[torch.Tensor] = None,  # bf16 [max_bs, max_seq_len]
    radix_topk_scratch: Optional[dict] = None,  # topk_kernel scratch bundle
    topk_block: int = 1024,
    process_group=None,
    reduce_ca=None,
    score_reduce_bf16: bool = False,
    scorer_norm: str = "off",
    head_agg: str = "max",
    absorbed_latent_fp8: Optional[torch.Tensor] = None,
    absorbed_latent_scales: Optional[torch.Tensor] = None,
    absorbed_w_sel: Optional[torch.Tensor] = None,
    absorbed_latent: Optional[torch.Tensor] = None,
    scratch_absorbed_v: Optional[torch.Tensor] = None,  # fp32 [max_bs, H, kv_lora_rank]
    scratch_absorbed_qsel: Optional[torch.Tensor] = None,  # fp32 [max_bs, H, label_dim]
    scratch_absorbed_sel_i64: Optional[torch.Tensor] = None,  # int64 [H, label_dim]
    scratch_absorbed_q: Optional[torch.Tensor] = None,  # fp32 [max_bs, H, nope_dim]
    include_current_slot: bool = False,
    key_norm_cache: Optional[torch.Tensor] = None,  # fp32 [L, max_tokens, H] (cosine)
    scratch_qnorm: Optional[torch.Tensor] = None,  # fp32 [max_bs, H] (cosine)
    q_pe: Optional[
        torch.Tensor
    ] = None,  # fp32 [bs, H, rope_dim] post-RoPE (rope-aware)
    k_pe: Optional[
        torch.Tensor
    ] = None,  # bf16 [max_tokens, rope_dim] resident RoPE key
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture-safe selection that writes results into caller-owned buffers.

    The returned top-K comes from the absorbed-latent score read straight off
    the resident MLA latent — no labels gather. The absorbed identity holds only
    for scorer_norm="off", and the rope dims are excluded by construction (no-PE
    queries + K-noPE W_UK rows in ``absorbed_w_sel``). ``written`` is the
    slot-validity bitmap ``[L, T]`` so a reused physical slot's stale latent is
    masked to ``-inf`` until its fresh KV write lands.

    On CUDA with Triton + the score scratch and radix top-k bundle provided, the
    pipeline is allocation-free after a single warmup call: the paged absorbed
    kernel fills ``scratch_scores`` in place, the optional ``per_request_valid``
    mask and current-slot force-include write the authoritative buffer in place,
    and the sequence-order radix top-k emits ascending ``-1``-padded indices.

    Fallback path (CPU, or scratch missing): calls the eager
    :func:`absorbed_topk_select`.  This branch is intended for unit tests;
    do NOT route production graph capture through it.
    """
    bs = req_pool_indices.shape[0]
    device = queries.device

    # scorer_norm="off" is the raw absorbed dot; "cosine" divides each per-head
    # dot by the query/key norms in the same kernel (the numerator IS the raw
    # dot, so the identity still holds). Any other value is unsupported.
    assert scorer_norm in ("off", "cosine"), (
        "Double Sparsity selection supports scorer_norm in ('off', 'cosine'); "
        f"got {scorer_norm!r}."
    )
    cosine = scorer_norm == "cosine"
    assert absorbed_w_sel is not None, (
        "Double Sparsity selection requires absorbed_w_sel (the bind-time K-noPE "
        "W_UK projection)."
    )

    use_triton_fast = (
        _TRITON_AVAILABLE
        and device.type == "cuda"
        and scratch_scores is not None
        and radix_topk_scratch is not None
    )

    # CPU / no-scratch fallback (unit tests): the eager absorbed_topk_select
    # scores the resident latent without the in-place graph-state scratch. The
    # graph-safe path below fills scratch_scores in place and shares the same
    # reduce + radix top-k.
    if not use_triton_fast:
        if q_pe is not None or k_pe is not None:
            # Fail closed: the eager fallback does NOT score the rope term, so
            # returning here with rope inputs would silently produce a no-PE
            # selection. rope-aware scoring is only defined on the graph-safe
            # scratch path above.
            raise RuntimeError(
                "Double Sparsity rope-aware scoring (q_pe/k_pe) requires the "
                "graph-safe scratch path; the eager absorbed_topk_select fallback "
                "does not score the rope term. Refusing to return a no-PE selection "
                "with rope inputs."
            )
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
            scorer_norm=scorer_norm,
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
    assert (
        sel_layer.dtype == torch.int32
    ), f"channel_selection must be int32, got {sel_layer.dtype}"
    assert (
        w_layer.dtype == torch.float32
    ), f"channel_weights must be float32, got {w_layer.dtype}"
    assert (
        req_pool_indices.dtype == torch.int32
    ), f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
    assert (
        req_to_token.dtype == torch.int32
    ), f"req_to_token must be int32, got {req_to_token.dtype}"
    assert (
        seq_lens.dtype == torch.int32
    ), f"seq_lens must be int32, got {seq_lens.dtype}"

    # NVTX ranges name the three DS-specific cost buckets (logical score /
    # score all-reduce / top-k select) so profiles can attribute them without
    # kernel-name matching. Host-side annotations: they mark eager decode and
    # the capture-time launches; CUDA-graph replay does not re-emit them.
    scores_view = scratch_scores[:bs, :max_seq_len]
    # Score the logical positions straight from the resident latent into
    # scratch_scores IN PLACE (v_h built into scratch_absorbed_v allocation-free),
    # then reduce + radix top-k. Scales are required only for the fp8 path; the
    # bf16 KV path passes the dequantized k_nope directly (scales=None).
    assert absorbed_latent_fp8 is not None, (
        "Double Sparsity graph-safe selection requires the resident latent "
        "(absorbed_latent_fp8)."
    )
    assert (
        absorbed_latent_scales is not None
        or absorbed_latent_fp8.dtype == torch.bfloat16
    ), "fp8 resident latent requires absorbed_latent_scales."
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
    # Cosine denominator inputs: this layer's key-norm cache slice ([max_tokens, H])
    # and the query-norm scratch. Fail closed — cosine without them would silently
    # route as raw dot. The cache slice is a cheap host-side view (constant offset),
    # like channel_selection[layer_id] above; 0-alloc and graph-safe.
    if cosine:
        assert key_norm_cache is not None and scratch_qnorm is not None, (
            "Double Sparsity cosine selection requires key_norm_cache "
            "[L, max_tokens, H] and scratch_qnorm [max_bs, H]; one is None."
        )
        k_norm_cache_layer = key_norm_cache[layer_id]
    else:
        k_norm_cache_layer = None
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
        cosine=cosine,
        key_norm_cache=k_norm_cache_layer,
        scratch_qnorm=scratch_qnorm,
        q_pe=q_pe,
        k_pe=k_pe,
    )
    torch.cuda.nvtx.range_pop()

    # The radix selector upcasts score loads in-register, so the reduced bf16
    # buffer can be its authoritative input directly: the compared values are
    # bit-identical to the fp32 copy-back (bf16→fp32 is exact) and the copy-back
    # kernel disappears.
    bf16_used = score_reduce_bf16 and scratch_scores_bf16 is not None
    bf16_authoritative = bf16_used
    topk_scores = scores_view
    if (
        process_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
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
        assert (
            scratch_pv_mask is not None
        ), "per_request_valid requires scratch_pv_mask in graph-safe path"
        pv_view = scratch_pv_mask[:bs, :max_seq_len]
        # copy_ handles dtype conversion in-place (no allocation when shapes match).
        pv_view.copy_(per_request_valid)
        # In-place flip: True = valid → True = invalid; then masked_fill_(invalid, -inf).
        torch.logical_not(pv_view, out=pv_view)
        # Masks the AUTHORITATIVE buffer: bf16(-inf) upcasts to fp32(-inf),
        # so the masked selection is identical on either dtype.
        topk_scores.masked_fill_(pv_view, float("-inf"))

    # Force-include the current decode slot (logical position seq_len-1): the
    # slot-validity bitmap masks the freshly-allocated slot to -inf during
    # scoring, but the current token's own KV is valid at attention time. Override
    # its score to +inf AFTER the validity mask and BEFORE the top-k. Only
    # seq_len-1 is touched, so every other reused slot stays -inf-masked.
    # FAIL-CLOSED: +inf only for a width-covered seq_len-1 (no clamp) and, for
    # cosine, only when the row's q-norm is finite.
    if include_current_slot and max_seq_len > 0:
        _force_include_current_slot(
            topk_scores,
            seq_lens,
            max_seq_len,
            bs,
            cosine=cosine,
            scratch_qnorm=scratch_qnorm,
        )

    # Sequence-aware deterministic radix top-k: work proportional to each row's
    # live window, exact (score desc, pos asc) selection emitted in ascending
    # order directly. Fixed grids; allocation-free with the scratch bundle.
    torch.cuda.nvtx.range_push("ds_topk_select")
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
    torch.cuda.nvtx.range_pop()

    return out_indices, out_lengths
