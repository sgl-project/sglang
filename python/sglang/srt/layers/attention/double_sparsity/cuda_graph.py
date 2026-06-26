"""CUDA-graph piecewise capture for the Double Sparsity decode path.

The DS selection pipeline must be replay-stable: every shape, every
device-side branch, and every allocation must be deterministic across
capture and replay. The replay-stability contract requires:

1. Static output buffers — ``selected_indices: [bs, max_top_k]`` (-1
   padded) and ``valid_lengths: [bs]`` are pre-allocated and reused
   across capture and replay.
2. Pre-allocated scratch — stage-1 partial-topk and stage-2 merge
   scratch tensors are allocated to the worst-case batch before capture.
3. Device-side branching only — all conditional logic inside the
   captured region uses ``tl.where`` / mask multiplies / kernel-internal
   predication. The captured Python region contains zero ``if``
   statements that read CUDA tensor values to the host.
4. ``max_top_k < top_k`` — caller validates at startup, not at capture.
5. The eager (per-step) path produces identical output to graph replay
   on a deterministic fixture.

This module exposes:

* :class:`DSGraphState` — owns the static buffers + scratch.
* :func:`capture_decode_step` — convenience wrapper that captures one
  decode-step call to ``selector.retrieve_topk`` and returns a replayable
  closure.
* :func:`assert_no_alloc_in_region` — context manager used by the
  CUDA-graph allocation regression probe: any ``torch.empty(...)`` /
  ``torch.zeros(...)`` call inside the region raises, proving the rule
  is enforced rather than just declared.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.double_sparsity.selector import (
        DoubleSparsitySelector,
    )

logger = logging.getLogger(__name__)


@dataclass
class DSGraphState:
    """Static buffers + scratch that survive CUDA graph capture / replay."""

    selected_indices: torch.Tensor  # int32 [max_bs, max_top_k], -1 padded
    valid_lengths: torch.Tensor  # int32 [max_bs]
    scratch_partial_scores: Optional[torch.Tensor] = (
        None  # [max_bs, num_blocks, partial_k]
    )
    scratch_partial_indices: Optional[torch.Tensor] = (
        None  # [max_bs, num_blocks, partial_k]
    )
    max_seq_len: int = 0  # static sequence width for graph-safe logical scoring

    # Allocation-free graph-safe path scratch (sized at allocate_graph_state time).
    # Populated when num_local_heads > 0, label_dim > 0, max_seq_len > 0.
    scratch_scores: Optional[torch.Tensor] = None  # fp32 [max_bs, max_seq_len]
    scratch_topk_values: Optional[torch.Tensor] = None  # fp32 [max_bs, max_top_k]
    scratch_topk_indices: Optional[torch.Tensor] = None  # int64 [max_bs, max_top_k]
    scratch_invalid_mask: Optional[torch.Tensor] = None  # bool [max_bs, max_top_k]
    scratch_sorted_vals: Optional[torch.Tensor] = None  # int64 [max_bs, max_top_k]
    scratch_boundary: Optional[torch.Tensor] = None  # int64 [max_bs, 1]
    scratch_valid_i64: Optional[torch.Tensor] = None  # int64 [max_bs, 1]
    scratch_pv_mask: Optional[torch.Tensor] = None  # bool [max_bs, max_seq_len]
    scratch_throwaway_idx: Optional[torch.Tensor] = None  # int64 [max_bs, max_top_k]
    # bf16 transport scratch for the cross-TP score reduce (score_reduce_dtype
    # == "bf16"): the fp32 scores are cast into this view, reduced, and cast
    # back — halving the reduce bytes over the static score width.
    scratch_scores_bf16: Optional[torch.Tensor] = None  # bf16 [max_bs, max_seq_len]
    # Absorbed-latent selection scratch. The query-side latent projection v_h is
    # built into scratch_absorbed_v in place, scored straight into scratch_scores
    # (paged fp8, in place), then the reduce + radix top-k runs.
    # scratch_absorbed_qsel holds the weighted channel-gathered query;
    # scratch_absorbed_sel_i64 is the int64 layer mask (copied from the int32
    # channel selection) so the gather does no per-step .long() allocation.
    # scratch_absorbed_v: fp32 [max_bs, num_local_heads, kv_lora_rank]
    scratch_absorbed_v: Optional[torch.Tensor] = None
    # scratch_absorbed_qsel: fp32 [max_bs, num_local_heads, label_dim]
    scratch_absorbed_qsel: Optional[torch.Tensor] = None
    # scratch_absorbed_sel_i64: int64 [num_local_heads, label_dim]
    scratch_absorbed_sel_i64: Optional[torch.Tensor] = None
    # scratch_absorbed_q: fp32 [max_bs, num_local_heads, qk_nope_head_dim] — the
    # served bf16/fp16 query is cast into this in place (copy_), so the absorbed
    # v_h build never calls queries.to(torch.float32) on the hot path.
    scratch_absorbed_q: Optional[torch.Tensor] = None
    # scratch_qnorm: fp32 [max_bs, num_local_heads] — per (batch, head) query norm
    # for the cosine denominator, L2-normed in place from scratch_absorbed_qsel
    # (the weighted channel-gather) before the absorbed paged score kernel runs.
    scratch_qnorm: Optional[torch.Tensor] = None
    # Scratch bundle for the sequence-aware deterministic radix top-k
    # (topk_kernel.select_topk_sequence_order_triton). When present, the
    # graph-safe selection replaces the two full-width torch.topk passes.
    scratch_topk_hist: Optional[torch.Tensor] = None  # int32 [max_bs, 256]
    scratch_topk_key_prefix: Optional[torch.Tensor] = None  # int64 [max_bs]
    scratch_topk_quota: Optional[torch.Tensor] = None  # int32 [max_bs]
    scratch_topk_block_above: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    scratch_topk_block_tie: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    scratch_topk_above_pref: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    scratch_topk_tie_pref: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    topk_block: int = 1024
    # Production input scratch — `forward_batch.req_pool_indices` is int64 in
    # production (scheduler + cuda_graph_runner.py:178) but the captured
    # selector region requires int32. `_select_topk_indices` does an in-place
    # copy_() into these views before calling retrieve_topk_graph_safe.
    scratch_req_pool_indices: Optional[torch.Tensor] = None  # int32 [max_bs]
    scratch_seq_lens: Optional[torch.Tensor] = None  # int32 [max_bs]
    # Per-row current decode logical index (seq_len-1), built in place for the
    # include_current_slot force-include scatter. int64 for scatter_'s index dtype.
    scratch_cur_index: Optional[torch.Tensor] = None  # int64 [max_bs]
    # logical_to_physical's Triton kernel atomically accumulates the
    # bad-req_pool count here. Zeroed on every call.
    lp_error_scratch: Optional[torch.Tensor] = None  # int32 [1]

    # Host-side replay identity for the selection-capture dump. The DSA
    # backend's pre-replay metadata init stamps these (plain Python values,
    # never touched inside a captured region): `last_replay_graph_key` is the
    # runner's graph-variant key this state belongs to (today the padded
    # decode batch size). Eager forwards allocate a fresh DSGraphState per
    # forward and never stamp it, so `last_replay_graph_key is None`
    # distinguishes the eager path from graph replay in the dumps.
    last_replay_graph_key: Optional[object] = None
    replay_prep_count: int = 0


def allocate_graph_state(
    *,
    max_bs: int,
    max_top_k: int,
    max_seq_len: int = 0,
    num_score_blocks: int = 1,
    partial_topk: int = 0,
    num_local_heads: int = 0,
    label_dim: int = 0,
    score_reduce_bf16: bool = False,
    topk_block: int = 1024,
    kv_lora_rank: int = 0,
    qk_nope_head_dim: int = 0,
    device: Optional[torch.device] = None,
) -> DSGraphState:
    """Pre-allocate replay-stable buffers for the DS decode path.

    ``max_bs`` is the worst-case batch size at the configured operating
    concurrencies (16 / 32 / 64 per the plan). ``num_score_blocks`` and
    ``partial_topk`` size the two-stage scratch; default 0 disables the
    stage-1 partial buffers (single-pass selection path).

    ``max_seq_len`` sets :attr:`DSGraphState.max_seq_len`, which
    ``capture_decode_step`` uses as a static Python int to avoid the
    ``seq_lens.max().item()`` host sync during CUDA graph capture. Set to
    the maximum sequence length at the configured concurrency.

    When ``max_seq_len > 0``, the allocation-free graph-safe path scratch
    buffers (``scratch_scores``, ``scratch_topk_values``,
    ``scratch_topk_indices``, ``scratch_invalid_mask``, ``scratch_sorted_vals``,
    ``scratch_boundary``, ``scratch_valid_i64``, ``scratch_pv_mask``) are also
    allocated, sized to the worst-case ``(max_bs, max_seq_len)`` / ``(max_bs,
    max_top_k)``.  ``num_local_heads`` and ``label_dim`` size the absorbed-latent
    scratch; ``kv_lora_rank`` is the latent width for ``scratch_absorbed_v``,
    ``qk_nope_head_dim`` is the served query width for ``scratch_absorbed_q``.
    """
    if max_bs <= 0:
        raise ValueError(f"max_bs must be positive, got {max_bs}.")
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = torch.full((max_bs, max_top_k), -1, dtype=torch.int32, device=device)
    valid = torch.zeros((max_bs,), dtype=torch.int32, device=device)
    scratch_scores_partial = None
    scratch_indices = None
    if num_score_blocks > 0 and partial_topk > 0:
        scratch_scores_partial = torch.zeros(
            (max_bs, num_score_blocks, partial_topk),
            dtype=torch.float32,
            device=device,
        )
        scratch_indices = torch.full(
            (max_bs, num_score_blocks, partial_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )

    # Allocation-free graph-safe scratch (only when max_seq_len is known).
    scratch_scores = None
    scratch_topk_values = None
    scratch_topk_indices = None
    scratch_invalid_mask = None
    scratch_sorted_vals = None
    scratch_boundary = None
    scratch_valid_i64 = None
    scratch_pv_mask = None
    scratch_throwaway_idx = None
    scratch_scores_bf16 = None
    scratch_absorbed_v = None
    scratch_absorbed_qsel = None
    scratch_absorbed_sel_i64 = None
    scratch_absorbed_q = None
    scratch_qnorm = None
    scratch_req_pool_indices = None
    scratch_seq_lens = None
    scratch_cur_index = None
    lp_error_scratch = None
    scratch_topk_hist = None
    scratch_topk_key_prefix = None
    scratch_topk_quota = None
    scratch_topk_block_above = None
    scratch_topk_block_tie = None
    scratch_topk_above_pref = None
    scratch_topk_tie_pref = None
    if max_seq_len > 0:
        scratch_scores = torch.zeros(
            (max_bs, max_seq_len),
            dtype=torch.float32,
            device=device,
        )
        scratch_topk_values = torch.zeros(
            (max_bs, max_top_k),
            dtype=torch.float32,
            device=device,
        )
        scratch_topk_indices = torch.zeros(
            (max_bs, max_top_k),
            dtype=torch.int64,
            device=device,
        )
        scratch_invalid_mask = torch.zeros(
            (max_bs, max_top_k),
            dtype=torch.bool,
            device=device,
        )
        scratch_sorted_vals = torch.zeros(
            (max_bs, max_top_k),
            dtype=torch.int64,
            device=device,
        )
        scratch_boundary = torch.full(
            (max_bs, 1),
            max_seq_len,
            dtype=torch.int64,
            device=device,
        )
        scratch_valid_i64 = torch.zeros(
            (max_bs, 1),
            dtype=torch.int64,
            device=device,
        )
        scratch_pv_mask = torch.zeros(
            (max_bs, max_seq_len),
            dtype=torch.bool,
            device=device,
        )
        scratch_throwaway_idx = torch.zeros(
            (max_bs, max_top_k),
            dtype=torch.int64,
            device=device,
        )
        scratch_req_pool_indices = torch.zeros(
            (max_bs,),
            dtype=torch.int32,
            device=device,
        )
        scratch_seq_lens = torch.zeros(
            (max_bs,),
            dtype=torch.int32,
            device=device,
        )
        scratch_cur_index = torch.zeros(
            (max_bs,),
            dtype=torch.int64,
            device=device,
        )
        lp_error_scratch = torch.zeros((1,), dtype=torch.int32, device=device)
        if score_reduce_bf16:
            scratch_scores_bf16 = torch.zeros(
                (max_bs, max_seq_len),
                dtype=torch.bfloat16,
                device=device,
            )
        # Absorbed-latent scratch: v_h ([max_bs, H, lora]), the weighted
        # channel-gathered query ([max_bs, H, label_dim]), and the int64 layer
        # mask ([H, label_dim]).
        if num_local_heads > 0 and kv_lora_rank > 0 and label_dim > 0:
            scratch_absorbed_v = torch.zeros(
                (max_bs, num_local_heads, kv_lora_rank),
                dtype=torch.float32,
                device=device,
            )
            scratch_absorbed_qsel = torch.zeros(
                (max_bs, num_local_heads, label_dim),
                dtype=torch.float32,
                device=device,
            )
            scratch_absorbed_sel_i64 = torch.zeros(
                (num_local_heads, label_dim),
                dtype=torch.int64,
                device=device,
            )
            # The served query is bf16/fp16 [bs, H, qk_nope_head_dim]; cast it into
            # this fp32 scratch in place so the v_h build never allocates. Fall back
            # to label_dim width only when qk_nope_head_dim was not supplied (CPU
            # unit fixtures); the production backend always passes it.
            _q_width = qk_nope_head_dim if qk_nope_head_dim > 0 else label_dim
            scratch_absorbed_q = torch.zeros(
                (max_bs, num_local_heads, _q_width),
                dtype=torch.float32,
                device=device,
            )
            # Per (batch, head) query norm for the cosine denominator — L2-normed
            # in place from scratch_absorbed_qsel before the paged score kernel.
            scratch_qnorm = torch.zeros(
                (max_bs, num_local_heads),
                dtype=torch.float32,
                device=device,
            )
        topk_nblocks = (max_seq_len + topk_block - 1) // topk_block
        scratch_topk_hist = torch.zeros(
            (max_bs, 256),
            dtype=torch.int32,
            device=device,
        )
        scratch_topk_key_prefix = torch.zeros(
            (max_bs,),
            dtype=torch.int64,
            device=device,
        )
        scratch_topk_quota = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        scratch_topk_block_above = torch.zeros(
            (max_bs, topk_nblocks),
            dtype=torch.int32,
            device=device,
        )
        scratch_topk_block_tie = torch.zeros(
            (max_bs, topk_nblocks),
            dtype=torch.int32,
            device=device,
        )
        scratch_topk_above_pref = torch.zeros(
            (max_bs, topk_nblocks),
            dtype=torch.int32,
            device=device,
        )
        scratch_topk_tie_pref = torch.zeros(
            (max_bs, topk_nblocks),
            dtype=torch.int32,
            device=device,
        )

    return DSGraphState(
        selected_indices=selected,
        valid_lengths=valid,
        scratch_partial_scores=scratch_scores_partial,
        scratch_partial_indices=scratch_indices,
        max_seq_len=max_seq_len,
        scratch_scores=scratch_scores,
        scratch_topk_values=scratch_topk_values,
        scratch_topk_indices=scratch_topk_indices,
        scratch_invalid_mask=scratch_invalid_mask,
        scratch_sorted_vals=scratch_sorted_vals,
        scratch_boundary=scratch_boundary,
        scratch_valid_i64=scratch_valid_i64,
        scratch_pv_mask=scratch_pv_mask,
        scratch_throwaway_idx=scratch_throwaway_idx,
        scratch_scores_bf16=scratch_scores_bf16,
        scratch_absorbed_v=scratch_absorbed_v,
        scratch_absorbed_qsel=scratch_absorbed_qsel,
        scratch_absorbed_sel_i64=scratch_absorbed_sel_i64,
        scratch_absorbed_q=scratch_absorbed_q,
        scratch_qnorm=scratch_qnorm,
        scratch_topk_hist=scratch_topk_hist,
        scratch_topk_key_prefix=scratch_topk_key_prefix,
        scratch_topk_quota=scratch_topk_quota,
        scratch_topk_block_above=scratch_topk_block_above,
        scratch_topk_block_tie=scratch_topk_block_tie,
        scratch_topk_above_pref=scratch_topk_above_pref,
        scratch_topk_tie_pref=scratch_topk_tie_pref,
        topk_block=topk_block,
        scratch_req_pool_indices=scratch_req_pool_indices,
        scratch_seq_lens=scratch_seq_lens,
        scratch_cur_index=scratch_cur_index,
        lp_error_scratch=lp_error_scratch,
    )


def radix_topk_scratch(state: Optional[DSGraphState]) -> Optional[dict]:
    """The radix top-k scratch bundle of a graph state, as kwargs for
    ``topk_kernel.select_topk_sequence_order_triton`` — or None when the
    state has no bundle (legacy torch.topk pipeline)."""
    if state is None or state.scratch_topk_hist is None:
        return None
    return {
        "scratch_hist": state.scratch_topk_hist,
        "scratch_key_prefix": state.scratch_topk_key_prefix,
        "scratch_quota": state.scratch_topk_quota,
        "scratch_block_above": state.scratch_topk_block_above,
        "scratch_block_tie": state.scratch_topk_block_tie,
        "scratch_above_pref": state.scratch_topk_above_pref,
        "scratch_tie_pref": state.scratch_topk_tie_pref,
    }


def capture_decode_step(
    selector: DoubleSparsitySelector,
    *,
    state: DSGraphState,
    queries: torch.Tensor,
    layer_id: int,
    req_pool_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: Optional[torch.Tensor] = None,
    max_seq_len: int = 0,
    absorbed_latent_fp8: Optional[torch.Tensor] = None,
    absorbed_latent_scales: Optional[torch.Tensor] = None,
    key_norm_cache: Optional[torch.Tensor] = None,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor]]:
    """Capture one ``retrieve_topk`` call and return a replayable closure.

    The captured region writes into the pre-allocated ``state`` buffers.
    Replay reuses the same buffers; the returned closure returns the same
    tensors so callers can read updated values.

    ``req_to_token`` must be provided for the logical-domain selection path
    (production TP operation). Without it the selector falls back to the
    physical-domain path, which produces the wrong top-K during graph replay
    when physical slots differ across TP ranks.

    ``max_seq_len`` provides the static sequence width for the graph-safe
    logical scoring path.  Resolution priority: ``state.max_seq_len`` (set
    at :func:`allocate_graph_state` time) > ``max_seq_len`` parameter >
    dynamic ``seq_lens.max().item()`` (host sync — only safe on CPU / before
    the CUDA graph capture region).

    On non-CUDA devices the function returns an eager closure that does no
    capture; this keeps unit tests portable.
    """

    if getattr(getattr(selector, "config", None), "rope_aware_score", False):
        # Fail closed: this convenience capture wrapper does not thread the rope
        # query/key, so capturing here while rope_aware_score is on would score
        # no-PE. The production rope-aware path is the model-side graph-safe
        # selector (deepseek_v2._select_topk_indices), not this helper.
        raise RuntimeError(
            "Double Sparsity 'rope_aware_score' is not wired through "
            "capture_decode_step (it does not thread the rope query/key); refusing "
            "to capture a no-PE selection while the flag is on."
        )

    # Resolve static max_seq_len BEFORE any capture region.
    # state.max_seq_len is the preferred source (set at allocate_graph_state time).
    # Fall through to the parameter, then to a one-time .item() that is safe here
    # because we have not yet entered a torch.cuda.graph() block.
    _max_seq_len: int = state.max_seq_len if state.max_seq_len > 0 else max_seq_len
    if _max_seq_len <= 0 and seq_lens.numel() > 0:
        _max_seq_len = int(seq_lens.max().item())

    if state.selected_indices.device.type != "cuda":
        logger.debug(
            "capture_decode_step: state on %s, skipping CUDA-graph capture.",
            state.selected_indices.device,
        )

        def _eager_replay() -> Tuple[torch.Tensor, torch.Tensor]:
            out_idx, out_len = selector.retrieve_topk(
                queries=queries,
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                sparse_mask=sparse_mask,
                seq_lens=seq_lens,
                req_to_token=req_to_token,
                max_seq_len=_max_seq_len,
            )
            bs = out_idx.shape[0]
            mtk = out_idx.shape[1]
            state.selected_indices[:bs, :mtk].copy_(out_idx)
            state.valid_lengths[:bs].copy_(out_len)
            return state.selected_indices, state.valid_lengths

        # Run once eagerly to warm up.
        _eager_replay()
        return _eager_replay

    # CUDA path: warm up, then capture.
    # When the selector is bound AND scratch is allocated, route through the
    # allocation-free graph-safe retrieve_topk_graph_safe. Otherwise fall back
    # to selector.retrieve_topk (allocates during capture; not zero-alloc).
    # Selection's graph-safe precondition is the bind-time absorbed projection +
    # the caller-supplied resident fp8 latent.
    use_graph_safe = (
        getattr(selector, "absorbed_w_sel", None) is not None
        and getattr(selector, "channel_mask", None) is not None
        and state.scratch_scores is not None
        and state.scratch_absorbed_v is not None
        and absorbed_latent_fp8 is not None
        and absorbed_latent_scales is not None
    )

    # Fail-fast future-proof guard. Every non-learned variant is graph-safe —
    # head_agg (mean) in the absorbed score kernel, anchor_mode (recency/global/
    # strided) as a tensorized post-topK force-include — so this does not fire.
    # It remains so a future non-graph-safe variant can re-enable the eager
    # requirement.
    if use_graph_safe:
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            ds_scorer_is_graph_safe,
        )

        if not ds_scorer_is_graph_safe(getattr(selector, "config", None)):
            raise RuntimeError(
                "Double Sparsity: a non-graph-safe selector variant cannot be "
                "captured; serve with --disable-cuda-graph or use the default selector."
            )

    def _call_into_state() -> None:
        if use_graph_safe:
            from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
                retrieve_topk_graph_safe,
            )

            retrieve_topk_graph_safe(
                queries=queries,
                written=None,
                channel_selection=selector.channel_mask.channel_selection,
                channel_weights=selector.channel_mask.channel_weights,
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                max_seq_len=_max_seq_len,
                max_top_k=selector.max_top_k,
                out_indices=state.selected_indices,
                out_lengths=state.valid_lengths,
                scratch_scores=state.scratch_scores,
                scratch_topk_values=state.scratch_topk_values,
                scratch_topk_indices=state.scratch_topk_indices,
                scratch_invalid_mask=state.scratch_invalid_mask,
                scratch_sorted_vals=state.scratch_sorted_vals,
                scratch_boundary=state.scratch_boundary,
                scratch_valid_i64=state.scratch_valid_i64,
                per_request_valid=sparse_mask,
                scratch_pv_mask=state.scratch_pv_mask,
                scratch_throwaway_idx=state.scratch_throwaway_idx,
                scratch_scores_bf16=state.scratch_scores_bf16,
                radix_topk_scratch=radix_topk_scratch(state),
                topk_block=state.topk_block,
                process_group=getattr(selector, "process_group", None),
                reduce_ca=getattr(selector, "reduce_ca", None),
                score_reduce_bf16=(
                    getattr(selector.config, "score_reduce_dtype", "bf16") == "bf16"
                ),
                scorer_norm=getattr(selector.config, "scorer_norm", "off"),
                head_agg=getattr(selector.config, "head_agg", "max"),
                anchor_mode=getattr(selector.config, "anchor_mode", "off"),
                anchor_budget=getattr(selector.config, "anchor_budget", 0),
                include_current_slot=bool(
                    getattr(selector.config, "include_current_slot", False)
                ),
                scratch_cur_index=state.scratch_cur_index,
                absorbed_latent_fp8=absorbed_latent_fp8,
                absorbed_latent_scales=absorbed_latent_scales,
                absorbed_w_sel=getattr(selector, "absorbed_w_sel", None),
                scratch_absorbed_v=state.scratch_absorbed_v,
                scratch_absorbed_qsel=state.scratch_absorbed_qsel,
                scratch_absorbed_sel_i64=state.scratch_absorbed_sel_i64,
                scratch_absorbed_q=state.scratch_absorbed_q,
                key_norm_cache=key_norm_cache,
                scratch_qnorm=state.scratch_qnorm,
            )
        else:
            out_idx, out_len = selector.retrieve_topk(
                queries=queries,
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                sparse_mask=sparse_mask,
                seq_lens=seq_lens,
                req_to_token=req_to_token,
                max_seq_len=_max_seq_len,
            )
            bs = out_idx.shape[0]
            mtk = out_idx.shape[1]
            state.selected_indices[:bs, :mtk].copy_(out_idx)
            state.valid_lengths[:bs].copy_(out_len)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call_into_state()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_into_state()

    def _replay() -> Tuple[torch.Tensor, torch.Tensor]:
        graph.replay()
        return state.selected_indices, state.valid_lengths

    return _replay


@contextlib.contextmanager
def assert_no_alloc_in_region(label: str = "DS decode capture"):
    """Context manager that fails if any tensor is allocated in the body.

    Used by the CUDA-graph allocation regression probe: wrap the captured region; if any
    ``torch.empty(...)`` / ``torch.zeros(...)`` / ``torch.tensor(...)``
    is called, raise ``RuntimeError`` so the test detects the rule
    violation. Implementation: snapshot the CUDA caching-allocator counter
    via ``torch.cuda.memory_stats`` before / after; growth indicates new
    allocations. On non-CUDA devices, this is a no-op.
    """

    if not torch.cuda.is_available():
        yield
        return
    before = torch.cuda.memory_stats(device=None).get("allocation.all.allocated", 0)
    try:
        yield
    finally:
        after = torch.cuda.memory_stats(device=None).get("allocation.all.allocated", 0)
        delta = after - before
        if delta > 0:
            raise RuntimeError(
                f"{label}: new CUDA allocation detected inside the captured region "
                f"({delta} new allocations). CUDA-graph capture requires all "
                "scratch to be preallocated and all branching to be device-side."
            )
