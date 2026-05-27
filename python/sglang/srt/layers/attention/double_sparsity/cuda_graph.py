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
    scratch_partial_scores: Optional[torch.Tensor] = None  # [max_bs, num_blocks, partial_k]
    scratch_partial_indices: Optional[torch.Tensor] = None  # [max_bs, num_blocks, partial_k]
    max_seq_len: int = 0  # static sequence width for graph-safe logical scoring

    # Allocation-free graph-safe path scratch (sized at allocate_graph_state time).
    # Populated when num_local_heads > 0, label_dim > 0, max_seq_len > 0.
    scratch_scores: Optional[torch.Tensor] = None         # fp32 [max_bs, max_seq_len]
    scratch_topk_values: Optional[torch.Tensor] = None    # fp32 [max_bs, max_top_k]
    scratch_topk_indices: Optional[torch.Tensor] = None   # int64 [max_bs, max_top_k]
    scratch_invalid_mask: Optional[torch.Tensor] = None   # bool [max_bs, max_top_k]
    scratch_sorted_vals: Optional[torch.Tensor] = None    # int64 [max_bs, max_top_k]
    scratch_boundary: Optional[torch.Tensor] = None       # int64 [max_bs, 1]
    scratch_valid_i64: Optional[torch.Tensor] = None      # int64 [max_bs, 1]
    scratch_pv_mask: Optional[torch.Tensor] = None        # bool [max_bs, max_seq_len]
    scratch_throwaway_idx: Optional[torch.Tensor] = None  # int64 [max_bs, max_top_k]


def allocate_graph_state(
    *,
    max_bs: int,
    max_top_k: int,
    max_seq_len: int = 0,
    num_score_blocks: int = 1,
    partial_topk: int = 0,
    num_local_heads: int = 0,
    label_dim: int = 0,
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
    max_top_k)``.  ``num_local_heads`` and ``label_dim`` are accepted for
    future-proofing the API contract; the scratch above does not actually
    depend on them (the Triton kernel reads heads/label_dim from the bound
    selector at call time).
    """
    del num_local_heads, label_dim  # API parity; not used for scratch sizing today.

    if max_bs <= 0:
        raise ValueError(f"max_bs must be positive, got {max_bs}.")
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = torch.full(
        (max_bs, max_top_k), -1, dtype=torch.int32, device=device
    )
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
    if max_seq_len > 0:
        scratch_scores = torch.zeros(
            (max_bs, max_seq_len), dtype=torch.float32, device=device,
        )
        scratch_topk_values = torch.zeros(
            (max_bs, max_top_k), dtype=torch.float32, device=device,
        )
        scratch_topk_indices = torch.zeros(
            (max_bs, max_top_k), dtype=torch.int64, device=device,
        )
        scratch_invalid_mask = torch.zeros(
            (max_bs, max_top_k), dtype=torch.bool, device=device,
        )
        scratch_sorted_vals = torch.zeros(
            (max_bs, max_top_k), dtype=torch.int64, device=device,
        )
        scratch_boundary = torch.full(
            (max_bs, 1), max_seq_len, dtype=torch.int64, device=device,
        )
        scratch_valid_i64 = torch.zeros(
            (max_bs, 1), dtype=torch.int64, device=device,
        )
        scratch_pv_mask = torch.zeros(
            (max_bs, max_seq_len), dtype=torch.bool, device=device,
        )
        scratch_throwaway_idx = torch.zeros(
            (max_bs, max_top_k), dtype=torch.int64, device=device,
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
    )


def capture_decode_step(
    selector: "DoubleSparsitySelector",
    *,
    state: DSGraphState,
    queries: torch.Tensor,
    layer_id: int,
    req_pool_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: Optional[torch.Tensor] = None,
    max_seq_len: int = 0,
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
    use_graph_safe = (
        getattr(selector, "token_label_table", None) is not None
        and getattr(selector, "channel_mask", None) is not None
        and state.scratch_scores is not None
    )

    def _call_into_state() -> None:
        if use_graph_safe:
            from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
                retrieve_topk_graph_safe,
            )

            retrieve_topk_graph_safe(
                queries=queries,
                token_signatures=selector.token_label_table.signatures,
                written=selector.token_label_table.written,
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
                process_group=getattr(selector, "process_group", None),
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
    before = torch.cuda.memory_stats(device=None).get(
        "allocation.all.allocated", 0
    )
    try:
        yield
    finally:
        after = torch.cuda.memory_stats(device=None).get(
            "allocation.all.allocated", 0
        )
        delta = after - before
        if delta > 0:
            raise RuntimeError(
                f"{label}: new CUDA allocation detected inside the captured region "
                f"({delta} new allocations). CUDA-graph capture requires all "
                "scratch to be preallocated and all branching to be device-side."
            )
