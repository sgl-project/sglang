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
  regression test from AC-6: any ``torch.empty(...)`` / ``torch.zeros(...)``
  call inside the region raises, proving the rule is enforced rather than
  just declared.
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


def allocate_graph_state(
    *,
    max_bs: int,
    max_top_k: int,
    num_score_blocks: int = 1,
    partial_topk: int = 0,
    device: Optional[torch.device] = None,
) -> DSGraphState:
    """Pre-allocate replay-stable buffers for the DS decode path.

    ``max_bs`` is the worst-case batch size at the configured operating
    concurrencies (16 / 32 / 64 per the plan). ``num_score_blocks`` and
    ``partial_topk`` size the two-stage scratch; default 0 disables the
    stage-1 partial buffers (single-pass selection path).
    """

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
    scratch_scores = None
    scratch_indices = None
    if num_score_blocks > 0 and partial_topk > 0:
        scratch_scores = torch.zeros(
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

    return DSGraphState(
        selected_indices=selected,
        valid_lengths=valid,
        scratch_partial_scores=scratch_scores,
        scratch_partial_indices=scratch_indices,
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
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor]]:
    """Capture one ``retrieve_topk`` call and return a replayable closure.

    The captured region writes into the pre-allocated ``state`` buffers.
    Replay reuses the same buffers; the returned closure returns the same
    tensors so callers can read updated values.

    On non-CUDA devices the function returns an eager closure that does no
    capture; this keeps unit tests portable.
    """

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
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out_idx, out_len = selector.retrieve_topk(
            queries=queries,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            seq_lens=seq_lens,
        )
        bs = out_idx.shape[0]
        mtk = out_idx.shape[1]
        state.selected_indices[:bs, :mtk].copy_(out_idx)
        state.valid_lengths[:bs].copy_(out_len)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_idx, out_len = selector.retrieve_topk(
            queries=queries,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            seq_lens=seq_lens,
        )
        bs = out_idx.shape[0]
        mtk = out_idx.shape[1]
        state.selected_indices[:bs, :mtk].copy_(out_idx)
        state.valid_lengths[:bs].copy_(out_len)

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
