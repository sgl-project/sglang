"""Per-batch-size CUDA-graph capture of the multi-layer EAGLE draft tree-select
glue (``MultiLayerEagleDraftWorker.draft_forward``).

That glue (``select_top_k_tokens`` + the per-step token/score/parent assembly +
``topk`` / ``sort`` / ``gather`` / ``cat``) runs eagerly between the draft-extend
graph and the verify graph — a handful of tiny launches per decode step. Its
output is a pure function of ``(topk_p, topk_index)`` — the repeat-interleaved
hidden state is discarded — so it captures cleanly into a per-bs CUDA graph
whose replay costs ~one launch instead of the per-op launches. ``hidden_states``
is deliberately NOT part of the graph: its leading dim is the token count (not
bs), so it varies between calls, and it does not affect the output; the graph
runs the core with ``hidden=None``.

Capture is lazy per (shape, dtype) key, via the canonical
warmup-on-side-stream → capture pattern, with ``capture_error_mode="thread_local"``
so the overlap scheduler's concurrent kernels on other threads do not trip
capture-safety. Each freshly captured graph is bit-exact self-checked ONCE (on
the first inputs for that key) against the eager path run with the real hidden
state; correctness for later same-key inputs relies on the captured ops being
deterministic, RNG-free and hidden-independent — which holds for the gated
topk==1 chain (a future core change that broke that would need re-validation).
Any capture failure or self-check mismatch permanently falls back to eager for
that key, so this can never change results.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

CoreFn = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class _Captured:
    __slots__ = ("graph", "in_p", "in_i", "out")

    def __init__(self, graph, in_p, in_i, out):
        self.graph = graph
        self.in_p = in_p
        self.in_i = in_i
        self.out = out


class DraftForwardCudaGraph:
    def __init__(self, core_fn: CoreFn):
        self._core = core_fn
        self._graphs: Dict[tuple, Optional[_Captured]] = {}

    def run(
        self,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            tuple(topk_p.shape),
            topk_p.dtype,
            tuple(topk_index.shape),
            topk_index.dtype,
        )
        if key not in self._graphs:
            self._graphs[key] = self._capture(topk_p, topk_index, hidden_states)
        cap = self._graphs[key]
        if cap is None:
            return self._core(topk_p, topk_index, hidden_states)
        cap.in_p.copy_(topk_p)
        cap.in_i.copy_(topk_index)
        cap.graph.replay()
        return tuple(o.clone() for o in cap.out)

    def _capture(self, topk_p, topk_index, hidden_states) -> Optional[_Captured]:
        try:
            in_p = topk_p.clone()
            in_i = topk_index.clone()

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._core(in_p, in_i, None)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                out = self._core(in_p, in_i, None)

            in_p.copy_(topk_p)
            in_i.copy_(topk_index)
            graph.replay()
            ref = self._core(topk_p, topk_index, hidden_states)
            if len(out) != len(ref) or any(
                not torch.equal(o, r) for o, r in zip(out, ref)
            ):
                return None
            return _Captured(graph, in_p, in_i, out)
        except Exception:
            return None
