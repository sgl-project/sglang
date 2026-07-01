"""CUDA graph capture context manager + forward-context propagation.

Owns two pieces of cross-cutting state used by *every* piecewise-style
backend (currently breakable + tc_piecewise):

* _in_tc_piecewise_cuda_graph — a process-global flag set true while we
  are inside the capture or replay window of a piecewise CUDA graph.
  Read by model code that needs to take the static-buffer / fixed-shape
  branch. See refactor/plan.md §6.5 for the full semantics.
* TcPiecewiseForwardContext — a dataclass propagated across attention/MoE
  layers during capture and replay so that submodules can reach the
  current ForwardBatch and per-layer metadata without threading
  arguments through every call site. Named TcPiecewise… (matches
  Backend.TC_PIECEWISE + enable_tc_piecewise_cuda_graph) to
  disambiguate from the per-forward-call
  sglang.srt.model_executor.forward_context.ForwardContext.

This module deliberately does **not** own torch.compile-specific state
(warmup flag, capture stream); those live in compilation/compile_phase.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

from sglang.srt.model_executor.runner_backend_utils import (
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_in_tc_piecewise_cuda_graph = False


def is_in_tc_piecewise_cuda_graph() -> bool:
    """True while inside tc_piecewise CUDA graph capture/replay."""
    return _in_tc_piecewise_cuda_graph


@contextmanager
def enable_tc_piecewise_cuda_graph():
    """Mark the enclosed scope as "we are inside a piecewise CUDA graph
    capture/replay". Sets _in_tc_piecewise_cuda_graph true for the duration.
    """
    global _in_tc_piecewise_cuda_graph
    _in_tc_piecewise_cuda_graph = True
    try:
        yield
    finally:
        _in_tc_piecewise_cuda_graph = False


@dataclass
class TcPiecewiseForwardContext:
    forward_batch: Optional[ForwardBatch] = None
    attention_layers: Optional[List[Any]] = field(default=None)
    quant_config: Any = None
    moe_layers: Optional[List[Any]] = field(default=None)
    moe_fusions: Optional[List[Any]] = field(default=None)
    dsa_indexers: Optional[List[Any]] = field(default=None)
    num_tokens: Optional[int] = None
    raw_num_tokens: Optional[int] = None
    mamba_track_variant: Optional[bool] = None


_tc_piecewise_forward_context: Optional[TcPiecewiseForwardContext] = None


def get_tc_piecewise_forward_context() -> Optional[TcPiecewiseForwardContext]:
    return _tc_piecewise_forward_context


@contextmanager
def set_tc_piecewise_forward_context(
    forward_batch: ForwardBatch,
    attention_layers: List[Any],
    quant_config: Any,
    moe_layers: List[Any],
    moe_fusions: List[Any],
    dsa_indexers: Optional[List[Any]] = None,
    num_tokens: Optional[int] = None,
    raw_num_tokens: Optional[int] = None,
    mamba_track_variant: Optional[bool] = None,
):
    global _tc_piecewise_forward_context
    _tc_piecewise_forward_context = TcPiecewiseForwardContext(
        forward_batch=forward_batch,
        attention_layers=attention_layers,
        quant_config=quant_config,
        moe_layers=moe_layers,
        moe_fusions=moe_fusions,
        dsa_indexers=dsa_indexers,
        num_tokens=num_tokens,
        raw_num_tokens=raw_num_tokens,
        mamba_track_variant=mamba_track_variant,
    )
    try:
        yield
    finally:
        _tc_piecewise_forward_context = None


TC_PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Piecewise CUDA graph failed.\n" + PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG
)
