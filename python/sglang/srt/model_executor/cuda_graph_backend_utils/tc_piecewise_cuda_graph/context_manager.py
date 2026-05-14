"""CUDA graph capture context manager + forward-context propagation.

Owns two pieces of cross-cutting state used by *every* piecewise-style
backend (currently breakable + tc_piecewise):

* ``_in_tc_piecewise_cuda_graph`` — a process-global flag set true while we
  are inside the capture or replay window of a piecewise CUDA graph.
  Read by model code that needs to take the static-buffer / fixed-shape
  branch. See ``refactor/plan.md`` §6.5 for the full semantics.
* ``ForwardContext`` — a dataclass propagated across attention/MoE
  layers during capture and replay so that submodules can reach the
  current ``ForwardBatch`` and per-layer metadata without threading
  arguments through every call site.

This module deliberately does **not** own torch.compile-specific state
(warmup flag, capture stream); those live in ``compilation/compile_phase.py``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_in_tc_piecewise_cuda_graph = False


def is_in_tc_piecewise_cuda_graph() -> bool:
    """True while inside tc_piecewise CUDA graph capture/replay."""
    return _in_tc_piecewise_cuda_graph


@contextmanager
def enable_tc_piecewise_cuda_graph():
    """Mark the enclosed scope as "we are inside a piecewise CUDA graph
    capture/replay". Sets ``_in_tc_piecewise_cuda_graph`` true for the duration.

    Errors during capture surface a hint that lets users disable the
    feature while filing a bug.
    """
    global _in_tc_piecewise_cuda_graph
    _in_tc_piecewise_cuda_graph = True
    try:
        yield
    except Exception as e:
        logger.error(
            "Piecewise CUDA Graph failed with error: %s\n%s",
            e,
            TC_PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG,
        )
        raise
    finally:
        _in_tc_piecewise_cuda_graph = False


@dataclass
class ForwardContext:
    forward_batch: Optional["ForwardBatch"] = None
    attention_layers: Optional[List[Any]] = field(default=None)
    quant_config: Any = None
    moe_layers: Optional[List[Any]] = field(default=None)
    moe_fusions: Optional[List[Any]] = field(default=None)


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> Optional[ForwardContext]:
    return _forward_context


@contextmanager
def set_forward_context(
    forward_batch: ForwardBatch,
    attention_layers: List[Any],
    quant_config: Any,
    moe_layers: List[Any],
    moe_fusions: List[Any],
):
    global _forward_context
    _forward_context = ForwardContext(
        forward_batch=forward_batch,
        attention_layers=attention_layers,
        quant_config=quant_config,
        moe_layers=moe_layers,
        moe_fusions=moe_fusions,
    )
    try:
        yield
    finally:
        _forward_context = None


TC_PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Piecewise CUDA Graph is enabled by default as an experimental feature.\n"
    "To work around this error, add --disable-piecewise-cuda-graph to your launch command.\n"
    "Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose"
)
