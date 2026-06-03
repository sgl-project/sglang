"""Backend interface for CUDA graph capture/replay.

A backend encapsulates *how* the model forward at one shape is captured
into a replayable artifact and how that artifact is invoked. The runner
above this interface is phase-aware (prefill vs decode) but
backend-agnostic — it never branches on backend type.

Today's three implementations:
- ``FullCudaGraphBackend``     — one ``torch.cuda.CUDAGraph`` per shape.
- ``BreakableCudaGraphBackend`` — segmented ``BreakableCUDAGraph`` per shape.
- ``TcPiecewiseCudaGraphBackend`` — torch.compile wraps the model;
  per-shape graphs live inside torch.compile's internal cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class BaseCudaGraphBackend(ABC):
    """Capture/replay protocol for one cuda-graph backend.

    Lifecycle:
        1. ``__init__(cuda_graph_runner, ...)`` — backend allocates its
           captured artifact tables and binds runner-derived handles
           (``device_module``, ``tp_group``). Subclasses do additional
           per-backend setup (memory saver, torch.compile, …) here.
        2. ``capture_session(stream)`` — context wrapping the runner's
           outer capture loop. Backend binds the stream/pool here and
           opens any per-backend "we are capturing now" flags.
        3. ``capture_one(shape_key, forward_fn, dummies)`` — record the
           replayable artifact for ``shape_key``. Called once per shape
           inside ``capture_session``.
        4. ``replay(shape_key, static_forward_batch, **kwargs)`` — invoke
           the captured artifact for ``shape_key`` with already-populated
           static buffers. May or may not consume ``static_forward_batch``
           depending on backend (Full/Breakable replay against static
           buffers and ignore it; TcPiecewise dispatches by shape via
           torch.compile and uses it).
        5. ``replay_session()`` — context wrapping replay-time model
           code. Sets per-backend global flags so model code takes the
           static-buffer / fixed-shape path.
        6. ``can_run(forward_batch, shape_key)`` — "can this backend
           replay for this batch at this shape?" Default: yes iff
           ``capture_one`` has produced a graph for ``shape_key``.
           Subclasses can override to AND in backend-specific
           eligibility (page-size constraints, sparsity caps, …).
        7. ``cleanup()`` — release pool, drop captured artifacts.
    """

    def __init__(self, cuda_graph_runner: BaseCudaGraphRunner) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None

    def can_run(self, forward_batch: ForwardBatch, shape_key: Any) -> bool:
        """Can this backend replay for the given shape? Default: yes iff
        ``capture_one`` has produced a graph for ``shape_key``."""
        return shape_key in self._graphs

    @abstractmethod
    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream) -> Iterator[None]:
        """Bind ``stream`` (and any pool handle) for the duration of the
        runner's outer capture loop. Implementations open their per-
        backend capture flag inside this context.
        """
        yield  # pragma: no cover

    @abstractmethod
    def capture_one(
        self,
        shape_key: Any,
        forward_fn,
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None: ...

    @abstractmethod
    def replay(
        self,
        shape_key: Any,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @contextmanager
    def replay_session(self):
        """Context wrapping replay-time model code. Sets per-backend
        global flags (``is_in_*_cuda_graph``) so model code takes the
        static-buffer / fixed-shape path. Default: no-op (Full doesn't
        set any flag).
        """
        yield
