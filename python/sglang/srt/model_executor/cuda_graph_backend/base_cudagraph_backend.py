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
from typing import TYPE_CHECKING, Any, Iterator, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseCudaGraphBackend(ABC):
    """Capture/replay protocol for one cuda-graph backend.

    Lifecycle:
        1. ``prepare(runner)`` — one-time setup (wrap the model with
           torch.compile, install compilation hooks, allocate the pool
           handle, etc.).
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
        5. ``runtime_session()`` — context wrapping replay-time model
           code. Sets per-backend global flags so model code takes the
           static-buffer / fixed-shape path.
        6. ``can_run(forward_batch)`` — backend-level "is this batch
           supported" check. Runner ANDs with phase-level checks.
        7. ``has_shape(shape_key)`` — whether ``capture_one`` has been
           called for ``shape_key``.
        8. ``cleanup()`` — release pool, drop captured artifacts.
    """

    @abstractmethod
    def prepare(self, runner) -> None: ...

    @abstractmethod
    def can_run(self, forward_batch: "ForwardBatch") -> bool: ...

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
    ) -> None: ...

    @abstractmethod
    def has_shape(self, shape_key: Any) -> bool:
        """Whether ``capture_one`` has been called for this shape."""

    @abstractmethod
    def replay(
        self,
        shape_key: Any,
        static_forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @contextmanager
    def runtime_session(self):
        """Context wrapping replay-time model code. Sets per-backend
        global flags (``is_in_*_cuda_graph``) so model code takes the
        static-buffer / fixed-shape path. Default: no-op (Full doesn't
        set any flag).
        """
        yield

    # -----------------------------------------------------------------
    # Optional prefill hooks. Backends that need stable addresses for
    # captured prefill segments (today: only Breakable) override these.
    # The default implementations are no-ops / pass-through so the
    # PrefillCudaGraphRunner code path stays uniform.
    # -----------------------------------------------------------------
    def setup_prefill_state(self, runner) -> None:
        """One-time hook called by ``PrefillCudaGraphRunner.__init__``
        before ``prepare()``. Breakable allocates its static prefill buffers
        here; other backends no-op.
        """
        pass

    def populate_prefill_dummy_inputs(
        self,
        kwargs: dict,
        *,
        bs: int,
        num_tokens: int,
    ) -> None:
        """Mutate ``kwargs`` in place during capture-time dummy
        ``ForwardBatch`` construction. Breakable swaps in stable-address
        static tensors so captured segments read from fixed pointers;
        other backends no-op (runner falls back to fresh literals).
        """
        pass

    def commit_prefill_serving_inputs(self, forward_batch: "ForwardBatch") -> None:
        """Replay-time hook: copy serving-batch values into backend
        static buffers (Breakable); other backends no-op.
        """
        pass
