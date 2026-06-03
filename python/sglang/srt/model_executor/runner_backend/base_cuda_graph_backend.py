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
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseCudaGraphBackend(ABC):
    """Capture/replay protocol for one cuda-graph backend. Pure
    interface — every method is abstract; the base provides no
    implementation or state. Concrete backends own their own state
    (per-shape ``_graphs`` / ``_outputs`` tables, memory saver,
    torch.compile config, etc.) and bind runner-derived handles
    (``_device_module``, ``_tp_group``, …) in their own ``__init__``.

    Each concrete backend takes the ``BaseCudaGraphRunner`` that owns it
    as the first constructor argument so it can pull the handles it
    needs (``cuda_graph_runner.device_module``,
    ``cuda_graph_runner.model_runner.tp_group``, …).

    Lifecycle:
        1. ``__init__(cuda_graph_runner, ...)`` — backend constructor.
           Binds runner-derived handles, allocates per-backend state.
        2. ``capture_session(stream)`` — context wrapping the runner's
           outer capture loop. Backend binds the stream / pool here and
           opens any per-backend "we are capturing now" flags.
        3. ``capture_one(shape_key, forward_fn, dummies)`` — record the
           replayable artifact for ``shape_key``. Called once per shape
           inside ``capture_session``.
        4. ``replay(shape_key, static_forward_batch, **kwargs)`` — invoke
           the captured artifact for ``shape_key`` with already-populated
           static buffers. May or may not consume ``static_forward_batch``
           depending on backend (Full / Breakable replay against static
           buffers and ignore it; TcPiecewise dispatches by shape via
           torch.compile and uses it).
        5. ``replay_session()`` — context wrapping replay-time model
           code. Sets per-backend global flags so model code takes the
           static-buffer / fixed-shape path. Backends without such
           a flag yield without doing anything.
        6. ``can_run(forward_batch, shape_key)`` — "can this backend
           replay for this batch at this shape?" Backends that maintain
           a per-shape ``_graphs`` table answer this with shape
           membership; TcPiecewise answers it as "always yes"
           (torch.compile manages its own cache). Subclasses can also
           AND in backend-specific eligibility (page-size constraints,
           sparsity caps, …).
        7. ``cleanup()`` — release pool, drop captured artifacts.
    """

    @abstractmethod
    def can_run(self, forward_batch: ForwardBatch, shape_key: Any) -> bool:
        """Can this backend replay for the given shape?"""

    @abstractmethod
    def capture_session(self, stream: torch.cuda.Stream) -> Iterator[None]:
        """Bind ``stream`` (and any pool handle) for the duration of the
        runner's outer capture loop. Implementations open their per-
        backend capture flag inside this context.
        """

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
    def replay_session(self) -> Iterator[None]:
        """Context wrapping replay-time model code. Sets per-backend
        global flags (``is_in_*_cuda_graph``) so model code takes the
        static-buffer / fixed-shape path. Backends without such a flag
        yield without doing anything.
        """

    @abstractmethod
    def cleanup(self) -> None: ...
