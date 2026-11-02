"""Admission accounting for multimodal preprocessing.

This module deliberately operates after request parsing.  It bounds work admitted
to the tokenizer-side multimodal preprocessing pipeline, but it does not bound
HTTP request-body parsing or memory retained by an ASGI server before the request
reaches :class:`TokenizerManager`.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Protocol


class _DoneCallbackFuture(Protocol):
    def add_done_callback(self, callback: Any) -> None: ...


class MultimodalPreprocessingRequestTooLarge(ValueError):
    """The request can never fit in this admission controller."""

    def __init__(self, item_count: int, max_inflight_items: int) -> None:
        self.item_count = item_count
        self.max_inflight_items = max_inflight_items
        super().__init__(
            f"request needs {item_count} item slots, but the limit is "
            f"{max_inflight_items}"
        )


class MultimodalPreprocessingBusy(RuntimeError):
    """The request fits by itself, but not alongside current work."""

    def __init__(
        self, item_count: int, inflight_items: int, max_inflight_items: int
    ) -> None:
        self.item_count = item_count
        self.inflight_items = inflight_items
        self.max_inflight_items = max_inflight_items
        super().__init__(
            f"request needs {item_count} item slots with {inflight_items}/"
            f"{max_inflight_items} currently reserved"
        )


_current_lease: contextvars.ContextVar[
    Optional[MultimodalPreprocessingAdmissionLease]
] = contextvars.ContextVar("current_mm_preprocessing_admission_lease", default=None)


def _count_media_items(value: Any) -> int:
    """Count media leaves while treating dictionaries and byte buffers as items."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_count_media_items(item) for item in value)
    return 1


def count_preprocessed_multimodal_items(request: Any) -> int:
    """Return the number of media items this normalized request will preprocess.

    Parallel sampling expands the normalized modality arrays, while the tokenizer
    manager preprocesses only the original ``batch_size`` inputs and reuses their
    features for the generated samples.  Slice those arrays back to that effective
    preprocessing batch so admission reflects actual work rather than output fanout.
    """
    is_single = getattr(request, "is_single", True)
    batch_size = getattr(request, "batch_size", 1)
    total = 0
    for field_name in ("image_data", "video_data", "audio_data"):
        value = getattr(request, field_name, None)
        if not is_single and isinstance(value, list):
            value = value[:batch_size]
        total += _count_media_items(value)
    return total


class MultimodalPreprocessingAdmissionLease:
    """An idempotently releasable reservation from an admission controller."""

    def __init__(
        self, controller: MultimodalPreprocessingAdmission, item_count: int
    ) -> None:
        self._controller: Optional[MultimodalPreprocessingAdmission] = controller
        self._owner_released = False
        self._pending_futures = 0
        self._lock = threading.Lock()
        self.item_count = item_count

    def release(self) -> None:
        with self._lock:
            controller = self._controller
            if controller is None or self._owner_released:
                return
            self._owner_released = True
            if self._pending_futures:
                return
            self._controller = None
        controller._release(self.item_count)

    def track_future(self, future: _DoneCallbackFuture) -> None:
        """Keep the reservation until submitted background work really finishes."""
        with self._lock:
            if self._controller is None:
                raise RuntimeError("cannot track work on a released reservation")
            self._pending_futures += 1
        # concurrent.futures.Future invokes callbacks synchronously when the
        # future is already complete. Register outside the lease lock so that
        # such a callback can safely decrement the pending count.
        try:
            future.add_done_callback(self._future_done)
        except BaseException:
            self._future_done(future)
            raise

    def _future_done(self, future: _DoneCallbackFuture) -> None:
        with self._lock:
            self._pending_futures -= 1
            if self._pending_futures < 0:
                raise RuntimeError(
                    "multimodal preprocessing future accounting underflow"
                )
            controller = (
                self._controller
                if self._owner_released and self._pending_futures == 0
                else None
            )
            if controller is not None:
                self._controller = None
        if controller is not None:
            controller._release(self.item_count)

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _current_lease.set(self)
        try:
            yield
        finally:
            _current_lease.reset(token)

    def __enter__(self) -> MultimodalPreprocessingAdmissionLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def track_mm_preprocessing_future(future: _DoneCallbackFuture) -> None:
    """Attach newly submitted executor work to the active request lease."""
    lease = _current_lease.get()
    if lease is not None:
        lease.track_future(future)


class MultimodalPreprocessingAdmission:
    """A non-blocking weighted limiter for tokenizer-side MM preprocessing."""

    def __init__(self, max_inflight_items: int) -> None:
        if max_inflight_items <= 0:
            raise ValueError("max_inflight_items must be positive")
        self.max_inflight_items = max_inflight_items
        self._inflight_items = 0
        self._lock = threading.Lock()

    @property
    def inflight_items(self) -> int:
        with self._lock:
            return self._inflight_items

    def acquire(self, item_count: int) -> MultimodalPreprocessingAdmissionLease:
        """Reserve items or classify a permanent limit from transient pressure."""
        if item_count <= 0:
            raise ValueError("item_count must be positive")
        with self._lock:
            if item_count > self.max_inflight_items:
                raise MultimodalPreprocessingRequestTooLarge(
                    item_count, self.max_inflight_items
                )
            if item_count > self.max_inflight_items - self._inflight_items:
                raise MultimodalPreprocessingBusy(
                    item_count, self._inflight_items, self.max_inflight_items
                )
            self._inflight_items += item_count
        return MultimodalPreprocessingAdmissionLease(self, item_count)

    def try_acquire(
        self, item_count: int
    ) -> Optional[MultimodalPreprocessingAdmissionLease]:
        """Reserve items, returning ``None`` only for transient pressure."""
        try:
            return self.acquire(item_count)
        except MultimodalPreprocessingBusy:
            return None

    def _release(self, item_count: int) -> None:
        with self._lock:
            if item_count > self._inflight_items:
                raise RuntimeError(
                    "multimodal preprocessing admission accounting underflow"
                )
            self._inflight_items -= item_count
