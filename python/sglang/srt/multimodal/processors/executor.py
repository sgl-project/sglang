import asyncio
import concurrent.futures
import copy
import logging
import threading
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class _WorkerState(threading.local):
    def __init__(self):
        self.processor = None


class MultimodalProcessorExecutor:
    """Run processor calls on isolated, thread-local processor clones."""

    def __init__(self, resolve_processor: Callable[[], Any], max_workers: int):
        # Resolved per clone rather than captured here: a subclass keeps
        # customizing `_processor` after `super().__init__()` has already built
        # this pool, and a clone taken now would miss every one of those edits.
        self._resolve_processor = resolve_processor
        # Probe once, so a processor that cannot be cloned still falls back to
        # synchronous processing at startup instead of failing inside a worker.
        copy.deepcopy(resolve_processor())
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sglang-mm-processor",
        )
        self._worker_state = _WorkerState()
        self._clone_lock = threading.Lock()

    async def run(
        self,
        function: Callable[..., T],
        *args: Any,
        on_cancel: Optional[Callable[[T], None]] = None,
        **kwargs: Any,
    ) -> T:
        future = self._executor.submit(self._run, function, args, kwargs)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # Running threads cannot be cancelled. Keep cleanup attached to the
            # concurrent future, even if the event loop has already shut down.
            future.cancel()
            future.add_done_callback(lambda done: self._discard_result(done, on_cancel))
            raise

    @staticmethod
    def _discard_result(future, on_cancel) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        if on_cancel is not None:
            try:
                on_cancel(future.result())
            except Exception:
                logger.exception("Failed to release abandoned multimodal features")

    def _run(
        self,
        function: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        processor = self._worker_state.processor
        if processor is None:
            # One clone at a time: cloning reads the shared processor, which the
            # worker path also reads for the token-count helpers.
            with self._clone_lock:
                processor = copy.deepcopy(self._resolve_processor())
            self._worker_state.processor = processor
        return function(*args, processor=processor, **kwargs)

    def shutdown(self) -> None:
        self._executor.shutdown()
