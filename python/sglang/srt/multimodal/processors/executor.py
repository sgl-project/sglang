import asyncio
import concurrent.futures
import copy
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


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

    async def run(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._run, function, args, kwargs
        )

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
