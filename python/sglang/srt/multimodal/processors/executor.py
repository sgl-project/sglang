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

    def __init__(self, processor: Any, max_workers: int):
        self._processor = processor
        self._processor_clones = [copy.deepcopy(processor) for _ in range(max_workers)]
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
            with self._clone_lock:
                processor = (
                    self._processor_clones.pop()
                    if self._processor_clones
                    else copy.deepcopy(self._processor)
                )
            self._worker_state.processor = processor
        return function(*args, processor=processor, **kwargs)

    def shutdown(self) -> None:
        self._executor.shutdown()
