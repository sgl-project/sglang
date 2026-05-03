"""Post-response task utilities.

Unlike post-process (for example, output tensor/image/video formatting, file
saving, and response payload construction), post-response is runtime
maintenance after the result is returned (for example, component cleanup, CPU
offload, next-request prefetch, and cache cleanup).

Performance concern:
1. When request frequency is high, the next request may wait for unfinished
   post-response work, so the maintenance cost moves to the next request.
2. When request frequency is low, post-response work usually finishes in the
   idle gap and has little user-visible impact.
"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class PostResponseTaskRunner:
    """Runs maintenance tasks after a response has been sent.

    The scheduler waits for pending tasks before executing the next request, so
    tasks can use model modules without racing the next forward pass.
    """

    def __init__(self, name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._lock = Lock()
        self._pending: list[Future[None]] = []

    def submit(self, name: str, fn: Callable[[], None]) -> None:
        with self._lock:
            self._pending.append(self._executor.submit(self._run, name, fn))

    def wait_pending(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = []

        for future in pending:
            future.result()

    def shutdown(self) -> None:
        self.wait_pending()
        self._executor.shutdown(wait=True)

    @staticmethod
    def _run(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("Post-response task failed: %s", name)
            raise
