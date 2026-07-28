from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

PostResponseTask = tuple[str, Callable[[], None]]


class PostResponseTaskRunner:
    """Runs response-tail maintenance before the next forward dispatch."""

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
