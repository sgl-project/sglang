import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from typing import Any, Callable, Optional, TypeVar

from transformers import PreTrainedTokenizerFast

T = TypeVar("T")

# Arbitrary cutoff: inputs this short hold the event loop for well under a
# millisecond, so preprocessing them inline when idle skips the thread hop.
INLINE_PREPROCESS_MAX_CHARS = 2048

# (shared tokenizer, the job's view of it), bound for the duration of a job.
_job_tokenizer: ContextVar[Optional[tuple[Any, Any]]] = ContextVar(
    "sglang_preprocess_job_tokenizer", default=None
)


def resolve_tokenizer(shared: Any) -> Any:
    """Return the running job's view of ``shared``, or ``shared`` outside a job."""
    binding = _job_tokenizer.get()
    if binding is not None and binding[0] is shared:
        return binding[1]
    return shared


def with_own_backend(tokenizer: Any) -> Any:
    """``tokenizer`` with a private copy of its Rust backend, for use on another
    thread. A fast tokenizer's backend raises ``Already borrowed`` (or, on newer
    ``tokenizers``, blocks) when two threads use it at once."""
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return tokenizer
    clone = copy.copy(tokenizer)
    clone._tokenizer = copy.deepcopy(tokenizer.backend_tokenizer)
    return clone


class PreprocessExecutor:
    """Runs blocking request preprocessing (chat templates, tokenization) off the
    event loop.

    Jobs never touch the Rust backend the event loop uses (multimodal processor
    calls, response parsing). Copying a backend takes hundreds of ms and holds the
    GIL, so it is copied once at construction; each job gets a shallow copy of
    the shared tokenizer on that backend, so attributes set after startup (e.g.
    the chat template) stay current.

    Requests leave preprocessing in the order they entered it: one FIFO worker
    runs every offloaded job, and a job may run inline only while nothing is
    outstanding, so it can never overtake an earlier request.
    """

    def __init__(self, get_shared_tokenizer: Callable[[], Any] = lambda: None):
        self._get_shared_tokenizer = get_shared_tokenizer
        self._worker_backend_source = None
        self._worker_backend = None
        self._copy_backend(get_shared_tokenizer())
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sglang-preprocess"
        )
        # Offloaded jobs whose caller has not resumed yet, including jobs still
        # running after their caller was cancelled.
        self._num_outstanding = 0

    async def run(
        self, func: Callable[..., T], *args: Any, inline_if_idle: bool = False
    ) -> T:
        """Run ``func(*args)`` on the worker thread and await its result.

        ``inline_if_idle`` is for work too cheap to be worth the thread hop: it
        runs on the event loop when no job is outstanding, and queues behind
        the outstanding jobs otherwise.
        """
        if inline_if_idle and self._num_outstanding == 0:
            return func(*args)

        loop = asyncio.get_running_loop()
        context = copy_context()
        context.run(_job_tokenizer.set, self._bind_tokenizer())
        self._num_outstanding += 1
        job = self._executor.submit(context.run, func, *args)
        try:
            result = await asyncio.wrap_future(job)
        except BaseException:
            if job.done():
                self._release()
            else:
                # Cancelled while running: stay outstanding until the worker
                # finishes, so no inline job overtakes it.
                job.add_done_callback(
                    lambda _: loop.call_soon_threadsafe(self._release)
                )
            raise
        self._release()
        return result

    def _copy_backend(self, shared: Any) -> None:
        if isinstance(shared, PreTrainedTokenizerFast):
            self._worker_backend_source = shared.backend_tokenizer
            self._worker_backend = copy.deepcopy(shared.backend_tokenizer)

    def _bind_tokenizer(self) -> Optional[tuple[Any, Any]]:
        shared = self._get_shared_tokenizer()
        if not isinstance(shared, PreTrainedTokenizerFast):
            # No Rust backend to borrow concurrently.
            return None
        if shared.backend_tokenizer is not self._worker_backend_source:
            # The tokenizer was replaced after startup; pay the copy once more.
            self._copy_backend(shared)
        view = copy.copy(shared)
        view._tokenizer = self._worker_backend
        return shared, view

    def _release(self) -> None:
        self._num_outstanding -= 1
