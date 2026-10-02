import asyncio
import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# (shared tokenizer, the worker's clone of it), bound for the duration of a job.
_job_tokenizer: ContextVar[Optional[tuple[Any, Any]]] = ContextVar(
    "sglang_preprocess_job_tokenizer", default=None
)


def resolve_tokenizer(shared: Any) -> Any:
    """Return the running job's clone of ``shared``, or ``shared`` outside a job."""
    binding = _job_tokenizer.get()
    if binding is not None and binding[0] is shared:
        return binding[1]
    return shared


class PreprocessExecutor:
    """Runs blocking request preprocessing (chat templates, tokenization) off the
    event loop.

    Jobs use a clone of the tokenizer: the event loop keeps using the shared one
    (multimodal processor calls, response parsing), and a fast tokenizer used
    from two threads at once raises ``Already borrowed``.

    Requests leave preprocessing in the order they entered it: one FIFO worker
    runs every offloaded job, and a job may run inline only while nothing is
    outstanding, so it can never overtake an earlier request.
    """

    def __init__(self, get_shared_tokenizer: Callable[[], Any] = lambda: None):
        self._get_shared_tokenizer = get_shared_tokenizer
        self._tokenizer_binding: Optional[tuple[Any, Any]] = None
        # Set if the tokenizer cannot be cloned: every job then runs inline, as
        # it did before preprocessing was offloaded.
        self._run_inline = False
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
        if (inline_if_idle and self._num_outstanding == 0) or self._run_inline:
            return func(*args)
        binding = self._bind_tokenizer()
        if self._run_inline:
            return func(*args)

        loop = asyncio.get_running_loop()
        context = copy_context()
        context.run(_job_tokenizer.set, binding)
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

    def _bind_tokenizer(self) -> Optional[tuple[Any, Any]]:
        # Cloned on first use, after startup has finished editing the tokenizer
        # (e.g. setting its chat template).
        shared = self._get_shared_tokenizer()
        if shared is None:
            return None
        binding = self._tokenizer_binding
        if binding is None or binding[0] is not shared:
            try:
                binding = (shared, copy.deepcopy(shared))
            except Exception:
                logger.warning(
                    "Unable to clone the tokenizer for the preprocessing worker; "
                    "preprocessing requests on the event loop.",
                    exc_info=True,
                )
                self._run_inline = True
                return None
            self._tokenizer_binding = binding
        return binding

    def _release(self) -> None:
        self._num_outstanding -= 1
