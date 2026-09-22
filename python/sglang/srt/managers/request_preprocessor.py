import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RequestPreprocessor:
    """Runs blocking request preprocessing (chat templates, tokenization) off the
    event loop.

    Requests leave preprocessing in the order they entered it: one FIFO worker
    runs every offloaded job, and a job may run inline only while nothing is
    outstanding, so it can never overtake an earlier request.
    """

    def __init__(self):
        # One worker keeps tokenizer and template use serialized, as it was when
        # everything ran on the event loop thread.
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
        self._num_outstanding += 1
        job = self._executor.submit(copy_context().run, func, *args)
        try:
            result = await asyncio.wrap_future(job)
        except BaseException:
            if job.done():
                self._release()
            else:
                # Cancelled while running: stay outstanding until the worker
                # finishes, so no inline job overlaps it.
                job.add_done_callback(
                    lambda _: loop.call_soon_threadsafe(self._release)
                )
            raise
        self._release()
        return result

    def _release(self) -> None:
        self._num_outstanding -= 1
