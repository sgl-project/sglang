"""Completion contract for ``Scheduler.defer_response(req, completion)``.

One completion holds each response; ``concurrent.futures.Future[None]`` works.
The scheduler polls it while busy, idle or paused, then calls ``result`` before
streaming; errors abort the response. Cancellation retires responses even if
work continues; completions must not cancel work needed by sibling requests.

Completions own payloads after KV release. Held responses do not affect
``Scheduler.is_fully_idle``: only the output rank holds them, so TP ranks must
agree on idle-gated operations. PD disaggregation is unsupported.
"""

from typing import Protocol


class ResponseCompletion(Protocol):
    def done(self) -> bool:
        """Whether ``result`` can be called without blocking."""
        ...

    def result(self) -> None:
        """Finalize metadata on the scheduler thread, or raise on failure."""
        ...

    def cancel(self) -> bool:
        """Attempt to cancel work without blocking; never emit a response."""
        ...
