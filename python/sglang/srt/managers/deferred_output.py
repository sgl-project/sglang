"""Completion contract for ``Scheduler.defer_response(req, completion)``.

An auxiliary-output consumer can defer a finished request's response until
its asynchronous work completes. The scheduler owns the request and polls the
completion while busy, idle or paused. It calls ``result`` before streaming,
on the scheduler thread, so a plugin can finalize response metadata there.
An unhandled exception aborts the response instead of terminating the server.

Only one completion may hold a request; consumers must combine their work
before deferring. A ``concurrent.futures.Future[None]`` satisfies this contract.
Cancellation retires the response immediately even if the work is already
running. A completion sharing a batch's work must not cancel its siblings.

Held requests have released their KV cache, so the completion must own its
payload. They do not affect ``Scheduler.is_fully_idle``: only the output rank
holds responses under TP, and idle-gated collective operations must agree on
all ranks. PD disaggregation is unsupported.
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
