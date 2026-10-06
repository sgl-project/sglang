"""Extension contract for holding finished requests' final responses.

An out-of-tree consumer that finishes per-request work asynchronously (for
example, a ``HostAuxiliaryOutput`` that publishes data to external storage)
can keep a finished request from streaming until that work completes:

1. register a ``DeferredOutputSource`` with
   ``Scheduler.register_deferred_output_source``;
2. set ``Req.defer_output`` on a finished request, e.g. from
   ``HostAuxiliaryOutput.consume``, which runs before the batch is streamed;
3. return the request from ``poll`` once its work is done.

The scheduler skips deferred requests when streaming, polls every source on
each iteration and while idle, streams the requests a source releases, and
does not report itself idle while a source has pending requests. All calls
run on the scheduler thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DeferredOutputSource(Protocol):
    def poll(self) -> List[Req]:
        """Return held requests that may stream now, without blocking."""
        ...

    def has_pending(self) -> bool:
        """Whether any request is still held by this source."""
        ...
