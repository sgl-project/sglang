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
keeps polling instead of sleeping while a source has pending requests. All
calls run on the scheduler thread.

Held requests do not count against ``Scheduler.is_fully_idle``. Under tensor
parallelism only the rank that streams outputs holds them, and idle-gated
operations (cache flushes, memory release, storage attach) must see the same
state on every rank. A held request has already released its KV cache, so a
source must own the data it publishes. Sources are not supported with PD
disaggregation.
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
