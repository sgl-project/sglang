"""ScriptedReqHandle: handle for a request submitted via ScriptedContext.

Convenience handle exposing a stable, named subset of req state for
the common test patterns. Test scripts are also explicitly allowed to
reach into ``t._scheduler`` (the raw ``Scheduler`` object) to assert
invariants on scheduler internals — see the project plan
``2026-05-26-direct-internals-access-plan.md`` for the rationale.
Coupling tests to internals is accepted as the price of catching
regressions that aren't otherwise observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook


@dataclass(frozen=True, slots=True)
class ScriptedReqHandle:
    rid: str
    scheduler_hook: "ScriptedSchedulerHook"

    @property
    def req(self) -> Optional["Req"]:
        """The raw engine ``Req`` this handle tracks, or ``None`` if the
        scheduler no longer holds it (finished / not yet admitted).

        Escape hatch for assertions on engine-internal fields not surfaced as
        named handle properties — see the module docstring on direct internals
        access. Re-fetched on every access since the req moves between
        scheduler structures.
        """
        return self.scheduler_hook._find_req_by_rid(self.rid)

    @property
    def finished(self) -> bool:
        """True iff the request has completed (stop / length / abort)."""
        return self.scheduler_hook._lookup_finished(self.rid)

    @property
    def is_chunking(self) -> bool:
        """True iff the req is currently in the middle of chunked prefill.

        Reflects the scheduler's singular ``chunked_req`` slot — True iff
        this rid is the current chunked_req.
        """
        return self.scheduler_hook._lookup_is_chunking(self.rid)
