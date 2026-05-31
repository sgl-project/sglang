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
        return self.scheduler_hook._find_req_by_rid(self.rid)

    @property
    def finished(self) -> bool:
        return self.scheduler_hook._lookup_finished(self.rid)

    @property
    def is_chunking(self) -> bool:
        return self.scheduler_hook._lookup_is_chunking(self.rid)
