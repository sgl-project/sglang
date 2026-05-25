"""ReqHandle: lazy-lookup handle for a request submitted via ScriptedRuntime.

Hides the raw ``Req`` object from test scripts so assertions cannot tunnel into
scheduler internals (which would defeat the refactor-safety of the harness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sglang.srt.test.scripted_runtime.runtime import ScriptedRuntime


ReqStatus = Literal["waiting", "running", "finished", "unknown"]


@dataclass(frozen=True, slots=True)
class ReqHandle:
    rid: str
    runtime: "ScriptedRuntime"

    @property
    def status(self) -> ReqStatus:
        return self.runtime._lookup_req_status(self.rid)
