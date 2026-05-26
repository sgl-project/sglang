"""ReqHandle: handle for a request submitted via ScriptedRuntime.

Hides the raw ``Req`` so test scripts cannot assert on scheduler
internals — preserves the harness's refactor-safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.runtime import ScriptedRuntime


ReqStatus = Literal["waiting", "running", "finished", "unknown"]


@dataclass(frozen=True, slots=True)
class ReqHandle:
    rid: str
    runtime: "ScriptedRuntime"

    @property
    def status(self) -> ReqStatus:
        return self.runtime._lookup_req_status(self.rid)
