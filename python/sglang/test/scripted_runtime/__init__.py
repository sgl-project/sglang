"""ScriptedRuntime: deterministic generator-driven scheduler harness for fine-grained tests.

See ``docs`` and the project notes under
``agent-context/projects/sglang/2026-05-25-chunked-prefill-rewrite/``
for the design rationale.
"""

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.req_handle import ReqHandle, ReqStatus
from sglang.test.scripted_runtime.runtime import (
    ScriptedRuntime,
    ScriptedRuntimeFinished,
)

__all__ = [
    "ScriptedRuntime",
    "ScriptedRuntimeFinished",
    "ReqHandle",
    "ReqStatus",
    "execute_scripted_runtime",
]
