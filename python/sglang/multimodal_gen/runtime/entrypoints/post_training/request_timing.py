"""Wall-clock marks for the rollout HTTP path.

The client reconstructs one waterfall per request across three processes, so it
needs absolute marks, not durations. They ride a response header: every mark is
taken before the response headers are sent, and a header leaves the msgpack
body contract untouched for clients that ignore it.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager

# Compact JSON rides under the header, e.g.
#   x-sgld-timing: {"srv_recv":1787472609.104512,...,"msgpack_end":1787472691.63172,"request_id":"abc123"}
#     mark name -> epoch seconds (6 dp), plus this request's id
TIMING_HEADER = "x-sgld-timing"

MARKS = (
    "srv_recv",
    "forward_start",
    "forward_end",
    "build_start",
    "build_end",
    "dump_end",
    "msgpack_end",
)


class RequestStamps:
    """Absolute wall-clock marks for one rollout request."""

    __slots__ = ("request_id", "_marks")

    def __init__(self, request_id: str = "") -> None:
        self.request_id = request_id
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        assert name in MARKS, f"unknown timing mark {name!r}"
        self._marks[name] = time.time()

    @contextmanager
    def span(self, name: str):
        """Mark ``{name}_start`` on entry and ``{name}_end`` on exit.

        dump/msgpack have no start mark: each begins where the previous span
        ended, so only the end boundary is recorded.
        """
        start = f"{name}_start"
        if start in MARKS:
            self.mark(start)
        try:
            yield
        finally:
            self.mark(f"{name}_end")

    def to_header(self) -> str:
        payload: dict[str, object] = {
            name: round(t, 6) for name, t in self._marks.items()
        }
        if self.request_id:
            payload["request_id"] = self.request_id
        return json.dumps(payload, separators=(",", ":"))

