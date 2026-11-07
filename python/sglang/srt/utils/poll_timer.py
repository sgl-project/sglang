import time
from typing import Optional

import msgspec


class PollTimer(msgspec.Struct):
    """Time an interval from its first blocked poll to its first ready poll."""

    started_at: Optional[float] = None

    @property
    def running(self) -> bool:
        return self.started_at is not None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at if self.running else 0.0

    def observe(self, ready: bool) -> float:
        if ready:
            elapsed = self.elapsed()
            self.started_at = None
            return elapsed
        if not self.running:
            self.started_at = time.perf_counter()
        return 0.0
