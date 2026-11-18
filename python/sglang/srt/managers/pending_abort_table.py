"""Remember rid-addressed aborts that arrive before their request.

`/abort_request` is a point-in-time message: the process it lands on aborts what
it holds right now. The generate and the abort are independent HTTP requests, so
the abort can arrive before the request it names reaches the process that every
request passes on its way to the scheduler: the multi-tokenizer router, or the
tokenizer manager itself with a single tokenizer worker. That process records
the abort here and applies it when the request passes.
"""

import time
from typing import Dict


class PendingAbortTable:
    """Rids whose abort arrived before the request; each entry is one-shot."""

    # Must outlive the gap between the abort and its request passing, which we
    # measured at up to 26 s under an abort storm with congested tokenizer
    # workers. An expired entry falls back to the old behavior: the abort is lost.
    TTL = 30.0

    def __init__(self, ttl: float = TTL) -> None:
        self.ttl = ttl
        # Insertion order is deadline order: the TTL is fixed and record()
        # re-inserts a refreshed rid at the end.
        self._deadlines: Dict[str, float] = {}

    def __len__(self) -> int:
        self._prune()
        return len(self._deadlines)

    def __contains__(self, rid: str) -> bool:
        self._prune()
        return rid in self._deadlines

    def record(self, rid: str) -> None:
        self._prune()
        self._deadlines.pop(rid, None)
        self._deadlines[rid] = time.monotonic() + self.ttl

    def consume(self, rid: str) -> bool:
        """Take the entry as the rid's request passes; True if one was pending."""
        self._prune()
        return self._deadlines.pop(rid, None) is not None

    def discard(self, rid: str) -> None:
        self._deadlines.pop(rid, None)

    def _prune(self) -> None:
        now = time.monotonic()
        while self._deadlines:
            rid, deadline = next(iter(self._deadlines.items()))
            if deadline > now:
                break
            del self._deadlines[rid]
