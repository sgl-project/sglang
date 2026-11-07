"""Bounded decode session affinity with capacity-aware spill."""

import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class _Session:
    rank: int
    peak_input: int
    footprint: int
    last_seen: float


class DecodeSessionPlacement:
    def __init__(
        self,
        num_ranks: int,
        max_sessions: int = 4096,
        idle_seconds: int = 1800,
        reserved_decode_tokens: int = 0,
    ):
        if min(num_ranks, max_sessions, idle_seconds) <= 0:
            raise ValueError("Session placement limits must be positive")
        if reserved_decode_tokens < 0:
            raise ValueError("Reserved decode tokens must be nonnegative")
        self.sessions: OrderedDict[tuple[str, str | None], _Session] = OrderedDict()
        self.retained = [0] * num_ranks
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        self.reserved_decode_tokens = reserved_decode_tokens

    def _remove(self, key):
        previous = self.sessions.pop(key)
        self.retained[previous.rank] -= previous.footprint
        return previous

    def select(
        self,
        key,
        input_tokens,
        output_tokens,
        active,
        live_tokens,
        live_requests,
        capacity,
        forced_rank=None,
    ):
        if not active:
            raise RuntimeError("No active DP workers are available for decode")
        if forced_rank is not None and forced_rank not in active:
            raise ValueError(f"DP rank {forced_rank} is not active")
        now = time.monotonic()
        while self.sessions:
            oldest_key, oldest = next(iter(self.sessions.items()))
            if now - oldest.last_seen < self.idle_seconds:
                break
            self._remove(oldest_key)
        previous = self._remove(key) if key in self.sessions else None
        peak = max(input_tokens, previous.peak_input) if previous else input_tokens
        growth = peak - previous.peak_input if previous else 0
        footprint = peak + growth + output_tokens
        if key and len(self.sessions) >= self.max_sessions:
            self._remove(next(iter(self.sessions)))
        rank = forced_rank
        if rank is None and previous and previous.rank in active:
            rank = previous.rank
            # Price the full incoming prompt without assuming prefix reuse.
            # Snapshots guide routing; native KV admission remains authoritative.
            if all(capacity[r] > 0 for r in active):

                def projected(worker):
                    # Include queued prompts and per-request decode reserves.
                    # Retained history is a cache preference, not live KV use.
                    return (
                        live_tokens[worker]
                        + input_tokens
                        + self.reserved_decode_tokens * (live_requests[worker] + 1)
                    )

                if projected(rank) > capacity[rank]:
                    fitting = [
                        r for r in active if r != rank and projected(r) <= capacity[r]
                    ]
                    if fitting:
                        rank = min(
                            fitting,
                            key=lambda r: (
                                projected(r) / capacity[r],
                                live_requests[r],
                                r,
                            ),
                        )
        if rank is None:
            normalize = all(capacity[r] > 0 for r in active)

            def score(worker):
                projected = max(self.retained[worker], live_tokens[worker]) + footprint
                usage = projected / capacity[worker] if normalize else projected
                return usage, live_requests[worker], worker

            rank = min(active, key=score)
        if key:
            self.sessions[key] = _Session(rank, peak, footprint, now)
            self.retained[rank] += footprint
        return rank
