"""Soft session affinity for DP ranks inside a prefill engine."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class _Session:
    rank: int
    peak_input: int
    footprint: int
    slot: int | None
    last_seen: float


class PrefillSessionPlacement:
    def __init__(
        self,
        num_ranks: int,
        *,
        slots_per_rank: int = 3,
        max_sessions: int = 4096,
        idle_seconds: int = 1800,
    ):
        if not 0 <= slots_per_rank <= 32:
            raise ValueError(
                "prefill session planning slots per rank must be between 0 and 32"
            )
        if max_sessions < 1 or idle_seconds < 1:
            raise ValueError(
                "prefill session history needs a positive size and timeout"
            )
        self.sessions: OrderedDict[str, _Session] = OrderedDict()
        self.retained = [0] * num_ranks
        self.occupied = [set() for _ in range(num_ranks)]
        self.slots_per_rank = slots_per_rank
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds

    def _remove(self, session_id: str) -> _Session:
        session = self.sessions.pop(session_id)
        self.retained[session.rank] -= session.footprint
        if session.slot is not None:
            self.occupied[session.rank].remove(session.slot)
        return session

    def drop_ranks(self, ranks: set[int]) -> None:
        for key in [
            key for key, session in self.sessions.items() if session.rank in ranks
        ]:
            self._remove(key)

    def _cold_slot(self, ranks, footprint, live_tokens, capacity):
        # Uniformly spaced small-to-large targets sum to each rank's capacity.
        # Account for the error of earlier placements in the remaining targets.
        count = self.slots_per_rank
        best = None
        for rank in ranks:
            free = [s for s in range(count) if s not in self.occupied[rank]]
            if not free:
                continue
            cap = capacity[rank]
            mean = cap / count
            targets = [
                mean if count == 1 else mean * (2 + 2 * s / (count - 1)) / 3
                for s in free
            ]
            used = max(self.retained[rank], live_tokens[rank])
            deviation = (used + sum(targets) - cap) / len(free)
            for slot, target in zip(free, targets):
                error = abs(footprint - target + deviation) / cap
                score = (error, len(self.occupied[rank]), used / cap, rank, slot)
                if best is None or score < best:
                    best = score
        return (best[-2], best[-1]) if best is not None else (None, None)

    def select(
        self,
        session_id: str | None,
        input_tokens: int,
        output_tokens: int,
        active: list[int],
        live_tokens: list[int],
        live_requests: list[int],
        capacity: list[int],
        forced_rank: int | None = None,
    ) -> int:
        if not active:
            raise RuntimeError("No active DP workers are available for prefill.")
        now = time.monotonic()
        while self.sessions:
            key, oldest = next(iter(self.sessions.items()))
            if now - oldest.last_seen < self.idle_seconds:
                break
            self._remove(key)

        previous = self._remove(session_id) if session_id in self.sessions else None
        peak = max(input_tokens, previous.peak_input) if previous else input_tokens
        growth = peak - previous.peak_input if previous else 0
        # Project one observed turn of growth and this request's output allowance.
        footprint = peak + growth + output_tokens
        if session_id and len(self.sessions) >= self.max_sessions:
            self._remove(next(iter(self.sessions)))

        normalize = all(capacity[r] > 0 for r in active)
        slot = None
        rank = forced_rank
        if rank is None and previous is not None and previous.rank in active:
            rank = previous.rank
        candidates = active
        if rank is None and normalize:
            fitting = [
                r
                for r in active
                if max(self.retained[r], live_tokens[r]) + footprint <= capacity[r]
            ]
            candidates = fitting or active
        if previous is not None and rank == previous.rank:
            slot = previous.slot
        if session_id and normalize and self.slots_per_rank and slot is None:
            slot_rank, slot = self._cold_slot(
                [rank] if rank is not None else candidates,
                footprint,
                live_tokens,
                capacity,
            )
            if rank is None:
                rank = slot_rank
        if rank is None:
            # Use raw token counts until every active rank has reported capacity.
            def score(r):
                used = max(self.retained[r], live_tokens[r])
                projected = used + footprint
                if normalize:
                    projected = (used + min(footprint, capacity[r])) / capacity[r]
                return projected, live_requests[r], r

            rank = min(candidates, key=score)

        if session_id:
            estimate = min(footprint, capacity[rank]) if capacity[rank] else footprint
            self.sessions[session_id] = _Session(rank, peak, estimate, slot, now)
            self.retained[rank] += estimate
            if slot is not None:
                self.occupied[rank].add(slot)
        return rank
