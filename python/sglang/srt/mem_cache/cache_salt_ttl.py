# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cache-salt TTL bookkeeping: which salts have expired, and when.

Pure bookkeeping with no engine state, so it is unit testable on CPU. The
reaper lives in the TokenizerManager process, which is the one place with a
clock every TP rank agrees on -- see ``TokenizerManager._cache_salt_ttl_loop``.
"""

from __future__ import annotations

import enum
from typing import Iterable, Optional

import msgspec


class CacheSaltTtlMode(str, enum.Enum):
    """Which event starts the clock for a salt."""

    # Bounded lifetime: the deadline is set once, at the salt's first request,
    # and no later request extends it.
    FIRST_USE = "first_use"
    # Idle timeout: every request carrying the salt pushes the deadline out.
    LAST_USE = "last_use"


class CacheSaltTtlPolicy(msgspec.Struct, frozen=True, kw_only=True):
    default_ttl_s: float
    # Ceiling on a client-supplied per-request TTL; a client may shorten its
    # retention window but never extend it past what the operator configured.
    max_ttl_s: float
    mode: CacheSaltTtlMode
    sweep_interval_s: float
    # A client can mint unbounded distinct salts (nothing upstream caps their
    # cardinality), so the table is bounded and overflow expires the salts
    # closest to their deadline rather than dropping them untracked.
    max_tracked_salts: int

    def resolve_ttl(self, requested_ttl_s: Optional[float]) -> float:
        if requested_ttl_s is None:
            return min(self.default_ttl_s, self.max_ttl_s)
        if requested_ttl_s < 0:
            raise ValueError(
                f"cache_salt_ttl_seconds must be >= 0, got {requested_ttl_s}"
            )
        return min(requested_ttl_s, self.max_ttl_s)


class _SaltState(msgspec.Struct):
    deadline: float
    ttl_s: float


class CacheSaltTtlReaper:
    """Tracks the live salts and reports which ones have outlived their TTL.

    ``now`` is supplied by the caller (a monotonic clock) so the sweep is
    testable without sleeping.
    """

    def __init__(self, policy: CacheSaltTtlPolicy):
        self.policy = policy
        self._salts: dict[str, _SaltState] = {}

    def __len__(self) -> int:
        return len(self._salts)

    def observe(
        self, salt: str, now: float, requested_ttl_s: Optional[float] = None
    ) -> None:
        """Record a request carrying ``salt``, arming or refreshing its TTL."""
        ttl_s = self.policy.resolve_ttl(requested_ttl_s)
        state = self._salts.get(salt)
        if state is None:
            self._salts[salt] = _SaltState(deadline=now + ttl_s, ttl_s=ttl_s)
            return
        # A shorter TTL always wins: a request may tighten its own retention
        # window, and honoring the tightest request is the safe direction.
        if ttl_s < state.ttl_s:
            state.deadline -= state.ttl_s - ttl_s
            state.ttl_s = ttl_s
        if self.policy.mode is CacheSaltTtlMode.LAST_USE:
            state.deadline = max(state.deadline, now + state.ttl_s)

    def forget(self, salts: Iterable[str]) -> None:
        """Drop salts the engine was told to expire; a later request re-arms a
        fresh TTL epoch for the same string."""
        for salt in salts:
            self._salts.pop(salt, None)

    def sweep(self, now: float) -> list[str]:
        """The salts whose TTL has elapsed, removed from the table.

        Sorted, so the command the engine receives is deterministic.
        """
        expired = [salt for salt, s in self._salts.items() if s.deadline <= now]
        expired.extend(self._overflow_salts(exclude=set(expired)))
        self.forget(expired)
        expired.sort()
        return expired

    def _overflow_salts(self, exclude: set[str]) -> list[str]:
        overflow = len(self._salts) - len(exclude) - self.policy.max_tracked_salts
        if overflow <= 0:
            return []
        candidates = sorted(
            (s.deadline, salt) for salt, s in self._salts.items() if salt not in exclude
        )
        return [salt for _, salt in candidates[:overflow]]


def build_cache_salt_ttl_reaper(server_args) -> Optional[CacheSaltTtlReaper]:
    """None when --cache-salt-ttl-seconds is unset."""
    if server_args.cache_salt_ttl_seconds is None:
        return None
    return CacheSaltTtlReaper(
        CacheSaltTtlPolicy(
            default_ttl_s=server_args.cache_salt_ttl_seconds,
            max_ttl_s=server_args.cache_salt_ttl_max_seconds,
            mode=CacheSaltTtlMode(server_args.cache_salt_ttl_mode),
            sweep_interval_s=server_args.cache_salt_ttl_sweep_interval_seconds,
            max_tracked_salts=server_args.cache_salt_ttl_max_tracked_salts,
        )
    )
