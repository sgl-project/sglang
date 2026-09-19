# Copyright 2026 SGLang Team
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
"""Attempt accounting independent of generation output and client request IDs.

Owned by one tokenizer event loop. A transport close requests cancellation; only
an explicit scheduler acknowledgement makes a dispatched child terminal. Snapshots
retain every child until the entire attempt is terminal, so a consumer can recover
from lost notifications without inspecting generation responses.
"""

import asyncio
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class Child:
    child_id: str
    rid: str
    kind: Literal["sample", "warmup"]
    dp_rank: int | None = None
    dispatched: bool = False
    prefill_complete: bool = False
    terminal: bool = False


@dataclass
class Attempt:
    attempt_id: str
    stage: str
    expires_at: float
    children: dict[str, Child] = field(default_factory=dict)
    version: int = 0
    sealed: bool = False
    cancel_requested: bool = False
    finished_at: float | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)


class RequestLifecycle:
    def __init__(
        self,
        *,
        max_attempts: int = 16384,
        max_children: int = 4096,
        max_total_children: int = 65536,
        max_tombstones: int = 1000000,
        retention_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.incarnation = uuid.uuid4().hex
        self._attempts: dict[str, Attempt] = {}
        self._children: dict[str, Attempt] = {}
        self._max_attempts = max_attempts
        self._max_children = max_children
        self._max_total_children = max_total_children
        self._max_tombstones = max_tombstones
        self._tombstones: set[str] = set()
        self._finished: deque[tuple[float, str]] = deque()
        self._retention_seconds = retention_seconds
        self._clock = clock

    def __contains__(self, attempt_id: str) -> bool:
        return attempt_id in self._attempts

    def is_sealed(self, attempt_id: str) -> bool:
        return self._attempts[attempt_id].sealed

    def is_cancelled(self, attempt_id: str) -> bool:
        attempt = self._attempts.get(attempt_id)
        return attempt is not None and attempt.cancel_requested

    def claim(self, attempt_id: str, stage: str, lease_seconds: float = 30) -> None:
        # The proxy owns this UUID. Never derive it from the native rid, which may
        # be reused, overlap another rid's prefix, or expand during sampling.
        if uuid.UUID(attempt_id).hex != attempt_id:
            raise ValueError("attempt_id must be a canonical UUID hex string")
        self._validate_lease(lease_seconds)
        self.prune()
        if attempt_id in self._attempts or attempt_id in self._tombstones:
            raise ValueError("attempt_id has already been claimed")
        if (
            len(self._attempts) >= self._max_attempts
            or len(self._tombstones) >= self._max_tombstones
        ):
            raise ValueError("request lifecycle capacity exceeded")
        self._attempts[attempt_id] = Attempt(
            attempt_id, stage, self._clock() + lease_seconds
        )

    def add_child(
        self, attempt_id: str, rid: str, kind: Literal["sample", "warmup"] = "sample"
    ) -> str:
        return self.add_children(attempt_id, [rid], kind)[0]

    def add_children(
        self,
        attempt_id: str,
        rids: list[str],
        kind: Literal["sample", "warmup"] = "sample",
    ) -> list[str]:
        attempt = self._attempts[attempt_id]
        self._expire(attempt)
        if attempt.sealed or attempt.cancel_requested:
            raise ValueError("attempt no longer accepts children")
        if (
            len(attempt.children) + len(rids) > self._max_children
            or len(self._children) + len(rids) > self._max_total_children
        ):
            raise ValueError("request lifecycle child capacity exceeded")
        child_ids = []
        for rid in rids:
            child_id = uuid.uuid4().hex
            attempt.children[child_id] = Child(child_id, rid, kind)
            self._children[child_id] = attempt
            child_ids.append(child_id)
        self._changed(attempt)
        return child_ids

    def dispatched(self, child_id: str) -> None:
        attempt = self._children[child_id]
        self._expire(attempt)
        child = attempt.children[child_id]
        if attempt.cancel_requested or child.terminal:
            raise ValueError("attempt was cancelled before dispatch")
        if not child.dispatched:
            child.dispatched = True
            self._changed(attempt)

    def scheduler_event(
        self, child_id: str, dp_rank: int, phase: Literal["prefill", "terminal"]
    ) -> None:
        if phase not in ("prefill", "terminal"):
            raise ValueError("unknown scheduler lifecycle phase")
        attempt = self._children.get(child_id)
        if attempt is None:
            # A delayed duplicate may arrive after the terminal retention window.
            return
        child = attempt.children[child_id]
        if not child.dispatched:
            raise ValueError("scheduler event for an undispatched child")
        if child.dp_rank is not None and child.dp_rank != dp_rank:
            raise ValueError("child moved between scheduler ranks")
        if child.terminal or (phase == "prefill" and child.prefill_complete):
            return
        child.dp_rank = dp_rank
        if phase == "prefill":
            child.prefill_complete = True
        else:
            child.terminal = True
        self._changed(attempt)

    def discard(self, child_id: str) -> None:
        """Local rejection is terminal only if it preceded scheduler dispatch."""
        attempt = self._children[child_id]
        child = attempt.children[child_id]
        if child.dispatched:
            raise ValueError("dispatched children require a scheduler acknowledgement")
        if not child.terminal:
            child.terminal = True
            self._changed(attempt)

    def seal(self, attempt_id: str) -> None:
        """The producer has stopped creating children, including sampling warmups."""
        attempt = self._attempts[attempt_id]
        if not attempt.sealed:
            attempt.sealed = True
            self._changed(attempt)

    def cancel(self, attempt_id: str) -> list[str]:
        attempt = self._attempts[attempt_id]
        if not attempt.cancel_requested and attempt.finished_at is None:
            attempt.cancel_requested = True
            self._changed(attempt)
        return [
            child.child_id
            for child in attempt.children.values()
            if child.dispatched and not child.terminal
        ]

    def renew(self, attempt_id: str, lease_seconds: float = 30) -> None:
        self._validate_lease(lease_seconds)
        attempt = self._attempts[attempt_id]
        self._expire(attempt)
        if attempt.cancel_requested:
            raise ValueError("an expired or cancelled attempt cannot be renewed")
        attempt.expires_at = self._clock() + lease_seconds

    def expired(self) -> list[str]:
        result = []
        for attempt in self._attempts.values():
            self._expire(attempt)
            if attempt.cancel_requested and attempt.finished_at is None:
                result.append(attempt.attempt_id)
        return result

    def snapshot(self, attempt_id: str) -> dict:
        attempt = self._attempts[attempt_id]
        self._expire(attempt)
        return {
            "incarnation": self.incarnation,
            "attempt_id": attempt.attempt_id,
            "stage": attempt.stage,
            "version": attempt.version,
            "sealed": attempt.sealed,
            "cancel_requested": attempt.cancel_requested,
            "terminal": attempt.finished_at is not None,
            "children": [asdict(child) for child in attempt.children.values()],
        }

    async def wait(self, attempt_id: str, after: int, timeout: float = 20) -> dict:
        attempt = self._attempts[attempt_id]
        self._expire(attempt)
        if attempt.version <= after and attempt.finished_at is None:
            # No await between testing the version and capturing the event.
            changed = attempt.changed
            remaining = (
                timeout
                if attempt.cancel_requested
                else max(0, attempt.expires_at - self._clock())
            )
            try:
                await asyncio.wait_for(changed.wait(), min(timeout, remaining))
            except asyncio.TimeoutError:
                pass
        return self.snapshot(attempt_id)

    def prune(self) -> None:
        now = self._clock()
        while self._finished and now - self._finished[0][0] >= self._retention_seconds:
            _, attempt_id = self._finished.popleft()
            self._tombstones.discard(attempt_id)
            self._remove(attempt_id)

    def acknowledge(self, attempt_id: str) -> None:
        """The coordinator persisted terminal state and released its reservations."""
        if attempt_id not in self._tombstones:
            if attempt_id in self._attempts:
                raise ValueError("cannot acknowledge an unfinished attempt")
            raise KeyError(attempt_id)
        self._remove(attempt_id)

    def _remove(self, attempt_id: str) -> None:
        attempt = self._attempts.pop(attempt_id, None)
        if attempt is not None:
            for child_id in attempt.children:
                del self._children[child_id]

    def _expire(self, attempt: Attempt) -> None:
        if (
            attempt.finished_at is None
            and not attempt.cancel_requested
            and self._clock() >= attempt.expires_at
        ):
            self.cancel(attempt.attempt_id)

    def _changed(self, attempt: Attempt) -> None:
        attempt.version += 1
        if (
            attempt.finished_at is None
            and attempt.sealed
            and all(child.terminal for child in attempt.children.values())
        ):
            attempt.finished_at = self._clock()
            self._tombstones.add(attempt.attempt_id)
            self._finished.append((attempt.finished_at, attempt.attempt_id))
        attempt.changed.set()
        attempt.changed = asyncio.Event()

    @staticmethod
    def _validate_lease(seconds: float) -> None:
        if not 0 < seconds <= 60:
            raise ValueError("lifecycle lease must be in (0, 60] seconds")


class SchedulerLifecycle:
    """Observe scheduler-owned state, including deferred KV cleanup.

    Only opted-in requests are retained here. Polling is rate limited and runs at
    the start of the next scheduler iteration: an abort response queued before
    cleanup is not itself a terminal acknowledgement. No generation output is
    consumed, and unary requests report prefill independently of output cadence.
    """

    def __init__(self, emit: Callable, clock=time.monotonic, interval: float = 0.02):
        self._emit = emit
        self._clock = clock
        self._interval = interval
        self._next_poll = 0.0
        self._requests: dict = {}
        self._prefilled: set[str] = set()
        self._retired: set[str] = set()

    def register(self, req) -> None:
        child_id = getattr(req, "lifecycle_id", None)
        if child_id is not None:
            self._requests[child_id] = req

    def retire(self, req) -> None:
        child_id = getattr(req, "lifecycle_id", None)
        if child_id is not None:
            self.register(req)
            self._retired.add(child_id)

    def poll(self) -> None:
        if not self._requests or self._clock() < self._next_poll:
            return
        self._next_poll = self._clock() + self._interval
        for child_id, req in list(self._requests.items()):
            if child_id not in self._prefilled and req.time_stats.prefill_finished_time:
                self._emit(req, "prefill")
                self._prefilled.add(child_id)
            if (
                (child_id in self._retired or req.finished())
                and not req.kv.holds_kv
                and not req.kv.holds_mamba
                and req.inflight_middle_chunks <= 0
                and req.metadata_buffer_index == -1
            ):
                self._emit(req, "terminal")
                del self._requests[child_id]
                self._prefilled.discard(child_id)
                self._retired.discard(child_id)
