from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class HandoffProtocolError(RuntimeError):
    """Raised when a token handoff message violates protocol invariants."""


class HandoffPhase(str, Enum):
    COPYING_PROMPT_KV = "copying_prompt_kv"
    REPLAYING = "replaying"
    DRAINING_FINAL_SUFFIX = "draining_final_suffix"
    READY_TO_COMMIT = "ready_to_commit"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OutputOwner(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class CatchUpEstimate:
    feasible: bool
    initial_backlog_tokens: int
    estimated_bridge_tokens: int
    estimated_catch_up_ms: float
    reason: str | None = None


@dataclass(frozen=True)
class BatchedReplayPlan:
    """Teacher-forced EXTEND inputs plus its sampled boundary token."""

    input_token_ids: list[int]
    expected_next_token_id: int


def build_batched_replay_plan(token_log: list[int]) -> BatchedReplayPlan:
    """Split a sealed bridge log into one EXTEND and one verification token."""

    if len(token_log) < 2:
        raise ValueError("batched replay requires at least two bridge tokens")
    normalized = [int(token_id) for token_id in token_log]
    return BatchedReplayPlan(
        input_token_ids=normalized[:-1],
        expected_next_token_id=normalized[-1],
    )


def estimate_catch_up(
    *,
    remaining_copy_ms: float,
    prefill_decode_tpot_ms: float,
    decode_replay_tokens_per_second: float,
    replay_startup_ms: float = 0.0,
) -> CatchUpEstimate:
    """Estimate whether Decode can catch a live Prefill token producer.

    The model is intentionally conservative and transport-independent. Prefill
    keeps producing during both the remaining KV copy and replay startup.
    Decode then consumes the accumulated token log at its measured extend rate.
    """

    if remaining_copy_ms < 0 or replay_startup_ms < 0:
        raise ValueError("time inputs must be non-negative")
    if prefill_decode_tpot_ms <= 0:
        raise ValueError("prefill_decode_tpot_ms must be positive")
    if decode_replay_tokens_per_second <= 0:
        raise ValueError("decode_replay_tokens_per_second must be positive")

    producer_rate = 1000.0 / prefill_decode_tpot_ms
    backlog_time_ms = remaining_copy_ms + replay_startup_ms
    initial_backlog = math.ceil(producer_rate * backlog_time_ms / 1000.0)

    if decode_replay_tokens_per_second <= producer_rate:
        return CatchUpEstimate(
            feasible=False,
            initial_backlog_tokens=initial_backlog,
            estimated_bridge_tokens=initial_backlog,
            estimated_catch_up_ms=math.inf,
            reason="decode replay rate does not exceed prefill decode rate",
        )

    replay_only_ms = (
        initial_backlog / (decode_replay_tokens_per_second - producer_rate) * 1000.0
    )
    total_bridge_ms = backlog_time_ms + replay_only_ms
    bridge_tokens = math.ceil(producer_rate * total_bridge_ms / 1000.0)
    return CatchUpEstimate(
        feasible=True,
        initial_backlog_tokens=initial_backlog,
        estimated_bridge_tokens=bridge_tokens,
        estimated_catch_up_ms=replay_startup_ms + replay_only_ms,
    )


@dataclass
class TokenHandoffState:
    """Pure protocol model for a single live P-to-D request handoff.

    Token indices are relative to the first generated output token. Prefill is
    the sole owner until an exact-count commit succeeds. The class deliberately
    contains no transport or scheduler calls so races can be tested
    deterministically before integration with a KV backend.
    """

    request_id: str
    epoch: int
    prompt_token_count: int
    phase: HandoffPhase = HandoffPhase.COPYING_PROMPT_KV
    owner: OutputOwner = OutputOwner.PREFILL
    replayed_count: int = 0
    committed_count: int = 0
    sealed_count: int | None = None
    token_log: list[int] = field(default_factory=list)
    failure_reason: str | None = None

    def _check_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            raise HandoffProtocolError(
                f"stale handoff epoch: got {epoch}, expected {self.epoch}"
            )

    def _check_active(self) -> None:
        if self.phase in {
            HandoffPhase.COMMITTED,
            HandoffPhase.CANCELLED,
            HandoffPhase.FAILED,
        }:
            raise HandoffProtocolError(
                f"handoff is terminal in phase {self.phase.value}"
            )

    @property
    def produced_count(self) -> int:
        return len(self.token_log)

    def append_tokens(
        self, *, epoch: int, first_output_index: int, token_ids: list[int]
    ) -> int:
        """Append a contiguous Prefill-produced token-log fragment.

        A fully published fragment is accepted as an idempotent transport
        retry only when its payload exactly matches the existing log. Partial
        overlaps are rejected so a sender must retry the original fragment or
        append a new fragment at ``produced_count``.
        """

        self._check_epoch(epoch)
        self._check_active()
        if self.owner is not OutputOwner.PREFILL:
            raise HandoffProtocolError("prefill cannot append after ownership commit")
        if self.sealed_count is not None:
            raise HandoffProtocolError(
                "prefill cannot append after the token log is sealed"
            )
        token_ids = [int(token_id) for token_id in token_ids]

        fragment_end = first_output_index + len(token_ids)
        if first_output_index < self.produced_count:
            if (
                fragment_end <= self.produced_count
                and self.token_log[first_output_index:fragment_end] == token_ids
            ):
                return self.produced_count
            raise HandoffProtocolError(
                "conflicting or partially overlapping token append"
            )
        if first_output_index != self.produced_count:
            raise HandoffProtocolError(
                "non-contiguous token append: "
                f"got index {first_output_index}, expected {self.produced_count}"
            )
        if not token_ids:
            return self.produced_count

        self.token_log.extend(token_ids)
        if self.phase is HandoffPhase.READY_TO_COMMIT:
            self.phase = HandoffPhase.REPLAYING
        return self.produced_count

    def mark_prompt_kv_ready(self, *, epoch: int) -> None:
        self._check_epoch(epoch)
        self._check_active()
        if self.phase is not HandoffPhase.COPYING_PROMPT_KV:
            raise HandoffProtocolError(
                f"prompt KV readiness is invalid in phase {self.phase.value}"
            )
        self.phase = (
            HandoffPhase.READY_TO_COMMIT
            if self.replayed_count == self.produced_count
            else HandoffPhase.REPLAYING
        )

    def seal_token_log(self, *, epoch: int) -> int:
        """Freeze the final output boundary so Decode can drain a finite suffix.

        A live producer can append another token during every replay launch,
        especially when short extends have a non-trivial fixed startup cost.
        The runtime therefore seals the log only after Prompt KV is ready and
        Decode is sufficiently close. Prefill remains the output owner until
        commit, but it must stop scheduling new Decode steps for this request.
        """

        self._check_epoch(epoch)
        self._check_active()
        if self.phase is HandoffPhase.COPYING_PROMPT_KV:
            raise HandoffProtocolError("cannot seal before prompt KV is ready")
        if self.sealed_count is not None:
            return self.sealed_count

        self.sealed_count = self.produced_count
        self.phase = (
            HandoffPhase.READY_TO_COMMIT
            if self.replayed_count == self.sealed_count
            else HandoffPhase.DRAINING_FINAL_SUFFIX
        )
        return self.sealed_count

    def acknowledge_replay(self, *, epoch: int, replayed_count: int) -> None:
        self._check_epoch(epoch)
        self._check_active()
        if self.phase is HandoffPhase.COPYING_PROMPT_KV:
            raise HandoffProtocolError("cannot acknowledge replay before prompt KV")
        if replayed_count < self.replayed_count:
            raise HandoffProtocolError("replayed_count must be monotonic")
        if replayed_count > self.produced_count:
            raise HandoffProtocolError(
                "decode cannot replay beyond the published token log"
            )

        self.replayed_count = replayed_count
        if replayed_count == self.produced_count:
            self.phase = HandoffPhase.READY_TO_COMMIT
        elif self.sealed_count is not None:
            self.phase = HandoffPhase.DRAINING_FINAL_SUFFIX
        else:
            self.phase = HandoffPhase.REPLAYING

    def commit(self, *, epoch: int, produced_count: int) -> None:
        """Atomically transfer output ownership at an exact token boundary."""

        self._check_epoch(epoch)
        self._check_active()
        if produced_count != self.produced_count:
            raise HandoffProtocolError(
                "decode is behind current producer position: "
                f"requested {produced_count}, current {self.produced_count}"
            )
        if self.replayed_count != produced_count:
            raise HandoffProtocolError(
                "decode has not materialized KV for the full token log"
            )
        if self.sealed_count != produced_count:
            raise HandoffProtocolError(
                "token log must be sealed at the committed output boundary"
            )

        self.committed_count = produced_count
        self.owner = OutputOwner.DECODE
        self.phase = HandoffPhase.COMMITTED

    def cancel(self, *, epoch: int) -> None:
        self._check_epoch(epoch)
        self._check_active()
        self.phase = HandoffPhase.CANCELLED

    def fail(self, *, epoch: int, reason: str) -> None:
        self._check_epoch(epoch)
        self._check_active()
        self.failure_reason = reason
        self.phase = HandoffPhase.FAILED
