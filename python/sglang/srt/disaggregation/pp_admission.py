from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Sequence, TypeVar


class DeferredBootstrapRequest(Protocol):
    rid: str
    pp_defer_body: Optional[int]


PollT = TypeVar("PollT")


@dataclass(frozen=True)
class PPAdmissionVerdict:
    """An admission decision produced by the first pipeline stage."""

    admitted: tuple[str, ...]
    failed: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> Optional[PPAdmissionVerdict]:
        if payload is None:
            return None
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise TypeError("PP admission payload must be a two-item sequence")
        if len(payload) != 2:
            raise ValueError(
                "PP admission payload must contain admitted and failed rids"
            )

        admitted = cls._parse_rids(payload[0], "admitted")
        failed = cls._parse_rids(payload[1], "failed")
        overlap = set(admitted) & set(failed)
        if overlap:
            raise ValueError(f"PP admission payload has conflicting rids: {overlap}")
        return cls(admitted=admitted, failed=failed)

    @staticmethod
    def _parse_rids(value: object, name: str) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"PP admission {name} rids must be a sequence")
        if not all(isinstance(rid, str) for rid in value):
            raise TypeError(f"PP admission {name} rids must be strings")
        return tuple(dict.fromkeys(value))

    def to_payload(self) -> list[list[str]]:
        return [list(self.admitted), list(self.failed)]

    def with_deferred(self, deferred_rids: Iterable[str]) -> PPAdmissionVerdict:
        failed = set(self.failed)
        admitted = tuple(
            rid
            for rid in dict.fromkeys((*self.admitted, *deferred_rids))
            if rid not in failed
        )
        return PPAdmissionVerdict(admitted=admitted, failed=self.failed)

    @property
    def all_rids(self) -> set[str]:
        return set(self.admitted) | set(self.failed)


@dataclass(frozen=True)
class PPAdmissionMessage:
    """Admission verdict plus failures reported by later pipeline stages."""

    verdict: PPAdmissionVerdict
    local_failures: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: object) -> Optional[PPAdmissionMessage]:
        if payload is None:
            return None
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise TypeError("PP admission message must be a sequence")
        if len(payload) not in (2, 3):
            raise ValueError("PP admission message must contain two or three items")

        verdict = PPAdmissionVerdict.from_payload(payload[:2])
        assert verdict is not None
        local_failures = (
            PPAdmissionVerdict._parse_rids(payload[2], "local failure")
            if len(payload) == 3
            else ()
        )
        return cls(verdict=verdict, local_failures=local_failures)

    def to_payload(self) -> list[list[str]]:
        return [*self.verdict.to_payload(), list(self.local_failures)]

    def with_verdict(self, verdict: PPAdmissionVerdict) -> PPAdmissionMessage:
        return PPAdmissionMessage(verdict, self.local_failures)

    def with_local_failures(self, local_failures: Iterable[str]) -> PPAdmissionMessage:
        failures = tuple(dict.fromkeys((*self.local_failures, *local_failures)))
        return PPAdmissionMessage(self.verdict, failures)


@dataclass
class PPAdmissionState:
    """Scheduler-local state for the optional PP admission fast path."""

    step: int = 0
    deferred_rids: dict[str, None] = field(default_factory=dict)
    deferred_bootstrap: list[DeferredBootstrapRequest] = field(default_factory=list)
    local_failures: dict[str, None] = field(default_factory=dict)
    uniform_failures_applied: dict[str, None] = field(default_factory=dict)

    def defer_verdict(self, rid: str) -> None:
        self.deferred_rids[rid] = None

    def defer_bootstrap(self, req: DeferredBootstrapRequest) -> None:
        req.pp_defer_body = self.step
        if req not in self.deferred_bootstrap:
            self.deferred_bootstrap.append(req)

    def clear_applied(self, verdict: PPAdmissionVerdict) -> None:
        for rid in verdict.all_rids:
            self.deferred_rids.pop(rid, None)

    def record_local_failure(self, rid: str) -> bool:
        is_new = rid not in self.local_failures
        self.local_failures[rid] = None
        return is_new

    def has_local_failure(self, rid: str) -> bool:
        return rid in self.local_failures

    def has_uniform_failure(self, rid: str) -> bool:
        return rid in self.uniform_failures_applied

    def consume_uniform_failures(self, rids: Iterable[str]) -> list[str]:
        reported = tuple(dict.fromkeys(rids))
        reported_set = set(reported)
        new_failures = [
            rid for rid in reported if rid not in self.uniform_failures_applied
        ]
        for rid in reported:
            self.local_failures.pop(rid, None)
            self.deferred_rids.pop(rid, None)
            self.uniform_failures_applied[rid] = None
        self.deferred_bootstrap = [
            req for req in self.deferred_bootstrap if req.rid not in reported_set
        ]
        return new_failures


def map_authoritative_polls(
    rids: Iterable[str],
    verdict: PPAdmissionVerdict,
    admitted_poll: PollT,
    failed_poll: PollT,
) -> list[Optional[PollT]]:
    """Map the first stage's verdict without polling local senders again."""
    admitted = set(verdict.admitted)
    failed = set(verdict.failed)
    return [
        failed_poll if rid in failed else admitted_poll if rid in admitted else None
        for rid in rids
    ]


def merge_deferred_send(
    previous: Optional[tuple[bool, Optional[int]]],
    last_chunk: bool,
    end_idx: Optional[int],
) -> tuple[bool, Optional[int]]:
    """Coalesce deferred sends without truncating a terminal transfer."""
    owed_last = last_chunk or bool(previous and previous[0])
    return owed_last, None if owed_last else end_idx


def publication_for_stage(
    is_first_rank: bool,
    intended: PPAdmissionVerdict,
    applied: Optional[PPAdmissionVerdict],
) -> PPAdmissionVerdict:
    """Publish the authoritative stage's applied verdict to later stages."""
    if is_first_rank and applied is not None:
        return applied
    return intended


def prepare_forward_message(
    message: PPAdmissionMessage,
    published: PPAdmissionVerdict,
    local_failures: Iterable[str],
) -> PPAdmissionMessage:
    """Update one payload without adding or reordering P2P operations."""
    return message.with_verdict(published).with_local_failures(local_failures)


def route_aborts_to_failed(
    good_rids: Iterable[str],
    bad_rids: Iterable[str],
    aborted_rids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Make aborts override admission while preserving input order."""
    aborted_order = tuple(dict.fromkeys(aborted_rids))
    if not aborted_order:
        return list(good_rids), list(bad_rids)
    aborted = set(aborted_order)
    good = [rid for rid in good_rids if rid not in aborted]
    bad = list(dict.fromkeys((*bad_rids, *aborted_order)))
    return good, bad
