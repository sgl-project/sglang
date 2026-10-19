"""Transport-independent state for an OFFLINE simulator's external driver.

The scheduler owns payloads and batch execution. This module only records the
driver's causal promise, checks the input barrier, and commits virtual events.
"""

from collections import deque
from typing import Literal

import msgspec

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """The run cannot continue without violating the coordination contract."""


class Submit(msgspec.Struct, tag="submit", forbid_unknown_fields=True, frozen=True):
    rid: str


class Cancel(msgspec.Struct, tag="cancel", forbid_unknown_fields=True, frozen=True):
    rid: str
    reason: str = "cancelled by driver"


Operation = Submit | Cancel
TerminalKind = Literal["COMPLETED", "CANCELLED", "ERROR"]


class Ready(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    run_id: str
    round: int
    operations: list[Operation]
    next_wakeup_ns: int | None = None
    finished: bool = False
    protocol_version: int = PROTOCOL_VERSION


class Terminal(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    rid: str
    kind: TerminalKind
    at_ns: int
    reason: str | None = None


class Grant(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    run_id: str
    round: int
    now_ns: int
    terminals: list[Terminal]
    status: Literal["CONTINUE", "FINISHED", "ERROR"] = "CONTINUE"
    error: str | None = None
    protocol_version: int = PROTOCOL_VERSION


_ENCODER = msgspec.json.Encoder()
_READY_DECODER = msgspec.json.Decoder(Ready)
_GRANT_DECODER = msgspec.json.Decoder(Grant)


def encode_message(message: Ready | Grant) -> bytes:
    return _ENCODER.encode(message)


def decode_ready(data: bytes) -> Ready:
    try:
        return _READY_DECODER.decode(data)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"Invalid Ready: {exc}") from exc


def decode_grant(data: bytes) -> Grant:
    try:
        return _GRANT_DECODER.decode(data)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"Invalid Grant: {exc}") from exc


class _Request(msgspec.Struct):
    sequence: int
    arrival_ns: int
    admitted: bool = False
    terminal: Terminal | None = None


class Coordinator:
    """One driver, one outstanding Ready, and one authoritative virtual clock."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id
        self.now_ns = 0
        self.next_round = 0
        self._ready: Ready | None = None
        self._requests: dict[str, _Request] = {}
        self._pending: deque[str] = deque()
        self._terminals: list[Terminal] = []
        self._driver_finished = False
        self._closed = False

    @property
    def awaiting_ready(self) -> bool:
        return self._ready is None and not self._closed

    @property
    def next_wakeup_ns(self) -> int | None:
        return self._ready.next_wakeup_ns if self._ready is not None else None

    @property
    def driver_finished(self) -> bool:
        return self._driver_finished

    @property
    def has_terminals(self) -> bool:
        return bool(self._terminals)

    @property
    def pending_rids(self) -> tuple[str, ...]:
        """Payloads required by this Ready, in manifest order."""
        self._skip_terminal_payloads()
        return tuple(rid for rid in self._pending if self.is_open(rid))

    def _skip_terminal_payloads(self) -> None:
        # Cancelled or rejected payloads can be skipped without an HTTP arrival.
        # Each entry leaves this queue once, including cancellations out of order.
        while self._pending and not self.is_open(self._pending[0]):
            self._pending.popleft()

    @property
    def open_rids(self) -> tuple[str, ...]:
        return tuple(rid for rid in self._requests if self.is_open(rid))

    def is_open(self, rid: str) -> bool:
        request = self._requests.get(rid)
        return request is not None and request.terminal is None

    def request_arrival_ns(self, rid: str) -> int:
        """Return the manifest arrival without a float clock conversion."""
        try:
            return self._requests[rid].arrival_ns
        except KeyError as exc:
            raise self._error(f"unknown request rid={rid!r}") from exc

    def is_terminal(self, rid: str) -> bool:
        request = self._requests.get(rid)
        return request is not None and request.terminal is not None

    def is_cancelled(self, rid: str) -> bool:
        request = self._requests.get(rid)
        return (
            request is not None
            and request.terminal is not None
            and request.terminal.kind == "CANCELLED"
        )

    def _error(self, message: str) -> ProtocolError:
        return ProtocolError(
            f"run={self.run_id!r} round={self.next_round} now_ns={self.now_ns}: "
            f"{message}"
        )

    def accept_ready(self, ready: Ready) -> None:
        """Validate the whole manifest before modifying any run state."""
        if self._closed:
            raise self._error("run already finished")
        if self._ready is not None:
            raise self._error("previous Ready has not received its Grant")
        if ready.protocol_version != PROTOCOL_VERSION:
            raise self._error(f"unsupported protocol version {ready.protocol_version}")
        if not ready.run_id or len(ready.run_id) > 128:
            raise self._error("run_id must contain 1 to 128 characters")
        if self.run_id is not None and ready.run_id != self.run_id:
            raise self._error(f"unexpected run_id {ready.run_id!r}")
        if type(ready.round) is not int or ready.round != self.next_round:
            raise self._error(f"expected round {self.next_round}, got {ready.round!r}")
        wakeup = ready.next_wakeup_ns
        if wakeup is not None and (type(wakeup) is not int or wakeup < self.now_ns):
            raise self._error(f"next_wakeup_ns precedes now: {wakeup!r}")
        if self._driver_finished and not ready.finished:
            raise self._error("finished driver cannot resume work")
        if ready.finished and wakeup is not None:
            raise self._error("finished driver cannot register a timer")

        submitted: set[str] = set()
        for op in ready.operations:
            if not isinstance(op, (Submit, Cancel)):
                raise self._error(f"unsupported operation {type(op).__name__}")
            if not op.rid or len(op.rid) > 256:
                raise self._error("rid must contain 1 to 256 characters")
            if isinstance(op, Submit):
                if ready.finished:
                    raise self._error(f"finished driver submitted rid={op.rid!r}")
                if op.rid in self._requests or op.rid in submitted:
                    raise self._error(f"duplicate Submit rid={op.rid!r}")
                submitted.add(op.rid)
            elif op.rid not in self._requests and op.rid not in submitted:
                raise self._error(f"Cancel for unknown rid={op.rid!r}")

        self.run_id = ready.run_id
        self._ready = ready
        self._driver_finished = ready.finished
        for op in ready.operations:
            if isinstance(op, Submit):
                self._requests[op.rid] = _Request(len(self._requests), self.now_ns)
                self._pending.append(op.rid)
            else:
                self.terminal(op.rid, "CANCELLED", reason=op.reason)

    def mark_admitted(self, rid: str) -> None:
        if self._ready is None:
            raise self._error(f"cannot admit rid={rid!r} before Ready")
        self._skip_terminal_payloads()
        if not self._pending or rid != self._pending[0]:
            raise self._error(
                f"payload rid={rid!r} is not next in manifest {self.pending_rids}"
            )
        self._pending.popleft()
        self._requests[rid].admitted = True

    def advance(self, now_ns: int) -> None:
        if self._ready is None:
            raise self._error("cannot advance before Ready")
        if type(now_ns) is not int or now_ns < self.now_ns:
            raise self._error(f"virtual time cannot go backwards: {now_ns!r}")
        if self.pending_rids:
            raise self._error(f"missing payloads for {self.pending_rids}")
        if self.has_terminals and now_ns > self.now_ns:
            raise self._error("must deliver terminal events before advancing")
        if self.next_wakeup_ns is not None and now_ns > self.next_wakeup_ns:
            raise self._error(f"cannot advance past timer at {self.next_wakeup_ns}")
        self.now_ns = now_ns

    def terminal(
        self, rid: str, kind: TerminalKind, *, reason: str | None = None
    ) -> bool:
        """Commit at most one terminal; late native completions are ignored."""
        request = self._requests.get(rid)
        if request is None:
            raise self._error(f"terminal for unknown rid={rid!r}")
        if kind not in ("COMPLETED", "CANCELLED", "ERROR"):
            raise self._error(f"invalid terminal kind {kind!r}")
        if request.terminal is not None:
            return False
        if self._ready is None:
            raise self._error(f"cannot commit rid={rid!r} before Ready")
        if kind == "COMPLETED" and not request.admitted:
            raise self._error(f"cannot complete unadmitted rid={rid!r}")
        event = Terminal(rid, kind, self.now_ns, reason)
        request.terminal = event
        self._terminals.append(event)
        return True

    def grant(self, *, finished: bool = False) -> Grant:
        """Close this round after the bridge has applied all scheduler work.

        The bridge must additionally check batch and native cleanup state before
        requesting FINISHED; a cancelled request can still occupy a batch.
        """
        if self._ready is None:
            raise self._error("no Ready to answer")
        if self.pending_rids:
            raise self._error(f"missing payloads for {self.pending_rids}")
        if finished and (not self.driver_finished or self.open_rids):
            raise self._error("cannot finish with an active driver or open requests")
        terminals = sorted(
            self._terminals, key=lambda event: self._requests[event.rid].sequence
        )
        result = Grant(
            self.run_id,
            self.next_round,
            self.now_ns,
            terminals,
            "FINISHED" if finished else "CONTINUE",
        )
        self.next_round += 1
        self._ready = None
        self._terminals = []
        self._closed = finished
        return result
