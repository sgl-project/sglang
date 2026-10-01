"""Background cohort ownership; request admission performs only local work."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import ClassVar, Literal

import msgspec
import torch
from sglang.srt.training_capture.cohort import (
    STATE_COLUMNS,
    CaptureCohort,
    CaptureCohortAllocator,
    CaptureCohortError,
)
from sglang.srt.training_capture.cohort_exchange import (
    agree_receipts,
    agree_snapshot,
    check_receipt,
    snapshot_offer,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    Identifier,
    Positive,
    StrictStruct,
    canonical_bytes,
    decode_manifest,
)
from sglang.srt.training_capture.snapshot_writer import OwnerWriteReceipt


class CaptureTicket(StrictStruct):
    capture_id: Identifier
    fencing_token: Positive
    request_sha256: Digest


@dataclass(eq=False)
class CaptureHandle:
    """Rank-local ownership, valid until finish; never serialize this object.

    Read cohort/invalid_reason through service.status(). The service can renew
    or invalidate a handle while its actor is copying or writing its tensors.
    """

    cohort: CaptureCohort
    ticket: CaptureTicket | None = None
    available: bool = False
    bound: bool = False
    drained: bool = False
    invalid_reason: str | None = None
    published: bool = False
    transfer_complete: bool = True
    claimed_at: float | None = None
    renewal_failed: bool = False
    snapshot_payload: bytes | None = None
    manifest_payload: bytes | None = None
    receipt_payload: bytes | None = None
    receipts_payload: bytes | None = None


def _request_words(fingerprint):
    payload = bytes.fromhex(fingerprint)
    return tuple(
        int.from_bytes(payload[offset : offset + 8], "little", signed=True)
        for offset in range(0, 32, 8)
    )


def _request_digest(words):
    return b"".join(word.to_bytes(8, "little", signed=True) for word in words).hex()


class CaptureCohortService:
    """One background thread per rank, activated after resource readiness.

    Rank zero claims tickets before entering the PP pipeline. Each rank binds
    the same ticket locally, then explicitly finishes after its CUDA/Store work.
    A failure invalidates admission immediately but cannot release a bound slot.
    Only the aux owner may report publication, after durable journal recovery if
    the publish response was ambiguous. This service never closes resources or
    destroys the control group; its owner must first obtain a successful close.
    """

    _retained: ClassVar[set[CaptureCohortService]] = set()

    def __init__(self, allocator: CaptureCohortAllocator, *, poll_seconds=0.05):
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ContractError("invalid cohort polling interval")
        self.allocator = allocator
        self.partition = allocator.layout.partitions[allocator.rank]
        self.aux_rank = next(
            rank
            for rank, part in enumerate(allocator.layout.partitions)
            if part.include_aux
        )
        self.poll_seconds = poll_seconds
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.records: dict[str, CaptureHandle] = {}
        self.stopping = False
        self.admission_ready = True
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None
        self.activation: threading.Event | None = None
        shape = (allocator.config.max_inflight_samples + 1, STATE_COLUMNS)
        if math.prod(shape) * 8 * (allocator.size + 1) > 64 << 20:
            raise ContractError("cohort registry exceeds its control memory budget")
        self.frame = torch.zeros(shape, dtype=torch.int64, device="cpu")
        self.frames = [torch.empty_like(self.frame) for _ in range(allocator.size)]

    def start(self, *, activation=None):
        """Call on every rank after collective startup validation succeeds."""
        with self.lock:
            if self.thread is not None or self.stopping:
                raise ContractError("cohort service cannot be restarted")
            self.activation = activation
            self.thread = threading.Thread(
                target=self._run, name="capture-cohorts", daemon=True
            )
            self.thread.start()

    def claim(self, request_sha256: str) -> CaptureTicket | None:
        fingerprint = msgspec.convert(request_sha256, type=Digest)
        with self.lock:
            if (
                self.allocator.rank != 0
                or self.stopping
                or self.error is not None
                or not self.admission_ready
            ):
                return None
            now = time.monotonic()
            for handle in self.records.values():
                if (
                    handle.available
                    and handle.ticket is None
                    and handle.invalid_reason is None
                    and now < handle.cohort.deadline
                ):
                    lease = handle.cohort.lease
                    ticket = CaptureTicket(
                        capture_id=lease.capture_id,
                        fencing_token=lease.fencing_token,
                        request_sha256=fingerprint,
                    )
                    handle.ticket, handle.claimed_at = ticket, now
                    handle.available = False
                    self.wake.set()
                    return ticket
        return None

    def set_admission_ready(self, ready: bool):
        """Pause new tickets until every rank's writer/supervisor is ready."""
        if type(ready) is not bool:
            raise ContractError("admission readiness must be boolean")
        with self.lock:
            self.admission_ready = ready
            self.wake.set()

    def bind(self, ticket: CaptureTicket, request_sha256: str) -> CaptureHandle | None:
        ticket = msgspec.convert(msgspec.to_builtins(ticket), type=CaptureTicket)
        fingerprint = msgspec.convert(request_sha256, type=Digest)
        with self.lock:
            handle = self.records.get(ticket.capture_id)
            if (
                handle is None
                or self.stopping
                or self.error is not None
                or handle.bound
                or handle.invalid_reason is not None
                or handle.cohort.lease.fencing_token != ticket.fencing_token
                or time.monotonic() >= handle.cohort.deadline
            ):
                return None
            if ticket.request_sha256 != fingerprint or (
                handle.ticket is not None and handle.ticket != ticket
            ):
                handle.invalid_reason = "request_identity_mismatch"
                self.wake.set()
                return None
            handle.ticket, handle.bound = ticket, True
            handle.available = False
            handle.claimed_at = handle.claimed_at or time.monotonic()
            self.wake.set()
            return handle

    def _check_handle(self, handle):
        if (
            self.records.get(handle.cohort.lease.capture_id) is not handle
            or not handle.bound
            or handle.drained
        ):
            raise ContractError("foreign, unbound or finished capture handle")

    def cancel(self, ticket: CaptureTicket, reason: str) -> bool:
        """Cancel a claimed ticket even if its request never reached bind()."""
        ticket = msgspec.convert(msgspec.to_builtins(ticket), type=CaptureTicket)
        with self.lock:
            handle = self.records.get(ticket.capture_id)
            if (
                handle is None
                or handle.cohort.lease.fencing_token != ticket.fencing_token
                or (handle.ticket is not None and handle.ticket != ticket)
            ):
                return False
            handle.ticket = ticket
            handle.available = False
            handle.invalid_reason = (
                handle.invalid_reason or reason or "request_cancelled"
            )
            self.wake.set()
            return True

    def status(self, handle: CaptureHandle):
        with self.lock:
            self._check_handle(handle)
            reason = handle.invalid_reason
            if time.monotonic() >= handle.cohort.deadline:
                reason = reason or "lease_expired"
            return handle.cohort, reason

    def fail(self, handle: CaptureHandle, reason: str):
        with self.lock:
            self._check_handle(handle)
            handle.invalid_reason = handle.invalid_reason or reason or "capture_failed"
            self.wake.set()

    def _check_live_handle(self, handle):
        self._check_handle(handle)
        if (
            handle.invalid_reason is not None
            or self.stopping
            or self.error is not None
            or time.monotonic() >= handle.cohort.deadline
        ):
            raise ContractError("capture handle is no longer writable")

    def submit_snapshot(
        self, handle, *, execution_sha256, prepared=None, metadata=None
    ):
        """Submit completed descriptors from a writer actor; performs no collective.

        Every rank submits, including inactive ranks. Only the aux owner passes
        metadata; each active owner passes its own prepared partition. Tensor
        storage remains owned by the actor until finish().
        """
        try:
            with self.lock:
                self._check_live_handle(handle)
                cohort, ticket = handle.cohort, handle.ticket
            payload = snapshot_offer(
                cohort, ticket, self.partition, execution_sha256, prepared, metadata
            )
            if len(payload) > self.allocator.config.manifest_buffer_bytes + 4096:
                raise ContractError("snapshot offer exceeds the control budget")
            with self.lock:
                self._check_live_handle(handle)
                if handle.snapshot_payload is not None:
                    raise ContractError("snapshot descriptors were already submitted")
                handle.snapshot_payload = payload
                self.wake.set()
        except Exception:
            self.fail(handle, "snapshot_submission_failed")
            raise

    def get_manifest(self, handle):
        """Return a fresh decoded view, or None while peer descriptors are pending."""
        with self.lock:
            self._check_live_handle(handle)
            payload = handle.manifest_payload
        return None if payload is None else decode_manifest(payload)

    def submit_receipt(self, handle, receipt):
        """Report a completed owner write to the exact agreed manifest."""
        try:
            with self.lock:
                self._check_live_handle(handle)
                if not self.partition.active or handle.manifest_payload is None:
                    raise ContractError("receipt requires an active owner and manifest")
                checked = check_receipt(
                    receipt,
                    cohort=handle.cohort,
                    owner_id=self.partition.owner_id,
                    manifest_payload=handle.manifest_payload,
                )
                if handle.receipt_payload is not None:
                    raise ContractError("owner receipt was already submitted")
                handle.receipt_payload = canonical_bytes(checked)
                self.wake.set()
        except Exception:
            self.fail(handle, "receipt_submission_failed")
            raise

    def get_receipts(self, handle):
        """Return fresh receipts after every canonical owner has completed writes."""
        with self.lock:
            self._check_live_handle(handle)
            payload = handle.receipts_payload
        return (
            None
            if payload is None
            else msgspec.json.decode(payload, type=tuple[OwnerWriteReceipt, ...])
        )

    def finish(
        self,
        handle: CaptureHandle,
        *,
        outcome: Literal["stored", "published", "failed"],
        transfer_complete: bool,
    ):
        """Return actor ownership; uncertain transfers are permanently quarantined.

        An aux owner must resolve its publication journal before calling this.
        Failed/expired leases alone are not proof that a publication did not win.
        Recovered publication may still leave the original source quarantined;
        report published with transfer_complete=False in that case.
        """
        with self.lock:
            self._check_handle(handle)
            if (
                outcome not in ("stored", "published", "failed")
                or (outcome == "published" and not self.partition.include_aux)
                or (outcome == "stored" and self.partition.include_aux)
                or type(transfer_complete) is not bool
                or (outcome == "stored" and not transfer_complete)
            ):
                raise ContractError("invalid local capture completion")
            if (
                outcome != "failed"
                and handle.snapshot_payload is not None
                and (
                    handle.manifest_payload is None
                    or (self.partition.active and handle.receipt_payload is None)
                    or (outcome == "published" and handle.receipts_payload is None)
                )
            ):
                raise ContractError("completion precedes agreed metadata or writes")
            handle.drained = True
            handle.transfer_complete = transfer_complete
            handle.published = outcome == "published"
            if outcome == "failed":
                handle.invalid_reason = handle.invalid_reason or "local_capture_failed"
            if self.error is not None:
                self._release(handle)
                self._retain_if_needed()
            self.wake.set()

    def _build_frame(self):
        with self.lock:
            now = time.monotonic()
            self.frame.zero_()
            self.frame[0, 0] = int(self.stopping)
            self.frame[0, 1] = int(self.admission_ready)
            ledger = []
            for index, handle in enumerate(self.records.values(), start=1):
                cohort = handle.cohort
                if self.stopping:
                    handle.invalid_reason = handle.invalid_reason or "service_stopping"
                if now >= cohort.deadline:
                    handle.invalid_reason = handle.invalid_reason or "lease_expired"
                if (
                    handle.claimed_at is not None
                    and now - handle.claimed_at
                    >= self.allocator.config.max_capture_seconds
                ):
                    handle.invalid_reason = handle.invalid_reason or "capture_timeout"
                drained = handle.drained or not handle.bound
                flags = (
                    handle.invalid_reason is not None,
                    handle.bound,
                    drained,
                    handle.published,
                    not handle.renewal_failed
                    and not handle.published
                    and now >= cohort.renew_at,
                    handle.ticket is not None,
                )
                for column, value in enumerate(flags):
                    self.frame[index, column] = int(value)
                if handle.ticket is not None:
                    for column, value in enumerate(
                        _request_words(handle.ticket.request_sha256), start=6
                    ):
                        self.frame[index, column] = value
                for column, payload in enumerate(
                    (
                        handle.snapshot_payload,
                        handle.receipt_payload,
                        handle.manifest_payload,
                        handle.receipts_payload,
                    ),
                    start=10,
                ):
                    self.frame[index, column] = int(payload is not None)
                ledger.append((cohort.lease, cohort.reserved_bytes))
            return ledger, self.frame, self.frames

    def _release(self, handle):
        if handle.cohort.slot is not None:
            self.allocator.resources.pool.release(
                handle.cohort.slot, transfer_complete=handle.transfer_complete
            )
        del self.records[handle.cohort.lease.capture_id]

    def _retain_if_needed(self):
        pool = self.allocator.resources.pool
        stats = pool.stats() if pool is not None else {}
        if self.records or stats.get("quarantined", 0) or stats.get("filling", 0):
            self._retained.add(self)
        else:
            self._retained.discard(self)

    def _cycle(self):
        frames = self.allocator.synchronize(self._build_frame)
        # Only the background thread inserts/removes records. Foreground calls
        # can bind during the exchange, so retirement needs a prior invalid vote
        # from EVERY rank or a completed, bound owner on the success path.
        with self.lock:
            handles = list(self.records.values())
            self.stopping |= any(frame[0][0] for frame in frames)
            for index, handle in enumerate(handles, start=1):
                rows = [frame[index] for frame in frames]
                fingerprints = {tuple(row[6:10]) for row in rows if row[5]}
                if len(fingerprints) > 1:
                    handle.invalid_reason = "request_identity_mismatch"
                elif fingerprints and handle.ticket is None:
                    lease = handle.cohort.lease
                    handle.ticket = CaptureTicket(
                        capture_id=lease.capture_id,
                        fencing_token=lease.fencing_token,
                        request_sha256=_request_digest(next(iter(fingerprints))),
                    )
                    handle.claimed_at = time.monotonic()
                if any(
                    row[3] for rank, row in enumerate(rows) if rank != self.aux_rank
                ):
                    raise ContractError("only the aux owner can publish a capture")
                if any(row[0] for row in rows) or self.stopping:
                    handle.invalid_reason = (
                        handle.invalid_reason or "peer_capture_failed"
                    )
                handle.available = (
                    handle.ticket is None
                    and handle.invalid_reason is None
                    and not self.stopping
                    and all(frame[0][1] for frame in frames)
                )

        for index, handle in enumerate(handles, start=1):
            rows = [frame[index] for frame in frames]
            published = bool(rows[self.aux_rank][3])
            actors_drained = all(row[2] for row in rows)
            owners_bound = all(
                row[1]
                for part, row in zip(
                    self.allocator.layout.partitions, rows, strict=True
                )
                if part.active
            )
            terminal = actors_drained and (
                (published and owners_bound) or all(row[0] for row in rows)
            )
            if terminal:
                if not published:
                    self.allocator.fail(handle.cohort)
                with self.lock:
                    self._release(handle)
            elif any(row[4] for row in rows) and not published:
                try:
                    renewed = self.allocator.renew(handle.cohort)
                except CaptureCohortError as error:
                    if self.allocator.poisoned:
                        raise
                    with self.lock:
                        handle.invalid_reason = f"renewal_failed:{error.phase}"
                        handle.renewal_failed = True
                else:
                    with self.lock:
                        handle.cohort = renewed
            elif not published and not any(row[0] for row in rows):
                kind = None
                if all(row[10] for row in rows) and not any(row[12] for row in rows):
                    kind = "snapshot"
                elif (
                    all(row[12] for row in rows)
                    and not any(row[13] for row in rows)
                    and all(
                        row[11]
                        for part, row in zip(
                            self.allocator.layout.partitions, rows, strict=True
                        )
                        if part.active
                    )
                ):
                    kind = "receipts"
                if kind is not None:
                    self._exchange(handle, kind)

        # Admission/stop changes during the exchange are applied on the NEXT
        # exchange. Every rank must choose reserve using the same voted state.
        if not any(frame[0][0] for frame in frames) and (
            len(self.records) < self.allocator.config.max_inflight_samples
        ):
            cohort = self.allocator.reserve()
            if cohort is not None:
                with self.lock:
                    self.records[cohort.lease.capture_id] = CaptureHandle(cohort=cohort)
        with self.lock:
            return any(frame[0][0] for frame in frames) and not self.records

    def _exchange(self, handle, kind):
        cohort = handle.cohort
        if kind == "snapshot":
            build_local = lambda: handle.snapshot_payload

            def validate(payloads):
                return agree_snapshot(
                    payloads,
                    cohort=cohort,
                    request_sha256=handle.ticket.request_sha256,
                    allocator=self.allocator,
                )

        else:
            build_local = lambda: (
                handle.receipt_payload if self.partition.active else b"null"
            )

            def validate(payloads):
                return agree_receipts(
                    payloads,
                    cohort=cohort,
                    manifest_payload=handle.manifest_payload,
                    layout=self.allocator.layout,
                )

        try:
            payload = self.allocator.exchange(
                cohort,
                kind=kind,
                max_bytes=self.allocator.config.manifest_buffer_bytes + 4096,
                build_local=build_local,
                validate=validate,
            )
        except CaptureCohortError as error:
            if self.allocator.poisoned:
                raise
            with self.lock:
                handle.invalid_reason = f"{kind}_exchange_failed:{error.phase}"
        else:
            with self.lock:
                if kind == "snapshot":
                    handle.manifest_payload = payload
                else:
                    handle.receipts_payload = payload

    def _run(self):
        try:
            if self.activation is not None:
                while not self.activation.wait(self.poll_seconds):
                    if self.stopping:
                        return
            while not self._cycle():
                self.wake.wait(self.poll_seconds)
                self.wake.clear()
        except Exception as error:  # noqa: BLE001 - never release live transfer buffers
            with self.lock:
                self.error = error
                self.stopping = True
                self._retained.add(self)
                for handle in list(self.records.values()):
                    handle.invalid_reason = handle.invalid_reason or "control_failed"
                    if handle.drained or not handle.bound:
                        self._release(handle)
                self._retain_if_needed()

    def close(self, timeout=5.0) -> bool:
        """Signal collective stop; false means resources must remain alive."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("invalid cohort shutdown timeout")
        with self.lock:
            self.stopping = True
            thread = self.thread
            self.wake.set()
        if thread is not None and thread.ident is not None:
            thread.join(timeout)
        with self.lock:
            self._retain_if_needed()
            return (
                not (thread is not None and thread.is_alive())
                and self not in self._retained
            )
