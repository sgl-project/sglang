"""All-owner Host reservations before a request enters the PP pipeline."""

from __future__ import annotations

import math
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import timedelta

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.training_capture.catalog import CaptureLease
from sglang.srt.training_capture.host_pool import HostSlot
from sglang.srt.training_capture.protocol import (
    ContractError,
    canonical_bytes,
    digest_bytes,
)

_VERSION = 3
_LEASE_BYTES = 4096
STATE_COLUMNS = 14
_PHASES = (
    "policy",
    "slots",
    "lease",
    "validation",
    "ready",
    "rollback",
    "renew",
    "retire",
    "retired",
    "state",
    "state_validation",
    "state_ready",
    "exchange",
    "exchange_policy",
    "exchange_sizes",
    "exchange_buffers",
    "exchange_validated",
    "exchange_ready",
)


def _lease_identity(lease):
    return (
        lease.capture_id,
        lease.fencing_token,
        lease.dataset_id,
        lease.sample_id,
        lease.generation_id,
    )


def _digest_words(payload):
    digest = bytes.fromhex(digest_bytes(payload))
    return tuple(
        int.from_bytes(digest[offset : offset + 8], "little", signed=True)
        for offset in range(0, 32, 8)
    )


class CaptureCohortError(ContractError):
    def __init__(self, phase, failed_ranks=()):
        self.phase, self.failed_ranks = phase, tuple(failed_ranks)
        super().__init__(f"capture cohort {phase} failed; ranks={self.failed_ranks}")


class CaptureCohort(msgspec.Struct, frozen=True, kw_only=True):
    lease: CaptureLease
    slot: HostSlot | None
    deadline: float
    renew_at: float
    reserved_bytes: int


class CaptureCohortAllocator:
    """Blocking background protocol on a dedicated, PP-major Gloo group.

    All ranks, including inactive partitions, call reserve in the same order.
    None means collective backpressure; an exception means capture must stop or
    explicitly recover. A transport/protocol failure poisons this allocator.
    Returned slots have not had any CUDA/Store transfer enqueued. Their caller
    owns renewal, binding to requests, cancellation and eventual safe release.
    Never call this protocol from a model forward or on an inference group.
    """

    def __init__(
        self, *, group, layout, config, teacher, kv, resources, timeout_seconds=120.0
    ):
        if (
            group is None
            or group is dist.group.WORLD
            or str(dist.get_backend(group)) != "gloo"
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ContractError("cohort reservations require a dedicated Gloo group")
        self.group, self.layout, self.config = group, layout, config
        self.teacher, self.kv, self.resources = teacher, kv, resources
        self.rank, self.size = dist.get_rank(group), dist.get_world_size(group)
        if self.rank < 0 or self.size < 1:
            raise ContractError("cohort caller is outside its process group")
        self.timeout = timedelta(seconds=timeout_seconds)
        self.round = 0
        self.poisoned = False
        self.lock = threading.Lock()
        # Allocate control storage before reserving any request payload slots.
        self.header = torch.zeros(10, dtype=torch.int64, device="cpu")
        self.headers = [torch.empty_like(self.header) for _ in range(self.size)]
        self.body = torch.zeros(_LEASE_BYTES, dtype=torch.uint8, device="cpu")
        self.bodies = [torch.empty_like(self.body) for _ in range(self.size)]

    def _gather(self, outputs, value):
        try:
            work = dist.all_gather(outputs, value, group=self.group, async_op=True)
            if not work.wait(self.timeout):
                raise TimeoutError("cohort collective did not complete")
        except Exception as error:
            self.poisoned = True
            raise CaptureCohortError("transport") from error

    def _vote(self, phase, error=None, *, nbytes=0, length=0, fingerprint=()):
        self.header.zero_()
        prefix = [_VERSION, self.round, _PHASES.index(phase)]
        for index, value in enumerate(
            prefix + [int(error is not None), nbytes, length]
        ):
            self.header[index] = value
        for index, word in enumerate(fingerprint, start=6):
            self.header[index] = word
        self._gather(self.headers, self.header)
        values = [item.tolist() for item in self.headers]
        if any(item[:3] != prefix or item[3] not in (0, 1) for item in values):
            self.poisoned = True
            raise CaptureCohortError("protocol", range(self.size)) from error
        failed = [rank for rank, item in enumerate(values) if item[3]]
        if failed:
            raise CaptureCohortError(phase, failed) from error
        return values

    def _policy(self):
        layout = self.layout
        if (
            len(layout.partitions) != self.size
            or layout.topology.tp_size * layout.topology.pp_size != self.size
            or len({part.owner_id for part in layout.partitions}) != self.size
            or {part.owner_id for part in layout.partitions if part.active}
            != set(layout.topology.owners)
            or [part.owner_id for part in layout.partitions if part.include_aux]
            != [layout.topology.aux_owner]
        ):
            raise ContractError("cohort group differs from canonical ownership")
        partition = layout.partitions[self.rank]
        partition.local_layers(self.kv)
        if partition.active != (self.resources.pool is not None):
            raise ContractError("cohort resources differ from local ownership")
        if partition.include_aux and self.resources.catalog is None:
            raise ContractError("aux owner requires a Catalog client")
        policy = canonical_bytes(
            (self.config.startup_policy, self.teacher, self.kv, layout)
        )
        if len(policy) > 1 << 20:
            raise ContractError("cohort policy exceeds its metadata budget")
        return _digest_words(policy)

    def _decode_lease(self, payload, *, slot, anchor, reserved_bytes):
        shared = msgspec.json.decode(payload, type=CaptureLease)
        if shared.dataset_id != self.config.dataset_id:
            raise ContractError("lease belongs to a different dataset")
        cohort = CaptureCohort(
            lease=shared,
            slot=slot,
            deadline=anchor + shared.expires_in_seconds,
            renew_at=anchor + shared.renew_after_seconds,
            reserved_bytes=reserved_bytes,
        )
        if time.monotonic() >= cohort.renew_at:
            raise ContractError("reserved lease already requires renewal")
        return cohort

    def reserve(self) -> CaptureCohort | None:
        with self._operation():
            return self._reserve()

    @contextmanager
    def _operation(self):
        if self.poisoned or not self.lock.acquire(blocking=False):
            raise ContractError("cohort allocator is poisoned or already in use")
        self.round += 1
        try:
            yield
        finally:
            self.lock.release()

    def _reserve(self):
        slot, lease, ready = None, None, False
        try:
            error, fingerprint = None, ()
            try:
                fingerprint = self._policy()
            except Exception as cause:  # noqa: BLE001 - every rank must vote
                error = cause
            votes = self._vote("policy", error, fingerprint=fingerprint)
            if len({tuple(item[6:]) for item in votes}) != 1:
                raise CaptureCohortError("policy_agreement", range(self.size))

            partition = self.layout.partitions[self.rank]
            error, nbytes = None, 0
            try:
                if partition.active:
                    slot = self.resources.pool.acquire()
                    if slot is not None:
                        nbytes = slot.storage.numel()
                        if not 0 < nbytes < 1 << 63:
                            raise ContractError("invalid local reservation size")
            except Exception as cause:  # noqa: BLE001 - peers must release their slots
                error = cause
            # Every rank records an anchor before any aux Catalog begin can run.
            # Local deadlines are conservative without comparing host clocks.
            anchor = time.monotonic()
            votes = self._vote("slots", error, nbytes=nbytes)
            if any(
                part.active and vote[4] == 0
                for part, vote in zip(self.layout.partitions, votes, strict=True)
            ):
                return None
            reserved_bytes = sum(vote[4] for vote in votes)

            error, payload = None, b""
            try:
                self.body.zero_()
                if partition.include_aux:
                    sample_id, generation_id = uuid.uuid4().hex, uuid.uuid4().hex
                    candidate = self.resources.catalog.begin(
                        {
                            "dataset_id": self.config.dataset_id,
                            "sample_id": sample_id,
                            "generation_id": generation_id,
                            "contract_id": self.config.contract_id,
                            "teacher": msgspec.to_builtins(self.teacher),
                            "kv": msgspec.to_builtins(self.kv),
                            "owners": list(self.layout.topology.owners),
                            "reserved_bytes": reserved_bytes,
                            "lease_seconds": self.config.capture_lease_seconds,
                            "idempotency_key": f"begin-{sample_id}-{generation_id}",
                        }
                    )
                    candidate = msgspec.convert(
                        msgspec.to_builtins(candidate), type=CaptureLease
                    )
                    if (
                        candidate.dataset_id,
                        candidate.sample_id,
                        candidate.generation_id,
                    ) != (self.config.dataset_id, sample_id, generation_id):
                        raise ContractError("Catalog changed the requested identity")
                    lease = candidate
                    payload = canonical_bytes(lease)
                    if len(payload) > _LEASE_BYTES:
                        raise ContractError("capture lease exceeds control capacity")
                    self.body[: len(payload)].copy_(
                        torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                    )
            except Exception as cause:  # noqa: BLE001 - Catalog failure is collective
                error = cause
            cohort = self._exchange_lease(
                payload, error, slot=slot, anchor=anchor, reserved_bytes=reserved_bytes
            )
            ready = True
            return cohort
        finally:
            if not ready:
                self._rollback(slot, lease)

    def _exchange_lease(
        self, payload, error, *, slot, anchor, reserved_bytes, expected=None
    ):
        votes = self._vote("lease", error, length=len(payload))
        self._gather(self.bodies, self.body)
        error, cohort, fingerprint = None, None, ()
        try:
            aux = next(
                index
                for index, part in enumerate(self.layout.partitions)
                if part.include_aux
            )
            if not 0 < votes[aux][5] <= _LEASE_BYTES or any(
                row[5] != 0 for index, row in enumerate(votes) if index != aux
            ):
                raise ContractError("lease payload was not supplied by the aux owner")
            cohort = self._decode_lease(
                memoryview(self.bodies[aux].numpy())[: votes[aux][5]],
                slot=slot,
                anchor=anchor,
                reserved_bytes=reserved_bytes,
            )
            if expected is not None and _lease_identity(
                cohort.lease
            ) != _lease_identity(expected):
                raise ContractError("heartbeat changed the capture identity or fence")
            fingerprint = _digest_words(canonical_bytes(cohort.lease))
        except Exception as cause:  # noqa: BLE001 - no rank may return alone
            error = cause
        votes = self._vote("validation", error, fingerprint=fingerprint)
        if len({tuple(item[6:]) for item in votes}) != 1:
            raise CaptureCohortError("lease_agreement", range(self.size))
        error = (
            ContractError("lease requires renewal before readiness")
            if time.monotonic() >= cohort.renew_at
            else None
        )
        self._vote("ready", error)
        return cohort

    def _agree_cohort(self, phase, cohort):
        error, fingerprint = None, ()
        try:
            lease = msgspec.convert(
                msgspec.to_builtins(cohort.lease), type=CaptureLease
            )
            fingerprint = _digest_words(canonical_bytes((lease, cohort.reserved_bytes)))
        except Exception as cause:  # noqa: BLE001 - peers must agree before Catalog I/O
            error = cause
        votes = self._vote(phase, error, fingerprint=fingerprint)
        if len({tuple(item[6:]) for item in votes}) != 1:
            raise CaptureCohortError(phase + "_agreement", range(self.size))

    def renew(self, cohort: CaptureCohort) -> CaptureCohort:
        """Renew one common lease; failure never releases an in-use Host slot."""
        with self._operation():
            anchor = time.monotonic()
            self._agree_cohort("renew", cohort)
            error, payload = None, b""
            try:
                if anchor >= cohort.deadline:
                    raise ContractError("cannot revive an expired local lease")
                self.body.zero_()
                if self.layout.partitions[self.rank].include_aux:
                    renewed = self.resources.catalog.heartbeat(cohort.lease)
                    renewed = msgspec.convert(
                        msgspec.to_builtins(renewed), type=CaptureLease
                    )
                    if _lease_identity(renewed) != _lease_identity(cohort.lease):
                        raise ContractError(
                            "heartbeat changed the capture identity or fence"
                        )
                    payload = canonical_bytes(renewed)
                    if len(payload) > _LEASE_BYTES:
                        raise ContractError("capture lease exceeds control capacity")
                    self.body[: len(payload)].copy_(
                        torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                    )
            except Exception as cause:  # noqa: BLE001 - renewal failure is collective
                error = cause
            return self._exchange_lease(
                payload,
                error,
                slot=cohort.slot,
                anchor=anchor,
                reserved_bytes=cohort.reserved_bytes,
                expected=cohort.lease,
            )

    def fail(self, cohort: CaptureCohort):
        """Fail a drained capture; the caller remains responsible for slot release."""
        with self._operation():
            self._agree_cohort("retire", cohort)
            error = None
            try:
                if self.layout.partitions[self.rank].include_aux:
                    result = self.resources.catalog.fail(cohort.lease, "cohort_failed")
                    if result.get("state") != "FAILED":
                        raise ContractError("Catalog did not confirm capture failure")
            except Exception as cause:  # noqa: BLE001 - every peer needs the outcome
                error = cause
            self._vote("retired", error)

    def synchronize(self, build_local):
        """Exchange a bounded registry frame built without holding inference locks."""
        with self._operation():
            error, fingerprint = None, ()
            send, outputs = None, None
            try:
                ledger, send, outputs = build_local()
                shape = (self.config.max_inflight_samples + 1, STATE_COLUMNS)
                if (
                    tuple(send.shape) != shape
                    or send.dtype != torch.int64
                    or send.device.type != "cpu"
                    or not send.is_contiguous()
                    or len(outputs) != self.size
                    or any(
                        tuple(output.shape) != shape
                        or output.dtype != send.dtype
                        or output.device.type != "cpu"
                        or not output.is_contiguous()
                        for output in outputs
                    )
                    or send.numel() * send.element_size() * (self.size + 1) > 64 << 20
                ):
                    raise ContractError("invalid or oversized cohort state frame")
                fingerprint = _digest_words(
                    canonical_bytes((self._policy(), ledger, shape))
                )
            except Exception as cause:  # noqa: BLE001 - local frame failure must be voted
                error = cause
            votes = self._vote("state", error, fingerprint=fingerprint)
            if len({tuple(item[6:]) for item in votes}) != 1:
                raise CaptureCohortError("state_agreement", range(self.size))
            self._gather(outputs, send)
            error, values = None, None
            try:
                values = [output.tolist() for output in outputs]
                if any(
                    value not in (0, 1)
                    for rows in values
                    for row in rows
                    for value in row[:6] + row[10:]
                ):
                    raise ContractError("invalid cohort state flags")
            except Exception as cause:  # noqa: BLE001 - validate before choosing an action
                error = cause
            self._vote("state_validation", error)
            self._vote("state_ready")
            return values

    def exchange(self, cohort, *, kind, max_bytes, build_local, validate):
        """Agree immutable metadata bytes, with voted allocation/validation failures.

        Payload tensors never enter this protocol. Callbacks run on the control
        thread; callers retain ownership of their registered transfer buffers.
        """
        with self._operation():
            self._agree_cohort("exchange", cohort)
            error, fingerprint = None, ()
            try:
                if kind not in ("snapshot", "receipts") or not (
                    type(max_bytes) is int and 0 < max_bytes <= 64 << 20
                ):
                    raise ContractError("invalid metadata exchange policy")
                fingerprint = _digest_words(canonical_bytes((kind, max_bytes)))
            except Exception as cause:  # noqa: BLE001 - no peer may advance alone
                error = cause
            votes = self._vote("exchange_policy", error, fingerprint=fingerprint)
            if len({tuple(item[6:]) for item in votes}) != 1:
                raise CaptureCohortError("exchange_policy_agreement", range(self.size))

            error, payload = None, b""
            try:
                payload = build_local()
                if type(payload) is not bytes or not 0 < len(payload) <= max_bytes:
                    raise ContractError("invalid or oversized local metadata")
            except Exception as cause:  # noqa: BLE001 - include local encoder failures
                error, payload = cause, b""
            votes = self._vote("exchange_sizes", error, length=len(payload))

            error, send, outputs = None, None, None
            try:
                lengths = [vote[5] for vote in votes]
                size = max(lengths)
                if any(not 0 < length <= max_bytes for length in lengths) or (
                    size * (self.size + 1) > 64 << 20
                ):
                    raise ContractError("metadata exchange exceeds control capacity")
                send = torch.zeros(size, dtype=torch.uint8, device="cpu")
                outputs = [torch.empty_like(send) for _ in range(self.size)]
                send[: len(payload)].copy_(
                    torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                )
            except Exception as cause:  # noqa: BLE001 - allocate before payload gather
                error = cause
            self._vote("exchange_buffers", error)
            self._gather(outputs, send)

            error, result, fingerprint = None, None, ()
            try:
                values = [
                    output[:length].numpy().tobytes()
                    for output, length in zip(outputs, lengths, strict=True)
                ]
                result = validate(values)
                if type(result) is not bytes or not 0 < len(result) <= max_bytes:
                    raise ContractError("invalid or oversized agreed metadata")
                fingerprint = _digest_words(result)
            except Exception as cause:  # noqa: BLE001 - parser failures are collective
                error = cause
            votes = self._vote("exchange_validated", error, fingerprint=fingerprint)
            if len({tuple(item[6:]) for item in votes}) != 1:
                raise CaptureCohortError("exchange_result_agreement", range(self.size))
            error = (
                ContractError("lease expired during metadata exchange")
                if time.monotonic() >= cohort.deadline
                else None
            )
            self._vote("exchange_ready", error)
            return result

    def _rollback(self, slot, lease):
        error = None
        try:
            if slot is not None:
                self.resources.pool.release(slot, transfer_complete=True)
        except Exception as cause:  # noqa: BLE001 - report cleanup failures to peers
            error = cause
        try:
            if lease is not None:
                self.resources.catalog.fail(lease, "cohort_reservation_failed")
        except Exception as cause:  # noqa: BLE001 - a failed lease cleanup needs a vote
            error = error or cause
        if not self.poisoned:
            self._vote("rollback", error)
        elif error is not None:
            raise CaptureCohortError("rollback", [self.rank]) from error
