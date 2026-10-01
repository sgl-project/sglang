"""All-owner Host reservations before a request enters the PP pipeline."""

from __future__ import annotations

import math
import threading
import time
import uuid
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

_VERSION = 1
_LEASE_BYTES = 4096
_PHASES = ("policy", "slots", "lease", "validation", "ready", "rollback")


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
        if self.poisoned or not self.lock.acquire(blocking=False):
            raise ContractError("cohort allocator is poisoned or already in use")
        self.round += 1
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
                    raise ContractError(
                        "lease payload was not supplied by the aux owner"
                    )
                cohort = self._decode_lease(
                    memoryview(self.bodies[aux].numpy())[: votes[aux][5]],
                    slot=slot,
                    anchor=anchor,
                    reserved_bytes=reserved_bytes,
                )
                fingerprint = _digest_words(canonical_bytes(cohort.lease))
            except Exception as cause:  # noqa: BLE001 - no rank may return alone
                error = cause
            votes = self._vote("validation", error, fingerprint=fingerprint)
            if len({tuple(item[6:]) for item in votes}) != 1:
                raise CaptureCohortError("lease_agreement", range(self.size))
            # A late participant must observe peers returning from validation.
            error = (
                ContractError("lease requires renewal before readiness")
                if time.monotonic() >= cohort.renew_at
                else None
            )
            self._vote("ready", error)
            ready = True
            return cohort
        finally:
            try:
                if not ready:
                    self._rollback(slot, lease)
            finally:
                self.lock.release()

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
