"""Single Store actor advancing multiple completed cohort partitions fairly."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import ClassVar

import msgspec
import torch
from sglang.srt.training_capture.catalog import CaptureLease
from sglang.srt.training_capture.cohort_service import CaptureHandle
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.mooncake_store import TransportError
from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    Manifest,
    canonical_bytes,
)
from sglang.srt.training_capture.snapshot import SnapshotMetadata
from sglang.srt.training_capture.snapshot_writer import (
    OwnerWriteReceipt,
    SnapshotWriter,
)
from sglang.srt.training_capture.timings import CaptureTimings

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _Write:
    handle: CaptureHandle
    context: RequestCaptureContext | None
    execution_sha256: str | None
    metadata_payload: bytes | None
    failure_reason: str | None
    queued_at: float = field(default_factory=time.monotonic)
    phase: str = "copies"
    transfer_complete: bool = False
    tensors: dict[str, torch.Tensor] | None = None
    manifest: Manifest | None = None
    receipts: tuple[OwnerWriteReceipt, ...] | None = None
    publication_lease: CaptureLease | None = None
    retry_at: float = 0.0


class CohortSnapshotWriter:
    """Own sealed/aborted contexts after submit, without reading serving requests.

    Start after resource readiness and run alongside the cohort service. There
    is one Store thread per rank. Pending peer metadata does not block another
    completed request: rank-local completion orders need not match. close() only
    stops this writer; the supervisor must separately drain the cohort service
    before closing transport or destroying its group.
    """

    _retained: ClassVar[set[CohortSnapshotWriter]] = set()

    def __init__(self, service, *, poll_seconds=0.01, retry_seconds=5.0):
        if any(
            not math.isfinite(value) or value <= 0
            for value in (poll_seconds, retry_seconds)
        ):
            raise ContractError("invalid cohort writer polling interval")
        self.service = service
        self.partition = service.partition
        self.resources = service.allocator.resources
        self.timings = CaptureTimings()
        self.writer = (
            SnapshotWriter(
                self.resources.store,
                self.resources.catalog,
                self.resources.journal,
                timings=self.timings,
            )
            if self.partition.active
            else None
        )
        self.poll_seconds, self.retry_seconds = poll_seconds, retry_seconds
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.jobs: dict[str, _Write] = {}
        self.thread = None
        self.activation = None
        self.stopping = False
        self.ready = False
        self.error = None
        self.recovery_error = None
        self.counters = Counter()

    def start(self, *, activation=None):
        with self.lock:
            if self.thread is not None or self.stopping:
                raise ContractError("cohort writer cannot be restarted")
            self.activation = activation
            self.thread = threading.Thread(
                target=self._run, name="cohort-writer", daemon=True
            )
            self.thread.start()

    def submit(
        self,
        handle,
        *,
        context=None,
        metadata=None,
        execution_sha256=None,
        failure_reason=None,
    ):
        """Transfer ownership on success; an exception leaves it with the caller.

        Every bound rank submits, including inactive ranks (no context/metadata).
        Active successful captures provide a sealed context and matching metadata.
        Failure submissions still drain outstanding CUDA copies before finish.
        A missing context on failure asserts that no CUDA copy was ever enqueued.
        """
        cohort, _ = self.service.status(handle)
        if context is not None and (
            context.slot is not cohort.slot
            or context.partition != self.partition
            or context.state not in ("SEALED", "FAILED")
        ):
            raise ContractError("writer requires its own sealed or aborted context")
        if not failure_reason and self.partition.active != (context is not None):
            raise ContractError("context does not match local payload ownership")
        if not self.partition.active and (context is not None or metadata is not None):
            raise ContractError("inactive actor cannot submit payload metadata")
        if not failure_reason:
            execution_sha256 = msgspec.convert(execution_sha256, type=Digest)
            if self.partition.active and metadata is None:
                raise ContractError("completed partition requires snapshot metadata")
        payload = canonical_bytes(metadata) if metadata is not None else None
        if (
            payload is not None
            and len(payload) > self.service.allocator.config.manifest_buffer_bytes
        ):
            raise ContractError("writer metadata exceeds its reserved budget")
        with self.lock:
            if (
                self.stopping
                or self.error is not None
                or self.thread is None
                or not self.ready
            ):
                raise ContractError("cohort writer is not accepting completed actors")
            capture_id = cohort.lease.capture_id
            if (
                capture_id in self.jobs
                or len(self.jobs) >= self.service.allocator.config.max_inflight_samples
            ):
                raise ContractError("duplicate actor or full cohort writer")
            self.jobs[capture_id] = _Write(
                handle, context, execution_sha256, payload, failure_reason
            )
            self.counters["submitted"] += 1
            self.wake.set()

    def stats(self, *, include_timings=True):
        with self.lock:
            return {
                "ready": self.ready,
                "stopping": self.stopping,
                "error": type(self.error).__name__ if self.error is not None else None,
                "recovery_error": self.recovery_error,
                "pending": len(self.jobs),
                "states": dict(Counter(job.phase for job in self.jobs.values())),
                "counters": dict(self.counters),
                **({"stage_timings": self.timings.stats()} if include_timings else {}),
                "oldest_seconds": max(
                    (time.monotonic() - job.queued_at for job in self.jobs.values()),
                    default=0.0,
                ),
            }

    def _finish(self, job, outcome):
        slot = job.handle.cohort.slot
        complete = job.transfer_complete and (
            slot is None
            or slot.storage.data_ptr() not in self.resources.store.quarantined
        )
        self.service.finish(job.handle, outcome=outcome, transfer_complete=complete)
        with self.lock:
            del self.jobs[job.handle.cohort.lease.capture_id]
            self.counters[outcome] += 1

    def _fail(self, job, reason):
        self.service.fail(job.handle, reason)
        self._finish(job, "failed")

    def _advance(self, job):
        try:
            if job.phase == "copies":
                self.timings.observe(
                    "queue_wait", max(0.0, time.monotonic() - job.queued_at)
                )
                if job.context is not None:
                    self.timings.call("copy_wait", job.context.wait_for_copies)
                job.transfer_complete = True
            # An ambiguous publication can have won even after cancel/expiry.
            # Keep its exact manifest/lease until idempotent recovery confirms it.
            if job.phase == "recovering":
                if time.monotonic() >= job.retry_at:
                    self.writer.recover_partitions(
                        job.manifest, job.receipts, job.publication_lease
                    )
                    self._finish(job, "published")
                return
            cohort, invalid = self.service.status(job.handle)
            if invalid or job.failure_reason or self.stopping:
                self._fail(job, invalid or job.failure_reason or "writer_stopping")
                return
            if job.phase == "copies":
                prepared, metadata = None, None
                if job.context is not None:
                    metadata = msgspec.json.decode(
                        job.metadata_payload, type=SnapshotMetadata
                    )
                    if metadata.sequence != job.context.sequence:
                        raise ContractError(
                            "writer metadata differs from sealed sequence"
                        )
                    prepared, job.tensors = self.timings.call(
                        "snapshot_build",
                        job.context.prepare_partition,
                        payload_hasher=self.resources.store.payload_hasher,
                        **{
                            name: getattr(metadata, name)
                            for name in SnapshotMetadata.__struct_fields__
                            if name != "sequence"
                        },
                    )
                self.service.submit_snapshot(
                    job.handle,
                    execution_sha256=job.execution_sha256,
                    prepared=prepared,
                    metadata=metadata if self.partition.include_aux else None,
                )
                job.phase = "manifest"
            elif job.phase == "manifest":
                manifest = self.service.get_manifest(job.handle)
                if manifest is None:
                    return
                job.manifest = manifest
                if not self.partition.active:
                    self._finish(job, "stored")
                    return
                receipt = self.writer.write_partition(
                    manifest,
                    job.tensors,
                    cohort.lease,
                    owner_id=self.partition.owner_id,
                )
                self.service.submit_receipt(job.handle, receipt)
                if self.partition.include_aux:
                    job.phase = "receipts"
                else:
                    self._finish(job, "stored")
            elif job.phase == "receipts":
                receipts = self.service.get_receipts(job.handle)
                if receipts is None:
                    return
                job.receipts, job.publication_lease = receipts, cohort.lease
                job.phase = "publishing"
                self.writer.publish_partitions(
                    job.manifest, receipts, cohort.slot.manifest_buffer, cohort.lease
                )
                self._finish(job, "published")
        except Exception as error:  # noqa: BLE001 - ownership outlives ambiguous I/O
            if isinstance(error, TransportError) or (
                job.context is not None and job.context.transfer_uncertain
            ):
                job.transfer_complete = False
            with self.lock:
                self.counters["error_" + type(error).__name__] += 1
            if job.phase in ("publishing", "recovering"):
                job.phase = "recovering"
                job.retry_at = time.monotonic() + self.retry_seconds
                logger.warning(
                    "Cohort publication recovery pending: %s", type(error).__name__
                )
            else:
                self._fail(job, "writer_" + type(error).__name__)

    def _run(self):
        recovery_at = 0.0
        try:
            if self.activation is not None:
                while not self.activation.wait(self.poll_seconds):
                    if self.stopping:
                        return
            torch.set_num_threads(1)
            if not self.partition.include_aux:
                with self.lock:
                    self.ready = True
            while True:
                with self.lock:
                    jobs = list(self.jobs.values())
                    if self.stopping and not jobs:
                        return
                if not self.ready and time.monotonic() >= recovery_at:
                    try:
                        self.writer.recover()
                    except Exception as error:  # noqa: BLE001 - retain journal
                        with self.lock:
                            self.recovery_error = type(error).__name__
                        recovery_at = time.monotonic() + self.retry_seconds
                    else:
                        with self.lock:
                            self.ready, self.recovery_error = True, None
                for job in jobs:
                    self._advance(job)
                self.wake.wait(self.poll_seconds)
                self.wake.clear()
        except Exception as error:  # noqa: BLE001 - retain all unacknowledged contexts
            with self.lock:
                self.error, self.stopping = error, True
        finally:
            with self.lock:
                if self.jobs:
                    self._retained.add(self)
                else:
                    self._retained.discard(self)

    def close(self, timeout=5.0):
        """Drain owned actors; false retains them and prohibits resource teardown."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("invalid cohort writer shutdown timeout")
        with self.lock:
            self.stopping = True
            thread = self.thread
            self.wake.set()
        if thread is not None and thread.ident is not None:
            thread.join(timeout)
        with self.lock:
            stopped = not (thread is not None and thread.is_alive()) and not self.jobs
            if stopped:
                self._retained.discard(self)
            else:
                self._retained.add(self)
            return stopped
