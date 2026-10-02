"""Request collection backed by all-rank cohort admission and owner-local writers."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter, deque
from typing import ClassVar

import torch.distributed as dist
from sglang.srt.training_capture.admission import CaptureAdmission
from sglang.srt.training_capture.cohort_service import CaptureCohortService
from sglang.srt.training_capture.cohort_writer import CohortSnapshotWriter
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.coordinator import (
    CaptureBatch,
    CaptureCoordinator,
    CaptureReservation,
    CaptureStep,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    canonical_bytes,
    digest_bytes,
)
from sglang.srt.training_capture.request_router import CaptureRequestRouter
from sglang.srt.training_capture.snapshot import SnapshotMetadata

logger = logging.getLogger(__name__)


class CohortCaptureCoordinator(CaptureCoordinator):
    """Reuse the validated AR/static-verify collector with partitioned ownership.

    The inference thread exclusively owns collecting contexts. Finalization
    detaches and queues them; the handoff thread gives them to the Store actor.
    The dedicated cohort service owns leases and buffer release on every rank.
    An owned control group is destroyed only after every actor and resource has
    closed. Otherwise the caller may destroy the group only after close=True.
    """

    _retained: ClassVar[set[CohortCaptureCoordinator]] = set()

    def __init__(
        self,
        *,
        allocator,
        req_to_token,
        enable_overlap=False,
        capture_mode="autoregressive",
        metrics=None,
        autostart=True,
        owns_control_group=False,
    ):
        self.config, self.teacher, self.kv = (
            allocator.config,
            allocator.teacher,
            allocator.kv,
        )
        self.resources = allocator.resources
        self.control_group = allocator.group
        self.owns_control_group = owns_control_group
        self.exporter, self.pool = self.resources.exporter, self.resources.pool
        self.req_to_token = req_to_token
        self.capture_mode, self.enable_overlap = capture_mode, enable_overlap
        self.layout, self.partition = (
            allocator.layout,
            allocator.layout.partitions[allocator.rank],
        )
        self.service = CaptureCohortService(allocator)
        self.service.set_admission_ready(False)
        self.writer_actor = CohortSnapshotWriter(self.service)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.activation = threading.Event()
        self.work = queue.Queue(maxsize=self.config.max_inflight_samples)
        self.available = deque()
        self.records = {}
        self.requests = {}
        self.counters = Counter()
        self.disabled_reason = None
        self.admission_paused = False
        self.error = None
        self.closed = False
        self.started = False
        self.admission = CaptureAdmission(
            self.config.sample_ratio, self.config.adaptive
        )
        self.policy_sha256 = digest_bytes(canonical_bytes(self.config.startup_policy))
        self.request_router = CaptureRequestRouter(
            self.service, sample_ratio=self._admission_ratio
        )
        self.metrics = metrics
        self.handoff_thread = threading.Thread(
            target=self._handoff_loop, name="capture-handoff", daemon=True
        )
        self.metrics_thread = (
            threading.Thread(
                target=self._metrics_loop, name="capture-metrics", daemon=True
            )
            if metrics is not None
            else None
        )
        if autostart:
            self.activate()

    def activate(self, *, defer=False):
        if self.started or self.closed or self.stop.is_set():
            raise ContractError("cohort coordinator cannot be restarted")
        self.started = True
        try:
            self.writer_actor.start(activation=self.activation)
            self.service.start(activation=self.activation)
            self.handoff_thread.start()
            if self.metrics_thread is not None:
                self.metrics_thread.start()
            if not defer:
                self.activation.set()
        except Exception:
            self.close()
            raise

    def _host_stats(self):
        if self.pool is not None:
            return self.pool.stats()
        return dict.fromkeys(
            (
                "free",
                "filling",
                "quarantined",
                "allocated_bytes",
                "device_allocated_bytes",
                "device_limit_bytes",
            ),
            0,
        )

    def _pressure(self, now):
        writer = self.writer_actor.stats(include_timings=False)
        return (
            min(
                1.0,
                (
                    len(self.records)
                    + writer["pending"]
                    + self._host_stats()["quarantined"]
                )
                / self.config.max_inflight_samples,
            ),
            max(
                writer["oldest_seconds"],
                max(
                    (
                        now - r.queued_at
                        for r in self.records.values()
                        if r.state == "queued"
                    ),
                    default=0.0,
                ),
            ),
        )

    def stats(self):
        writer = self.writer_actor.stats()
        with self.service.lock:
            reservations = len(self.service.records)
            available = sum(
                handle.available for handle in self.service.records.values()
            )
        with self.lock:
            occupancy, age = self._pressure(time.monotonic())
            states = Counter(record.state for record in self.records.values())
            states["available"] += available
            states["writing"] += writer["pending"] - writer["states"].get(
                "recovering", 0
            )
            states["pending_publication"] += writer["states"].get("recovering", 0)
            counters = dict(self.counters)
            counters["ready"] = writer["counters"].get("published", 0)
            return {
                "counters": counters,
                "disabled_reason": self.disabled_reason,
                "admission_paused": self.admission_paused,
                "enable_overlap": self.enable_overlap,
                "reservations": reservations,
                "states": dict(states),
                "queued": self.work.qsize(),
                "occupied_fraction": occupancy,
                "writer_age_seconds": age,
                "stage_timings": writer["stage_timings"],
                "host_pool": self._host_stats(),
                "admission": self.admission.stats(
                    time.monotonic(),
                    disabled=self.disabled_reason is not None or self.admission_paused,
                ),
                "cohort_writer": writer,
                "request_router": dict(self.request_router.counters),
            }

    def _admission_ratio(self):
        writer = self.writer_actor.stats(include_timings=False)
        if (
            self.disabled_reason
            or not writer["ready"]
            or writer["error"]
            or writer["states"].get("recovering", 0)
        ):
            return 0.0
        return super()._admission_ratio()

    def _admit(self, req):
        req.training_capture_attempted = True
        if self.disabled_reason:
            self.request_router.cancel(req, self.disabled_reason)
            return
        route = self.request_router.bind(req)
        if route is None:
            return
        record = None
        try:
            cohort, invalid = self.service.status(route.handle)
            record = CaptureReservation(
                cohort.lease,
                cohort.slot,
                cohort.deadline,
                cohort.renew_at,
                state="active",
                started=time.monotonic(),
                cohort_handle=route.handle,
                execution_sha256=route.execution_sha256,
            )
            with self.lock:
                self.records[cohort.lease.capture_id] = record
            if invalid:
                raise ContractError(invalid)
            record.context = RequestCaptureContext(
                slot=cohort.slot,
                prompt_ids=tuple(req.origin_input_ids),
                max_tokens=self.config.max_sample_tokens,
                vocab_size=self.teacher.vocab_size,
                partition=self.partition,
            )
            record.provenance = self._provenance(req, config_sha256=self.policy_sha256)
            req.training_capture_context = record
            req.training_capture_finalize = self.on_release
            self.requests[cohort.lease.capture_id] = req
            self._count("admitted")
        except Exception:  # noqa: BLE001 - admission failure must not reject inference
            reason = "admission_invalid_request"
            self.service.fail(route.handle, reason)
            if record is None:
                # bind transferred ownership, but no context or DMA exists yet.
                self.service.finish(
                    route.handle, outcome="failed", transfer_complete=True
                )
            else:
                record.invalid_reason = reason
                if record.context is not None:
                    record.context.abort(reason)
                self._detach(req, record)
            self._count("admission_invalid_request")

    def _refresh(self, record):
        cohort, reason = self.service.status(record.cohort_handle)
        record.lease, record.deadline, record.renew_at = (
            cohort.lease,
            cohort.deadline,
            cohort.renew_at,
        )
        record.invalid_reason = record.invalid_reason or reason or self.disabled_reason

    def before_forward(self, reqs):
        for req in list(self.requests.values()):
            self._refresh(req.training_capture_context)
        super().before_forward(reqs)

    def on_release(self, req):
        if req.training_capture_context is not None:
            self._refresh(req.training_capture_context)
        super().on_release(req)

    def after_result(self, ticket, *, requests=()):
        # PP relays committed tokens, not the rank-local forward ticket. Read the
        # scheduler's finalized request ledger after its ordinary result handler.
        if ticket is None:
            ticket = CaptureBatch(
                tuple(
                    CaptureStep(req, req.training_capture_context, None)
                    for req in requests
                    if req.training_capture_context is not None
                )
            )
        for step in ticket.steps:
            if step.request.training_capture_context is step.reservation:
                self._refresh(step.reservation)
        super().after_result(ticket, requests=requests)

    def _fail_request(self, req, record, reason):
        self.service.fail(record.cohort_handle, reason)
        super()._fail_request(req, record, reason)

    def disable(self, reason):
        self.service.set_admission_ready(False)
        super().disable(reason)

    def _set_admission_paused(self, paused):
        with self.lock:
            self.admission_paused = paused
            if paused:
                self.service.set_admission_ready(False)

    def control(self, action):
        super().control(action)
        if action == "abort":
            self.service.cancel_unbound("operator_aborted")

    def _handoff(self, record):
        context, metadata = record.context, None
        failure = record.invalid_reason
        try:
            if not failure and self.partition.active:
                metadata = SnapshotMetadata(
                    dataset_id=record.lease.dataset_id,
                    sample_id=record.lease.sample_id,
                    generation_id=record.lease.generation_id,
                    teacher=self.teacher,
                    sequence=context.sequence,
                    kv=self.kv,
                    provenance=record.provenance,
                    contract_id=self.config.contract_id,
                    topology=self.layout.topology,
                )
            self.writer_actor.submit(
                record.cohort_handle,
                context=context if self.partition.active else None,
                metadata=metadata,
                execution_sha256=record.execution_sha256,
                failure_reason=failure,
            )
        except Exception as error:  # noqa: BLE001 - rejected handoff retains ownership
            # submit rejected ownership, so this thread still must fence D2H.
            self.service.fail(record.cohort_handle, "writer_handoff_failed")
            complete = True
            try:
                if context is not None:
                    context.wait_for_copies()
            except Exception:  # noqa: BLE001 - quarantine failed copies
                complete = False
            self.service.finish(
                record.cohort_handle, outcome="failed", transfer_complete=complete
            )
            self._count("writer_failed_" + type(error).__name__)
            self._admission_failure("writer_error")
        with self.lock:
            record.state = "done"
            self.records.pop(record.lease.capture_id)

    def _handoff_loop(self):
        try:
            while not self.activation.wait(0.05):
                if self.stop.is_set():
                    return
            while not self.stop.is_set() or not self.work.empty():
                writer = self.writer_actor.stats(include_timings=False)
                with self.lock:
                    self.service.set_admission_ready(
                        bool(
                            not self.disabled_reason
                            and not self.admission_paused
                            and not self.stop.is_set()
                            and writer["ready"]
                            and not writer["error"]
                            and not writer["stopping"]
                            and not writer["states"].get("recovering", 0)
                        )
                    )
                try:
                    record = self.work.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._handoff(record)
                finally:
                    self.work.task_done()
        except Exception as error:  # noqa: BLE001 - never discard undrained contexts
            with self.lock:
                self.error = error
                self.disabled_reason = "handoff_failed"
                self._retained.add(self)
            logger.error("Capture handoff stopped: %s", type(error).__name__)
            self.service.set_admission_ready(False)

    def on_idle(self):
        if (
            self.work.unfinished_tasks
            or self.writer_actor.stats(include_timings=False)["pending"]
        ):
            time.sleep(0)

    def close(self):
        if self.closed:
            return True
        self.disable("producer_shutdown")
        self.stop.set()
        if self.metrics_thread is not None and self.metrics_thread.ident is not None:
            self.metrics_thread.join(timeout=5)
        if self.handoff_thread.ident is not None:
            self.handoff_thread.join(timeout=20)
        if self.handoff_thread.is_alive() or self.records:
            self._retained.add(self)
            return False
        if not self.writer_actor.close(timeout=20) or not self.service.close(
            timeout=20
        ):
            self._retained.add(self)
            return False
        try:
            self.resources.close()
            if self.owns_control_group:
                dist.destroy_process_group(self.control_group)
                self.owns_control_group = False
        except Exception:
            self._retained.add(self)
            raise
        self.closed = True
        self._retained.discard(self)
        return True
