# SPDX-License-Identifier: Apache-2.0
"""Native FULL-prefix storage writes over the real HiCache worker queues."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import msgspec
import torch

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    CopyTo,
    ExecutionUpdate,
    Location,
    MoveTo,
    Status,
    TargetOutcome,
    TokenRange,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

FULL = ComponentType.FULL


@dataclass
class _Write:
    key: bytes
    work: object
    staging: object
    interests: set[str] = field(default_factory=set)
    gpu_locks: list = field(default_factory=list)
    host_locks: list = field(default_factory=list)
    staged: bool = False
    operation_id: int | None = None
    completed: int = 0
    failure: tuple[str, str] | None = None
    failure_status: Status = Status.FAILED
    reclaimed: set[int] = field(default_factory=set)


@dataclass
class _Interest:
    work: object
    job: _Write
    cancelled: bool = False
    result: ExecutionUpdate | None = None


class StorageCacheTransfers:
    """One-prefix writes with independent interests and ACK-based retirement.

    A write uses one immutable, complete page-hash chain and one storage
    operation, so completed_tokens describes a contiguous reusable prefix even
    when ordinary radix matching splits its nodes during worker access. Neither
    cancellation nor a short ACK frees the source before worker quiescence.
    """

    def __init__(self, executor):
        self.executor, self.cache = executor, executor.cache
        self.host = executor.transfers
        self.jobs: dict[bytes, _Write] = {}
        self.interests: dict[str, _Interest] = {}

    def _local_work(self, work):
        backend = self.cache.cache_controller.storage_backend_type
        if (
            work.payload.src.tier not in ("gpu", "host")
            or work.payload.dst.tier != "storage"
            or isinstance(work.payload, MoveTo)
            and work.payload.src.tier != "gpu"
            or work.payload.dst.backend not in (None, backend)
            or work.payload.src.backend is not None
        ):
            raise ControlError(
                "unsupported", "Storage supports GPU/host COPY and GPU-to-storage MOVE"
            )
        if len(work.target.branches) != 1:
            raise ControlError(
                "unsupported", "Storage writes require one complete prefix"
            )
        if min(work.max_result_targets, self.executor.config.max_result_targets) < 2:
            raise ControlError("capacity", "Storage writes reserve two outcome ranges")
        normalized = msgspec.structs.replace(
            work,
            payload=msgspec.structs.replace(
                work.payload,
                dst=msgspec.structs.replace(work.payload.dst, backend=None),
            ),
        )
        self.host._prefixes(normalized, validate_direction=False)
        staging = msgspec.structs.replace(
            work,
            payload=CopyTo(
                target=work.payload.target,
                src=work.payload.src,
                dst=Location(tier="host" if work.payload.src.tier == "gpu" else "gpu"),
            ),
        )
        # Host-source writes don't perform H2D. This projection only reuses the
        # bounded logical path/source validator.
        self.host._paths(staging)
        return staging

    def start(self, work):
        staging = self._local_work(work)
        prefixes = self.host._prefixes(staging)
        key = hashlib.sha256(
            msgspec.json.encode(
                (work.payload.src.tier, isinstance(work.payload, MoveTo), prefixes)
            )
        ).digest()
        job = self.jobs.get(key)
        if job is None:
            job = _Write(key, work, staging)
            self.jobs[key] = job
        identity = work.ref.operation_id
        interest = _Interest(work, job)
        self.interests[identity] = interest
        job.interests.add(identity)
        if job.operation_id is None and not job.staged:
            self._stage(job)
        return self._take(interest)

    def _hold_gpu(self, job):
        if not job.gpu_locks:
            rows = self.host._paths(job.staging, source=False)[0][2]
            job.gpu_locks.append(self.cache.lock(rows[-1][0].id))

    def _stage(self, job):
        try:
            if job.work.payload.src.tier == "host":
                rows = self.host._paths(job.staging)[0][2]
                if any(n.write_through_pending_id is not None for n, _, _ in rows):
                    return
                self._hold_gpu(job)
                job.staged = True
            else:
                update = (
                    self.host.poll(job.staging.ref)
                    if job.staging.ref.operation_id in self.host.interests
                    else self.host.start(job.staging)
                )
                if update is None:
                    raise ControlError("transfer_error", "Lost host staging interest")
                if update.status in (Status.RUNNING, Status.COMPLETED):
                    self._hold_gpu(job)
                if update.status.terminal:
                    if update.status != Status.COMPLETED:
                        job.failure = (update.code or "staging_failed", update.reason)
                        job.failure_status = update.status
                        self._retire(job)
                        return
                    job.staged = True
            if job.staged:
                self._write(job)
        except Exception as error:
            job.failure = (getattr(error, "code", "storage_error"), str(error)[:512])
            if job.operation_id is None:
                job.failure_status = {
                    "missing": Status.MISSING,
                    "unsupported": Status.UNSUPPORTED,
                    "capacity": Status.DEFERRED,
                }.get(job.failure[0], Status.FAILED)
            if job.staging.ref.operation_id not in self.host.interests:
                self._retire(job)

    def _write(self, job):
        paths = self.host._paths(job.staging, source=False)
        rows = paths[0][2]
        if any(
            n.component_data[FULL].host_value is None
            or n.write_through_pending_id is not None
            for n, _, _ in rows
        ):
            raise ControlError("missing", "Host source is not fully published")
        self.host._host_hold(job, [n for n, _, _ in rows])
        hashes = [
            h for n, _, _ in rows for h in self.cache.tree_core.get_hash_values(n.id)
        ]
        tokens = paths[0][0]
        if len(hashes) * self.cache.page_size != len(tokens):
            raise ControlError("unsupported", "Incomplete source page-hash chain")
        indices = torch.cat([n.component_data[FULL].host_value for n, _, _ in rows])
        # Capture the entire logical scope now; subsequent radix splits cannot
        # change either the worker's buffers or the ACK's token geometry.
        operation_id = self.cache.cache_controller.write_storage(
            indices,
            list(tokens),
            hashes,
            [] if self.cache.hicache_storage_pass_prefix_keys else None,
            verify_prefix=True,
        )
        job.operation_id = operation_id
        self.cache._native_storage_write_results[operation_id] = None

    def begin_poll(self, *, limit=64):
        if not self.jobs:
            return
        self.cache.drain_storage_control_queues()
        for job in tuple(self.jobs.values())[:limit]:
            self.jobs.pop(job.key)
            self.jobs[job.key] = job
            if job.operation_id is None:
                if job.failure or all(
                    self.interests[i].cancelled for i in job.interests
                ):
                    update = self.host.cancel(job.staging.ref)
                    if update.quiescent:
                        if update.status not in (Status.COMPLETED, Status.CANCELLED):
                            job.failure = (
                                update.code or "staging_failed",
                                update.reason,
                            )
                            job.failure_status = update.status
                        self._retire(job)
                else:
                    self._stage(job)
                continue
            completed = self.cache._native_storage_write_results[job.operation_id]
            if completed is not None:
                end = job.work.target.branches[0].requested.end_token
                if (
                    type(completed) is not int
                    or not 0 <= completed <= end
                    or completed % self.cache.page_size
                ):
                    job.failure = (
                        "invalid_storage_ack",
                        "Worker returned invalid token coverage",
                    )
                else:
                    job.completed = completed
                    if completed != end:
                        job.failure = (
                            "incomplete_storage_write",
                            "Storage worker stopped before the full prefix was written",
                        )
                self._retire(job)

    def _outcomes(self, interest):
        job, work = interest.job, interest.work
        tokens, compatibility, rows = self.host._paths(job.staging, source=False)[0]
        boundaries = [0, job.completed, len(tokens)]
        if job.completed == len(tokens) and isinstance(work.payload, MoveTo):
            boundaries.extend(end for _, _, end in rows)
        output = []
        for start, end in zip(sorted(set(boundaries)), sorted(set(boundaries))[1:]):
            ready = end <= job.completed
            reclaimed = ready and all(
                n.id in job.reclaimed for n, a, b in rows if a < end and b > start
            )
            steps = ("destination_ready",) if ready else ()
            complete = ready
            if isinstance(work.payload, MoveTo):
                complete &= reclaimed
                if reclaimed:
                    steps += ("source_reclaimed",)
            outcome = TargetOutcome(
                coverage=TokenRange(component="full", start_token=start, end_token=end),
                status=Status.COMPLETED
                if complete
                else Status.PARTIAL
                if ready
                else Status.MISSING,
                location=Location(
                    tier="storage",
                    engine_id=self.executor.engine_id,
                    backend=self.cache.cache_controller.storage_backend_type,
                ),
                completed_steps=steps,
                code="" if complete else "source_busy" if ready else "missing_copy",
            )
            if output and (
                output[-1].status,
                output[-1].code,
                output[-1].completed_steps,
            ) == (outcome.status, outcome.code, outcome.completed_steps):
                output[-1] = msgspec.structs.replace(
                    outcome,
                    coverage=msgspec.structs.replace(
                        outcome.coverage, start_token=output[-1].coverage.start_token
                    ),
                )
            else:
                output.append(outcome)
        return tuple(
            msgspec.structs.replace(
                o,
                content_id=self.executor._content_id(
                    tokens[: o.coverage.end_token], FULL, compatibility
                ),
            )
            for o in output
        )

    def _retire(self, job):
        self.host._release_gpu(job)
        active = any(not self.interests[i].cancelled for i in job.interests)
        if (
            job.operation_id is not None
            and job.failure is None
            and active
            and isinstance(job.work.payload, MoveTo)
        ):
            rows = self.host._paths(job.staging, source=False)[0][2]
            for node, _, _ in reversed(rows):
                if (
                    node.component_data[FULL].value is not None
                    and node in self.cache.tree_core.evictable_device_leaves
                ):
                    result = self.cache.tree_core.demote(node.id)
                    self.cache._free_values(result.device_frees, result.host_frees)
                    job.reclaimed.add(node.id)
        try:
            for identity in job.interests:
                interest = self.interests[identity]
                outcomes = self._outcomes(interest) if job.staged else ()
                status = (
                    job.failure_status
                    if job.failure
                    else Status.CANCELLED
                    if interest.cancelled
                    else Status.COMPLETED
                    if outcomes and all(o.status == Status.COMPLETED for o in outcomes)
                    else Status.PARTIAL
                )
                steps = tuple(
                    s
                    for s in ("destination_ready", "source_reclaimed")
                    if outcomes and all(s in o.completed_steps for o in outcomes)
                )
                interest.result = ExecutionUpdate(
                    status=status,
                    quiescent=True,
                    code=job.failure[0]
                    if job.failure
                    else "cancelled"
                    if interest.cancelled
                    else "",
                    reason=job.failure[1] if job.failure else "",
                    completed_steps=steps,
                    target_outcomes=outcomes,
                )
        finally:
            self.host._release_host(job)
            if job.operation_id is not None:
                self.cache._native_storage_write_results.pop(job.operation_id)
            self.jobs.pop(job.key)

    def _take(self, interest):
        if interest.result is not None:
            self.interests.pop(interest.work.ref.operation_id)
            return interest.result
        nested = self.host.interests.get(interest.job.staging.ref.operation_id)
        waiting = (nested is not None and not nested.job.started) or (
            interest.job.operation_id is None and not interest.job.gpu_locks
        )
        return ExecutionUpdate(
            status=Status.CANCELLING
            if interest.cancelled or interest.job.failure
            else Status.QUEUED
            if waiting
            else Status.RUNNING,
            quiescent=waiting,
            code="draining" if interest.cancelled else "storage_pending",
        )

    def poll(self, ref):
        interest = self.interests.get(ref.operation_id)
        return self._take(interest) if interest is not None else None

    def cancel(self, ref):
        interest = self.interests.get(ref.operation_id)
        if interest is None:
            return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)
        if interest.result is not None:
            return self._take(interest)
        interest.cancelled = True
        job = interest.job
        if job.failure is None and any(
            not self.interests[i].cancelled for i in job.interests
        ):
            job.interests.remove(ref.operation_id)
            self.interests.pop(ref.operation_id)
            return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)
        if job.operation_id is None:
            update = self.host.cancel(job.staging.ref)
            if update.quiescent:
                if update.status not in (Status.COMPLETED, Status.CANCELLED):
                    job.failure = (update.code or "staging_failed", update.reason)
                    job.failure_status = update.status
                self._retire(job)
        return self._take(interest)
