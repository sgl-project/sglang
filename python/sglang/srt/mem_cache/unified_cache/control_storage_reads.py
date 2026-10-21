# SPDX-License-Identifier: Apache-2.0
"""Exact native prefix restoration through asynchronous HiCache storage reads."""

from __future__ import annotations

import hashlib
import uuid
from array import array
from dataclasses import dataclass, field

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    CopyTo,
    ExecutionUpdate,
    Location,
    Status,
    TargetOutcome,
    TokenRange,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

FULL = ComponentType.FULL


@dataclass
class _Read:
    key: bytes
    work: object
    tokens: tuple
    compatibility: object
    handle: CacheRequestHandle = field(
        default_factory=lambda: CacheRequestHandle("native-kv:" + uuid.uuid4().hex, 0)
    )
    interests: set[str] = field(default_factory=set)
    gpu_locks: list = field(default_factory=list)
    host_locks: list = field(default_factory=list)
    anchor: int | None = None
    operation: object = None
    loading: bool = False
    failure: tuple[str, str] | None = None
    failure_status: Status = Status.FAILED
    cancel_requested: bool = False
    waiting: str = "whole_scope_capacity"


@dataclass
class _Interest:
    work: object
    job: _Read
    cancelled: bool = False
    result: ExecutionUpdate | None = None


class StorageCacheReads:
    """Shared one-prefix restores with complete-scope approval and real fences.

    Different scopes serialize their cold stage and recheck current host hits
    before starting. An existing prefix is counted once, and compatible readers
    of the same destination share one interest-independent physical job.
    """

    def __init__(self, executor):
        self.executor, self.cache = executor, executor.cache
        self.host = executor.transfers
        self.jobs: dict[bytes, _Read] = {}
        self.interests: dict[str, _Interest] = {}

    def _scope(self, work):
        payload = work.payload
        if (
            not isinstance(payload, CopyTo)
            or payload.src.tier != "storage"
            or payload.dst.tier not in ("gpu", "host")
            or payload.src.backend not in (None, "file")
            or payload.dst.backend is not None
            or any(
                p.engine_id not in (None, self.executor.engine_id)
                for p in (payload.src, payload.dst)
            )
        ):
            raise ControlError(
                "unsupported", "File storage reads support local COPY to host/GPU"
            )
        if len(work.target.branches) != 1:
            raise ControlError(
                "unsupported", "Storage reads require one complete prefix"
            )
        if min(work.max_result_targets, self.executor.config.max_result_targets) < 2:
            raise ControlError("capacity", "Storage reads reserve two outcome ranges")
        projection = msgspec.structs.replace(
            work,
            payload=CopyTo(
                target=payload.target,
                src=Location(tier="gpu"),
                dst=Location(tier="host"),
            ),
        )
        return self.host._prefixes(projection)[0]

    def _rows(self, job):
        path = self.executor._walk(job.tokens, job.compatibility)
        if any(end > len(job.tokens) or matched != end for _, _, end, matched in path):
            raise ControlError(
                "unsupported", "Resident prefix must end at an existing radix boundary"
            )
        return [(node, start, end) for node, start, end, _ in path]

    def _hold(self, job):
        rows = self._rows(job)
        if rows and rows[-1][0].id != job.anchor:
            new = self.cache.lock(rows[-1][0].id)
            self.host._release_gpu(job)
            job.gpu_locks.append(new)
            job.anchor = rows[-1][0].id
        self.host._host_hold(job, [n for n, _, _ in rows])

    def _published(self, job):
        # Called on the owner thread inside actual host publication, before
        # ordinary admissions/eviction can consume the newly inserted prefix.
        try:
            self._hold(job)
        except Exception as error:
            job.failure = ("publication_error", str(error)[:512])

    def _release(self, job):
        self.host._release_host(job)
        self.host._release_gpu(job)
        job.anchor = None

    def _length(self, job, tier):
        length = 0
        for node, _, end in self._rows(job):
            cd = node.component_data[FULL]
            ready = (
                cd.host_value is not None and node.write_through_pending_id is None
                if tier == "host"
                else cd.value is not None and not self.host._load_pending(node)
            )
            if not ready:
                break
            length = end
        return length

    def start(self, work):
        tokens, compatibility = self._scope(work)
        key = hashlib.sha256(
            msgspec.json.encode((tokens, compatibility, work.payload.dst.tier))
        ).digest()
        job = self.jobs.get(key)
        if job is None:
            job = _Read(key, work, tokens, compatibility)
            self._rows(job)
            self.jobs[key] = job
        interest = _Interest(work, job)
        self.interests[work.ref.operation_id] = interest
        job.interests.add(work.ref.operation_id)
        if job.operation is None and not job.loading:
            self._start(job)
        return self._take(interest)

    def _start(self, job):
        try:
            if any(
                other is not job and (other.operation is not None or other.loading)
                for other in self.jobs.values()
            ):
                job.waiting = "shared_prefix_serialization"
                return
            pool = self.cache.cache_controller.mem_pool_host
            if len(job.tokens) > pool.size or (
                job.work.payload.dst.tier == "gpu"
                and len(job.tokens) > self.cache.token_to_kv_pool_allocator.size
            ):
                self._retire(
                    job,
                    status=Status.DEFERRED,
                    code="scope_exceeds_destination_capacity",
                )
                return
            rows = self._rows(job)
            if any(
                n.write_through_pending_id is not None or self.host._load_pending(n)
                for n, _, _ in rows
            ):
                job.waiting = "source_pending"
                return
            self._hold(job)
            host_length = self._length(job, "host")
            gpu_length = self._length(job, "gpu")
            if self._length(job, job.work.payload.dst.tier) == len(job.tokens):
                self._retire(job)
                return
            if len(job.tokens) - host_length > pool.available_size() or (
                job.work.payload.dst.tier == "gpu"
                and len(job.tokens) - gpu_length > self.host._gpu_available()
            ):
                self._release(job)
                return
            if host_length == len(job.tokens):
                self._load_gpu(job)
                return
            if self.cache.cache_controller.prefetch_rate_limited():
                job.waiting = "storage_rate_limit"
                self._release(job)
                return
            anchor = next(
                (n for n, _, end in rows if end == host_length),
                self.cache.tree_core.root_node,
            )
            self.cache.prefetch_from_storage(
                job.handle,
                anchor.id,
                array("q", job.tokens[host_length:]),
                last_hash=self.cache.get_last_hash_value(anchor.id),
                prefix_keys=self.cache.get_prefix_hash_values(anchor.id)
                + self.cache.tree_core.get_hash_values(anchor.id)
                if self.cache.hicache_storage_pass_prefix_keys
                else None,
                matched_prefix_tokens=list(job.tokens[:host_length]),
                cache_salt=job.compatibility.cache_salt,
                minimum_prefetch_tokens=self.cache.page_size,
            )
            info = self.cache.ongoing_prefetch.get(job.handle)
            if info is None:
                job.waiting = "storage_rate_limit"
                self._release(job)
                return
            job.operation = info.operation
            job.operation.native_quiescent = False
            job.operation.native_on_publish = lambda: self._published(job)
        except Exception as error:
            job.failure = (getattr(error, "code", "storage_error"), str(error)[:512])
            if job.operation is None and not job.loading:
                self._retire(job)

    def _load_work(self, job):
        return msgspec.structs.replace(
            job.work,
            payload=CopyTo(
                target=job.work.payload.target,
                src=Location(tier="host"),
                dst=Location(tier="gpu"),
            ),
        )

    def _load_gpu(self, job):
        work = self._load_work(job)
        result = self.host.start(work)
        job.loading = not result.status.terminal
        if result.status.terminal:
            if result.status != Status.COMPLETED:
                job.failure = (result.code or "load_failed", result.reason)
            self._retire(job)

    def _observe_cold_result(self, job):
        if (
            job.failure is None
            and not job.cancel_requested
            and self._length(job, "host") != len(job.tokens)
        ):
            missing = job.operation.storage_hit_count == 0
            job.failure_status = Status.MISSING if missing else Status.FAILED
            job.failure = (
                "missing_copy" if missing else "incomplete_storage_read",
                "Storage did not publish the complete requested prefix",
            )

    def begin_poll(self, *, limit=64):
        if not self.jobs:
            return
        self.cache.drain_storage_control_queues()
        for job in tuple(self.jobs.values())[:limit]:
            self.jobs.pop(job.key)
            self.jobs[job.key] = job
            cancelled = all(self.interests[i].cancelled for i in job.interests)
            if job.loading:
                result = (
                    self.host.cancel(job.work.ref)
                    if cancelled or job.failure
                    else self.host.poll(job.work.ref)
                )
                if result is not None and result.status.terminal:
                    job.loading = False
                    if result.status not in (Status.COMPLETED, Status.CANCELLED):
                        job.failure = (result.code or "load_failed", result.reason)
                    self._retire(job)
            elif job.operation is None:
                if cancelled:
                    self._retire(job)
                else:
                    self._start(job)
            elif getattr(job.operation, "native_quiescent", False):
                self._observe_cold_result(job)
                if cancelled or job.failure:
                    self._retire(job)
                elif job.work.payload.dst.tier == "gpu":
                    self._load_gpu(job)
                else:
                    self._retire(job)

    def _retire(self, job, *, status=None, code=""):
        try:
            if job.operation is not None and getattr(
                job.operation, "native_quiescent", False
            ):
                # Final ACK publication may have queued unused scratch pages
                # after the previous drain's release-count snapshot.
                self.cache.drain_storage_control_queues()
            length = self._length(job, job.work.payload.dst.tier)
            if (
                length == len(job.tokens)
                and job.work.payload.dst.tier == "gpu"
                and job.failure is None
                and status in (None, Status.COMPLETED)
            ):
                for identity in job.interests:
                    interest = self.interests[identity]
                    if not interest.cancelled:
                        self.executor._capture_preparation(interest.work)
            outcomes = []
            for start, end in ((0, length), (length, len(job.tokens))):
                if start == end:
                    continue
                ready = end <= length
                outcomes.append(
                    TargetOutcome(
                        coverage=TokenRange(
                            component="full", start_token=start, end_token=end
                        ),
                        status=Status.COMPLETED if ready else Status.MISSING,
                        completed_steps=("destination_ready",) if ready else (),
                        location=Location(
                            tier=job.work.payload.dst.tier,
                            engine_id=self.executor.engine_id,
                        ),
                        code="" if ready else "missing_copy",
                        content_id=self.executor._content_id(
                            job.tokens[:end], FULL, job.compatibility
                        ),
                    )
                )
            for identity in job.interests:
                interest = self.interests[identity]
                terminal = (
                    job.failure_status
                    if job.failure
                    else Status.CANCELLED
                    if interest.cancelled
                    else status
                    or (
                        Status.COMPLETED
                        if length == len(job.tokens)
                        else Status.PARTIAL
                    )
                )
                interest.result = ExecutionUpdate(
                    status=terminal,
                    quiescent=True,
                    code=job.failure[0]
                    if job.failure
                    else "cancelled"
                    if interest.cancelled
                    else code,
                    reason=job.failure[1] if job.failure else "",
                    completed_steps=("destination_ready",)
                    if length == len(job.tokens)
                    else (),
                    target_outcomes=tuple(outcomes),
                )
        finally:
            self._release(job)
            self.cache.pop_prefetch_loaded_span(job.handle)
            self.cache.discard_storage_prefetch_accounting(job.handle)
            self.cache.storage_prefetch_retries.cancel(job.handle.rid)
            self.cache._storage_prefetch_wait.pop(job.handle, None)
            if job.operation is not None:
                job.operation.native_on_publish = None
            self.jobs.pop(job.key)

    def _take(self, interest):
        if interest.result is not None:
            self.interests.pop(interest.work.ref.operation_id)
            return interest.result
        job = interest.job
        waiting = job.operation is None and not job.loading
        return ExecutionUpdate(
            status=Status.CANCELLING
            if interest.cancelled or job.failure
            else Status.QUEUED
            if waiting
            else Status.RUNNING,
            quiescent=waiting,
            code="draining"
            if interest.cancelled
            else job.waiting
            if waiting
            else "storage_pending",
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
        if job.loading:
            update = self.host.cancel(job.work.ref)
            if update.quiescent:
                job.loading = False
                if update.status not in (Status.COMPLETED, Status.CANCELLED):
                    job.failure = (update.code or "load_failed", update.reason)
                self._retire(job)
        elif job.operation is not None:
            if getattr(job.operation, "native_quiescent", False):
                self._observe_cold_result(job)
                self._retire(job)
            else:
                job.cancel_requested = True
                self.cache.cache_controller.terminate_prefetch(job.operation)
        else:
            self._retire(job)
        return self._take(interest)
