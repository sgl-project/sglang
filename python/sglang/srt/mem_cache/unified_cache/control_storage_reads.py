# SPDX-License-Identifier: Apache-2.0
"""Exact native prefix restoration through asynchronous HiCache storage reads."""

from __future__ import annotations

import hashlib
import time
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
# Limit demand preference without letting a stream of arrivals starve background
# reads. Physical conflict and capacity retain precedence.
_MAX_DEMAND_BURST = 8


@dataclass
class _Read:
    key: bytes
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
    failure: tuple[str, str] | None = None
    failure_status: Status = Status.FAILED
    cancel_requested: bool = False
    waiting: str = "whole_scope_capacity"
    demands: set[CacheRequestHandle] = field(default_factory=set)
    retired: bool = False
    loaded_span: tuple[int, int | None] = (0, None)
    reserved_host_tokens: int = 0
    admission_order: int = 0
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None


@dataclass
class _Demand:
    job: _Read
    # Already resident at join time is L2, not this request's storage hit.
    storage_start: int
    joined_at: float = field(default_factory=time.monotonic)


@dataclass
class _Interest:
    work: object
    job: _Read
    cancelled: bool = False
    result: ExecutionUpdate | None = None


class StorageCacheReads:
    """Shared one-prefix restores with complete-scope approval and real fences.

    Cold host scopes pipeline across independent salt domains with bounded
    admission. A domain serializes until publication. Compatible readers
    share one interest-independent physical job.
    """

    def __init__(self, executor):
        self.executor, self.cache = executor, executor.cache
        self.host = executor.transfers
        self.jobs: dict[bytes, _Read] = {}
        self.interests: dict[str, _Interest] = {}
        self.demands: dict[CacheRequestHandle, _Demand] = {}
        self._joinable: dict[tuple, _Read] = {}
        self._host_reclaim_retry_at = 0.0
        self._next_admission_order = 0
        self._demand_admissions = 0
        for name in (
            "max_storage_read_inflight_tokens",
            "max_speculative_storage_read_tokens",
            "demand_host_reserve_tokens",
        ):
            if getattr(executor.config, name) % self.cache.page_size:
                raise ValueError(f"{name} must be page aligned")
        if (
            executor.config.demand_host_reserve_tokens
            >= self.cache.cache_controller.mem_pool_host.size
        ):
            raise ValueError("demand_host_reserve_tokens must be below host capacity")
        self.stats = dict(
            demand_coalesced=0,
            demand_consumed=0,
            demand_aborted=0,
            demand_limit_declined=0,
            jobs_created=0,
            interests_coalesced=0,
            jobs_started=0,
            jobs_completed=0,
            jobs_failed=0,
            jobs_deferred=0,
            jobs_cancelled=0,
            requested_tokens=0,
            loaded_tokens=0,
            queue_seconds=0.0,
            restore_seconds=0.0,
            demand_wait_seconds=0.0,
        )

    def snapshot(self):
        """Owner-local counters and gauges; jobs, not logical interests, cost I/O."""
        stats = dict(self.stats)
        active = [job for job in self.jobs.values() if job.operation is not None]
        stats.update(
            active_jobs=len(active),
            queued_jobs=sum(job.operation is None for job in self.jobs.values()),
            demand_waiters=len(self.demands),
            reserved_host_tokens=self._unallocated_reservations(None),
            inflight_tokens=sum(job.reserved_host_tokens for job in active),
            speculative_inflight_tokens=sum(
                job.reserved_host_tokens for job in active if not job.demands
            ),
        )
        pool = self.cache.cache_controller.mem_pool_host
        stats.update(
            host_capacity_tokens=pool.size,
            host_available_tokens=pool.available_size(),
            host_bytes_per_token=pool.size_per_token,
        )
        for reason in (
            "whole_scope_capacity",
            "storage_concurrency_limit",
            "shared_prefix_serialization",
            "storage_rate_limit",
            "source_pending",
            "storage_inflight_budget",
            "speculative_read_budget",
            "demand_priority",
        ):
            stats["waiting_" + reason] = sum(
                job.operation is None and job.waiting == reason
                for job in self.jobs.values()
            )
        return stats

    @staticmethod
    def _demand_key(tokens, extra_key, cache_salt):
        return tokens, extra_key, cache_salt

    def join_demand(self, request, prefix, suffix, extra_key, cache_salt):
        if request in self.demands:
            return True
        # Other stop policies retain their existing deadline/early-compute behavior.
        if self.cache.prefetch_stop_policy != "wait_complete":
            return False
        if len(self.demands) >= self.executor.config.max_prepared_prefixes:
            self.stats["demand_limit_declined"] += 1
            return False
        if (
            len(prefix or ()) + len(suffix)
            > self.cache.kv_control_service.config.limits.max_scope_tokens
        ):
            return False
        tokens = tuple(prefix or ()) + tuple(suffix.token_ids)
        job = self._joinable.get(self._demand_key(tokens, extra_key, cache_salt))
        if job is None or job.cancel_requested or job.failure or job.retired:
            return False
        if job.operation is None and not self._has_live_interest(job):
            return False
        self.demands[request] = _Demand(
            job, max(len(prefix or ()), self._host_length(job))
        )
        job.demands.add(request)
        self.stats["demand_coalesced"] += 1
        return True

    def demand_ready(self, request):
        """Admission may proceed after drain, including partial/missing results."""
        demand = self.demands.get(request)
        return None if demand is None else demand.job.retired

    def release_demand(self, request, *, aborted=False):
        demand = self.demands.pop(request, None)
        if demand is None:
            return None
        job = demand.job
        job.demands.remove(request)
        self.stats["demand_wait_seconds"] += time.monotonic() - demand.joined_at
        self.stats["demand_aborted" if aborted else "demand_consumed"] += 1
        count, start = job.loaded_span
        left = max(start or 0, demand.storage_start)
        right = (start or 0) + count
        span = (right - left, left) if right > left else (0, None)
        if job.retired and not job.demands:
            self._release(job)
        return span

    def _scope(self, work):
        payload = work.payload
        if (
            not isinstance(payload, CopyTo)
            or (payload.src.tier, payload.dst.tier) != ("storage", "host")
            or (payload.src.tier, payload.dst.tier)
            not in self.executor.storage_adapter.copy_paths
            or payload.src.backend not in (None, self.executor.storage_adapter.backend)
            or payload.dst.backend is not None
            or any(
                p.engine_id not in (None, self.executor.engine_id)
                for p in (payload.src, payload.dst)
            )
        ):
            raise ControlError(
                "unsupported", "Storage prefetch requires a local host destination"
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

    def _host_length(self, job):
        length = 0
        for node, _, end in self._rows(job):
            if (
                node.component_data[FULL].host_value is None
                or node.write_through_pending_id is not None
            ):
                break
            length = end
        return length

    def start(self, work):
        tokens, compatibility = self._scope(work)
        key = hashlib.sha256(msgspec.json.encode((tokens, compatibility))).digest()
        job = self.jobs.get(key)
        if job is None:
            job = _Read(
                key,
                tokens,
                compatibility,
                admission_order=self._next_admission_order,
            )
            self._next_admission_order += 1
            self._rows(job)
            self.jobs[key] = job
            self.stats["jobs_created"] += 1
        else:
            self.stats["interests_coalesced"] += 1
        interest = _Interest(work, job)
        self.interests[work.ref.operation_id] = interest
        job.interests.add(work.ref.operation_id)
        # Demand may join a queued job before any I/O is issued.
        self._joinable[
            self._demand_key(
                job.tokens, job.compatibility.extra_key, job.compatibility.cache_salt
            )
        ] = job
        if job.operation is None:
            self._start(job)
        return self._take(interest)

    def _start(self, job):
        try:
            pool = self.cache.cache_controller.mem_pool_host
            if len(job.tokens) > pool.size:
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
            host_length = self._host_length(job)
            if host_length == len(job.tokens):
                self._retire(job)
                return
            waiting = self._admission_wait(job)
            if waiting is not None:
                job.waiting = waiting
                self._release(job)
                return
            if self.cache.cache_controller.prefetch_rate_limited():
                job.waiting = "storage_rate_limit"
                self._release(job)
                return
            needed = len(job.tokens) - host_length
            budget = self._read_budget(job, needed)
            if budget is not None:
                if budget.startswith("scope_exceeds_"):
                    self._retire(job, status=Status.DEFERRED, code=budget)
                else:
                    job.waiting = budget
                    self._release(job)
                return
            if not self._host_capacity(job, needed):
                job.waiting = "whole_scope_capacity"
                if job.demands:
                    # Ordinary demand must be able to continue computation
                    # when PIN/host pressure prevents a complete cold restore.
                    self._retire(job, status=Status.DEFERRED, code=job.waiting)
                else:
                    self._release(job)
                return
            # The query has not allocated rows yet. Charge its complete missing
            # span until the owner observes real allocation or quiescence.
            job.reserved_host_tokens = len(job.tokens) - host_length
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
                job.reserved_host_tokens = 0
                job.waiting = "storage_rate_limit"
                self._release(job)
                return
            job.operation = info.operation
            job.started_at = time.monotonic()
            self.stats["jobs_started"] += 1
            self.stats["queue_seconds"] += job.started_at - job.created_at
            self.stats["requested_tokens"] += job.reserved_host_tokens
            if self.executor.config.storage_demand_priority:
                self._demand_admissions = (
                    min(_MAX_DEMAND_BURST, self._demand_admissions + 1)
                    if job.demands
                    else 0
                )
            job.operation.native_quiescent = False
            job.operation.native_on_publish = lambda: self._published(job)
        except Exception as error:
            job.failure = (getattr(error, "code", "storage_error"), str(error)[:512])
            if job.operation is None:
                self._retire(job)

    def _read_budget(self, job, needed):
        """Gate new native work; ordinary recovery keeps its existing fallback.

        Controller occupancy includes ordinary and native query/GET allocations.
        Native speculative spans are charged once per physical in-flight job;
        completed residency is managed by Router policy and ordinary eviction.
        """
        config = self.executor.config
        limit = config.max_storage_read_inflight_tokens
        if limit:
            if needed > limit:
                return "scope_exceeds_read_budget"
            if self.cache.cache_controller.prefetch_tokens_occupied + needed > limit:
                return "storage_inflight_budget"
        if not job.demands:
            limit = config.max_speculative_storage_read_tokens
            if limit:
                if needed > limit:
                    return "scope_exceeds_speculative_budget"
                occupied = sum(
                    other.reserved_host_tokens
                    for other in self.jobs.values()
                    if other is not job
                    and other.operation is not None
                    and not other.demands
                )
                if occupied + needed > limit:
                    return "speculative_read_budget"
            if (
                needed + config.demand_host_reserve_tokens
                > self.cache.cache_controller.mem_pool_host.size
            ):
                return "scope_exceeds_speculative_capacity"
        return None

    def _admission_wait(self, job):
        active = [
            other
            for other in self.jobs.values()
            if other is not job and other.operation is not None
        ]
        if len(active) >= self.executor.config.max_concurrent_storage_reads:
            return "storage_concurrency_limit"
        if any(
            job.compatibility.cache_salt == other.compatibility.cache_salt
            for other in active
        ):
            return "shared_prefix_serialization"
        if (
            self.executor.config.storage_demand_priority
            and not job.demands
            and self._demand_admissions < _MAX_DEMAND_BURST
        ):
            if any(
                other is not job
                and other.demands
                and other.operation is None
                and self._has_live_interest(other)
                for other in self.jobs.values()
            ):
                return "demand_priority"
        return None

    def _has_live_interest(self, job):
        if job.demands:
            return True
        control = self.cache.kv_control_service.control
        now = control._clock()
        return any(
            not self.interests[identity].cancelled
            and (
                (pending := control._pending.get(identity)) is None
                or pending.deadline is None
                or now < pending.deadline
            )
            for identity in job.interests
        )

    def _unallocated_reservations(self, job):
        # Allocated scratch is already absent from pool.available_size(). A
        # cancelled query keeps its promise until its owner-observed drain.
        return sum(
            other.reserved_host_tokens
            for other in self.jobs.values()
            if other is not job
            and other.operation is not None
            and other.operation.host_indices is None
            and not getattr(other.operation, "native_quiescent", False)
        )

    def _host_capacity(self, job, needed):
        """Approve the full missing span, reclaiming only its current shortfall.

        The caller holds the requested existing prefix and has checked device
        and transfer budgets. Native eviction honors PIN/transfer references;
        whole radix nodes may make its actual reclamation exceed the shortfall.
        """
        pool = self.cache.cache_controller.mem_pool_host
        reserved = self._unallocated_reservations(job)
        # Capacity and outstanding promises use token slots exactly once.
        # The reserve is an admission floor, never a physical allocation.
        if not job.demands:
            needed += self.executor.config.demand_host_reserve_tokens
        available = pool.available_size() - reserved
        if needed <= available:
            return True
        control = self.cache.kv_control_service.control
        now = control._clock()
        if now < self._host_reclaim_retry_at:
            return False
        # begin_poll precedes controller deadline handling. Do not introduce
        # host eviction for an expired caller; a live coalesced sibling or
        # ordinary demand still authorizes the shared physical preparation.
        if not self._has_live_interest(job):
            return False
        self.cache.evict_host(needed - available)
        if needed + reserved > pool.available_size():
            # Shared across jobs on this host pool: pinned pressure must not
            # cause a full eviction walk on every owner idle poll/joiner.
            self._host_reclaim_retry_at = now + 0.1
            return False
        self._host_reclaim_retry_at = 0.0
        return True

    def _observe_cold_result(self, job):
        if (
            job.failure is None
            and not job.cancel_requested
            and self._host_length(job) != len(job.tokens)
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
        jobs = tuple(self.jobs.values())
        if self.executor.config.storage_demand_priority:
            active = [job for job in jobs if job.operation is not None]
            queued = [job for job in jobs if job.operation is None]
            demand = sorted(
                (job for job in queued if job.demands),
                key=lambda job: job.admission_order,
            )
            # Keep rotating background candidates: a bounded poll must not keep
            # retrying the same blocked scopes while feasible work waits later.
            background = [job for job in queued if not job.demands]
            if self._demand_admissions >= _MAX_DEMAND_BURST:
                # One background candidate gets a turn; a blocked candidate
                # must not hide runnable demand behind a full background queue.
                jobs = active + background[:1] + demand + background[1:]
            else:
                jobs = active + demand + background
        for job in jobs[:limit]:
            self.jobs.pop(job.key)
            self.jobs[job.key] = job
            cancelled = not job.demands and all(
                self.interests[i].cancelled for i in job.interests
            )
            if cancelled and job.operation is not None and not job.cancel_requested:
                job.cancel_requested = True
                self.cache.cache_controller.terminate_prefetch(job.operation)
            if job.operation is None:
                if cancelled:
                    self._retire(job)
                else:
                    self._start(job)
            elif getattr(job.operation, "native_quiescent", False):
                self._observe_cold_result(job)
                self._retire(job)

    def _retire(self, job, *, status=None, code=""):
        length = 0
        try:
            if job.operation is not None and getattr(
                job.operation, "native_quiescent", False
            ):
                # Final ACK publication may have queued unused scratch pages
                # after the previous drain's release-count snapshot.
                self.cache.drain_storage_control_queues()
            length = self._host_length(job)
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
                            tier="host",
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
            job.reserved_host_tokens = 0
            job.loaded_span = self.cache.pop_prefetch_loaded_span(job.handle)
            self.stats["loaded_tokens"] += job.loaded_span[0]
            if job.started_at is not None:
                self.stats["restore_seconds"] += time.monotonic() - job.started_at
            outcome = (
                "failed"
                if job.failure
                else "deferred"
                if status == Status.DEFERRED
                else "cancelled"
                if job.cancel_requested
                or all(self.interests[i].cancelled for i in job.interests)
                else "completed"
                if length == len(job.tokens)
                else "failed"
            )
            self.stats["jobs_" + outcome] += 1
            job.retired = True
            key = self._demand_key(
                job.tokens, job.compatibility.extra_key, job.compatibility.cache_salt
            )
            if self._joinable.get(key) is job:
                self._joinable.pop(key)
            # Demand consumers hold the published prefix until their next
            # admission pass (or abort). pop_prefetch_loaded_span hands off to
            # ordinary match/load-back, just like a completed ordinary restore.
            if not job.demands:
                self._release(job)
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
        waiting = job.operation is None
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
        if job.demands:
            # The ordinary request now owns demand for this physical operation.
            # Keep the cancelled native receipt draining; never terminate its I/O.
            return self._take(interest)
        if job.failure is None and any(
            not self.interests[i].cancelled for i in job.interests
        ):
            job.interests.remove(ref.operation_id)
            self.interests.pop(ref.operation_id)
            return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)
        if job.operation is not None:
            if getattr(job.operation, "native_quiescent", False):
                self._observe_cold_result(job)
                self._retire(job)
            else:
                job.cancel_requested = True
                self.cache.cache_controller.terminate_prefetch(job.operation)
        else:
            self._retire(job)
        return self._take(interest)
