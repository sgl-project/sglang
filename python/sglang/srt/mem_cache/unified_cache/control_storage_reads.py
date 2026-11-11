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
    CacheCompatibility,
    ControlError,
    CopyTo,
    ExecutionUpdate,
    Location,
    Status,
    TargetOutcome,
    TokenRange,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, DecLockRefParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

FULL = ComponentType.FULL


@dataclass
class _Read:
    key: bytes | CacheRequestHandle
    tokens: tuple
    compatibility: object
    handle: CacheRequestHandle = field(
        default_factory=lambda: CacheRequestHandle("native-kv:" + uuid.uuid4().hex, 0)
    )
    interests: set[str] = field(default_factory=set)
    host_locks: list = field(default_factory=list)
    operation: object = None
    failure: tuple[str, str] | None = None
    failure_status: Status = Status.FAILED
    cancel_requested: bool = False
    waiting: str = "whole_scope_capacity"
    demands: set[CacheRequestHandle] = field(default_factory=set)
    retired: bool = False
    loaded_span: tuple[int, int | None] = (0, None)
    published_host_node: int | None = None
    reserved_host_tokens: int = 0
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None

    @property
    def borrowed(self) -> bool:
        return isinstance(self.key, CacheRequestHandle)


@dataclass
class _Demand:
    job: _Read
    # Already resident at join time is L2, not this request's storage hit.
    storage_start: int
    joined_at: float = field(default_factory=time.monotonic)
    resume_tokens: tuple[int, ...] | None = None
    tail_started: bool = False


@dataclass
class _AdmissionHold:
    receipt: DecLockRefParams
    users: int = 0


@dataclass
class _Interest:
    work: object
    job: _Read
    cancelled: bool = False
    result: ExecutionUpdate | None = None


class StorageCacheReads:
    """Shared one-prefix restores with complete-scope approval and real fences.

    Cold host scopes pipeline across independent radix branches with bounded
    admission. Reads sharing a radix root page serialize until publication.
    Compatible readers share one interest-independent physical job.
    """

    def __init__(self, executor):
        self.executor, self.cache = executor, executor.cache
        self.jobs: dict[bytes | CacheRequestHandle, _Read] = {}
        self.interests: dict[str, _Interest] = {}
        self.demands: dict[CacheRequestHandle, _Demand] = {}
        self._joinable: dict[tuple, _Read] = {}
        self._admission_holds: dict[CacheRequestHandle, tuple[int, int]] = {}
        self._admission_scopes: dict[tuple[int, int], _AdmissionHold] = {}
        self._admission_tokens = 0
        self._host_reclaim_retry_at = 0.0
        self.stats = dict(
            demand_coalesced=0,
            demand_prefix_coalesced=0,
            demand_consumed=0,
            demand_aborted=0,
            demand_limit_declined=0,
            ordinary_reads_reused=0,
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
            admission_held_tokens=self._admission_tokens,
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
        ):
            stats["waiting_" + reason] = sum(
                job.operation is None and job.waiting == reason
                for job in self.jobs.values()
            )
        return stats

    @staticmethod
    def _demand_key(tokens, extra_key, cache_salt):
        return tokens, extra_key, cache_salt

    def hold_for_admission(self, request, node_id, end_token):
        if (
            node_id is None
            or self.cache.is_write_back
            or self.cache.tree_core.component_types != (FULL,)
        ):
            return
        old = self._admission_holds.get(request)
        previous_end = old[1] if old is not None else 0
        if end_token <= previous_end:
            return
        scope = (node_id, end_token)
        hold = self._admission_scopes.get(scope)
        released = previous_end if old and self._admission_scopes[old].users == 1 else 0
        # Cache mode limits in-flight prefetch to half the pool. Leave
        # another quarter available for ordinary write-through and eviction.
        budget = self.cache.cache_controller.prefetch_capacity_limit // 2
        if hold is None:
            if self._admission_tokens - released + end_token > budget:
                return
            hold = _AdmissionHold(self.cache.inc_host_lock_ref(node_id).to_dec_params())
            self._admission_scopes[scope] = hold
            self._admission_tokens += end_token
        self.release_admission(request)
        hold.users += 1
        self._admission_holds[request] = scope

    def release_admission(self, request):
        scope = self._admission_holds.pop(request, None)
        if scope is None:
            return
        hold = self._admission_scopes[scope]
        hold.users -= 1
        if not hold.users:
            node_id, end_token = scope
            self.cache.dec_host_lock_ref(node_id, hold.receipt)
            del self._admission_scopes[scope]
            self._admission_tokens -= end_token

    def clear_admission_holds(self):
        for request in tuple(self._admission_holds):
            self.release_admission(request)

    def join_demand(self, request, prefix, suffix, extra_key, cache_salt):
        if request in self.demands:
            return not self.demands[request].tail_started
        # Other stop policies retain their existing deadline/early-compute behavior.
        if self.cache.prefetch_stop_policy != "wait_complete":
            return False
        if len(self.demands) >= self.executor.config.max_prepared_prefixes:
            self.stats["demand_limit_declined"] += 1
            return False
        prefix_length = len(prefix or ())
        if (
            prefix_length + len(suffix)
            > self.cache.kv_control_service.config.limits.max_scope_tokens
        ):
            return False
        tokens = tuple(prefix or ()) + tuple(suffix.token_ids)
        key = self._demand_key(tokens, extra_key, cache_salt)
        job = self._joinable.get(key)
        resume_tokens = None
        if job is None:
            # Join only submitted reads. A speculative ancestor still waiting
            # for admission must not block an ordinary request.
            job = max(
                (
                    candidate
                    for candidate in self._joinable.values()
                    if candidate.operation is not None
                    and not candidate.cancel_requested
                    and not candidate.failure
                    and not candidate.retired
                    and candidate.compatibility.extra_key == extra_key
                    and candidate.compatibility.cache_salt == cache_salt
                    and prefix_length < len(candidate.tokens) < len(tokens)
                    and tokens[: len(candidate.tokens)] == candidate.tokens
                ),
                key=lambda candidate: len(candidate.tokens),
                default=None,
            )
            if job is not None:
                resume_tokens = tokens
        if (
            job is None
            and len(self.demands) + 1 < self.executor.config.max_prepared_prefixes
        ):
            # Borrowing adds the existing reader as well as this new demand.
            job = self._borrow_ordinary(
                tokens,
                CacheCompatibility(extra_key=extra_key, cache_salt=cache_salt),
                exact=True,
            )
            if job is not None:
                self._joinable[key] = job
        if job is None or job.cancel_requested or job.failure or job.retired:
            return False
        if job.operation is None:
            if not self._has_live_interest(job):
                return False
            if (
                self._admission_wait(job) == "storage_concurrency_limit"
                or self.cache.cache_controller.prefetch_rate_limited()
            ):
                # The speculative queue has not issued I/O for this scope.
                # Preserve ordinary admission (including the rate-limit
                # fallback) and retire the hint so it cannot issue a late GET.
                self._retire(job, status=Status.DEFERRED, code="ordinary_restore")
                return False
        self.demands[request] = _Demand(
            job, max(prefix_length, self._host_length(job)), resume_tokens=resume_tokens
        )
        job.demands.add(request)
        self.stats["demand_coalesced"] += 1
        if resume_tokens is not None:
            self.stats["demand_prefix_coalesced"] += 1
        return True

    def demand_ready(self, request):
        """Admission may proceed after drain, including partial/missing results."""
        demand = self.demands.get(request)
        return None if demand is None else demand.job.retired

    def check_demand_progress(self, request):
        demand = self.demands.get(request)
        if demand is None:
            return None
        if not demand.job.retired:
            return False
        if demand.resume_tokens is None:
            return True
        if not demand.tail_started:
            demand.tail_started = True
            tokens, compatibility = demand.resume_tokens, demand.job.compatibility
            anchor, length = self.cache.tree_core.root_node, 0
            for node, _, end, matched in self.executor._walk(tokens, compatibility):
                if not self._host_ready(node):
                    break
                if matched == len(tokens):
                    return None
                if matched != end:
                    break
                anchor, length = node, end
            self.cache.prefetch_from_storage(
                request,
                anchor.id,
                array("q", tokens[length:]),
                last_hash=self.cache.get_last_hash_value(anchor.id),
                prefix_keys=self.cache.get_prefix_hash_values(anchor.id)
                if self.cache.hicache_storage_pass_prefix_keys
                else None,
                matched_prefix_tokens=list(tokens[:length]),
                extra_key=compatibility.extra_key,
                cache_salt=compatibility.cache_salt,
            )
        # The ordinary progress path owns the suffix operation, if any.
        return None

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
        if not aborted and span[0]:
            # The scheduler reads this span before it can commit admission.
            # Keep just the bounded Host hold when releasing the shared job.
            self.hold_for_admission(request, job.published_host_node, right)
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
        if min(work.max_result_targets, self.executor.config.max_result_targets) < 2:
            raise ControlError("capacity", "Storage reads reserve two outcome ranges")
        return self.executor._storage_prefix(work)

    def _rows(self, job):
        path = self.executor._walk(job.tokens, job.compatibility)
        return [(node, start, matched) for node, start, _, matched in path]

    def _hold(self, job, rows):
        # Host locks keep the restore path alive without blocking device eviction.
        self.executor._hold_host(job.host_locks, [n for n, _, _ in rows])

    def _published(self, job, inserted_host_node):
        # Called on the owner thread inside actual host publication, before
        # ordinary admissions/eviction can consume the newly inserted prefix.
        try:
            self._hold(job, self._rows(job))
            job.published_host_node = inserted_host_node
        except Exception as error:
            job.failure = ("publication_error", str(error)[:512])

    def _release(self, job):
        self.executor._release_host(job.host_locks)

    def _host_length(self, job, rows=None):
        length = 0
        for node, _, end in self._rows(job) if rows is None else rows:
            if not self._host_ready(node):
                break
            length = end
        return length

    @staticmethod
    def _host_ready(node):
        return (
            node.component_data[FULL].host_value is not None
            and node.write_through_pending_id is None
        )

    def _borrow_ordinary(self, tokens, compatibility, *, exact=False):
        end = len(tokens)
        for request, info in self.cache.ongoing_prefetch.items():
            op, suffix = info.operation, info.prefetch_key
            prefix = getattr(op, "control_prefix", None)
            start = op.storage_start
            suffix_end = start + len(suffix)
            if (
                prefix is None
                or not start < end <= suffix_end
                or (exact and end != suffix_end)
                or suffix.cache_salt != compatibility.cache_salt
                or suffix.extra_key != compatibility.extra_key
                or op.is_terminated()
                or suffix.token_ids[end - start - 1] != tokens[-1]
                or tuple(prefix) != tokens[:start]
                or tuple(suffix.token_ids[: end - start]) != tokens[start:]
            ):
                continue
            if (job := self.jobs.get(request)) is not None:
                return job
            if request in self.demands:
                # An ordinary suffix may still own a shared ancestor. A new
                # job must not replace its demand and strand the prefix hold.
                if exact:
                    continue
                raise ControlError(
                    "unavailable", "Ordinary suffix still owns a shared prefix"
                )
            if self.cache.prefetch_stop_policy != "wait_complete":
                raise ControlError(
                    "unavailable", "Compatible ordinary restore uses early stop"
                )
            if (
                len(self.demands) >= self.executor.config.max_prepared_prefixes
                or start + len(suffix)
                > self.cache.kv_control_service.config.limits.max_scope_tokens
            ):
                raise ControlError(
                    "unavailable", "Ordinary read interest capacity exhausted"
                )
            # The existing operation keeps its request handle, allocation and
            # stop policy. Only its lifetime and publication become shared.
            job = _Read(
                request,
                tokens[:start] + tuple(suffix.token_ids),
                compatibility,
                handle=request,
                operation=op,
            )
            self.jobs[request] = job
            self.demands[request] = _Demand(job, start)
            job.demands.add(request)
            self._observe_publication(job)
            self.stats["ordinary_reads_reused"] += 1
            return job
        return None

    def _observe_publication(self, job):
        job.operation.native_quiescent = False
        job.operation.native_on_publish = lambda node: self._published(job, node)

    def start(self, work):
        tokens, compatibility = self._scope(work)
        key = hashlib.sha256(msgspec.json.encode((tokens, compatibility))).digest()
        job = self.jobs.get(key)
        if job is None:
            job = self._borrow_ordinary(tokens, compatibility)
        rows = None
        if job is None:
            job = _Read(key, tokens, compatibility)
            # Validate before registering the job. The first attempt can use
            # this owner-local walk; queued retries must inspect the tree again.
            rows = self._rows(job)
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
            self._start(job, rows=rows)
        return self._take(interest)

    def _start(self, job, *, rows=None):
        # The owner polls storage before applying individual deadlines. A
        # newly available slot must not start a read for expired hints only.
        if not self._has_live_interest(job):
            return
        try:
            pool = self.cache.cache_controller.mem_pool_host
            if len(job.tokens) > pool.size:
                self._retire(
                    job,
                    status=Status.DEFERRED,
                    code="scope_exceeds_destination_capacity",
                )
                return
            if rows is None:
                rows = self._rows(job)
            if any(
                n.write_through_pending_id is not None or self.executor._load_pending(n)
                for n, _, _ in rows
            ):
                job.waiting = "source_pending"
                return
            host_length = self._host_length(job, rows)
            if host_length == len(job.tokens):
                self._retire(job, host_length=host_length)
                return
            if rows:
                node, start, end = rows[-1]
                if end - start < len(node.key):
                    # A matched slice of a compressed sibling is readable,
                    # but only a full node can anchor a storage suffix.
                    rows.pop()
                    host_length = min(host_length, start)
            waiting = self._admission_wait(job)
            if waiting is not None:
                job.waiting = waiting
                return
            if self.cache.cache_controller.prefetch_rate_limited():
                job.waiting = "storage_rate_limit"
                return
            self._hold(job, rows)
            needed = len(job.tokens) - host_length
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
            self._observe_publication(job)
        except Exception as error:
            job.failure = (getattr(error, "code", "storage_error"), str(error)[:512])
            if job.operation is None:
                self._retire(job)

    def _admission_wait(self, job):
        active = [
            other
            for other in self.jobs.values()
            if other is not job and other.operation is not None
        ]
        if len(active) >= self.executor.config.max_concurrent_storage_reads:
            return "storage_concurrency_limit"
        page_size = self.cache.page_size
        root_page = job.tokens[:page_size]
        if any(
            job.compatibility.cache_salt == other.compatibility.cache_salt
            and job.compatibility.extra_key == other.compatibility.extra_key
            and root_page == other.tokens[:page_size]
            for other in active
        ):
            return "shared_prefix_serialization"
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

        The caller holds the existing prefix. Eviction honors PIN and transfer
        references and may reclaim whole nodes beyond the shortfall.
        """
        pool = self.cache.cache_controller.mem_pool_host
        reserved = self._unallocated_reservations(job)
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

    def _retire(self, job, *, status=None, code="", host_length=None):
        length = 0
        try:
            if job.operation is not None and getattr(
                job.operation, "native_quiescent", False
            ):
                # Final ACK publication may have queued unused scratch pages
                # after the previous drain's release-count snapshot.
                self.cache.drain_storage_control_queues()
            length = self._host_length(job) if host_length is None else host_length
            scopes = {}
            for identity in job.interests:
                interest = self.interests[identity]
                end = interest.work.target.requested.end_token
                ready_end = min(end, length)
                ready = ready_end == end
                if end not in scopes:
                    scopes[end] = self._outcomes(job, ready_end, end)
                failure = job.failure
                if (
                    job.borrowed
                    and ready
                    and failure
                    and failure[0]
                    in (
                        "missing_copy",
                        "incomplete_storage_read",
                    )
                ):
                    failure = None
                terminal = (
                    Status.CANCELLED
                    if interest.cancelled
                    else job.failure_status
                    if failure
                    else status or (Status.COMPLETED if ready else Status.PARTIAL)
                )
                interest.result = ExecutionUpdate(
                    status=terminal,
                    quiescent=True,
                    code="cancelled"
                    if interest.cancelled
                    else failure[0]
                    if failure
                    else code,
                    reason=failure[1] if failure else "",
                    completed_steps=("destination_ready",) if ready else (),
                    target_outcomes=scopes[end],
                )
        finally:
            job.reserved_host_tokens = 0
            job.loaded_span = self.cache.pop_prefetch_loaded_span(job.handle)
            if not job.borrowed:
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
            if not job.borrowed:
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
            if not job.borrowed or job.handle not in job.demands:
                self.cache.discard_storage_prefetch_accounting(job.handle)
                self.cache.storage_prefetch_retries.cancel(job.handle.rid)
                self.cache._storage_prefetch_wait.pop(job.handle, None)
            if job.operation is not None:
                job.operation.native_on_publish = None
            self.jobs.pop(job.key)

    def _outcomes(self, job, ready_end, end):
        location = Location(tier="host", engine_id=self.executor.engine_id)
        return tuple(
            TargetOutcome(
                coverage=TokenRange(
                    component="full", start_token=left, end_token=right
                ),
                status=Status.COMPLETED if right == ready_end else Status.MISSING,
                completed_steps=("destination_ready",) if right == ready_end else (),
                location=location,
                code="" if right == ready_end else "missing_copy",
                content_id=self.executor._content_id(
                    job.tokens[:right], FULL, job.compatibility
                ),
            )
            for left, right in ((0, ready_end), (ready_end, end))
            if left < right
        )

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
