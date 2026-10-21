# SPDX-License-Identifier: Apache-2.0
"""Bounded native interests over the existing persistent-host HiCache path."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    ExecutionUpdate,
    Location,
    MoveTo,
    Status,
    TargetOutcome,
    TokenRange,
    TokensTarget,
)
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

FULL = ComponentType.FULL


@dataclass
class _Job:
    key: bytes
    work: object
    interests: set[str] = field(default_factory=set)
    gpu_locks: list = field(default_factory=list)
    host_locks: list = field(default_factory=list)
    started: bool = False
    failure: tuple[str, str] | None = None
    reclaimed: set[int] = field(default_factory=set)
    waiting: str = "resource_approval"


@dataclass
class _Interest:
    work: object
    job: _Job
    cancelled: bool = False
    result: ExecutionUpdate | None = None


class HostCacheTransfers:
    """Real FULL-prefix copies, shared jobs, and independently cancellable users.

    HiCache owns allocation, queued DMA, fences and physical publication. This
    adapter holds the complete reusable scope until its last required fence.
    H2D admission uses a scheduler prefill barrier while the installed indices
    are still pending, since a native load has no inference-batch consumer index.
    """

    def __init__(self, executor):
        self.executor, self.cache, self.core = executor, executor.cache, executor.core
        self.jobs: dict[bytes, _Job] = {}
        self.interests: dict[str, _Interest] = {}

    @property
    def blocks_prefill(self):
        return any(
            j.started and j.work.payload.dst.tier == "gpu" for j in self.jobs.values()
        )

    def _prefixes(self, work):
        self.executor._check(work.target)
        payload = work.payload
        if any(
            location.engine_id not in (None, self.executor.engine_id)
            or location.backend is not None
            for location in (payload.src, payload.dst)
        ):
            raise ControlError(
                "unsupported", "Transfer must address this owner's local tiers"
            )
        direction = payload.src.tier, payload.dst.tier
        if direction not in (("gpu", "host"), ("host", "gpu")) or (
            isinstance(payload, MoveTo) and direction != ("gpu", "host")
        ):
            raise ControlError(
                "unsupported",
                "This adapter supports GPU/host COPY and GPU-to-host MOVE",
            )
        prefixes = []
        for branch in work.target.branches:
            if not isinstance(branch.content, TokensTarget):
                raise ControlError(
                    "unsupported",
                    "Host movement currently requires a token-backed prefix",
                )
            end = branch.requested.end_token
            if branch.requested.start_token or end % self.core.page_size:
                raise ControlError(
                    "unsupported",
                    "Host movement requires a complete page-aligned prefix",
                )
            if (
                branch.missing
                or branch.unaligned
                or end > len(branch.content.token_ids)
            ):
                raise ControlError("missing", "Requested prefix history is incomplete")
            compatibility = self.executor._compatibility(branch.content)
            prefixes.append((branch.content.token_ids[:end], compatibility))
        # Reserve traversal room for every possible ordinary radix split while
        # I/O is pending; completion must not become impossible to inspect.
        if (
            sum(len(tokens) // self.core.page_size for tokens, _ in prefixes)
            > self.executor.config.max_query_nodes
        ):
            raise ControlError(
                "capacity",
                "Transfer scope exceeds its worst-case page traversal budget",
            )
        return prefixes

    def _paths(self, work, *, source=True):
        paths = []
        count = 0
        result_budget = 0
        for tokens, compatibility in self._prefixes(work):
            path = self.executor._walk(tokens, compatibility)
            if not path or path[-1][3] != len(tokens):
                raise ControlError("missing", "Prefix no longer exists in this cache")
            rows = []
            for node, start, end, matched in path:
                if matched != end or end > len(tokens):
                    raise ControlError(
                        "unsupported",
                        "Prefix endpoint is not a materialized radix boundary",
                    )
                cd = node.component_data[FULL]
                value = cd.value if work.payload.src.tier == "gpu" else cd.host_value
                if source and value is None:
                    raise ControlError(
                        "missing", "Selected source prefix is not resident"
                    )
                if source and value.numel() != len(node.key):
                    raise ControlError(
                        "unsupported", "Source has an unsupported physical span layout"
                    )
                rows.append((node, start, end))
            count += len(rows)
            if count > self.executor.config.max_query_nodes:
                raise ControlError(
                    "capacity", "Transfer traversal exceeds query budget"
                )
            destination = [
                (
                    node.component_data[FULL].host_value
                    if work.payload.dst.tier == "host"
                    else node.component_data[FULL].value
                )
                is not None
                for node, _, _ in rows
            ]
            runs = 1 + sum(a != b for a, b in zip(destination, destination[1:]))
            # Existing destination runs cannot disappear under our holds.
            # A failed top-down submission can add one published prefix;
            # FULL MOVE leaves a retained prefix and a reclaimed suffix.
            result_budget += max(
                2 if isinstance(work.payload, MoveTo) else 1,
                runs + (not all(destination)),
            )
            paths.append((tokens, compatibility, rows))
        if source and result_budget > min(
            work.max_result_targets, self.executor.config.max_result_targets
        ):
            raise ControlError(
                "capacity", "Partial transfer coverage exceeds result budget"
            )
        return paths

    def _load_pending(self, node):
        if node.load_back_pending_id is not None:
            return True
        # Write-through loads need not set load_back_pending_id. The existing
        # ongoing anchor still owns every copied node until its ack fence.
        for anchor in self.cache.ongoing_load_back:
            current = self.core.node_by_id(anchor)
            while current is not None and current is not self.core.root_node:
                if current is node:
                    return True
                current = current.parent
        return False

    def start(self, work):
        prefixes = self._prefixes(work)
        self._paths(work)  # Each joiner approves its own immutable scope/budget.
        key = hashlib.sha256(
            msgspec.json.encode(
                (
                    work.payload.src.tier,
                    work.payload.dst.tier,
                    isinstance(work.payload, MoveTo),
                    sorted(set(prefixes), key=msgspec.json.encode),
                )
            )
        ).digest()
        job = self.jobs.get(key)
        if job is None:
            job = _Job(key, work)
            self.jobs[key] = job
        identity = work.ref.operation_id
        interest = _Interest(work, job)
        job.interests.add(identity)
        self.interests[identity] = interest
        if not job.started:
            self._try_start(job)
        result = self._current(interest)
        if result.status.terminal:
            self.interests.pop(identity)
        return result

    def _release_gpu(self, job):
        for lock in reversed(job.gpu_locks):
            self.cache.unlock(lock)
        job.gpu_locks.clear()

    def _release_host(self, job):
        for node_id, receipt in reversed(job.host_locks):
            self.cache.dec_host_lock_ref(node_id, receipt)
        job.host_locks.clear()

    def _host_hold(self, job, nodes):
        known = {node_id for node_id, _ in job.host_locks}
        for node in nodes:
            if (
                node.id not in known
                and node.component_data[FULL].host_value is not None
            ):
                receipt = self.cache.inc_host_lock_ref(node.id).to_dec_params()
                job.host_locks.append((node.id, receipt))

    def _try_start(self, job):
        try:
            paths = self._paths(job.work)
            nodes = {n.id: n for _, _, rows in paths for n, _, _ in rows}
            src = job.work.payload.src.tier
            pool = (
                self.cache.cache_controller.mem_pool_host.anchor_entry.host_pool
                if src == "gpu"
                else self.cache.token_to_kv_pool_allocator
            )
            capacity = getattr(pool, "size", None)
            if (
                type(capacity) is int
                and sum(len(n.key) for n in nodes.values()) > capacity
            ):
                for identity in job.interests:
                    self.interests[identity].result = ExecutionUpdate(
                        status=Status.DEFERRED,
                        quiescent=True,
                        code="scope_exceeds_destination_capacity",
                        reason="Select a smaller reusable scope; the complete scope exceeds the destination pool",
                    )
                self.jobs.pop(job.key, None)
                return
            if any(
                (
                    self._load_pending(n)
                    if src == "gpu"
                    else n.write_through_pending_id is not None
                )
                for n in nodes.values()
            ):
                job.waiting = "source_pending"
                return
            anchors = {rows[-1][0].id: rows[-1][0] for _, _, rows in paths}
            job.gpu_locks = [self.cache.lock(n.id) for n in anchors.values()]
            self._host_hold(job, nodes.values())
            needed = sum(
                len(n.key)
                for n in nodes.values()
                if (
                    n.component_data[FULL].host_value
                    if src == "gpu"
                    else n.component_data[FULL].value
                )
                is None
            )
            if src == "gpu":
                available = self.cache.cache_controller.mem_pool_host.available_size()
            else:
                available = self._gpu_available()
            if needed > available:
                # No destination is reserved until the entire unique scope fits.
                self._release_host(job)
                self._release_gpu(job)
                job.waiting = "whole_scope_capacity"
                return
            job.started = True  # Partial submission must retain holds on error.
            if src == "gpu":
                self.cache._execute_and_commit_kv_backup(BackupKV(node_ids=list(nodes)))
                self._host_hold(job, nodes.values())
                if any(
                    n.component_data[FULL].host_value is None for n in nodes.values()
                ):
                    raise ControlError(
                        "capacity",
                        "Host allocation stopped before the full scope was queued",
                    )
            else:
                for _, _, rows in paths:
                    anchor = rows[-1][0]
                    if any(n.component_data[FULL].value is None for n, _, _ in rows):
                        quota = self._gpu_available()
                        if not self.cache.load_back(
                            anchor.id, mem_quota=quota, minimum_load_tokens=1
                        ):
                            raise ControlError(
                                "capacity",
                                "Device allocation stopped before the full scope was queued",
                            )
        except Exception as error:
            job.failure = (getattr(error, "code", "transfer_error"), str(error)[:512])
            if not job.started:
                self._retire(job)

    def _gpu_available(self):
        if self.executor.gpu_copy_capacity is not None:
            return self.executor.gpu_copy_capacity()
        return (
            self.cache._component_available_size(FULL) + self.core.full_evictable_size()
        )

    def begin_poll(self, *, limit=64):
        if not self.jobs:
            return
        selected = tuple(self.jobs.values())[:limit]
        for job in selected:
            self.jobs.pop(job.key)
            self.jobs[job.key] = job
            if not job.started:
                self._try_start(job)
        if not any(j.started for j in self.jobs.values()):
            return
        try:
            self.cache.flush_pending_backups()
            self.cache.writing_check()
            if self.blocks_prefill:
                self.cache.ready_to_load_host_cache()
            self.cache.loading_check()
        except Exception as error:
            for job in self.jobs.values():
                if job.started and job.failure is None:
                    job.failure = ("transfer_error", str(error)[:512])
        for job in selected:
            if job.key not in self.jobs or not job.started:
                continue
            try:
                paths = self._paths(job.work, source=False)
                nodes = [n for _, _, rows in paths for n, _, _ in rows]
                pending = any(
                    n.write_through_pending_id is not None or self._load_pending(n)
                    for n in nodes
                )
                # An exception may occur between allocation and tree tracking.
                if job.failure:
                    pending |= (
                        self.cache.cache_controller.has_inflight_device_transfers()
                    )
                if not pending:
                    self._retire(job)
            except Exception as error:
                if job.failure is None:
                    job.failure = ("transfer_error", str(error)[:512])

    def _outcomes(self, work, reclaimed):
        outcomes = []
        paths = self._paths(work, source=False)
        for branch, (_, _, rows) in enumerate(paths):
            for node, start, end in rows:
                cd = node.component_data[FULL]
                destination = (
                    cd.host_value if work.payload.dst.tier == "host" else cd.value
                )
                ready = destination is not None and (
                    node.write_through_pending_id is None
                    if work.payload.dst.tier == "host"
                    else not self._load_pending(node)
                )
                source = cd.value if work.payload.src.tier == "gpu" else cd.host_value
                steps = ("destination_ready",) if ready else ()
                complete = ready
                if isinstance(work.payload, MoveTo):
                    complete &= source is None and node.id in reclaimed
                    if complete:
                        steps += ("source_reclaimed",)
                outcomes.append(
                    TargetOutcome(
                        branch_index=branch,
                        coverage=TokenRange(
                            component="full", start_token=start, end_token=end
                        ),
                        status=Status.COMPLETED
                        if complete
                        else Status.PARTIAL
                        if ready
                        else Status.MISSING,
                        location=Location(
                            tier=work.payload.dst.tier,
                            engine_id=self.executor.engine_id,
                        ),
                        completed_steps=steps,
                        code="source_busy"
                        if ready and not complete
                        else ""
                        if complete
                        else "missing_copy",
                    )
                )
        merged = []
        for outcome in outcomes:
            if merged:
                previous = merged[-1]
                if (
                    previous.branch_index,
                    previous.status,
                    previous.code,
                    previous.completed_steps,
                    previous.location,
                    previous.coverage.end_token,
                ) == (
                    outcome.branch_index,
                    outcome.status,
                    outcome.code,
                    outcome.completed_steps,
                    outcome.location,
                    outcome.coverage.start_token,
                ):
                    merged[-1] = msgspec.structs.replace(
                        previous,
                        coverage=TokenRange(
                            component="full",
                            start_token=previous.coverage.start_token,
                            end_token=outcome.coverage.end_token,
                        ),
                        content_id=outcome.content_id,
                    )
                    continue
            merged.append(outcome)
        # Hash the final logical runs, avoiding a full-prefix hash for every
        # physical fragment when ordinary matching split a copy during DMA.
        return tuple(
            msgspec.structs.replace(
                outcome,
                content_id=self.executor._content_id(
                    paths[outcome.branch_index][0][: outcome.coverage.end_token],
                    FULL,
                    paths[outcome.branch_index][1],
                ),
            )
            for outcome in merged
        )

    def _retire(self, job):
        active = [
            self.interests[i] for i in job.interests if not self.interests[i].cancelled
        ]
        self._release_gpu(job)
        if (
            job.started
            and job.failure is None
            and active
            and isinstance(job.work.payload, MoveTo)
        ):
            paths = self._paths(job.work, source=False)
            nodes = {n.id: (n, end) for _, _, rows in paths for n, _, end in rows}
            for node, _ in sorted(
                nodes.values(), key=lambda item: item[1], reverse=True
            ):
                cd = node.component_data[FULL]
                if (
                    cd.value is not None
                    and cd.host_value is not None
                    and node in self.core.evictable_device_leaves
                ):
                    result = self.core.demote(node.id)
                    self.cache._free_values(result.device_frees, result.host_frees)
                    job.reclaimed.add(node.id)
        try:
            for identity in job.interests:
                interest = self.interests[identity]
                outcomes = (
                    self._outcomes(interest.work, job.reclaimed) if job.started else ()
                )
                if job.failure:
                    code, reason = job.failure
                    status = Status.MISSING if code == "missing" else Status.FAILED
                elif interest.cancelled:
                    status, code, reason = Status.CANCELLED, "cancelled", ""
                else:
                    status = (
                        Status.COMPLETED
                        if outcomes
                        and all(o.status == Status.COMPLETED for o in outcomes)
                        else Status.PARTIAL
                    )
                    code, reason = "", ""
                steps = tuple(
                    step
                    for step in ("destination_ready", "source_reclaimed")
                    if outcomes and all(step in o.completed_steps for o in outcomes)
                )
                interest.result = ExecutionUpdate(
                    status=status,
                    quiescent=True,
                    code=code,
                    reason=reason,
                    completed_steps=steps,
                    target_outcomes=outcomes,
                )
        finally:
            self._release_host(job)
            self.jobs.pop(job.key, None)

    @staticmethod
    def _current(interest):
        if interest.result is not None:
            return interest.result
        if interest.job.failure:
            code, reason = interest.job.failure
            return ExecutionUpdate(
                status=Status.CANCELLING, quiescent=False, code=code, reason=reason
            )
        return ExecutionUpdate(
            status=Status.CANCELLING
            if interest.cancelled
            else Status.RUNNING
            if interest.job.started
            else Status.QUEUED,
            quiescent=not interest.job.started,
            code="draining"
            if interest.cancelled
            else "transfer_pending"
            if interest.job.started
            else interest.job.waiting,
        )

    def poll(self, ref):
        interest = self.interests.get(ref.operation_id)
        if interest is None:
            return None
        result = self._current(interest)
        if result.status.terminal:
            self.interests.pop(ref.operation_id)
        return result

    def cancel(self, ref):
        interest = self.interests.get(ref.operation_id)
        if interest is None:
            return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)
        if interest.result is not None:
            return self.poll(ref)
        interest.cancelled = True
        job = interest.job
        if job.failure is None and any(
            not self.interests[i].cancelled for i in job.interests
        ):
            job.interests.remove(ref.operation_id)
            self.interests.pop(ref.operation_id)
            return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)
        if not job.started:
            self._retire(job)
        return self.poll(ref)
