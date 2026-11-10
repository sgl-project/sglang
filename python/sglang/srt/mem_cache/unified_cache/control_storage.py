# SPDX-License-Identifier: Apache-2.0
"""Native FULL-prefix storage writes over the real HiCache worker queues."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import msgspec
import torch

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    ExecutionUpdate,
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
    tokens: tuple[int, ...]
    compatibility: object
    interests: set[str] = field(default_factory=set)
    host_locks: list = field(default_factory=list)
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
        self.jobs: dict[bytes, _Write] = {}
        self.interests: dict[str, _Interest] = {}

    def start(self, work):
        payload, adapter = work.payload, self.executor.storage_adapter
        if (
            not isinstance(payload, MoveTo)
            or (payload.src.tier, payload.dst.tier) not in adapter.move_paths
            or payload.dst.backend not in (None, adapter.backend)
            or payload.src.backend is not None
            or any(
                loc.engine_id not in (None, self.executor.engine_id)
                for loc in (payload.src, payload.dst)
            )
        ):
            raise ControlError(
                "unsupported", "DEMOTE requires local host KV and compatible storage"
            )
        tokens, compatibility = self.executor._storage_prefix(work)
        if max(2, len(tokens) // self.cache.page_size) > min(
            work.max_result_targets, self.executor.config.max_result_targets
        ):
            raise ControlError("capacity", "DEMOTE scope exceeds the result budget")
        key = hashlib.sha256(msgspec.json.encode((tokens, compatibility))).digest()
        job = self.jobs.get(key)
        if job is None:
            job = _Write(key, tokens, compatibility)
            self.jobs[key] = job
        identity = work.ref.operation_id
        interest = _Interest(work, job)
        self.interests[identity] = interest
        job.interests.add(identity)
        if job.operation_id is None:
            self._stage(job)
        return self._take(interest)

    def _rows(self, job):
        path = self.executor._walk(job.tokens, job.compatibility)
        if not path or path[-1][3] != len(job.tokens):
            raise ControlError("missing", "Prefix no longer exists in this cache")
        if any(matched != end for _, _, end, matched in path):
            raise ControlError(
                "unsupported", "Prefix endpoint is not a materialized radix boundary"
            )
        return [(node, start, end) for node, start, end, _ in path]

    def _stage(self, job):
        try:
            rows = self._rows(job)
            if any(node.write_through_pending_id is not None for node, _, _ in rows):
                blocked = "inflight_backup"
            else:
                values = [node.component_data[FULL].host_value for node, _, _ in rows]
                if any(value is None for value in values):
                    raise ControlError("missing", "Host source is not fully published")
                # Reclamation proceeds from the end of the path. A blocked
                # endpoint prevents reclaiming any of its ancestors.
                blocked = self._host_reclamation_reason(rows[-1][0])
            if blocked:
                job.failure_status = Status.DEFERRED
                raise ControlError(blocked, "No host range can be reclaimed")
            if any(
                value.numel() != end - start
                for value, (_, start, end) in zip(values, rows)
            ):
                raise ControlError(
                    "unsupported", "Host source has an unsupported layout"
                )
            hashes = [
                h
                for node, _, _ in rows
                for h in self.cache.tree_core.get_hash_values(node.id)
            ]
            if len(hashes) * self.cache.page_size != len(job.tokens):
                raise ControlError("unsupported", "Incomplete source page-hash chain")
            self.executor._hold_host(job.host_locks, (node for node, _, _ in rows))
            # The worker owns the complete hash chain and source slots through ACK.
            job.operation_id = self.cache.cache_controller.write_storage(
                torch.cat(values),
                list(job.tokens),
                hashes,
                [] if self.cache.hicache_storage_pass_prefix_keys else None,
                verify_prefix=True,
            )
            self.cache._native_storage_write_results[job.operation_id] = None
        except Exception as error:
            job.failure = (getattr(error, "code", "storage_error"), str(error)[:512])
            job.failure_status = {
                "missing": Status.MISSING,
                "unsupported": Status.UNSUPPORTED,
                "capacity": Status.DEFERRED,
            }.get(job.failure[0], job.failure_status)
            self._retire(job)

    def begin_poll(self, *, limit=64):
        if not self.jobs:
            return
        self.cache.drain_storage_control_queues()
        for job in tuple(self.jobs.values())[:limit]:
            self.jobs.pop(job.key)
            self.jobs[job.key] = job
            completed = self.cache._native_storage_write_results[job.operation_id]
            if completed is not None:
                end = len(job.tokens)
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

    def _host_reclamation_reason(self, node):
        if node.write_through_pending_id is not None:
            return "inflight_backup"
        if self.executor._load_pending(node):
            return "inflight_load"
        if any(cd.host_lock_ref or cd.lock_ref for cd in node.component_data):
            return "active_reference"
        if not node.evicted:
            return "source_gpu_resident"
        if node.children:
            return "structural_dependency"
        return "" if self.cache.tree_core._is_host_leaf(node) else "not_reclaimable"

    def _reclaim_host(self, job, rows):
        # Worker ACK is quiescent. Only this job's source receipts have been
        # released; PINs, computation and other transfers keep their own holds.
        for node, _, _ in reversed(rows):
            if self._host_reclamation_reason(node):
                continue
            result = self.cache.tree_core.evict_host_leaf(node.id)
            freed = sum(value.numel() for value in result.host_frees.get(FULL, ()))
            self.cache._free_values(result.device_frees, result.host_frees)
            self.cache._apply_cache_actions(result.cache_actions)
            if freed:
                job.reclaimed.add(node.id)

    def _outcomes(self, interest, rows):
        job = interest.job
        boundaries = {0, job.completed, len(job.tokens)}
        if job.completed == len(job.tokens):
            boundaries.update(end for _, _, end in rows)
        boundaries = sorted(boundaries)
        output = []
        for start, end in zip(boundaries, boundaries[1:]):
            ready = end <= job.completed
            reclaimed = ready and all(
                n.id in job.reclaimed for n, a, b in rows if a < end and b > start
            )
            steps = ("destination_ready",) if ready else ()
            if reclaimed:
                steps += ("source_reclaimed",)
            code = "" if reclaimed else "missing_copy"
            if ready and not reclaimed:
                code = (
                    job.failure[0]
                    if job.failure
                    else "cancelled"
                    if interest.cancelled
                    else next(
                        (
                            self._host_reclamation_reason(n)
                            for n, a, b in rows
                            if a < end and b > start and n.id not in job.reclaimed
                        ),
                        "not_reclaimable",
                    )
                )
            outcome = TargetOutcome(
                coverage=TokenRange(component="full", start_token=start, end_token=end),
                status=Status.COMPLETED
                if reclaimed
                else Status.PARTIAL
                if ready
                else Status.MISSING,
                location=self.executor.storage_adapter.location(),
                completed_steps=steps,
                code=code,
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
                    job.tokens[: o.coverage.end_token], FULL, job.compatibility
                ),
            )
            for o in output
        )

    def _retire(self, job):
        # Freeze current radix geometry before a host MOVE removes any nodes.
        # Ordinary matching may have split the path while the worker ran.
        rows = self._rows(job) if job.operation_id is not None else ()
        self.executor._release_host(job.host_locks)
        active = any(not self.interests[i].cancelled for i in job.interests)
        if rows and job.failure is None and active:
            self._reclaim_host(job, rows)
        try:
            for identity in job.interests:
                interest = self.interests[identity]
                outcomes = self._outcomes(interest, rows) if rows else ()
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
            self.executor._release_host(job.host_locks)
            if job.operation_id is not None:
                self.cache._native_storage_write_results.pop(job.operation_id)
            self.jobs.pop(job.key)

    def _take(self, interest):
        if interest.result is not None:
            self.interests.pop(interest.work.ref.operation_id)
            return interest.result
        return ExecutionUpdate(
            status=Status.CANCELLING
            if interest.cancelled or interest.job.failure
            else Status.RUNNING,
            quiescent=False,
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
        return self._take(interest)
