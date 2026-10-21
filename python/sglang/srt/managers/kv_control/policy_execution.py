# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous policy plans; native control owns physical lifetimes."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass, field

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    CopyTo,
    ExecutionUpdate,
    MoveTo,
    SetPriority,
    Status,
    TokensTarget,
)
from sglang.srt.managers.kv_control.policy import placement, scopes_overlap
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


@dataclass
class _Plan:
    remaining: int
    outcomes: dict = field(default_factory=dict)


@dataclass
class _Work:
    ref: object
    entry: object
    plan: _Plan
    dependencies: tuple
    policy: str
    owner: str
    tasks: tuple
    binding: tuple | None
    wait_binding: tuple | None
    not_before: float
    deadline: float | None
    selected: str | None = None
    lease: object = None
    started: bool = False
    started_at: float | None = None
    raw_done: bool = False
    ready: bool = False
    cancelled: bool = False
    notified: bool = False


def scope_tokens(scope):
    return sum(
        branch.requested.end_token - branch.requested.start_token
        for branch in scope.branches
    )


class PolicyExecution:
    def __init__(self, runtime):
        self.runtime, self.control, self.service = (
            runtime,
            runtime.control,
            runtime.service,
        )
        self.works = {}
        self.paused = OrderedDict()
        self.outcomes = {}
        self.stats = {
            name: 0
            for name in (
                "proposals",
                "started",
                "completed",
                "deferred",
                "cancelled",
                "preparations_ready",
                "preparations_consumed",
                "preparations_expired",
                "policy_expired",
                "preparations_revoked",
                "consumed_prefix_tokens",
            )
        }

    def previous_scope(self, policy, owner):
        record = self.paused.get((policy, owner))
        return record[0] if record is not None else None

    def remember_pause(self, policy, signal):
        self.paused[policy, signal.owner_key] = (signal.scope, signal.tasks)
        self.paused.move_to_end((policy, signal.owner_key))
        limit = self.service.config.max_policy_snapshot_tokens
        policy_limit = self.runtime.policies[policy].options.max_contexts
        while sum(key[0] == policy for key in self.paused) > policy_limit:
            first = next(key for key in self.paused if key[0] == policy)
            self.paused.pop(first)
        while (
            len(self.paused) > self.service.config.max_policy_actions
            or sum(scope_tokens(value[0]) for value in self.paused.values()) > limit
        ):
            self.paused.popitem(last=False)

    def reserve(self, parent, signal, plan, changed_policies):
        accepted = sum(entry.accepted for entry in plan)
        if accepted + len(self.works) > self.service.config.max_policy_actions:
            raise ControlError("capacity", "Policy work capacity exhausted")
        if (
            sum(scope_tokens(w.entry.proposal.scope) for w in self.works.values())
            + sum(scope_tokens(e.proposal.scope) for e in plan if e.accepted)
            > self.service.config.max_policy_snapshot_tokens
        ):
            raise ControlError("capacity", "Policy frozen-scope capacity exhausted")
        now = self.control._clock()
        if any(
            e.proposal.not_before - now > self.service.config.policy_max_delay_seconds
            for e in plan
        ):
            raise ControlError("capacity", "Policy prediction exceeds its horizon")
        refs = (
            self.control._reserve_policy_operations(
                tuple((self.runtime.action_id(e.ref), e.proposal.action) for e in plan),
                resolutions=tuple(e.proposal.scope for e in plan),
            )
            if plan
            else ()
        )
        group = _Plan(accepted, self.outcomes)
        by_ref = {entry.ref: child for entry, child in zip(plan, refs, strict=True)}
        binding = (
            (parent.context.caller_id, parent.context.binding_id)
            if parent.context.binding_id is not None
            else None
        )
        created = set()
        for entry, child in zip(plan, refs, strict=True):
            self.stats["proposals"] += 1
            if not entry.accepted:
                result = self.control._finish(
                    child,
                    ExecutionUpdate(
                        status=Status.DEFERRED, quiescent=True, code=entry.code
                    ),
                )
                group.outcomes[child.operation_id] = (result.status, False)
                self.stats["deferred"] += 1
                continue
            proposal = entry.proposal
            deadline = (
                max(now, proposal.not_before)
                + self.service.config.policy_max_wait_seconds
                if proposal.prepare_for_admission
                else None
                if binding is not None and isinstance(proposal.action, MoveTo)
                else max(now, proposal.not_before)
                + self.service.config.policy_max_delay_seconds
            )
            work = _Work(
                child,
                entry,
                group,
                tuple(by_ref[dep].operation_id for dep in entry.dependencies),
                entry.ref.policy,
                signal.owner_key,
                signal.tasks,
                binding,
                binding if isinstance(proposal.action, MoveTo) else None,
                proposal.not_before,
                deadline,
            )
            self.works[child.operation_id] = work
            created.add(child.operation_id)
        # A resume supersedes saves, while an in-flight compatible COPY retains
        # its interest so a new reader can join without terminating shared IO.
        for old in tuple(self.works.values()):
            if old.ref.operation_id in created or old.owner != signal.owner_key:
                continue
            if old.policy not in changed_policies:
                continue
            action = old.entry.proposal.action
            if isinstance(action, MoveTo) or old.binding is None and not old.started:
                self.cancel(old, "newer_policy_signal")
        return refs

    def _retire(self, work):
        if work.lease is not None:
            work.lease.release()
        self.works.pop(work.ref.operation_id, None)
        work.plan.remaining -= 1

    def cancel(self, work, reason):
        if work.cancelled:
            return
        work.cancelled = True
        if work.lease is not None:
            work.lease.release()
        if not work.raw_done:
            current = self.control.registry.get(work.ref)
            if not current.status.terminal:
                self.control.cancel(work.ref)
        # Keep the work, snapshot and result interest until the core certifies
        # that all worker access has stopped. Reaping never invents quiescence.
        self.stats["cancelled"] += 1

    def _notify(self, work, result):
        if work.notified:
            return
        work.notified = True
        elapsed = (
            0.0
            if work.started_at is None
            else max(0.0, self.control._clock() - work.started_at)
        )
        try:
            self.runtime.policies[work.policy].on_result(
                work.entry.proposal, result, elapsed_seconds=elapsed
            )
        except Exception as error:
            self.runtime.errors[work.policy] = f"{type(error).__name__}: {error}"[:512]

    def reap(self):
        for work in tuple(self.works.values()):
            if work.raw_done:
                if work.cancelled or work.lease is not None and work.lease.released:
                    self._retire(work)
                continue
            result = self.control.registry.get(work.ref)
            if not result.status.terminal:
                continue
            work.raw_done = True
            satisfied = result.status == Status.COMPLETED or (
                work.lease is not None
                and work.lease.ready
                and result.update.code == "gpu_already_ready"
            )
            work.plan.outcomes[work.ref.operation_id] = (result.status, satisfied)
            self._notify(work, result)
            if result.status == Status.COMPLETED:
                self.stats["completed"] += 1
            if (
                work.entry.proposal.prepare_for_admission
                and work.lease is not None
                and work.lease.ready
                and not work.cancelled
            ):
                work.ready = True
                self.stats["preparations_ready"] += 1
            else:
                self._retire(work)
        needed = {w.ref.operation_id for w in self.works.values()} | {
            dep for w in self.works.values() for dep in w.dependencies
        }
        for identity in tuple(self.outcomes):
            if identity not in needed:
                self.outcomes.pop(identity)

    def _materialize(self, scope):
        # The execution owner may split a page-aligned boundary with an ordinary
        # native match. Policies never receive the cache or perform this edit.
        executor = self.control.executor
        if (
            sum(
                branch.requested.end_token // executor.core.page_size
                for branch in scope.branches
            )
            > executor.config.max_query_nodes
        ):
            raise ControlError(
                "capacity", "Movement scope exceeds its traversal budget"
            )
        for branch in scope.branches:
            if (
                not isinstance(branch.content, TokensTarget)
                or branch.requested.start_token
                or branch.requested.end_token % executor.core.page_size
            ):
                raise ControlError(
                    "unsupported",
                    "Movement policies require complete aligned token prefixes",
                )
            content = branch.content
            compatibility = executor._compatibility(content)
            self.service.cache.touch_prefix(
                RadixKey(
                    array("q", content.token_ids[: branch.requested.end_token]),
                    cache_salt=compatibility.cache_salt,
                    extra_key=compatibility.extra_key,
                )
            )

    def _conflicts(self, action, scope):
        destination = placement(action)
        if not destination:
            return False
        for pending in self.control._pending.values():
            if (
                pending.origin == "explicit"
                and pending.target is not None
                and placement(pending.payload) not in ("", destination)
                and scopes_overlap(scope, pending.target)
            ):
                return True
        if isinstance(action, MoveTo):
            executor = self.control.executor
            for branch in scope.branches:
                if not isinstance(branch.content, TokensTarget):
                    return True
                for node, _, _, _ in executor._walk(
                    branch.content.token_ids[: branch.requested.end_token],
                    executor._compatibility(branch.content),
                ):
                    if (
                        executor.claims.explicit_priority(node, ComponentType.FULL)
                        is not None
                    ):
                        return True
        return False

    def progress(self):
        self.reap()
        now = self.control._clock()
        started = 0
        for work in tuple(self.works.values()):
            if (
                work.deadline is not None
                and now >= work.deadline
                and not work.cancelled
            ):
                self.stats[
                    "preparations_expired"
                    if work.entry.proposal.prepare_for_admission
                    else "policy_expired"
                ] += 1
                self.cancel(work, "preparation_deadline")
            if (
                work.cancelled
                or work.raw_done
                or work.started
                or work.wait_binding is not None
                or work.not_before > now
            ):
                continue
            if started >= self.service.config.max_policy_actions_per_step:
                break
            if any(dep not in work.plan.outcomes for dep in work.dependencies):
                continue
            if any(not work.plan.outcomes[dep][1] for dep in work.dependencies):
                self.control._finish(
                    work.ref,
                    ExecutionUpdate(
                        status=Status.DEFERRED, quiescent=True, code="dependency_failed"
                    ),
                )
                continue
            proposal = work.entry.proposal
            if proposal.prepare_for_admission and work.selected is None:
                continue
            try:
                if isinstance(proposal.action, SetPriority):
                    pending_transfer = False
                    executor = self.control.executor
                    for branch in proposal.scope.branches:
                        if isinstance(branch.content, TokensTarget):
                            for node, _, _, _ in executor._walk(
                                branch.content.token_ids[: branch.requested.end_token],
                                executor._compatibility(branch.content),
                            ):
                                pending_transfer |= (
                                    node.write_through_pending_id is not None
                                    or node.load_back_pending_id is not None
                                    or (
                                        executor.transfers is not None
                                        and executor.transfers._load_pending(node)
                                    )
                                )
                    if pending_transfer:
                        continue
                if self._conflicts(proposal.action, proposal.scope):
                    self.control._finish(
                        work.ref,
                        ExecutionUpdate(
                            status=Status.DEFERRED,
                            quiescent=True,
                            code="explicit_intent",
                        ),
                    )
                    self.stats["deferred"] += 1
                    continue
                if isinstance(proposal.action, (CopyTo, MoveTo)):
                    self._materialize(proposal.scope)
                if proposal.prepare_for_admission:
                    if work.lease is None:
                        work.lease = self.control.executor.prepare_prefix(work.ref)
                    if work.lease.capture():
                        # The owner has the actual ready scope; no new COPY was
                        # necessary. Keep its separate request-preparation lease.
                        self.control._finish(
                            work.ref,
                            ExecutionUpdate(
                                status=Status.DEFERRED,
                                quiescent=True,
                                code="gpu_already_ready",
                            ),
                        )
                        continue
                work.started, work.started_at = True, now
                self.stats["started"] += 1
                self.control._start_policy_operation(work.ref)
                started += 1
            except ControlError as error:
                self.control._finish(
                    work.ref,
                    ExecutionUpdate(
                        status=Status.DEFERRED
                        if error.code == "capacity"
                        else Status.UNSUPPORTED,
                        quiescent=True,
                        code=error.code,
                        reason=str(error)[:4096],
                    ),
                )
        self.reap()

    def before_explicit(self, action, scope):
        destination = placement(action)
        if destination:
            for work in tuple(self.works.values()):
                other = placement(work.entry.proposal.action)
                if (
                    other
                    and destination != other
                    and scopes_overlap(scope, work.entry.proposal.scope)
                ):
                    self.cancel(work, "explicit_intent")

    def _matches(self, work, req):
        tasks = getattr(req, "kv_control_policy_tasks", ())
        if not set(tasks).intersection(work.tasks):
            return False
        context = getattr(req, "kv_control_context", None)
        if work.binding is not None and (
            context is None or (context.caller_id, context.binding_id) != work.binding
        ):
            return False
        scope = work.entry.proposal.scope
        if len(scope.branches) != 1 or not isinstance(
            scope.branches[0].content, TokensTarget
        ):
            return False
        branch = scope.branches[0]
        end = branch.requested.end_token
        content = branch.content
        raw_salt = getattr(req, "kv_control_logical_cache_salt", req.cache_salt)
        return (
            content.cache_salt == raw_salt
            and content.extra_key == req.extra_key
            and len(req.origin_input_ids) >= end
            and tuple(req.origin_input_ids[:end]) == content.token_ids[:end]
            and req._compute_max_prefix_len(len(req.origin_input_ids)) >= end
        )

    def before_admission(self, req):
        matching = [
            w
            for w in self.works.values()
            if w.entry.proposal.prepare_for_admission
            and not w.cancelled
            and self._matches(w, req)
        ]
        if not matching:
            return True
        if (
            self.runtime.waiting_seconds(req)
            >= self.service.config.policy_max_wait_seconds
        ):
            for work in matching:
                self.cancel(work, "admission_wait_limit")
            return True
        if req.cache_request_handle in self.service.cache.ongoing_prefetch:
            # Let the real inference prefetch publish its host result rather
            # than issuing a duplicate cold query/allocation for this request.
            return False
        if (
            getattr(req, "lora_id", None) is not None
            or getattr(req, "multimodal_inputs", None) is not None
            or getattr(req, "positional_embed_overrides", None) is not None
            or req.kv.holds_kv
            or getattr(req, "beam_group", None) is not None
            or getattr(req, "dllm_config", None) is not None
        ):
            for work in matching:
                self.cancel(work, "unsupported_preparation_consumer")
            return True
        observed = next(
            (
                r
                for r in (
                    self.runtime._view.requests
                    if self.runtime._view is not None
                    else ()
                )
                if r.identity == req.kv_control_policy_identity
            ),
            None,
        )
        available = self.control.executor.transfers._gpu_available()
        if observed is None or observed.estimated_incremental_tokens is None:
            for work in matching:
                self.cancel(work, "unknown_admission_demand")
            return True
        if (
            not all(w.ready for w in matching)
            and observed.estimated_incremental_tokens > available
        ):
            return False
        for work in matching:
            work.selected = req.kv_control_policy_identity
            work.deadline = (
                req.kv_control_policy_arrival
                + self.service.config.policy_max_wait_seconds
            )
            work.not_before = min(work.not_before, self.control._clock())
        self.progress()
        remaining = [
            w for w in matching if w.ref.operation_id in self.works and not w.cancelled
        ]
        return all(w.ready and w.lease is not None and w.lease.ready for w in remaining)

    def allows_candidate(self, req):
        if (
            self.runtime.waiting_seconds(req)
            >= self.service.config.policy_max_wait_seconds
        ):
            return True
        return not any(
            w.entry.proposal.prepare_for_admission
            and not w.cancelled
            and self._matches(w, req)
            and not w.ready
            for w in self.works.values()
        )

    def admitted(self, req):
        identity = getattr(req, "kv_control_policy_identity", None)
        for work in tuple(self.works.values()):
            if work.selected == identity and work.ready:
                if req.lock is not None:
                    self.stats["preparations_consumed"] += 1
                    self.stats["consumed_prefix_tokens"] += min(
                        req.prefix_len, scope_tokens(work.entry.proposal.scope)
                    )
                self._retire(work)

    def request_finished(self, req, *, success, context=None):
        identity = getattr(req, "kv_control_policy_identity", None)
        context = context or getattr(req, "kv_control_context", None)
        binding = (
            (context.caller_id, context.binding_id) if context is not None else None
        )
        for work in tuple(self.works.values()):
            if binding is not None and work.binding == binding and not success:
                self.cancel(work, "request_failed")
            if work.selected == identity and identity is not None:
                self.cancel(work, "request_finished")
            if binding is not None and work.wait_binding == binding:
                if success:
                    work.wait_binding = None
                    work.deadline = (
                        self.control._clock()
                        + self.service.config.policy_max_delay_seconds
                    )
                else:
                    self.cancel(work, "request_failed")
        self.progress()

    def cleanup(self, tasks):
        removed = frozenset(tasks)
        for work in tuple(self.works.values()):
            if removed.intersection(work.tasks):
                self.cancel(work, "task_finished")
        for key, (_, owners) in tuple(self.paused.items()):
            if removed.intersection(owners):
                self.paused.pop(key)
        self.reap()

    def cancel_all(self):
        for work in tuple(self.works.values()):
            self.cancel(work, "owner_reset")
        self.reap()

    def before_cancel(self, ref):
        work = self.works.get(ref.operation_id)
        if work is not None and not self.control.registry.get(ref).status.terminal:
            self.cancel(work, "caller_cancelled")
