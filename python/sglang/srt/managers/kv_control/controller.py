# SPDX-License-Identifier: Apache-2.0
"""Callable native control core; serving/cache integration supplies Executor."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    PROTOCOL_VERSION,
    Associate,
    ContentTarget,
    ControlContext,
    ControlError,
    CopyTo,
    ExecutionUpdate,
    FinishTask,
    Limits,
    LogicalChange,
    MoveTo,
    OperationRef,
    OperationResult,
    Pin,
    Prefetch,
    Primitive,
    Signal,
    Status,
    Submission,
    TaskIdentity,
    TokensTarget,
    Value,
    parse_action,
    validated,
)
from sglang.srt.managers.kv_control.index import LogicalIndex, ResolvedTarget
from sglang.srt.managers.kv_control.public import (
    PUBLIC_ACTIONS,
    normalize_public,
    requested_outcomes,
)
from sglang.srt.managers.kv_control.registry import OperationRegistry, snapshot_envelope
from sglang.srt.managers.kv_hints import KvHintsEnvelope


class ExecutionRequest(Value):
    ref: OperationRef
    payload: Primitive
    target: ResolvedTarget
    max_result_targets: int = 4096
    # Owner-selected semantics, never a caller-controlled primitive field.
    public_demote: bool = False


class Executor(Protocol):
    """Scheduler-owned physical boundary, called only on the owner thread.

    start revalidates namespace, component/layout support, actual copies and
    current capacity, then acquires real allocator reservations/transfer holds.
    It may return DEFERRED/UNSUPPORTED/MISSING without submitting physical work.
    ACCEPTED/RUNNING certify approval, never successful publication. All updates
    are cumulative. Terminal updates certify quiescence of this caller's resource
    obligations; shared jobs keep their own holds until all worker access stops.

    COPY requires destination publication after every required component/rank;
    MOVE reports destination publication and source reclamation separately.
    Cancelling work drains outstanding backend access before releasing holds.
    """

    def start(self, work: ExecutionRequest) -> ExecutionUpdate: ...

    def poll(self, ref: OperationRef) -> ExecutionUpdate | None: ...

    def cancel(self, ref: OperationRef) -> ExecutionUpdate | None: ...


@dataclass
class _Pending:
    ref: OperationRef
    payload: Primitive | Pin
    target: ResolvedTarget | None
    started: bool = False
    cancel_sent: bool = False
    # Keep a bounded cause, never an exception/traceback retaining worker state.
    failure: tuple[str, str] | None = None
    deadline: float | None = None


class KVControl:
    """Opt-in, mechanism-only core with no scheduler, policy or backend imports."""

    def __init__(
        self,
        engine_id: str,
        namespace: str,
        *,
        enabled: bool = False,
        executor: Executor | None = None,
        limits: Limits = Limits(),
        clock: Callable[[], float] = time.monotonic,
    ):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        self.enabled = enabled
        self.executor = executor
        self.limits = limits
        self._clock = clock
        self.registry = OperationRegistry(
            engine_id, namespace, limits=limits, clock=clock
        )
        self.index = LogicalIndex(namespace, self.registry.engine_epoch, limits=limits)
        self._pending: dict[str, _Pending] = {}
        self._requests: dict[tuple[str, str], set[str]] = {}
        self.association_feedback = None

    def context(
        self, caller_id: str, *, binding_id: str | None = None
    ) -> ControlContext:
        return self.registry.validate_context(
            ControlContext(
                engine_id=self.registry.engine_id,
                engine_epoch=self.registry.engine_epoch,
                namespace=self.registry.namespace,
                caller_id=caller_id,
                binding_id=binding_id,
            )
        )

    def submit(
        self,
        envelope: KvHintsEnvelope,
        context: ControlContext,
        *,
        execute_primitives: bool = True,
        public_only: bool = False,
    ) -> Submission:
        if type(execute_primitives) is not bool:
            raise ControlError("invalid_payload", "Execution support must be a bool")
        batch, unhandled = snapshot_envelope(envelope, self.limits)
        if not batch.actions:
            return Submission(replayed=False, unhandled_actions=unhandled)
        submission = self.registry.reserve(batch, context)
        if submission.replayed:
            return msgspec.structs.replace(submission, unhandled_actions=unhandled)
        # Resolve every historical selector against the state at submission,
        # before any sibling can mutate the index. List order is no dependency API.
        prepared = []
        public_payloads = {}
        for frozen in batch.actions:
            try:
                if not self.enabled:
                    raise ControlError("unsupported", "Native KV control is disabled")
                if batch.protocol_version != PROTOCOL_VERSION:
                    raise ControlError(
                        "unsupported", "Unsupported native protocol version"
                    )
                if public_only and frozen.action_type not in PUBLIC_ACTIONS:
                    raise ControlError("unsupported", "Use PIN, PREFETCH, or DEMOTE")
                payload = parse_action(frozen.thaw())
                if frozen.invalid_numbers:
                    raise ControlError(
                        "invalid_payload",
                        "JSON numbers must be finite and within the exact integer bound",
                    )
                scope = None
                if isinstance(payload, (Prefetch, Pin)):
                    public_payloads[frozen.action_id] = payload
                    if (
                        payload.target != "request_cached"
                        and payload.target.namespace != context.namespace
                    ):
                        raise ControlError(
                            "incompatible_namespace", "Target namespace differs"
                        )
                    if self.executor is None:
                        raise ControlError(
                            "unsupported", "No cache executor is installed"
                        )
                    describe = getattr(self.executor, "capabilities", None)
                    payload = normalize_public(payload, describe() if describe else {})
                if hasattr(payload, "target") and payload.target != "request_cached":
                    if payload.target.namespace != context.namespace:
                        raise ControlError(
                            "incompatible_namespace", "Target namespace differs"
                        )
                if not execute_primitives and isinstance(
                    payload, (CopyTo, MoveTo, Pin)
                ):
                    raise ControlError(
                        "unsupported",
                        "This request profile cannot execute native cache operations",
                    )
                if isinstance(payload, Pin):
                    if payload.location.engine_id not in (None, context.engine_id):
                        raise ControlError(
                            "unsupported", "Remote engines approve their own protection"
                        )
                    if payload.mode == "acquire":
                        if payload.target == "request_cached":
                            if context.binding_id is None:
                                raise ControlError(
                                    "invalid_payload",
                                    "request_cached requires a request incarnation",
                                )
                        else:
                            scope = self.index.resolve(payload.target)
                if isinstance(payload, (CopyTo, MoveTo)):
                    locations = (payload.src, payload.dst)
                    if any(
                        location.engine_id not in (None, context.engine_id)
                        for location in locations
                    ):
                        raise ControlError(
                            "unsupported",
                            "Remote engines must approve their own resources",
                        )
                    if (
                        isinstance(payload, (CopyTo, MoveTo))
                        and payload.src == payload.dst
                    ):
                        raise ControlError(
                            "invalid_payload", "Source and destination must differ"
                        )
                    if payload.target == "request_cached":
                        if context.binding_id is None:
                            raise ControlError(
                                "invalid_payload",
                                "request_cached requires a request incarnation",
                            )
                    else:
                        scope = self.index.resolve(payload.target)
                    if self.executor is None:
                        raise ControlError(
                            "unsupported", "No cache executor is installed"
                        )
                prepared.append((payload, scope))
            except ControlError as error:
                prepared.append(error)
        # A finish_task sibling must also see interests whose immutable targets
        # have been prepared but whose dispatch has not yet been visited.
        for result, item in zip(submission.results, prepared):
            if not isinstance(item, ControlError):
                payload, scope = item
                if isinstance(payload, (CopyTo, MoveTo)) or (
                    isinstance(payload, Pin)
                    and payload.mode == "acquire"
                    and payload.target == "request_cached"
                ):
                    public = public_payloads.get(result.ref.action_id)
                    self._pending[result.ref.operation_id] = _Pending(
                        result.ref,
                        payload,
                        scope,
                        deadline=self._clock() + public.timeout_ms / 1000
                        if isinstance(public, Prefetch)
                        and public.timeout_ms is not None
                        else None,
                    )
                    if public is not None:
                        self.registry.set_public_scope(
                            result.ref,
                            requested_outcomes(scope) if scope is not None else (),
                            self.executor.capabilities().get("cache_generation"),
                        )
        for result, item in zip(submission.results, prepared):
            if self.registry.get(result.ref).status.terminal:
                continue
            try:
                if isinstance(item, ControlError):
                    raise item
                payload, scope = item
                self._dispatch(result.ref, payload, scope)
            except ControlError as error:
                self._error(result.ref, error)
        return msgspec.structs.replace(
            self.registry.query(context, batch.message_id), unhandled_actions=unhandled
        )

    def _dispatch(self, ref, payload, scope):
        if isinstance(payload, Pin):
            if payload.mode == "acquire" and payload.target == "request_cached":
                key = ref.context.caller_id, ref.context.binding_id
                self._requests.setdefault(key, set()).add(ref.operation_id)
                self.registry.update(
                    ref,
                    ExecutionUpdate(
                        status=Status.QUEUED, quiescent=True, code="request_completion"
                    ),
                )
            else:
                self._execute_pin(ref, payload, scope)
        elif isinstance(payload, Signal):
            # No consumer: do not resolve a session or build observations.
            self.registry.update(
                ref,
                ExecutionUpdate(
                    status=Status.UNCONSUMED, quiescent=True, code="no_policy"
                ),
            )
        elif isinstance(payload, Associate):
            record = self.index.associate(payload)
            if self.association_feedback is not None:
                self.association_feedback(ref, record)
            change = LogicalChange(
                index_generation=self.index.generation,
                session_revision=self.index.session_revision(payload.session_id),
                continuation_revision=record.revision,
                incarnation=record.incarnation,
            )
            self.registry.update(
                ref,
                ExecutionUpdate(
                    status=Status.COMPLETED, quiescent=True, logical_change=change
                ),
            )
        elif isinstance(payload, FinishTask):
            change = self.index.finish_task(
                payload.session_id, expected_revision=payload.expected_revision
            )
            self._cancel_task_interests(change.removed_tasks)
            cleanup = getattr(self.executor, "release_task_protections", None)
            if cleanup is not None:
                cleanup(caller_id=ref.context.caller_id, tasks=change.removed_tasks)
            self.registry.update(
                ref,
                ExecutionUpdate(
                    status=Status.COMPLETED, quiescent=True, logical_change=change
                ),
            )
        else:
            pending = self._pending[ref.operation_id]
            wait = ref.context.binding_id is not None and (
                payload.target == "request_cached" or payload.src.tier == "gpu"
            )
            if wait:
                key = ref.context.caller_id, ref.context.binding_id
                self._requests.setdefault(key, set()).add(ref.operation_id)
                self.registry.update(
                    ref,
                    ExecutionUpdate(
                        status=Status.QUEUED, quiescent=True, code="request_completion"
                    ),
                )
            else:
                self._start(pending)

    def _execute_pin(self, ref, payload, scope):
        if (
            payload.owner_task is not None
            and payload.mode != "release"
            and self.index.task_identity(payload.owner_task.session_id)
            != payload.owner_task
        ):
            raise ControlError(
                "stale_revision", "PIN demand owner task is no longer live"
            )
        if scope is not None:
            self.registry.set_public_scope(
                ref,
                requested_outcomes(scope),
                self.executor.capabilities().get("cache_generation"),
            )
        update, claim = self.executor.pin(ref=ref, payload=payload, scope=scope)
        self._finish(ref, update)
        if claim is not None:
            self.registry.set_public_claim(ref, claim)

    def _cancel_task_interests(self, removed_tasks: tuple[TaskIdentity, ...]) -> None:
        removed = frozenset(removed_tasks)
        for pending in tuple(self._pending.values()):
            if pending.target is not None and any(
                branch.continuation is not None and branch.continuation.task in removed
                for branch in pending.target.branches
            ):
                # Content equality alone establishes no task ownership. cancel
                # retires queued work or asks the executor to drain started work.
                self.cancel(pending.ref)

    def request_event(
        self,
        context: ControlContext,
        event: Literal["success", "abort", "error", "preempt", "checkpoint"],
        *,
        committed: ContentTarget | TokensTarget | None = None,
    ) -> tuple[OperationResult, ...]:
        """success must run before request references are released.

        The owner supplies authoritative committed content for request_cached;
        its executor acquires separate holds synchronously in start(). This core
        never observes a live Req or guesses a generated suffix.
        """
        self.registry.validate_context(context)
        if context.binding_id is None or event not in (
            "success",
            "abort",
            "error",
            "preempt",
            "checkpoint",
        ):
            raise ControlError(
                "invalid_payload", "Expected a request incarnation and lifecycle event"
            )
        if event in ("preempt", "checkpoint"):
            return ()
        key = context.caller_id, context.binding_id
        ids = tuple(self._requests.pop(key, ()))
        results = []
        for operation_id in ids:
            pending = self._pending[operation_id]
            if event != "success":
                results.append(
                    self._finish(
                        pending.ref,
                        ExecutionUpdate(
                            status=Status.CANCELLED,
                            quiescent=True,
                            code="request_failed",
                        ),
                    )
                )
                continue
            try:
                if pending.target is None:
                    if committed is None:
                        raise ControlError(
                            "missing", "No committed request content supplied"
                        )
                    pending.target = self.index.resolve(committed)
                self._start(pending)
            except ControlError as error:
                self._error(pending.ref, error)
            results.append(self.registry.get(pending.ref))
        return tuple(results)

    def _start(self, pending):
        assert self.executor is not None and pending.target is not None
        if isinstance(pending.payload, Pin):
            self._execute_pin(pending.ref, pending.payload, pending.target)
            return
        if pending.deadline is not None and self._clock() >= pending.deadline:
            self.cancel(pending.ref)
            return
        current = self.registry.get(pending.ref)
        if (
            current.public_result is not None
            and not current.public_result.requested_scope
        ):
            self.registry.set_public_scope(
                pending.ref,
                requested_outcomes(pending.target),
                self.executor.capabilities().get("cache_generation"),
            )
        if (
            any(branch.unaligned for branch in pending.target.branches)
            and not pending.target.allow_clip
        ):
            raise ControlError(
                "unaligned", "Exact range splits an indivisible descriptor"
            )
        pending.started = True
        try:
            update = self.executor.start(
                ExecutionRequest(
                    ref=pending.ref,
                    payload=pending.payload,
                    target=pending.target,
                    max_result_targets=self.limits.max_scope_spans,
                    public_demote=current.action_type == "sglang.demote",
                )
            )
            self._execution_update(pending, update)
        except Exception as error:
            self._executor_error(pending, error)

    def _execution_update(self, pending, update):
        update = validated(update, ExecutionUpdate)
        current = self.registry.get(pending.ref)
        if current.status == Status.CANCELLING:
            update = self._merge_drain_progress(current.update, update)
        # Validate bounds before retaining a reported failure cause on _Pending.
        self.registry._validate_update(update)
        self._validate_execution(pending, update)
        if update.status == Status.RUNNING and current.status in (
            Status.RECEIVED,
            Status.QUEUED,
        ):
            self.registry.update(
                pending.ref, ExecutionUpdate(status=Status.ACCEPTED, quiescent=False)
            )
        if current.status == Status.CANCELLING:
            if update.status == Status.FAILED and pending.failure is None:
                pending.failure = (update.code, update.reason)
            update = msgspec.structs.replace(
                update,
                status=(Status.FAILED if pending.failure else Status.CANCELLED)
                if update.status.terminal
                else Status.CANCELLING,
            )
            if pending.failure is not None:
                update = msgspec.structs.replace(
                    update, code=pending.failure[0], reason=pending.failure[1]
                )
        self._finish(pending.ref, update)

    @staticmethod
    def _merge_drain_progress(previous, update):
        """A drain receipt can omit effects already observed before cancellation."""

        def steps(old, new):
            return tuple(dict.fromkeys((*old, *new)))

        def target_key(outcome):
            return (
                outcome.branch_index,
                outcome.coverage,
                outcome.content_id,
                outcome.location,
                outcome.coverage_kind,
                outcome.window_size,
            )

        outcomes = list(previous.target_outcomes)
        positions = {target_key(outcome): i for i, outcome in enumerate(outcomes)}
        for outcome in update.target_outcomes:
            key = target_key(outcome)
            position = positions.get(key)
            if position is not None:
                old = outcomes[position]
                if old.completed_steps and not outcome.completed_steps:
                    continue
                outcome = msgspec.structs.replace(
                    outcome,
                    completed_steps=steps(old.completed_steps, outcome.completed_steps),
                    status=old.status
                    if old.status == Status.COMPLETED
                    else outcome.status,
                )
                outcomes[position] = outcome
            else:
                positions[key] = len(outcomes)
                outcomes.append(outcome)
        # Registry validation still enforces the existing result/step limits.
        return msgspec.structs.replace(
            update,
            completed_steps=steps(previous.completed_steps, update.completed_steps),
            target_outcomes=tuple(outcomes),
        )

    @staticmethod
    def _validate_execution(pending, update):
        if update.logical_change is not None or update.derived_operations:
            raise ControlError(
                "invalid_completion",
                "Executors cannot report logical mutations or create child operations",
            )
        payload = pending.payload
        location = payload.dst
        required = (
            ("destination_ready", "source_reclaimed")
            if isinstance(payload, MoveTo)
            else ("destination_ready",)
        )
        successful = []
        if "source_reclaimed" in update.completed_steps and (
            not isinstance(payload, MoveTo)
            or "destination_ready" not in update.completed_steps
            or update.completed_steps.index("source_reclaimed")
            < update.completed_steps.index("destination_ready")
        ):
            raise ControlError(
                "invalid_completion",
                "Only MOVE can reclaim a source, after destination publication",
            )
        for outcome in update.target_outcomes:
            if outcome.branch_index >= len(pending.target.branches):
                raise ControlError(
                    "invalid_completion", "Unknown snapshot branch in result"
                )
            if not outcome.completed_steps:
                continue
            if "source_reclaimed" in outcome.completed_steps and (
                not isinstance(payload, MoveTo)
                or "destination_ready" not in outcome.completed_steps
                or outcome.completed_steps.index("source_reclaimed")
                < outcome.completed_steps.index("destination_ready")
            ):
                raise ControlError(
                    "invalid_completion",
                    "Source reclamation requires destination publication",
                )
            coverage = outcome.coverage
            actual = outcome.location
            if (
                actual is None
                or actual.tier != location.tier
                or actual.engine_id not in (None, pending.ref.context.engine_id)
                or (location.backend is not None and actual.backend != location.backend)
                or (
                    location.backend_namespace is not None
                    and actual.backend_namespace != location.backend_namespace
                )
                or coverage.component is None
            ):
                raise ControlError(
                    "invalid_completion",
                    "Actual coverage must identify the requested location and component",
                )
            in_scope = False
            for branch in (pending.target.branches[outcome.branch_index],):
                if isinstance(branch.content, TokensTarget):
                    in_scope |= (
                        branch.requested.start_token
                        <= coverage.start_token
                        <= coverage.end_token
                        <= min(
                            branch.requested.end_token, len(branch.content.token_ids)
                        )
                        and (
                            outcome.coverage_kind != "checkpoint"
                            or coverage.end_token > branch.requested.start_token
                        )
                    )
                else:
                    in_scope |= any(
                        span.component == coverage.component
                        and span.content_id == outcome.content_id
                        and span.coverage_kind == outcome.coverage_kind
                        and span.window_size == outcome.window_size
                        and span.start_token
                        <= coverage.start_token
                        <= coverage.end_token
                        <= span.end_token
                        for span in branch.spans
                    )
            if not in_scope:
                raise ControlError(
                    "invalid_completion", "Actual coverage exceeds the bound target"
                )
            successful.append(outcome)
        if update.status != Status.COMPLETED:
            return
        if not successful or not set(required).issubset(update.completed_steps):
            raise ControlError(
                "invalid_completion",
                "Completed primitive needs actual coverage and completed stages",
            )
        if any(
            branch.missing or branch.unaligned for branch in pending.target.branches
        ):
            raise ControlError(
                "invalid_completion",
                "Incomplete logical coverage must be reported as partial",
            )
        complete = [
            outcome
            for outcome in successful
            if outcome.status == Status.COMPLETED
            and set(required).issubset(outcome.completed_steps)
        ]

        def covers(branch_index, component, content_id, start, end):
            intervals = sorted(
                (o.coverage.start_token, o.coverage.end_token)
                for o in complete
                if o.branch_index == branch_index
                and o.coverage.component == component
                and (content_id is None or o.content_id == content_id)
            )
            if start == end:
                return (start, end) in intervals
            cursor = start
            for left, right in intervals:
                if left > cursor:
                    break
                cursor = max(cursor, right)
            return cursor >= end

        for branch_index, branch in enumerate(pending.target.branches):
            if isinstance(branch.content, TokensTarget):
                # The executor certifies its required layout. Tokens alone do
                # not enumerate components, but each reported one must be whole.
                components = {
                    outcome.coverage.component
                    for outcome in complete
                    if outcome.branch_index == branch_index
                }
                covered = bool(components)
                for component in components:
                    members = [
                        o
                        for o in complete
                        if o.branch_index == branch_index
                        and o.coverage.component == component
                    ]
                    kinds = {(o.coverage_kind, o.window_size) for o in members}
                    if len(kinds) != 1:
                        covered = False
                        break
                    kind, window = kinds.pop()
                    start, end = (
                        branch.requested.start_token,
                        branch.requested.end_token,
                    )
                    if kind == "checkpoint":
                        start = end
                    elif kind == "window":
                        start = max(start, end - window)
                    covered &= covers(branch_index, component, None, start, end)
            else:
                covered = bool(branch.spans) and all(
                    covers(
                        branch_index,
                        span.component,
                        span.content_id,
                        span.start_token,
                        span.end_token,
                    )
                    for span in branch.spans
                )
            if not covered:
                raise ControlError(
                    "invalid_completion",
                    "Completed result omits required logical coverage",
                )

    def _finish(self, ref, update):
        result = self.registry.update(ref, update)
        if result.status.terminal:
            self._pending.pop(ref.operation_id, None)
            key = ref.context.caller_id, ref.context.binding_id
            ids = self._requests.get(key)
            if ids is not None:
                ids.discard(ref.operation_id)
                if not ids:
                    del self._requests[key]
        return result

    def _error(self, ref, error):
        status = (
            Status.UNSUPPORTED
            if error.code in ("unsupported", "unaligned")
            else (Status.MISSING if error.code == "missing" else Status.REJECTED)
        )
        return self._finish(
            ref,
            ExecutionUpdate(
                status=status, quiescent=True, code=error.code, reason=str(error)[:4096]
            ),
        )

    def _executor_error(self, pending, error):
        current = self.registry.get(pending.ref)
        if pending.failure is None:
            pending.failure = (
                "executor_error",
                f"{type(error).__name__}: {error}"[:4096],
            )
        update = msgspec.structs.replace(
            current.update,
            status=Status.CANCELLING,
            quiescent=False,
            code=pending.failure[0],
            reason=pending.failure[1],
        )
        self.registry.update(pending.ref, update)

    def cancel(self, ref: OperationRef) -> OperationResult:
        current = self.registry.get(ref)  # Includes expected incarnation check.
        if current.status.terminal:
            return current
        pending = self._pending[ref.operation_id]
        if not pending.started:
            return self._finish(
                ref,
                ExecutionUpdate(
                    status=Status.CANCELLED,
                    quiescent=True,
                    code="cancelled_before_dispatch",
                ),
            )
        if pending.cancel_sent:
            return current
        if current.status != Status.CANCELLING:
            self.registry.update(
                ref,
                msgspec.structs.replace(
                    current.update, status=Status.CANCELLING, quiescent=False
                ),
            )
        pending.cancel_sent = True
        try:
            update = self.executor.cancel(ref)
            if update is not None:
                self._execution_update(pending, update)
        except Exception as error:
            self._executor_error(pending, error)
        return self.registry.get(ref)

    def progress(self, *, limit: int = 64) -> tuple[OperationResult, ...]:
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ControlError("invalid_payload", "Progress batch exceeds limit")
        if (
            self._pending
            and (begin_poll := getattr(self.executor, "begin_poll", None)) is not None
        ):
            try:
                begin_poll(limit=limit)
            except Exception as error:
                for pending in tuple(self._pending.values()):
                    if pending.started:
                        self._executor_error(pending, error)
        results = []
        # Rotate polled entries for bounded fair progress, including idle serving.
        for operation_id in tuple(self._pending):
            pending = self._pending[operation_id]
            if pending.deadline is not None and self._clock() >= pending.deadline:
                self.cancel(pending.ref)
                if operation_id not in self._pending:
                    results.append(self.registry.get(pending.ref))
                    if len(results) >= limit:
                        break
                    continue
            if not pending.started:
                continue
            self._pending.pop(operation_id)
            self._pending[operation_id] = pending
            try:
                if (
                    self.registry.get(pending.ref).status == Status.CANCELLING
                    and not pending.cancel_sent
                ):
                    self.cancel(pending.ref)
                if operation_id in self._pending:
                    update = self.executor.poll(pending.ref)
                    if update is not None:
                        self._execution_update(pending, update)
            except Exception as error:
                self._executor_error(pending, error)
            results.append(self.registry.get(pending.ref))
            if len(results) >= limit:
                break
        return tuple(results)

    def reset(self, namespace: str) -> None:
        """Rotate logical generation after the owner has drained/reset real caches."""
        if self.registry.capacity()["active"]:
            raise ControlError(
                "busy", "Drain active operations before rotating the epoch"
            )
        registry = OperationRegistry(
            self.registry.engine_id, namespace, limits=self.limits, clock=self._clock
        )
        index = LogicalIndex(namespace, registry.engine_epoch, limits=self.limits)
        self.registry, self.index = registry, index
