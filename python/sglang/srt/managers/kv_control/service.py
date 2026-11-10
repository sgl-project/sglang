# SPDX-License-Identifier: Apache-2.0
"""Typed, scheduler-owned control service; no model request is needed for RPCs."""

from __future__ import annotations

import atexit
import hashlib
import threading
import time
import uuid
from typing import Annotated

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    CacheWatermark,
    ControlContext,
    ControlError,
    Identifier,
    Limits,
    Location,
    OperationRef,
    Position,
    PublicLocation,
    PublicTarget,
    TaskIdentity,
    TokensTarget,
    Value,
    resolve_prefix,
    validated,
)
from sglang.srt.managers.kv_control.controller import KVControl
from sglang.srt.managers.kv_control.feedback import (
    FeedbackCursor,
    FeedbackJournal,
    FeedbackLimits,
    FeedbackPublisherConfig,
    RequestAssociation,
)
from sglang.srt.managers.kv_hints import KvHintsEnvelope
from sglang.srt.mem_cache.unified_cache.control_protection import (
    HostCacheProtection,
    ProtectionLimits,
)
from sglang.srt.mem_cache.unified_cache.control_state import CacheControlConfig


class CacheLimits(Value):
    max_query_nodes: int = 4096
    max_result_targets: int = 4096
    max_prepared_prefixes: int = 128
    max_concurrent_storage_reads: Annotated[int, msgspec.Meta(ge=1, le=64)] = 4
    # Zero disables the optional native admission budget. Ordinary restore
    # occupancy counts against it, but ordinary inference is never gated here.
    max_storage_read_inflight_tokens: Annotated[int, msgspec.Meta(ge=0, le=1 << 40)] = 0
    max_speculative_storage_read_tokens: Annotated[
        int, msgspec.Meta(ge=0, le=1 << 40)
    ] = 0
    demand_host_reserve_tokens: Annotated[int, msgspec.Meta(ge=0, le=1 << 40)] = 0
    storage_demand_priority: bool = False


class ServiceConfig(Value):
    namespace: Identifier
    engine_id: Identifier | None = None
    limits: Limits = Limits()
    cache_limits: CacheLimits = CacheLimits()
    protection_limits: ProtectionLimits = ProtectionLimits()
    feedback_limits: FeedbackLimits = FeedbackLimits()
    feedback_publisher: FeedbackPublisherConfig | None = None

    def cache_config(self):
        return CacheControlConfig(
            namespace=self.namespace, **msgspec.to_builtins(self.cache_limits)
        )


def parse_config(value):
    if value is None:
        return None
    config = validated(value, ServiceConfig)
    config.cache_config()
    return config


def runtime_config():
    from sglang.srt.runtime_context import (
        get_disagg,
        get_memory,
        get_parallel,
        get_schedule,
        get_spec,
    )

    config = parse_config(get_memory().kv_control_config)
    if config is None:
        return None
    parallel = get_parallel()
    if any(
        size != 1
        for size in (
            parallel.tp_size,
            parallel.pp_size,
            parallel.attn_cp_size,
            parallel.num_dp_ranks,
            parallel.nnodes,
        )
    ):
        raise ValueError(
            "KV control currently requires one TP/PP/CP/DP owner on one node"
        )
    if get_spec().speculative_algorithm or get_disagg().disaggregation_mode != "null":
        raise ValueError(
            "KV control does not support speculative decoding or PD disaggregation"
        )
    memory = get_memory()
    if memory.enable_hierarchical_cache and not get_schedule().disable_overlap_schedule:
        raise ValueError(
            "Native host transfers currently require --disable-overlap-schedule"
        )
    if (
        memory.disable_radix_cache
        or memory.radix_cache_backend
        or memory.enable_lmcache
        or memory.enable_flexkv
    ):
        raise ValueError("KV control requires the enabled built-in unified radix cache")
    return config


PageSize = Annotated[int, msgspec.Meta(ge=1, le=256)]


class Capabilities(Value, tag="capabilities", tag_field="kind"):
    pass


class Submit(Value, tag="submit", tag_field="kind"):
    context: ControlContext
    envelope: KvHintsEnvelope


class Query(Value, tag="query", tag_field="kind"):
    context: ControlContext
    message_id: Identifier


class Lookup(Value, tag="lookup", tag_field="kind"):
    context: ControlContext
    message_id: Identifier
    limit: PageSize = 256


class Ack(Value, tag="ack", tag_field="kind"):
    context: ControlContext
    message_id: Identifier


class Cancel(Value, tag="cancel", tag_field="kind"):
    ref: OperationRef


class Residency(Value, tag="residency", tag_field="kind"):
    context: ControlContext
    target: TokensTarget
    limit: PageSize = 256
    location: Location = Location(tier="gpu")


class StorageTargets(Value, tag="storage_targets", tag_field="kind"):
    context: ControlContext
    target: PublicTarget
    location: PublicLocation


class Capacity(Value, tag="capacity", tag_field="kind"):
    context: ControlContext


class Claim(Value, tag="claim", tag_field="kind"):
    context: ControlContext
    claim_id: Identifier
    owner_task: TaskIdentity | None = None


class ClaimEvents(Value, tag="claim_events", tag_field="kind"):
    context: ControlContext
    after_sequence: Position = 0
    limit: PageSize = 256


class Events(Value, tag="events", tag_field="kind"):
    context: ControlContext
    cursor: FeedbackCursor | None = None
    limit: PageSize = 256


class BindRequest(Value, tag="bind_request", tag_field="kind"):
    context: ControlContext
    binding_id: Identifier
    association: RequestAssociation


Command = (
    Capabilities
    | Submit
    | Query
    | Lookup
    | Ack
    | Cancel
    | Residency
    | StorageTargets
    | Capacity
    | Claim
    | ClaimEvents
    | Events
    | BindRequest
)


class KVControlService:
    def __init__(
        self,
        cache,
        config: ServiceConfig,
        *,
        clock=time.monotonic,
        namespace_keys=False,
    ):
        if type(namespace_keys) is not bool:
            raise ValueError("namespace_keys must be a bool")
        self.namespace_keys = namespace_keys
        config = parse_config(config)
        if config is None:
            raise ValueError("An enabled KV service requires configuration")
        self.cache, self.config = cache, config
        if self.namespace_keys:
            self.cache_salt(None)  # Reject an unencodable namespace at startup.
        self.control = KVControl(
            config.engine_id or uuid.uuid4().hex,
            config.namespace,
            enabled=True,
            limits=config.limits,
            clock=clock,
        )
        self.control.executor = self._bind_executor()
        capabilities = self.control.executor.capabilities()
        if capabilities["reason"]:
            raise ValueError(
                "KV control cache is unsupported: " + capabilities["reason"]
            )
        cache.kv_control_service = self
        self.feedback = None
        self._bind_feedback()
        if config.feedback_publisher is not None:
            atexit.register(self.close)

    def _bind_feedback(self):
        registry = self.control.registry
        if self.feedback is not None:
            self.feedback.close()
        self.feedback = FeedbackJournal(
            engine_id=registry.engine_id,
            engine_epoch=registry.engine_epoch,
            namespace=registry.namespace,
            limits=self.config.feedback_limits,
            watermark=self.cache_watermark,
            publisher_config=self.config.feedback_publisher,
            clock=self.control._clock,
        )
        registry.on_result = self.feedback.operation
        registry.cache_watermark = self.cache_watermark
        self._feedback_backend = self._backend_identity()

    def _backend_identity(self):
        controller = self.cache.cache_controller
        store = controller.storage_backend if controller is not None else None
        adapter = self.control.executor.storage_adapter
        return (
            self.cache.enable_storage,
            store,
            adapter.backend_namespace
            if self.cache.enable_storage
            and adapter is not None
            and adapter.store is store
            else None,
        )

    def cache_watermark(self):
        recorder = self.cache.tree_core.kv_events
        return CacheWatermark(
            source_id="cache:"
            + hashlib.sha256(self.control.registry.engine_id.encode()).hexdigest(),
            source_epoch=recorder.state_epoch,
            generation=recorder.state_generation,
            version=recorder.state_version,
        )

    def _feedback_context(self, req):
        context = getattr(req, "kv_control_feedback_context", None)
        if (
            context is None
            or context.engine_epoch != self.control.registry.engine_epoch
        ):
            context = self.control.context("request", binding_id=uuid.uuid4().hex)
            req.kv_control_feedback_context = context
            self.feedback.register_request(context, req.rid)
        return context

    def _bind_executor(self):
        executor = self.cache.create_kv_control_executor(
            engine_id=self.control.registry.engine_id,
            engine_epoch=self.control.registry.engine_epoch,
        )
        if self.namespace_keys:
            executor.cache_salt_transform = self.cache_salt
        if executor.host_control:
            executor.protection = HostCacheProtection(
                executor,
                limits=self.config.protection_limits,
                clock=self.control._clock,
                max_result_targets=self.config.limits.max_scope_spans,
            )
        return executor

    def cache_salt(self, original):
        from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt

        return encode_cache_salt(self.config.namespace, original)

    def prepare_request(self, req):
        self._feedback_context(req)
        if not self.namespace_keys:
            return
        if hasattr(req, "kv_control_logical_cache_salt"):
            if req.kv_control_key_namespace != self.config.namespace:
                raise ControlError(
                    "stale_epoch", "Request key belongs to an older model namespace"
                )
            return
        original = req.cache_salt
        encoded = self.cache_salt(original)
        req.kv_control_logical_cache_salt = original
        req.kv_control_key_namespace = self.config.namespace
        req.cache_salt = encoded

    def capabilities(self):
        from sglang.srt.managers.kv_control.key_namespace import SALT_ENCODING
        from sglang.srt.managers.kv_control.public import public_capabilities

        registry = self.control.registry
        executor = self.control.executor.capabilities()
        public = public_capabilities(executor)
        return {
            "enabled": True,
            "public_protocol": public,
            "feedback": {
                **self.feedback.describe(),
                "cache_watermark": self.cache_watermark(),
            },
            "engine_id": registry.engine_id,
            "engine_epoch": registry.engine_epoch,
            "namespace": registry.namespace,
            "executor": {**executor, "actions": tuple(public["actions"])},
            "topology": "single_owner",
            "attached_caller_id": "request",
            "weight_updates": "unsupported",
            "cache_salt_namespace": registry.namespace if self.namespace_keys else None,
            "cache_salt_encoding": SALT_ENCODING
            if self.namespace_keys
            else "owner_assertion",
        }

    @property
    def has_pending(self):
        reads = self.control.executor.storage_reads
        return self.control.registry.capacity()["active"] != 0 or (
            reads is not None and bool(reads.demands)
        )

    @property
    def has_background_work(self):
        protection = self.control.executor.protection
        return (
            self.has_pending
            or self.feedback.has_outbox
            or (protection is not None and protection.has_background_work)
        )

    @property
    def next_deadline(self):
        protection = self.control.executor.protection
        deadlines = (
            self.feedback.next_deadline,
            protection.next_deadline if protection is not None else None,
        )
        return min(
            (deadline for deadline in deadlines if deadline is not None), default=None
        )

    def invalidate_protection(self, *, reason):
        protection = self.control.executor.protection
        self.feedback.append(
            "invalidation",
            None,
            {
                "reason": reason,
                "tiers": ("gpu", "host"),
                "semantics": "owner_local_only",
            },
        )
        changes = protection.invalidate(reason=reason) if protection is not None else ()
        for claim in changes:
            self.feedback.append("claim", claim.caller_id, claim)
        if reason == "host_cache_detached":
            self.close()
        return changes

    def close(self):
        if self.feedback is not None:
            self.feedback.close()
        atexit.unregister(self.close)

    def progress_protection(self, *, limit=64):
        protection = self.control.executor.protection
        if protection is not None:
            for claim in protection.progress(limit=limit):
                self.feedback.append("claim", claim.caller_id, claim)

    def host_protection_evicted(self, values):
        protection = self.control.executor.protection
        if protection is not None and protection.registry.priorities:
            for claim in protection.host_evicted(values):
                self.feedback.append("claim", claim.caller_id, claim)

    def progress(self):
        backend = self._backend_identity()
        if (
            backend[0] != self._feedback_backend[0]
            or backend[1] is not self._feedback_backend[1]
            or backend[2] != self._feedback_backend[2]
        ):
            self.feedback.append(
                "invalidation",
                None,
                {
                    "reason": "storage_attachment_changed",
                    "tiers": ("storage",),
                    "semantics": "owner_storage_observations_only",
                    "previous_backend_namespace": self._feedback_backend[2],
                    "backend_namespace": backend[2],
                },
            )
            self._feedback_backend = backend

        if self.has_pending:
            self.control.progress()
        self.progress_protection()
        self.feedback.flush()

    def reset_after_cache_flush(self):
        self.control.reset(self.config.namespace)
        self.control.executor = self._bind_executor()
        self._bind_feedback()

    def attach(self, req, envelope, *, unsupported_request=False):
        if envelope is None or not any(
            a.action_type.startswith("sglang.") for a in envelope.actions
        ):
            return
        context = self._feedback_context(req)
        # The request retains only a context, never registry/worker objects. A
        # reused rid or another sample gets a different incarnation.
        unsupported = unsupported_request or (
            getattr(req, "session", None) is not None and req.session.streaming
        )
        unsupported |= getattr(req, "lora_id", None) is not None
        unsupported |= getattr(req, "multimodal_inputs", None) is not None
        # Refuse this request's physical work without removing the owner that
        # must still cancel/drain existing transfers on logical task cleanup.
        submission = self.control.submit(
            envelope, context, execute_primitives=not unsupported
        )
        req.kv_control_context = context
        req.kv_hints = msgspec.structs.replace(
            envelope, actions=list(submission.unhandled_actions)
        )

    def finish_request(
        self, req, *, committed_len=0, checkpointed=False, aborted=False
    ):
        context = getattr(req, "kv_control_context", None)
        terminal = aborted or (getattr(req, "finished", lambda: False)())
        if not terminal:
            return
        feedback_context = getattr(req, "kv_control_feedback_context", None)
        if context is None and feedback_context is None:
            return
        if context is not None:
            req.kv_control_context = None
        active_context = context or feedback_context
        if active_context.engine_epoch != self.control.registry.engine_epoch:
            return
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        success = (
            checkpointed
            and req.finished()
            and not isinstance(req.finished_reason, FINISH_ABORT)
        )
        committed = None
        if (
            success
            and committed_len
            and (
                context is not None
                or committed_len <= self.config.limits.max_scope_tokens
            )
        ):
            committed = TokensTarget(
                namespace=self.config.namespace,
                token_ids=tuple(req.full_untruncated_fill_ids[:committed_len]),
                cache_salt=getattr(
                    req, "kv_control_logical_cache_salt", req.cache_salt
                ),
                extra_key=req.extra_key,
                is_multimodal=req.multimodal_inputs is not None,
                allow_clip=True,
            )
        if feedback_context is not None:
            self.feedback.finish_request(
                feedback_context,
                req,
                checkpointed=success and not aborted,
                aborted=aborted,
                committed=committed,
                committed_tokens=committed_len,
            )
            req.kv_control_feedback_context = None
        if context is not None:
            self.control.request_event(
                context,
                "success" if success and not aborted else "abort",
                committed=committed,
            )

    def execute(self, command):
        if threading.get_ident() != self.cache._kv_control_owner_thread:
            raise ControlError(
                "wrong_owner", "Control commands must run on the cache owner"
            )
        control, registry = self.control, self.control.registry
        if isinstance(command, Capabilities):
            return self.capabilities()
        context = (
            command.ref.context if isinstance(command, Cancel) else command.context
        )
        registry.validate_context(context)
        if isinstance(command, Events):
            self.progress()
            return self.feedback.query(
                caller_id=context.caller_id, cursor=command.cursor, limit=command.limit
            )
        if isinstance(command, BindRequest):
            return self.feedback.bind_request(
                context, command.binding_id, command.association
            )
        if isinstance(command, StorageTargets):
            if command.target == "request_cached" or command.location.tier != "storage":
                raise ControlError(
                    "invalid_payload",
                    "Storage descriptions require a definite target and storage location",
                )
            return control.executor.storage_targets(command.target, command.location)
        if isinstance(command, Submit):
            if context.binding_id is not None:
                raise ControlError(
                    "invalid_payload",
                    "Independent submission cannot bind an inference request",
                )
            # Independent control has no external consumer. Opaque actions
            # remain in the reply for their owner to handle.
            return control.submit(command.envelope, context)
        if isinstance(command, Query):
            return registry.query(context, command.message_id)
        if isinstance(command, Lookup):
            return registry.lookup(context, command.message_id, limit=command.limit)
        if isinstance(command, Ack):
            return registry.acknowledge(context, command.message_id)
        if isinstance(command, Cancel):
            return control.cancel(command.ref)
        if isinstance(command, (Claim, ClaimEvents)):
            protection = control.executor.protection
            if protection is None:
                raise ControlError("unsupported", "L2 protection is unavailable")
            if context.binding_id is not None:
                raise ControlError(
                    "invalid_payload",
                    "Claim queries require an independent caller context",
                )
            if isinstance(command, Claim):
                return protection.get(
                    caller_id=context.caller_id,
                    claim_id=command.claim_id,
                    owner_task=command.owner_task,
                )
            return protection.changes(
                caller_id=context.caller_id,
                after_sequence=command.after_sequence,
                limit=command.limit,
            )
        if isinstance(command, Residency):
            snapshot = control.executor.residency(
                resolve_prefix(command.target, context, self.config.limits),
                limit=command.limit,
                location=command.location,
            )
            return {
                **msgspec.to_builtins(snapshot),
                "cache_watermark": self.cache_watermark(),
                "coverage_semantics": "requested_scope_only",
            }
        return {
            "operations": registry.capacity(),
            "storage_reads": self.cache.prefetch_outcome_stats_snapshot()
            if control.executor.storage_reads is not None
            else None,
            "protection": {
                "active_claims": control.executor.protection.registry.active_claims,
                "protected_tokens": control.executor.protection.registry.protected_tokens,
            }
            if control.executor.protection is not None
            else None,
        }


def dispatch(service, data):
    """Bounded JSON-compatible reply used by typed IPC and in-process callers."""
    try:
        if len(msgspec.json.encode(data)) > 1 << 20:
            raise ControlError("capacity", "Control request exceeds byte limit")
        command = validated(data, Command)
        if service is None:
            if isinstance(command, Capabilities):
                return {"success": True, "data": {"enabled": False, "actions": []}}
            raise ControlError("unsupported", "Native KV control is disabled")
        return {"success": True, "data": msgspec.to_builtins(service.execute(command))}
    except (ControlError, ValueError, TypeError, RecursionError) as error:
        return {
            "success": False,
            "error": {
                "code": getattr(error, "code", "invalid_payload"),
                "reason": str(error)[:512],
            },
        }
