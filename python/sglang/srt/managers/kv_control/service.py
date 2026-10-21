# SPDX-License-Identifier: Apache-2.0
"""Typed, scheduler-owned control service; no model request is needed for RPCs."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace
from typing import Annotated

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ControlContext,
    ControlError,
    Identifier,
    Limits,
    Location,
    LogicalTarget,
    OperationRef,
    TokensTarget,
    Value,
    validated,
)
from sglang.srt.managers.kv_control.controller import KVControl
from sglang.srt.managers.kv_control.index import IndexCursor
from sglang.srt.managers.kv_control.policy_runtime import (
    PolicyDefinition,
    validate_definitions,
)
from sglang.srt.managers.kv_hints import KvHintsEnvelope
from sglang.srt.mem_cache.unified_cache.priority_claims import CacheControlConfig


class CacheLimits(Value):
    max_copy_units: int = 65536
    max_claims_per_copy: int = 16
    max_policy_claims_per_copy: int = 4
    max_query_nodes: int = 4096
    max_result_targets: int = 4096


class ServiceConfig(Value):
    namespace: Identifier
    engine_id: Identifier | None = None
    limits: Limits = Limits()
    cache_limits: CacheLimits = CacheLimits()
    policies: tuple[PolicyDefinition, ...] = ()

    def cache_config(self):
        return CacheControlConfig(
            namespace=self.namespace, **msgspec.to_builtins(self.cache_limits)
        )


def parse_config(value):
    if value is None:
        return None
    config = validated(value, ServiceConfig)
    config.cache_config()
    validate_definitions(config.policies)
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
    target: LogicalTarget
    limit: PageSize = 256
    location: Location = Location(tier="gpu")


class Associations(Value, tag="associations", tag_field="kind"):
    context: ControlContext
    session_id: Identifier | None = None
    cursor: IndexCursor | None = None
    limit: PageSize = 256


class Capacity(Value, tag="capacity", tag_field="kind"):
    context: ControlContext


Command = (
    Capabilities
    | Submit
    | Query
    | Lookup
    | Ack
    | Cancel
    | Residency
    | Associations
    | Capacity
)


class KVControlService:
    def __init__(
        self,
        cache,
        config: ServiceConfig,
        *,
        clock=time.monotonic,
        namespace_keys=False,
        idle_check=None,
        weight_version=None,
        cache_available=None,
    ):
        if type(namespace_keys) is not bool:
            raise ValueError("namespace_keys must be a bool")
        self.namespace_keys = namespace_keys
        self.idle_check = idle_check
        self.cache_available = cache_available
        self.cache, self.config = cache, validated(config, ServiceConfig)
        validate_definitions(self.config.policies)
        self.base_namespace = self.config.namespace
        if self.namespace_keys:
            self.cache_salt(None)  # Reject an unencodable namespace at startup.
        from sglang.srt.managers.kv_control.weight_lifecycle import WeightLifecycle

        self.weights = WeightLifecycle(self, weight_version)
        self.control = KVControl(
            config.engine_id or uuid.uuid4().hex,
            config.namespace,
            enabled=True,
            limits=config.limits,
            clock=clock,
        )
        self.control.executor = self._bind_executor()
        capabilities = self.control.executor.capabilities()
        if not capabilities["actions"]:
            raise ValueError(
                "KV control cache is unsupported: " + capabilities["reason"]
            )
        cache.kv_control_service = self
        self._bind_policy_runtime()

    def _bind_policy_runtime(self):
        from sglang.srt.managers.kv_control.policy_runtime import KVPolicyRuntime

        self.policy_runtime = KVPolicyRuntime(self) if self.config.policies else None
        self.control.policy_runtime = self.policy_runtime

    def _bind_executor(self, capacity=None):
        executor = self.cache.create_kv_control_executor(
            engine_id=self.control.registry.engine_id,
            engine_epoch=self.control.registry.engine_epoch,
        )
        executor.gpu_copy_capacity = capacity
        executor.cache_available = self.cache_is_available
        if self.namespace_keys:
            executor.cache_salt_transform = self.cache_salt
        return executor

    def cache_is_available(self):
        return self.cache_available is None or self.cache_available()

    def cache_salt(self, original):
        from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt

        return encode_cache_salt(self.config.namespace, original)

    def prepare_request(self, req):
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

    def rebind_namespace(self, namespace):
        """Owner boundary after draining model work; never mutates parameters.

        Old physical copies may remain under their old encoded keys. Logical
        results/declarations/claims rotate, and fresh ordinary/store lookup uses
        the new namespace even when a caller deliberately retains old buffers.
        """
        if threading.get_ident() != self.cache._kv_control_owner_thread:
            raise ControlError(
                "wrong_owner", "Namespace rotation must run on the cache owner"
            )
        namespace = validated(namespace, Identifier)
        if self.weights.active and not self.weights.fence.quiescent:
            raise ControlError("busy", "Parameter work is not quiescent")
        if not self.namespace_keys or self.idle_check is None:
            raise ControlError(
                "unsupported",
                "Namespace rotation requires namespaced request keys and an owner idle check",
            )
        if namespace == self.config.namespace:
            raise ControlError("conflict", "A changed model requires a fresh namespace")
        self.require_quiescent_cache()
        capacity = self.control.executor.gpu_copy_capacity
        self.control.reset(namespace)
        self.config = msgspec.structs.replace(self.config, namespace=namespace)
        claims = self.cache.tree_core.priority_claims
        claims.config = replace(claims.config, namespace=namespace)
        claims.reset()
        self.control.executor = self._bind_executor(capacity)
        self._bind_policy_runtime()

    def require_quiescent_cache(self):
        if threading.get_ident() != self.cache._kv_control_owner_thread:
            raise ControlError(
                "wrong_owner", "Cache transitions must run on their owner"
            )
        if self.idle_check is None:
            raise ControlError("unsupported", "An owner idle check is required")
        if self.has_pending or not self.idle_check():
            raise ControlError(
                "busy",
                "Drain inference and control operations before namespace rotation",
            )
        if self.cache.session.any_holding_kv():
            raise ControlError(
                "busy", "Close StreamingSessions holding KV before namespace rotation"
            )
        if any(
            getattr(self.cache, name, {})
            for name in (
                "ongoing_write_through",
                "ongoing_load_back",
                "ongoing_backup",
                "ongoing_prefetch",
            )
        ):
            raise ControlError(
                "busy", "Drain cache transfers before namespace rotation"
            )
        controller = self.cache.cache_controller
        if controller is not None and controller.has_inflight_device_transfers():
            raise ControlError("busy", "Cache controller still owns transfer buffers")
        for node in self.cache.tree_core._node_arena.values():
            if node is not self.cache.tree_core.root_node and any(
                cd.lock_ref or cd.host_lock_ref for cd in node.component_data
            ):
                raise ControlError(
                    "busy", "Cache content still has active correctness references"
                )

    def capabilities(self):
        from sglang.srt.managers.kv_control.key_namespace import SALT_ENCODING

        registry = self.control.registry
        return dict(
            enabled=True,
            engine_id=registry.engine_id,
            engine_epoch=registry.engine_epoch,
            namespace=registry.namespace,
            executor=self.control.executor.capabilities(),
            policies=tuple(p.name for p in self.config.policies),
            topology="single_owner",
            attached_caller_id="request",
            weight_updates="guarded_transactions"
            if self.namespace_keys
            else "unsupported",
            weight_version=self.weights.version,
            weight_update_devices=("cpu", "cuda") if self.namespace_keys else (),
            updating_weights=self.weights.active,
            cache_salt_namespace=registry.namespace if self.namespace_keys else None,
            cache_salt_encoding=SALT_ENCODING
            if self.namespace_keys
            else "owner_assertion",
        )

    @property
    def has_pending(self):
        return self.control.registry.capacity()["active"] != 0

    @property
    def blocks_prefill(self):
        transfers = self.control.executor.transfers
        return transfers is not None and transfers.blocks_prefill

    def progress(self):
        if self.has_pending:
            self.control.progress()

    def reset_after_cache_flush(self):
        capacity = self.control.executor.gpu_copy_capacity
        self.control.reset(self.config.namespace)
        self.control.executor = self._bind_executor(capacity)
        self._bind_policy_runtime()

    def attach(self, req, envelope, *, unsupported_request=False):
        if envelope is None or not any(
            a.action_type.startswith("sglang.") for a in envelope.actions
        ):
            return
        context = self.control.context("request", binding_id=uuid.uuid4().hex)
        # The request retains only a context, never registry/worker objects. A
        # reused rid or another sample gets a different incarnation.
        unsupported = unsupported_request or (
            getattr(req, "session", None) is not None and req.session.streaming
        )
        unsupported |= getattr(req, "lora_id", None) is not None
        unsupported |= getattr(req, "multimodal_inputs", None) is not None
        if unsupported:
            # Preserve ordinary inference, but give every native sibling an
            # inspectable unsupported result. Opaque consumers retain ownership.
            executor = self.control.executor
            self.control.executor = None
            try:
                submission = self.control.submit(envelope, context)
            finally:
                self.control.executor = executor
        else:
            submission = self.control.submit(envelope, context)
        req.kv_control_context = context
        req.kv_hints = msgspec.structs.replace(
            envelope, actions=list(submission.unhandled_actions)
        )

    def finish_request(
        self, req, *, committed_len=0, checkpointed=False, aborted=False
    ):
        context = getattr(req, "kv_control_context", None)
        if context is None or (not aborted and not req.finished()):
            return
        req.kv_control_context = None
        if context.engine_epoch != self.control.registry.engine_epoch:
            return
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        success = (
            checkpointed
            and req.finished()
            and not isinstance(req.finished_reason, FINISH_ABORT)
        )
        committed = None
        if success and committed_len:
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
        self.control.request_event(
            context,
            "success" if success and not aborted else "abort",
            committed=committed,
        )

    def execute(self, command):
        control, registry = self.control, self.control.registry
        if isinstance(command, Capabilities):
            return self.capabilities()
        context = (
            command.ref.context if isinstance(command, Cancel) else command.context
        )
        registry.validate_context(context)
        if isinstance(command, Submit):
            if self.weights.active:
                raise ControlError(
                    "busy", "Finish the weight update before submitting KV actions"
                )
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
        if isinstance(command, Residency):
            return control.executor.residency(
                control.index.resolve(command.target),
                limit=command.limit,
                location=command.location,
            )
        if isinstance(command, Associations):
            return control.index.query(
                session_id=command.session_id,
                cursor=command.cursor,
                limit=command.limit,
            )
        return dict(
            operations=registry.capacity(),
            associations=control.index.capacity(),
            copies=control.executor.claims.capacity(),
        )


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
