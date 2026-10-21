# SPDX-License-Identifier: Apache-2.0
"""Typed, scheduler-owned control service; no model request is needed for RPCs."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ControlContext,
    ControlError,
    Identifier,
    Limits,
    LogicalTarget,
    OperationRef,
    TokensTarget,
    Value,
    validated,
)
from sglang.srt.managers.kv_control.controller import KVControl
from sglang.srt.managers.kv_control.index import IndexCursor
from sglang.srt.managers.kv_hints import KvHintsEnvelope
from sglang.srt.mem_cache.unified_cache.priority_claims import CacheControlConfig


class CacheLimits(Value):
    max_copy_units: int = 65536
    max_claims_per_copy: int = 16
    max_query_nodes: int = 4096
    max_result_targets: int = 4096


class ServiceConfig(Value):
    namespace: Identifier
    engine_id: Identifier | None = None
    limits: Limits = Limits()
    cache_limits: CacheLimits = CacheLimits()

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
    def __init__(self, cache, config: ServiceConfig, *, clock=time.monotonic):
        self.cache, self.config = cache, validated(config, ServiceConfig)
        self.control = KVControl(
            config.engine_id or uuid.uuid4().hex,
            config.namespace,
            enabled=True,
            limits=config.limits,
            clock=clock,
        )
        self.control.executor = cache.create_kv_control_executor(
            engine_id=self.control.registry.engine_id,
            engine_epoch=self.control.registry.engine_epoch,
        )
        capabilities = self.control.executor.capabilities()
        if not capabilities["actions"]:
            raise ValueError(
                "KV control cache is unsupported: " + capabilities["reason"]
            )
        cache.kv_control_service = self

    def capabilities(self):
        registry = self.control.registry
        return dict(
            enabled=True,
            engine_id=registry.engine_id,
            engine_epoch=registry.engine_epoch,
            namespace=registry.namespace,
            executor=self.control.executor.capabilities(),
            policies=(),
            topology="single_owner",
            attached_caller_id="request",
        )

    @property
    def has_pending(self):
        return self.control.registry.capacity()["active"] != 0

    def progress(self):
        if self.has_pending:
            self.control.progress()

    def reset_after_cache_flush(self):
        self.control.reset(self.config.namespace)
        self.control.executor = self.cache.create_kv_control_executor(
            engine_id=self.control.registry.engine_id,
            engine_epoch=self.control.registry.engine_epoch,
        )

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
                cache_salt=req.cache_salt,
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
                control.index.resolve(command.target), limit=command.limit
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
