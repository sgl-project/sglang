# SPDX-License-Identifier: Apache-2.0
"""Public protocol adapters. Capabilities describe execution, not schema presence."""

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ACTION_VERSION,
    HASH_SCHEME,
    PROTOCOL_VERSION,
    ControlError,
    CopyTo,
    Demote,
    Location,
    MoveTo,
    Pin,
    PrefixTarget,
    PublicLocation,
    PublicResult,
    Status,
    TargetOutcome,
    TokensTarget,
)

PUBLIC_ACTIONS = ("sglang.prefetch", "sglang.demote", "sglang.pin")


def public_capabilities(caps):
    directions = caps.get("directions", {})
    paths = {}
    for name, primitive, permitted in (
        (
            "sglang.prefetch",
            "copy_to",
            (("storage", "host"),),
        ),
        (
            "sglang.demote",
            "move_to",
            (("host", "storage"),),
        ),
    ):
        available = (
            directions.get(primitive, ())
            if caps.get("available", True)
            and "sglang." + primitive in caps.get("actions", ())
            else ()
        )
        paths[name] = tuple(p for p in permitted if p in available)
    pins = caps.get("pin_protections", ()) if caps.get("available", True) else ()
    actions = {name: (ACTION_VERSION,) for name, values in paths.items() if values}
    if pins:
        actions["sglang.pin"] = (ACTION_VERSION,)
    return dict(
        protocol_version=PROTOCOL_VERSION,
        schema_versions={name: (ACTION_VERSION,) for name in PUBLIC_ACTIONS},
        actions=actions,
        paths=paths,
        pin_protections=pins,
        pin_query="claim" if pins else None,
        pin_submission=("independent", "attached") if pins else (),
        pin_request_cached="successful_checkpoint_existing_host_prefix"
        if pins
        else None,
        pin_attached_claim_id="returned_effective_claim_id_per_binding"
        if pins
        else None,
        pin_owner_task="optional_local_index_identity" if pins else None,
        hash_only_execution=False,
        prefix=dict(
            namespace=caps.get("namespace"),
            layout_id=caps.get("layout_id"),
            page_size=directions.get("page_alignment"),
            components=caps.get("components", ()),
            hash_scheme=HASH_SCHEME,
            token_material="complete_prefix",
        ),
        storage_backend=directions.get("storage_backend"),
        backend_namespace_selection=bool(directions.get("backend_namespace")),
        backend_namespace=directions.get("backend_namespace"),
        storage_targets=directions.get("storage_targets", False),
        storage_profile=directions.get("storage_profile"),
        timeout="relative_ms_cancel_and_drain",
        result_version="1.0",
    )


def normalize_public(payload, caps):
    capability = public_capabilities(caps)
    if isinstance(payload, Pin):
        if (
            not (
                capability["pin_protections"]
                if payload.mode == "acquire"
                else caps.get("pin_protections", ())
            )
            or payload.location.tier != "host"
            or payload.required_protection != "eviction_exclusion"
        ):
            raise ControlError(
                "unsupported", "Only local host eviction exclusion is executable"
            )
        locations = (payload.location,)
    else:
        name = "sglang.demote" if isinstance(payload, Demote) else "sglang.prefetch"
        path = payload.source.tier, payload.destination.tier
        if path not in capability["paths"][name]:
            raise ControlError("unsupported", "Public transfer path is not executable")
        locations = (payload.source, payload.destination)
    for location in locations:
        if (
            location.backend_namespace is not None
            and location.backend_namespace != capability["backend_namespace"]
        ):
            raise ControlError(
                "incompatible_namespace", "Storage backend namespace differs"
            )
        if location.tier == "storage" and location.backend not in (
            None,
            capability["storage_backend"],
        ):
            raise ControlError("unsupported", "Storage backend is not executable")
    target = payload.target
    if isinstance(payload, Pin) and payload.mode != "acquire":
        # Claim identity suffices even after its content/session was removed.
        return payload
    if (
        target != "request_cached"
        and not isinstance(target, PrefixTarget)
        and target.allow_clip
        and (isinstance(payload, Pin) or not payload.allow_partial)
    ):
        raise ControlError("invalid_payload", "Clipping requires allow_partial")
    if isinstance(target, PrefixTarget):
        c = target.compatibility
        expected = capability["prefix"]
        if c.namespace != expected["namespace"]:
            raise ControlError("incompatible_namespace", "Prefix namespace differs")
        if (
            c.layout_id != expected["layout_id"]
            or c.page_size != expected["page_size"]
            or set(c.components) != set(expected["components"])
            or c.hash_scheme != HASH_SCHEME
        ):
            raise ControlError(
                "unsupported", "Prefix layout/page/components/hash scheme differs"
            )
        if target.start_token != 0 or target.token_ids is None:
            raise ControlError(
                "unsupported", "Execution requires complete token material from zero"
            )
        # Blocks are observations, not key material. The executor derives its own
        # storage keys from tokens and applies encode_cache_salt exactly once.
        target = TokensTarget(
            namespace=c.namespace,
            token_ids=target.token_ids,
            cache_salt=c.cache_salt or None,
            layout_id=c.layout_id,
        )
    if isinstance(payload, Pin):
        return msgspec.structs.replace(payload, target=target)
    primitive = MoveTo if isinstance(payload, Demote) else CopyTo
    return primitive(
        target=target,
        src=Location(
            tier=payload.source.tier,
            engine_id=payload.source.engine_id,
            backend=payload.source.backend,
            backend_namespace=payload.source.backend_namespace,
        ),
        dst=Location(
            tier=payload.destination.tier,
            engine_id=payload.destination.engine_id,
            backend=payload.destination.backend,
            backend_namespace=payload.destination.backend_namespace,
        ),
    )


def project_result(current, update, sequence):
    """Project only per-target completion evidence; queued I/O proves no readiness."""
    public = current.public_result or PublicResult(
        owner_epoch=current.ref.context.engine_epoch,
        state_version=sequence,
        outcome_sequence=sequence,
    )
    ready = tuple(
        o for o in update.target_outcomes if "destination_ready" in o.completed_steps
    )
    reclaimed = tuple(
        o for o in update.target_outcomes if "source_reclaimed" in o.completed_steps
    )
    demote = current.action_type == "sglang.demote"
    accepted = (
        tuple(
            o
            for o in update.target_outcomes
            if o.status in (Status.ACCEPTED, Status.RUNNING)
            or "destination_ready" in o.completed_steps
            or "protection_active" in o.completed_steps
        )
        or public.accepted_scope
    )
    missing = tuple(o for o in update.target_outcomes if o.status == Status.MISSING)
    blocked = tuple(
        o
        for o in update.target_outcomes
        if o.status in (Status.DEFERRED, Status.PARTIAL)
        and "source_reclaimed" not in o.completed_steps
    )
    if not update.target_outcomes and update.status in (
        Status.MISSING,
        Status.DEFERRED,
    ):
        scope = tuple(
            msgspec.structs.replace(
                o, status=update.status, code=update.code, reason=update.reason
            )
            for o in public.requested_scope
        )
        if update.status == Status.MISSING:
            missing = scope
        else:
            blocked = scope
    return msgspec.structs.replace(
        public,
        state_version=sequence,
        outcome_sequence=sequence,
        accepted_scope=accepted,
        ready_scope=ready,
        ready_location=PublicLocation(**msgspec.to_builtins(ready[0].location))
        if ready
        else None,
        preserved_scope=ready if demote else (),
        reclaimed_scope=reclaimed if demote else (),
        missing_scope=missing,
        blocked_scope=blocked,
    )


def requested_outcomes(scope):
    return tuple(
        TargetOutcome(coverage=branch.requested, branch_index=i, status=Status.RECEIVED)
        for i, branch in enumerate(scope.branches)
    )
