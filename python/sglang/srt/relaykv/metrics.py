from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional

from .planner import RelayKVPlan

logger = logging.getLogger(__name__)

POLICY_STATES = ("off", "shadow", "applied_candidate", "fallback_candidate")
POLICY_EVENT_LOG_KEYS = (
    "runtime_policy_state",
    "request_id",
    "layer_idx",
    "full_kv_fits",
    "budget_pressure",
    "available_kv_budget_tokens",
    "estimated_full_kv_tokens",
    "resident_budget_tokens",
    "coverage_ratio",
    "risk_level",
    "policy_reason",
)


def log_shadow_plan(
    plan: RelayKVPlan,
    *,
    prefix: str = "relaykv_shadow_plan",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a compact JSON log line for MVP-0 shadow planning."""

    payload = plan.to_log_dict()
    if extra:
        payload.update(extra)
    logger.info("%s=%s", prefix, json.dumps(payload, sort_keys=True))


def should_log(step_idx: Optional[int], interval: int) -> bool:
    if interval <= 0:
        return True
    if step_idx is None:
        return True
    return step_idx % interval == 0


def _event_value(event: RelayKVPlan | Mapping[str, Any], key: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(key)
    return getattr(event, key, None)


def _bool_counter_key(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def summarize_policy_events(
    events: Iterable[RelayKVPlan | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize RelayKV runtime policy metadata without touching runtime state."""

    state_counts = Counter({state: 0 for state in POLICY_STATES})
    budget_pressure_counts = Counter({"true": 0, "false": 0, "unknown": 0})
    full_kv_fits_counts = Counter({"true": 0, "false": 0, "unknown": 0})
    policy_reason_counts: Counter[str] = Counter()

    for event in events:
        state = str(_event_value(event, "runtime_policy_state") or "unknown")
        state_counts[state] += 1
        budget_pressure_counts[
            _bool_counter_key(_event_value(event, "budget_pressure"))
        ] += 1
        full_kv_fits_counts[
            _bool_counter_key(_event_value(event, "full_kv_fits"))
        ] += 1
        policy_reason = _event_value(event, "policy_reason")
        if policy_reason:
            policy_reason_counts[str(policy_reason)] += 1

    return {
        "relaykv_policy_off_count": state_counts["off"],
        "relaykv_policy_shadow_count": state_counts["shadow"],
        "relaykv_policy_applied_candidate_count": state_counts["applied_candidate"],
        "relaykv_policy_fallback_candidate_count": state_counts["fallback_candidate"],
        "policy_state_counts": dict(state_counts),
        "budget_pressure_counts": dict(budget_pressure_counts),
        "full_kv_fits_counts": dict(full_kv_fits_counts),
        "policy_reason_counts": dict(sorted(policy_reason_counts.items())),
    }


def summarize_candidate_events(
    events: Iterable[RelayKVPlan | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize candidate runtime event payloads and their no-op guards."""

    candidate_counts = Counter({"applied_candidate": 0, "fallback_candidate": 0})
    noop_guard_counts = Counter(
        {
            "fallback_candidate_noop_guard_true": 0,
            "applied_candidate_log_only_true": 0,
            "dry_copy_candidate_true": 0,
            "scheduler_policy_noop_true": 0,
            "kv_cache_mutation_false": 0,
            "attention_override_false": 0,
            "host_backup_copy_false": 0,
            "host_backup_copy_candidate_true": 0,
            "host_backup_copy_candidate_false": 0,
            "host_backup_copy_executed_false": 0,
            "host_backup_copy_executed_true": 0,
            "snapshot_created_true": 0,
            "snapshot_created_false": 0,
            "runtime_writeback_false": 0,
        }
    )

    for event in events:
        state = str(_event_value(event, "runtime_policy_state") or "unknown")
        if state in candidate_counts:
            candidate_counts[state] += 1
        if _event_value(event, "fallback_candidate_noop_guard") is True:
            noop_guard_counts["fallback_candidate_noop_guard_true"] += 1
        if _event_value(event, "applied_candidate_log_only") is True:
            noop_guard_counts["applied_candidate_log_only_true"] += 1
        if _event_value(event, "dry_copy_candidate") is True:
            noop_guard_counts["dry_copy_candidate_true"] += 1
        if _event_value(event, "scheduler_policy_noop") is True:
            noop_guard_counts["scheduler_policy_noop_true"] += 1
        if _event_value(event, "kv_cache_mutation") is False:
            noop_guard_counts["kv_cache_mutation_false"] += 1
        if _event_value(event, "attention_override") is False:
            noop_guard_counts["attention_override_false"] += 1
        if _event_value(event, "host_backup_copy") is False:
            noop_guard_counts["host_backup_copy_false"] += 1
        if _event_value(event, "host_backup_copy_candidate") is True:
            noop_guard_counts["host_backup_copy_candidate_true"] += 1
        if _event_value(event, "host_backup_copy_candidate") is False:
            noop_guard_counts["host_backup_copy_candidate_false"] += 1
        if _event_value(event, "host_backup_copy_executed") is False:
            noop_guard_counts["host_backup_copy_executed_false"] += 1
        if _event_value(event, "host_backup_copy_executed") is True:
            noop_guard_counts["host_backup_copy_executed_true"] += 1
        if _event_value(event, "snapshot_created") is True:
            noop_guard_counts["snapshot_created_true"] += 1
        if _event_value(event, "snapshot_created") is False:
            noop_guard_counts["snapshot_created_false"] += 1
        if _event_value(event, "runtime_writeback") is False:
            noop_guard_counts["runtime_writeback_false"] += 1

    return {
        "candidate_event_counts": dict(candidate_counts),
        "noop_guard_counts": dict(noop_guard_counts),
    }


def _candidate_copy_count_template() -> Counter[str]:
    return Counter(
        {
            "total_candidate_events": 0,
            "applied_candidate_count": 0,
            "fallback_candidate_count": 0,
            "snapshot_created_count": 0,
            "snapshot_skipped_count": 0,
            "host_backup_copy_candidate_count": 0,
            "host_backup_copy_executed_count": 0,
            "host_backup_copy_skipped_count": 0,
            "copy_equal_true_count": 0,
            "copy_equal_false_count": 0,
            "source_mutated_true_count": 0,
            "source_mutated_false_count": 0,
            "fallback_candidate_noop_guard_count": 0,
            "attention_override_true_count": 0,
            "attention_override_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "kv_cache_mutation_false_count": 0,
            "runtime_writeback_true_count": 0,
            "runtime_writeback_false_count": 0,
            "scheduler_policy_noop_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
        }
    )


def _increment_candidate_copy_counts(
    counts: Counter[str],
    event: RelayKVPlan | Mapping[str, Any],
) -> None:
    counts["total_candidate_events"] += 1
    state = str(_event_value(event, "runtime_policy_state") or "unknown")
    if state == "applied_candidate":
        counts["applied_candidate_count"] += 1
    elif state == "fallback_candidate":
        counts["fallback_candidate_count"] += 1

    if _event_value(event, "snapshot_created") is True:
        counts["snapshot_created_count"] += 1
    else:
        counts["snapshot_skipped_count"] += 1
    if _event_value(event, "host_backup_copy_candidate") is True:
        counts["host_backup_copy_candidate_count"] += 1
    if _event_value(event, "host_backup_copy_executed") is True:
        counts["host_backup_copy_executed_count"] += 1
    else:
        counts["host_backup_copy_skipped_count"] += 1
    if _event_value(event, "copy_equal") is True:
        counts["copy_equal_true_count"] += 1
    elif _event_value(event, "copy_equal") is False:
        counts["copy_equal_false_count"] += 1
    if _event_value(event, "source_mutated") is True:
        counts["source_mutated_true_count"] += 1
    elif _event_value(event, "source_mutated") is False:
        counts["source_mutated_false_count"] += 1
    if _event_value(event, "fallback_candidate_noop_guard") is True:
        counts["fallback_candidate_noop_guard_count"] += 1

    for field, prefix in (
        ("attention_override", "attention_override"),
        ("kv_cache_mutation", "kv_cache_mutation"),
        ("runtime_writeback", "runtime_writeback"),
        ("scheduler_policy_noop", "scheduler_policy_noop"),
    ):
        value = _event_value(event, field)
        if value is True:
            counts[f"{prefix}_true_count"] += 1
        elif value is False:
            counts[f"{prefix}_false_count"] += 1


def _counter_payload(counts: Counter[str]) -> dict[str, int]:
    return dict(counts)


def summarize_host_backup_copy_candidates_for_smoke(
    events: Iterable[RelayKVPlan | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize smoke-only host-backup copy candidate events.

    This reads event payloads only. It does not connect the candidate path to
    attention, KV freeing, KV pool mutation, scheduler decisions, or writeback.
    """

    total_counts = _candidate_copy_count_template()
    state_counts = Counter({"applied_candidate": 0, "fallback_candidate": 0})
    skipped_reason_counts: Counter[str] = Counter()
    per_layer: dict[str, Counter[str]] = {}
    per_request: dict[str, Counter[str]] = {}
    per_batch: dict[str, Counter[str]] = {}

    for event in events:
        state = str(_event_value(event, "runtime_policy_state") or "unknown")
        if state in state_counts:
            state_counts[state] += 1
        skipped_reason = _event_value(event, "host_backup_copy_skipped_reason")
        if skipped_reason:
            skipped_reason_counts[str(skipped_reason)] += 1

        layer_key = str(_event_value(event, "layer_idx"))
        request_key = str(_event_value(event, "request_id"))
        batch_value = _event_value(event, "batch_id")
        if batch_value is None:
            batch_value = _event_value(event, "batch_index")
        batch_key = str(batch_value)
        if layer_key not in per_layer:
            per_layer[layer_key] = _candidate_copy_count_template()
        if request_key not in per_request:
            per_request[request_key] = _candidate_copy_count_template()
        if batch_key not in per_batch:
            per_batch[batch_key] = _candidate_copy_count_template()

        _increment_candidate_copy_counts(total_counts, event)
        _increment_candidate_copy_counts(per_layer[layer_key], event)
        _increment_candidate_copy_counts(per_request[request_key], event)
        _increment_candidate_copy_counts(per_batch[batch_key], event)

    return {
        **_counter_payload(total_counts),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "policy_state_counts": dict(state_counts),
        "per_layer_counts": {
            key: _counter_payload(value) for key, value in sorted(per_layer.items())
        },
        "per_request_counts": {
            key: _counter_payload(value) for key, value in sorted(per_request.items())
        },
        "per_batch_counts": {
            key: _counter_payload(value) for key, value in sorted(per_batch.items())
        },
    }


def log_host_backup_copy_candidate_summary(
    events: Iterable[RelayKVPlan | Mapping[str, Any]],
    *,
    prefix: str = "relaykv_host_backup_copy_candidate_summary",
) -> None:
    payload = summarize_host_backup_copy_candidates_for_smoke(events)
    logger.info("%s=%s", prefix, json.dumps(payload, sort_keys=True))


def _event_sequence_from_input(
    value: Mapping[str, Any] | list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    event_keys: tuple[str, ...],
) -> list[Mapping[str, Any]] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, Mapping):
        raise TypeError("RelayKV join inputs must be mapping, list, or tuple objects")
    for key in event_keys:
        events = value.get(key)
        if isinstance(events, (list, tuple)):
            return list(events)
    return None


def _summary_count_value(summary: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, int):
            return value
    return 0


def _event_layer_value(event: Mapping[str, Any]) -> Any:
    layer_value = _event_value(event, "layer_id")
    if layer_value is None:
        layer_value = _event_value(event, "layer_idx")
    return layer_value


def _event_req_pool_idx_value(event: Mapping[str, Any]) -> Any:
    req_pool_idx = _event_value(event, "req_pool_idx")
    if req_pool_idx is None:
        req_pool_idx = _event_value(event, "req_pool_index")
    if req_pool_idx is None:
        req_pool_idx = _event_value(event, "request_pool_idx")
    return req_pool_idx


def _runtime_host_backup_join_key(
    event: Mapping[str, Any],
) -> tuple[str, str, str] | None:
    request_id = _event_value(event, "request_id")
    req_pool_idx = _event_req_pool_idx_value(event)
    layer_id = _event_layer_value(event)
    if request_id is None or req_pool_idx is None or layer_id is None:
        return None
    return (str(request_id), str(req_pool_idx), str(layer_id))


def _increment_join_safety_counts(
    counts: Counter[str],
    event: Mapping[str, Any],
) -> None:
    if _event_value(event, "source_mutated") is True:
        counts["source_mutated_true_count"] += 1
    if _event_value(event, "attention_override") is True:
        counts["attention_override_true_count"] += 1
    if _event_value(event, "kv_cache_mutation") is True:
        counts["kv_cache_mutation_true_count"] += 1
    if _event_value(event, "runtime_writeback") is True:
        counts["runtime_writeback_true_count"] += 1
    if _event_value(event, "scheduler_policy_noop") is False:
        counts["scheduler_policy_noop_false_count"] += 1


def join_runtime_observation_with_host_backup_candidates_for_smoke(
    runtime_payloads_or_summary: Mapping[str, Any]
    | list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    host_backup_candidate_summary_or_events: Mapping[str, Any]
    | list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Join read-only runtime observation metadata with host-backup candidates.

    The join is summary-only plumbing for smoke tests. It reads dictionaries and
    lists only, does not touch tensors, KV pools, snapshots, host backup copy,
    attention state, scheduler decisions, or runtime writeback.
    """

    runtime_payloads = _event_sequence_from_input(
        runtime_payloads_or_summary,
        event_keys=("runtime_payloads", "payloads", "events"),
    )
    candidate_events = _event_sequence_from_input(
        host_backup_candidate_summary_or_events,
        event_keys=("host_backup_candidate_events", "candidate_events", "events"),
    )

    safety_counts: Counter[str] = Counter(
        {
            "source_mutated_true_count": 0,
            "attention_override_true_count": 0,
            "kv_cache_mutation_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
        }
    )

    if runtime_payloads is None or candidate_events is None:
        if isinstance(runtime_payloads_or_summary, Mapping):
            for key in safety_counts:
                safety_counts[key] += _summary_count_value(
                    runtime_payloads_or_summary, key
                )
        if isinstance(host_backup_candidate_summary_or_events, Mapping):
            for key in safety_counts:
                safety_counts[key] += _summary_count_value(
                    host_backup_candidate_summary_or_events, key
                )
        return {
            "join_granularity": "summary_only_unjoinable",
            "total_runtime_payloads": _summary_count_value(
                runtime_payloads_or_summary, "total_runtime_payloads", "total_payloads"
            )
            if isinstance(runtime_payloads_or_summary, Mapping)
            else 0,
            "total_host_backup_candidate_events": _summary_count_value(
                host_backup_candidate_summary_or_events,
                "total_host_backup_candidate_events",
                "total_candidate_events",
            )
            if isinstance(host_backup_candidate_summary_or_events, Mapping)
            else 0,
            "joined_count": 0,
            "unmatched_runtime_count": 0,
            "unmatched_candidate_count": 0,
            "per_request_join_counts": {},
            "per_layer_join_counts": {},
            "req_pool_idx_joined_count": 0,
            "req_pool_idx_missing_count": 0,
            **dict(safety_counts),
        }

    candidate_key_counts: Counter[tuple[str, str, str]] = Counter()
    candidate_invalid_key_count = 0
    req_pool_idx_missing_count = 0
    for event in candidate_events:
        if not isinstance(event, Mapping):
            raise TypeError("RelayKV candidate join events must be mappings")
        _increment_join_safety_counts(safety_counts, event)
        if _event_req_pool_idx_value(event) is None:
            req_pool_idx_missing_count += 1
        key = _runtime_host_backup_join_key(event)
        if key is None:
            candidate_invalid_key_count += 1
        else:
            candidate_key_counts[key] += 1

    joined_count = 0
    unmatched_runtime_count = 0
    req_pool_idx_joined_count = 0
    per_request_join_counts: Counter[str] = Counter()
    per_layer_join_counts: Counter[str] = Counter()

    for payload in runtime_payloads:
        if not isinstance(payload, Mapping):
            raise TypeError("RelayKV runtime join payloads must be mappings")
        _increment_join_safety_counts(safety_counts, payload)
        if _event_req_pool_idx_value(payload) is None:
            req_pool_idx_missing_count += 1
        key = _runtime_host_backup_join_key(payload)
        if key is not None and candidate_key_counts[key] > 0:
            candidate_key_counts[key] -= 1
            joined_count += 1
            req_pool_idx_joined_count += 1
            request_id, _, layer_id = key
            per_request_join_counts[request_id] += 1
            per_layer_join_counts[layer_id] += 1
        else:
            unmatched_runtime_count += 1

    unmatched_candidate_count = candidate_invalid_key_count + sum(
        candidate_key_counts.values()
    )
    return {
        "join_granularity": "event",
        "total_runtime_payloads": len(runtime_payloads),
        "total_host_backup_candidate_events": len(candidate_events),
        "joined_count": joined_count,
        "unmatched_runtime_count": unmatched_runtime_count,
        "unmatched_candidate_count": unmatched_candidate_count,
        "per_request_join_counts": dict(sorted(per_request_join_counts.items())),
        "per_layer_join_counts": dict(sorted(per_layer_join_counts.items())),
        "req_pool_idx_joined_count": req_pool_idx_joined_count,
        "req_pool_idx_missing_count": req_pool_idx_missing_count,
        **dict(safety_counts),
    }


_REPORT_SAFETY_COUNTER_KEYS = (
    "source_mutated_true_count",
    "attention_override_true_count",
    "kv_cache_mutation_true_count",
    "runtime_writeback_true_count",
    "scheduler_policy_noop_false_count",
)

_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS = (
    "host_backup_copy_executed_count",
    "kv_pool_read_count",
    "kv_snapshot_count",
)


def _readonly_report_value(
    missing_field_counts: Counter[str],
    *sources_and_keys: tuple[Mapping[str, Any], tuple[str, ...]],
) -> int:
    for source, keys in sources_and_keys:
        for key in keys:
            value = source.get(key)
            if isinstance(value, int):
                return value
    missing_field_counts["missing_field_count"] += 1
    return 0


def _readonly_report_safety_counter(
    missing_field_counts: Counter[str],
    key: str,
    *sources: Mapping[str, Any],
) -> int:
    values = [
        source.get(key)
        for source in sources
        if isinstance(source, Mapping) and isinstance(source.get(key), int)
    ]
    if not values:
        missing_field_counts["missing_field_count"] += 1
        return 0
    return max(values)


def build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
    runtime_observation_summary: Mapping[str, Any],
    host_backup_candidate_summary: Mapping[str, Any],
    join_summary: Mapping[str, Any],
    policy_dry_run_summary: Mapping[str, Any] | None = None,
    materialization_summary: Mapping[str, Any] | None = None,
    host_backup_copy_request_summary: Mapping[str, Any] | None = None,
    host_backup_copy_boundary_result_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only runtime/candidate/join report for smoke tests.

    This combines precomputed summary dictionaries only. It does not execute
    host backup copy, read KV pools, create KV snapshots, connect attention,
    change scheduler decisions, or write runtime state.
    """

    if not isinstance(runtime_observation_summary, Mapping):
        raise TypeError("runtime_observation_summary must be a mapping")
    if not isinstance(host_backup_candidate_summary, Mapping):
        raise TypeError("host_backup_candidate_summary must be a mapping")
    if not isinstance(join_summary, Mapping):
        raise TypeError("join_summary must be a mapping")
    if policy_dry_run_summary is not None and not isinstance(
        policy_dry_run_summary, Mapping
    ):
        raise TypeError("policy_dry_run_summary must be a mapping or None")
    if materialization_summary is not None and not isinstance(
        materialization_summary, Mapping
    ):
        raise TypeError("materialization_summary must be a mapping or None")
    if host_backup_copy_request_summary is not None and not isinstance(
        host_backup_copy_request_summary, Mapping
    ):
        raise TypeError("host_backup_copy_request_summary must be a mapping or None")
    if host_backup_copy_boundary_result_summary is not None and not isinstance(
        host_backup_copy_boundary_result_summary, Mapping
    ):
        raise TypeError(
            "host_backup_copy_boundary_result_summary must be a mapping or None"
        )

    missing_field_counts: Counter[str] = Counter({"missing_field_count": 0})
    total_runtime_payloads = _readonly_report_value(
        missing_field_counts,
        (runtime_observation_summary, ("total_runtime_payloads", "total_payloads")),
        (join_summary, ("total_runtime_payloads", "total_payloads")),
    )
    total_host_backup_candidate_events = _readonly_report_value(
        missing_field_counts,
        (
            host_backup_candidate_summary,
            ("total_host_backup_candidate_events", "total_candidate_events"),
        ),
        (join_summary, ("total_host_backup_candidate_events", "total_candidate_events")),
    )
    joined_count = _readonly_report_value(
        missing_field_counts,
        (join_summary, ("joined_count",)),
    )
    unmatched_runtime_count = _readonly_report_value(
        missing_field_counts,
        (join_summary, ("unmatched_runtime_count",)),
    )
    unmatched_candidate_count = _readonly_report_value(
        missing_field_counts,
        (join_summary, ("unmatched_candidate_count",)),
    )
    req_pool_idx_joined_count = _readonly_report_value(
        missing_field_counts,
        (join_summary, ("req_pool_idx_joined_count",)),
    )
    req_pool_idx_missing_count = _readonly_report_value(
        missing_field_counts,
        (join_summary, ("req_pool_idx_missing_count",)),
    )

    safety_counts = {
        key: _readonly_report_safety_counter(
            missing_field_counts,
            key,
            runtime_observation_summary,
            host_backup_candidate_summary,
            join_summary,
            policy_dry_run_summary or {},
            materialization_summary or {},
            host_backup_copy_request_summary or {},
            host_backup_copy_boundary_result_summary or {},
        )
        for key in _REPORT_SAFETY_COUNTER_KEYS
    }
    materialization_safety_counts = {
        key: (
            _readonly_report_safety_counter(
                missing_field_counts,
                key,
                materialization_summary,
                host_backup_copy_request_summary,
                host_backup_copy_boundary_result_summary,
            )
            if materialization_summary is not None
            or host_backup_copy_request_summary is not None
            or host_backup_copy_boundary_result_summary is not None
            else 0
        )
        for key in _MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS
    }
    host_backup_copy_request_summary_included = (
        host_backup_copy_request_summary is not None
    )
    host_backup_copy_request_total = 0
    host_backup_copy_request_ready_count = 0
    host_backup_copy_request_blocked_count = 0
    host_backup_copy_request_materialized_kv_count = 0
    if host_backup_copy_request_summary is not None:
        host_backup_copy_request_total = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_request_summary, ("total_copy_requests",)),
        )
        host_backup_copy_request_ready_count = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_request_summary, ("request_ready_count",)),
        )
        host_backup_copy_request_blocked_count = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_request_summary, ("blocked_count",)),
        )
        host_backup_copy_request_materialized_kv_count = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_request_summary, ("materialized_kv_count",)),
        )
    host_backup_copy_boundary_result_summary_included = (
        host_backup_copy_boundary_result_summary is not None
    )
    host_backup_copy_boundary_result_total = 0
    host_backup_copy_boundary_noop_count = 0
    host_backup_copy_boundary_blocked_count = 0
    host_backup_copy_boundary_error_count = 0
    host_backup_copy_boundary_materialized_kv_count = 0
    if host_backup_copy_boundary_result_summary is not None:
        host_backup_copy_boundary_result_total = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_boundary_result_summary, ("total_boundary_results",)),
        )
        host_backup_copy_boundary_noop_count = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_boundary_result_summary, ("boundary_noop_count",)),
        )
        host_backup_copy_boundary_blocked_count = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_boundary_result_summary, ("blocked_count",)),
        )
        host_backup_copy_boundary_error_count = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_boundary_result_summary, ("error_count",)),
        )
        host_backup_copy_boundary_materialized_kv_count = _readonly_report_value(
            missing_field_counts,
            (host_backup_copy_boundary_result_summary, ("materialized_kv_count",)),
        )
    policy_dry_run_included = policy_dry_run_summary is not None
    policy_dry_run_total_events = 0
    policy_dry_run_selected_event_count = 0
    if policy_dry_run_summary is not None:
        policy_dry_run_total_events = _readonly_report_value(
            missing_field_counts,
            (policy_dry_run_summary, ("total_events", "policy_dry_run_total_events")),
        )
        for key in ("selected_event_count", "policy_dry_run_selected_event_count"):
            value = policy_dry_run_summary.get(key)
            if isinstance(value, int):
                policy_dry_run_selected_event_count = value
                break
    materialization_summary_included = materialization_summary is not None
    materialization_total_results = 0
    materialization_result_count = 0
    materialization_fake_count = 0
    materialization_guarded_noop_count = 0
    materialization_candidate_event_count = 0
    materialization_host_backup_copy_count = 0
    materialization_blocked_count = 0
    materialization_skipped_count = 0
    materialization_error_count = 0
    materialized_kv_count = 0
    materialized_token_count = 0
    if materialization_summary is not None:
        materialization_total_results = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("total_materialization_results",)),
        )
        materialization_result_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("materialized_result_count",)),
        )
        materialization_fake_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("fake_materialized_count",)),
        )
        materialization_guarded_noop_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("guarded_noop_count",)),
        )
        materialization_candidate_event_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("candidate_event_materialized_count",)),
        )
        materialization_host_backup_copy_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("host_backup_copy_materialized_count",)),
        )
        materialization_blocked_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("blocked_count",)),
        )
        materialization_skipped_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("skipped_count",)),
        )
        materialization_error_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("error_count",)),
        )
        materialized_kv_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("materialized_kv_count",)),
        )
        materialized_token_count = _readonly_report_value(
            missing_field_counts,
            (materialization_summary, ("materialized_token_count",)),
        )
    report_generated_from_readonly_inputs = True
    all_safety_counts = {**safety_counts, **materialization_safety_counts}
    overall_safety_status = (
        "pass"
        if report_generated_from_readonly_inputs
        and all(value == 0 for value in all_safety_counts.values())
        else "fail"
    )
    report_warning_counts = {
        "missing_field_warning_count": missing_field_counts["missing_field_count"]
    }

    return {
        "report_type": "relaykv_readonly_runtime_candidate_join_report",
        "report_generated_from_readonly_inputs": report_generated_from_readonly_inputs,
        "runtime_observation_summary": dict(runtime_observation_summary),
        "host_backup_candidate_summary": dict(host_backup_candidate_summary),
        "join_summary": dict(join_summary),
        "policy_dry_run_summary": (
            dict(policy_dry_run_summary) if policy_dry_run_summary is not None else None
        ),
        "policy_dry_run_included": policy_dry_run_included,
        "policy_dry_run_total_events": policy_dry_run_total_events,
        "policy_dry_run_selected_event_count": policy_dry_run_selected_event_count,
        "materialization_summary": (
            dict(materialization_summary)
            if materialization_summary is not None
            else None
        ),
        "materialization_summary_included": materialization_summary_included,
        "materialization_total_results": materialization_total_results,
        "materialization_result_count": materialization_result_count,
        "materialization_fake_count": materialization_fake_count,
        "materialization_guarded_noop_count": materialization_guarded_noop_count,
        "materialization_candidate_event_count": materialization_candidate_event_count,
        "materialization_host_backup_copy_count": materialization_host_backup_copy_count,
        "materialization_blocked_count": materialization_blocked_count,
        "materialization_skipped_count": materialization_skipped_count,
        "materialization_error_count": materialization_error_count,
        "materialized_kv_count": materialized_kv_count,
        "materialized_token_count": materialized_token_count,
        "host_backup_copy_request_summary": (
            dict(host_backup_copy_request_summary)
            if host_backup_copy_request_summary is not None
            else None
        ),
        "host_backup_copy_request_summary_included": (
            host_backup_copy_request_summary_included
        ),
        "host_backup_copy_request_total": host_backup_copy_request_total,
        "host_backup_copy_request_ready_count": host_backup_copy_request_ready_count,
        "host_backup_copy_request_blocked_count": (
            host_backup_copy_request_blocked_count
        ),
        "host_backup_copy_request_materialized_kv_count": (
            host_backup_copy_request_materialized_kv_count
        ),
        "host_backup_copy_boundary_result_summary": (
            dict(host_backup_copy_boundary_result_summary)
            if host_backup_copy_boundary_result_summary is not None
            else None
        ),
        "host_backup_copy_boundary_result_summary_included": (
            host_backup_copy_boundary_result_summary_included
        ),
        "host_backup_copy_boundary_result_total": (
            host_backup_copy_boundary_result_total
        ),
        "host_backup_copy_boundary_noop_count": (
            host_backup_copy_boundary_noop_count
        ),
        "host_backup_copy_boundary_blocked_count": (
            host_backup_copy_boundary_blocked_count
        ),
        "host_backup_copy_boundary_error_count": (
            host_backup_copy_boundary_error_count
        ),
        "host_backup_copy_boundary_materialized_kv_count": (
            host_backup_copy_boundary_materialized_kv_count
        ),
        "overall_safety_status": overall_safety_status,
        "total_runtime_payloads": total_runtime_payloads,
        "total_host_backup_candidate_events": total_host_backup_candidate_events,
        "joined_count": joined_count,
        "unmatched_runtime_count": unmatched_runtime_count,
        "unmatched_candidate_count": unmatched_candidate_count,
        "join_granularity": str(join_summary.get("join_granularity", "unknown")),
        "req_pool_idx_joined_count": req_pool_idx_joined_count,
        "req_pool_idx_missing_count": req_pool_idx_missing_count,
        **all_safety_counts,
        "missing_field_counts": dict(missing_field_counts),
        "report_warning_counts": report_warning_counts,
    }


def assess_relaykv_readonly_materialization_readiness_for_smoke(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess read-only RelayKV dry-run readiness for safe materialization.

    This consumes a precomputed report dictionary only. It does not materialize
    KV data, execute host backup copy, read KV pools, snapshot KV, connect
    attention, alter scheduler decisions, or write runtime state.
    """

    if not isinstance(report, Mapping):
        raise TypeError("RelayKV materialization readiness report must be a mapping")

    safety_counts = {
        key: report.get(key) if isinstance(report.get(key), int) else 0
        for key in _REPORT_SAFETY_COUNTER_KEYS
    }
    report_generated_from_readonly_inputs = (
        report.get("report_generated_from_readonly_inputs") is True
    )
    observed_overall_safety_status = str(
        report.get("overall_safety_status", "unknown")
    )
    observed_policy_dry_run_included = report.get("policy_dry_run_included") is True
    observed_policy_dry_run_total_events = (
        report.get("policy_dry_run_total_events")
        if isinstance(report.get("policy_dry_run_total_events"), int)
        else 0
    )
    observed_joined_count = (
        report.get("joined_count") if isinstance(report.get("joined_count"), int) else 0
    )
    observed_unmatched_runtime_count = (
        report.get("unmatched_runtime_count")
        if isinstance(report.get("unmatched_runtime_count"), int)
        else 0
    )
    observed_unmatched_candidate_count = (
        report.get("unmatched_candidate_count")
        if isinstance(report.get("unmatched_candidate_count"), int)
        else 0
    )
    observed_req_pool_idx_missing_count = (
        report.get("req_pool_idx_missing_count")
        if isinstance(report.get("req_pool_idx_missing_count"), int)
        else 0
    )
    observed_join_granularity = str(report.get("join_granularity", "unknown"))

    blocker_state_by_reason = {
        "safety_counter_nonzero": "blocked_safety_counter_nonzero",
        "not_readonly_report": "blocked_not_readonly_report",
        "overall_safety_not_pass": "blocked_overall_safety_not_pass",
        "policy_dry_run_missing": "blocked_policy_dry_run_missing",
        "no_joined_events": "blocked_no_joined_events",
        "summary_only_unjoinable": "blocked_summary_only_unjoinable",
        "req_pool_idx_missing": "blocked_req_pool_idx_missing",
    }
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = []

    if any(value != 0 for value in safety_counts.values()):
        blocking_reasons.append("safety_counter_nonzero")
    if not report_generated_from_readonly_inputs:
        blocking_reasons.append("not_readonly_report")
    if observed_overall_safety_status != "pass":
        blocking_reasons.append("overall_safety_not_pass")
    if (
        not observed_policy_dry_run_included
        or observed_policy_dry_run_total_events <= 0
    ):
        blocking_reasons.append("policy_dry_run_missing")
    if observed_joined_count <= 0:
        blocking_reasons.append("no_joined_events")
    if observed_join_granularity == "summary_only_unjoinable":
        blocking_reasons.append("summary_only_unjoinable")
    if observed_req_pool_idx_missing_count > 0:
        blocking_reasons.append("req_pool_idx_missing")
    if observed_unmatched_runtime_count > 0:
        warning_reasons.append("unmatched_runtime_events_present")
    if observed_unmatched_candidate_count > 0:
        warning_reasons.append("unmatched_candidate_events_present")

    ready_for_materialization = not blocking_reasons
    if ready_for_materialization:
        readiness_state = "ready_for_safe_materialization_dry_run_complete"
        readiness_reasons = ["readonly_dry_run_report_ready"]
    elif len(blocking_reasons) == 1:
        readiness_state = blocker_state_by_reason[blocking_reasons[0]]
        readiness_reasons = []
    else:
        readiness_state = "blocked_multiple_reasons"
        readiness_reasons = []

    return {
        "readiness_type": "relaykv_readonly_materialization_readiness",
        "ready_for_materialization": ready_for_materialization,
        "readiness_state": readiness_state,
        "readiness_reasons": readiness_reasons,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "observed_join_granularity": observed_join_granularity,
        "observed_overall_safety_status": observed_overall_safety_status,
        "observed_policy_dry_run_included": observed_policy_dry_run_included,
        "observed_policy_dry_run_total_events": observed_policy_dry_run_total_events,
        "observed_joined_count": observed_joined_count,
        "observed_unmatched_runtime_count": observed_unmatched_runtime_count,
        "observed_unmatched_candidate_count": observed_unmatched_candidate_count,
        "observed_req_pool_idx_missing_count": observed_req_pool_idx_missing_count,
        "report_generated_from_readonly_inputs": report_generated_from_readonly_inputs,
        **safety_counts,
    }


def assess_relaykv_readonly_attention_readiness_for_smoke(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess read-only RelayKV metadata readiness for attention connection.

    This consumes a precomputed report dictionary only. It does not connect
    attention, materialize KV data, execute host backup copy, read KV pools,
    snapshot KV, alter scheduler decisions, or write runtime state.
    """

    if not isinstance(report, Mapping):
        raise TypeError("RelayKV attention readiness report must be a mapping")

    safety_counter_keys = (
        *_REPORT_SAFETY_COUNTER_KEYS,
        *_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS,
    )
    safety_counts = {
        key: report.get(key) if isinstance(report.get(key), int) else 0
        for key in safety_counter_keys
    }

    report_generated_from_readonly_inputs = (
        report.get("report_generated_from_readonly_inputs") is True
    )
    observed_overall_safety_status = str(
        report.get("overall_safety_status", "unknown")
    )
    observed_policy_dry_run_included = report.get("policy_dry_run_included") is True
    observed_policy_dry_run_total_events = (
        report.get("policy_dry_run_total_events")
        if isinstance(report.get("policy_dry_run_total_events"), int)
        else 0
    )
    observed_materialization_summary_included = (
        report.get("materialization_summary_included") is True
    )
    observed_materialization_total_results = (
        report.get("materialization_total_results")
        if isinstance(report.get("materialization_total_results"), int)
        else 0
    )
    observed_materialization_result_count = (
        report.get("materialization_result_count")
        if isinstance(report.get("materialization_result_count"), int)
        else 0
    )
    observed_materialized_kv_count = (
        report.get("materialized_kv_count")
        if isinstance(report.get("materialized_kv_count"), int)
        else 0
    )
    observed_candidate_event_materialized_count = (
        report.get("materialization_candidate_event_count")
        if isinstance(report.get("materialization_candidate_event_count"), int)
        else 0
    )
    observed_fake_materialized_count = (
        report.get("materialization_fake_count")
        if isinstance(report.get("materialization_fake_count"), int)
        else 0
    )
    observed_guarded_noop_count = (
        report.get("materialization_guarded_noop_count")
        if isinstance(report.get("materialization_guarded_noop_count"), int)
        else 0
    )
    observed_blocked_count = (
        report.get("materialization_blocked_count")
        if isinstance(report.get("materialization_blocked_count"), int)
        else 0
    )
    observed_error_count = (
        report.get("materialization_error_count")
        if isinstance(report.get("materialization_error_count"), int)
        else 0
    )
    observed_host_backup_copy_materialized_count = (
        report.get("materialization_host_backup_copy_count")
        if isinstance(report.get("materialization_host_backup_copy_count"), int)
        else 0
    )
    observed_host_backup_copy_executed_count = safety_counts[
        "host_backup_copy_executed_count"
    ]
    observed_kv_pool_read_count = safety_counts["kv_pool_read_count"]
    observed_kv_snapshot_count = safety_counts["kv_snapshot_count"]

    blocker_state_by_reason = {
        "not_readonly_report": "blocked_not_readonly_report",
        "overall_safety_not_pass": "blocked_overall_safety_not_pass",
        "policy_dry_run_missing": "blocked_policy_dry_run_missing",
        "materialization_summary_missing": "blocked_materialization_summary_missing",
        "no_materialization_results": "blocked_no_materialization_results",
        "no_materialized_kv": "blocked_no_materialized_kv",
        "guarded_noop_present": "blocked_guarded_noop_present",
        "materialization_blocked": "blocked_materialization_blocked",
        "materialization_error": "blocked_materialization_error",
        "host_backup_copy_executed": "blocked_host_backup_copy_executed",
        "kv_pool_read": "blocked_kv_pool_read",
        "kv_snapshot": "blocked_kv_snapshot",
        "safety_counter_nonzero": "blocked_safety_counter_nonzero",
    }
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = [
        "metadata_only_readiness_does_not_connect_attention"
    ]

    if not report_generated_from_readonly_inputs:
        blocking_reasons.append("not_readonly_report")
    if observed_overall_safety_status != "pass":
        blocking_reasons.append("overall_safety_not_pass")
    if (
        not observed_policy_dry_run_included
        or observed_policy_dry_run_total_events <= 0
    ):
        blocking_reasons.append("policy_dry_run_missing")
    if not observed_materialization_summary_included:
        blocking_reasons.append("materialization_summary_missing")
    if (
        observed_materialization_summary_included
        and observed_materialization_total_results <= 0
    ):
        blocking_reasons.append("no_materialization_results")
    if (
        observed_materialization_summary_included
        and observed_materialization_result_count <= 0
    ):
        blocking_reasons.append("no_materialization_results")
    if observed_materialized_kv_count <= 0:
        blocking_reasons.append("no_materialized_kv")
    if (
        observed_candidate_event_materialized_count <= 0
        and observed_fake_materialized_count <= 0
    ):
        blocking_reasons.append("no_materialized_kv")
    if observed_guarded_noop_count > 0:
        blocking_reasons.append("guarded_noop_present")
    if observed_blocked_count > 0:
        blocking_reasons.append("materialization_blocked")
    if observed_error_count > 0:
        blocking_reasons.append("materialization_error")
    if observed_host_backup_copy_executed_count > 0:
        blocking_reasons.append("host_backup_copy_executed")
    if observed_kv_pool_read_count > 0:
        blocking_reasons.append("kv_pool_read")
    if observed_kv_snapshot_count > 0:
        blocking_reasons.append("kv_snapshot")
    if any(
        safety_counts[key] != 0
        for key in (
            "source_mutated_true_count",
            "attention_override_true_count",
            "kv_cache_mutation_true_count",
            "runtime_writeback_true_count",
            "scheduler_policy_noop_false_count",
        )
    ):
        blocking_reasons.append("safety_counter_nonzero")

    # Preserve order while dropping duplicate blockers from overlapping rules.
    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    ready_for_attention_connection = not blocking_reasons
    if ready_for_attention_connection:
        readiness_state = "ready_for_attention_connection_metadata_only"
        readiness_reasons = ["readonly_metadata_materialization_report_ready"]
    elif len(blocking_reasons) == 1:
        readiness_state = blocker_state_by_reason[blocking_reasons[0]]
        readiness_reasons = []
    else:
        readiness_state = "blocked_multiple_reasons"
        readiness_reasons = []

    return {
        "readiness_type": "relaykv_readonly_attention_readiness",
        "ready_for_attention_connection": ready_for_attention_connection,
        "readiness_state": readiness_state,
        "readiness_reasons": readiness_reasons,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "observed_overall_safety_status": observed_overall_safety_status,
        "observed_materialization_summary_included": (
            observed_materialization_summary_included
        ),
        "observed_materialization_total_results": (
            observed_materialization_total_results
        ),
        "observed_materialization_result_count": (
            observed_materialization_result_count
        ),
        "observed_materialized_kv_count": observed_materialized_kv_count,
        "observed_candidate_event_materialized_count": (
            observed_candidate_event_materialized_count
        ),
        "observed_fake_materialized_count": observed_fake_materialized_count,
        "observed_guarded_noop_count": observed_guarded_noop_count,
        "observed_host_backup_copy_materialized_count": (
            observed_host_backup_copy_materialized_count
        ),
        "observed_host_backup_copy_executed_count": (
            observed_host_backup_copy_executed_count
        ),
        "observed_kv_pool_read_count": observed_kv_pool_read_count,
        "observed_kv_snapshot_count": observed_kv_snapshot_count,
        "report_generated_from_readonly_inputs": (
            report_generated_from_readonly_inputs
        ),
        **safety_counts,
    }


def assess_relaykv_host_backup_copy_readiness_for_smoke(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess read-only readiness for host backup copy boundary requests.

    This consumes a precomputed report dictionary only. It does not execute
    host backup copy, materialize KV data, read KV pools, snapshot KV, connect
    attention, alter scheduler decisions, or write runtime state.
    """

    if not isinstance(report, Mapping):
        raise TypeError("RelayKV host backup copy readiness report must be a mapping")

    safety_counter_keys = (
        *_REPORT_SAFETY_COUNTER_KEYS,
        *_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS,
    )
    safety_counts = {
        key: report.get(key) if isinstance(report.get(key), int) else 0
        for key in safety_counter_keys
    }

    observed_report_generated_from_readonly_inputs = (
        report.get("report_generated_from_readonly_inputs") is True
    )
    observed_overall_safety_status = str(
        report.get("overall_safety_status", "unknown")
    )
    observed_policy_dry_run_included = report.get("policy_dry_run_included") is True
    observed_policy_dry_run_total_events = (
        report.get("policy_dry_run_total_events")
        if isinstance(report.get("policy_dry_run_total_events"), int)
        else 0
    )
    observed_materialization_summary_included = (
        report.get("materialization_summary_included") is True
    )
    observed_materialization_total_results = (
        report.get("materialization_total_results")
        if isinstance(report.get("materialization_total_results"), int)
        else 0
    )
    observed_materialization_result_count = (
        report.get("materialization_result_count")
        if isinstance(report.get("materialization_result_count"), int)
        else 0
    )
    observed_materialized_kv_count = (
        report.get("materialized_kv_count")
        if isinstance(report.get("materialized_kv_count"), int)
        else 0
    )
    observed_candidate_event_materialized_count = (
        report.get("materialization_candidate_event_count")
        if isinstance(report.get("materialization_candidate_event_count"), int)
        else 0
    )
    observed_fake_materialized_count = (
        report.get("materialization_fake_count")
        if isinstance(report.get("materialization_fake_count"), int)
        else 0
    )
    observed_guarded_noop_count = (
        report.get("materialization_guarded_noop_count")
        if isinstance(report.get("materialization_guarded_noop_count"), int)
        else 0
    )
    observed_materialization_blocked_count = (
        report.get("materialization_blocked_count")
        if isinstance(report.get("materialization_blocked_count"), int)
        else 0
    )
    observed_materialization_error_count = (
        report.get("materialization_error_count")
        if isinstance(report.get("materialization_error_count"), int)
        else 0
    )
    observed_host_backup_copy_executed_count = safety_counts[
        "host_backup_copy_executed_count"
    ]
    observed_kv_pool_read_count = safety_counts["kv_pool_read_count"]
    observed_kv_snapshot_count = safety_counts["kv_snapshot_count"]

    blocker_state_by_reason = {
        "not_readonly_report": "blocked_not_readonly_report",
        "overall_safety_not_pass": "blocked_overall_safety_not_pass",
        "policy_dry_run_missing": "blocked_policy_dry_run_missing",
        "materialization_summary_missing": "blocked_materialization_summary_missing",
        "no_materialization_results": "blocked_no_materialization_results",
        "no_materialized_kv": "blocked_no_materialized_kv",
        "candidate_event_materialization_missing": (
            "blocked_candidate_event_materialization_missing"
        ),
        "guarded_noop_present": "blocked_guarded_noop_present",
        "materialization_blocked": "blocked_materialization_blocked",
        "materialization_error": "blocked_materialization_error",
        "host_backup_copy_already_executed": (
            "blocked_host_backup_copy_already_executed"
        ),
        "kv_pool_read_observed": "blocked_kv_pool_read_observed",
        "kv_snapshot_observed": "blocked_kv_snapshot_observed",
        "safety_counter_nonzero": "blocked_safety_counter_nonzero",
    }
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = [
        "readiness_only_does_not_execute_host_backup_copy"
    ]

    if not observed_report_generated_from_readonly_inputs:
        blocking_reasons.append("not_readonly_report")
    if observed_overall_safety_status != "pass":
        blocking_reasons.append("overall_safety_not_pass")
    if (
        not observed_policy_dry_run_included
        or observed_policy_dry_run_total_events <= 0
    ):
        blocking_reasons.append("policy_dry_run_missing")
    if not observed_materialization_summary_included:
        blocking_reasons.append("materialization_summary_missing")
    if (
        observed_materialization_summary_included
        and observed_materialization_total_results <= 0
    ):
        blocking_reasons.append("no_materialization_results")
    if (
        observed_materialization_summary_included
        and observed_materialization_result_count <= 0
    ):
        blocking_reasons.append("no_materialization_results")
    if observed_materialized_kv_count <= 0:
        blocking_reasons.append("no_materialized_kv")
    if observed_candidate_event_materialized_count <= 0:
        blocking_reasons.append("candidate_event_materialization_missing")
    if observed_guarded_noop_count > 0:
        blocking_reasons.append("guarded_noop_present")
    if observed_materialization_blocked_count > 0:
        blocking_reasons.append("materialization_blocked")
    if observed_materialization_error_count > 0:
        blocking_reasons.append("materialization_error")
    if observed_host_backup_copy_executed_count > 0:
        blocking_reasons.append("host_backup_copy_already_executed")
    if observed_kv_pool_read_count > 0:
        blocking_reasons.append("kv_pool_read_observed")
    if observed_kv_snapshot_count > 0:
        blocking_reasons.append("kv_snapshot_observed")
    if any(
        safety_counts[key] != 0
        for key in (
            "source_mutated_true_count",
            "attention_override_true_count",
            "kv_cache_mutation_true_count",
            "runtime_writeback_true_count",
            "scheduler_policy_noop_false_count",
        )
    ):
        blocking_reasons.append("safety_counter_nonzero")

    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    ready_for_host_backup_copy_boundary = not blocking_reasons
    if ready_for_host_backup_copy_boundary:
        readiness_state = "ready_for_host_backup_copy_boundary_smoke"
        readiness_reasons = ["readonly_candidate_event_materialization_ready"]
    elif len(blocking_reasons) == 1:
        readiness_state = blocker_state_by_reason[blocking_reasons[0]]
        readiness_reasons = []
    else:
        readiness_state = "blocked_multiple_reasons"
        readiness_reasons = []

    return {
        "readiness_type": "relaykv_host_backup_copy_readiness",
        "ready_for_host_backup_copy_boundary": ready_for_host_backup_copy_boundary,
        "readiness_state": readiness_state,
        "readiness_reasons": readiness_reasons,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "observed_overall_safety_status": observed_overall_safety_status,
        "observed_report_generated_from_readonly_inputs": (
            observed_report_generated_from_readonly_inputs
        ),
        "observed_policy_dry_run_included": observed_policy_dry_run_included,
        "observed_policy_dry_run_total_events": observed_policy_dry_run_total_events,
        "observed_materialization_summary_included": (
            observed_materialization_summary_included
        ),
        "observed_materialization_total_results": (
            observed_materialization_total_results
        ),
        "observed_materialization_result_count": (
            observed_materialization_result_count
        ),
        "observed_materialized_kv_count": observed_materialized_kv_count,
        "observed_candidate_event_materialized_count": (
            observed_candidate_event_materialized_count
        ),
        "observed_fake_materialized_count": observed_fake_materialized_count,
        "observed_guarded_noop_count": observed_guarded_noop_count,
        "observed_materialization_blocked_count": (
            observed_materialization_blocked_count
        ),
        "observed_materialization_error_count": observed_materialization_error_count,
        "observed_host_backup_copy_executed_count": (
            observed_host_backup_copy_executed_count
        ),
        "observed_kv_pool_read_count": observed_kv_pool_read_count,
        "observed_kv_snapshot_count": observed_kv_snapshot_count,
        **safety_counts,
    }


def assess_relaykv_actual_host_backup_copy_readiness_for_smoke(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess read-only readiness for a future actual host backup copy smoke.

    This consumes a precomputed report dictionary only. It does not execute
    host backup copy, materialize KV data, read KV pools, snapshot KV, connect
    attention, alter scheduler decisions, or write runtime state.
    """

    if not isinstance(report, Mapping):
        raise TypeError(
            "RelayKV actual host backup copy readiness report must be a mapping"
        )

    safety_counter_keys = (
        *_REPORT_SAFETY_COUNTER_KEYS,
        *_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS,
    )
    safety_counts = {
        key: report.get(key) if isinstance(report.get(key), int) else 0
        for key in safety_counter_keys
    }

    observed_report_generated_from_readonly_inputs = (
        report.get("report_generated_from_readonly_inputs") is True
    )
    observed_overall_safety_status = str(
        report.get("overall_safety_status", "unknown")
    )
    observed_policy_dry_run_included = report.get("policy_dry_run_included") is True
    observed_policy_dry_run_total_events = (
        report.get("policy_dry_run_total_events")
        if isinstance(report.get("policy_dry_run_total_events"), int)
        else 0
    )
    observed_materialization_summary_included = (
        report.get("materialization_summary_included") is True
    )
    observed_materialization_result_count = (
        report.get("materialization_result_count")
        if isinstance(report.get("materialization_result_count"), int)
        else 0
    )
    observed_host_backup_copy_request_summary_included = (
        report.get("host_backup_copy_request_summary_included") is True
    )
    observed_host_backup_copy_request_ready_count = (
        report.get("host_backup_copy_request_ready_count")
        if isinstance(report.get("host_backup_copy_request_ready_count"), int)
        else 0
    )
    observed_host_backup_copy_request_blocked_count = (
        report.get("host_backup_copy_request_blocked_count")
        if isinstance(report.get("host_backup_copy_request_blocked_count"), int)
        else 0
    )
    observed_host_backup_copy_boundary_result_summary_included = (
        report.get("host_backup_copy_boundary_result_summary_included") is True
    )
    observed_host_backup_copy_boundary_noop_count = (
        report.get("host_backup_copy_boundary_noop_count")
        if isinstance(report.get("host_backup_copy_boundary_noop_count"), int)
        else 0
    )
    observed_host_backup_copy_boundary_blocked_count = (
        report.get("host_backup_copy_boundary_blocked_count")
        if isinstance(report.get("host_backup_copy_boundary_blocked_count"), int)
        else 0
    )
    observed_host_backup_copy_boundary_error_count = (
        report.get("host_backup_copy_boundary_error_count")
        if isinstance(report.get("host_backup_copy_boundary_error_count"), int)
        else 0
    )
    observed_host_backup_copy_executed_count = safety_counts[
        "host_backup_copy_executed_count"
    ]
    observed_kv_pool_read_count = safety_counts["kv_pool_read_count"]
    observed_kv_snapshot_count = safety_counts["kv_snapshot_count"]

    blocker_state_by_reason = {
        "not_readonly_report": "blocked_not_readonly_report",
        "overall_safety_not_pass": "blocked_overall_safety_not_pass",
        "policy_dry_run_missing": "blocked_policy_dry_run_missing",
        "materialization_summary_missing": "blocked_materialization_summary_missing",
        "copy_request_summary_missing": "blocked_copy_request_summary_missing",
        "no_copy_requests_ready": "blocked_no_copy_requests_ready",
        "copy_request_blocked": "blocked_copy_request_blocked",
        "boundary_result_summary_missing": "blocked_boundary_result_summary_missing",
        "no_boundary_noop_results": "blocked_no_boundary_noop_results",
        "boundary_result_blocked": "blocked_boundary_result_blocked",
        "boundary_result_error": "blocked_boundary_result_error",
        "host_backup_copy_already_executed": (
            "blocked_host_backup_copy_already_executed"
        ),
        "kv_pool_read_observed": "blocked_kv_pool_read_observed",
        "kv_snapshot_observed": "blocked_kv_snapshot_observed",
        "safety_counter_nonzero": "blocked_safety_counter_nonzero",
    }
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = [
        "readiness_only_does_not_execute_actual_host_backup_copy"
    ]

    if not observed_report_generated_from_readonly_inputs:
        blocking_reasons.append("not_readonly_report")
    if observed_overall_safety_status != "pass":
        blocking_reasons.append("overall_safety_not_pass")
    if (
        not observed_policy_dry_run_included
        or observed_policy_dry_run_total_events <= 0
    ):
        blocking_reasons.append("policy_dry_run_missing")
    if (
        not observed_materialization_summary_included
        or observed_materialization_result_count <= 0
    ):
        blocking_reasons.append("materialization_summary_missing")
    if not observed_host_backup_copy_request_summary_included:
        blocking_reasons.append("copy_request_summary_missing")
    if observed_host_backup_copy_request_ready_count <= 0:
        blocking_reasons.append("no_copy_requests_ready")
    if observed_host_backup_copy_request_blocked_count > 0:
        blocking_reasons.append("copy_request_blocked")
    if not observed_host_backup_copy_boundary_result_summary_included:
        blocking_reasons.append("boundary_result_summary_missing")
    if observed_host_backup_copy_boundary_noop_count <= 0:
        blocking_reasons.append("no_boundary_noop_results")
    if observed_host_backup_copy_boundary_blocked_count > 0:
        blocking_reasons.append("boundary_result_blocked")
    if observed_host_backup_copy_boundary_error_count > 0:
        blocking_reasons.append("boundary_result_error")
    if observed_host_backup_copy_executed_count > 0:
        blocking_reasons.append("host_backup_copy_already_executed")
    if observed_kv_pool_read_count > 0:
        blocking_reasons.append("kv_pool_read_observed")
    if observed_kv_snapshot_count > 0:
        blocking_reasons.append("kv_snapshot_observed")
    if any(
        safety_counts[key] != 0
        for key in (
            "source_mutated_true_count",
            "attention_override_true_count",
            "kv_cache_mutation_true_count",
            "runtime_writeback_true_count",
            "scheduler_policy_noop_false_count",
        )
    ):
        blocking_reasons.append("safety_counter_nonzero")

    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    ready_for_actual_host_backup_copy = not blocking_reasons
    if ready_for_actual_host_backup_copy:
        readiness_state = "ready_for_actual_host_backup_copy_smoke_boundary_complete"
        readiness_reasons = ["readonly_copy_boundary_report_ready"]
    elif len(blocking_reasons) == 1:
        readiness_state = blocker_state_by_reason[blocking_reasons[0]]
        readiness_reasons = []
    else:
        readiness_state = "blocked_multiple_reasons"
        readiness_reasons = []

    return {
        "readiness_type": "relaykv_actual_host_backup_copy_readiness",
        "ready_for_actual_host_backup_copy": ready_for_actual_host_backup_copy,
        "readiness_state": readiness_state,
        "readiness_reasons": readiness_reasons,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "observed_overall_safety_status": observed_overall_safety_status,
        "observed_report_generated_from_readonly_inputs": (
            observed_report_generated_from_readonly_inputs
        ),
        "observed_policy_dry_run_included": observed_policy_dry_run_included,
        "observed_policy_dry_run_total_events": observed_policy_dry_run_total_events,
        "observed_materialization_summary_included": (
            observed_materialization_summary_included
        ),
        "observed_materialization_result_count": (
            observed_materialization_result_count
        ),
        "observed_host_backup_copy_request_summary_included": (
            observed_host_backup_copy_request_summary_included
        ),
        "observed_host_backup_copy_request_ready_count": (
            observed_host_backup_copy_request_ready_count
        ),
        "observed_host_backup_copy_request_blocked_count": (
            observed_host_backup_copy_request_blocked_count
        ),
        "observed_host_backup_copy_boundary_result_summary_included": (
            observed_host_backup_copy_boundary_result_summary_included
        ),
        "observed_host_backup_copy_boundary_noop_count": (
            observed_host_backup_copy_boundary_noop_count
        ),
        "observed_host_backup_copy_boundary_blocked_count": (
            observed_host_backup_copy_boundary_blocked_count
        ),
        "observed_host_backup_copy_boundary_error_count": (
            observed_host_backup_copy_boundary_error_count
        ),
        "observed_host_backup_copy_executed_count": (
            observed_host_backup_copy_executed_count
        ),
        "observed_kv_pool_read_count": observed_kv_pool_read_count,
        "observed_kv_snapshot_count": observed_kv_snapshot_count,
        **safety_counts,
    }


def _policy_dry_run_int_config(
    policy_config: Mapping[str, Any],
    key: str,
) -> int:
    value = policy_config.get(key)
    if not isinstance(value, int):
        raise TypeError(f"RelayKV dry-run policy config {key} must be an int")
    return value


def _policy_dry_run_block_ids(
    block_metadata: Mapping[str, Any],
    key: str,
) -> list[Any]:
    value = block_metadata.get(key, ())
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"RelayKV dry-run block metadata {key} must be list or tuple")
    return list(value)


def build_relaykv_policy_dry_run_events_for_smoke(
    runtime_payloads: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    block_metadata_by_request: Mapping[str, Mapping[str, Any]],
    policy_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build smoke-only RelayKV policy dry-run events from read-only metadata.

    This is a log/report layer only. It does not read KV pools, create
    snapshots, execute host backup copy, connect attention, change scheduler
    decisions, or write runtime state.
    """

    if not isinstance(runtime_payloads, (list, tuple)):
        raise TypeError("runtime_payloads must be a list or tuple")
    if not isinstance(block_metadata_by_request, Mapping):
        raise TypeError("block_metadata_by_request must be a mapping")
    if not isinstance(policy_config, Mapping):
        raise TypeError("policy_config must be a mapping")

    kv_budget_tokens = _policy_dry_run_int_config(policy_config, "kv_budget_tokens")
    recent_tokens = _policy_dry_run_int_config(policy_config, "recent_tokens")
    anchor_tokens = _policy_dry_run_int_config(policy_config, "anchor_tokens")
    transient_tokens = _policy_dry_run_int_config(policy_config, "transient_tokens")
    retrieval_top_k = _policy_dry_run_int_config(policy_config, "retrieval_top_k")
    if retrieval_top_k < 0:
        raise ValueError("RelayKV dry-run retrieval_top_k must be non-negative")
    layer_budget_policy = policy_config.get("layer_budget_policy")
    if not isinstance(layer_budget_policy, str):
        raise TypeError("RelayKV dry-run layer_budget_policy must be a string")

    retrieval_budget_tokens = max(
        kv_budget_tokens - recent_tokens - anchor_tokens - transient_tokens,
        0,
    )
    events: list[dict[str, Any]] = []
    for payload in runtime_payloads:
        if not isinstance(payload, Mapping):
            raise TypeError("RelayKV dry-run runtime payloads must be mappings")
        request_id = _event_value(payload, "request_id")
        layer_id = _event_layer_value(payload)
        if request_id is None or layer_id is None:
            raise ValueError("RelayKV dry-run payload requires request_id and layer_id")

        block_metadata = block_metadata_by_request.get(str(request_id), {})
        if not isinstance(block_metadata, Mapping):
            raise TypeError("RelayKV dry-run block metadata must be mappings")
        anchor_block_ids = _policy_dry_run_block_ids(
            block_metadata, "anchor_block_ids"
        )
        recent_block_ids = _policy_dry_run_block_ids(
            block_metadata, "recent_block_ids"
        )
        candidate_block_ids = _policy_dry_run_block_ids(
            block_metadata, "candidate_block_ids"
        )
        selected_block_ids = candidate_block_ids[:retrieval_top_k]
        kv_classes_present: list[str] = []
        if recent_block_ids:
            kv_classes_present.append("RECENT")
        if anchor_block_ids:
            kv_classes_present.append("ANCHOR")
        if selected_block_ids:
            kv_classes_present.append("RETRIEVED")
        if len(candidate_block_ids) > len(selected_block_ids):
            kv_classes_present.append("COLD_CANDIDATE")

        events.append(
            {
                "event_type": "relaykv_policy_dry_run",
                "request_id": str(request_id),
                "req_pool_idx": _event_req_pool_idx_value(payload),
                "seq_len": _event_value(payload, "seq_len"),
                "layer_id": layer_id,
                "policy_state": "dry_run",
                "kv_budget_tokens": kv_budget_tokens,
                "recent_tokens": recent_tokens,
                "anchor_tokens": anchor_tokens,
                "transient_tokens": transient_tokens,
                "retrieval_budget_tokens": retrieval_budget_tokens,
                "candidate_block_ids": candidate_block_ids,
                "selected_block_ids": selected_block_ids,
                "anchor_block_ids": anchor_block_ids,
                "recent_block_ids": recent_block_ids,
                "kv_classes_present": kv_classes_present,
                "layer_budget_policy": layer_budget_policy,
                "retrieval_top_k": retrieval_top_k,
                "source": "readonly_metadata_policy_dry_run",
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
            }
        )
    return events


def summarize_relaykv_policy_dry_run_events_for_smoke(
    events: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize smoke-only RelayKV policy dry-run events."""

    if not isinstance(events, (list, tuple)):
        raise TypeError("RelayKV dry-run events must be a list or tuple")
    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    kv_class_counts: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "source_mutated_true_count": 0,
            "attention_override_true_count": 0,
            "kv_cache_mutation_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
        }
    )
    event_type_counts: Counter[str] = Counter()

    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("RelayKV dry-run event must be a mapping")
        event_type_counts[str(_event_value(event, "event_type"))] += 1
        per_request[str(_event_value(event, "request_id"))] += 1
        per_layer[str(_event_layer_value(event))] += 1
        kv_classes_present = _event_value(event, "kv_classes_present")
        if isinstance(kv_classes_present, (list, tuple)):
            for kv_class in kv_classes_present:
                kv_class_counts[str(kv_class)] += 1
        _increment_join_safety_counts(safety_counts, event)

    return {
        "total_events": len(events),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "kv_class_counts": dict(sorted(kv_class_counts.items())),
        **dict(safety_counts),
    }


def _fake_materialization_block_ids(
    event: Mapping[str, Any],
    key: str,
) -> list[Any]:
    value = _event_value(event, key)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"RelayKV fake materialization {key} must be list or tuple")
    return list(value)


def build_relaykv_fake_materialization_results_for_smoke(
    policy_dry_run_events: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    readiness: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build fake materialization result payloads from dry-run events only.

    This validates result schema without reading KV pools, taking snapshots,
    executing host backup copy, connecting attention, mutating scheduler state,
    or writing runtime state.
    """

    if not isinstance(policy_dry_run_events, (list, tuple)):
        raise TypeError("policy_dry_run_events must be a list or tuple")
    if readiness is not None and not isinstance(readiness, Mapping):
        raise TypeError("readiness must be a mapping or None")

    readiness_ready = (
        True
        if readiness is None
        else readiness.get("ready_for_materialization") is True
    )
    readiness_blocking_reasons: list[str] = []
    if readiness is not None:
        blocking_reasons = readiness.get("blocking_reasons")
        if isinstance(blocking_reasons, (list, tuple)):
            readiness_blocking_reasons = [str(reason) for reason in blocking_reasons]
    if readiness is not None and not readiness_ready and not readiness_blocking_reasons:
        readiness_blocking_reasons = ["readiness_not_met"]

    results: list[dict[str, Any]] = []
    for event in policy_dry_run_events:
        if not isinstance(event, Mapping):
            raise TypeError("RelayKV fake materialization events must be mappings")
        selected_block_ids = _fake_materialization_block_ids(
            event, "selected_block_ids"
        )
        anchor_block_ids = _fake_materialization_block_ids(event, "anchor_block_ids")
        recent_block_ids = _fake_materialization_block_ids(event, "recent_block_ids")
        candidate_block_ids = _fake_materialization_block_ids(
            event, "candidate_block_ids"
        )
        blocking_reasons: list[str] = []
        warning_reasons: list[str] = []
        materialized_block_ids: list[Any] = []
        skipped_block_ids: list[Any] = []
        retrieved_block_ids: list[Any] = []

        if not readiness_ready:
            materialization_state = "blocked"
            blocking_reasons = list(readiness_blocking_reasons)
            skipped_block_ids = list(selected_block_ids)
        elif not selected_block_ids:
            materialization_state = "skipped"
            warning_reasons.append("no_selected_blocks")
        else:
            materialization_state = "fake_materialized"
            materialized_block_ids = list(selected_block_ids)
            retrieved_block_ids = list(selected_block_ids)
        if readiness is None:
            warning_reasons.append("readiness_not_provided")

        results.append(
            {
                "event_type": "relaykv_materialization_result",
                "request_id": _event_value(event, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(event),
                "seq_len": _event_value(event, "seq_len"),
                "layer_id": _event_layer_value(event),
                "materialization_state": materialization_state,
                "materialization_mode": "fake",
                "selected_block_ids": selected_block_ids,
                "materialized_block_ids": materialized_block_ids,
                "skipped_block_ids": skipped_block_ids,
                "fallback_block_ids": [],
                "anchor_block_ids": anchor_block_ids,
                "recent_block_ids": recent_block_ids,
                "retrieved_block_ids": retrieved_block_ids,
                "candidate_block_ids": candidate_block_ids,
                "materialized_kv_count": len(materialized_block_ids),
                "materialized_token_count": 0,
                "source": "policy_dry_run_fake_materialization",
                "blocking_reasons": blocking_reasons,
                "warning_reasons": warning_reasons,
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "host_backup_copy_executed": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
            }
        )
    return results


def _guarded_noop_materialization_block_ids_from_aliases(
    event: Mapping[str, Any],
    keys: tuple[str, ...],
) -> list[Any] | None:
    for key in keys:
        value = _event_value(event, key)
        if value is None:
            continue
        if key in ("block_id", "candidate_block_id"):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"RelayKV guarded no-op materialization {key} must be list or tuple"
            )
        return list(value)
    return None


def build_relaykv_guarded_noop_materialization_results_for_smoke(
    host_backup_candidate_events: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    readiness: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build guarded no-op materialization results from candidate events only.

    This validates candidate-event-to-materialization-result schema without
    reading KV pools, taking snapshots, executing host backup copy, connecting
    attention, mutating scheduler state, or writing runtime state.
    """

    if not isinstance(host_backup_candidate_events, (list, tuple)):
        raise TypeError("host_backup_candidate_events must be a list or tuple")
    if readiness is not None and not isinstance(readiness, Mapping):
        raise TypeError("readiness must be a mapping or None")

    readiness_ready = (
        True
        if readiness is None
        else readiness.get("ready_for_materialization") is True
    )
    readiness_blocking_reasons: list[str] = []
    if readiness is not None:
        blocking_reasons = readiness.get("blocking_reasons")
        if isinstance(blocking_reasons, (list, tuple)):
            readiness_blocking_reasons = [str(reason) for reason in blocking_reasons]
    if readiness is not None and not readiness_ready and not readiness_blocking_reasons:
        readiness_blocking_reasons = ["readiness_not_met"]

    results: list[dict[str, Any]] = []
    for event in host_backup_candidate_events:
        if not isinstance(event, Mapping):
            raise TypeError(
                "RelayKV guarded no-op materialization events must be mappings"
            )

        selected_block_ids = (
            _guarded_noop_materialization_block_ids_from_aliases(
                event,
                (
                    "selected_block_ids",
                    "copied_block_ids",
                    "block_ids",
                    "block_id",
                    "candidate_block_id",
                    "candidate_block_ids",
                ),
            )
            or []
        )
        candidate_block_ids = (
            _guarded_noop_materialization_block_ids_from_aliases(
                event, ("candidate_block_ids",)
            )
            or list(selected_block_ids)
        )
        anchor_block_ids = _fake_materialization_block_ids(event, "anchor_block_ids")
        recent_block_ids = _fake_materialization_block_ids(event, "recent_block_ids")

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = ["guarded_noop_no_kv_copy"]
        skipped_block_ids = list(selected_block_ids)

        if not readiness_ready:
            materialization_state = "blocked"
            blocking_reasons = list(readiness_blocking_reasons)
        elif not selected_block_ids:
            materialization_state = "skipped"
            warning_reasons.append("no_selected_blocks")
            skipped_block_ids = []
        else:
            materialization_state = "guarded_noop"
        if readiness is None:
            warning_reasons.append("readiness_not_provided")

        results.append(
            {
                "event_type": "relaykv_materialization_result",
                "request_id": _event_value(event, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(event),
                "seq_len": _event_value(event, "seq_len"),
                "layer_id": _event_layer_value(event),
                "materialization_state": materialization_state,
                "materialization_mode": "noop_guarded",
                "selected_block_ids": selected_block_ids,
                "materialized_block_ids": [],
                "skipped_block_ids": skipped_block_ids,
                "fallback_block_ids": [],
                "anchor_block_ids": anchor_block_ids,
                "recent_block_ids": recent_block_ids,
                "retrieved_block_ids": [],
                "candidate_block_ids": candidate_block_ids,
                "materialized_kv_count": 0,
                "materialized_token_count": 0,
                "source": "host_backup_candidate_guarded_noop_materialization",
                "blocking_reasons": blocking_reasons,
                "warning_reasons": warning_reasons,
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "host_backup_copy_executed": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
            }
        )
    return results


def _candidate_event_materialization_block_ids_from_aliases(
    event: Mapping[str, Any],
    keys: tuple[str, ...],
) -> list[Any] | None:
    for key in keys:
        value = _event_value(event, key)
        if value is None:
            continue
        if key in ("block_id", "candidate_block_id"):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "RelayKV candidate-event materialization "
                f"{key} must be list or tuple"
            )
        return list(value)
    return None


def build_relaykv_candidate_event_materialization_results_for_smoke(
    host_backup_candidate_events: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    readiness: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build metadata-only materialization results from candidate events.

    This treats candidate event payloads as materialized candidate metadata
    without reading KV pools, taking snapshots, executing host backup copy,
    connecting attention, mutating scheduler state, or writing runtime state.
    """

    if not isinstance(host_backup_candidate_events, (list, tuple)):
        raise TypeError("host_backup_candidate_events must be a list or tuple")
    if readiness is not None and not isinstance(readiness, Mapping):
        raise TypeError("readiness must be a mapping or None")

    readiness_ready = (
        True
        if readiness is None
        else readiness.get("ready_for_materialization") is True
    )
    readiness_blocking_reasons: list[str] = []
    if readiness is not None:
        blocking_reasons = readiness.get("blocking_reasons")
        if isinstance(blocking_reasons, (list, tuple)):
            readiness_blocking_reasons = [str(reason) for reason in blocking_reasons]
    if readiness is not None and not readiness_ready and not readiness_blocking_reasons:
        readiness_blocking_reasons = ["readiness_not_met"]

    results: list[dict[str, Any]] = []
    for event in host_backup_candidate_events:
        if not isinstance(event, Mapping):
            raise TypeError(
                "RelayKV candidate-event materialization events must be mappings"
            )

        selected_block_ids = (
            _candidate_event_materialization_block_ids_from_aliases(
                event,
                (
                    "selected_block_ids",
                    "materialized_block_ids",
                    "retrieved_block_ids",
                    "copied_block_ids",
                    "block_ids",
                    "block_id",
                    "candidate_block_id",
                    "candidate_block_ids",
                ),
            )
            or []
        )
        materialized_block_ids = (
            _candidate_event_materialization_block_ids_from_aliases(
                event, ("materialized_block_ids",)
            )
            or list(selected_block_ids)
        )
        retrieved_block_ids = (
            _candidate_event_materialization_block_ids_from_aliases(
                event, ("retrieved_block_ids",)
            )
            or list(materialized_block_ids)
        )
        candidate_block_ids = (
            _candidate_event_materialization_block_ids_from_aliases(
                event, ("candidate_block_ids",)
            )
            or list(selected_block_ids)
        )
        anchor_block_ids = _fake_materialization_block_ids(event, "anchor_block_ids")
        recent_block_ids = _fake_materialization_block_ids(event, "recent_block_ids")

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = ["candidate_event_metadata_only_no_kv_copy"]
        skipped_block_ids: list[Any] = []

        if not readiness_ready:
            materialization_state = "blocked"
            blocking_reasons = list(readiness_blocking_reasons)
            skipped_block_ids = list(selected_block_ids)
            materialized_block_ids = []
            retrieved_block_ids = []
        elif not selected_block_ids or not materialized_block_ids:
            materialization_state = "skipped"
            warning_reasons.append("no_selected_blocks")
            materialized_block_ids = []
            retrieved_block_ids = []
        else:
            materialization_state = "candidate_event_materialized"
        if readiness is None:
            warning_reasons.append("readiness_not_provided")

        results.append(
            {
                "event_type": "relaykv_materialization_result",
                "request_id": _event_value(event, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(event),
                "seq_len": _event_value(event, "seq_len"),
                "layer_id": _event_layer_value(event),
                "materialization_state": materialization_state,
                "materialization_mode": "candidate_event",
                "selected_block_ids": selected_block_ids,
                "materialized_block_ids": materialized_block_ids,
                "skipped_block_ids": skipped_block_ids,
                "fallback_block_ids": [],
                "anchor_block_ids": anchor_block_ids,
                "recent_block_ids": recent_block_ids,
                "retrieved_block_ids": retrieved_block_ids,
                "candidate_block_ids": candidate_block_ids,
                "materialized_kv_count": len(materialized_block_ids),
                "materialized_token_count": 0,
                "source": "host_backup_candidate_event_materialization",
                "blocking_reasons": blocking_reasons,
                "warning_reasons": warning_reasons,
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "host_backup_copy_executed": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
            }
        )
    return results


def summarize_relaykv_materialization_results_for_smoke(
    results: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize fake RelayKV materialization result payloads."""

    if not isinstance(results, (list, tuple)):
        raise TypeError("RelayKV materialization results must be a list or tuple")

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_state: Counter[str] = Counter()
    per_mode: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "source_mutated_true_count": 0,
            "attention_override_true_count": 0,
            "kv_cache_mutation_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "host_backup_copy_executed_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
        }
    )
    materialized_kv_count = 0
    materialized_token_count = 0
    guarded_noop_count = 0
    fake_materialized_count = 0
    candidate_event_materialized_count = 0
    host_backup_copy_materialized_count = 0
    skipped_count = 0
    fallback_count = 0
    blocked_count = 0
    error_count = 0

    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError("RelayKV materialization result must be a mapping")
        state = str(_event_value(result, "materialization_state") or "unknown")
        mode = str(_event_value(result, "materialization_mode") or "unknown")
        per_state[state] += 1
        per_mode[mode] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1
        if state == "fake_materialized":
            fake_materialized_count += 1
        elif state == "candidate_event_materialized":
            candidate_event_materialized_count += 1
        elif state == "guarded_noop":
            guarded_noop_count += 1
        elif state == "skipped":
            skipped_count += 1
            guarded_noop_count += 1
        elif state == "blocked":
            blocked_count += 1
            guarded_noop_count += 1
        elif state == "fallback":
            fallback_count += 1
        elif state == "error":
            error_count += 1

        kv_count = _event_value(result, "materialized_kv_count")
        if isinstance(kv_count, int):
            materialized_kv_count += kv_count
        token_count = _event_value(result, "materialized_token_count")
        if isinstance(token_count, int):
            materialized_token_count += token_count

        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "host_backup_copy_executed") is True:
            safety_counts["host_backup_copy_executed_count"] += 1
            host_backup_copy_materialized_count += 1
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1

    return {
        "summary_type": "relaykv_materialization_summary",
        "total_materialization_results": len(results),
        "materialized_result_count": (
            fake_materialized_count + candidate_event_materialized_count
        ),
        "fake_materialized_count": fake_materialized_count,
        "guarded_noop_count": guarded_noop_count,
        "candidate_event_materialized_count": candidate_event_materialized_count,
        "host_backup_copy_materialized_count": host_backup_copy_materialized_count,
        "skipped_count": skipped_count,
        "fallback_count": fallback_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_state_counts": dict(sorted(per_state.items())),
        "per_mode_counts": dict(sorted(per_mode.items())),
        "materialized_kv_count": materialized_kv_count,
        "materialized_token_count": materialized_token_count,
        **dict(safety_counts),
    }


def _host_backup_copy_request_block_ids(
    result: Mapping[str, Any],
    key: str,
) -> list[Any]:
    value = _event_value(result, key)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"RelayKV host backup copy request {key} must be list or tuple")
    return list(value)


def build_relaykv_host_backup_copy_requests_for_smoke(
    candidate_event_materialization_results: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    copy_readiness: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build host backup copy boundary request payloads without copying KV."""

    if not isinstance(candidate_event_materialization_results, (list, tuple)):
        raise TypeError(
            "candidate_event_materialization_results must be a list or tuple"
        )
    if copy_readiness is not None and not isinstance(copy_readiness, Mapping):
        raise TypeError("copy_readiness must be a mapping or None")

    copy_ready = (
        True
        if copy_readiness is None
        else copy_readiness.get("ready_for_host_backup_copy_boundary") is True
    )
    readiness_state = (
        "copy_readiness_not_provided"
        if copy_readiness is None
        else str(copy_readiness.get("readiness_state", "unknown"))
    )
    readiness_blocking_reasons: list[str] = []
    if copy_readiness is not None:
        blocking_reasons = copy_readiness.get("blocking_reasons")
        if isinstance(blocking_reasons, (list, tuple)):
            readiness_blocking_reasons = [str(reason) for reason in blocking_reasons]
    if copy_readiness is not None and not copy_ready and not readiness_blocking_reasons:
        readiness_blocking_reasons = ["copy_readiness_not_met"]

    requests: list[dict[str, Any]] = []
    for result in candidate_event_materialization_results:
        if not isinstance(result, Mapping):
            raise TypeError(
                "RelayKV host backup copy request inputs must be mappings"
            )

        selected_block_ids = _host_backup_copy_request_block_ids(
            result, "selected_block_ids"
        )
        materialized_block_ids = _host_backup_copy_request_block_ids(
            result, "materialized_block_ids"
        )
        retrieved_block_ids = _host_backup_copy_request_block_ids(
            result, "retrieved_block_ids"
        )
        candidate_block_ids = _host_backup_copy_request_block_ids(
            result, "candidate_block_ids"
        )
        anchor_block_ids = _host_backup_copy_request_block_ids(
            result, "anchor_block_ids"
        )
        recent_block_ids = _host_backup_copy_request_block_ids(
            result, "recent_block_ids"
        )
        blocking_reasons: list[str] = []
        warning_reasons: list[str] = []

        if copy_readiness is None:
            warning_reasons.append("copy_readiness_not_provided")
        if not copy_ready:
            blocking_reasons.extend(readiness_blocking_reasons)
        if (
            _event_value(result, "materialization_state")
            != "candidate_event_materialized"
            or _event_value(result, "materialization_mode") != "candidate_event"
        ):
            blocking_reasons.append("not_candidate_event_materialized")
        if not materialized_block_ids:
            blocking_reasons.append("no_materialized_blocks")

        copy_state = "blocked" if blocking_reasons else "request_ready"
        materialized_kv_count = _event_value(result, "materialized_kv_count")
        materialized_token_count = _event_value(result, "materialized_token_count")

        requests.append(
            {
                "event_type": "relaykv_host_backup_copy_request",
                "request_id": _event_value(result, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(result),
                "seq_len": _event_value(result, "seq_len"),
                "layer_id": _event_layer_value(result),
                "selected_block_ids": selected_block_ids,
                "materialized_block_ids": materialized_block_ids,
                "retrieved_block_ids": retrieved_block_ids,
                "candidate_block_ids": candidate_block_ids,
                "anchor_block_ids": anchor_block_ids,
                "recent_block_ids": recent_block_ids,
                "materialized_kv_count": (
                    materialized_kv_count if isinstance(materialized_kv_count, int) else 0
                ),
                "materialized_token_count": (
                    materialized_token_count
                    if isinstance(materialized_token_count, int)
                    else 0
                ),
                "materialization_source": _event_value(result, "source"),
                "readiness_state": readiness_state,
                "copy_state": copy_state,
                "copy_mode": "host_backup_copy_boundary",
                "copy_source": "host_backup_candidate",
                "copy_destination": "materialization_result_only",
                "copy_guard_state": "pre_attention_no_runtime_writeback",
                "copy_reason": "candidate_event_metadata_ready",
                "source": "candidate_event_materialization_to_host_backup_copy_request",
                "blocking_reasons": blocking_reasons,
                "warning_reasons": warning_reasons,
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "host_backup_copy_executed": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
            }
        )
    return requests


def summarize_relaykv_host_backup_copy_requests_for_smoke(
    requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize host backup copy boundary request payloads."""

    if not isinstance(requests, (list, tuple)):
        raise TypeError("RelayKV host backup copy requests must be a list or tuple")

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_copy_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "source_mutated_true_count": 0,
            "attention_override_true_count": 0,
            "kv_cache_mutation_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "host_backup_copy_executed_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
        }
    )
    materialized_kv_count = 0
    materialized_token_count = 0
    request_ready_count = 0
    blocked_count = 0

    for request in requests:
        if not isinstance(request, Mapping):
            raise TypeError("RelayKV host backup copy request must be a mapping")
        copy_state = str(_event_value(request, "copy_state") or "unknown")
        per_copy_state[copy_state] += 1
        per_request[str(_event_value(request, "request_id"))] += 1
        per_layer[str(_event_layer_value(request))] += 1
        if copy_state == "request_ready":
            request_ready_count += 1
        elif copy_state == "blocked":
            blocked_count += 1

        kv_count = _event_value(request, "materialized_kv_count")
        if isinstance(kv_count, int):
            materialized_kv_count += kv_count
        token_count = _event_value(request, "materialized_token_count")
        if isinstance(token_count, int):
            materialized_token_count += token_count

        if _event_value(request, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1
        if _event_value(request, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(request, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(request, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(request, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(request, "host_backup_copy_executed") is True:
            safety_counts["host_backup_copy_executed_count"] += 1
        if _event_value(request, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(request, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1

    return {
        "summary_type": "relaykv_host_backup_copy_request_summary",
        "total_copy_requests": len(requests),
        "request_ready_count": request_ready_count,
        "blocked_count": blocked_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_copy_state_counts": dict(sorted(per_copy_state.items())),
        "materialized_kv_count": materialized_kv_count,
        "materialized_token_count": materialized_token_count,
        **dict(safety_counts),
    }


def build_relaykv_host_backup_copy_boundary_results_for_smoke(
    host_backup_copy_requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    execute_copy: bool = False,
) -> list[dict[str, Any]]:
    """Build host backup copy boundary result payloads without copying KV."""

    if not isinstance(host_backup_copy_requests, (list, tuple)):
        raise TypeError("host_backup_copy_requests must be a list or tuple")
    if not isinstance(execute_copy, bool):
        raise TypeError("execute_copy must be a bool")

    results: list[dict[str, Any]] = []
    for request in host_backup_copy_requests:
        if not isinstance(request, Mapping):
            raise TypeError(
                "RelayKV host backup copy boundary result inputs must be mappings"
            )

        selected_block_ids = _host_backup_copy_request_block_ids(
            request, "selected_block_ids"
        )
        materialized_block_ids = _host_backup_copy_request_block_ids(
            request, "materialized_block_ids"
        )
        retrieved_block_ids = _host_backup_copy_request_block_ids(
            request, "retrieved_block_ids"
        )
        candidate_block_ids = _host_backup_copy_request_block_ids(
            request, "candidate_block_ids"
        )
        anchor_block_ids = _host_backup_copy_request_block_ids(
            request, "anchor_block_ids"
        )
        recent_block_ids = _host_backup_copy_request_block_ids(
            request, "recent_block_ids"
        )

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = []
        if execute_copy:
            blocking_reasons.append("execute_copy_not_allowed_in_smoke")
        else:
            warning_reasons.append("execute_copy_false_boundary_noop")
        if _event_value(request, "event_type") != "relaykv_host_backup_copy_request":
            blocking_reasons.append("not_host_backup_copy_request")
        if _event_value(request, "copy_state") != "request_ready":
            blocking_reasons.append("copy_request_not_ready")
        if not materialized_block_ids:
            blocking_reasons.append("no_materialized_blocks")

        copy_state = "blocked" if blocking_reasons else "boundary_noop"
        materialized_kv_count = _event_value(request, "materialized_kv_count")
        materialized_token_count = _event_value(request, "materialized_token_count")

        results.append(
            {
                "event_type": "relaykv_host_backup_copy_boundary_result",
                "materialization_state": (
                    "blocked"
                    if blocking_reasons
                    else "host_backup_copy_boundary_noop"
                ),
                "materialization_mode": "host_backup_copy_boundary",
                "copy_state": copy_state,
                "copy_mode": _event_value(request, "copy_mode"),
                "request_id": _event_value(request, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(request),
                "seq_len": _event_value(request, "seq_len"),
                "layer_id": _event_layer_value(request),
                "selected_block_ids": selected_block_ids,
                "materialized_block_ids": materialized_block_ids,
                "retrieved_block_ids": retrieved_block_ids,
                "candidate_block_ids": candidate_block_ids,
                "anchor_block_ids": anchor_block_ids,
                "recent_block_ids": recent_block_ids,
                "materialized_kv_count": (
                    materialized_kv_count if isinstance(materialized_kv_count, int) else 0
                ),
                "materialized_token_count": (
                    materialized_token_count
                    if isinstance(materialized_token_count, int)
                    else 0
                ),
                "source": "host_backup_copy_request_to_boundary_result",
                "blocking_reasons": blocking_reasons,
                "warning_reasons": warning_reasons,
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "host_backup_copy_executed": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
            }
        )
    return results


def summarize_relaykv_host_backup_copy_boundary_results_for_smoke(
    results: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize host backup copy boundary result payloads."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV host backup copy boundary results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_copy_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "source_mutated_true_count": 0,
            "attention_override_true_count": 0,
            "kv_cache_mutation_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "host_backup_copy_executed_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
        }
    )
    materialized_kv_count = 0
    materialized_token_count = 0
    boundary_noop_count = 0
    blocked_count = 0
    error_count = 0

    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError("RelayKV host backup copy boundary result must be a mapping")
        copy_state = str(_event_value(result, "copy_state") or "unknown")
        per_copy_state[copy_state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1
        if copy_state == "boundary_noop":
            boundary_noop_count += 1
        elif copy_state == "blocked":
            blocked_count += 1
        elif copy_state == "error":
            error_count += 1

        kv_count = _event_value(result, "materialized_kv_count")
        if isinstance(kv_count, int):
            materialized_kv_count += kv_count
        token_count = _event_value(result, "materialized_token_count")
        if isinstance(token_count, int):
            materialized_token_count += token_count

        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "host_backup_copy_executed") is True:
            safety_counts["host_backup_copy_executed_count"] += 1
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1

    return {
        "summary_type": "relaykv_host_backup_copy_boundary_result_summary",
        "total_boundary_results": len(results),
        "boundary_noop_count": boundary_noop_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_copy_state_counts": dict(sorted(per_copy_state.items())),
        "materialized_kv_count": materialized_kv_count,
        "materialized_token_count": materialized_token_count,
        **dict(safety_counts),
    }


def build_relaykv_actual_host_backup_copy_results_for_smoke(
    host_backup_copy_requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    actual_copy_readiness: Mapping[str, Any] | None = None,
    execute_copy: bool = True,
) -> list[dict[str, Any]]:
    """Build isolated actual host backup copy smoke results without runtime IO."""

    if not isinstance(host_backup_copy_requests, (list, tuple)):
        raise TypeError("host_backup_copy_requests must be a list or tuple")
    if actual_copy_readiness is not None and not isinstance(
        actual_copy_readiness, Mapping
    ):
        raise TypeError("actual_copy_readiness must be a mapping or None")
    if not isinstance(execute_copy, bool):
        raise TypeError("execute_copy must be a bool")

    if actual_copy_readiness is None:
        readiness_ready = False
        readiness_state = "actual_copy_readiness_not_provided"
        readiness_blocking_reasons = ["actual_copy_readiness_not_provided"]
    else:
        readiness_ready = (
            actual_copy_readiness.get("ready_for_actual_host_backup_copy") is True
            or actual_copy_readiness.get("ready_for_actual_host_backup_copy_smoke")
            is True
            or actual_copy_readiness.get("ready_for_host_backup_copy_boundary") is True
            or actual_copy_readiness.get("readiness_state")
            == "ready_for_actual_host_backup_copy_smoke_boundary_complete"
        )
        readiness_state = str(
            actual_copy_readiness.get("readiness_state", "unknown")
        )
        readiness_blocking_reasons = []
        blocking_reasons = actual_copy_readiness.get("blocking_reasons")
        if isinstance(blocking_reasons, (list, tuple)):
            readiness_blocking_reasons = [str(reason) for reason in blocking_reasons]
        if not readiness_ready and not readiness_blocking_reasons:
            readiness_blocking_reasons = ["actual_copy_readiness_not_met"]

    results: list[dict[str, Any]] = []
    for request in host_backup_copy_requests:
        if not isinstance(request, Mapping):
            raise TypeError(
                "RelayKV actual host backup copy smoke inputs must be mappings"
            )

        selected_block_ids = _host_backup_copy_request_block_ids(
            request, "selected_block_ids"
        )
        materialized_block_ids = _host_backup_copy_request_block_ids(
            request, "materialized_block_ids"
        )
        retrieved_block_ids = _host_backup_copy_request_block_ids(
            request, "retrieved_block_ids"
        )
        candidate_block_ids = _host_backup_copy_request_block_ids(
            request, "candidate_block_ids"
        )
        anchor_block_ids = _host_backup_copy_request_block_ids(
            request, "anchor_block_ids"
        )
        recent_block_ids = _host_backup_copy_request_block_ids(
            request, "recent_block_ids"
        )

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = ["isolated_smoke_no_runtime_connection"]
        host_backup_copy_executed = False

        if actual_copy_readiness is None:
            blocking_reasons.append("actual_copy_readiness_not_provided")
        elif not readiness_ready:
            blocking_reasons.extend(readiness_blocking_reasons)
        if not execute_copy:
            blocking_reasons.append("execute_copy_required_for_actual_copy_smoke")
        if _event_value(request, "event_type") != "relaykv_host_backup_copy_request":
            blocking_reasons.append("not_host_backup_copy_request")
        if _event_value(request, "copy_state") != "request_ready":
            blocking_reasons.append("copy_request_not_ready")
        if not materialized_block_ids:
            blocking_reasons.append("no_materialized_blocks")

        copy_state = "blocked" if blocking_reasons else "copy_executed"
        if not blocking_reasons:
            host_backup_copy_executed = True

        materialized_kv_count = _event_value(request, "materialized_kv_count")
        materialized_token_count = _event_value(request, "materialized_token_count")
        if not isinstance(materialized_kv_count, int):
            materialized_kv_count = len(materialized_block_ids)
        if not isinstance(materialized_token_count, int):
            materialized_token_count = 0

        results.append(
            {
                "event_type": "relaykv_materialization_result",
                "materialization_state": (
                    "host_backup_copy_materialized"
                    if host_backup_copy_executed
                    else "blocked"
                ),
                "materialization_mode": "host_backup_copy",
                "copy_state": copy_state,
                "copy_mode": "host_backup_copy_isolated_smoke",
                "request_id": _event_value(request, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(request),
                "seq_len": _event_value(request, "seq_len"),
                "layer_id": _event_layer_value(request),
                "selected_block_ids": selected_block_ids,
                "materialized_block_ids": materialized_block_ids,
                "retrieved_block_ids": retrieved_block_ids,
                "candidate_block_ids": candidate_block_ids,
                "anchor_block_ids": anchor_block_ids,
                "recent_block_ids": recent_block_ids,
                "materialized_kv_count": materialized_kv_count,
                "materialized_token_count": materialized_token_count,
                "source": "host_backup_copy_request_to_isolated_materialization_result",
                "readiness_state": readiness_state,
                "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
                "warning_reasons": warning_reasons,
                "source_mutated": False,
                "attention_override": False,
                "kv_cache_mutation": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "host_backup_copy_executed": host_backup_copy_executed,
                "kv_pool_read": False,
                "kv_snapshot": False,
            }
        )
    return results


def summarize_relaykv_actual_host_backup_copy_results_for_smoke(
    results: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize isolated actual host backup copy smoke results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV actual host backup copy smoke results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_copy_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "source_mutated_true_count": 0,
            "attention_override_true_count": 0,
            "kv_cache_mutation_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "host_backup_copy_executed_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
        }
    )
    materialized_kv_count = 0
    materialized_token_count = 0
    host_backup_copy_materialized_count = 0
    blocked_count = 0
    error_count = 0

    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError(
                "RelayKV actual host backup copy smoke result must be a mapping"
            )
        copy_state = str(_event_value(result, "copy_state") or "unknown")
        per_copy_state[copy_state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1
        if copy_state == "copy_executed":
            host_backup_copy_materialized_count += 1
        elif copy_state == "blocked":
            blocked_count += 1
        elif copy_state == "error":
            error_count += 1

        kv_count = _event_value(result, "materialized_kv_count")
        if isinstance(kv_count, int):
            materialized_kv_count += kv_count
        token_count = _event_value(result, "materialized_token_count")
        if isinstance(token_count, int):
            materialized_token_count += token_count

        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "host_backup_copy_executed") is True:
            safety_counts["host_backup_copy_executed_count"] += 1
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1

    return {
        "summary_type": "relaykv_actual_host_backup_copy_result_summary",
        "total_copy_results": len(results),
        "host_backup_copy_materialized_count": host_backup_copy_materialized_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "materialized_kv_count": materialized_kv_count,
        "materialized_token_count": materialized_token_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_copy_state_counts": dict(sorted(per_copy_state.items())),
        **dict(safety_counts),
    }


def log_policy_summary(
    events: Iterable[RelayKVPlan | Mapping[str, Any]],
    *,
    prefix: str = "relaykv_policy_summary",
) -> None:
    payload = summarize_policy_events(events)
    logger.info("%s=%s", prefix, json.dumps(payload, sort_keys=True))


def policy_event_payload(
    event: RelayKVPlan | Mapping[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {key: _event_value(event, key) for key in POLICY_EVENT_LOG_KEYS}
    state = payload["runtime_policy_state"]
    dry_copy_candidate = state == "applied_candidate"
    payload["runtime_policy_action"] = (
        "dry_copy_candidate_log_only" if dry_copy_candidate else "log_only_noop"
    )
    payload["dry_copy_candidate"] = dry_copy_candidate
    payload["applied_candidate_log_only"] = state == "applied_candidate"
    payload["fallback_candidate_noop_guard"] = state == "fallback_candidate"
    payload["scheduler_policy_noop"] = True
    payload["kv_cache_mutation"] = False
    payload["attention_override"] = False
    payload["runtime_writeback"] = False
    payload["host_backup_copy"] = False
    payload["host_backup_copy_executed"] = False
    if dry_copy_candidate:
        payload["host_backup_copy_skipped_reason"] = (
            "dry_copy_candidate_metadata_only_no_host_backup_copy"
        )
    elif state == "fallback_candidate":
        payload["host_backup_copy_skipped_reason"] = "fallback_candidate_noop_guard"
    else:
        payload["host_backup_copy_skipped_reason"] = (
            "runtime_policy_state_not_applied_candidate"
        )
    if extra:
        payload.update(extra)
    return payload


def log_policy_event(
    event: RelayKVPlan | Mapping[str, Any],
    *,
    prefix: str = "relaykv_runtime_policy_event",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = policy_event_payload(event, extra=extra)
    logger.info("%s=%s", prefix, json.dumps(payload, sort_keys=True))
