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
    values = [source.get(key) for source in sources if isinstance(source.get(key), int)]
    if not values:
        missing_field_counts["missing_field_count"] += 1
        return 0
    return max(values)


def build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
    runtime_observation_summary: Mapping[str, Any],
    host_backup_candidate_summary: Mapping[str, Any],
    join_summary: Mapping[str, Any],
    policy_dry_run_summary: Mapping[str, Any] | None = None,
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
        )
        for key in _REPORT_SAFETY_COUNTER_KEYS
    }
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
    report_generated_from_readonly_inputs = True
    overall_safety_status = (
        "pass"
        if report_generated_from_readonly_inputs
        and all(value == 0 for value in safety_counts.values())
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
        "overall_safety_status": overall_safety_status,
        "total_runtime_payloads": total_runtime_payloads,
        "total_host_backup_candidate_events": total_host_backup_candidate_events,
        "joined_count": joined_count,
        "unmatched_runtime_count": unmatched_runtime_count,
        "unmatched_candidate_count": unmatched_candidate_count,
        "join_granularity": str(join_summary.get("join_granularity", "unknown")),
        "req_pool_idx_joined_count": req_pool_idx_joined_count,
        "req_pool_idx_missing_count": req_pool_idx_missing_count,
        **safety_counts,
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
            candidate_event_materialized_count += 1
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
        "materialized_result_count": candidate_event_materialized_count,
        "fake_materialized_count": candidate_event_materialized_count,
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
