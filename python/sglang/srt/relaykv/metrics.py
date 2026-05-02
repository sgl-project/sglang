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
