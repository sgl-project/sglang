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
