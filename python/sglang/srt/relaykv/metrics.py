from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class RelayKVBlockMeta:
    """Reserved RelayKV block metadata for future retrieval-only annotations."""

    query_block_score: Optional[float] = None
    middle_layer_query_block_score: Optional[float] = None
    retrieval_criticality_rank: Optional[int] = None
    gather_anchor_score: Optional[float] = None
    aggregate_retrieval_score: Optional[float] = None
    massive_qk_score: Optional[float] = None
    working_set_stability_score: Optional[float] = None
    last_retrieved_step: Optional[int] = None
    retrieval_reuse_count: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayKVGroupMeta:
    """Reserved RelayKV per-kv-head-group metadata.

    For GQA-style models, any future per-head retrieval scores should be
    aggregated into `kv_head_group` scores before populating this schema.
    """

    layer_id: int
    kv_head_group: int
    retrieval_head_score: Optional[float] = None
    query_dependent_group_score: Optional[float] = None
    group_budget_bonus: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayKVPolicyDecision:
    """Reserved RelayKV policy metadata for future temporal-reuse bookkeeping."""

    temporal_reuse_enabled: bool = False
    reused_block_ids: list[int] = field(default_factory=list)
    newly_retrieved_block_ids: list[int] = field(default_factory=list)
    selection_stability_ratio: Optional[float] = None
    selection_reason_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def build_relaykv_actual_host_backup_copy_report_for_smoke(
    readonly_report: Mapping[str, Any],
    actual_host_backup_copy_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build an isolated actual host backup copy report for smoke tests only.

    This combines a readonly diagnostic report with an isolated actual-copy
    smoke summary. It does not execute copy, read KV pools, snapshot KV,
    connect attention, alter scheduler behavior, or write runtime state.
    """

    if not isinstance(readonly_report, Mapping):
        raise TypeError("readonly_report must be a mapping")
    if actual_host_backup_copy_summary is not None and not isinstance(
        actual_host_backup_copy_summary, Mapping
    ):
        raise TypeError("actual_host_backup_copy_summary must be a mapping or None")

    actual_summary_included = actual_host_backup_copy_summary is not None
    actual_summary = actual_host_backup_copy_summary or {}

    actual_safety_counts = {
        key: actual_summary.get(key) if isinstance(actual_summary.get(key), int) else 0
        for key in (*_REPORT_SAFETY_COUNTER_KEYS, *_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS)
    }
    readonly_safety_counts = {
        key: readonly_report.get(key) if isinstance(readonly_report.get(key), int) else 0
        for key in (*_REPORT_SAFETY_COUNTER_KEYS, *_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS)
    }

    actual_host_backup_copy_total = (
        actual_summary.get("total_copy_results")
        if isinstance(actual_summary.get("total_copy_results"), int)
        else 0
    )
    actual_host_backup_copy_materialized_count = (
        actual_summary.get("host_backup_copy_materialized_count")
        if isinstance(actual_summary.get("host_backup_copy_materialized_count"), int)
        else 0
    )
    actual_host_backup_copy_blocked_count = (
        actual_summary.get("blocked_count")
        if isinstance(actual_summary.get("blocked_count"), int)
        else 0
    )
    actual_host_backup_copy_error_count = (
        actual_summary.get("error_count")
        if isinstance(actual_summary.get("error_count"), int)
        else 0
    )
    actual_host_backup_copy_materialized_kv_count = (
        actual_summary.get("materialized_kv_count")
        if isinstance(actual_summary.get("materialized_kv_count"), int)
        else 0
    )
    actual_host_backup_copy_materialized_token_count = (
        actual_summary.get("materialized_token_count")
        if isinstance(actual_summary.get("materialized_token_count"), int)
        else 0
    )
    actual_host_backup_copy_executed_count = actual_safety_counts[
        "host_backup_copy_executed_count"
    ]
    actual_host_backup_copy_kv_pool_read_count = actual_safety_counts[
        "kv_pool_read_count"
    ]
    actual_host_backup_copy_kv_snapshot_count = actual_safety_counts[
        "kv_snapshot_count"
    ]

    actual_copy_safety_reasons: list[str] = []
    if readonly_report.get("overall_safety_status") != "pass":
        actual_copy_safety_reasons.append("readonly_report_overall_safety_not_pass")
    if not actual_summary_included:
        actual_copy_safety_reasons.append("actual_host_backup_copy_summary_missing")
    if actual_host_backup_copy_kv_pool_read_count != 0:
        actual_copy_safety_reasons.append("kv_pool_read_observed")
    if actual_host_backup_copy_kv_snapshot_count != 0:
        actual_copy_safety_reasons.append("kv_snapshot_observed")
    for key in _REPORT_SAFETY_COUNTER_KEYS:
        if actual_safety_counts[key] != 0:
            actual_copy_safety_reasons.append(key)
    if actual_summary_included and not isinstance(
        actual_summary.get("summary_type"), str
    ):
        actual_copy_safety_reasons.append("actual_host_backup_copy_summary_malformed")

    actual_copy_safety_reasons = list(dict.fromkeys(actual_copy_safety_reasons))
    actual_copy_safety_status = (
        "pass" if not actual_copy_safety_reasons else "fail"
    )

    return {
        "report_type": "relaykv_actual_host_backup_copy_report",
        "source_report_type": readonly_report.get("report_type"),
        "report_generated_from_readonly_inputs": (
            readonly_report.get("report_generated_from_readonly_inputs") is True
        ),
        "overall_safety_status": readonly_report.get("overall_safety_status"),
        "actual_copy_report_generated_from_isolated_smoke_inputs": True,
        "actual_copy_safety_status": actual_copy_safety_status,
        "actual_copy_safety_reasons": actual_copy_safety_reasons,
        "readonly_report": dict(readonly_report),
        "actual_host_backup_copy_summary": (
            dict(actual_host_backup_copy_summary)
            if actual_host_backup_copy_summary is not None
            else None
        ),
        "policy_dry_run_included": readonly_report.get("policy_dry_run_included")
        is True,
        "policy_dry_run_total_events": (
            readonly_report.get("policy_dry_run_total_events")
            if isinstance(readonly_report.get("policy_dry_run_total_events"), int)
            else 0
        ),
        "materialization_summary_included": (
            readonly_report.get("materialization_summary_included") is True
        ),
        "materialization_result_count": (
            readonly_report.get("materialization_result_count")
            if isinstance(readonly_report.get("materialization_result_count"), int)
            else 0
        ),
        "host_backup_copy_request_summary_included": (
            readonly_report.get("host_backup_copy_request_summary_included") is True
        ),
        "host_backup_copy_request_ready_count": (
            readonly_report.get("host_backup_copy_request_ready_count")
            if isinstance(readonly_report.get("host_backup_copy_request_ready_count"), int)
            else 0
        ),
        "host_backup_copy_boundary_result_summary_included": (
            readonly_report.get("host_backup_copy_boundary_result_summary_included")
            is True
        ),
        "host_backup_copy_boundary_noop_count": (
            readonly_report.get("host_backup_copy_boundary_noop_count")
            if isinstance(readonly_report.get("host_backup_copy_boundary_noop_count"), int)
            else 0
        ),
        "actual_host_backup_copy_summary_included": actual_summary_included,
        "actual_host_backup_copy_total": actual_host_backup_copy_total,
        "actual_host_backup_copy_materialized_count": (
            actual_host_backup_copy_materialized_count
        ),
        "actual_host_backup_copy_blocked_count": actual_host_backup_copy_blocked_count,
        "actual_host_backup_copy_error_count": actual_host_backup_copy_error_count,
        "actual_host_backup_copy_materialized_kv_count": (
            actual_host_backup_copy_materialized_kv_count
        ),
        "actual_host_backup_copy_materialized_token_count": (
            actual_host_backup_copy_materialized_token_count
        ),
        "actual_host_backup_copy_executed_count": (
            actual_host_backup_copy_executed_count
        ),
        "actual_host_backup_copy_kv_pool_read_count": (
            actual_host_backup_copy_kv_pool_read_count
        ),
        "actual_host_backup_copy_kv_snapshot_count": (
            actual_host_backup_copy_kv_snapshot_count
        ),
        **readonly_safety_counts,
        **actual_safety_counts,
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


def assess_relaykv_attention_connection_readiness_for_smoke(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess isolated actual-copy report readiness for attention design only."""

    if not isinstance(report, Mapping):
        raise TypeError(
            "RelayKV attention connection readiness report must be a mapping"
        )

    safety_counts = {
        key: report.get(key) if isinstance(report.get(key), int) else 0
        for key in (*_REPORT_SAFETY_COUNTER_KEYS, *_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS)
    }

    observed_report_type = str(report.get("report_type", "unknown"))
    observed_actual_copy_safety_status = str(
        report.get("actual_copy_safety_status", "unknown")
    )
    observed_actual_host_backup_copy_summary_included = (
        report.get("actual_host_backup_copy_summary_included") is True
    )
    observed_actual_host_backup_copy_materialized_count = (
        report.get("actual_host_backup_copy_materialized_count")
        if isinstance(report.get("actual_host_backup_copy_materialized_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_executed_count = (
        report.get("actual_host_backup_copy_executed_count")
        if isinstance(report.get("actual_host_backup_copy_executed_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_blocked_count = (
        report.get("actual_host_backup_copy_blocked_count")
        if isinstance(report.get("actual_host_backup_copy_blocked_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_error_count = (
        report.get("actual_host_backup_copy_error_count")
        if isinstance(report.get("actual_host_backup_copy_error_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_kv_pool_read_count = (
        report.get("actual_host_backup_copy_kv_pool_read_count")
        if isinstance(report.get("actual_host_backup_copy_kv_pool_read_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_kv_snapshot_count = (
        report.get("actual_host_backup_copy_kv_snapshot_count")
        if isinstance(report.get("actual_host_backup_copy_kv_snapshot_count"), int)
        else 0
    )

    blocker_state_by_reason = {
        "not_actual_copy_report": "blocked_not_actual_copy_report",
        "actual_copy_safety_not_pass": "blocked_actual_copy_safety_not_pass",
        "actual_copy_summary_missing": "blocked_actual_copy_summary_missing",
        "no_actual_copy_materialized": "blocked_no_actual_copy_materialized",
        "actual_copy_not_executed": "blocked_actual_copy_not_executed",
        "actual_copy_blocked": "blocked_actual_copy_blocked",
        "actual_copy_error": "blocked_actual_copy_error",
        "kv_pool_read_observed": "blocked_kv_pool_read_observed",
        "kv_snapshot_observed": "blocked_kv_snapshot_observed",
        "safety_counter_nonzero": "blocked_safety_counter_nonzero",
    }
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = [
        "design_only_readiness_does_not_connect_attention"
    ]

    if observed_report_type != "relaykv_actual_host_backup_copy_report":
        blocking_reasons.append("not_actual_copy_report")
    if observed_actual_copy_safety_status != "pass":
        blocking_reasons.append("actual_copy_safety_not_pass")
    if not observed_actual_host_backup_copy_summary_included:
        blocking_reasons.append("actual_copy_summary_missing")
    if observed_actual_host_backup_copy_materialized_count <= 0:
        blocking_reasons.append("no_actual_copy_materialized")
    if observed_actual_host_backup_copy_executed_count <= 0:
        blocking_reasons.append("actual_copy_not_executed")
    if observed_actual_host_backup_copy_blocked_count > 0:
        blocking_reasons.append("actual_copy_blocked")
    if observed_actual_host_backup_copy_error_count > 0:
        blocking_reasons.append("actual_copy_error")
    if observed_actual_host_backup_copy_kv_pool_read_count > 0:
        blocking_reasons.append("kv_pool_read_observed")
    if observed_actual_host_backup_copy_kv_snapshot_count > 0:
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
    ready_for_attention_connection = not blocking_reasons
    if ready_for_attention_connection:
        readiness_state = "ready_for_attention_connection_design_only"
        readiness_reasons = ["isolated_actual_copy_report_ready_for_design_only"]
    elif len(blocking_reasons) == 1:
        readiness_state = blocker_state_by_reason[blocking_reasons[0]]
        readiness_reasons = []
    else:
        readiness_state = "blocked_multiple_reasons"
        readiness_reasons = []

    return {
        "readiness_type": "relaykv_attention_connection_readiness",
        "ready_for_attention_connection": ready_for_attention_connection,
        "readiness_state": readiness_state,
        "readiness_reasons": readiness_reasons,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "observed_actual_copy_safety_status": observed_actual_copy_safety_status,
        "observed_actual_host_backup_copy_summary_included": (
            observed_actual_host_backup_copy_summary_included
        ),
        "observed_actual_host_backup_copy_materialized_count": (
            observed_actual_host_backup_copy_materialized_count
        ),
        "observed_actual_host_backup_copy_executed_count": (
            observed_actual_host_backup_copy_executed_count
        ),
        "observed_actual_host_backup_copy_blocked_count": (
            observed_actual_host_backup_copy_blocked_count
        ),
        "observed_actual_host_backup_copy_error_count": (
            observed_actual_host_backup_copy_error_count
        ),
        "observed_actual_host_backup_copy_kv_pool_read_count": (
            observed_actual_host_backup_copy_kv_pool_read_count
        ),
        "observed_actual_host_backup_copy_kv_snapshot_count": (
            observed_actual_host_backup_copy_kv_snapshot_count
        ),
        **safety_counts,
    }


def assess_relaykv_attention_handoff_readiness_for_smoke(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess metadata-only readiness for RelayKV attention handoff smoke."""

    if not isinstance(report, Mapping):
        raise TypeError("RelayKV attention handoff readiness report must be a mapping")

    safety_counts = {
        key: report.get(key) if isinstance(report.get(key), int) else 0
        for key in (*_REPORT_SAFETY_COUNTER_KEYS, *_MATERIALIZATION_REPORT_SAFETY_COUNTER_KEYS)
    }

    observed_report_type = str(report.get("report_type", "unknown"))
    observed_actual_copy_safety_status = str(
        report.get("actual_copy_safety_status", "unknown")
    )
    observed_actual_host_backup_copy_summary_included = (
        report.get("actual_host_backup_copy_summary_included") is True
    )
    observed_actual_host_backup_copy_materialized_count = (
        report.get("actual_host_backup_copy_materialized_count")
        if isinstance(report.get("actual_host_backup_copy_materialized_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_executed_count = (
        report.get("actual_host_backup_copy_executed_count")
        if isinstance(report.get("actual_host_backup_copy_executed_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_kv_pool_read_count = (
        report.get("actual_host_backup_copy_kv_pool_read_count")
        if isinstance(report.get("actual_host_backup_copy_kv_pool_read_count"), int)
        else 0
    )
    observed_actual_host_backup_copy_kv_snapshot_count = (
        report.get("actual_host_backup_copy_kv_snapshot_count")
        if isinstance(report.get("actual_host_backup_copy_kv_snapshot_count"), int)
        else 0
    )

    blocker_state_by_reason = {
        "not_actual_copy_report": "blocked_not_actual_copy_report",
        "actual_copy_safety_not_pass": "blocked_actual_copy_safety_not_pass",
        "actual_copy_summary_missing": "blocked_actual_copy_summary_missing",
        "no_actual_copy_materialized": "blocked_no_actual_copy_materialized",
        "actual_copy_not_executed": "blocked_actual_copy_not_executed",
        "kv_pool_read_observed": "blocked_kv_pool_read_observed",
        "kv_snapshot_observed": "blocked_kv_snapshot_observed",
        "attention_override_observed": "blocked_attention_override_observed",
        "runtime_writeback_observed": "blocked_runtime_writeback_observed",
        "scheduler_mutation_observed": "blocked_scheduler_mutation_observed",
    }
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = [
        "metadata_only_handoff_does_not_connect_attention_backend"
    ]

    if observed_report_type != "relaykv_actual_host_backup_copy_report":
        blocking_reasons.append("not_actual_copy_report")
    if observed_actual_copy_safety_status != "pass":
        blocking_reasons.append("actual_copy_safety_not_pass")
    if not observed_actual_host_backup_copy_summary_included:
        blocking_reasons.append("actual_copy_summary_missing")
    if observed_actual_host_backup_copy_materialized_count <= 0:
        blocking_reasons.append("no_actual_copy_materialized")
    if observed_actual_host_backup_copy_executed_count <= 0:
        blocking_reasons.append("actual_copy_not_executed")
    if observed_actual_host_backup_copy_kv_pool_read_count > 0:
        blocking_reasons.append("kv_pool_read_observed")
    if observed_actual_host_backup_copy_kv_snapshot_count > 0:
        blocking_reasons.append("kv_snapshot_observed")
    if safety_counts["attention_override_true_count"] > 0:
        blocking_reasons.append("attention_override_observed")
    if safety_counts["runtime_writeback_true_count"] > 0:
        blocking_reasons.append("runtime_writeback_observed")
    if safety_counts["scheduler_policy_noop_false_count"] > 0:
        blocking_reasons.append("scheduler_mutation_observed")

    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    ready_for_attention_handoff = not blocking_reasons
    if ready_for_attention_handoff:
        readiness_state = "ready_for_attention_handoff_metadata_only"
        readiness_reasons = ["actual_copy_report_ready_for_metadata_only_handoff"]
    elif len(blocking_reasons) == 1:
        readiness_state = blocker_state_by_reason[blocking_reasons[0]]
        readiness_reasons = []
    else:
        readiness_state = "blocked_multiple_reasons"
        readiness_reasons = []

    return {
        "readiness_type": "relaykv_attention_handoff_readiness",
        "ready_for_attention_handoff": ready_for_attention_handoff,
        "readiness_state": readiness_state,
        "readiness_reasons": readiness_reasons,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "observed_actual_copy_safety_status": observed_actual_copy_safety_status,
        "observed_actual_host_backup_copy_summary_included": (
            observed_actual_host_backup_copy_summary_included
        ),
        "observed_actual_host_backup_copy_materialized_count": (
            observed_actual_host_backup_copy_materialized_count
        ),
        "observed_actual_host_backup_copy_executed_count": (
            observed_actual_host_backup_copy_executed_count
        ),
        "observed_actual_host_backup_copy_kv_pool_read_count": (
            observed_actual_host_backup_copy_kv_pool_read_count
        ),
        "observed_actual_host_backup_copy_kv_snapshot_count": (
            observed_actual_host_backup_copy_kv_snapshot_count
        ),
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


def build_relaykv_attention_handoff_candidates_for_smoke(
    actual_copy_results: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    attention_handoff_readiness: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build metadata-only attention handoff candidates from actual-copy results."""

    if not isinstance(actual_copy_results, (list, tuple)):
        raise TypeError("actual_copy_results must be a list or tuple")
    if attention_handoff_readiness is not None and not isinstance(
        attention_handoff_readiness, Mapping
    ):
        raise TypeError("attention_handoff_readiness must be a mapping or None")

    if attention_handoff_readiness is None:
        readiness_ready = False
        readiness_state = "attention_handoff_readiness_not_provided"
        readiness_blocking_reasons = ["attention_handoff_readiness_not_provided"]
    else:
        readiness_ready = (
            attention_handoff_readiness.get("ready_for_attention_handoff") is True
        )
        readiness_state = str(
            attention_handoff_readiness.get("readiness_state", "unknown")
        )
        readiness_blocking_reasons = []
        blocking_reasons = attention_handoff_readiness.get("blocking_reasons")
        if isinstance(blocking_reasons, (list, tuple)):
            readiness_blocking_reasons = [str(reason) for reason in blocking_reasons]
        if not readiness_ready and not readiness_blocking_reasons:
            readiness_blocking_reasons = ["attention_handoff_readiness_not_met"]

    candidates: list[dict[str, Any]] = []
    for result in actual_copy_results:
        if not isinstance(result, Mapping):
            raise TypeError("RelayKV attention handoff inputs must be mappings")

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
        if attention_handoff_readiness is None:
            blocking_reasons.append("attention_handoff_readiness_not_provided")
        elif not readiness_ready:
            blocking_reasons.extend(readiness_blocking_reasons)
        if _event_value(result, "event_type") != "relaykv_materialization_result":
            blocking_reasons.append("not_materialization_result")
        if _event_value(result, "materialization_state") != "host_backup_copy_materialized":
            blocking_reasons.append("not_host_backup_copy_materialized")
        if _event_value(result, "copy_state") != "copy_executed":
            blocking_reasons.append("copy_not_executed")
        if not materialized_block_ids:
            blocking_reasons.append("no_materialized_blocks")

        handoff_state = "blocked" if blocking_reasons else "handoff_ready"
        materialized_token_count = _event_value(result, "materialized_token_count")
        working_kv_token_count = (
            materialized_token_count if isinstance(materialized_token_count, int) else 0
        )
        candidates.append(
            {
                "event_type": "relaykv_attention_handoff_candidate",
                "handoff_state": handoff_state,
                "handoff_mode": "metadata_only",
                "source": "actual_host_backup_copy_result_to_attention_handoff_candidate",
                "request_id": _event_value(result, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(result),
                "seq_len": _event_value(result, "seq_len"),
                "layer_id": _event_layer_value(result),
                "recent_block_ids": recent_block_ids,
                "anchor_block_ids": anchor_block_ids,
                "retrieved_block_ids": retrieved_block_ids,
                "candidate_block_ids": candidate_block_ids,
                "materialized_block_ids": materialized_block_ids,
                "working_kv_block_ids": materialized_block_ids,
                "working_kv_block_count": len(materialized_block_ids),
                "working_kv_token_count": working_kv_token_count,
                "attention_target_layer_id": _event_layer_value(result),
                "attention_target_backend": "unconnected",
                "attention_override_allowed": False,
                "attention_connection_attempted": False,
                "attention_override": False,
                "attention_override_noop": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
                "warning_reasons": [],
                "readiness_state": readiness_state,
            }
        )
    return candidates


def summarize_relaykv_attention_handoff_candidates_for_smoke(
    candidates: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize metadata-only RelayKV attention handoff candidates."""

    if not isinstance(candidates, (list, tuple)):
        raise TypeError("RelayKV attention handoff candidates must be a list or tuple")

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_handoff_state: Counter[str] = Counter()
    attention_connection_attempted_count = 0
    attention_override_noop_count = 0
    working_kv_block_count = 0
    working_kv_token_count = 0
    handoff_ready_count = 0
    blocked_count = 0
    safety_counts: Counter[str] = Counter(
        {
            "attention_override_true_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("RelayKV attention handoff candidate must be a mapping")
        handoff_state = str(_event_value(candidate, "handoff_state") or "unknown")
        per_handoff_state[handoff_state] += 1
        per_request[str(_event_value(candidate, "request_id"))] += 1
        per_layer[str(_event_layer_value(candidate))] += 1
        if handoff_state == "handoff_ready":
            handoff_ready_count += 1
        elif handoff_state == "blocked":
            blocked_count += 1

        block_count = _event_value(candidate, "working_kv_block_count")
        if isinstance(block_count, int):
            working_kv_block_count += block_count
        token_count = _event_value(candidate, "working_kv_token_count")
        if isinstance(token_count, int):
            working_kv_token_count += token_count
        if _event_value(candidate, "attention_connection_attempted") is True:
            attention_connection_attempted_count += 1
        if _event_value(candidate, "attention_override_noop") is True:
            attention_override_noop_count += 1
        if _event_value(candidate, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(candidate, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(candidate, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(candidate, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(candidate, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(candidate, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(candidate, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_attention_handoff_candidate_summary",
        "total_handoff_candidates": len(candidates),
        "handoff_ready_count": handoff_ready_count,
        "blocked_count": blocked_count,
        "working_kv_block_count": working_kv_block_count,
        "working_kv_token_count": working_kv_token_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_handoff_state_counts": dict(sorted(per_handoff_state.items())),
        "attention_connection_attempted_count": attention_connection_attempted_count,
        "attention_override_noop_count": attention_override_noop_count,
        **dict(safety_counts),
    }


def build_relaykv_attention_connection_dry_run_results_for_smoke(
    handoff_candidates: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    execute_attention: bool = False,
) -> list[dict[str, Any]]:
    """Build metadata-only attention connection dry-run results."""

    if not isinstance(handoff_candidates, (list, tuple)):
        raise TypeError("handoff_candidates must be a list or tuple")
    if not isinstance(execute_attention, bool):
        raise TypeError("execute_attention must be a bool")

    results: list[dict[str, Any]] = []
    for candidate in handoff_candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError(
                "RelayKV attention connection dry-run inputs must be mappings"
            )

        working_kv_block_ids = _host_backup_copy_request_block_ids(
            candidate, "working_kv_block_ids"
        )
        working_kv_block_count = _event_value(candidate, "working_kv_block_count")
        if not isinstance(working_kv_block_count, int):
            working_kv_block_count = len(working_kv_block_ids)
        working_kv_token_count = _event_value(candidate, "working_kv_token_count")
        if not isinstance(working_kv_token_count, int):
            working_kv_token_count = 0

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = ["metadata_only_attention_connection_dry_run"]
        attention_connection_attempted = False

        if execute_attention:
            blocking_reasons.append("execute_attention_not_allowed_in_dry_run_smoke")
        if _event_value(candidate, "event_type") != "relaykv_attention_handoff_candidate":
            blocking_reasons.append("not_attention_handoff_candidate")
        if _event_value(candidate, "handoff_state") != "handoff_ready":
            blocking_reasons.append("handoff_not_ready")
        if _event_value(candidate, "handoff_mode") != "metadata_only":
            blocking_reasons.append("handoff_not_metadata_only")
        if not working_kv_block_ids:
            blocking_reasons.append("no_working_kv_blocks")

        if not blocking_reasons:
            attention_connection_attempted = True

        results.append(
            {
                "event_type": "relaykv_attention_connection_dry_run_result",
                "attention_connection_state": (
                    "dry_run" if attention_connection_attempted else "blocked"
                ),
                "attention_connection_mode": "metadata_only",
                "source": "attention_handoff_candidate_to_connection_dry_run_result",
                "request_id": _event_value(candidate, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(candidate),
                "seq_len": _event_value(candidate, "seq_len"),
                "layer_id": _event_layer_value(candidate),
                "working_kv_block_ids": working_kv_block_ids,
                "working_kv_block_count": working_kv_block_count,
                "working_kv_token_count": working_kv_token_count,
                "attention_target_layer_id": (
                    _event_value(candidate, "attention_target_layer_id")
                    if _event_value(candidate, "attention_target_layer_id") is not None
                    else _event_layer_value(candidate)
                ),
                "attention_target_backend": "unconnected",
                "attention_connection_attempted": attention_connection_attempted,
                "attention_override": False,
                "attention_override_noop": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
                "warning_reasons": warning_reasons,
            }
        )
    return results


def summarize_relaykv_attention_connection_dry_run_results_for_smoke(
    results: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize metadata-only attention connection dry-run results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV attention connection dry-run results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_attention_connection_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "attention_override_true_count": 0,
            "attention_override_noop_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    attention_connection_attempted_count = 0
    attention_connection_dry_run_count = 0
    blocked_count = 0
    error_count = 0
    working_kv_block_count = 0
    working_kv_token_count = 0

    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError(
                "RelayKV attention connection dry-run result must be a mapping"
            )
        state = str(_event_value(result, "attention_connection_state") or "unknown")
        per_attention_connection_state[state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1
        if state == "dry_run":
            attention_connection_dry_run_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        block_count = _event_value(result, "working_kv_block_count")
        if isinstance(block_count, int):
            working_kv_block_count += block_count
        token_count = _event_value(result, "working_kv_token_count")
        if isinstance(token_count, int):
            working_kv_token_count += token_count
        if _event_value(result, "attention_connection_attempted") is True:
            attention_connection_attempted_count += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "attention_override_noop") is True:
            safety_counts["attention_override_noop_count"] += 1
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_attention_connection_dry_run_result_summary",
        "total_attention_connection_dry_run_results": len(results),
        "attention_connection_dry_run_count": attention_connection_dry_run_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "working_kv_block_count": working_kv_block_count,
        "working_kv_token_count": working_kv_token_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_attention_connection_state_counts": dict(
            sorted(per_attention_connection_state.items())
        ),
        "attention_connection_attempted_count": attention_connection_attempted_count,
        **dict(safety_counts),
    }


def build_relaykv_attention_override_noop_results_for_smoke(
    attention_connection_dry_run_results: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    allow_override: bool = False,
) -> list[dict[str, Any]]:
    """Build guarded no-op attention override results from dry-run results."""

    if not isinstance(attention_connection_dry_run_results, (list, tuple)):
        raise TypeError(
            "attention_connection_dry_run_results must be a list or tuple"
        )
    if not isinstance(allow_override, bool):
        raise TypeError("allow_override must be a bool")

    results: list[dict[str, Any]] = []
    for dry_run_result in attention_connection_dry_run_results:
        if not isinstance(dry_run_result, Mapping):
            raise TypeError(
                "RelayKV attention override noop inputs must be mappings"
            )

        working_kv_block_ids = _host_backup_copy_request_block_ids(
            dry_run_result, "working_kv_block_ids"
        )
        working_kv_block_count = _event_value(
            dry_run_result, "working_kv_block_count"
        )
        if not isinstance(working_kv_block_count, int):
            working_kv_block_count = len(working_kv_block_ids)
        working_kv_token_count = _event_value(
            dry_run_result, "working_kv_token_count"
        )
        if not isinstance(working_kv_token_count, int):
            working_kv_token_count = 0

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = [
            "attention_override_noop_guarded",
            "no_runtime_attention_backend_connection",
        ]
        attention_connection_attempted = False
        attention_override_noop = False

        if allow_override:
            blocking_reasons.append("attention_override_not_allowed_in_phase4_noop")
        if (
            _event_value(dry_run_result, "event_type")
            != "relaykv_attention_connection_dry_run_result"
        ):
            blocking_reasons.append("not_attention_connection_dry_run_result")
        if _event_value(dry_run_result, "attention_connection_state") != "dry_run":
            blocking_reasons.append("attention_connection_not_dry_run")
        if _event_value(dry_run_result, "attention_connection_mode") != "metadata_only":
            blocking_reasons.append("attention_connection_not_metadata_only")
        if _event_value(dry_run_result, "attention_connection_attempted") is not True:
            blocking_reasons.append("attention_connection_not_attempted")
        if not working_kv_block_ids:
            blocking_reasons.append("no_working_kv_blocks")

        if not blocking_reasons:
            attention_connection_attempted = True
            attention_override_noop = True

        results.append(
            {
                "event_type": "relaykv_attention_override_noop_result",
                "attention_connection_state": (
                    "override_noop" if attention_override_noop else "blocked"
                ),
                "attention_connection_mode": (
                    "noop_guarded" if attention_override_noop else "metadata_only"
                ),
                "source": (
                    "attention_connection_dry_run_result_to_override_noop_result"
                ),
                "request_id": _event_value(dry_run_result, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(dry_run_result),
                "seq_len": _event_value(dry_run_result, "seq_len"),
                "layer_id": _event_layer_value(dry_run_result),
                "working_kv_block_ids": working_kv_block_ids,
                "working_kv_block_count": working_kv_block_count,
                "working_kv_token_count": working_kv_token_count,
                "attention_target_layer_id": (
                    _event_value(dry_run_result, "attention_target_layer_id")
                    if _event_value(dry_run_result, "attention_target_layer_id")
                    is not None
                    else _event_layer_value(dry_run_result)
                ),
                "attention_target_backend": "unconnected",
                "attention_connection_attempted": attention_connection_attempted,
                "attention_override": False,
                "attention_override_noop": attention_override_noop,
                "attention_override_allowed": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
                "warning_reasons": warning_reasons,
            }
        )
    return results


def summarize_relaykv_attention_override_noop_results_for_smoke(
    results: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize guarded no-op attention override results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV attention override noop results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_attention_connection_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "attention_override_true_count": 0,
            "attention_override_noop_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    attention_connection_attempted_count = 0
    attention_override_noop_count = 0
    blocked_count = 0
    error_count = 0
    working_kv_block_count = 0
    working_kv_token_count = 0

    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError("RelayKV attention override noop result must be a mapping")
        state = str(_event_value(result, "attention_connection_state") or "unknown")
        per_attention_connection_state[state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1
        if state == "override_noop":
            attention_override_noop_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        block_count = _event_value(result, "working_kv_block_count")
        if isinstance(block_count, int):
            working_kv_block_count += block_count
        token_count = _event_value(result, "working_kv_token_count")
        if isinstance(token_count, int):
            working_kv_token_count += token_count
        if _event_value(result, "attention_connection_attempted") is True:
            attention_connection_attempted_count += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "attention_override_noop") is True:
            safety_counts["attention_override_noop_count"] += 1
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_attention_override_noop_result_summary",
        "total_attention_override_noop_results": len(results),
        "attention_override_noop_count": attention_override_noop_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "working_kv_block_count": working_kv_block_count,
        "working_kv_token_count": working_kv_token_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_attention_connection_state_counts": dict(
            sorted(per_attention_connection_state.items())
        ),
        "attention_connection_attempted_count": attention_connection_attempted_count,
        **dict(safety_counts),
    }


def _unique_sorted_int_list(values: Iterable[Any]) -> list[int]:
    unique_values: set[int] = set()
    for value in values:
        if isinstance(value, int):
            unique_values.add(value)
    return sorted(unique_values)


def _resolve_attention_comparison_full_kv_block_ids(
    result: Mapping[str, Any],
    full_kv_block_ids_by_request_layer: Mapping[Any, Any] | None,
) -> list[int]:
    request_id = _event_value(result, "request_id")
    layer_id = _event_layer_value(result)

    if full_kv_block_ids_by_request_layer is not None:
        lookup_keys = (
            (request_id, layer_id),
            f"{request_id}:{layer_id}",
            request_id,
        )
        for lookup_key in lookup_keys:
            if lookup_key in full_kv_block_ids_by_request_layer:
                lookup_value = full_kv_block_ids_by_request_layer[lookup_key]
                if isinstance(lookup_value, (list, tuple)):
                    return _unique_sorted_int_list(lookup_value)
                return []

    synthesized_values: list[int] = []
    for field_name in (
        "recent_block_ids",
        "anchor_block_ids",
        "candidate_block_ids",
        "retrieved_block_ids",
        "materialized_block_ids",
        "working_kv_block_ids",
    ):
        field_value = _event_value(result, field_name)
        if isinstance(field_value, (list, tuple)):
            synthesized_values.extend(field_value)
    return _unique_sorted_int_list(synthesized_values)


def build_relaykv_attention_comparison_plans_for_smoke(
    attention_override_noop_results: list[Mapping[str, Any]]
    | tuple[Mapping[str, Any], ...],
    full_kv_block_ids_by_request_layer: Mapping[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build metadata-only attention comparison plans."""

    if not isinstance(attention_override_noop_results, (list, tuple)):
        raise TypeError("attention_override_noop_results must be a list or tuple")
    if full_kv_block_ids_by_request_layer is not None and not isinstance(
        full_kv_block_ids_by_request_layer, Mapping
    ):
        raise TypeError(
            "full_kv_block_ids_by_request_layer must be a mapping or None"
        )

    plans: list[dict[str, Any]] = []
    for noop_result in attention_override_noop_results:
        if not isinstance(noop_result, Mapping):
            raise TypeError(
                "RelayKV attention comparison plan inputs must be mappings"
            )

        working_kv_block_ids = _host_backup_copy_request_block_ids(
            noop_result, "working_kv_block_ids"
        )
        working_kv_block_count = _event_value(noop_result, "working_kv_block_count")
        if not isinstance(working_kv_block_count, int):
            working_kv_block_count = len(working_kv_block_ids)
        working_kv_token_count = _event_value(noop_result, "working_kv_token_count")
        if not isinstance(working_kv_token_count, int):
            working_kv_token_count = 0

        full_kv_block_ids = _resolve_attention_comparison_full_kv_block_ids(
            noop_result, full_kv_block_ids_by_request_layer
        )
        full_kv_block_count = len(full_kv_block_ids)
        working_kv_block_id_set = set(working_kv_block_ids)
        full_kv_block_id_set = set(full_kv_block_ids)
        coverage_block_count = sum(
            1 for block_id in working_kv_block_ids if block_id in full_kv_block_id_set
        )
        missing_from_full_block_ids = [
            block_id for block_id in working_kv_block_ids if block_id not in full_kv_block_id_set
        ]
        full_only_block_ids = [
            block_id for block_id in full_kv_block_ids if block_id not in working_kv_block_id_set
        ]
        reduced_block_count = max(full_kv_block_count - working_kv_block_count, 0)
        working_to_full_block_ratio = (
            working_kv_block_count / full_kv_block_count
            if full_kv_block_count > 0
            else None
        )
        coverage_ratio = (
            coverage_block_count / full_kv_block_count
            if full_kv_block_count > 0
            else None
        )

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = ["metadata_only_attention_comparison_plan"]

        if (
            _event_value(noop_result, "event_type")
            != "relaykv_attention_override_noop_result"
        ):
            blocking_reasons.append("not_attention_override_noop_result")
        if _event_value(noop_result, "attention_connection_state") != "override_noop":
            blocking_reasons.append("attention_connection_not_override_noop")
        if _event_value(noop_result, "attention_connection_mode") != "noop_guarded":
            blocking_reasons.append("attention_connection_not_noop_guarded")
        if _event_value(noop_result, "attention_override_noop") is not True:
            blocking_reasons.append("attention_override_noop_not_true")
        if _event_value(noop_result, "attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if not working_kv_block_ids:
            blocking_reasons.append("no_working_kv_blocks")
        if not full_kv_block_ids:
            blocking_reasons.append("no_full_kv_blocks")

        attention_connection_attempted = not blocking_reasons and (
            _event_value(noop_result, "attention_connection_attempted") is True
        )
        comparison_state = "plan_ready" if not blocking_reasons else "blocked"

        plans.append(
            {
                "event_type": "relaykv_attention_comparison_plan",
                "comparison_state": comparison_state,
                "comparison_mode": "metadata_only",
                "source": "attention_override_noop_result_to_comparison_plan",
                "request_id": _event_value(noop_result, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(noop_result),
                "seq_len": _event_value(noop_result, "seq_len"),
                "layer_id": _event_layer_value(noop_result),
                "full_kv_block_ids": full_kv_block_ids,
                "relaykv_working_kv_block_ids": working_kv_block_ids,
                "relaykv_working_kv_block_count": working_kv_block_count,
                "full_kv_block_count": full_kv_block_count,
                "reduced_block_count": reduced_block_count,
                "working_to_full_block_ratio": working_to_full_block_ratio,
                "coverage_block_count": coverage_block_count,
                "coverage_ratio": coverage_ratio,
                "missing_from_full_block_ids": missing_from_full_block_ids,
                "full_only_block_ids": full_only_block_ids,
                "attention_comparison_executed": False,
                "attention_connection_attempted": attention_connection_attempted,
                "attention_override": False,
                "attention_override_noop": (
                    True if not blocking_reasons else False
                ),
                "kv_pool_read": False,
                "kv_snapshot": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
                "warning_reasons": warning_reasons,
            }
        )
    return plans


def summarize_relaykv_attention_comparison_plans_for_smoke(
    plans: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize metadata-only attention comparison plans."""

    if not isinstance(plans, (list, tuple)):
        raise TypeError("RelayKV attention comparison plans must be a list or tuple")

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_comparison_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "attention_override_true_count": 0,
            "attention_override_noop_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    comparison_plan_ready_count = 0
    blocked_count = 0
    error_count = 0
    full_kv_block_count = 0
    relaykv_working_kv_block_count = 0
    reduced_block_count = 0
    working_to_full_ratio_sum = 0.0
    working_to_full_ratio_count = 0
    coverage_ratio_sum = 0.0
    coverage_ratio_count = 0
    missing_from_full_block_count = 0
    full_only_block_count = 0
    attention_comparison_executed_count = 0
    attention_connection_attempted_count = 0

    for plan in plans:
        if not isinstance(plan, Mapping):
            raise TypeError("RelayKV attention comparison plan must be a mapping")
        state = str(_event_value(plan, "comparison_state") or "unknown")
        per_comparison_state[state] += 1
        per_request[str(_event_value(plan, "request_id"))] += 1
        per_layer[str(_event_layer_value(plan))] += 1
        if state == "plan_ready":
            comparison_plan_ready_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = _event_value(plan, "full_kv_block_count")
        if isinstance(value, int):
            full_kv_block_count += value
        value = _event_value(plan, "relaykv_working_kv_block_count")
        if isinstance(value, int):
            relaykv_working_kv_block_count += value
        value = _event_value(plan, "reduced_block_count")
        if isinstance(value, int):
            reduced_block_count += value
        value = _event_value(plan, "working_to_full_block_ratio")
        if isinstance(value, (int, float)):
            working_to_full_ratio_sum += float(value)
            working_to_full_ratio_count += 1
        value = _event_value(plan, "coverage_ratio")
        if isinstance(value, (int, float)):
            coverage_ratio_sum += float(value)
            coverage_ratio_count += 1
        value = _event_value(plan, "missing_from_full_block_ids")
        if isinstance(value, (list, tuple)):
            missing_from_full_block_count += len(value)
        value = _event_value(plan, "full_only_block_ids")
        if isinstance(value, (list, tuple)):
            full_only_block_count += len(value)
        if _event_value(plan, "attention_comparison_executed") is True:
            attention_comparison_executed_count += 1
        if _event_value(plan, "attention_connection_attempted") is True:
            attention_connection_attempted_count += 1
        if _event_value(plan, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(plan, "attention_override_noop") is True:
            safety_counts["attention_override_noop_count"] += 1
        if _event_value(plan, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(plan, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(plan, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(plan, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(plan, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(plan, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_attention_comparison_plan_summary",
        "total_attention_comparison_plans": len(plans),
        "comparison_plan_ready_count": comparison_plan_ready_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "full_kv_block_count": full_kv_block_count,
        "relaykv_working_kv_block_count": relaykv_working_kv_block_count,
        "reduced_block_count": reduced_block_count,
        "mean_working_to_full_block_ratio": (
            working_to_full_ratio_sum / working_to_full_ratio_count
            if working_to_full_ratio_count > 0
            else None
        ),
        "mean_coverage_ratio": (
            coverage_ratio_sum / coverage_ratio_count
            if coverage_ratio_count > 0
            else None
        ),
        "missing_from_full_block_count": missing_from_full_block_count,
        "full_only_block_count": full_only_block_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_comparison_state_counts": dict(sorted(per_comparison_state.items())),
        "attention_comparison_executed_count": attention_comparison_executed_count,
        "attention_connection_attempted_count": attention_connection_attempted_count,
        **dict(safety_counts),
    }


def build_relaykv_attention_shadow_capture_results_for_smoke(
    attention_comparison_plans: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    capture_attention_output: bool = False,
) -> list[dict[str, Any]]:
    """Build metadata-only RelayKV attention shadow capture results."""

    if not isinstance(attention_comparison_plans, (list, tuple)):
        raise TypeError("attention_comparison_plans must be a list or tuple")
    if not isinstance(capture_attention_output, bool):
        raise TypeError("capture_attention_output must be a bool")

    results: list[dict[str, Any]] = []
    for plan in attention_comparison_plans:
        if not isinstance(plan, dict):
            raise TypeError(
                "RelayKV attention shadow capture inputs must be dict comparison plans"
            )

        relaykv_working_kv_block_ids = _host_backup_copy_request_block_ids(
            plan, "relaykv_working_kv_block_ids"
        )
        full_kv_block_ids = _host_backup_copy_request_block_ids(plan, "full_kv_block_ids")

        relaykv_working_kv_block_count = _event_value(
            plan, "relaykv_working_kv_block_count"
        )
        if not isinstance(relaykv_working_kv_block_count, int):
            relaykv_working_kv_block_count = len(relaykv_working_kv_block_ids)

        full_kv_block_count = _event_value(plan, "full_kv_block_count")
        if not isinstance(full_kv_block_count, int):
            full_kv_block_count = len(full_kv_block_ids)

        reduced_block_count = _event_value(plan, "reduced_block_count")
        if not isinstance(reduced_block_count, int):
            reduced_block_count = max(
                full_kv_block_count - relaykv_working_kv_block_count, 0
            )

        working_to_full_block_ratio = _event_value(plan, "working_to_full_block_ratio")
        if not isinstance(working_to_full_block_ratio, (int, float)):
            if full_kv_block_count > 0:
                working_to_full_block_ratio = (
                    relaykv_working_kv_block_count / full_kv_block_count
                )
            else:
                working_to_full_block_ratio = None

        coverage_ratio = _event_value(plan, "coverage_ratio")
        if not isinstance(coverage_ratio, (int, float)):
            coverage_block_count = _event_value(plan, "coverage_block_count")
            if isinstance(coverage_block_count, int) and full_kv_block_count > 0:
                coverage_ratio = coverage_block_count / full_kv_block_count
            elif full_kv_block_count > 0:
                coverage_ratio = relaykv_working_kv_block_count / full_kv_block_count
            else:
                coverage_ratio = None

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = [
            "metadata_only_attention_shadow_capture",
            "no_attention_output_tensor_captured",
        ]

        if capture_attention_output:
            blocking_reasons.append(
                "capture_attention_output_not_allowed_in_metadata_smoke"
            )
        if _event_value(plan, "event_type") != "relaykv_attention_comparison_plan":
            blocking_reasons.append("not_attention_comparison_plan")
        if _event_value(plan, "comparison_state") != "plan_ready":
            blocking_reasons.append("comparison_plan_not_ready")
        if _event_value(plan, "comparison_mode") != "metadata_only":
            blocking_reasons.append("comparison_plan_not_metadata_only")
        if _event_value(plan, "attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_already_executed")
        if _event_value(plan, "attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if _event_value(plan, "attention_override_noop") is not True:
            blocking_reasons.append("attention_override_noop_not_true")
        if not relaykv_working_kv_block_ids:
            blocking_reasons.append("no_relaykv_working_kv_blocks")
        if not full_kv_block_ids:
            blocking_reasons.append("no_full_kv_blocks")

        shadow_capture_attempted = not blocking_reasons
        shadow_capture_count = 1 if shadow_capture_attempted else 0

        results.append(
            {
                "event_type": "relaykv_attention_shadow_capture_result",
                "shadow_capture_state": (
                    "metadata_shadow_captured"
                    if shadow_capture_attempted
                    else "blocked"
                ),
                "shadow_capture_mode": "metadata_only",
                "source": "attention_comparison_plan_to_shadow_capture_result",
                "request_id": _event_value(plan, "request_id"),
                "req_pool_idx": _event_req_pool_idx_value(plan),
                "seq_len": _event_value(plan, "seq_len"),
                "layer_id": _event_layer_value(plan),
                "full_kv_block_ids": list(full_kv_block_ids),
                "relaykv_working_kv_block_ids": list(relaykv_working_kv_block_ids),
                "full_kv_block_count": full_kv_block_count,
                "relaykv_working_kv_block_count": relaykv_working_kv_block_count,
                "reduced_block_count": reduced_block_count,
                "working_to_full_block_ratio": working_to_full_block_ratio,
                "coverage_ratio": coverage_ratio,
                "shadow_capture_attempted": shadow_capture_attempted,
                "attention_shadow_capture_count": shadow_capture_count,
                "attention_output_captured": False,
                "attention_comparison_executed": False,
                "attention_connection_attempted": shadow_capture_attempted,
                "attention_override": False,
                "attention_override_noop": True if shadow_capture_attempted else False,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
                "warning_reasons": warning_reasons,
            }
        )
    return results


def summarize_relaykv_attention_shadow_capture_results_for_smoke(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize metadata-only RelayKV attention shadow capture results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV attention shadow capture results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_shadow_capture_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "attention_override_true_count": 0,
            "attention_override_noop_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    shadow_capture_count = 0
    blocked_count = 0
    error_count = 0
    full_kv_block_count = 0
    relaykv_working_kv_block_count = 0
    reduced_block_count = 0
    working_to_full_ratio_sum = 0.0
    working_to_full_ratio_count = 0
    coverage_ratio_sum = 0.0
    coverage_ratio_count = 0
    shadow_capture_attempted_count = 0
    attention_shadow_capture_count = 0
    attention_output_captured_count = 0
    attention_comparison_executed_count = 0
    attention_connection_attempted_count = 0

    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV attention shadow capture result must be a dict"
            )

        state = str(_event_value(result, "shadow_capture_state") or "unknown")
        per_shadow_capture_state[state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1
        if state == "metadata_shadow_captured":
            shadow_capture_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = _event_value(result, "full_kv_block_count")
        if isinstance(value, int):
            full_kv_block_count += value
        value = _event_value(result, "relaykv_working_kv_block_count")
        if isinstance(value, int):
            relaykv_working_kv_block_count += value
        value = _event_value(result, "reduced_block_count")
        if isinstance(value, int):
            reduced_block_count += value
        value = _event_value(result, "working_to_full_block_ratio")
        if isinstance(value, (int, float)):
            working_to_full_ratio_sum += float(value)
            working_to_full_ratio_count += 1
        value = _event_value(result, "coverage_ratio")
        if isinstance(value, (int, float)):
            coverage_ratio_sum += float(value)
            coverage_ratio_count += 1
        if _event_value(result, "shadow_capture_attempted") is True:
            shadow_capture_attempted_count += 1
        value = _event_value(result, "attention_shadow_capture_count")
        if isinstance(value, int):
            attention_shadow_capture_count += value
        if _event_value(result, "attention_output_captured") is True:
            attention_output_captured_count += 1
        if _event_value(result, "attention_comparison_executed") is True:
            attention_comparison_executed_count += 1
        if _event_value(result, "attention_connection_attempted") is True:
            attention_connection_attempted_count += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "attention_override_noop") is True:
            safety_counts["attention_override_noop_count"] += 1
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(result, "tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_attention_shadow_capture_result_summary",
        "total_attention_shadow_capture_results": len(results),
        "shadow_capture_count": shadow_capture_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "full_kv_block_count": full_kv_block_count,
        "relaykv_working_kv_block_count": relaykv_working_kv_block_count,
        "reduced_block_count": reduced_block_count,
        "mean_working_to_full_block_ratio": (
            working_to_full_ratio_sum / working_to_full_ratio_count
            if working_to_full_ratio_count > 0
            else None
        ),
        "mean_coverage_ratio": (
            coverage_ratio_sum / coverage_ratio_count
            if coverage_ratio_count > 0
            else None
        ),
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_shadow_capture_state_counts": dict(
            sorted(per_shadow_capture_state.items())
        ),
        "shadow_capture_attempted_count": shadow_capture_attempted_count,
        "attention_shadow_capture_count": attention_shadow_capture_count,
        "attention_output_captured_count": attention_output_captured_count,
        "attention_comparison_executed_count": attention_comparison_executed_count,
        "attention_connection_attempted_count": attention_connection_attempted_count,
        **dict(safety_counts),
    }


def _resolve_block_metadata_entry_for_smoke(
    block_metadata_by_id: dict[Any, Any] | None,
    request_id: Any,
    layer_id: Any,
    block_id: Any,
) -> tuple[Any, str | None]:
    if block_metadata_by_id is None:
        return None, None

    lookup_items = (
        ((request_id, layer_id, block_id), "request_layer_block_tuple"),
        (f"{request_id}:{layer_id}:{block_id}", "request_layer_block_string"),
        (block_id, "block_id"),
        (str(block_id), "block_id_string"),
    )
    for lookup_key, source_name in lookup_items:
        if lookup_key in block_metadata_by_id:
            return block_metadata_by_id[lookup_key], source_name
    return None, None


def _normalize_block_span_for_smoke(
    metadata_value: Any,
    request_id: Any,
    layer_id: Any,
    block_id: Any,
    seq_len: int,
    span_source: str,
) -> tuple[dict[str, Any] | None, str | None]:
    token_start: Any = None
    token_end: Any = None

    if isinstance(metadata_value, dict):
        if (
            isinstance(metadata_value.get("token_start"), int)
            and isinstance(metadata_value.get("token_end"), int)
        ):
            token_start = metadata_value.get("token_start")
            token_end = metadata_value.get("token_end")
            span_source = f"{span_source}:token_start_token_end"
        elif (
            isinstance(metadata_value.get("start_token"), int)
            and isinstance(metadata_value.get("end_token"), int)
        ):
            token_start = metadata_value.get("start_token")
            token_end = metadata_value.get("end_token")
            span_source = f"{span_source}:start_token_end_token"
        else:
            token_span = metadata_value.get("token_span")
            if (
                isinstance(token_span, (list, tuple))
                and len(token_span) == 2
                and isinstance(token_span[0], int)
                and isinstance(token_span[1], int)
            ):
                token_start = token_span[0]
                token_end = token_span[1]
                span_source = f"{span_source}:token_span"
    elif (
        isinstance(metadata_value, (list, tuple))
        and len(metadata_value) == 2
        and isinstance(metadata_value[0], int)
        and isinstance(metadata_value[1], int)
    ):
        token_start = metadata_value[0]
        token_end = metadata_value[1]
        span_source = f"{span_source}:sequence_span"

    if not isinstance(token_start, int) or not isinstance(token_end, int):
        return None, "invalid_block_span"
    if token_start < 0 or token_end <= token_start:
        return None, "invalid_block_span"
    if token_end > seq_len:
        return None, "block_span_out_of_seq_len"

    return (
        {
            "block_id": block_id,
            "token_start": token_start,
            "token_end": token_end,
            "token_count": token_end - token_start,
            "request_id": request_id,
            "layer_id": layer_id,
            "span_source": span_source,
        },
        None,
    )


def _resolve_block_spans_for_smoke(
    block_ids: list[int],
    *,
    request_id: Any,
    layer_id: Any,
    seq_len: int,
    block_metadata_by_id: dict[Any, Any] | None,
) -> tuple[list[dict[str, Any]], list[int], list[int], list[str]]:
    spans: list[dict[str, Any]] = []
    missing_block_ids: list[int] = []
    invalid_block_ids: list[int] = []
    blocking_reasons: list[str] = []

    for block_id in block_ids:
        metadata_value, key_source = _resolve_block_metadata_entry_for_smoke(
            block_metadata_by_id,
            request_id,
            layer_id,
            block_id,
        )
        if metadata_value is None or key_source is None:
            missing_block_ids.append(block_id)
            continue

        span, span_error = _normalize_block_span_for_smoke(
            metadata_value,
            request_id,
            layer_id,
            block_id,
            seq_len,
            key_source,
        )
        if span_error is not None:
            invalid_block_ids.append(block_id)
            blocking_reasons.append(span_error)
            continue
        if span is not None:
            spans.append(span)

    if missing_block_ids:
        blocking_reasons.append("missing_block_metadata")
    if invalid_block_ids and "invalid_block_span" not in blocking_reasons:
        blocking_reasons.append("invalid_block_span")

    return (
        spans,
        missing_block_ids,
        invalid_block_ids,
        list(dict.fromkeys(blocking_reasons)),
    )


def build_relaykv_kv_index_resolution_plans_for_smoke(
    attention_shadow_capture_results: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
    block_metadata_by_id: dict[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build metadata-only KV index resolution plans from shadow capture results."""

    if not isinstance(attention_shadow_capture_results, (list, tuple)):
        raise TypeError(
            "attention_shadow_capture_results must be a list or tuple"
        )
    if block_metadata_by_id is not None and not isinstance(block_metadata_by_id, dict):
        raise TypeError("block_metadata_by_id must be a dict or None")

    plans: list[dict[str, Any]] = []
    for result in attention_shadow_capture_results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV KV index resolution plan inputs must be dict results"
            )

        request_id = _event_value(result, "request_id")
        req_pool_idx = _event_req_pool_idx_value(result)
        seq_len = _event_value(result, "seq_len")
        layer_id = _event_layer_value(result)
        relaykv_working_kv_block_ids = _host_backup_copy_request_block_ids(
            result, "relaykv_working_kv_block_ids"
        )
        full_kv_block_ids = _host_backup_copy_request_block_ids(result, "full_kv_block_ids")

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = [
            "metadata_only_kv_index_resolution_plan",
            "no_req_to_token_pool_read",
        ]

        if _event_value(result, "event_type") != "relaykv_attention_shadow_capture_result":
            blocking_reasons.append("not_attention_shadow_capture_result")
        if _event_value(result, "shadow_capture_state") != "metadata_shadow_captured":
            blocking_reasons.append("shadow_capture_not_metadata_captured")
        if _event_value(result, "shadow_capture_mode") != "metadata_only":
            blocking_reasons.append("shadow_capture_not_metadata_only")
        if _event_value(result, "attention_output_captured") is True:
            blocking_reasons.append("attention_output_captured_not_allowed")
        if _event_value(result, "attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if _event_value(result, "attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if not relaykv_working_kv_block_ids:
            blocking_reasons.append("no_relaykv_working_kv_blocks")
        if not full_kv_block_ids:
            blocking_reasons.append("no_full_kv_blocks")
        if not isinstance(seq_len, int) or seq_len <= 0:
            blocking_reasons.append("seq_len_missing_or_invalid")

        relaykv_working_block_spans: list[dict[str, Any]] = []
        full_kv_block_spans: list[dict[str, Any]] = []
        missing_block_ids: list[int] = []
        invalid_block_ids: list[int] = []

        if not blocking_reasons:
            (
                full_kv_block_spans,
                full_missing_block_ids,
                full_invalid_block_ids,
                full_blocking_reasons,
            ) = _resolve_block_spans_for_smoke(
                full_kv_block_ids,
                request_id=request_id,
                layer_id=layer_id,
                seq_len=seq_len,
                block_metadata_by_id=block_metadata_by_id,
            )
            (
                relaykv_working_block_spans,
                working_missing_block_ids,
                working_invalid_block_ids,
                working_blocking_reasons,
            ) = _resolve_block_spans_for_smoke(
                relaykv_working_kv_block_ids,
                request_id=request_id,
                layer_id=layer_id,
                seq_len=seq_len,
                block_metadata_by_id=block_metadata_by_id,
            )

            missing_block_ids = list(
                dict.fromkeys(full_missing_block_ids + working_missing_block_ids)
            )
            invalid_block_ids = list(
                dict.fromkeys(full_invalid_block_ids + working_invalid_block_ids)
            )
            blocking_reasons.extend(full_blocking_reasons)
            blocking_reasons.extend(working_blocking_reasons)

        blocking_reasons = list(dict.fromkeys(blocking_reasons))
        resolution_state = (
            "block_span_resolved" if not blocking_reasons else "blocked"
        )

        if blocking_reasons:
            relaykv_working_block_spans = []
            full_kv_block_spans = []
            resolved_block_count = 0
            token_span_count = 0
            total_token_count = 0
        else:
            resolved_block_count = len(full_kv_block_spans)
            token_span_count = len(full_kv_block_spans)
            total_token_count = sum(
                span["token_count"] for span in full_kv_block_spans
            )

        plans.append(
            {
                "event_type": "relaykv_kv_index_resolution_plan",
                "resolution_state": resolution_state,
                "resolution_mode": "metadata_only",
                "source": "attention_shadow_capture_result_to_kv_index_resolution_plan",
                "request_id": request_id,
                "req_pool_idx": req_pool_idx,
                "seq_len": seq_len,
                "layer_id": layer_id,
                "relaykv_working_kv_block_ids": list(relaykv_working_kv_block_ids),
                "full_kv_block_ids": list(full_kv_block_ids),
                "relaykv_working_block_spans": relaykv_working_block_spans,
                "full_kv_block_spans": full_kv_block_spans,
                "resolved_block_count": resolved_block_count,
                "missing_block_ids": missing_block_ids,
                "invalid_block_ids": invalid_block_ids,
                "token_span_count": token_span_count,
                "total_token_count": total_token_count,
                "req_to_token_read": False,
                "token_to_kv_pool_read": False,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": blocking_reasons,
                "warning_reasons": warning_reasons,
            }
        )
    return plans


def summarize_relaykv_kv_index_resolution_plans_for_smoke(
    plans: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize metadata-only KV index resolution plans."""

    if not isinstance(plans, (list, tuple)):
        raise TypeError(
            "RelayKV KV index resolution plans must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_resolution_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    block_span_resolved_count = 0
    blocked_count = 0
    error_count = 0
    resolved_block_count = 0
    missing_block_count = 0
    invalid_block_count = 0
    token_span_count = 0
    total_token_count = 0

    for plan in plans:
        if not isinstance(plan, dict):
            raise TypeError("RelayKV KV index resolution plan must be a dict")

        state = str(_event_value(plan, "resolution_state") or "unknown")
        per_resolution_state[state] += 1
        per_request[str(_event_value(plan, "request_id"))] += 1
        per_layer[str(_event_layer_value(plan))] += 1

        if state == "block_span_resolved":
            block_span_resolved_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = _event_value(plan, "resolved_block_count")
        if isinstance(value, int):
            resolved_block_count += value
        value = _event_value(plan, "missing_block_ids")
        if isinstance(value, (list, tuple)):
            missing_block_count += len(value)
        value = _event_value(plan, "invalid_block_ids")
        if isinstance(value, (list, tuple)):
            invalid_block_count += len(value)
        value = _event_value(plan, "token_span_count")
        if isinstance(value, int):
            token_span_count += value
        value = _event_value(plan, "total_token_count")
        if isinstance(value, int):
            total_token_count += value

        if _event_value(plan, "req_to_token_read") is True:
            safety_counts["req_to_token_read_count"] += 1
        if _event_value(plan, "token_to_kv_pool_read") is True:
            safety_counts["token_to_kv_pool_read_count"] += 1
        if _event_value(plan, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(plan, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(plan, "tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if _event_value(plan, "attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if _event_value(plan, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(plan, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(plan, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(plan, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(plan, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_kv_index_resolution_plan_summary",
        "total_kv_index_resolution_plans": len(plans),
        "block_span_resolved_count": block_span_resolved_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "resolved_block_count": resolved_block_count,
        "missing_block_count": missing_block_count,
        "invalid_block_count": invalid_block_count,
        "token_span_count": token_span_count,
        "total_token_count": total_token_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_resolution_state_counts": dict(sorted(per_resolution_state.items())),
        **dict(safety_counts),
    }


def _req_to_token_table_for_req_pool_idx_for_smoke(
    req_to_token_table_by_req_pool_idx: dict[Any, Any] | None,
    req_pool_idx: Any,
) -> Any:
    if not isinstance(req_to_token_table_by_req_pool_idx, dict):
        return None
    if req_pool_idx in req_to_token_table_by_req_pool_idx:
        return req_to_token_table_by_req_pool_idx[req_pool_idx]
    req_pool_idx_as_str = str(req_pool_idx)
    if req_pool_idx_as_str in req_to_token_table_by_req_pool_idx:
        return req_to_token_table_by_req_pool_idx[req_pool_idx_as_str]
    return None


def _req_to_token_int_like_value_for_smoke(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _normalize_req_to_token_span_for_smoke(
    span: Mapping[str, Any],
    *,
    request_id: Any,
    req_pool_idx: int,
    layer_id: Any,
    seq_len: int,
    req_to_token_table: list[Any] | tuple[Any, ...],
) -> tuple[dict[str, Any] | None, str | None]:
    block_id = _event_value(span, "block_id")
    token_start = _event_value(span, "token_start")
    token_end = _event_value(span, "token_end")
    token_count = _event_value(span, "token_count")

    if not isinstance(token_start, int) or not isinstance(token_end, int):
        return None, "invalid_block_span"
    if not isinstance(token_count, int) or token_count != token_end - token_start:
        return None, "invalid_block_span"
    if token_start < 0 or token_end <= token_start:
        return None, "invalid_block_span"
    if token_end > seq_len:
        return None, "token_span_out_of_seq_len"

    req_to_token_entries: list[int] = []
    for position in range(token_start, token_end):
        if position >= len(req_to_token_table):
            return None, "token_position_out_of_req_to_token_table"
        entry = req_to_token_table[position]
        if not isinstance(entry, int) or isinstance(entry, bool):
            return None, "req_to_token_entry_not_int"
        req_to_token_entries.append(entry)

    return (
        {
            "block_id": block_id,
            "token_start": token_start,
            "token_end": token_end,
            "token_count": token_count,
            "request_id": request_id,
            "req_pool_idx": req_pool_idx,
            "layer_id": layer_id,
            "req_to_token_entries": req_to_token_entries,
            "entry_count": len(req_to_token_entries),
            "resolution_source": "synthetic_req_to_token_table",
        },
        None,
    )


def _resolve_req_to_token_spans_for_smoke(
    spans: list[dict[str, Any]],
    *,
    request_id: Any,
    req_pool_idx: int,
    layer_id: Any,
    seq_len: int,
    req_to_token_table: list[Any] | tuple[Any, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved_spans: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []

    for span in spans:
        if not isinstance(span, Mapping):
            blocking_reasons.append("invalid_block_span")
            continue
        resolved_span, span_error = _normalize_req_to_token_span_for_smoke(
            span,
            request_id=request_id,
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            seq_len=seq_len,
            req_to_token_table=req_to_token_table,
        )
        if span_error is not None:
            blocking_reasons.append(span_error)
            continue
        if resolved_span is not None:
            resolved_spans.append(resolved_span)

    return resolved_spans, list(dict.fromkeys(blocking_reasons))


def build_relaykv_req_to_token_resolution_results_for_smoke(
    kv_index_resolution_plans: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
    req_to_token_table_by_req_pool_idx: dict[Any, Any] | None = None,
    read_req_to_token: bool = False,
) -> list[dict[str, Any]]:
    """Build readonly synthetic req_to_token resolution results for smoke."""

    if not isinstance(kv_index_resolution_plans, (list, tuple)):
        raise TypeError("kv_index_resolution_plans must be a list or tuple")
    if (
        req_to_token_table_by_req_pool_idx is not None
        and not isinstance(req_to_token_table_by_req_pool_idx, dict)
    ):
        raise TypeError(
            "req_to_token_table_by_req_pool_idx must be a dict or None"
        )

    results: list[dict[str, Any]] = []
    for plan in kv_index_resolution_plans:
        if not isinstance(plan, dict):
            raise TypeError(
                "RelayKV req_to_token resolution inputs must be dict plans"
            )

        request_id = _event_value(plan, "request_id")
        req_pool_idx_value = _event_req_pool_idx_value(plan)
        req_pool_idx = _req_to_token_int_like_value_for_smoke(req_pool_idx_value)
        seq_len = _event_value(plan, "seq_len")
        layer_id = _event_layer_value(plan)
        relaykv_working_block_spans = _event_value(plan, "relaykv_working_block_spans")
        full_kv_block_spans = _event_value(plan, "full_kv_block_spans")

        blocking_reasons: list[str] = []
        warning_reasons: list[str] = [
            "readonly_req_to_token_resolution",
            "no_token_to_kv_pool_read",
        ]

        if _event_value(plan, "event_type") != "relaykv_kv_index_resolution_plan":
            blocking_reasons.append("not_kv_index_resolution_plan")
        if _event_value(plan, "resolution_state") != "block_span_resolved":
            blocking_reasons.append("kv_index_resolution_not_block_span_resolved")
        if _event_value(plan, "resolution_mode") != "metadata_only":
            blocking_reasons.append("kv_index_resolution_not_metadata_only")
        if read_req_to_token is not True:
            blocking_reasons.append("read_req_to_token_not_enabled")
        if req_pool_idx is None:
            blocking_reasons.append("req_pool_idx_missing_or_invalid")
        if not isinstance(seq_len, int) or seq_len <= 0:
            blocking_reasons.append("seq_len_missing_or_invalid")
        if not isinstance(relaykv_working_block_spans, (list, tuple)):
            blocking_reasons.append("invalid_block_span")
        elif not relaykv_working_block_spans:
            blocking_reasons.append("invalid_block_span")
        if not isinstance(full_kv_block_spans, (list, tuple)):
            blocking_reasons.append("invalid_block_span")
        elif not full_kv_block_spans:
            blocking_reasons.append("invalid_block_span")
        if _event_value(plan, "token_to_kv_pool_read") is True:
            blocking_reasons.append("token_to_kv_pool_read_not_allowed")
        if _event_value(plan, "kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if _event_value(plan, "tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if _event_value(plan, "attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if _event_value(plan, "attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if read_req_to_token and req_to_token_table_by_req_pool_idx is None:
            blocking_reasons.append("req_to_token_table_missing")

        req_to_token_table: list[Any] | tuple[Any, ...] | None = None
        relaykv_working_req_to_token_spans: list[dict[str, Any]] = []
        full_kv_req_to_token_spans: list[dict[str, Any]] = []

        if not blocking_reasons and req_pool_idx is not None:
            req_to_token_table = _req_to_token_table_for_req_pool_idx_for_smoke(
                req_to_token_table_by_req_pool_idx,
                req_pool_idx,
            )
            if req_to_token_table is None:
                blocking_reasons.append("req_to_token_table_for_req_pool_missing")
            elif not isinstance(req_to_token_table, (list, tuple)):
                blocking_reasons.append("req_to_token_table_not_indexable")

        if not blocking_reasons and req_to_token_table is not None:
            (
                full_kv_req_to_token_spans,
                full_blocking_reasons,
            ) = _resolve_req_to_token_spans_for_smoke(
                list(full_kv_block_spans),
                request_id=request_id,
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                seq_len=seq_len,
                req_to_token_table=req_to_token_table,
            )
            (
                relaykv_working_req_to_token_spans,
                working_blocking_reasons,
            ) = _resolve_req_to_token_spans_for_smoke(
                list(relaykv_working_block_spans),
                request_id=request_id,
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                seq_len=seq_len,
                req_to_token_table=req_to_token_table,
            )
            blocking_reasons.extend(full_blocking_reasons)
            blocking_reasons.extend(working_blocking_reasons)

        blocking_reasons = list(dict.fromkeys(blocking_reasons))
        resolution_state = (
            "req_to_token_resolved" if not blocking_reasons else "blocked"
        )

        if blocking_reasons:
            relaykv_working_req_to_token_spans = []
            full_kv_req_to_token_spans = []
            resolved_block_count = 0
            resolved_token_count = 0
            req_to_token_entry_count = 0
            req_to_token_read_flag = False
            req_to_token_read_count = 0
        else:
            resolved_block_count = len(full_kv_req_to_token_spans)
            resolved_token_count = sum(
                span["token_count"] for span in full_kv_req_to_token_spans
            )
            req_to_token_entry_count = sum(
                span["entry_count"] for span in full_kv_req_to_token_spans
            )
            req_to_token_read_flag = True
            req_to_token_read_count = req_to_token_entry_count

        results.append(
            {
                "event_type": "relaykv_req_to_token_resolution_result",
                "resolution_state": resolution_state,
                "resolution_mode": "readonly_synthetic_table",
                "source": (
                    "kv_index_resolution_plan_to_req_to_token_resolution_result"
                ),
                "request_id": request_id,
                "req_pool_idx": req_pool_idx_value,
                "seq_len": seq_len,
                "layer_id": layer_id,
                "relaykv_working_req_to_token_spans": (
                    relaykv_working_req_to_token_spans
                ),
                "full_kv_req_to_token_spans": full_kv_req_to_token_spans,
                "resolved_block_count": resolved_block_count,
                "resolved_token_count": resolved_token_count,
                "req_to_token_entry_count": req_to_token_entry_count,
                "req_to_token_read": req_to_token_read_flag,
                "req_to_token_read_count": req_to_token_read_count,
                "token_to_kv_pool_read": False,
                "token_to_kv_pool_read_count": 0,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": blocking_reasons,
                "warning_reasons": warning_reasons,
            }
        )
    return results


def summarize_relaykv_req_to_token_resolution_results_for_smoke(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize readonly synthetic req_to_token resolution results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV req_to_token resolution results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_resolution_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    req_to_token_resolved_count = 0
    blocked_count = 0
    error_count = 0
    resolved_block_count = 0
    resolved_token_count = 0
    req_to_token_entry_count = 0

    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV req_to_token resolution result must be a dict"
            )

        state = str(_event_value(result, "resolution_state") or "unknown")
        per_resolution_state[state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1

        if state == "req_to_token_resolved":
            req_to_token_resolved_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = _event_value(result, "resolved_block_count")
        if isinstance(value, int):
            resolved_block_count += value
        value = _event_value(result, "resolved_token_count")
        if isinstance(value, int):
            resolved_token_count += value
        value = _event_value(result, "req_to_token_entry_count")
        if isinstance(value, int):
            req_to_token_entry_count += value
        value = _event_value(result, "req_to_token_read_count")
        if isinstance(value, int):
            safety_counts["req_to_token_read_count"] += value
        value = _event_value(result, "token_to_kv_pool_read_count")
        if isinstance(value, int):
            safety_counts["token_to_kv_pool_read_count"] += value
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(result, "tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if _event_value(result, "attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_req_to_token_resolution_result_summary",
        "total_req_to_token_resolution_results": len(results),
        "req_to_token_resolved_count": req_to_token_resolved_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "resolved_block_count": resolved_block_count,
        "resolved_token_count": resolved_token_count,
        "req_to_token_entry_count": req_to_token_entry_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_resolution_state_counts": dict(sorted(per_resolution_state.items())),
        "req_to_token_read_count": safety_counts["req_to_token_read_count"],
        "token_to_kv_pool_read_count": (
            safety_counts["token_to_kv_pool_read_count"]
        ),
        "kv_pool_read_count": safety_counts["kv_pool_read_count"],
        "kv_snapshot_count": safety_counts["kv_snapshot_count"],
        "tensor_read_count": safety_counts["tensor_read_count"],
        "attention_comparison_executed_count": (
            safety_counts["attention_comparison_executed_count"]
        ),
        "attention_override_true_count": (
            safety_counts["attention_override_true_count"]
        ),
        "runtime_writeback_true_count": (
            safety_counts["runtime_writeback_true_count"]
        ),
        "scheduler_policy_noop_false_count": (
            safety_counts["scheduler_policy_noop_false_count"]
        ),
        "kv_cache_mutation_true_count": (
            safety_counts["kv_cache_mutation_true_count"]
        ),
        "source_mutated_true_count": safety_counts["source_mutated_true_count"],
    }


def _blocked_req_to_token_readonly_adapter_payload_for_smoke(
    plan: dict[str, Any],
    *,
    blocking_reasons: list[str],
    warning_reasons: list[str],
) -> dict[str, Any]:
    return {
        "event_type": "relaykv_req_to_token_readonly_adapter_payload",
        "adapter_state": "blocked",
        "adapter_mode": "readonly_bounded_preview",
        "source": (
            "kv_index_resolution_plan_to_req_to_token_readonly_adapter_payload"
        ),
        "request_id": plan.get("request_id"),
        "req_pool_idx": plan.get("req_pool_idx"),
        "seq_len": plan.get("seq_len"),
        "layer_id": plan.get("layer_id"),
        "requested_block_count": 0,
        "requested_token_count": 0,
        "read_token_count": 0,
        "preview_entry_count": 0,
        "preview_entries": [],
        "entry_count": 0,
        "entry_min": None,
        "entry_max": None,
        "entry_checksum": None,
        "truncated_preview": False,
        "req_to_token_read": False,
        "req_to_token_read_count": 0,
        "token_to_kv_pool_read": False,
        "token_to_kv_pool_read_count": 0,
        "kv_pool_read": False,
        "kv_snapshot": False,
        "tensor_read": False,
        "attention_comparison_executed": False,
        "attention_override": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
    }


def _readonly_adapter_unique_spans_for_smoke(
    working_spans: list[Any] | tuple[Any, ...],
    full_spans: list[Any] | tuple[Any, ...],
    *,
    seq_len: int,
) -> tuple[list[dict[str, int]], list[str]]:
    unique_spans: list[dict[str, int]] = []
    blocking_reasons: list[str] = []
    seen: set[tuple[int, int, int]] = set()

    for span in list(full_spans) + list(working_spans):
        if not isinstance(span, dict):
            blocking_reasons.append("invalid_block_span")
            continue
        block_id = span.get("block_id")
        token_start = span.get("token_start")
        token_end = span.get("token_end")
        token_count = span.get("token_count")
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            blocking_reasons.append("invalid_block_span")
            continue
        if not isinstance(token_start, int) or not isinstance(token_end, int):
            blocking_reasons.append("invalid_block_span")
            continue
        if not isinstance(token_count, int) or token_count != token_end - token_start:
            blocking_reasons.append("invalid_block_span")
            continue
        if token_start < 0 or token_end <= token_start:
            blocking_reasons.append("invalid_block_span")
            continue
        if token_end > seq_len:
            blocking_reasons.append("token_span_out_of_seq_len")
            continue
        span_key = (block_id, token_start, token_end)
        if span_key in seen:
            continue
        seen.add(span_key)
        unique_spans.append(
            {
                "block_id": block_id,
                "token_start": token_start,
                "token_end": token_end,
                "token_count": token_count,
            }
        )

    return unique_spans, list(dict.fromkeys(blocking_reasons))


def _read_req_to_token_entries_for_smoke(
    spans: list[dict[str, int]],
    *,
    req_to_token_backing: list[Any] | tuple[Any, ...],
) -> tuple[list[int], list[str]]:
    read_entries: list[int] = []
    blocking_reasons: list[str] = []

    for span in spans:
        token_start = span["token_start"]
        token_end = span["token_end"]
        for position in range(token_start, token_end):
            if position >= len(req_to_token_backing):
                blocking_reasons.append("token_position_out_of_req_to_token_table")
                return [], list(dict.fromkeys(blocking_reasons))
            entry = req_to_token_backing[position]
            if isinstance(entry, bool) or not isinstance(entry, int):
                blocking_reasons.append("req_to_token_entry_not_int")
                return [], list(dict.fromkeys(blocking_reasons))
            read_entries.append(entry)

    return read_entries, list(dict.fromkeys(blocking_reasons))


def build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
    kv_index_resolution_plans: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
    req_to_token_backing_by_req_pool_idx: dict[Any, Any] | None = None,
    read_req_to_token: bool = False,
    max_tokens_per_request: int = 256,
    max_blocks_per_request: int = 4,
    max_total_tokens: int = 512,
    max_preview_entries: int = 8,
) -> list[dict[str, Any]]:
    """Build bounded readonly req_to_token adapter payloads for smoke."""

    if not isinstance(kv_index_resolution_plans, (list, tuple)):
        raise TypeError("kv_index_resolution_plans must be a list or tuple")
    if (
        read_req_to_token
        and req_to_token_backing_by_req_pool_idx is not None
        and not isinstance(req_to_token_backing_by_req_pool_idx, dict)
    ):
        raise TypeError(
            "req_to_token_backing_by_req_pool_idx must be a dict when "
            "read_req_to_token=True"
        )

    total_requested_token_count = 0
    payloads: list[dict[str, Any]] = []

    for plan in kv_index_resolution_plans:
        if not isinstance(plan, dict):
            raise TypeError(
                "RelayKV req_to_token readonly adapter inputs must be dict plans"
            )

        blocking_reasons: list[str] = []
        warning_reasons = [
            "readonly_bounded_req_to_token_adapter",
            "no_token_to_kv_pool_read",
            "preview_only_no_full_entries_logged",
        ]

        request_id = plan.get("request_id")
        req_pool_idx_value = plan.get("req_pool_idx")
        req_pool_idx = _req_to_token_int_like_value_for_smoke(req_pool_idx_value)
        seq_len = plan.get("seq_len")
        layer_id = plan.get("layer_id")
        working_spans = plan.get("relaykv_working_block_spans")
        full_spans = plan.get("full_kv_block_spans")

        if plan.get("event_type") != "relaykv_kv_index_resolution_plan":
            blocking_reasons.append("not_kv_index_resolution_plan")
        if plan.get("resolution_state") != "block_span_resolved":
            blocking_reasons.append("kv_index_resolution_not_block_span_resolved")
        if plan.get("resolution_mode") != "metadata_only":
            blocking_reasons.append("kv_index_resolution_not_metadata_only")
        if read_req_to_token is not True:
            blocking_reasons.append("read_req_to_token_not_enabled")
        if req_pool_idx is None:
            blocking_reasons.append("req_pool_idx_missing_or_invalid")
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            blocking_reasons.append("seq_len_missing_or_invalid")
        if not isinstance(working_spans, (list, tuple)) or not working_spans:
            blocking_reasons.append("invalid_block_span")
        if not isinstance(full_spans, (list, tuple)) or not full_spans:
            blocking_reasons.append("invalid_block_span")
        if plan.get("token_to_kv_pool_read") is True:
            blocking_reasons.append("token_to_kv_pool_read_not_allowed")
        if plan.get("kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if plan.get("tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if plan.get("attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if plan.get("attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if read_req_to_token and req_to_token_backing_by_req_pool_idx is None:
            blocking_reasons.append("req_to_token_backing_missing")

        if blocking_reasons:
            payloads.append(
                _blocked_req_to_token_readonly_adapter_payload_for_smoke(
                    plan,
                    blocking_reasons=list(dict.fromkeys(blocking_reasons)),
                    warning_reasons=warning_reasons,
                )
            )
            continue

        assert isinstance(seq_len, int)
        assert isinstance(working_spans, (list, tuple))
        assert isinstance(full_spans, (list, tuple))
        assert isinstance(req_to_token_backing_by_req_pool_idx, dict)

        req_to_token_backing = _req_to_token_table_for_req_pool_idx_for_smoke(
            req_to_token_backing_by_req_pool_idx,
            req_pool_idx,
        )
        if req_to_token_backing is None:
            blocking_reasons.append("req_to_token_backing_for_req_pool_missing")
        elif not isinstance(req_to_token_backing, (list, tuple)):
            blocking_reasons.append("req_to_token_backing_not_list_or_tuple")

        unique_spans: list[dict[str, int]] = []
        if not blocking_reasons:
            unique_spans, span_blocking_reasons = _readonly_adapter_unique_spans_for_smoke(
                working_spans,
                full_spans,
                seq_len=seq_len,
            )
            blocking_reasons.extend(span_blocking_reasons)

        requested_block_count = len(unique_spans)
        requested_token_count = sum(span["token_count"] for span in unique_spans)

        if requested_block_count > max_blocks_per_request:
            blocking_reasons.append("requested_block_count_exceeds_limit")
        if requested_token_count > max_tokens_per_request:
            blocking_reasons.append("requested_token_count_exceeds_limit")
        if total_requested_token_count + requested_token_count > max_total_tokens:
            blocking_reasons.append("total_requested_token_count_exceeds_limit")

        if blocking_reasons:
            payloads.append(
                _blocked_req_to_token_readonly_adapter_payload_for_smoke(
                    plan,
                    blocking_reasons=list(dict.fromkeys(blocking_reasons)),
                    warning_reasons=warning_reasons,
                )
            )
            continue

        assert isinstance(req_to_token_backing, (list, tuple))
        read_entries, read_blocking_reasons = _read_req_to_token_entries_for_smoke(
            unique_spans,
            req_to_token_backing=req_to_token_backing,
        )
        if read_blocking_reasons:
            payloads.append(
                _blocked_req_to_token_readonly_adapter_payload_for_smoke(
                    plan,
                    blocking_reasons=read_blocking_reasons,
                    warning_reasons=warning_reasons,
                )
            )
            continue

        total_requested_token_count += requested_token_count
        preview_entries = list(read_entries[:max_preview_entries])
        read_token_count = len(read_entries)
        payloads.append(
            {
                "event_type": "relaykv_req_to_token_readonly_adapter_payload",
                "adapter_state": "adapter_payload_ready",
                "adapter_mode": "readonly_bounded_preview",
                "source": (
                    "kv_index_resolution_plan_to_req_to_token_readonly_adapter_payload"
                ),
                "request_id": request_id,
                "req_pool_idx": req_pool_idx_value,
                "seq_len": seq_len,
                "layer_id": layer_id,
                "requested_block_count": requested_block_count,
                "requested_token_count": requested_token_count,
                "read_token_count": read_token_count,
                "preview_entry_count": len(preview_entries),
                "preview_entries": preview_entries,
                "entry_count": read_token_count,
                "entry_min": min(read_entries) if read_entries else None,
                "entry_max": max(read_entries) if read_entries else None,
                "entry_checksum": (
                    sum((index + 1) * entry for index, entry in enumerate(read_entries))
                    % 1000000007
                ),
                "truncated_preview": read_token_count > len(preview_entries),
                "req_to_token_read": True,
                "req_to_token_read_count": read_token_count,
                "token_to_kv_pool_read": False,
                "token_to_kv_pool_read_count": 0,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": [],
                "warning_reasons": warning_reasons,
            }
        )

    return payloads


def summarize_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
    payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize bounded readonly req_to_token adapter payloads for smoke."""

    if not isinstance(payloads, (list, tuple)):
        raise TypeError(
            "RelayKV req_to_token readonly adapter payloads must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_adapter_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    adapter_payload_ready_count = 0
    blocked_count = 0
    error_count = 0
    requested_block_count = 0
    requested_token_count = 0
    read_token_count = 0
    preview_entry_count = 0
    truncated_preview_count = 0

    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(
                "RelayKV req_to_token readonly adapter payload must be a dict"
            )

        state = str(payload.get("adapter_state") or "unknown")
        per_adapter_state[state] += 1
        per_request[str(payload.get("request_id"))] += 1
        per_layer[str(payload.get("layer_id"))] += 1

        if state == "adapter_payload_ready":
            adapter_payload_ready_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = payload.get("requested_block_count")
        if isinstance(value, int) and not isinstance(value, bool):
            requested_block_count += value
        value = payload.get("requested_token_count")
        if isinstance(value, int) and not isinstance(value, bool):
            requested_token_count += value
        value = payload.get("read_token_count")
        if isinstance(value, int) and not isinstance(value, bool):
            read_token_count += value
        value = payload.get("preview_entry_count")
        if isinstance(value, int) and not isinstance(value, bool):
            preview_entry_count += value
        if payload.get("truncated_preview") is True:
            truncated_preview_count += 1

        value = payload.get("req_to_token_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["req_to_token_read_count"] += value
        value = payload.get("token_to_kv_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["token_to_kv_pool_read_count"] += value
        if payload.get("kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if payload.get("kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if payload.get("tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if payload.get("attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if payload.get("attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if payload.get("runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if payload.get("scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if payload.get("kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if payload.get("source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_req_to_token_readonly_adapter_payload_summary",
        "total_req_to_token_adapter_payloads": len(payloads),
        "adapter_payload_ready_count": adapter_payload_ready_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "requested_block_count": requested_block_count,
        "requested_token_count": requested_token_count,
        "read_token_count": read_token_count,
        "preview_entry_count": preview_entry_count,
        "truncated_preview_count": truncated_preview_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_adapter_state_counts": dict(sorted(per_adapter_state.items())),
        "req_to_token_read_count": safety_counts["req_to_token_read_count"],
        "token_to_kv_pool_read_count": (
            safety_counts["token_to_kv_pool_read_count"]
        ),
        "kv_pool_read_count": safety_counts["kv_pool_read_count"],
        "kv_snapshot_count": safety_counts["kv_snapshot_count"],
        "tensor_read_count": safety_counts["tensor_read_count"],
        "attention_comparison_executed_count": (
            safety_counts["attention_comparison_executed_count"]
        ),
        "attention_override_true_count": (
            safety_counts["attention_override_true_count"]
        ),
        "runtime_writeback_true_count": (
            safety_counts["runtime_writeback_true_count"]
        ),
        "scheduler_policy_noop_false_count": (
            safety_counts["scheduler_policy_noop_false_count"]
        ),
        "kv_cache_mutation_true_count": (
            safety_counts["kv_cache_mutation_true_count"]
        ),
        "source_mutated_true_count": safety_counts["source_mutated_true_count"],
    }


def _blocked_actual_req_to_token_pool_adapter_payload_for_smoke(
    plan: dict[str, Any],
    *,
    blocking_reasons: list[str],
    warning_reasons: list[str],
    req_to_token_source: str | None,
    req_to_token_backing_type: str | None,
    req_to_token_shape: Any,
    req_to_token_device: Any,
    req_to_token_dtype: Any,
) -> dict[str, Any]:
    return {
        "event_type": "relaykv_req_to_token_readonly_adapter_payload",
        "adapter_state": "blocked",
        "adapter_mode": "actual_pool_readonly_bounded_preview",
        "source": (
            "kv_index_resolution_plan_to_actual_req_to_token_readonly_adapter_payload"
        ),
        "request_id": plan.get("request_id"),
        "req_pool_idx": plan.get("req_pool_idx"),
        "seq_len": plan.get("seq_len"),
        "layer_id": plan.get("layer_id"),
        "requested_block_count": 0,
        "requested_token_count": 0,
        "read_token_count": 0,
        "preview_entry_count": 0,
        "preview_entries": [],
        "entry_count": 0,
        "entry_min": None,
        "entry_max": None,
        "entry_checksum": None,
        "truncated_preview": False,
        "req_to_token_source": req_to_token_source,
        "req_to_token_backing_type": req_to_token_backing_type,
        "req_to_token_shape": req_to_token_shape,
        "req_to_token_device": req_to_token_device,
        "req_to_token_dtype": req_to_token_dtype,
        "req_to_token_read": False,
        "req_to_token_read_count": 0,
        "actual_req_to_token_pool_read": False,
        "actual_req_to_token_pool_read_count": 0,
        "token_to_kv_pool_read": False,
        "token_to_kv_pool_read_count": 0,
        "kv_pool_read": False,
        "kv_snapshot": False,
        "tensor_read": False,
        "attention_comparison_executed": False,
        "attention_override": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
    }


def _safe_req_to_token_backing_metadata_for_smoke(
    req_to_token_backing: Any,
) -> tuple[str | None, Any, Any, Any]:
    if req_to_token_backing is None:
        return None, None, None, None

    backing_type = type(req_to_token_backing).__name__
    shape = None
    device = None
    dtype = None

    if isinstance(req_to_token_backing, (dict, list, tuple)):
        return backing_type, shape, device, dtype

    device = getattr(req_to_token_backing, "device", None)
    shape = getattr(req_to_token_backing, "shape", None)
    dtype = getattr(req_to_token_backing, "dtype", None)
    return backing_type, shape, device, dtype


def _actual_req_to_token_backing_row_for_smoke(
    req_to_token_backing: Any,
    *,
    req_pool_idx: int,
) -> Any:
    if isinstance(req_to_token_backing, dict):
        if req_pool_idx in req_to_token_backing:
            return req_to_token_backing[req_pool_idx]
        req_pool_idx_as_str = str(req_pool_idx)
        if req_pool_idx_as_str in req_to_token_backing:
            return req_to_token_backing[req_pool_idx_as_str]
        return None

    if isinstance(req_to_token_backing, (list, tuple)):
        if req_pool_idx < 0 or req_pool_idx >= len(req_to_token_backing):
            return None
        return req_to_token_backing[req_pool_idx]

    return None


def build_relaykv_actual_req_to_token_pool_adapter_payloads_for_smoke(
    kv_index_resolution_plans: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
    req_to_token_pool: Any = None,
    read_actual_req_to_token_pool: bool = False,
    max_tokens_per_request: int = 256,
    max_blocks_per_request: int = 4,
    max_total_tokens: int = 512,
    max_preview_entries: int = 8,
    allow_cpu_tensor_read: bool = False,
    allow_gpu_tensor_read: bool = False,
) -> list[dict[str, Any]]:
    """Build bounded readonly adapter payloads from a fake actual req_to_token pool."""

    if not isinstance(kv_index_resolution_plans, (list, tuple)):
        raise TypeError("kv_index_resolution_plans must be a list or tuple")

    total_requested_token_count = 0
    payloads: list[dict[str, Any]] = []

    req_to_token_source: str | None = None
    req_to_token_backing: Any = None
    req_to_token_backing_type: str | None = None
    req_to_token_shape: Any = None
    req_to_token_device: Any = None
    req_to_token_dtype: Any = None
    pool_blocking_reasons: list[str] = []

    if read_actual_req_to_token_pool is not True:
        pool_blocking_reasons.append("actual_req_to_token_pool_read_not_enabled")
    elif req_to_token_pool is None:
        pool_blocking_reasons.append("req_to_token_pool_missing")
    else:
        req_to_token_backing = getattr(req_to_token_pool, "req_to_token", None)
        if req_to_token_backing is None:
            pool_blocking_reasons.append("req_to_token_attr_missing")
        else:
            req_to_token_source = "actual_pool_attr"
            (
                req_to_token_backing_type,
                req_to_token_shape,
                req_to_token_device,
                req_to_token_dtype,
            ) = _safe_req_to_token_backing_metadata_for_smoke(req_to_token_backing)
            if isinstance(req_to_token_backing, (dict, list, tuple)):
                pass
            else:
                device_string = str(req_to_token_device).lower()
                if "cuda" in device_string or "gpu" in device_string:
                    pool_blocking_reasons.append("req_to_token_tensor_device_not_allowed")
                    if allow_gpu_tensor_read is not True:
                        pool_blocking_reasons.append("gpu_tensor_read_not_allowed")
                else:
                    if allow_cpu_tensor_read is not True:
                        pool_blocking_reasons.append("cpu_tensor_read_not_allowed")
                pool_blocking_reasons.append("req_to_token_backing_not_supported")

    for plan in kv_index_resolution_plans:
        if not isinstance(plan, dict):
            raise TypeError(
                "RelayKV actual req_to_token adapter inputs must be dict plans"
            )

        blocking_reasons = list(pool_blocking_reasons)
        warning_reasons = [
            "actual_req_to_token_pool_readonly_adapter",
            "bounded_preview_only",
            "no_token_to_kv_pool_read",
            "preview_only_no_full_entries_logged",
        ]

        request_id = plan.get("request_id")
        req_pool_idx_value = plan.get("req_pool_idx")
        req_pool_idx = _req_to_token_int_like_value_for_smoke(req_pool_idx_value)
        seq_len = plan.get("seq_len")
        layer_id = plan.get("layer_id")
        working_spans = plan.get("relaykv_working_block_spans")
        full_spans = plan.get("full_kv_block_spans")

        if plan.get("event_type") != "relaykv_kv_index_resolution_plan":
            blocking_reasons.append("not_kv_index_resolution_plan")
        if plan.get("resolution_state") != "block_span_resolved":
            blocking_reasons.append("kv_index_resolution_not_block_span_resolved")
        if plan.get("resolution_mode") != "metadata_only":
            blocking_reasons.append("kv_index_resolution_not_metadata_only")
        if req_pool_idx is None:
            blocking_reasons.append("req_pool_idx_missing_or_invalid")
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            blocking_reasons.append("seq_len_missing_or_invalid")
        if not isinstance(working_spans, (list, tuple)) or not working_spans:
            blocking_reasons.append("invalid_block_span")
        if not isinstance(full_spans, (list, tuple)) or not full_spans:
            blocking_reasons.append("invalid_block_span")
        if plan.get("token_to_kv_pool_read") is True:
            blocking_reasons.append("token_to_kv_pool_read_not_allowed")
        if plan.get("kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if plan.get("tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if plan.get("attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if plan.get("attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")

        if blocking_reasons:
            payloads.append(
                _blocked_actual_req_to_token_pool_adapter_payload_for_smoke(
                    plan,
                    blocking_reasons=list(dict.fromkeys(blocking_reasons)),
                    warning_reasons=warning_reasons,
                    req_to_token_source=req_to_token_source,
                    req_to_token_backing_type=req_to_token_backing_type,
                    req_to_token_shape=req_to_token_shape,
                    req_to_token_device=req_to_token_device,
                    req_to_token_dtype=req_to_token_dtype,
                )
            )
            continue

        assert isinstance(seq_len, int)
        assert isinstance(req_pool_idx, int)
        assert isinstance(working_spans, (list, tuple))
        assert isinstance(full_spans, (list, tuple))

        req_to_token_backing_row = _actual_req_to_token_backing_row_for_smoke(
            req_to_token_backing,
            req_pool_idx=req_pool_idx,
        )
        if req_to_token_backing_row is None:
            blocking_reasons.append("req_to_token_backing_for_req_pool_missing")
        elif not isinstance(req_to_token_backing_row, (list, tuple)):
            blocking_reasons.append("req_to_token_backing_not_list_or_tuple")

        unique_spans: list[dict[str, int]] = []
        if not blocking_reasons:
            unique_spans, span_blocking_reasons = _readonly_adapter_unique_spans_for_smoke(
                working_spans,
                full_spans,
                seq_len=seq_len,
            )
            blocking_reasons.extend(span_blocking_reasons)

        requested_block_count = len(unique_spans)
        requested_token_count = sum(span["token_count"] for span in unique_spans)

        if requested_block_count > max_blocks_per_request:
            blocking_reasons.append("requested_block_count_exceeds_limit")
        if requested_token_count > max_tokens_per_request:
            blocking_reasons.append("requested_token_count_exceeds_limit")
        if total_requested_token_count + requested_token_count > max_total_tokens:
            blocking_reasons.append("total_requested_token_count_exceeds_limit")

        if blocking_reasons:
            payloads.append(
                _blocked_actual_req_to_token_pool_adapter_payload_for_smoke(
                    plan,
                    blocking_reasons=list(dict.fromkeys(blocking_reasons)),
                    warning_reasons=warning_reasons,
                    req_to_token_source=req_to_token_source,
                    req_to_token_backing_type=req_to_token_backing_type,
                    req_to_token_shape=req_to_token_shape,
                    req_to_token_device=req_to_token_device,
                    req_to_token_dtype=req_to_token_dtype,
                )
            )
            continue

        assert isinstance(req_to_token_backing_row, (list, tuple))
        read_entries, read_blocking_reasons = _read_req_to_token_entries_for_smoke(
            unique_spans,
            req_to_token_backing=req_to_token_backing_row,
        )
        if read_blocking_reasons:
            payloads.append(
                _blocked_actual_req_to_token_pool_adapter_payload_for_smoke(
                    plan,
                    blocking_reasons=read_blocking_reasons,
                    warning_reasons=warning_reasons,
                    req_to_token_source=req_to_token_source,
                    req_to_token_backing_type=req_to_token_backing_type,
                    req_to_token_shape=req_to_token_shape,
                    req_to_token_device=req_to_token_device,
                    req_to_token_dtype=req_to_token_dtype,
                )
            )
            continue

        total_requested_token_count += requested_token_count
        preview_entries = list(read_entries[:max_preview_entries])
        read_token_count = len(read_entries)
        payloads.append(
            {
                "event_type": "relaykv_req_to_token_readonly_adapter_payload",
                "adapter_state": "adapter_payload_ready",
                "adapter_mode": "actual_pool_readonly_bounded_preview",
                "source": (
                    "kv_index_resolution_plan_to_actual_req_to_token_readonly_adapter_payload"
                ),
                "request_id": request_id,
                "req_pool_idx": req_pool_idx_value,
                "seq_len": seq_len,
                "layer_id": layer_id,
                "requested_block_count": requested_block_count,
                "requested_token_count": requested_token_count,
                "read_token_count": read_token_count,
                "preview_entry_count": len(preview_entries),
                "preview_entries": preview_entries,
                "entry_count": read_token_count,
                "entry_min": min(read_entries) if read_entries else None,
                "entry_max": max(read_entries) if read_entries else None,
                "entry_checksum": (
                    sum((index + 1) * entry for index, entry in enumerate(read_entries))
                    % 1000000007
                ),
                "truncated_preview": read_token_count > len(preview_entries),
                "req_to_token_source": req_to_token_source,
                "req_to_token_backing_type": req_to_token_backing_type,
                "req_to_token_shape": req_to_token_shape,
                "req_to_token_device": req_to_token_device,
                "req_to_token_dtype": req_to_token_dtype,
                "req_to_token_read": True,
                "req_to_token_read_count": read_token_count,
                "actual_req_to_token_pool_read": True,
                "actual_req_to_token_pool_read_count": read_token_count,
                "token_to_kv_pool_read": False,
                "token_to_kv_pool_read_count": 0,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": [],
                "warning_reasons": warning_reasons,
            }
        )

    return payloads


def summarize_relaykv_actual_req_to_token_pool_adapter_payloads_for_smoke(
    payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize fake actual req_to_token pool adapter payloads for smoke."""

    if not isinstance(payloads, (list, tuple)):
        raise TypeError(
            "RelayKV actual req_to_token adapter payloads must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_adapter_state: Counter[str] = Counter()
    per_adapter_mode: Counter[str] = Counter()
    per_req_to_token_source: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "actual_req_to_token_pool_read_true_count": 0,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    adapter_payload_ready_count = 0
    blocked_count = 0
    error_count = 0
    requested_block_count = 0
    requested_token_count = 0
    read_token_count = 0
    preview_entry_count = 0
    truncated_preview_count = 0

    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(
                "RelayKV actual req_to_token adapter payload must be a dict"
            )

        state = str(payload.get("adapter_state") or "unknown")
        mode = str(payload.get("adapter_mode") or "unknown")
        source = str(payload.get("req_to_token_source") or "unknown")
        per_adapter_state[state] += 1
        per_adapter_mode[mode] += 1
        per_req_to_token_source[source] += 1
        per_request[str(payload.get("request_id"))] += 1
        per_layer[str(payload.get("layer_id"))] += 1

        if state == "adapter_payload_ready":
            adapter_payload_ready_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = payload.get("requested_block_count")
        if isinstance(value, int) and not isinstance(value, bool):
            requested_block_count += value
        value = payload.get("requested_token_count")
        if isinstance(value, int) and not isinstance(value, bool):
            requested_token_count += value
        value = payload.get("read_token_count")
        if isinstance(value, int) and not isinstance(value, bool):
            read_token_count += value
        value = payload.get("preview_entry_count")
        if isinstance(value, int) and not isinstance(value, bool):
            preview_entry_count += value
        if payload.get("truncated_preview") is True:
            truncated_preview_count += 1

        value = payload.get("req_to_token_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["req_to_token_read_count"] += value
        value = payload.get("actual_req_to_token_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["actual_req_to_token_pool_read_count"] += value
        if payload.get("actual_req_to_token_pool_read") is True:
            safety_counts["actual_req_to_token_pool_read_true_count"] += 1
        value = payload.get("token_to_kv_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["token_to_kv_pool_read_count"] += value
        if payload.get("kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if payload.get("kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if payload.get("tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if payload.get("attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if payload.get("attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if payload.get("runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if payload.get("scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if payload.get("kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if payload.get("source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_actual_req_to_token_pool_adapter_payload_summary",
        "total_actual_req_to_token_adapter_payloads": len(payloads),
        "adapter_payload_ready_count": adapter_payload_ready_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "requested_block_count": requested_block_count,
        "requested_token_count": requested_token_count,
        "read_token_count": read_token_count,
        "preview_entry_count": preview_entry_count,
        "truncated_preview_count": truncated_preview_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_adapter_state_counts": dict(sorted(per_adapter_state.items())),
        "per_adapter_mode_counts": dict(sorted(per_adapter_mode.items())),
        "per_req_to_token_source_counts": dict(
            sorted(per_req_to_token_source.items())
        ),
        "req_to_token_read_count": safety_counts["req_to_token_read_count"],
        "actual_req_to_token_pool_read_count": (
            safety_counts["actual_req_to_token_pool_read_count"]
        ),
        "actual_req_to_token_pool_read_true_count": (
            safety_counts["actual_req_to_token_pool_read_true_count"]
        ),
        "token_to_kv_pool_read_count": (
            safety_counts["token_to_kv_pool_read_count"]
        ),
        "kv_pool_read_count": safety_counts["kv_pool_read_count"],
        "kv_snapshot_count": safety_counts["kv_snapshot_count"],
        "tensor_read_count": safety_counts["tensor_read_count"],
        "attention_comparison_executed_count": (
            safety_counts["attention_comparison_executed_count"]
        ),
        "attention_override_true_count": (
            safety_counts["attention_override_true_count"]
        ),
        "runtime_writeback_true_count": (
            safety_counts["runtime_writeback_true_count"]
        ),
        "scheduler_policy_noop_false_count": (
            safety_counts["scheduler_policy_noop_false_count"]
        ),
        "kv_cache_mutation_true_count": (
            safety_counts["kv_cache_mutation_true_count"]
        ),
        "source_mutated_true_count": safety_counts["source_mutated_true_count"],
    }


def _physical_kv_index_blocked_result_for_smoke(
    result: Mapping[str, Any],
    *,
    blocking_reasons: list[str],
    warning_reasons: list[str],
) -> dict[str, Any]:
    blocked_result = normalize_relaykv_sglang_adapter_schema_for_smoke(result)
    existing_engine_block_ref = blocked_result.get("engine_block_ref")
    if isinstance(existing_engine_block_ref, Mapping):
        engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
            dict(existing_engine_block_ref)
        )
    else:
        engine_block_ref = {}

    engine_block_ref["cache_position"] = engine_block_ref.get("cache_position")
    engine_block_ref["token_to_kv_pool_index"] = None
    engine_block_ref["physical_kv_index_preview"] = []
    engine_block_ref["physical_kv_index_count"] = 0
    engine_block_ref["physical_kv_index_checksum"] = None

    blocked_result.update(
        {
            "event_type": "relaykv_physical_kv_index_resolution_result",
            "resolution_state": "blocked",
            "resolution_mode": "readonly_synthetic_table",
            "source": (
                "req_to_token_resolution_result_to_"
                "physical_kv_index_resolution_result"
            ),
            "resolved_block_count": 0,
            "resolved_token_count": 0,
            "req_to_token_entry_count": 0,
            "physical_kv_index_count": 0,
            "physical_kv_index_preview_count": 0,
            "physical_kv_index_checksum": None,
            "truncated_physical_kv_index_preview": False,
            "req_to_token_read": False,
            "req_to_token_read_count": 0,
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read": False,
            "kv_snapshot": False,
            "tensor_read": False,
            "attention_comparison_executed": False,
            "attention_override": False,
            "runtime_writeback": False,
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "source_mutated": False,
            "engine_block_ref": engine_block_ref,
            "blocking_reasons": blocking_reasons,
            "warning_reasons": warning_reasons,
        }
    )
    blocked_result["decision_state"] = blocked_result.get("decision_state") or "blocked"
    blocked_result["fallback_reason"] = (
        blocked_result.get("fallback_reason")
        if blocked_result.get("fallback_reason") is not None
        else (blocking_reasons[0] if blocking_reasons else None)
    )
    return blocked_result


def _token_to_kv_pool_entry_for_smoke(
    token_to_kv_pool_table: Any,
    req_to_token_entry: int,
) -> Any:
    if isinstance(token_to_kv_pool_table, dict):
        if req_to_token_entry in token_to_kv_pool_table:
            return token_to_kv_pool_table[req_to_token_entry]
        req_to_token_entry_as_str = str(req_to_token_entry)
        if req_to_token_entry_as_str in token_to_kv_pool_table:
            return token_to_kv_pool_table[req_to_token_entry_as_str]
        return None

    if isinstance(token_to_kv_pool_table, (list, tuple)):
        if req_to_token_entry < 0 or req_to_token_entry >= len(token_to_kv_pool_table):
            return None
        return token_to_kv_pool_table[req_to_token_entry]

    try:
        physical_kv_index = token_to_kv_pool_table[req_to_token_entry]
    except Exception:
        req_to_token_entry_as_str = str(req_to_token_entry)
        try:
            physical_kv_index = token_to_kv_pool_table[req_to_token_entry_as_str]
        except Exception:
            raise
    return physical_kv_index


def _token_to_kv_pool_object_type_for_smoke(token_to_kv_pool_object: Any) -> str | None:
    if token_to_kv_pool_object is None:
        return None
    return type(token_to_kv_pool_object).__name__


def _token_to_kv_pool_object_shape_for_smoke(
    token_to_kv_pool_object: Any,
) -> list[int] | None:
    if token_to_kv_pool_object is None:
        return None
    shape = getattr(token_to_kv_pool_object, "shape", None)
    if isinstance(shape, (list, tuple)):
        dims: list[int] = []
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, int):
                return None
            dims.append(dim)
        return dims
    return None


def _token_to_kv_pool_object_is_indexable_for_smoke(
    token_to_kv_pool_object: Any,
) -> bool:
    if isinstance(token_to_kv_pool_object, (dict, list, tuple)):
        return True
    if token_to_kv_pool_object is None:
        return False
    getitem = getattr(token_to_kv_pool_object, "__getitem__", None)
    return callable(getitem)


def _token_to_kv_pool_entry_from_live_object_for_smoke(
    token_to_kv_pool_object: Any,
    req_to_token_entry: int,
) -> tuple[Any, str | None]:
    try:
        physical_kv_index = _token_to_kv_pool_entry_for_smoke(
            token_to_kv_pool_object,
            req_to_token_entry,
        )
    except Exception:
        return None, "token_to_kv_pool_index_read_failed"
    return physical_kv_index, None


def _read_live_physical_kv_indexes_for_smoke(
    req_to_token_spans: list[Mapping[str, Any]],
    *,
    token_to_kv_pool_object: Any,
) -> tuple[list[int], list[str]]:
    physical_kv_indexes: list[int] = []
    blocking_reasons: list[str] = []

    for span in req_to_token_spans:
        req_to_token_entries = _event_value(span, "req_to_token_entries")
        if not isinstance(req_to_token_entries, (list, tuple)):
            blocking_reasons.append("req_to_token_entries_missing")
            return [], list(dict.fromkeys(blocking_reasons))
        for entry in req_to_token_entries:
            if isinstance(entry, bool) or not isinstance(entry, int):
                blocking_reasons.append("req_to_token_entry_not_int")
                return [], list(dict.fromkeys(blocking_reasons))
            physical_kv_index, read_error = _token_to_kv_pool_entry_from_live_object_for_smoke(
                token_to_kv_pool_object,
                entry,
            )
            if read_error is not None:
                blocking_reasons.append(read_error)
                return [], list(dict.fromkeys(blocking_reasons))
            if physical_kv_index is None:
                blocking_reasons.append("token_to_kv_pool_entry_missing")
                return [], list(dict.fromkeys(blocking_reasons))
            if isinstance(physical_kv_index, bool) or not isinstance(
                physical_kv_index, int
            ):
                blocking_reasons.append("token_to_kv_pool_entry_not_int")
                return [], list(dict.fromkeys(blocking_reasons))
            physical_kv_indexes.append(physical_kv_index)

    return physical_kv_indexes, []


def _read_physical_kv_indexes_for_smoke(
    req_to_token_spans: list[Mapping[str, Any]],
    *,
    token_to_kv_pool_table: Any,
) -> tuple[list[int], list[str]]:
    physical_kv_indexes: list[int] = []
    blocking_reasons: list[str] = []

    for span in req_to_token_spans:
        req_to_token_entries = _event_value(span, "req_to_token_entries")
        if not isinstance(req_to_token_entries, (list, tuple)):
            blocking_reasons.append("req_to_token_entries_missing")
            return [], list(dict.fromkeys(blocking_reasons))
        for entry in req_to_token_entries:
            if isinstance(entry, bool) or not isinstance(entry, int):
                blocking_reasons.append("req_to_token_entry_not_int")
                return [], list(dict.fromkeys(blocking_reasons))
            physical_kv_index = _token_to_kv_pool_entry_for_smoke(
                token_to_kv_pool_table,
                entry,
            )
            if physical_kv_index is None:
                blocking_reasons.append("token_to_kv_pool_entry_missing")
                return [], list(dict.fromkeys(blocking_reasons))
            if isinstance(physical_kv_index, bool) or not isinstance(
                physical_kv_index, int
            ):
                blocking_reasons.append("token_to_kv_pool_entry_not_int")
                return [], list(dict.fromkeys(blocking_reasons))
            physical_kv_indexes.append(physical_kv_index)

    return physical_kv_indexes, []


def _blocked_live_token_to_kv_pool_index_read_result_for_smoke(
    result: Mapping[str, Any],
    *,
    blocking_reasons: list[str],
    warning_reasons: list[str],
    source_path: str | None,
    token_to_kv_pool_type: str | None,
    token_to_kv_pool_shape: list[int] | None,
    read_token_to_kv_pool_index: bool,
    max_tokens_per_request: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    blocked_result = normalize_relaykv_sglang_adapter_schema_for_smoke(result)
    adapter_metadata = blocked_result.get("adapter_metadata")
    if isinstance(adapter_metadata, Mapping):
        payload_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
            dict(adapter_metadata)
        )
    else:
        payload_adapter_metadata = {}
    payload_adapter_metadata.update(
        {
            "token_to_kv_pool_source_path": source_path,
            "token_to_kv_pool_type": token_to_kv_pool_type,
            "token_to_kv_pool_shape": token_to_kv_pool_shape,
            "live_index_read_enabled": read_token_to_kv_pool_index,
            "max_tokens_per_request": max_tokens_per_request,
            "max_total_tokens": max_total_tokens,
            "truncated_preview": False,
        }
    )

    existing_engine_block_ref = blocked_result.get("engine_block_ref")
    if isinstance(existing_engine_block_ref, Mapping):
        engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
            dict(existing_engine_block_ref)
        )
    else:
        engine_block_ref = {}
    engine_block_ref["token_to_kv_pool_index"] = None
    engine_block_ref["cache_position"] = None
    engine_block_ref["physical_kv_index_preview"] = []
    engine_block_ref["physical_kv_index_count"] = 0
    engine_block_ref["physical_kv_index_checksum"] = None

    blocked_result.update(
        {
            "event_type": "relaykv_live_token_to_kv_pool_index_read_result",
            "resolution_state": "blocked",
            "adapter_mode": "live_token_to_kv_pool_bounded_index_read",
            "source": (
                "req_to_token_resolution_result_to_"
                "live_token_to_kv_pool_index_read_result"
            ),
            "adapter_metadata": payload_adapter_metadata,
            "engine_block_ref": engine_block_ref,
            "requested_token_count": 0,
            "read_token_count": 0,
            "physical_kv_index_count": 0,
            "physical_kv_index_preview_count": 0,
            "physical_kv_index_checksum": None,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
            "req_to_token_read": False,
            "actual_req_to_token_pool_read": False,
            "token_to_kv_pool_read": False,
            "actual_token_to_kv_pool_read": False,
            "live_token_to_kv_pool_index_read": False,
            "kv_pool_read": False,
            "kv_snapshot": False,
            "tensor_read": False,
            "attention_comparison_executed": False,
            "attention_override": False,
            "runtime_writeback": False,
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "source_mutated": False,
            "blocking_reasons": blocking_reasons,
            "warning_reasons": warning_reasons,
        }
    )
    blocked_result["decision_state"] = "SHADOW_ONLY"
    blocked_result["fallback_reason"] = (
        blocked_result.get("fallback_reason")
        if blocked_result.get("fallback_reason") is not None
        else (blocking_reasons[0] if blocking_reasons else None)
    )
    return blocked_result


def build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
    req_to_token_resolution_results: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
    token_to_kv_pool_object: Any = None,
    read_token_to_kv_pool_index: bool = False,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
    source_path: str | None = None,
    max_preview_entries: int = 8,
) -> list[dict[str, Any]]:
    """Build guarded live-like bounded token_to_kv_pool index read results."""

    if not isinstance(req_to_token_resolution_results, (list, tuple)):
        raise TypeError("req_to_token_resolution_results must be a list or tuple")

    token_to_kv_pool_type = _token_to_kv_pool_object_type_for_smoke(
        token_to_kv_pool_object
    )
    token_to_kv_pool_shape = _token_to_kv_pool_object_shape_for_smoke(
        token_to_kv_pool_object
    )
    total_requested_token_count = 0
    results: list[dict[str, Any]] = []

    for result in req_to_token_resolution_results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV live token_to_kv_pool index read inputs must be dict results"
            )

        blocking_reasons: list[str] = []
        warning_reasons = [
            "guarded_live_token_to_kv_pool_index_read",
            "bounded_index_lookup_only",
            "no_kv_pool_read",
            "preview_only_no_full_indices_logged",
        ]

        full_kv_req_to_token_spans = _event_value(result, "full_kv_req_to_token_spans")
        if _event_value(result, "event_type") != "relaykv_req_to_token_resolution_result":
            blocking_reasons.append("not_req_to_token_resolution_result")
        if _event_value(result, "resolution_state") != "req_to_token_resolved":
            blocking_reasons.append("req_to_token_resolution_not_resolved")
        if read_token_to_kv_pool_index is not True:
            blocking_reasons.append("live_index_read_not_enabled")
        if read_token_to_kv_pool_index and token_to_kv_pool_object is None:
            blocking_reasons.append("token_to_kv_pool_object_missing")
        elif read_token_to_kv_pool_index and not _token_to_kv_pool_object_is_indexable_for_smoke(
            token_to_kv_pool_object
        ):
            blocking_reasons.append("token_to_kv_pool_object_not_indexable")
        if not isinstance(full_kv_req_to_token_spans, (list, tuple)):
            blocking_reasons.append("req_to_token_entries_missing")
        elif not full_kv_req_to_token_spans:
            blocking_reasons.append("req_to_token_entries_missing")
        if _event_value(result, "kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if _event_value(result, "kv_snapshot") is True:
            blocking_reasons.append("kv_snapshot_not_allowed")
        if _event_value(result, "tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if _event_value(result, "attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if _event_value(result, "attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if _event_value(result, "runtime_writeback") is True:
            blocking_reasons.append("runtime_writeback_not_allowed")
        if _event_value(result, "scheduler_policy_noop") is False:
            blocking_reasons.append("scheduler_mutation_not_allowed")
        if _event_value(result, "kv_cache_mutation") is True:
            blocking_reasons.append("kv_cache_mutation_not_allowed")
        if _event_value(result, "source_mutated") is True:
            blocking_reasons.append("source_mutation_not_allowed")

        requested_token_count = 0
        if not blocking_reasons:
            assert isinstance(full_kv_req_to_token_spans, (list, tuple))
            for span in full_kv_req_to_token_spans:
                if not isinstance(span, Mapping):
                    blocking_reasons.append("req_to_token_entries_missing")
                    break
                req_to_token_entries = _event_value(span, "req_to_token_entries")
                if not isinstance(req_to_token_entries, (list, tuple)):
                    blocking_reasons.append("req_to_token_entries_missing")
                    break
                requested_token_count += len(req_to_token_entries)

        if requested_token_count > max_tokens_per_request:
            blocking_reasons.append("max_tokens_per_request_exceeded")
        if total_requested_token_count + requested_token_count > max_total_tokens:
            blocking_reasons.append("max_total_tokens_exceeded")

        blocking_reasons = list(dict.fromkeys(blocking_reasons))
        if blocking_reasons:
            results.append(
                _blocked_live_token_to_kv_pool_index_read_result_for_smoke(
                    result,
                    blocking_reasons=blocking_reasons,
                    warning_reasons=warning_reasons,
                    source_path=source_path,
                    token_to_kv_pool_type=token_to_kv_pool_type,
                    token_to_kv_pool_shape=token_to_kv_pool_shape,
                    read_token_to_kv_pool_index=read_token_to_kv_pool_index,
                    max_tokens_per_request=max_tokens_per_request,
                    max_total_tokens=max_total_tokens,
                )
            )
            continue

        assert isinstance(full_kv_req_to_token_spans, (list, tuple))
        physical_kv_indexes, read_blocking_reasons = _read_live_physical_kv_indexes_for_smoke(
            list(full_kv_req_to_token_spans),
            token_to_kv_pool_object=token_to_kv_pool_object,
        )
        if read_blocking_reasons:
            results.append(
                _blocked_live_token_to_kv_pool_index_read_result_for_smoke(
                    result,
                    blocking_reasons=read_blocking_reasons,
                    warning_reasons=warning_reasons,
                    source_path=source_path,
                    token_to_kv_pool_type=token_to_kv_pool_type,
                    token_to_kv_pool_shape=token_to_kv_pool_shape,
                    read_token_to_kv_pool_index=read_token_to_kv_pool_index,
                    max_tokens_per_request=max_tokens_per_request,
                    max_total_tokens=max_total_tokens,
                )
            )
            continue

        total_requested_token_count += requested_token_count
        preview_entries = list(physical_kv_indexes[:max_preview_entries])
        checksum = (
            sum((index + 1) * entry for index, entry in enumerate(physical_kv_indexes))
            % 1000000007
        )
        normalized_result = normalize_relaykv_sglang_adapter_schema_for_smoke(result)
        adapter_metadata = normalized_result.get("adapter_metadata")
        if isinstance(adapter_metadata, Mapping):
            payload_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
                dict(adapter_metadata)
            )
        else:
            payload_adapter_metadata = {}
        payload_adapter_metadata.update(
            {
                "token_to_kv_pool_source_path": source_path,
                "token_to_kv_pool_type": token_to_kv_pool_type,
                "token_to_kv_pool_shape": token_to_kv_pool_shape,
                "live_index_read_enabled": read_token_to_kv_pool_index,
                "max_tokens_per_request": max_tokens_per_request,
                "max_total_tokens": max_total_tokens,
                "truncated_preview": len(physical_kv_indexes) > len(preview_entries),
            }
        )

        existing_engine_block_ref = normalized_result.get("engine_block_ref")
        if isinstance(existing_engine_block_ref, Mapping):
            engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
                dict(existing_engine_block_ref)
            )
        else:
            engine_block_ref = {}
        engine_block_ref["token_to_kv_pool_index"] = None
        engine_block_ref["cache_position"] = None
        engine_block_ref["physical_kv_index_preview"] = preview_entries
        engine_block_ref["physical_kv_index_count"] = len(physical_kv_indexes)
        engine_block_ref["physical_kv_index_checksum"] = checksum

        normalized_result.update(
            {
                "event_type": "relaykv_live_token_to_kv_pool_index_read_result",
                "resolution_state": "physical_kv_index_resolved",
                "adapter_mode": "live_token_to_kv_pool_bounded_index_read",
                "source": (
                    "req_to_token_resolution_result_to_"
                    "live_token_to_kv_pool_index_read_result"
                ),
                "adapter_metadata": payload_adapter_metadata,
                "engine_block_ref": engine_block_ref,
                "requested_token_count": requested_token_count,
                "read_token_count": len(physical_kv_indexes),
                "physical_kv_index_count": len(physical_kv_indexes),
                "physical_kv_index_preview_count": len(preview_entries),
                "physical_kv_index_checksum": checksum,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": len(physical_kv_indexes),
                "actual_token_to_kv_pool_read_count": len(physical_kv_indexes),
                "live_token_to_kv_pool_index_read_count": len(physical_kv_indexes),
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
                "req_to_token_read": False,
                "actual_req_to_token_pool_read": False,
                "token_to_kv_pool_read": True,
                "actual_token_to_kv_pool_read": True,
                "live_token_to_kv_pool_index_read": True,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": [],
                "warning_reasons": warning_reasons,
            }
        )
        normalized_result["decision_state"] = "SHADOW_ONLY"
        results.append(normalized_result)

    return results


def summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize guarded live-like bounded token_to_kv_pool index read results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV live token_to_kv_pool index read results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_resolution_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    resolved_count = 0
    blocked_count = 0
    error_count = 0
    requested_token_count = 0
    read_token_count = 0
    physical_kv_index_count = 0
    physical_kv_index_preview_count = 0
    truncated_preview_count = 0

    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV live token_to_kv_pool index read result must be a dict"
            )

        state = str(_event_value(result, "resolution_state") or "unknown")
        per_resolution_state[state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1

        if state == "physical_kv_index_resolved":
            resolved_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        for key in (
            "requested_token_count",
            "read_token_count",
            "physical_kv_index_count",
            "physical_kv_index_preview_count",
        ):
            value = _event_value(result, key)
            if isinstance(value, int) and not isinstance(value, bool):
                if key == "requested_token_count":
                    requested_token_count += value
                elif key == "read_token_count":
                    read_token_count += value
                elif key == "physical_kv_index_count":
                    physical_kv_index_count += value
                else:
                    physical_kv_index_preview_count += value

        adapter_metadata = _event_value(result, "adapter_metadata")
        if isinstance(adapter_metadata, Mapping) and adapter_metadata.get(
            "truncated_preview"
        ) is True:
            truncated_preview_count += 1

        for key in safety_counts:
            value = _event_value(result, key)
            if isinstance(value, int) and not isinstance(value, bool):
                safety_counts[key] += value

    return {
        "summary_type": "relaykv_live_token_to_kv_pool_index_read_result_summary",
        "total_live_token_to_kv_pool_index_read_results": len(results),
        "physical_kv_index_resolved_count": resolved_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "requested_token_count": requested_token_count,
        "read_token_count": read_token_count,
        "physical_kv_index_count": physical_kv_index_count,
        "physical_kv_index_preview_count": physical_kv_index_preview_count,
        "truncated_preview_count": truncated_preview_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_resolution_state_counts": dict(sorted(per_resolution_state.items())),
        **{key: safety_counts[key] for key in sorted(safety_counts)},
    }


def _relaykv_runtime_req_to_token_source_payloads_for_smoke(
    runtime_observation_payloads: Any,
) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    if runtime_observation_payloads is None:
        return None, None
    if not isinstance(runtime_observation_payloads, (list, tuple)):
        return [], "runtime_observation_payloads"
    normalized_payloads: list[Mapping[str, Any]] = []
    for payload in runtime_observation_payloads:
        if not isinstance(payload, Mapping):
            return [], "runtime_observation_payloads"
        normalized_payloads.append(payload)
    return normalized_payloads, "runtime_observation_payloads"


def _relaykv_runtime_req_to_token_entries_for_smoke(
    explicit_req_to_token_entries: Any,
) -> tuple[list[int] | None, str | None]:
    if explicit_req_to_token_entries is None:
        return None, "explicit_req_to_token_entries_missing"
    if not isinstance(explicit_req_to_token_entries, (list, tuple)):
        return None, "explicit_req_to_token_entries_not_list"

    normalized_entries: list[int] = []
    for entry in explicit_req_to_token_entries:
        if isinstance(entry, bool) or not isinstance(entry, int):
            return None, "explicit_req_to_token_entry_not_int"
        normalized_entries.append(entry)
    if not normalized_entries:
        return None, "explicit_req_to_token_entries_missing"
    return normalized_entries, None


def build_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
    *,
    runtime_observation_payloads: Any = None,
    kv_index_resolution_plans: Any = None,
    production_enabled: bool = False,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """Build smoke-only synthetic req_to_token entries from safe runtime metadata."""

    results: list[dict[str, Any]] = []
    if production_enabled is not True:
        return [
            {
                "event_type": (
                    "relaykv_runtime_metadata_derived_req_to_token_entries_result"
                ),
                "derivation_state": "blocked",
                "derivation_mode": "runtime_metadata_only",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "token_span": None,
                "layer_id": None,
                "kv_head_group": None,
                "derived_req_to_token_entries": [],
                "derived_entry_count": 0,
                "blocked_reason": "runtime_metadata_derivation_not_enabled",
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    payloads, source_path = _relaykv_runtime_req_to_token_source_payloads_for_smoke(
        runtime_observation_payloads
    )
    normalized_plans: list[Mapping[str, Any] | None] = []
    if kv_index_resolution_plans is None:
        normalized_plans = []
    elif not isinstance(kv_index_resolution_plans, (list, tuple)):
        payloads = []
        source_path = "runtime_observation_payloads"
    else:
        for plan in kv_index_resolution_plans:
            if plan is not None and not isinstance(plan, Mapping):
                payloads = []
                source_path = "runtime_observation_payloads"
                break
            normalized_plans.append(plan)

    if payloads is None:
        return [
            {
                "event_type": (
                    "relaykv_runtime_metadata_derived_req_to_token_entries_result"
                ),
                "derivation_state": "blocked",
                "derivation_mode": "runtime_metadata_only",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "token_span": None,
                "layer_id": None,
                "kv_head_group": None,
                "derived_req_to_token_entries": [],
                "derived_entry_count": 0,
                "blocked_reason": "runtime_observation_payloads_missing",
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    if payloads == [] and runtime_observation_payloads is not None:
        return [
            {
                "event_type": (
                    "relaykv_runtime_metadata_derived_req_to_token_entries_result"
                ),
                "derivation_state": "blocked",
                "derivation_mode": "runtime_metadata_only",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "token_span": None,
                "layer_id": None,
                "kv_head_group": None,
                "derived_req_to_token_entries": [],
                "derived_entry_count": 0,
                "blocked_reason": "runtime_observation_payload_invalid",
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    assert payloads is not None
    total_derived_entries = 0
    for index, payload in enumerate(payloads):
        plan = normalized_plans[index] if index < len(normalized_plans) else None
        normalized = normalize_relaykv_sglang_adapter_schema_for_smoke(payload)
        adapter_metadata = normalized.get("adapter_metadata")
        if isinstance(adapter_metadata, Mapping):
            result_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
                dict(adapter_metadata)
            )
        else:
            result_adapter_metadata = {}
        result_adapter_metadata["runtime_metadata_derivation_source_path"] = source_path
        if isinstance(plan, Mapping):
            result_adapter_metadata["kv_index_resolution_plan_metadata"] = (
                _copy_relaykv_metadata_value_for_smoke(dict(plan))
            )

        engine_request_id = normalized.get("engine_request_id")
        logical_sequence_id = normalized.get("logical_sequence_id")
        token_span = _relaykv_smoke_token_span_from_value(normalized.get("token_span"))
        if token_span is None:
            token_span = _relaykv_smoke_token_span_from_value(payload.get("token_span"))
        seq_len = _relaykv_smoke_first_present_value(
            payload,
            "seq_len",
        )
        if seq_len is None and isinstance(plan, Mapping):
            seq_len = _relaykv_smoke_first_present_value(plan, "seq_len")
        if seq_len is None and isinstance(result_adapter_metadata, Mapping):
            seq_len = result_adapter_metadata.get("seq_len")

        blocked_reason = None
        derived_entries: list[int] = []
        derived_token_span = token_span
        if token_span is not None:
            token_start, token_end = token_span
            if token_start < 0 or token_end <= token_start:
                blocked_reason = "invalid_token_span"
            else:
                derived_count = token_end - token_start
                if derived_count > max_tokens_per_request:
                    blocked_reason = "max_tokens_per_request_exceeded"
                elif total_derived_entries + derived_count > max_total_tokens:
                    blocked_reason = "max_total_tokens_exceeded"
                else:
                    derived_entries = list(range(token_start, token_end))
                    total_derived_entries += derived_count
        else:
            if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
                blocked_reason = "invalid_seq_len"
            else:
                derived_end = min(seq_len, max_tokens_per_request)
                if derived_end <= 0:
                    blocked_reason = "invalid_seq_len"
                elif total_derived_entries + derived_end > max_total_tokens:
                    blocked_reason = "max_total_tokens_exceeded"
                else:
                    derived_entries = list(range(0, derived_end))
                    derived_token_span = [0, derived_end]
                    total_derived_entries += derived_end

        if token_span is None and seq_len is None and blocked_reason is None:
            blocked_reason = "runtime_metadata_missing"

        results.append(
            {
                "event_type": (
                    "relaykv_runtime_metadata_derived_req_to_token_entries_result"
                ),
                "derivation_state": "derived" if blocked_reason is None else "blocked",
                "derivation_mode": "runtime_metadata_only",
                "engine_request_id": engine_request_id,
                "logical_sequence_id": logical_sequence_id,
                "token_span": derived_token_span,
                "layer_id": normalized.get("layer_id"),
                "kv_head_group": normalized.get("kv_head_group"),
                "derived_req_to_token_entries": derived_entries,
                "derived_entry_count": len(derived_entries),
                "blocked_reason": blocked_reason,
                "source_mutated": False,
                "adapter_metadata": result_adapter_metadata,
                "engine_block_ref": normalized.get("engine_block_ref"),
                "synthetic_metadata_only": True,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        )

    if not results:
        return [
            {
                "event_type": (
                    "relaykv_runtime_metadata_derived_req_to_token_entries_result"
                ),
                "derivation_state": "blocked",
                "derivation_mode": "runtime_metadata_only",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "token_span": None,
                "layer_id": None,
                "kv_head_group": None,
                "derived_req_to_token_entries": [],
                "derived_entry_count": 0,
                "blocked_reason": "runtime_metadata_missing",
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    return results


def summarize_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    production_enabled: bool,
    max_tokens_per_request: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    """Summarize smoke-only runtime metadata-derived req_to_token entries."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV runtime metadata-derived req_to_token results must be a list or tuple"
        )

    derived_count = 0
    blocked_count = 0
    error_count = 0
    total_derived_entries = 0
    totals: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV runtime metadata-derived req_to_token result must be a dict"
            )
        state = str(result.get("derivation_state") or "unknown")
        if state == "derived":
            derived_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = result.get("derived_entry_count")
        if isinstance(value, int) and not isinstance(value, bool):
            total_derived_entries += value

        for key in totals:
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value

    return {
        "event_type": "relaykv_runtime_metadata_derived_req_to_token_entries_summary",
        "production_enabled": production_enabled,
        "result_count": len(results),
        "derived_count": derived_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "total_derived_entries": total_derived_entries,
        "max_tokens_per_request": max_tokens_per_request,
        "max_total_tokens": max_total_tokens,
        **{key: totals[key] for key in sorted(totals)},
    }


def _relaykv_runtime_req_to_token_blocked_result_for_smoke(
    source_payload: Mapping[str, Any] | None,
    *,
    production_enabled: bool,
    blocking_reasons: list[str],
    max_tokens_per_request: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    normalized = (
        normalize_relaykv_sglang_adapter_schema_for_smoke(source_payload)
        if isinstance(source_payload, Mapping)
        else normalize_relaykv_sglang_adapter_schema_for_smoke({})
    )
    adapter_metadata = normalized.get("adapter_metadata")
    if isinstance(adapter_metadata, Mapping):
        payload_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
            dict(adapter_metadata)
        )
    else:
        payload_adapter_metadata = {}
    payload_adapter_metadata.update(
        {
            "runtime_req_to_token_payload_production_enabled": production_enabled,
            "max_tokens_per_request": max_tokens_per_request,
            "max_total_tokens": max_total_tokens,
        }
    )

    normalized.update(
        {
            "event_type": "relaykv_req_to_token_resolution_result",
            "resolution_state": "blocked",
            "adapter_mode": "runtime_req_to_token_payload_production",
            "source": "runtime_metadata_to_req_to_token_resolution_result",
            "decision_state": "SHADOW_ONLY",
            "engine_name": "sglang",
            "adapter_name": "sglang",
            "adapter_metadata": payload_adapter_metadata,
            "full_kv_req_to_token_spans": [],
            "relaykv_working_req_to_token_spans": [],
            "resolved_block_count": 0,
            "resolved_token_count": 0,
            "req_to_token_entry_count": 0,
            "req_to_token_read": False,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read": False,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read": False,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read": False,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read": False,
            "kv_snapshot": False,
            "tensor_read": False,
            "attention_comparison_executed": False,
            "attention_override": False,
            "runtime_writeback": False,
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "source_mutated": False,
            "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
            "warning_reasons": [
                "runtime_req_to_token_payload_production",
                "metadata_only_no_live_req_to_token_read",
                "no_token_to_kv_pool_read",
            ],
        }
    )
    return normalized


def build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
    *,
    runtime_observation_payloads: Any = None,
    kv_index_resolution_plans: Any = None,
    explicit_req_to_token_entries: Any = None,
    production_enabled: bool = False,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """Build smoke-only req_to_token resolution payloads from safe metadata."""

    source_payloads, source_path = _relaykv_runtime_req_to_token_source_payloads_for_smoke(
        runtime_observation_payloads
    )
    explicit_entries, entries_error = _relaykv_runtime_req_to_token_entries_for_smoke(
        explicit_req_to_token_entries
    )

    if production_enabled is not True:
        return [
            _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                None,
                production_enabled=False,
                blocking_reasons=[
                    "runtime_req_to_token_payload_production_not_enabled"
                ],
                max_tokens_per_request=max_tokens_per_request,
                max_total_tokens=max_total_tokens,
            )
        ]

    if source_payloads == [] and runtime_observation_payloads is not None:
        return [
            _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                None,
                production_enabled=True,
                blocking_reasons=["source_payload_invalid"],
                max_tokens_per_request=max_tokens_per_request,
                max_total_tokens=max_total_tokens,
            )
        ]

    if entries_error is not None:
        return [
            _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                source_payloads[0] if source_payloads else None,
                production_enabled=True,
                blocking_reasons=[entries_error],
                max_tokens_per_request=max_tokens_per_request,
                max_total_tokens=max_total_tokens,
            )
        ]

    assert explicit_entries is not None
    if len(explicit_entries) > max_tokens_per_request:
        return [
            _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                source_payloads[0] if source_payloads else None,
                production_enabled=True,
                blocking_reasons=["max_tokens_per_request_exceeded"],
                max_tokens_per_request=max_tokens_per_request,
                max_total_tokens=max_total_tokens,
            )
        ]

    resolved_source_payloads = source_payloads or [{}]
    if len(explicit_entries) * len(resolved_source_payloads) > max_total_tokens:
        return [
            _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                resolved_source_payloads[0],
                production_enabled=True,
                blocking_reasons=["max_total_tokens_exceeded"],
                max_tokens_per_request=max_tokens_per_request,
                max_total_tokens=max_total_tokens,
            )
        ]

    normalized_plans: list[Mapping[str, Any] | None] = []
    if kv_index_resolution_plans is None:
        normalized_plans = [None] * len(resolved_source_payloads)
    elif not isinstance(kv_index_resolution_plans, (list, tuple)):
        return [
            _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                resolved_source_payloads[0],
                production_enabled=True,
                blocking_reasons=["source_payload_invalid"],
                max_tokens_per_request=max_tokens_per_request,
                max_total_tokens=max_total_tokens,
            )
        ]
    else:
        for plan in kv_index_resolution_plans:
            if plan is not None and not isinstance(plan, Mapping):
                return [
                    _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                        resolved_source_payloads[0],
                        production_enabled=True,
                        blocking_reasons=["source_payload_invalid"],
                        max_tokens_per_request=max_tokens_per_request,
                        max_total_tokens=max_total_tokens,
                    )
                ]
            normalized_plans.append(plan)
        if len(normalized_plans) < len(resolved_source_payloads):
            normalized_plans.extend(
                [None] * (len(resolved_source_payloads) - len(normalized_plans))
            )

    results: list[dict[str, Any]] = []
    for index, source_payload in enumerate(resolved_source_payloads):
        if not isinstance(source_payload, Mapping):
            results.append(
                _relaykv_runtime_req_to_token_blocked_result_for_smoke(
                    None,
                    production_enabled=True,
                    blocking_reasons=["source_payload_invalid"],
                    max_tokens_per_request=max_tokens_per_request,
                    max_total_tokens=max_total_tokens,
                )
            )
            continue

        normalized = normalize_relaykv_sglang_adapter_schema_for_smoke(source_payload)
        adapter_metadata = normalized.get("adapter_metadata")
        if isinstance(adapter_metadata, Mapping):
            payload_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
                dict(adapter_metadata)
            )
        else:
            payload_adapter_metadata = {}
        payload_adapter_metadata.update(
            {
                "runtime_req_to_token_payload_production_enabled": True,
                "runtime_req_to_token_payload_source_path": source_path,
                "max_tokens_per_request": max_tokens_per_request,
                "max_total_tokens": max_total_tokens,
            }
        )
        plan = normalized_plans[index] if index < len(normalized_plans) else None
        if isinstance(plan, Mapping):
            payload_adapter_metadata["kv_index_resolution_plan_metadata"] = (
                _copy_relaykv_metadata_value_for_smoke(dict(plan))
            )

        token_span = normalized.get("token_span")
        if not _relaykv_smoke_token_span_from_value(token_span):
            token_span = [0, len(explicit_entries)]
            normalized["token_span"] = token_span
        logical_block_id = normalized.get("logical_block_id")
        if logical_block_id is None:
            logical_block_id = _relaykv_smoke_first_present_value(
                source_payload,
                "logical_block_id",
                "block_id",
            )
            if logical_block_id is None and isinstance(plan, Mapping):
                logical_block_id = _relaykv_smoke_logical_block_id(plan)
            normalized["logical_block_id"] = logical_block_id

        token_start = 0
        token_end = len(explicit_entries)
        if (
            isinstance(token_span, list)
            and len(token_span) == 2
            and isinstance(token_span[0], int)
            and isinstance(token_span[1], int)
        ):
            token_start = token_span[0]
            token_end = token_span[1]

        req_to_token_span = {
            "block_id": logical_block_id,
            "token_start": token_start,
            "token_end": token_end,
            "token_count": len(explicit_entries),
            "req_to_token_entries": list(explicit_entries),
            "entry_count": len(explicit_entries),
            "resolution_source": "runtime_metadata_explicit_req_to_token_entries",
        }
        normalized.update(
            {
                "event_type": "relaykv_req_to_token_resolution_result",
                "resolution_state": "req_to_token_resolved",
                "adapter_mode": "runtime_req_to_token_payload_production",
                "source": "runtime_metadata_to_req_to_token_resolution_result",
                "decision_state": "SHADOW_ONLY",
                "engine_name": "sglang",
                "adapter_name": "sglang",
                "adapter_metadata": payload_adapter_metadata,
                "full_kv_req_to_token_spans": [req_to_token_span],
                "relaykv_working_req_to_token_spans": [req_to_token_span],
                "resolved_block_count": 1,
                "resolved_token_count": len(explicit_entries),
                "req_to_token_entry_count": len(explicit_entries),
                "req_to_token_read": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read": False,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read": False,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read": False,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read": False,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": [],
                "warning_reasons": [
                    "runtime_req_to_token_payload_production",
                    "metadata_only_no_live_req_to_token_read",
                    "no_token_to_kv_pool_read",
                ],
            }
        )
        results.append(normalized)

    return results


def summarize_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
    payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    production_enabled: bool,
    max_tokens_per_request: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    """Summarize smoke-only runtime req_to_token payload production results."""

    if not isinstance(payloads, (list, tuple)):
        raise TypeError(
            "RelayKV runtime req_to_token payloads must be a list or tuple"
        )

    resolved_count = 0
    blocked_count = 0
    error_count = 0
    totals: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(
                "RelayKV runtime req_to_token payload result must be a dict"
            )
        state = str(payload.get("resolution_state") or "unknown")
        if state == "req_to_token_resolved":
            resolved_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1
        for key in (
            "req_to_token_read_count",
            "actual_req_to_token_pool_read_count",
            "token_to_kv_pool_read_count",
            "actual_token_to_kv_pool_read_count",
            "live_token_to_kv_pool_index_read_count",
        ):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
        if payload.get("kv_pool_read") is True:
            totals["kv_pool_read_count"] += 1
        if payload.get("kv_snapshot") is True:
            totals["kv_snapshot_count"] += 1
        if payload.get("tensor_read") is True:
            totals["tensor_read_count"] += 1
        if payload.get("attention_comparison_executed") is True:
            totals["attention_comparison_executed_count"] += 1
        if payload.get("attention_override") is True:
            totals["attention_override_true_count"] += 1
        if payload.get("runtime_writeback") is True:
            totals["runtime_writeback_true_count"] += 1
        if payload.get("scheduler_policy_noop") is False:
            totals["scheduler_policy_noop_false_count"] += 1
        if payload.get("kv_cache_mutation") is True:
            totals["kv_cache_mutation_true_count"] += 1
        if payload.get("source_mutated") is True:
            totals["source_mutated_true_count"] += 1

    return {
        "event_type": (
            "relaykv_runtime_req_to_token_resolution_payload_production_summary"
        ),
        "production_enabled": production_enabled,
        "payload_count": len(payloads),
        "resolved_count": resolved_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "max_tokens_per_request": max_tokens_per_request,
        "max_total_tokens": max_total_tokens,
        **{key: totals[key] for key in sorted(totals)},
    }


def _relaykv_runtime_req_to_token_payload_source_for_smoke(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
) -> tuple[Any, str | None]:
    for owner, path in (
        (forward_batch, "relaykv_runtime_observation_payloads"),
        (model_runner, "relaykv_runtime_observation_payloads"),
    ):
        if owner is None:
            continue
        try:
            value = getattr(owner, path, None)
        except Exception:
            continue
        if value is None:
            continue
        owner_name = "forward_batch" if owner is forward_batch else "model_runner"
        if isinstance(value, Mapping):
            return [value], f"{owner_name}.{path}"
        return value, f"{owner_name}.{path}"
    return None, None


def _relaykv_runtime_observation_metadata_source_bridge_source_for_smoke(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
    explicit_runtime_observation_payloads: Any = None,
) -> tuple[Any, str | None]:
    if explicit_runtime_observation_payloads is not None:
        return explicit_runtime_observation_payloads, "explicit_runtime_observation_payloads"

    for owner, path in (
        (forward_batch, "relaykv_runtime_observation_payloads"),
        (forward_batch, "relaykv_runtime_observation_metadata"),
        (model_runner, "relaykv_runtime_observation_payloads"),
        (model_runner, "relaykv_runtime_observation_metadata"),
    ):
        if owner is None:
            continue
        try:
            value = getattr(owner, path, None)
        except Exception:
            continue
        if value is not None:
            owner_name = "forward_batch" if owner is forward_batch else "model_runner"
            return value, f"{owner_name}.{path}"
    return None, None


def _relaykv_runtime_observation_bridge_payloads_from_mapping_for_smoke(
    payload: Mapping[str, Any],
    *,
    source_path: str | None,
) -> list[dict[str, Any]] | None:
    request_id = _relaykv_smoke_first_present_value(
        payload,
        "engine_request_id",
        "request_id",
        "rid",
    )
    adapter_metadata_value = payload.get("adapter_metadata")
    if isinstance(adapter_metadata_value, Mapping):
        adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
            dict(adapter_metadata_value)
        )
    else:
        adapter_metadata = {}
    seq_len = _relaykv_smoke_first_present_value(payload, "seq_len")
    if seq_len is None and isinstance(adapter_metadata, Mapping):
        seq_len = adapter_metadata.get("seq_len")
    token_span = _relaykv_smoke_token_span_from_value(payload.get("token_span"))
    if request_id is None:
        return None
    if token_span is None and (
        isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0
    ):
        return None

    layer_values: list[int | None] = []
    layer_id = _relaykv_smoke_first_present_value(payload, "layer_id")
    if isinstance(layer_id, int) and not isinstance(layer_id, bool):
        layer_values = [layer_id]
    else:
        raw_layer_ids = payload.get("layer_ids")
        if isinstance(raw_layer_ids, (list, tuple)) and raw_layer_ids:
            for raw_layer_id in raw_layer_ids:
                if isinstance(raw_layer_id, int) and not isinstance(raw_layer_id, bool):
                    layer_values.append(raw_layer_id)
                else:
                    return None
        else:
            layer_values = [None]

    engine_block_ref = payload.get("engine_block_ref")
    if isinstance(engine_block_ref, Mapping):
        normalized_engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
            dict(engine_block_ref)
        )
    else:
        normalized_engine_block_ref = {}
    req_pool_idx = _relaykv_smoke_first_present_value(payload, "req_pool_idx", "req_pool_index")
    if req_pool_idx is not None:
        normalized_engine_block_ref["req_pool_idx"] = _copy_relaykv_metadata_value_for_smoke(
            req_pool_idx
        )

    adapter_metadata["runtime_observation_source_bridge"] = True
    adapter_metadata["runtime_observation_source_bridge_source_path"] = source_path

    payloads: list[dict[str, Any]] = []
    for layer_value in layer_values:
        bridged_payload: dict[str, Any] = {
            "event_type": "relaykv_runtime_observation_metadata_source_bridge_payload",
            "engine_name": "sglang",
            "adapter_name": "sglang",
            "engine_request_id": request_id,
            "request_id": request_id,
            "logical_sequence_id": _relaykv_smoke_first_present_value(
                payload,
                "logical_sequence_id",
                "sequence_id",
                "request_id",
                "rid",
            ),
            "logical_block_id": _relaykv_smoke_first_present_value(
                payload,
                "logical_block_id",
                "block_id",
            ),
            "layer_id": layer_value,
            "kv_head_group": _relaykv_smoke_first_present_value(
                payload,
                "kv_head_group",
                "kv_group",
                "head_group",
            ),
            "kv_class": _relaykv_smoke_first_present_value(
                payload,
                "kv_class",
                "kv_cache_class",
            )
            or "UNKNOWN",
            "position_check_state": "not_checked_metadata_only",
            "attention_mask_mode": "unknown",
            "rope_position_consistency": "not_checked",
            "adapter_metadata": _copy_relaykv_metadata_value_for_smoke(
                dict(adapter_metadata)
            ),
            "engine_block_ref": _copy_relaykv_metadata_value_for_smoke(
                dict(normalized_engine_block_ref)
            )
            if normalized_engine_block_ref
            else None,
            "source_mutated": False,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
        if token_span is not None:
            bridged_payload["token_span"] = token_span
        if isinstance(seq_len, int) and not isinstance(seq_len, bool):
            bridged_payload["seq_len"] = seq_len
        payloads.append(bridged_payload)
    return payloads


def build_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
    explicit_runtime_observation_payloads: Any = None,
    bridge_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Bridge explicit runtime observation metadata into payloads for smoke only."""

    source_value, source_path = (
        _relaykv_runtime_observation_metadata_source_bridge_source_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
            explicit_runtime_observation_payloads=explicit_runtime_observation_payloads,
        )
    )

    bridge_state = "bridged"
    blocked_reason = None
    payload_count = 0
    blocked_payload_count = 0
    valid_payloads: list[dict[str, Any]] = []

    if bridge_enabled is not True:
        bridge_state = "blocked"
        blocked_reason = "bridge_not_enabled"
    elif source_value is None:
        bridge_state = "blocked"
        blocked_reason = "runtime_observation_source_missing"
    else:
        source_items: list[Any]
        if isinstance(source_value, Mapping):
            source_items = [source_value]
        elif isinstance(source_value, (list, tuple)):
            source_items = list(source_value)
        else:
            source_items = []
            bridge_state = "blocked"
            blocked_reason = "runtime_observation_source_not_list_or_tuple"

        if bridge_state != "blocked":
            payload_count = len(source_items)
            if payload_count == 0:
                bridge_state = "blocked"
                blocked_reason = "runtime_observation_source_empty"
            else:
                for item in source_items:
                    if not isinstance(item, Mapping):
                        blocked_payload_count += 1
                        continue
                    bridged_payloads = (
                        _relaykv_runtime_observation_bridge_payloads_from_mapping_for_smoke(
                            item,
                            source_path=source_path,
                        )
                    )
                    if not bridged_payloads:
                        blocked_payload_count += 1
                        continue
                    valid_payloads.extend(bridged_payloads)
                if not valid_payloads:
                    bridge_state = "blocked"
                    blocked_reason = "runtime_observation_source_invalid"

    return [
        {
            "event_type": "relaykv_runtime_observation_metadata_source_bridge_result",
            "bridge_state": bridge_state,
            "bridge_mode": "runtime_observation_metadata_source_bridge",
            "payload_count": payload_count,
            "valid_payload_count": len(valid_payloads),
            "blocked_payload_count": blocked_payload_count,
            "bridge_source_path": source_path,
            "blocked_reason": blocked_reason,
            "runtime_observation_payloads": valid_payloads,
            "source_mutated": False,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    ]


def summarize_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
    bridge_results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize runtime observation metadata source bridge payloads for smoke."""

    if not isinstance(bridge_results, (list, tuple)):
        raise TypeError(
            "RelayKV runtime observation metadata source bridge results must be a list or tuple"
        )

    summary: dict[str, Any] = {
        "event_type": "relaykv_runtime_observation_metadata_source_bridge_summary",
        "bridge_enabled": False,
        "bridge_state": "blocked",
        "payload_count": 0,
        "valid_payload_count": 0,
        "blocked_payload_count": 0,
        "bridge_source_path": None,
        "blocked_reason": None,
        "req_to_token_read_count": 0,
        "actual_req_to_token_pool_read_count": 0,
        "token_to_kv_pool_read_count": 0,
        "actual_token_to_kv_pool_read_count": 0,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
        "tensor_read_count": 0,
        "attention_comparison_executed_count": 0,
        "attention_override_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "kv_cache_mutation_true_count": 0,
        "source_mutated_true_count": 0,
    }
    for result in bridge_results:
        if not isinstance(result, Mapping):
            raise TypeError(
                "RelayKV runtime observation metadata source bridge result must be a mapping"
            )
        summary["bridge_enabled"] = bool(
            result.get("bridge_state") in {"bridged", "blocked", "error"}
        )
        summary["bridge_state"] = result.get("bridge_state") or "blocked"
        for key in ("payload_count", "valid_payload_count", "blocked_payload_count"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                summary[key] += value
        summary["bridge_source_path"] = result.get("bridge_source_path")
        summary["blocked_reason"] = result.get("blocked_reason")
        for key in (
            "req_to_token_read_count",
            "actual_req_to_token_pool_read_count",
            "token_to_kv_pool_read_count",
            "actual_token_to_kv_pool_read_count",
            "kv_pool_read_count",
            "kv_snapshot_count",
            "tensor_read_count",
            "attention_comparison_executed_count",
            "attention_override_true_count",
            "runtime_writeback_true_count",
            "scheduler_policy_noop_false_count",
            "kv_cache_mutation_true_count",
            "source_mutated_true_count",
        ):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                summary[key] += value
    return summary


def _relaykv_req_pool_idx_from_runtime_observation_payload_for_smoke(
    payload: Mapping[str, Any],
) -> Any:
    req_pool_idx = _relaykv_smoke_first_present_value(
        payload,
        "req_pool_idx",
        "req_pool_index",
    )
    if req_pool_idx is not None:
        return req_pool_idx
    engine_block_ref = payload.get("engine_block_ref")
    if isinstance(engine_block_ref, Mapping):
        return _relaykv_smoke_first_present_value(
            engine_block_ref,
            "req_pool_idx",
            "req_pool_index",
        )
    return None


def _relaykv_bounded_req_to_token_positions_for_smoke(
    payload: Mapping[str, Any],
) -> tuple[list[int] | None, list[int] | None, Any, str | None]:
    token_span = _relaykv_smoke_token_span(payload)
    if token_span is not None:
        token_start, token_end = token_span
        if token_start < 0 or token_end <= token_start:
            return None, token_span, None, "token_span_invalid"
        return list(range(token_start, token_end)), token_span, None, None

    seq_len = _relaykv_smoke_first_present_value(payload, "seq_len")
    adapter_metadata = payload.get("adapter_metadata")
    if seq_len is None and isinstance(adapter_metadata, Mapping):
        seq_len = adapter_metadata.get("seq_len")
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
        return None, None, seq_len, "seq_len_invalid"
    return list(range(0, seq_len)), [0, seq_len], seq_len, None


def _relaykv_read_req_to_token_value_for_smoke(
    req_to_token_pool_object: Any,
    req_pool_idx: Any,
    token_position: int,
) -> tuple[Any, str | None]:
    try:
        if isinstance(req_to_token_pool_object, dict):
            pool_row = req_to_token_pool_object.get(req_pool_idx)
            if pool_row is None and not isinstance(req_pool_idx, str):
                pool_row = req_to_token_pool_object.get(str(req_pool_idx))
            if pool_row is None:
                return None, "req_to_token_pool_index_error"
        else:
            pool_row = req_to_token_pool_object[req_pool_idx]
    except Exception:
        return None, "req_to_token_pool_index_error"

    try:
        if isinstance(pool_row, dict):
            value = pool_row.get(token_position)
            if value is None:
                value = pool_row.get(str(token_position))
        else:
            value = pool_row[token_position]
    except Exception:
        return None, "req_to_token_pool_index_error"
    return value, None


def _relaykv_shallow_noncallable_attr_for_smoke(
    value: Any,
    attr_name: str,
) -> tuple[bool, Any]:
    try:
        attr_value = getattr(value, attr_name)
    except Exception:
        return False, None
    if callable(attr_value):
        return False, None
    return True, attr_value


def _relaykv_normalize_shape_metadata_for_smoke(value: Any) -> list[int] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        normalized: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                return None
            normalized.append(item)
        return normalized
    return None


def _relaykv_req_to_token_value_shape_observation_for_smoke(
    value: Any,
) -> dict[str, Any]:
    value_type = value.__class__
    value_type_name = value_type.__name__
    value_module = value_type.__module__
    value_qualname = value_type.__qualname__

    has_shape, raw_shape = _relaykv_shallow_noncallable_attr_for_smoke(value, "shape")
    normalized_shape = _relaykv_normalize_shape_metadata_for_smoke(raw_shape)
    if not has_shape:
        has_size, raw_size = _relaykv_shallow_noncallable_attr_for_smoke(value, "size")
        if has_size:
            size_shape = _relaykv_normalize_shape_metadata_for_smoke(raw_size)
            if size_shape is not None:
                has_shape = True
                normalized_shape = size_shape

    has_dtype, raw_dtype = _relaykv_shallow_noncallable_attr_for_smoke(value, "dtype")
    has_device, raw_device = _relaykv_shallow_noncallable_attr_for_smoke(value, "device")
    has_ndim, raw_ndim = _relaykv_shallow_noncallable_attr_for_smoke(value, "ndim")

    value_len = None
    value_has_len = False
    try:
        computed_len = len(value)
    except Exception:
        computed_len = None
    else:
        if isinstance(computed_len, int) and not isinstance(computed_len, bool):
            value_has_len = True
            value_len = computed_len

    value_is_int = isinstance(value, int) and not isinstance(value, bool)
    value_is_bool = isinstance(value, bool)
    value_is_float = isinstance(value, float)
    value_is_list = isinstance(value, list)
    value_is_tuple = isinstance(value, tuple)
    value_is_tensor_like = bool(has_shape or has_dtype or has_device)

    value_is_scalar_like = False
    if normalized_shape == []:
        value_is_scalar_like = True
    elif isinstance(raw_ndim, int) and not isinstance(raw_ndim, bool) and raw_ndim == 0:
        value_is_scalar_like = True

    value_is_one_element_like = False
    if normalized_shape is not None:
        if normalized_shape == []:
            value_is_one_element_like = True
        elif normalized_shape:
            product = 1
            for dim in normalized_shape:
                if dim < 0:
                    product = 0
                    break
                product *= dim
            if product == 1:
                value_is_one_element_like = True
    elif value_has_len and value_len == 1:
        value_is_one_element_like = True

    return {
        "value_type_name": value_type_name,
        "value_module": value_module,
        "value_qualname": value_qualname,
        "value_has_shape": has_shape,
        "value_shape": normalized_shape,
        "value_has_dtype": has_dtype,
        "value_dtype": str(raw_dtype) if has_dtype else None,
        "value_has_device": has_device,
        "value_device": str(raw_device) if has_device else None,
        "value_has_len": value_has_len,
        "value_len": value_len,
        "value_is_int": value_is_int,
        "value_is_bool": value_is_bool,
        "value_is_float": value_is_float,
        "value_is_list": value_is_list,
        "value_is_tuple": value_is_tuple,
        "value_is_tensor_like": value_is_tensor_like,
        "value_is_scalar_like": value_is_scalar_like,
        "value_is_one_element_like": value_is_one_element_like,
    }


def build_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke(
    *,
    runtime_observation_payloads: Any = None,
    req_to_token_pool_object: Any = None,
    inspect_req_to_token_pool_value_shape: bool = False,
    max_tokens_per_request: int = 8,
    max_total_tokens: int = 16,
    source_path: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect shallow metadata for bounded req_to_token pool values without conversion."""

    source_payloads, payload_source_path = _relaykv_runtime_req_to_token_source_payloads_for_smoke(
        runtime_observation_payloads
    )
    if source_path is None:
        source_path = payload_source_path

    base_result = {
        "event_type": "relaykv_req_to_token_pool_value_shape_inspection_result",
        "adapter_mode": "req_to_token_pool_value_shape_inspection",
        "decision_state": "SHADOW_ONLY",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": None,
        "logical_sequence_id": None,
        "req_pool_idx": None,
        "token_span": None,
        "seq_len": None,
        "source_path": source_path,
        "probed_value_count": 0,
        "value_shape_observations": [],
        "blocked_reason": None,
        "source_mutated": False,
        "req_to_token_value_shape_inspection_count": 0,
        "req_to_token_read_count": 0,
        "actual_req_to_token_pool_read_count": 0,
        "token_to_kv_pool_read_count": 0,
        "actual_token_to_kv_pool_read_count": 0,
        "live_token_to_kv_pool_index_read_count": 0,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
        "tensor_read_count": 0,
        "attention_comparison_executed_count": 0,
        "attention_override_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "kv_cache_mutation_true_count": 0,
        "source_mutated_true_count": 0,
    }

    if inspect_req_to_token_pool_value_shape is not True:
        blocked = dict(base_result)
        blocked["inspection_state"] = "blocked"
        blocked["blocked_reason"] = "req_to_token_pool_value_shape_inspection_not_enabled"
        return [blocked]

    if source_payloads is None or (
        source_payloads == [] and runtime_observation_payloads is not None
    ):
        blocked = dict(base_result)
        blocked["inspection_state"] = "blocked"
        blocked["blocked_reason"] = "runtime_observation_payloads_missing"
        return [blocked]

    if req_to_token_pool_object is None:
        blocked = dict(base_result)
        blocked["inspection_state"] = "blocked"
        blocked["blocked_reason"] = "req_to_token_pool_object_missing"
        return [blocked]

    assert source_payloads is not None
    results: list[dict[str, Any]] = []
    total_tokens = 0

    for payload in source_payloads:
        if not isinstance(payload, Mapping):
            raise TypeError(
                "RelayKV req_to_token pool value shape inspection inputs must be mappings"
            )
        normalized = normalize_relaykv_sglang_adapter_schema_for_smoke(payload)
        req_pool_idx = _relaykv_req_pool_idx_from_runtime_observation_payload_for_smoke(
            payload
        )
        token_positions, token_span, seq_len, token_error = (
            _relaykv_bounded_req_to_token_positions_for_smoke(payload)
        )
        blocked_reason = None
        if req_pool_idx is None:
            blocked_reason = "req_pool_idx_missing"
        elif token_positions is None and token_error == "token_span_invalid":
            blocked_reason = "token_span_invalid"
        elif token_positions is None and token_error == "seq_len_invalid":
            blocked_reason = "seq_len_invalid"
        elif token_positions is None:
            blocked_reason = "token_span_or_seq_len_missing"

        if blocked_reason is None:
            assert token_positions is not None
            if len(token_positions) > max_tokens_per_request:
                blocked_reason = "max_tokens_per_request_exceeded"
            elif total_tokens + len(token_positions) > max_total_tokens:
                blocked_reason = "max_total_tokens_exceeded"

        observations: list[dict[str, Any]] = []
        probe_count = 0
        if blocked_reason is None:
            assert token_positions is not None
            for token_position in token_positions:
                value, value_error = _relaykv_read_req_to_token_value_for_smoke(
                    req_to_token_pool_object,
                    req_pool_idx,
                    token_position,
                )
                if value_error is not None:
                    blocked_reason = value_error
                    break
                probe_count += 1
                observations.append(
                    _relaykv_req_to_token_value_shape_observation_for_smoke(value)
                )
            total_tokens += probe_count

        result = dict(base_result)
        result.update(
            {
                "inspection_state": (
                    "inspected" if blocked_reason is None else "blocked"
                ),
                "engine_request_id": normalized.get("engine_request_id"),
                "logical_sequence_id": normalized.get("logical_sequence_id"),
                "req_pool_idx": req_pool_idx,
                "token_span": token_span,
                "seq_len": seq_len,
                "probed_value_count": probe_count,
                "value_shape_observations": observations,
                "blocked_reason": blocked_reason,
                "req_to_token_value_shape_inspection_count": probe_count,
                "req_to_token_read_count": probe_count,
                "actual_req_to_token_pool_read_count": probe_count,
            }
        )
        results.append(result)

    return results


def summarize_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    inspection_enabled: bool,
    max_tokens_per_request: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    """Summarize shallow req_to_token pool value shape inspections."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV req_to_token pool value shape inspection results must be a list or tuple"
        )

    inspected_count = 0
    blocked_count = 0
    error_count = 0
    total_probed_value_count = 0
    observed_type_counts: Counter[str] = Counter()
    observed_shape_counts: Counter[str] = Counter()
    observed_dtype_counts: Counter[str] = Counter()
    observed_device_counts: Counter[str] = Counter()
    observed_scalar_like_count = 0
    observed_one_element_like_count = 0
    safety_counts: Counter[str] = Counter(
        {
            "req_to_token_value_shape_inspection_count": 0,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )

    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError(
                "RelayKV req_to_token pool value shape inspection result must be a mapping"
            )
        state = str(result.get("inspection_state") or "unknown")
        if state == "inspected":
            inspected_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        probed = result.get("probed_value_count")
        if isinstance(probed, int) and not isinstance(probed, bool):
            total_probed_value_count += probed

        observations = result.get("value_shape_observations")
        if isinstance(observations, list):
            for observation in observations:
                if not isinstance(observation, Mapping):
                    continue
                observed_type_counts[str(observation.get("value_type_name") or "unknown")] += 1
                shape_value = observation.get("value_shape")
                observed_shape_counts[json.dumps(shape_value, sort_keys=True)] += 1
                dtype_value = observation.get("value_dtype")
                observed_dtype_counts[str(dtype_value)] += 1
                device_value = observation.get("value_device")
                observed_device_counts[str(device_value)] += 1
                if observation.get("value_is_scalar_like") is True:
                    observed_scalar_like_count += 1
                if observation.get("value_is_one_element_like") is True:
                    observed_one_element_like_count += 1

        for key in safety_counts:
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                safety_counts[key] += value

    return {
        "event_type": "relaykv_req_to_token_pool_value_shape_inspection_summary",
        "inspection_enabled": inspection_enabled,
        "result_count": len(results),
        "inspected_count": inspected_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "total_probed_value_count": total_probed_value_count,
        "observed_type_counts": dict(observed_type_counts),
        "observed_shape_counts": dict(observed_shape_counts),
        "observed_dtype_counts": dict(observed_dtype_counts),
        "observed_device_counts": dict(observed_device_counts),
        "observed_scalar_like_count": observed_scalar_like_count,
        "observed_one_element_like_count": observed_one_element_like_count,
        "max_tokens_per_request": max_tokens_per_request,
        "max_total_tokens": max_total_tokens,
        **{key: safety_counts[key] for key in sorted(safety_counts)},
    }


def build_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
    *,
    runtime_observation_payloads: Any = None,
    req_to_token_pool_object: Any = None,
    read_req_to_token_pool: bool = False,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
    source_path: str | None = None,
) -> list[dict[str, Any]]:
    """Read bounded req_to_token entries from a fake readonly req_to_token pool."""

    source_payloads, payload_source_path = _relaykv_runtime_req_to_token_source_payloads_for_smoke(
        runtime_observation_payloads
    )
    if source_path is None:
        source_path = payload_source_path

    if read_req_to_token_pool is not True:
        return [
            {
                "event_type": "relaykv_real_req_to_token_pool_bounded_read_result",
                "read_state": "blocked",
                "adapter_mode": "real_req_to_token_pool_bounded_read",
                "decision_state": "SHADOW_ONLY",
                "engine_name": "sglang",
                "adapter_name": "sglang",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "req_pool_idx": None,
                "token_span": None,
                "seq_len": None,
                "layer_id": None,
                "kv_head_group": None,
                "req_to_token_index_preview": [],
                "req_to_token_index_count": 0,
                "req_to_token_index_checksum": None,
                "blocked_reason": "req_to_token_pool_read_not_enabled",
                "source_path": source_path,
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    if source_payloads is None:
        return [
            {
                "event_type": "relaykv_real_req_to_token_pool_bounded_read_result",
                "read_state": "blocked",
                "adapter_mode": "real_req_to_token_pool_bounded_read",
                "decision_state": "SHADOW_ONLY",
                "engine_name": "sglang",
                "adapter_name": "sglang",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "req_pool_idx": None,
                "token_span": None,
                "seq_len": None,
                "layer_id": None,
                "kv_head_group": None,
                "req_to_token_index_preview": [],
                "req_to_token_index_count": 0,
                "req_to_token_index_checksum": None,
                "blocked_reason": "runtime_observation_payloads_missing",
                "source_path": source_path,
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    if req_to_token_pool_object is None:
        return [
            {
                "event_type": "relaykv_real_req_to_token_pool_bounded_read_result",
                "read_state": "blocked",
                "adapter_mode": "real_req_to_token_pool_bounded_read",
                "decision_state": "SHADOW_ONLY",
                "engine_name": "sglang",
                "adapter_name": "sglang",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "req_pool_idx": None,
                "token_span": None,
                "seq_len": None,
                "layer_id": None,
                "kv_head_group": None,
                "req_to_token_index_preview": [],
                "req_to_token_index_count": 0,
                "req_to_token_index_checksum": None,
                "blocked_reason": "req_to_token_pool_object_missing",
                "source_path": source_path,
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    if source_payloads == [] and runtime_observation_payloads is not None:
        return [
            {
                "event_type": "relaykv_real_req_to_token_pool_bounded_read_result",
                "read_state": "blocked",
                "adapter_mode": "real_req_to_token_pool_bounded_read",
                "decision_state": "SHADOW_ONLY",
                "engine_name": "sglang",
                "adapter_name": "sglang",
                "engine_request_id": None,
                "logical_sequence_id": None,
                "req_pool_idx": None,
                "token_span": None,
                "seq_len": None,
                "layer_id": None,
                "kv_head_group": None,
                "req_to_token_index_preview": [],
                "req_to_token_index_count": 0,
                "req_to_token_index_checksum": None,
                "blocked_reason": "runtime_observation_payloads_missing",
                "source_path": source_path,
                "source_mutated": False,
                "req_to_token_read_count": 0,
                "actual_req_to_token_pool_read_count": 0,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read_count": 0,
                "kv_snapshot_count": 0,
                "tensor_read_count": 0,
                "attention_comparison_executed_count": 0,
                "attention_override_true_count": 0,
                "runtime_writeback_true_count": 0,
                "scheduler_policy_noop_false_count": 0,
                "kv_cache_mutation_true_count": 0,
                "source_mutated_true_count": 0,
            }
        ]

    assert source_payloads is not None
    results: list[dict[str, Any]] = []
    total_tokens = 0
    max_preview_entries = 8
    checksum_modulus = 1000000007

    for payload in source_payloads:
        if not isinstance(payload, Mapping):
            raise TypeError(
                "RelayKV real req_to_token pool bounded read inputs must be mappings"
            )

        normalized = normalize_relaykv_sglang_adapter_schema_for_smoke(payload)
        req_pool_idx = _relaykv_req_pool_idx_from_runtime_observation_payload_for_smoke(
            payload
        )
        token_positions, token_span, seq_len, token_error = (
            _relaykv_bounded_req_to_token_positions_for_smoke(payload)
        )
        blocked_reason = None
        if req_pool_idx is None:
            blocked_reason = "req_pool_idx_missing"
        elif token_positions is None and token_error in {
            "token_span_invalid",
            "seq_len_invalid",
        }:
            blocked_reason = token_error
        elif token_positions is None:
            blocked_reason = "token_span_or_seq_len_missing"

        if blocked_reason is None:
            assert token_positions is not None
            if len(token_positions) > max_tokens_per_request:
                blocked_reason = "max_tokens_per_request_exceeded"
            elif total_tokens + len(token_positions) > max_total_tokens:
                blocked_reason = "max_total_tokens_exceeded"

        read_values: list[int] = []
        if blocked_reason is None:
            assert token_positions is not None
            for token_position in token_positions:
                value, value_error = _relaykv_read_req_to_token_value_for_smoke(
                    req_to_token_pool_object,
                    req_pool_idx,
                    token_position,
                )
                if value_error is not None:
                    blocked_reason = value_error
                    read_values = []
                    break
                if isinstance(value, bool) or not isinstance(value, int):
                    blocked_reason = "req_to_token_pool_value_not_int"
                    read_values = []
                    break
                read_values.append(value)

        checksum = None
        if read_values:
            checksum = (
                sum((index + 1) * value for index, value in enumerate(read_values))
                % checksum_modulus
            )

        if blocked_reason is None:
            total_tokens += len(read_values)

        result = {
            "event_type": "relaykv_real_req_to_token_pool_bounded_read_result",
            "read_state": (
                "req_to_token_pool_resolved" if blocked_reason is None else "blocked"
            ),
            "adapter_mode": "real_req_to_token_pool_bounded_read",
            "decision_state": "SHADOW_ONLY",
            "engine_name": "sglang",
            "adapter_name": "sglang",
            "engine_request_id": normalized.get("engine_request_id"),
            "logical_sequence_id": normalized.get("logical_sequence_id"),
            "req_pool_idx": req_pool_idx,
            "token_span": token_span,
            "seq_len": seq_len,
            "layer_id": normalized.get("layer_id"),
            "kv_head_group": normalized.get("kv_head_group"),
            "req_to_token_index_preview": list(read_values[:max_preview_entries]),
            "req_to_token_index_count": len(read_values),
            "req_to_token_index_checksum": checksum,
            "blocked_reason": blocked_reason,
            "source_path": source_path,
            "source_mutated": False,
            "req_to_token_read_count": len(read_values) if blocked_reason is None else 0,
            "actual_req_to_token_pool_read_count": (
                len(read_values) if blocked_reason is None else 0
            ),
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
            "adapter_metadata": normalized.get("adapter_metadata"),
            "engine_block_ref": normalized.get("engine_block_ref"),
        }
        results.append(result)

    return results


def summarize_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    read_enabled: bool,
    max_tokens_per_request: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    """Summarize bounded real req_to_token pool readonly read smoke results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV real req_to_token pool bounded read results must be a list or tuple"
        )

    resolved_count = 0
    blocked_count = 0
    error_count = 0
    total_req_to_token_index_count = 0
    safety_counts: Counter[str] = Counter(
        {
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "live_token_to_kv_pool_index_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError(
                "RelayKV real req_to_token pool bounded read result must be a mapping"
            )
        state = str(result.get("read_state") or "unknown")
        if state == "req_to_token_pool_resolved":
            resolved_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1
        value = result.get("req_to_token_index_count")
        if isinstance(value, int) and not isinstance(value, bool):
            total_req_to_token_index_count += value
        for key in safety_counts:
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                safety_counts[key] += value

    return {
        "event_type": "relaykv_real_req_to_token_pool_bounded_read_summary",
        "read_enabled": read_enabled,
        "result_count": len(results),
        "resolved_count": resolved_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "total_req_to_token_index_count": total_req_to_token_index_count,
        "max_tokens_per_request": max_tokens_per_request,
        "max_total_tokens": max_total_tokens,
        **{key: safety_counts[key] for key in sorted(safety_counts)},
    }


def build_relaykv_req_to_token_resolution_payloads_from_real_pool_read_for_smoke(
    real_pool_read_results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Convert bounded real req_to_token pool read results into resolution payloads."""

    if not isinstance(real_pool_read_results, (list, tuple)):
        raise TypeError(
            "RelayKV real req_to_token pool bounded read results must be a list or tuple"
        )

    payloads: list[dict[str, Any]] = []
    for result in real_pool_read_results:
        if not isinstance(result, Mapping):
            raise TypeError(
                "RelayKV real req_to_token pool bounded read result must be a mapping"
            )
        normalized = normalize_relaykv_sglang_adapter_schema_for_smoke(result)
        adapter_metadata = normalized.get("adapter_metadata")
        if isinstance(adapter_metadata, Mapping):
            payload_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
                dict(adapter_metadata)
            )
        else:
            payload_adapter_metadata = {}
        payload_adapter_metadata["real_req_to_token_pool_bounded_read_source_path"] = (
            result.get("source_path")
        )

        if result.get("read_state") != "req_to_token_pool_resolved":
            normalized.update(
                {
                    "event_type": "relaykv_req_to_token_resolution_result",
                    "resolution_state": "blocked",
                    "adapter_mode": "real_req_to_token_pool_bounded_read_conversion",
                    "source": (
                        "real_req_to_token_pool_bounded_read_result_to_"
                        "req_to_token_resolution_result"
                    ),
                    "decision_state": "SHADOW_ONLY",
                    "adapter_metadata": payload_adapter_metadata,
                    "full_kv_req_to_token_spans": [],
                    "relaykv_working_req_to_token_spans": [],
                    "resolved_block_count": 0,
                    "resolved_token_count": 0,
                    "req_to_token_entry_count": 0,
                    "req_to_token_read": False,
                    "req_to_token_read_count": 0,
                    "actual_req_to_token_pool_read": False,
                    "actual_req_to_token_pool_read_count": 0,
                    "token_to_kv_pool_read": False,
                    "token_to_kv_pool_read_count": 0,
                    "actual_token_to_kv_pool_read": False,
                    "actual_token_to_kv_pool_read_count": 0,
                    "live_token_to_kv_pool_index_read": False,
                    "live_token_to_kv_pool_index_read_count": 0,
                    "kv_pool_read": False,
                    "kv_snapshot": False,
                    "tensor_read": False,
                    "attention_comparison_executed": False,
                    "attention_override": False,
                    "runtime_writeback": False,
                    "scheduler_policy_noop": True,
                    "kv_cache_mutation": False,
                    "source_mutated": False,
                    "blocking_reasons": [result.get("blocked_reason")],
                    "warning_reasons": [
                        "real_req_to_token_pool_bounded_read_conversion"
                    ],
                }
            )
            payloads.append(normalized)
            continue

        token_span = _relaykv_smoke_token_span_from_value(result.get("token_span")) or [
            0,
            int(result.get("req_to_token_index_count") or 0),
        ]
        req_to_token_entries = list(result.get("req_to_token_index_preview") or [])
        req_to_token_count = int(result.get("req_to_token_index_count") or 0)
        engine_block_ref = normalized.get("engine_block_ref")
        if isinstance(engine_block_ref, Mapping):
            payload_engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
                dict(engine_block_ref)
            )
        else:
            payload_engine_block_ref = {}
        payload_engine_block_ref["req_to_token_index_count"] = req_to_token_count
        payload_engine_block_ref["req_to_token_index_checksum"] = result.get(
            "req_to_token_index_checksum"
        )

        req_to_token_span = {
            "block_id": normalized.get("logical_block_id"),
            "token_start": token_span[0],
            "token_end": token_span[1],
            "token_count": req_to_token_count,
            "req_to_token_entries": req_to_token_entries,
            "entry_count": req_to_token_count,
            "resolution_source": "real_req_to_token_pool_bounded_read",
        }
        normalized.update(
            {
                "event_type": "relaykv_req_to_token_resolution_result",
                "resolution_state": "req_to_token_resolved",
                "adapter_mode": "real_req_to_token_pool_bounded_read_conversion",
                "source": (
                    "real_req_to_token_pool_bounded_read_result_to_"
                    "req_to_token_resolution_result"
                ),
                "decision_state": "SHADOW_ONLY",
                "adapter_metadata": payload_adapter_metadata,
                "engine_block_ref": payload_engine_block_ref,
                "full_kv_req_to_token_spans": [req_to_token_span],
                "relaykv_working_req_to_token_spans": [req_to_token_span],
                "resolved_block_count": 1,
                "resolved_token_count": req_to_token_count,
                "req_to_token_entry_count": req_to_token_count,
                "req_to_token_read": True,
                "req_to_token_read_count": req_to_token_count,
                "actual_req_to_token_pool_read": True,
                "actual_req_to_token_pool_read_count": req_to_token_count,
                "token_to_kv_pool_read": False,
                "token_to_kv_pool_read_count": 0,
                "actual_token_to_kv_pool_read": False,
                "actual_token_to_kv_pool_read_count": 0,
                "live_token_to_kv_pool_index_read": False,
                "live_token_to_kv_pool_index_read_count": 0,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": [],
                "warning_reasons": [
                    "real_req_to_token_pool_bounded_read_conversion",
                    "bounded_preview_only",
                    "no_token_to_kv_pool_read",
                ],
            }
        )
        payloads.append(normalized)

    return payloads


def _relaykv_real_req_to_token_pool_source_for_smoke(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
) -> tuple[Any, str | None]:
    for owner, owner_name in (
        (model_runner, "model_runner"),
        (forward_batch, "forward_batch"),
    ):
        if owner is None:
            continue
        try:
            direct_value = getattr(owner, "req_to_token_pool", None)
        except Exception:
            direct_value = None
        if direct_value is not None:
            try:
                nested_value = getattr(direct_value, "req_to_token", None)
            except Exception:
                nested_value = None
            if nested_value is not None:
                return nested_value, f"{owner_name}.req_to_token_pool.req_to_token"
            return direct_value, f"{owner_name}.req_to_token_pool"
    return None, None


def run_model_runner_real_req_to_token_pool_bounded_read_hook_for_smoke(
    model_runner: Any,
    forward_batch: Any = None,
) -> dict[str, Any]:
    """Run a smoke-only bounded real req_to_token_pool read hook."""

    runtime_observation_payloads, runtime_observation_source_path = (
        _relaykv_runtime_req_to_token_payload_source_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
        )
    )
    runtime_observation_source_bridge_state = "not_attempted"
    runtime_observation_source_bridge_payload_count = 0
    runtime_observation_source_bridge_valid_count = 0
    runtime_observation_source_bridge_source_path = None
    runtime_observation_source_bridge_blocked_reason = None

    if runtime_observation_payloads is None:
        runtime_observation_bridge_results = (
            build_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
                forward_batch=forward_batch,
                model_runner=model_runner,
                bridge_enabled=True,
            )
        )
        runtime_observation_bridge_summary = (
            summarize_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
                runtime_observation_bridge_results
            )
        )
        runtime_observation_source_bridge_state = str(
            runtime_observation_bridge_summary.get("bridge_state") or "blocked"
        )
        runtime_observation_source_bridge_payload_count = int(
            runtime_observation_bridge_summary.get("payload_count") or 0
        )
        runtime_observation_source_bridge_valid_count = int(
            runtime_observation_bridge_summary.get("valid_payload_count") or 0
        )
        runtime_observation_source_bridge_source_path = (
            runtime_observation_bridge_summary.get("bridge_source_path")
        )
        runtime_observation_source_bridge_blocked_reason = (
            runtime_observation_bridge_summary.get("blocked_reason")
        )
        if runtime_observation_bridge_results:
            first_result = runtime_observation_bridge_results[0]
            if isinstance(first_result, Mapping):
                bridged_payloads = first_result.get("runtime_observation_payloads")
                if (
                    runtime_observation_source_bridge_state == "bridged"
                    and isinstance(bridged_payloads, list)
                    and bridged_payloads
                ):
                    runtime_observation_payloads = bridged_payloads
                    runtime_observation_source_path = (
                        runtime_observation_source_bridge_source_path
                    )
    else:
        runtime_observation_source_bridge_state = "not_needed"

    req_to_token_pool_object, req_to_token_pool_source_path = (
        _relaykv_real_req_to_token_pool_source_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
        )
    )
    read_results = build_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        req_to_token_pool_object=req_to_token_pool_object,
        read_req_to_token_pool=True,
        max_tokens_per_request=256,
        max_total_tokens=1024,
        source_path=req_to_token_pool_source_path,
    )
    summary = summarize_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
        read_results,
        read_enabled=True,
        max_tokens_per_request=256,
        max_total_tokens=1024,
    )

    payloads = build_relaykv_req_to_token_resolution_payloads_from_real_pool_read_for_smoke(
        read_results
    )
    resolved_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, Mapping)
        and payload.get("resolution_state") == "req_to_token_resolved"
    ]

    req_to_token_payload_attached = False
    req_to_token_payload_attach_target = None
    relaykv_payload_attr_write_count = 0
    if resolved_payloads:
        for owner, target in (
            (forward_batch, "forward_batch.relaykv_req_to_token_resolution_payloads"),
            (model_runner, "model_runner.relaykv_req_to_token_resolution_payloads"),
        ):
            if owner is None:
                continue
            try:
                setattr(owner, "relaykv_req_to_token_resolution_payloads", payloads)
            except Exception:
                continue
            req_to_token_payload_attached = True
            req_to_token_payload_attach_target = target
            relaykv_payload_attr_write_count = len(payloads)
            break

    blocked_reason = None
    for result in read_results:
        if isinstance(result, Mapping):
            blocked_reason = result.get("blocked_reason")
            if blocked_reason is not None:
                break

    summary["hook_enabled"] = True
    summary["req_to_token_pool_source_path"] = req_to_token_pool_source_path
    summary["runtime_observation_payload_source_path"] = (
        runtime_observation_source_path
    )
    summary["runtime_observation_source_bridge_state"] = (
        runtime_observation_source_bridge_state
    )
    summary["runtime_observation_source_bridge_payload_count"] = (
        runtime_observation_source_bridge_payload_count
    )
    summary["runtime_observation_source_bridge_valid_count"] = (
        runtime_observation_source_bridge_valid_count
    )
    summary["runtime_observation_source_bridge_source_path"] = (
        runtime_observation_source_bridge_source_path
    )
    summary["runtime_observation_source_bridge_blocked_reason"] = (
        runtime_observation_source_bridge_blocked_reason
    )
    summary["req_to_token_payload_attached"] = req_to_token_payload_attached
    summary["req_to_token_payload_attach_target"] = req_to_token_payload_attach_target
    summary["relaykv_payload_attr_write_count"] = relaykv_payload_attr_write_count
    summary["payload_count"] = len(payloads)
    summary["blocked_reason"] = blocked_reason

    return {
        "read_results": read_results,
        "payloads": payloads,
        "summary": summary,
    }


def run_model_runner_req_to_token_pool_value_shape_inspection_hook_for_smoke(
    model_runner: Any,
    forward_batch: Any = None,
) -> dict[str, Any]:
    """Run a smoke-only req_to_token_pool value-shape inspection hook."""

    runtime_observation_payloads, runtime_observation_source_path = (
        _relaykv_runtime_req_to_token_payload_source_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
        )
    )
    runtime_observation_source_bridge_state = "not_attempted"
    runtime_observation_source_bridge_payload_count = 0
    runtime_observation_source_bridge_valid_count = 0
    runtime_observation_source_bridge_source_path = None
    runtime_observation_source_bridge_blocked_reason = None

    if runtime_observation_payloads is None:
        runtime_observation_bridge_results = (
            build_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
                forward_batch=forward_batch,
                model_runner=model_runner,
                bridge_enabled=True,
            )
        )
        runtime_observation_bridge_summary = (
            summarize_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
                runtime_observation_bridge_results
            )
        )
        runtime_observation_source_bridge_state = str(
            runtime_observation_bridge_summary.get("bridge_state") or "blocked"
        )
        runtime_observation_source_bridge_payload_count = int(
            runtime_observation_bridge_summary.get("payload_count") or 0
        )
        runtime_observation_source_bridge_valid_count = int(
            runtime_observation_bridge_summary.get("valid_payload_count") or 0
        )
        runtime_observation_source_bridge_source_path = (
            runtime_observation_bridge_summary.get("bridge_source_path")
        )
        runtime_observation_source_bridge_blocked_reason = (
            runtime_observation_bridge_summary.get("blocked_reason")
        )
        if runtime_observation_bridge_results:
            first_result = runtime_observation_bridge_results[0]
            if isinstance(first_result, Mapping):
                bridged_payloads = first_result.get("runtime_observation_payloads")
                if (
                    runtime_observation_source_bridge_state == "bridged"
                    and isinstance(bridged_payloads, list)
                    and bridged_payloads
                ):
                    runtime_observation_payloads = bridged_payloads
                    runtime_observation_source_path = (
                        runtime_observation_source_bridge_source_path
                    )
    else:
        runtime_observation_source_bridge_state = "not_needed"

    req_to_token_pool_object, req_to_token_pool_source_path = (
        _relaykv_real_req_to_token_pool_source_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
        )
    )
    inspection_results = (
        build_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke(
            runtime_observation_payloads=runtime_observation_payloads,
            req_to_token_pool_object=req_to_token_pool_object,
            inspect_req_to_token_pool_value_shape=True,
            max_tokens_per_request=8,
            max_total_tokens=16,
            source_path=req_to_token_pool_source_path,
        )
    )
    summary = summarize_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke(
        inspection_results,
        inspection_enabled=True,
        max_tokens_per_request=8,
        max_total_tokens=16,
    )

    blocked_reason = None
    for result in inspection_results:
        if isinstance(result, Mapping):
            blocked_reason = result.get("blocked_reason")
            if blocked_reason is not None:
                break

    summary["hook_enabled"] = True
    summary["req_to_token_pool_source_path"] = req_to_token_pool_source_path
    summary["runtime_observation_payload_source_path"] = (
        runtime_observation_source_path
    )
    summary["runtime_observation_source_bridge_state"] = (
        runtime_observation_source_bridge_state
    )
    summary["runtime_observation_source_bridge_payload_count"] = (
        runtime_observation_source_bridge_payload_count
    )
    summary["runtime_observation_source_bridge_valid_count"] = (
        runtime_observation_source_bridge_valid_count
    )
    summary["runtime_observation_source_bridge_source_path"] = (
        runtime_observation_source_bridge_source_path
    )
    summary["runtime_observation_source_bridge_blocked_reason"] = (
        runtime_observation_source_bridge_blocked_reason
    )
    summary["blocked_reason"] = blocked_reason

    return {
        "inspection_results": inspection_results,
        "summary": summary,
    }


def _relaykv_runtime_kv_index_resolution_plans_for_smoke(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
) -> tuple[Any, str | None]:
    for owner, path in (
        (forward_batch, "relaykv_kv_index_resolution_plans"),
        (model_runner, "relaykv_kv_index_resolution_plans"),
    ):
        if owner is None:
            continue
        try:
            value = getattr(owner, path, None)
        except Exception:
            continue
        if value is None:
            continue
        owner_name = "forward_batch" if owner is forward_batch else "model_runner"
        if isinstance(value, Mapping):
            return [value], f"{owner_name}.{path}"
        return value, f"{owner_name}.{path}"
    return None, None


def _relaykv_runtime_explicit_req_to_token_entries_for_smoke(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
) -> tuple[Any, str | None]:
    for owner, path in (
        (forward_batch, "relaykv_explicit_req_to_token_entries_for_smoke"),
        (model_runner, "relaykv_explicit_req_to_token_entries_for_smoke"),
    ):
        if owner is None:
            continue
        try:
            value = getattr(owner, path, None)
        except Exception:
            continue
        if value is not None:
            owner_name = "forward_batch" if owner is forward_batch else "model_runner"
            return value, f"{owner_name}.{path}"
    return None, None


def run_model_runner_runtime_req_to_token_payload_production_hook_for_smoke(
    model_runner: Any,
    forward_batch: Any = None,
) -> dict[str, Any]:
    """Run a smoke-only runtime req_to_token payload production hook."""

    runtime_observation_payloads, runtime_observation_source_path = (
        _relaykv_runtime_req_to_token_payload_source_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
        )
    )
    kv_index_resolution_plans, kv_index_resolution_plan_source_path = (
        _relaykv_runtime_kv_index_resolution_plans_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
        )
    )
    explicit_req_to_token_entries, explicit_req_to_token_entries_source_path = (
        _relaykv_runtime_explicit_req_to_token_entries_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
        )
    )
    explicit_req_to_token_entries_present = (
        explicit_req_to_token_entries_source_path is not None
    )
    metadata_derived_entries_enabled = (
        os.getenv("SGLANG_RELAYKV_RUNTIME_METADATA_DERIVED_REQ_TO_TOKEN_ENTRIES")
        == "1"
    )
    metadata_derived_entries_state = "not_attempted"
    metadata_derived_entry_count = 0
    metadata_derived_result_count = 0
    metadata_derived_blocked_count = 0
    metadata_derived_source_path = None
    metadata_derived_blocked_reason = None
    runtime_observation_source_bridge_enabled = metadata_derived_entries_enabled
    runtime_observation_source_bridge_state = "not_attempted"
    runtime_observation_source_bridge_payload_count = 0
    runtime_observation_source_bridge_valid_count = 0
    runtime_observation_source_bridge_source_path = None
    runtime_observation_source_bridge_blocked_reason = None
    req_to_token_entry_source = "none"

    effective_req_to_token_entries = explicit_req_to_token_entries
    if explicit_req_to_token_entries_present:
        req_to_token_entry_source = "explicit"
    elif metadata_derived_entries_enabled:
        if runtime_observation_payloads is None:
            runtime_observation_bridge_results = (
                build_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
                    forward_batch=forward_batch,
                    model_runner=model_runner,
                    bridge_enabled=True,
                )
            )
            runtime_observation_bridge_summary = (
                summarize_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
                    runtime_observation_bridge_results
                )
            )
            runtime_observation_source_bridge_state = str(
                runtime_observation_bridge_summary.get("bridge_state") or "blocked"
            )
            runtime_observation_source_bridge_payload_count = int(
                runtime_observation_bridge_summary.get("payload_count") or 0
            )
            runtime_observation_source_bridge_valid_count = int(
                runtime_observation_bridge_summary.get("valid_payload_count") or 0
            )
            runtime_observation_source_bridge_source_path = (
                runtime_observation_bridge_summary.get("bridge_source_path")
            )
            runtime_observation_source_bridge_blocked_reason = (
                runtime_observation_bridge_summary.get("blocked_reason")
            )
            bridge_payloads = []
            if runtime_observation_bridge_results:
                first_result = runtime_observation_bridge_results[0]
                if isinstance(first_result, Mapping):
                    bridge_payloads = first_result.get("runtime_observation_payloads") or []
            if (
                runtime_observation_source_bridge_state == "bridged"
                and isinstance(bridge_payloads, list)
                and bridge_payloads
            ):
                runtime_observation_payloads = bridge_payloads
                runtime_observation_source_path = (
                    runtime_observation_source_bridge_source_path
                )
        else:
            runtime_observation_source_bridge_state = "not_needed"
            runtime_observation_source_bridge_payload_count = 0
            runtime_observation_source_bridge_valid_count = 0
            runtime_observation_source_bridge_source_path = None
            runtime_observation_source_bridge_blocked_reason = None

        metadata_derived_results = (
            build_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
                runtime_observation_payloads=runtime_observation_payloads,
                kv_index_resolution_plans=kv_index_resolution_plans,
                production_enabled=True,
                max_tokens_per_request=256,
                max_total_tokens=1024,
            )
        )
        metadata_derived_summary = (
            summarize_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
                metadata_derived_results,
                production_enabled=True,
                max_tokens_per_request=256,
                max_total_tokens=1024,
            )
        )
        metadata_derived_result_count = len(metadata_derived_results)
        metadata_derived_blocked_count = int(
            metadata_derived_summary.get("blocked_count") or 0
        )
        derived_entries_found = None
        for result in metadata_derived_results:
            if not isinstance(result, Mapping):
                continue
            derived_entries = result.get("derived_req_to_token_entries")
            if result.get("derivation_state") == "derived" and isinstance(
                derived_entries, list
            ):
                derived_entries_found = list(derived_entries)
                metadata_derived_entry_count = len(derived_entries_found)
                metadata_derived_entries_state = "derived"
                metadata_derived_blocked_reason = None
                adapter_metadata = result.get("adapter_metadata")
                if isinstance(adapter_metadata, Mapping):
                    metadata_derived_source_path = adapter_metadata.get(
                        "runtime_metadata_derivation_source_path"
                    )
                if metadata_derived_source_path is None:
                    metadata_derived_source_path = runtime_observation_source_path
                effective_req_to_token_entries = derived_entries_found
                req_to_token_entry_source = "metadata_derived"
                break
        if derived_entries_found is None:
            metadata_derived_entries_state = "blocked"
            for result in metadata_derived_results:
                if not isinstance(result, Mapping):
                    continue
                metadata_derived_blocked_reason = result.get("blocked_reason")
                adapter_metadata = result.get("adapter_metadata")
                if isinstance(adapter_metadata, Mapping):
                    metadata_derived_source_path = adapter_metadata.get(
                        "runtime_metadata_derivation_source_path"
                    )
                if metadata_derived_source_path is None:
                    metadata_derived_source_path = runtime_observation_source_path
                break

    payloads = build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        kv_index_resolution_plans=kv_index_resolution_plans,
        explicit_req_to_token_entries=effective_req_to_token_entries,
        production_enabled=True,
        max_tokens_per_request=256,
        max_total_tokens=1024,
    )
    payload_attached = False
    payload_attach_target = None
    relaykv_payload_attr_write_count = 0
    if any(
        payload.get("resolution_state") == "req_to_token_resolved"
        for payload in payloads
        if isinstance(payload, Mapping)
    ):
        for owner, target in (
            (forward_batch, "forward_batch.relaykv_req_to_token_resolution_payloads"),
            (model_runner, "model_runner.relaykv_req_to_token_resolution_payloads"),
        ):
            if owner is None:
                continue
            try:
                setattr(owner, "relaykv_req_to_token_resolution_payloads", payloads)
            except Exception:
                continue
            payload_attached = True
            payload_attach_target = target
            relaykv_payload_attr_write_count = len(payloads)
            break

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        adapter_metadata = payload.get("adapter_metadata")
        if not isinstance(adapter_metadata, dict):
            adapter_metadata = {}
            payload["adapter_metadata"] = adapter_metadata
        adapter_metadata["runtime_observation_payload_source_path"] = (
            runtime_observation_source_path
        )
        adapter_metadata["kv_index_resolution_plan_source_path"] = (
            kv_index_resolution_plan_source_path
        )
        adapter_metadata["explicit_req_to_token_entries_source_path"] = (
            explicit_req_to_token_entries_source_path
        )
        adapter_metadata["metadata_derived_entries_enabled"] = (
            metadata_derived_entries_enabled
        )
        adapter_metadata["metadata_derived_entries_state"] = (
            metadata_derived_entries_state
        )
        adapter_metadata["metadata_derived_entry_count"] = (
            metadata_derived_entry_count
        )
        adapter_metadata["metadata_derived_result_count"] = (
            metadata_derived_result_count
        )
        adapter_metadata["metadata_derived_blocked_count"] = (
            metadata_derived_blocked_count
        )
        adapter_metadata["metadata_derived_source_path"] = (
            metadata_derived_source_path
        )
        adapter_metadata["metadata_derived_blocked_reason"] = (
            metadata_derived_blocked_reason
        )
        adapter_metadata["runtime_observation_source_bridge_enabled"] = (
            runtime_observation_source_bridge_enabled
        )
        adapter_metadata["runtime_observation_source_bridge_state"] = (
            runtime_observation_source_bridge_state
        )
        adapter_metadata["runtime_observation_source_bridge_payload_count"] = (
            runtime_observation_source_bridge_payload_count
        )
        adapter_metadata["runtime_observation_source_bridge_valid_count"] = (
            runtime_observation_source_bridge_valid_count
        )
        adapter_metadata["runtime_observation_source_bridge_source_path"] = (
            runtime_observation_source_bridge_source_path
        )
        adapter_metadata["runtime_observation_source_bridge_blocked_reason"] = (
            runtime_observation_source_bridge_blocked_reason
        )
        adapter_metadata["explicit_req_to_token_entries_present"] = (
            explicit_req_to_token_entries_present
        )
        adapter_metadata["req_to_token_entry_source"] = req_to_token_entry_source
        adapter_metadata["payload_attached"] = payload_attached
        adapter_metadata["payload_attach_target"] = payload_attach_target

    summary = summarize_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        payloads,
        production_enabled=True,
        max_tokens_per_request=256,
        max_total_tokens=1024,
    )
    summary["payload_attached"] = payload_attached
    summary["payload_attach_target"] = payload_attach_target
    summary["relaykv_payload_attr_write_count"] = relaykv_payload_attr_write_count
    summary["runtime_observation_payload_source_path"] = (
        runtime_observation_source_path
    )
    summary["kv_index_resolution_plan_source_path"] = (
        kv_index_resolution_plan_source_path
    )
    summary["explicit_req_to_token_entries_source_path"] = (
        explicit_req_to_token_entries_source_path
    )
    summary["metadata_derived_entries_enabled"] = metadata_derived_entries_enabled
    summary["metadata_derived_entries_state"] = metadata_derived_entries_state
    summary["metadata_derived_entry_count"] = metadata_derived_entry_count
    summary["metadata_derived_result_count"] = metadata_derived_result_count
    summary["metadata_derived_blocked_count"] = metadata_derived_blocked_count
    summary["metadata_derived_source_path"] = metadata_derived_source_path
    summary["metadata_derived_blocked_reason"] = metadata_derived_blocked_reason
    summary["runtime_observation_source_bridge_enabled"] = (
        runtime_observation_source_bridge_enabled
    )
    summary["runtime_observation_source_bridge_state"] = (
        runtime_observation_source_bridge_state
    )
    summary["runtime_observation_source_bridge_payload_count"] = (
        runtime_observation_source_bridge_payload_count
    )
    summary["runtime_observation_source_bridge_valid_count"] = (
        runtime_observation_source_bridge_valid_count
    )
    summary["runtime_observation_source_bridge_source_path"] = (
        runtime_observation_source_bridge_source_path
    )
    summary["runtime_observation_source_bridge_blocked_reason"] = (
        runtime_observation_source_bridge_blocked_reason
    )
    summary["explicit_req_to_token_entries_present"] = (
        explicit_req_to_token_entries_present
    )
    summary["req_to_token_entry_source"] = req_to_token_entry_source
    return {"payloads": payloads, "summary": summary}


def _relaykv_req_to_token_resolution_payload_bridge_source(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
    explicit_payloads: Any = None,
) -> tuple[Any, str | None]:
    if explicit_payloads is not None:
        return explicit_payloads, "explicit_payloads"

    for owner, path in (
        (forward_batch, "relaykv_req_to_token_resolution_results"),
        (forward_batch, "relaykv_req_to_token_resolution_payloads"),
        (model_runner, "relaykv_req_to_token_resolution_results"),
        (model_runner, "relaykv_req_to_token_resolution_payloads"),
    ):
        if owner is None:
            continue
        value = getattr(owner, path, None)
        if value is not None:
            owner_name = "forward_batch" if owner is forward_batch else "model_runner"
            return value, f"{owner_name}.{path}"
    return None, None


def _is_valid_req_to_token_resolution_payload_for_smoke(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("event_type") != "relaykv_req_to_token_resolution_result":
        return False
    if payload.get("resolution_state") != "req_to_token_resolved":
        return False
    spans = payload.get("full_kv_req_to_token_spans")
    if not isinstance(spans, (list, tuple)) or not spans:
        return False
    for span in spans:
        if not isinstance(span, Mapping):
            return False
        entries = _event_value(span, "req_to_token_entries")
        if not isinstance(entries, (list, tuple)) or not entries:
            return False
    return True


def build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
    explicit_payloads: Any = None,
    bridge_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Build shallow req_to_token resolution bridge payloads for smoke only."""

    source_value, source_path = _relaykv_req_to_token_resolution_payload_bridge_source(
        forward_batch=forward_batch,
        model_runner=model_runner,
        explicit_payloads=explicit_payloads,
    )

    blocked_reason = None
    bridge_state = "bridged"
    valid_payloads: list[dict[str, Any]] = []
    payload_count = 0
    blocked_payload_count = 0

    if bridge_enabled is not True:
        bridge_state = "blocked"
        blocked_reason = "bridge_not_enabled"
    elif source_value is None:
        bridge_state = "blocked"
        blocked_reason = "bridge_source_missing"
    elif not isinstance(source_value, (list, tuple)):
        bridge_state = "blocked"
        blocked_reason = "bridge_source_not_list_or_tuple"
    elif len(source_value) == 0:
        bridge_state = "blocked"
        blocked_reason = "bridge_source_empty"
    else:
        payload_count = len(source_value)
        for payload in source_value:
            if _is_valid_req_to_token_resolution_payload_for_smoke(payload):
                valid_payloads.append(_copy_relaykv_metadata_value_for_smoke(dict(payload)))
            else:
                blocked_payload_count += 1
        if not valid_payloads:
            bridge_state = "blocked"
            blocked_reason = "bridge_payload_invalid"
        elif blocked_payload_count > 0:
            bridge_state = "bridged"

    result = {
        "event_type": "relaykv_req_to_token_resolution_payload_bridge_result",
        "bridge_state": bridge_state,
        "bridge_mode": "runtime_payload_bridge",
        "payload_count": payload_count,
        "valid_payload_count": len(valid_payloads),
        "blocked_payload_count": blocked_payload_count,
        "bridge_source_path": source_path,
        "blocked_reason": blocked_reason,
        "req_to_token_resolution_payloads": valid_payloads,
        "source_mutated": False,
        "req_to_token_read_count": 0,
        "actual_req_to_token_pool_read_count": 0,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
        "tensor_read_count": 0,
        "attention_comparison_executed_count": 0,
        "attention_override_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "kv_cache_mutation_true_count": 0,
        "source_mutated_true_count": 0,
    }
    return [result]


def summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
    bridge_results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize shallow req_to_token resolution bridge payloads for smoke only."""

    if not isinstance(bridge_results, (list, tuple)):
        raise TypeError(
            "RelayKV req_to_token resolution bridge results must be a list or tuple"
        )

    per_bridge_state: Counter[str] = Counter()
    per_source_path: Counter[str] = Counter()
    totals: Counter[str] = Counter(
        {
            "payload_count": 0,
            "valid_payload_count": 0,
            "blocked_payload_count": 0,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    bridged_count = 0
    blocked_count = 0
    error_count = 0

    for result in bridge_results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV req_to_token resolution bridge result must be a dict"
            )
        state = str(result.get("bridge_state") or "unknown")
        per_bridge_state[state] += 1
        per_source_path[str(result.get("bridge_source_path"))] += 1
        if state == "bridged":
            bridged_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1
        for key in totals:
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value

    return {
        "summary_type": "relaykv_req_to_token_resolution_payload_bridge_summary",
        "total_bridge_results": len(bridge_results),
        "bridged_count": bridged_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "per_bridge_state_counts": dict(sorted(per_bridge_state.items())),
        "per_source_path_counts": dict(sorted(per_source_path.items())),
        **{key: totals[key] for key in sorted(totals)},
    }


def build_relaykv_physical_kv_index_resolution_results_for_smoke(
    req_to_token_resolution_results: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
    token_to_kv_pool_table: Any = None,
    read_token_to_kv_pool: bool = False,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 512,
    max_preview_entries: int = 8,
) -> list[dict[str, Any]]:
    """Build bounded readonly synthetic physical KV index resolution results."""

    if not isinstance(req_to_token_resolution_results, (list, tuple)):
        raise TypeError("req_to_token_resolution_results must be a list or tuple")
    total_requested_token_count = 0
    results: list[dict[str, Any]] = []

    for result in req_to_token_resolution_results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV physical KV index resolution inputs must be dict results"
            )

        blocking_reasons: list[str] = []
        warning_reasons = [
            "readonly_synthetic_token_to_kv_pool_resolution",
            "no_kv_pool_read",
            "preview_only_no_full_indices_logged",
        ]

        full_kv_req_to_token_spans = _event_value(result, "full_kv_req_to_token_spans")
        if _event_value(result, "event_type") != "relaykv_req_to_token_resolution_result":
            blocking_reasons.append("not_req_to_token_resolution_result")
        if _event_value(result, "resolution_state") != "req_to_token_resolved":
            blocking_reasons.append("req_to_token_resolution_not_resolved")
        if read_token_to_kv_pool is not True:
            blocking_reasons.append("read_token_to_kv_pool_not_enabled")
        if read_token_to_kv_pool and token_to_kv_pool_table is None:
            blocking_reasons.append("token_to_kv_pool_table_missing")
        if read_token_to_kv_pool and token_to_kv_pool_table is not None and not isinstance(
            token_to_kv_pool_table, (dict, list, tuple)
        ):
            blocking_reasons.append("token_to_kv_pool_table_not_indexable")
        if not isinstance(full_kv_req_to_token_spans, (list, tuple)):
            blocking_reasons.append("req_to_token_entries_missing")
        elif not full_kv_req_to_token_spans:
            blocking_reasons.append("req_to_token_entries_missing")
        if _event_value(result, "kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if _event_value(result, "kv_snapshot") is True:
            blocking_reasons.append("kv_snapshot_not_allowed")
        if _event_value(result, "tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if _event_value(result, "attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if _event_value(result, "runtime_writeback") is True:
            blocking_reasons.append("runtime_writeback_not_allowed")
        if _event_value(result, "scheduler_policy_noop") is False:
            blocking_reasons.append("scheduler_mutation_not_allowed")
        if _event_value(result, "attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if _event_value(result, "kv_cache_mutation") is True:
            blocking_reasons.append("kv_cache_mutation_not_allowed")
        if _event_value(result, "source_mutated") is True:
            blocking_reasons.append("source_mutated_not_allowed")

        requested_token_count = 0
        if not blocking_reasons:
            assert isinstance(full_kv_req_to_token_spans, (list, tuple))
            for span in full_kv_req_to_token_spans:
                if not isinstance(span, Mapping):
                    blocking_reasons.append("req_to_token_entries_missing")
                    break
                req_to_token_entries = _event_value(span, "req_to_token_entries")
                if not isinstance(req_to_token_entries, (list, tuple)):
                    blocking_reasons.append("req_to_token_entries_missing")
                    break
                requested_token_count += len(req_to_token_entries)

        if requested_token_count > max_tokens_per_request:
            blocking_reasons.append("max_tokens_per_request_exceeded")
        if total_requested_token_count + requested_token_count > max_total_tokens:
            blocking_reasons.append("max_total_tokens_exceeded")

        blocking_reasons = list(dict.fromkeys(blocking_reasons))
        if blocking_reasons:
            results.append(
                _physical_kv_index_blocked_result_for_smoke(
                    result,
                    blocking_reasons=blocking_reasons,
                    warning_reasons=warning_reasons,
                )
            )
            continue

        assert isinstance(full_kv_req_to_token_spans, (list, tuple))
        assert isinstance(token_to_kv_pool_table, (dict, list, tuple))

        physical_kv_indexes, read_blocking_reasons = _read_physical_kv_indexes_for_smoke(
            list(full_kv_req_to_token_spans),
            token_to_kv_pool_table=token_to_kv_pool_table,
        )
        if read_blocking_reasons:
            results.append(
                _physical_kv_index_blocked_result_for_smoke(
                    result,
                    blocking_reasons=read_blocking_reasons,
                    warning_reasons=warning_reasons,
                )
            )
            continue

        total_requested_token_count += requested_token_count
        physical_kv_index_preview = list(physical_kv_indexes[:max_preview_entries])
        normalized_result = normalize_relaykv_sglang_adapter_schema_for_smoke(result)

        existing_engine_block_ref = normalized_result.get("engine_block_ref")
        if isinstance(existing_engine_block_ref, Mapping):
            engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
                dict(existing_engine_block_ref)
            )
        else:
            engine_block_ref = {}

        engine_block_ref["cache_position"] = engine_block_ref.get("cache_position")
        engine_block_ref["token_to_kv_pool_index"] = (
            physical_kv_indexes[0] if len(physical_kv_indexes) == 1 else None
        )
        engine_block_ref["physical_kv_index_preview"] = physical_kv_index_preview
        engine_block_ref["physical_kv_index_count"] = len(physical_kv_indexes)
        engine_block_ref["physical_kv_index_checksum"] = (
            sum(
                (index + 1) * physical_kv_index
                for index, physical_kv_index in enumerate(physical_kv_indexes)
            )
            % 1000000007
        )

        normalized_result.update(
            {
                "event_type": "relaykv_physical_kv_index_resolution_result",
                "resolution_state": "physical_kv_index_resolved",
                "resolution_mode": "readonly_synthetic_table",
                "source": (
                    "req_to_token_resolution_result_to_"
                    "physical_kv_index_resolution_result"
                ),
                "resolved_block_count": _event_value(result, "resolved_block_count"),
                "resolved_token_count": requested_token_count,
                "req_to_token_entry_count": requested_token_count,
                "physical_kv_index_count": len(physical_kv_indexes),
                "physical_kv_index_preview_count": len(physical_kv_index_preview),
                "physical_kv_index_checksum": engine_block_ref[
                    "physical_kv_index_checksum"
                ],
                "truncated_physical_kv_index_preview": (
                    len(physical_kv_indexes) > len(physical_kv_index_preview)
                ),
                "req_to_token_read": False,
                "req_to_token_read_count": 0,
                "token_to_kv_pool_read": True,
                "token_to_kv_pool_read_count": len(physical_kv_indexes),
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "engine_block_ref": engine_block_ref,
                "blocking_reasons": [],
                "warning_reasons": warning_reasons,
            }
        )
        results.append(normalized_result)

    return results


def summarize_relaykv_physical_kv_index_resolution_results_for_smoke(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize bounded readonly synthetic physical KV index resolution results."""

    if not isinstance(results, (list, tuple)):
        raise TypeError(
            "RelayKV physical KV index resolution results must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_resolution_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    physical_kv_index_resolved_count = 0
    blocked_count = 0
    error_count = 0
    physical_kv_index_count = 0
    physical_kv_index_preview_count = 0
    truncated_physical_kv_index_preview_count = 0

    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV physical KV index resolution result must be a dict"
            )

        state = str(_event_value(result, "resolution_state") or "unknown")
        per_resolution_state[state] += 1
        per_request[str(_event_value(result, "request_id"))] += 1
        per_layer[str(_event_layer_value(result))] += 1

        if state == "physical_kv_index_resolved":
            physical_kv_index_resolved_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = _event_value(result, "physical_kv_index_count")
        if isinstance(value, int) and not isinstance(value, bool):
            physical_kv_index_count += value
        value = _event_value(result, "physical_kv_index_preview_count")
        if isinstance(value, int) and not isinstance(value, bool):
            physical_kv_index_preview_count += value
        if _event_value(result, "truncated_physical_kv_index_preview") is True:
            truncated_physical_kv_index_preview_count += 1

        value = _event_value(result, "token_to_kv_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["token_to_kv_pool_read_count"] += value
        if _event_value(result, "kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if _event_value(result, "kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if _event_value(result, "tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if _event_value(result, "attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if _event_value(result, "attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if _event_value(result, "runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if _event_value(result, "scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if _event_value(result, "kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if _event_value(result, "source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_physical_kv_index_resolution_result_summary",
        "total_physical_kv_index_resolution_results": len(results),
        "physical_kv_index_resolved_count": physical_kv_index_resolved_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "physical_kv_index_count": physical_kv_index_count,
        "physical_kv_index_preview_count": physical_kv_index_preview_count,
        "truncated_physical_kv_index_preview_count": (
            truncated_physical_kv_index_preview_count
        ),
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_resolution_state_counts": dict(sorted(per_resolution_state.items())),
        "token_to_kv_pool_read_count": (
            safety_counts["token_to_kv_pool_read_count"]
        ),
        "kv_pool_read_count": safety_counts["kv_pool_read_count"],
        "kv_snapshot_count": safety_counts["kv_snapshot_count"],
        "tensor_read_count": safety_counts["tensor_read_count"],
        "attention_comparison_executed_count": (
            safety_counts["attention_comparison_executed_count"]
        ),
        "attention_override_true_count": (
            safety_counts["attention_override_true_count"]
        ),
        "runtime_writeback_true_count": (
            safety_counts["runtime_writeback_true_count"]
        ),
        "scheduler_policy_noop_false_count": (
            safety_counts["scheduler_policy_noop_false_count"]
        ),
        "kv_cache_mutation_true_count": (
            safety_counts["kv_cache_mutation_true_count"]
        ),
        "source_mutated_true_count": safety_counts["source_mutated_true_count"],
    }


def _blocked_token_to_kv_pool_readonly_adapter_payload_for_smoke(
    result: Mapping[str, Any],
    *,
    blocking_reasons: list[str],
    warning_reasons: list[str],
    token_to_kv_pool_source: str | None,
    token_to_kv_pool_backing_type: str | None,
) -> dict[str, Any]:
    blocked_payload = normalize_relaykv_sglang_adapter_schema_for_smoke(result)
    adapter_metadata = blocked_payload.get("adapter_metadata")
    if isinstance(adapter_metadata, Mapping):
        payload_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
            dict(adapter_metadata)
        )
    else:
        payload_adapter_metadata = {}
    if token_to_kv_pool_source is not None:
        payload_adapter_metadata["pool_source_path"] = token_to_kv_pool_source
    if token_to_kv_pool_backing_type is not None:
        payload_adapter_metadata["token_to_kv_pool_backing_type"] = (
            token_to_kv_pool_backing_type
        )

    existing_engine_block_ref = blocked_payload.get("engine_block_ref")
    if isinstance(existing_engine_block_ref, Mapping):
        engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
            dict(existing_engine_block_ref)
        )
    else:
        engine_block_ref = {}
    engine_block_ref["cache_position"] = engine_block_ref.get("cache_position")
    engine_block_ref["token_to_kv_pool_index"] = None
    engine_block_ref["token_to_kv_pool_index_preview"] = []
    engine_block_ref["physical_kv_index_preview"] = []
    engine_block_ref["physical_kv_index_count"] = 0
    engine_block_ref["physical_kv_index_checksum"] = None

    blocked_payload.update(
        {
            "event_type": "relaykv_token_to_kv_pool_readonly_adapter_payload",
            "adapter_state": "blocked",
            "adapter_mode": "fake_actual_token_to_kv_pool_readonly",
            "source": (
                "req_to_token_resolution_result_to_"
                "token_to_kv_pool_readonly_adapter_payload"
            ),
            "adapter_metadata": payload_adapter_metadata,
            "engine_block_ref": engine_block_ref,
            "requested_token_count": 0,
            "read_token_count": 0,
            "preview_entry_count": 0,
            "entry_count": 0,
            "entry_min": None,
            "entry_max": None,
            "entry_checksum": None,
            "truncated_preview": False,
            "physical_kv_index_count": 0,
            "physical_kv_index_preview_count": 0,
            "physical_kv_index_checksum": None,
            "truncated_physical_kv_index_preview": False,
            "req_to_token_read": False,
            "req_to_token_read_count": 0,
            "actual_token_to_kv_pool_read": False,
            "actual_token_to_kv_pool_read_count": 0,
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read": False,
            "kv_snapshot": False,
            "tensor_read": False,
            "attention_comparison_executed": False,
            "attention_override": False,
            "runtime_writeback": False,
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "source_mutated": False,
            "blocking_reasons": blocking_reasons,
            "warning_reasons": warning_reasons,
        }
    )
    blocked_payload["decision_state"] = blocked_payload.get("decision_state") or "blocked"
    blocked_payload["fallback_reason"] = (
        blocked_payload.get("fallback_reason")
        if blocked_payload.get("fallback_reason") is not None
        else (blocking_reasons[0] if blocking_reasons else None)
    )
    return blocked_payload


def build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
    req_to_token_resolution_results: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
    token_to_kv_pool_pool: Any = None,
    read_token_to_kv_pool: bool = False,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 512,
    max_preview_entries: int = 8,
) -> list[dict[str, Any]]:
    """Build bounded readonly fake-actual token_to_kv_pool adapter payloads."""

    if not isinstance(req_to_token_resolution_results, (list, tuple)):
        raise TypeError("req_to_token_resolution_results must be a list or tuple")

    token_to_kv_pool_source = None
    token_to_kv_pool_backing = None
    token_to_kv_pool_backing_type = None
    pool_blocking_reasons: list[str] = []

    if read_token_to_kv_pool is not True:
        pool_blocking_reasons.append("read_token_to_kv_pool_not_enabled")
    elif token_to_kv_pool_pool is None:
        pool_blocking_reasons.append("token_to_kv_pool_object_missing")
    else:
        try:
            token_to_kv_pool_backing = getattr(
                token_to_kv_pool_pool, "token_to_kv_pool", None
            )
        except Exception:
            pool_blocking_reasons.append("token_to_kv_pool_attr_access_failed")
        else:
            token_to_kv_pool_source = "token_to_kv_pool_pool.token_to_kv_pool"
            if token_to_kv_pool_backing is None:
                pool_blocking_reasons.append("token_to_kv_pool_attr_missing")
            else:
                token_to_kv_pool_backing_type = type(token_to_kv_pool_backing).__name__
                if not isinstance(token_to_kv_pool_backing, (dict, list, tuple)):
                    pool_blocking_reasons.append(
                        "token_to_kv_pool_backing_not_indexable"
                    )

    total_requested_token_count = 0
    payloads: list[dict[str, Any]] = []

    for result in req_to_token_resolution_results:
        if not isinstance(result, dict):
            raise TypeError(
                "RelayKV token_to_kv_pool adapter inputs must be dict results"
            )

        blocking_reasons = list(pool_blocking_reasons)
        warning_reasons = [
            "fake_actual_token_to_kv_pool_readonly_adapter",
            "bounded_preview_only",
            "no_kv_pool_read",
            "preview_only_no_full_indices_logged",
        ]

        full_kv_req_to_token_spans = _event_value(result, "full_kv_req_to_token_spans")
        if _event_value(result, "event_type") != "relaykv_req_to_token_resolution_result":
            blocking_reasons.append("not_req_to_token_resolution_result")
        if _event_value(result, "resolution_state") != "req_to_token_resolved":
            blocking_reasons.append("req_to_token_resolution_not_resolved")
        if _event_value(result, "kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if _event_value(result, "kv_snapshot") is True:
            blocking_reasons.append("kv_snapshot_not_allowed")
        if _event_value(result, "tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if _event_value(result, "attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if _event_value(result, "runtime_writeback") is True:
            blocking_reasons.append("runtime_writeback_not_allowed")
        if _event_value(result, "scheduler_policy_noop") is False:
            blocking_reasons.append("scheduler_mutation_not_allowed")
        if _event_value(result, "attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if _event_value(result, "kv_cache_mutation") is True:
            blocking_reasons.append("kv_cache_mutation_not_allowed")
        if _event_value(result, "source_mutated") is True:
            blocking_reasons.append("source_mutated_not_allowed")
        if not isinstance(full_kv_req_to_token_spans, (list, tuple)):
            blocking_reasons.append("req_to_token_entries_missing")
        elif not full_kv_req_to_token_spans:
            blocking_reasons.append("req_to_token_entries_missing")

        requested_token_count = 0
        if not blocking_reasons:
            assert isinstance(full_kv_req_to_token_spans, (list, tuple))
            for span in full_kv_req_to_token_spans:
                if not isinstance(span, Mapping):
                    blocking_reasons.append("req_to_token_entries_missing")
                    break
                req_to_token_entries = _event_value(span, "req_to_token_entries")
                if not isinstance(req_to_token_entries, (list, tuple)):
                    blocking_reasons.append("req_to_token_entries_missing")
                    break
                requested_token_count += len(req_to_token_entries)

        if requested_token_count > max_tokens_per_request:
            blocking_reasons.append("max_tokens_per_request_exceeded")
        if total_requested_token_count + requested_token_count > max_total_tokens:
            blocking_reasons.append("max_total_tokens_exceeded")

        blocking_reasons = list(dict.fromkeys(blocking_reasons))
        if blocking_reasons:
            payloads.append(
                _blocked_token_to_kv_pool_readonly_adapter_payload_for_smoke(
                    result,
                    blocking_reasons=blocking_reasons,
                    warning_reasons=warning_reasons,
                    token_to_kv_pool_source=token_to_kv_pool_source,
                    token_to_kv_pool_backing_type=token_to_kv_pool_backing_type,
                )
            )
            continue

        assert isinstance(full_kv_req_to_token_spans, (list, tuple))
        assert isinstance(token_to_kv_pool_backing, (dict, list, tuple))

        physical_kv_indexes, read_blocking_reasons = _read_physical_kv_indexes_for_smoke(
            list(full_kv_req_to_token_spans),
            token_to_kv_pool_table=token_to_kv_pool_backing,
        )
        if read_blocking_reasons:
            payloads.append(
                _blocked_token_to_kv_pool_readonly_adapter_payload_for_smoke(
                    result,
                    blocking_reasons=read_blocking_reasons,
                    warning_reasons=warning_reasons,
                    token_to_kv_pool_source=token_to_kv_pool_source,
                    token_to_kv_pool_backing_type=token_to_kv_pool_backing_type,
                )
            )
            continue

        total_requested_token_count += requested_token_count
        normalized_payload = normalize_relaykv_sglang_adapter_schema_for_smoke(result)
        adapter_metadata = normalized_payload.get("adapter_metadata")
        if isinstance(adapter_metadata, Mapping):
            payload_adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
                dict(adapter_metadata)
            )
        else:
            payload_adapter_metadata = {}
        if token_to_kv_pool_source is not None:
            payload_adapter_metadata["pool_source_path"] = token_to_kv_pool_source
        if token_to_kv_pool_backing_type is not None:
            payload_adapter_metadata["token_to_kv_pool_backing_type"] = (
                token_to_kv_pool_backing_type
            )

        existing_engine_block_ref = normalized_payload.get("engine_block_ref")
        if isinstance(existing_engine_block_ref, Mapping):
            engine_block_ref = _copy_relaykv_metadata_value_for_smoke(
                dict(existing_engine_block_ref)
            )
        else:
            engine_block_ref = {}

        preview_entries = list(physical_kv_indexes[:max_preview_entries])
        entry_count = len(physical_kv_indexes)
        engine_block_ref["cache_position"] = engine_block_ref.get("cache_position")
        engine_block_ref["token_to_kv_pool_index"] = (
            physical_kv_indexes[0] if len(physical_kv_indexes) == 1 else None
        )
        engine_block_ref["token_to_kv_pool_index_preview"] = preview_entries
        engine_block_ref["physical_kv_index_preview"] = preview_entries
        engine_block_ref["physical_kv_index_count"] = entry_count
        engine_block_ref["physical_kv_index_checksum"] = (
            sum((index + 1) * entry for index, entry in enumerate(physical_kv_indexes))
            % 1000000007
        )

        normalized_payload.update(
            {
                "event_type": "relaykv_token_to_kv_pool_readonly_adapter_payload",
                "adapter_state": "adapter_payload_ready",
                "adapter_mode": "fake_actual_token_to_kv_pool_readonly",
                "source": (
                    "req_to_token_resolution_result_to_"
                    "token_to_kv_pool_readonly_adapter_payload"
                ),
                "adapter_metadata": payload_adapter_metadata,
                "engine_block_ref": engine_block_ref,
                "requested_token_count": requested_token_count,
                "read_token_count": entry_count,
                "preview_entry_count": len(preview_entries),
                "entry_count": entry_count,
                "entry_min": min(physical_kv_indexes) if physical_kv_indexes else None,
                "entry_max": max(physical_kv_indexes) if physical_kv_indexes else None,
                "entry_checksum": engine_block_ref["physical_kv_index_checksum"],
                "truncated_preview": entry_count > len(preview_entries),
                "physical_kv_index_count": entry_count,
                "physical_kv_index_preview_count": len(preview_entries),
                "physical_kv_index_checksum": engine_block_ref[
                    "physical_kv_index_checksum"
                ],
                "truncated_physical_kv_index_preview": (
                    entry_count > len(preview_entries)
                ),
                "req_to_token_read": False,
                "req_to_token_read_count": 0,
                "actual_token_to_kv_pool_read": True,
                "actual_token_to_kv_pool_read_count": entry_count,
                "token_to_kv_pool_read": True,
                "token_to_kv_pool_read_count": entry_count,
                "kv_pool_read": False,
                "kv_snapshot": False,
                "tensor_read": False,
                "attention_comparison_executed": False,
                "attention_override": False,
                "runtime_writeback": False,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "source_mutated": False,
                "blocking_reasons": [],
                "warning_reasons": warning_reasons,
            }
        )
        payloads.append(normalized_payload)

    return payloads


def summarize_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
    payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize fake-actual token_to_kv_pool adapter payloads for smoke."""

    if not isinstance(payloads, (list, tuple)):
        raise TypeError(
            "RelayKV token_to_kv_pool readonly adapter payloads must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_adapter_state: Counter[str] = Counter()
    per_adapter_mode: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "actual_token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_true_count": 0,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    adapter_payload_ready_count = 0
    blocked_count = 0
    error_count = 0
    requested_token_count = 0
    read_token_count = 0
    preview_entry_count = 0
    physical_kv_index_count = 0
    truncated_preview_count = 0

    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(
                "RelayKV token_to_kv_pool readonly adapter payload must be a dict"
            )

        state = str(payload.get("adapter_state") or "unknown")
        mode = str(payload.get("adapter_mode") or "unknown")
        per_adapter_state[state] += 1
        per_adapter_mode[mode] += 1
        per_request[str(payload.get("request_id"))] += 1
        per_layer[str(payload.get("layer_id"))] += 1

        if state == "adapter_payload_ready":
            adapter_payload_ready_count += 1
        elif state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        value = payload.get("requested_token_count")
        if isinstance(value, int) and not isinstance(value, bool):
            requested_token_count += value
        value = payload.get("read_token_count")
        if isinstance(value, int) and not isinstance(value, bool):
            read_token_count += value
        value = payload.get("preview_entry_count")
        if isinstance(value, int) and not isinstance(value, bool):
            preview_entry_count += value
        value = payload.get("physical_kv_index_count")
        if isinstance(value, int) and not isinstance(value, bool):
            physical_kv_index_count += value
        if payload.get("truncated_preview") is True:
            truncated_preview_count += 1

        value = payload.get("actual_token_to_kv_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["actual_token_to_kv_pool_read_count"] += value
        if payload.get("actual_token_to_kv_pool_read") is True:
            safety_counts["actual_token_to_kv_pool_read_true_count"] += 1
        value = payload.get("token_to_kv_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["token_to_kv_pool_read_count"] += value
        if payload.get("kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if payload.get("kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if payload.get("tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if payload.get("attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if payload.get("attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if payload.get("runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if payload.get("scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if payload.get("kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if payload.get("source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_token_to_kv_pool_readonly_adapter_payload_summary",
        "total_token_to_kv_pool_readonly_adapter_payloads": len(payloads),
        "adapter_payload_ready_count": adapter_payload_ready_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "requested_token_count": requested_token_count,
        "read_token_count": read_token_count,
        "preview_entry_count": preview_entry_count,
        "physical_kv_index_count": physical_kv_index_count,
        "truncated_preview_count": truncated_preview_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_adapter_state_counts": dict(sorted(per_adapter_state.items())),
        "per_adapter_mode_counts": dict(sorted(per_adapter_mode.items())),
        "actual_token_to_kv_pool_read_count": (
            safety_counts["actual_token_to_kv_pool_read_count"]
        ),
        "actual_token_to_kv_pool_read_true_count": (
            safety_counts["actual_token_to_kv_pool_read_true_count"]
        ),
        "token_to_kv_pool_read_count": (
            safety_counts["token_to_kv_pool_read_count"]
        ),
        "kv_pool_read_count": safety_counts["kv_pool_read_count"],
        "kv_snapshot_count": safety_counts["kv_snapshot_count"],
        "tensor_read_count": safety_counts["tensor_read_count"],
        "attention_comparison_executed_count": (
            safety_counts["attention_comparison_executed_count"]
        ),
        "attention_override_true_count": (
            safety_counts["attention_override_true_count"]
        ),
        "runtime_writeback_true_count": (
            safety_counts["runtime_writeback_true_count"]
        ),
        "scheduler_policy_noop_false_count": (
            safety_counts["scheduler_policy_noop_false_count"]
        ),
        "kv_cache_mutation_true_count": (
            safety_counts["kv_cache_mutation_true_count"]
        ),
        "source_mutated_true_count": safety_counts["source_mutated_true_count"],
    }


def _copy_relaykv_metadata_value_for_smoke(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _copy_relaykv_metadata_value_for_smoke(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_relaykv_metadata_value_for_smoke(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_relaykv_metadata_value_for_smoke(item) for item in value)
    return value


def _relaykv_smoke_first_present_value(
    payload: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if value is not None:
                return value
    return None


def _relaykv_smoke_first_span(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for key in (
        "relaykv_working_block_spans",
        "full_kv_block_spans",
        "relaykv_working_req_to_token_spans",
        "full_kv_req_to_token_spans",
    ):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and value:
            first_item = value[0]
            if isinstance(first_item, Mapping):
                return first_item
    return None


def _relaykv_smoke_token_span_from_value(value: Any) -> list[int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], int)
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    ):
        return [value[0], value[1]]
    return None


def _relaykv_smoke_logical_block_id(payload: Mapping[str, Any]) -> Any:
    logical_block_id = _relaykv_smoke_first_present_value(
        payload,
        "logical_block_id",
        "block_id",
    )
    if logical_block_id is not None:
        return logical_block_id

    span = _relaykv_smoke_first_span(payload)
    if span is not None:
        block_id = _relaykv_smoke_first_present_value(span, "logical_block_id", "block_id")
        if block_id is not None:
            return block_id

    for key in ("relaykv_working_kv_block_ids", "full_kv_block_ids"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and value:
            return value[0]
    return None


def _relaykv_smoke_token_span(payload: Mapping[str, Any]) -> list[int] | None:
    token_span = _relaykv_smoke_token_span_from_value(payload.get("token_span"))
    if token_span is not None:
        return token_span

    span = _relaykv_smoke_first_span(payload)
    if span is None:
        return None

    token_start = _relaykv_smoke_first_present_value(span, "token_start", "start_token")
    token_end = _relaykv_smoke_first_present_value(span, "token_end", "end_token")
    if (
        isinstance(token_start, int)
        and not isinstance(token_start, bool)
        and isinstance(token_end, int)
        and not isinstance(token_end, bool)
    ):
        return [token_start, token_end]

    return _relaykv_smoke_token_span_from_value(span.get("token_span"))


def _relaykv_smoke_adapter_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    existing_metadata = payload.get("adapter_metadata")
    if isinstance(existing_metadata, Mapping):
        adapter_metadata = _copy_relaykv_metadata_value_for_smoke(
            dict(existing_metadata)
        )
    else:
        adapter_metadata = {}
    for key in (
        "request_id",
        "req_pool_idx",
        "seq_len",
        "pool_source_path",
        "req_to_token_source",
        "req_to_token_backing_type",
        "req_to_token_shape",
        "req_to_token_device",
        "req_to_token_dtype",
    ):
        if key in payload:
            adapter_metadata[key] = _copy_relaykv_metadata_value_for_smoke(payload[key])
    return adapter_metadata


def _relaykv_smoke_engine_block_ref(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    req_pool_idx = _event_req_pool_idx_value(payload)
    preview_present = "req_to_token_entries_preview" in payload
    cache_position_present = "cache_position" in payload
    existing_block_ref = payload.get("engine_block_ref")
    if (
        req_pool_idx is None
        and not preview_present
        and not cache_position_present
        and not isinstance(existing_block_ref, Mapping)
    ):
        return None

    if isinstance(existing_block_ref, Mapping):
        engine_block_ref = _copy_relaykv_metadata_value_for_smoke(dict(existing_block_ref))
    else:
        engine_block_ref = {}
    engine_block_ref["req_pool_idx"] = _copy_relaykv_metadata_value_for_smoke(
        req_pool_idx
    )
    engine_block_ref["cache_position"] = (
        _copy_relaykv_metadata_value_for_smoke(payload.get("cache_position"))
        if cache_position_present
        else engine_block_ref.get("cache_position")
    )
    if "token_to_kv_pool_index" not in engine_block_ref:
        engine_block_ref["token_to_kv_pool_index"] = None
    if preview_present:
        engine_block_ref["req_to_token_entries_preview"] = (
            _copy_relaykv_metadata_value_for_smoke(
                payload.get("req_to_token_entries_preview")
            )
        )
    return engine_block_ref


def _relaykv_smoke_decision_state(payload: Mapping[str, Any]) -> Any:
    return _relaykv_smoke_first_present_value(
        payload,
        "decision_state",
        "adapter_state",
        "resolution_state",
        "shadow_capture_state",
        "runtime_policy_state",
    )


def _relaykv_smoke_fallback_reason(payload: Mapping[str, Any]) -> Any:
    fallback_reason = _relaykv_smoke_first_present_value(
        payload,
        "fallback_reason",
        "fallback_reason_code",
    )
    if fallback_reason is not None:
        return fallback_reason

    decision_state = _relaykv_smoke_decision_state(payload)
    if decision_state in {"blocked", "fallback_candidate", "fallback"}:
        blocking_reasons = payload.get("blocking_reasons")
        if isinstance(blocking_reasons, (list, tuple)) and blocking_reasons:
            return blocking_reasons[0]
        return "blocked"
    return None


def normalize_relaykv_sglang_adapter_schema_for_smoke(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Add Core/Adapter schema fields for smoke payloads without mutating inputs."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    normalized = _copy_relaykv_metadata_value_for_smoke(dict(payload))
    if not isinstance(normalized, dict):
        raise TypeError("normalized payload must be a dict")

    request_id = _relaykv_smoke_first_present_value(
        payload,
        "engine_request_id",
        "request_id",
    )
    logical_sequence_id = _relaykv_smoke_first_present_value(
        payload,
        "logical_sequence_id",
        "sequence_id",
        "request_id",
    )
    decision_state = _relaykv_smoke_decision_state(payload)

    normalized["engine_name"] = (
        normalized.get("engine_name")
        if normalized.get("engine_name") is not None
        else "sglang"
    )
    normalized["adapter_name"] = (
        normalized.get("adapter_name")
        if normalized.get("adapter_name") is not None
        else "sglang"
    )
    normalized["engine_request_id"] = request_id
    normalized["logical_sequence_id"] = logical_sequence_id
    normalized["logical_block_id"] = _relaykv_smoke_logical_block_id(payload)
    normalized["token_span"] = _relaykv_smoke_token_span(payload)
    normalized["layer_id"] = (
        normalized.get("layer_id")
        if normalized.get("layer_id") is not None
        else _event_layer_value(payload)
    )
    normalized["kv_head_group"] = _relaykv_smoke_first_present_value(
        payload,
        "kv_head_group",
        "kv_group",
        "head_group",
    )
    normalized["kv_class"] = (
        _relaykv_smoke_first_present_value(payload, "kv_class", "kv_cache_class")
        or "UNKNOWN"
    )
    normalized["decision_state"] = decision_state or "SHADOW_ONLY"
    normalized["fallback_reason"] = _relaykv_smoke_fallback_reason(payload)
    normalized["position_check_state"] = (
        normalized.get("position_check_state")
        if normalized.get("position_check_state") is not None
        else "not_checked_metadata_only"
    )
    normalized["attention_mask_mode"] = (
        normalized.get("attention_mask_mode")
        if normalized.get("attention_mask_mode") is not None
        else "unknown"
    )
    normalized["rope_position_consistency"] = (
        normalized.get("rope_position_consistency")
        if normalized.get("rope_position_consistency") is not None
        else "not_checked"
    )
    normalized["adapter_metadata"] = _relaykv_smoke_adapter_metadata(payload)
    normalized["engine_block_ref"] = _relaykv_smoke_engine_block_ref(payload)
    return normalized


def _relaykv_smoke_count_value(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1 if value is True else 0


def summarize_relaykv_sglang_adapter_schema_alignment_for_smoke(
    payloads: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Summarize normalized Core/Adapter schema alignment for smoke payloads."""

    if not isinstance(payloads, (list, tuple)):
        raise TypeError("payloads must be a list or tuple")

    normalized_payloads = [
        normalize_relaykv_sglang_adapter_schema_for_smoke(payload)
        for payload in payloads
    ]

    per_event_type: Counter[str] = Counter()
    per_decision_state: Counter[str] = Counter()
    token_span_present_count = 0
    logical_block_id_present_count = 0
    adapter_metadata_req_pool_idx_count = 0
    engine_block_ref_req_pool_idx_count = 0
    engine_block_ref_present_count = 0
    token_to_kv_pool_index_none_count = 0
    position_check_default_count = 0
    safety_counts: Counter[str] = Counter(
        {
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )

    for payload in normalized_payloads:
        per_event_type[str(payload.get("event_type") or "unknown")] += 1
        per_decision_state[str(payload.get("decision_state") or "unknown")] += 1
        if payload.get("token_span") is not None:
            token_span_present_count += 1
        if payload.get("logical_block_id") is not None:
            logical_block_id_present_count += 1
        adapter_metadata = payload.get("adapter_metadata")
        if isinstance(adapter_metadata, Mapping) and "req_pool_idx" in adapter_metadata:
            adapter_metadata_req_pool_idx_count += 1
        engine_block_ref = payload.get("engine_block_ref")
        if isinstance(engine_block_ref, Mapping):
            engine_block_ref_present_count += 1
            if "req_pool_idx" in engine_block_ref:
                engine_block_ref_req_pool_idx_count += 1
            if engine_block_ref.get("token_to_kv_pool_index") is None:
                token_to_kv_pool_index_none_count += 1
        if payload.get("position_check_state") == "not_checked_metadata_only":
            position_check_default_count += 1

        safety_counts["token_to_kv_pool_read_count"] += _relaykv_smoke_count_value(
            payload, "token_to_kv_pool_read_count"
        )
        safety_counts["kv_pool_read_count"] += _relaykv_smoke_count_value(
            payload, "kv_pool_read_count"
        )
        safety_counts["kv_snapshot_count"] += _relaykv_smoke_count_value(
            payload, "kv_snapshot_count"
        )
        safety_counts["tensor_read_count"] += _relaykv_smoke_count_value(
            payload, "tensor_read_count"
        )
        safety_counts["attention_comparison_executed_count"] += (
            _relaykv_smoke_count_value(payload, "attention_comparison_executed_count")
            + (1 if payload.get("attention_comparison_executed") is True else 0)
        )
        safety_counts["attention_override_true_count"] += (
            _relaykv_smoke_count_value(payload, "attention_override_true_count")
            + (1 if payload.get("attention_override") is True else 0)
        )
        safety_counts["runtime_writeback_true_count"] += (
            _relaykv_smoke_count_value(payload, "runtime_writeback_true_count")
            + (1 if payload.get("runtime_writeback") is True else 0)
        )
        if payload.get("scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        safety_counts["scheduler_policy_noop_false_count"] += _relaykv_smoke_count_value(
            payload, "scheduler_policy_noop_false_count"
        )
        safety_counts["kv_cache_mutation_true_count"] += (
            _relaykv_smoke_count_value(payload, "kv_cache_mutation_true_count")
            + (1 if payload.get("kv_cache_mutation") is True else 0)
        )
        safety_counts["source_mutated_true_count"] += (
            _relaykv_smoke_count_value(payload, "source_mutated_true_count")
            + (1 if payload.get("source_mutated") is True else 0)
        )

    return {
        "summary_type": "relaykv_sglang_adapter_schema_alignment_summary",
        "total_payload_count": len(normalized_payloads),
        "normalized_payload_count": len(normalized_payloads),
        "token_span_present_count": token_span_present_count,
        "logical_block_id_present_count": logical_block_id_present_count,
        "adapter_metadata_req_pool_idx_count": adapter_metadata_req_pool_idx_count,
        "engine_block_ref_present_count": engine_block_ref_present_count,
        "engine_block_ref_req_pool_idx_count": engine_block_ref_req_pool_idx_count,
        "token_to_kv_pool_index_none_count": token_to_kv_pool_index_none_count,
        "position_check_default_count": position_check_default_count,
        "per_event_type_counts": dict(sorted(per_event_type.items())),
        "per_decision_state_counts": dict(sorted(per_decision_state.items())),
        **dict(safety_counts),
    }


def build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
    forward_batch_like: Any = None,
    req_to_token_pool: Any = None,
    inspect_req_to_token: bool = False,
) -> list[dict[str, Any]]:
    """Build metadata-only req_to_token runtime inspection payloads for smoke."""

    request_id = None
    layer_id = None
    batch_id = None
    blocking_reasons: list[str] = []

    if forward_batch_like is not None and isinstance(forward_batch_like, dict):
        request_id = forward_batch_like.get("request_id")
        layer_id = forward_batch_like.get("layer_id")
        batch_id = forward_batch_like.get("batch_id")
        if forward_batch_like.get("token_to_kv_pool_read") is True:
            blocking_reasons.append("token_to_kv_pool_read_not_allowed")
        if forward_batch_like.get("kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if forward_batch_like.get("tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if forward_batch_like.get("attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if forward_batch_like.get("attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")

    req_to_token_attr_present = False
    req_to_token_attr_observed = False
    req_to_token_type = None
    req_to_token_module = None
    req_to_token_qualname = None
    req_to_token_shape = None
    req_to_token_device = None
    req_to_token_dtype = None

    if inspect_req_to_token is not True:
        blocking_reasons.append("inspect_req_to_token_not_enabled")
    elif req_to_token_pool is None:
        blocking_reasons.append("req_to_token_pool_missing")
    else:
        try:
            req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        except Exception:
            blocking_reasons.append("req_to_token_attr_access_failed")
            req_to_token = None
        else:
            if req_to_token is None:
                blocking_reasons.append("req_to_token_attr_missing")
            else:
                req_to_token_attr_present = True
                req_to_token_attr_observed = True
                req_to_token_type = type(req_to_token).__name__
                req_to_token_module = type(req_to_token).__module__
                req_to_token_qualname = type(req_to_token).__qualname__
                req_to_token_shape = getattr(req_to_token, "shape", None)
                req_to_token_device = getattr(req_to_token, "device", None)
                req_to_token_dtype = getattr(req_to_token, "dtype", None)

    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    inspection_state = "metadata_observed" if not blocking_reasons else "blocked"

    payload = {
        "event_type": "relaykv_req_to_token_runtime_inspection_payload",
        "inspection_state": inspection_state,
        "inspection_mode": "metadata_only",
        "source": "req_to_token_pool_to_runtime_inspection_payload",
        "request_id": request_id,
        "layer_id": layer_id,
        "batch_id": batch_id,
        "metadata_observed": not blocking_reasons,
        "req_to_token_attr_present": req_to_token_attr_present,
        "req_to_token_attr_observed": req_to_token_attr_observed,
        "actual_req_to_token_pool_inspection": not blocking_reasons,
        "req_to_token_type": req_to_token_type,
        "req_to_token_module": req_to_token_module,
        "req_to_token_qualname": req_to_token_qualname,
        "req_to_token_shape": req_to_token_shape,
        "req_to_token_device": req_to_token_device,
        "req_to_token_dtype": req_to_token_dtype,
        "req_to_token_read": False,
        "req_to_token_read_count": 0,
        "actual_req_to_token_pool_read": False,
        "actual_req_to_token_pool_read_count": 0,
        "token_to_kv_pool_read": False,
        "token_to_kv_pool_read_count": 0,
        "kv_pool_read": False,
        "kv_snapshot": False,
        "tensor_read": False,
        "attention_comparison_executed": False,
        "attention_override": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": [
            "req_to_token_metadata_only_inspection",
            "no_req_to_token_values_read",
            "no_token_to_kv_pool_read",
        ],
    }
    return [payload]


def summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
    payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize metadata-only req_to_token runtime inspection payloads."""

    if not isinstance(payloads, (list, tuple)):
        raise TypeError(
            "RelayKV req_to_token runtime inspection payloads must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_inspection_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "metadata_observed_count": 0,
            "req_to_token_attr_present_count": 0,
            "actual_req_to_token_pool_inspection_count": 0,
            "req_to_token_attr_observed_count": 0,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    blocked_count = 0
    error_count = 0

    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(
                "RelayKV req_to_token runtime inspection payload must be a dict"
            )

        state = str(payload.get("inspection_state") or "unknown")
        per_inspection_state[state] += 1
        per_request[str(payload.get("request_id"))] += 1
        per_layer[str(payload.get("layer_id"))] += 1

        if state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        if payload.get("metadata_observed") is True:
            safety_counts["metadata_observed_count"] += 1
        if payload.get("req_to_token_attr_present") is True:
            safety_counts["req_to_token_attr_present_count"] += 1
        if payload.get("actual_req_to_token_pool_inspection") is True:
            safety_counts["actual_req_to_token_pool_inspection_count"] += 1
        if payload.get("req_to_token_attr_observed") is True:
            safety_counts["req_to_token_attr_observed_count"] += 1

        value = payload.get("req_to_token_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["req_to_token_read_count"] += value
        value = payload.get("actual_req_to_token_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["actual_req_to_token_pool_read_count"] += value
        value = payload.get("token_to_kv_pool_read_count")
        if isinstance(value, int) and not isinstance(value, bool):
            safety_counts["token_to_kv_pool_read_count"] += value
        if payload.get("kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if payload.get("kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if payload.get("tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if payload.get("attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if payload.get("attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if payload.get("runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if payload.get("scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if payload.get("kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if payload.get("source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_req_to_token_runtime_inspection_payload_summary",
        "total_req_to_token_runtime_inspection_payloads": len(payloads),
        "blocked_count": blocked_count,
        "error_count": error_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_inspection_state_counts": dict(sorted(per_inspection_state.items())),
        "metadata_observed_count": safety_counts["metadata_observed_count"],
        "req_to_token_attr_present_count": (
            safety_counts["req_to_token_attr_present_count"]
        ),
        "actual_req_to_token_pool_inspection_count": (
            safety_counts["actual_req_to_token_pool_inspection_count"]
        ),
        "req_to_token_attr_observed_count": (
            safety_counts["req_to_token_attr_observed_count"]
        ),
        "req_to_token_read_count": safety_counts["req_to_token_read_count"],
        "actual_req_to_token_pool_read_count": (
            safety_counts["actual_req_to_token_pool_read_count"]
        ),
        "token_to_kv_pool_read_count": (
            safety_counts["token_to_kv_pool_read_count"]
        ),
        "kv_pool_read_count": safety_counts["kv_pool_read_count"],
        "kv_snapshot_count": safety_counts["kv_snapshot_count"],
        "tensor_read_count": safety_counts["tensor_read_count"],
        "attention_comparison_executed_count": (
            safety_counts["attention_comparison_executed_count"]
        ),
        "attention_override_true_count": (
            safety_counts["attention_override_true_count"]
        ),
        "runtime_writeback_true_count": (
            safety_counts["runtime_writeback_true_count"]
        ),
        "scheduler_policy_noop_false_count": (
            safety_counts["scheduler_policy_noop_false_count"]
        ),
        "kv_cache_mutation_true_count": (
            safety_counts["kv_cache_mutation_true_count"]
        ),
        "source_mutated_true_count": safety_counts["source_mutated_true_count"],
    }


def run_model_runner_req_to_token_runtime_inspection_hook_for_smoke(
    model_runner: Any,
    forward_batch: Any = None,
) -> dict[str, Any]:
    """Run a fake model-runner req_to_token metadata inspection hook for smoke."""

    req_to_token_pool = None
    req_to_token_pool_path = None

    try:
        req_to_token_pool = getattr(model_runner, "req_to_token_pool", None)
    except Exception:
        req_to_token_pool = None
    else:
        if req_to_token_pool is not None:
            req_to_token_pool_path = "model_runner.req_to_token_pool"

    if req_to_token_pool is None:
        try:
            allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
        except Exception:
            allocator = None
        if allocator is not None:
            try:
                req_to_token_pool = getattr(allocator, "req_to_token_pool", None)
            except Exception:
                req_to_token_pool = None
            else:
                if req_to_token_pool is not None:
                    req_to_token_pool_path = (
                        "model_runner.token_to_kv_pool_allocator.req_to_token_pool"
                    )

    if req_to_token_pool is None:
        try:
            memory_pool = getattr(model_runner, "memory_pool", None)
        except Exception:
            memory_pool = None
        if memory_pool is not None:
            try:
                req_to_token_pool = getattr(memory_pool, "req_to_token_pool", None)
            except Exception:
                req_to_token_pool = None
            else:
                if req_to_token_pool is not None:
                    req_to_token_pool_path = (
                        "model_runner.memory_pool.req_to_token_pool"
                    )

    if req_to_token_pool is None and forward_batch is not None:
        try:
            req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        except Exception:
            req_to_token_pool = None
        else:
            if req_to_token_pool is not None:
                req_to_token_pool_path = "forward_batch.req_to_token_pool"

    payloads = build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
        forward_batch_like=forward_batch,
        req_to_token_pool=req_to_token_pool,
        inspect_req_to_token=True,
    )
    for payload in payloads:
        payload["hook_path"] = req_to_token_pool_path
        payload["hook_source"] = "model_runner_req_to_token_runtime_inspection_hook"

    summary = summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
        payloads
    )
    summary["hook_path_counts"] = {
        "none": 0,
        "model_runner.req_to_token_pool": 0,
        "model_runner.token_to_kv_pool_allocator.req_to_token_pool": 0,
        "model_runner.memory_pool.req_to_token_pool": 0,
        "forward_batch.req_to_token_pool": 0,
    }
    hook_key = req_to_token_pool_path or "none"
    if hook_key not in summary["hook_path_counts"]:
        summary["hook_path_counts"][hook_key] = 0
    summary["hook_path_counts"][hook_key] += len(payloads)

    return {
        "payloads": payloads,
        "summary": summary,
        "req_to_token_pool_path": req_to_token_pool_path,
    }


def build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
    runtime_like: Mapping[str, Any] | None = None,
    token_to_kv_pool: Any = None,
    inspect_token_to_kv_pool: bool = False,
) -> list[dict[str, Any]]:
    """Build metadata-only token_to_kv_pool runtime inspection payloads."""

    if runtime_like is not None and not isinstance(runtime_like, Mapping):
        raise TypeError("runtime_like must be a mapping or None")

    request_id = None
    layer_id = None
    batch_id = None
    kv_head_group = None
    decision_state = None
    kv_class = "UNKNOWN"
    token_to_kv_pool_source_path = None
    token_to_kv_pool_lookup_error = False
    blocking_reasons: list[str] = []

    if runtime_like is not None:
        request_id = runtime_like.get("request_id")
        layer_id = runtime_like.get("layer_id")
        batch_id = runtime_like.get("batch_id")
        kv_head_group = runtime_like.get("kv_head_group")
        decision_state = runtime_like.get("decision_state")
        kv_class = runtime_like.get("kv_class") or "UNKNOWN"
        token_to_kv_pool_source_path = runtime_like.get("token_to_kv_pool_source_path")
        token_to_kv_pool_lookup_error = (
            runtime_like.get("token_to_kv_pool_lookup_error") is True
        )

        if runtime_like.get("token_to_kv_pool_read") is True:
            blocking_reasons.append("token_to_kv_pool_value_read_not_allowed")
        if runtime_like.get("token_to_kv_pool_index_read") is True:
            blocking_reasons.append("token_to_kv_pool_index_read_not_allowed")
        if runtime_like.get("kv_pool_read") is True:
            blocking_reasons.append("kv_pool_read_not_allowed")
        if runtime_like.get("kv_snapshot") is True:
            blocking_reasons.append("kv_snapshot_not_allowed")
        if runtime_like.get("tensor_read") is True:
            blocking_reasons.append("tensor_read_not_allowed")
        if runtime_like.get("attention_override") is True:
            blocking_reasons.append("attention_override_true_not_allowed")
        if runtime_like.get("attention_comparison_executed") is True:
            blocking_reasons.append("attention_comparison_executed_not_allowed")
        if runtime_like.get("runtime_writeback") is True:
            blocking_reasons.append("runtime_writeback_not_allowed")
        if runtime_like.get("scheduler_policy_noop") is False:
            blocking_reasons.append("scheduler_mutation_not_allowed")
        if runtime_like.get("source_mutated") is True:
            blocking_reasons.append("source_mutation_not_allowed")
        if runtime_like.get("token_to_kv_pool_lookup_error") is True:
            blocking_reasons.append("token_to_kv_pool_attr_access_failed")

    token_to_kv_pool_attr_present = False
    token_to_kv_pool_attr_observed = False
    token_to_kv_pool_type = None
    token_to_kv_pool_module = None
    token_to_kv_pool_qualname = None
    token_to_kv_pool_shape = None
    token_to_kv_pool_device = None
    token_to_kv_pool_dtype = None

    if inspect_token_to_kv_pool is not True:
        blocking_reasons.append("inspection_not_enabled")
    elif token_to_kv_pool is None and token_to_kv_pool_lookup_error is not True:
        blocking_reasons.append("token_to_kv_pool_missing")
    else:
        try:
            token_to_kv_pool_type = type(token_to_kv_pool).__name__
            token_to_kv_pool_module = type(token_to_kv_pool).__module__
            token_to_kv_pool_qualname = type(token_to_kv_pool).__qualname__
            token_to_kv_pool_shape = getattr(token_to_kv_pool, "shape", None)
            token_to_kv_pool_device = getattr(token_to_kv_pool, "device", None)
            token_to_kv_pool_dtype = getattr(token_to_kv_pool, "dtype", None)
        except Exception:
            blocking_reasons.append("token_to_kv_pool_attr_access_failed")
        else:
            token_to_kv_pool_attr_present = True
            token_to_kv_pool_attr_observed = True

    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    inspection_state = "metadata_observed" if not blocking_reasons else "blocked"
    payload = normalize_relaykv_sglang_adapter_schema_for_smoke(
        {
            "event_type": "relaykv_token_to_kv_pool_runtime_inspection_payload",
            "inspection_mode": "runtime_metadata_only",
            "inspection_state": inspection_state,
            "source": "token_to_kv_pool_to_runtime_inspection_payload",
            "request_id": request_id,
            "layer_id": layer_id,
            "batch_id": batch_id,
            "kv_head_group": kv_head_group,
            "kv_class": kv_class,
            "decision_state": decision_state or "SHADOW_ONLY",
            "metadata_observed": not blocking_reasons,
            "token_to_kv_pool_attr_present": token_to_kv_pool_attr_present,
            "token_to_kv_pool_attr_observed": token_to_kv_pool_attr_observed,
            "actual_token_to_kv_pool_inspection": not blocking_reasons,
            "token_to_kv_pool_type": token_to_kv_pool_type,
            "token_to_kv_pool_module": token_to_kv_pool_module,
            "token_to_kv_pool_qualname": token_to_kv_pool_qualname,
            "token_to_kv_pool_shape": token_to_kv_pool_shape,
            "token_to_kv_pool_device": token_to_kv_pool_device,
            "token_to_kv_pool_dtype": token_to_kv_pool_dtype,
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read": False,
            "actual_token_to_kv_pool_read_count": 0,
            "req_to_token_read": False,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read": False,
            "actual_req_to_token_pool_read_count": 0,
            "kv_pool_read": False,
            "kv_snapshot": False,
            "tensor_read": False,
            "attention_comparison_executed": False,
            "attention_override": False,
            "runtime_writeback": False,
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "source_mutated": False,
            "blocking_reasons": blocking_reasons,
            "warning_reasons": [
                "token_to_kv_pool_metadata_only_inspection",
                "no_token_to_kv_pool_values_read",
                "no_req_to_token_read",
            ],
        }
    )
    payload["adapter_metadata"]["token_to_kv_pool_type"] = token_to_kv_pool_type
    payload["adapter_metadata"]["token_to_kv_pool_module"] = token_to_kv_pool_module
    payload["adapter_metadata"]["token_to_kv_pool_qualname"] = (
        token_to_kv_pool_qualname
    )
    payload["adapter_metadata"]["token_to_kv_pool_shape"] = token_to_kv_pool_shape
    payload["adapter_metadata"]["token_to_kv_pool_device"] = token_to_kv_pool_device
    payload["adapter_metadata"]["token_to_kv_pool_dtype"] = token_to_kv_pool_dtype
    payload["adapter_metadata"]["token_to_kv_pool_source_path"] = (
        token_to_kv_pool_source_path
    )
    engine_block_ref = payload.get("engine_block_ref")
    if not isinstance(engine_block_ref, dict):
        engine_block_ref = {}
        payload["engine_block_ref"] = engine_block_ref
    engine_block_ref["token_to_kv_pool_index"] = None
    engine_block_ref["physical_kv_index"] = None
    engine_block_ref["cache_position"] = None
    if not blocking_reasons:
        payload["fallback_reason"] = None
    return [payload]


def summarize_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
    payloads: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Summarize metadata-only token_to_kv_pool runtime inspection payloads."""

    if not isinstance(payloads, (list, tuple)):
        raise TypeError(
            "RelayKV token_to_kv_pool runtime inspection payloads must be a list or tuple"
        )

    per_request: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_inspection_state: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter(
        {
            "metadata_observed_count": 0,
            "token_to_kv_pool_attr_present_count": 0,
            "actual_token_to_kv_pool_inspection_count": 0,
            "token_to_kv_pool_attr_observed_count": 0,
            "token_to_kv_pool_read_count": 0,
            "actual_token_to_kv_pool_read_count": 0,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    blocked_count = 0
    error_count = 0

    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(
                "RelayKV token_to_kv_pool runtime inspection payload must be a dict"
            )

        state = str(payload.get("inspection_state") or "unknown")
        per_inspection_state[state] += 1
        per_request[str(payload.get("request_id"))] += 1
        per_layer[str(payload.get("layer_id"))] += 1

        if state == "blocked":
            blocked_count += 1
        elif state == "error":
            error_count += 1

        if payload.get("metadata_observed") is True:
            safety_counts["metadata_observed_count"] += 1
        if payload.get("token_to_kv_pool_attr_present") is True:
            safety_counts["token_to_kv_pool_attr_present_count"] += 1
        if payload.get("actual_token_to_kv_pool_inspection") is True:
            safety_counts["actual_token_to_kv_pool_inspection_count"] += 1
        if payload.get("token_to_kv_pool_attr_observed") is True:
            safety_counts["token_to_kv_pool_attr_observed_count"] += 1

        for key in (
            "token_to_kv_pool_read_count",
            "actual_token_to_kv_pool_read_count",
            "req_to_token_read_count",
            "actual_req_to_token_pool_read_count",
        ):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                safety_counts[key] += value
        if payload.get("kv_pool_read") is True:
            safety_counts["kv_pool_read_count"] += 1
        if payload.get("kv_snapshot") is True:
            safety_counts["kv_snapshot_count"] += 1
        if payload.get("tensor_read") is True:
            safety_counts["tensor_read_count"] += 1
        if payload.get("attention_comparison_executed") is True:
            safety_counts["attention_comparison_executed_count"] += 1
        if payload.get("attention_override") is True:
            safety_counts["attention_override_true_count"] += 1
        if payload.get("runtime_writeback") is True:
            safety_counts["runtime_writeback_true_count"] += 1
        if payload.get("scheduler_policy_noop") is False:
            safety_counts["scheduler_policy_noop_false_count"] += 1
        if payload.get("kv_cache_mutation") is True:
            safety_counts["kv_cache_mutation_true_count"] += 1
        if payload.get("source_mutated") is True:
            safety_counts["source_mutated_true_count"] += 1

    return {
        "summary_type": "relaykv_token_to_kv_pool_runtime_inspection_payload_summary",
        "total_token_to_kv_pool_runtime_inspection_payloads": len(payloads),
        "blocked_count": blocked_count,
        "error_count": error_count,
        "per_request_counts": dict(sorted(per_request.items())),
        "per_layer_counts": dict(sorted(per_layer.items())),
        "per_inspection_state_counts": dict(sorted(per_inspection_state.items())),
        "metadata_observed_count": safety_counts["metadata_observed_count"],
        "token_to_kv_pool_attr_present_count": (
            safety_counts["token_to_kv_pool_attr_present_count"]
        ),
        "actual_token_to_kv_pool_inspection_count": (
            safety_counts["actual_token_to_kv_pool_inspection_count"]
        ),
        "token_to_kv_pool_attr_observed_count": (
            safety_counts["token_to_kv_pool_attr_observed_count"]
        ),
        "token_to_kv_pool_read_count": (
            safety_counts["token_to_kv_pool_read_count"]
        ),
        "actual_token_to_kv_pool_read_count": (
            safety_counts["actual_token_to_kv_pool_read_count"]
        ),
        "req_to_token_read_count": safety_counts["req_to_token_read_count"],
        "actual_req_to_token_pool_read_count": (
            safety_counts["actual_req_to_token_pool_read_count"]
        ),
        "kv_pool_read_count": safety_counts["kv_pool_read_count"],
        "kv_snapshot_count": safety_counts["kv_snapshot_count"],
        "tensor_read_count": safety_counts["tensor_read_count"],
        "attention_comparison_executed_count": (
            safety_counts["attention_comparison_executed_count"]
        ),
        "attention_override_true_count": (
            safety_counts["attention_override_true_count"]
        ),
        "runtime_writeback_true_count": (
            safety_counts["runtime_writeback_true_count"]
        ),
        "scheduler_policy_noop_false_count": (
            safety_counts["scheduler_policy_noop_false_count"]
        ),
        "kv_cache_mutation_true_count": (
            safety_counts["kv_cache_mutation_true_count"]
        ),
        "source_mutated_true_count": safety_counts["source_mutated_true_count"],
    }


def run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
    model_runner: Any,
    forward_batch: Any = None,
) -> dict[str, Any]:
    """Run a fake model-runner token_to_kv_pool metadata inspection hook for smoke."""

    token_to_kv_pool = None
    token_to_kv_pool_path = None
    token_to_kv_pool_lookup_error = False

    try:
        token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    except Exception:
        token_to_kv_pool = None
        token_to_kv_pool_lookup_error = True
        token_to_kv_pool_path = "model_runner.token_to_kv_pool"
    else:
        if token_to_kv_pool is not None:
            token_to_kv_pool_path = "model_runner.token_to_kv_pool"

    if token_to_kv_pool is None and token_to_kv_pool_lookup_error is not True:
        try:
            allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
        except Exception:
            allocator = None
        if allocator is not None:
            try:
                nested_token_to_kv_pool = getattr(allocator, "token_to_kv_pool", None)
            except Exception:
                token_to_kv_pool = None
                token_to_kv_pool_lookup_error = True
                token_to_kv_pool_path = (
                    "model_runner.token_to_kv_pool_allocator.token_to_kv_pool"
                )
            else:
                if nested_token_to_kv_pool is not None:
                    token_to_kv_pool = nested_token_to_kv_pool
                    token_to_kv_pool_path = (
                        "model_runner.token_to_kv_pool_allocator.token_to_kv_pool"
                    )
                else:
                    token_to_kv_pool = allocator
                    token_to_kv_pool_path = "model_runner.token_to_kv_pool_allocator"

    if token_to_kv_pool is None and token_to_kv_pool_lookup_error is not True:
        try:
            allocator = getattr(model_runner, "kv_pool_allocator", None)
        except Exception:
            allocator = None
        if allocator is not None:
            try:
                token_to_kv_pool = getattr(allocator, "token_to_kv_pool", None)
            except Exception:
                token_to_kv_pool = None
                token_to_kv_pool_lookup_error = True
                token_to_kv_pool_path = (
                    "model_runner.kv_pool_allocator.token_to_kv_pool"
                )
            else:
                if token_to_kv_pool is not None:
                    token_to_kv_pool_path = (
                        "model_runner.kv_pool_allocator.token_to_kv_pool"
                    )

    if token_to_kv_pool is None and token_to_kv_pool_lookup_error is not True:
        try:
            memory_pool = getattr(model_runner, "memory_pool", None)
        except Exception:
            memory_pool = None
        if memory_pool is not None:
            try:
                token_to_kv_pool = getattr(memory_pool, "token_to_kv_pool", None)
            except Exception:
                token_to_kv_pool = None
                token_to_kv_pool_lookup_error = True
                token_to_kv_pool_path = "model_runner.memory_pool.token_to_kv_pool"
            else:
                if token_to_kv_pool is not None:
                    token_to_kv_pool_path = "model_runner.memory_pool.token_to_kv_pool"

    if (
        token_to_kv_pool is None
        and token_to_kv_pool_lookup_error is not True
        and forward_batch is not None
    ):
        try:
            token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        except Exception:
            token_to_kv_pool = None
            token_to_kv_pool_lookup_error = True
            token_to_kv_pool_path = "forward_batch.token_to_kv_pool"
        else:
            if token_to_kv_pool is not None:
                token_to_kv_pool_path = "forward_batch.token_to_kv_pool"

    runtime_like = {
        "request_id": getattr(forward_batch, "request_id", None)
        if forward_batch is not None
        else None,
        "layer_id": getattr(forward_batch, "layer_id", None)
        if forward_batch is not None
        else None,
        "batch_id": getattr(forward_batch, "batch_id", None)
        if forward_batch is not None
        else None,
        "kv_head_group": getattr(forward_batch, "kv_head_group", None)
        if forward_batch is not None
        else None,
        "decision_state": getattr(forward_batch, "decision_state", None)
        if forward_batch is not None
        else None,
        "kv_class": getattr(forward_batch, "kv_class", None)
        if forward_batch is not None
        else None,
        "token_to_kv_pool_source_path": token_to_kv_pool_path,
        "token_to_kv_pool_lookup_error": token_to_kv_pool_lookup_error,
    }
    payloads = build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
        runtime_like=runtime_like,
        token_to_kv_pool=token_to_kv_pool,
        inspect_token_to_kv_pool=True,
    )
    for payload in payloads:
        payload["hook_path"] = token_to_kv_pool_path
        payload["hook_source"] = (
            "model_runner_token_to_kv_pool_runtime_inspection_hook"
        )

    summary = summarize_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
        payloads
    )
    summary["hook_path_counts"] = {
        "none": 0,
        "model_runner.token_to_kv_pool": 0,
        "model_runner.token_to_kv_pool_allocator.token_to_kv_pool": 0,
        "model_runner.token_to_kv_pool_allocator": 0,
        "model_runner.kv_pool_allocator.token_to_kv_pool": 0,
        "model_runner.memory_pool.token_to_kv_pool": 0,
        "forward_batch.token_to_kv_pool": 0,
    }
    hook_key = token_to_kv_pool_path or "none"
    if hook_key not in summary["hook_path_counts"]:
        summary["hook_path_counts"][hook_key] = 0
    summary["hook_path_counts"][hook_key] += len(payloads)

    return {
        "payloads": payloads,
        "summary": summary,
        "token_to_kv_pool_path": token_to_kv_pool_path,
    }


def run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke(
    model_runner: Any,
    forward_batch: Any = None,
    explicit_req_to_token_resolution_payloads: Any = None,
) -> dict[str, Any]:
    """Run a guarded live-like token_to_kv_pool index read hook for smoke only."""

    token_to_kv_pool = None
    token_to_kv_pool_path = None
    token_to_kv_pool_lookup_error = False

    try:
        token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    except Exception:
        token_to_kv_pool = None
        token_to_kv_pool_lookup_error = True
        token_to_kv_pool_path = "model_runner.token_to_kv_pool"
    else:
        if token_to_kv_pool is not None:
            token_to_kv_pool_path = "model_runner.token_to_kv_pool"

    if token_to_kv_pool is None and token_to_kv_pool_lookup_error is not True:
        try:
            allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
        except Exception:
            allocator = None
        if allocator is not None:
            try:
                nested_token_to_kv_pool = getattr(allocator, "token_to_kv_pool", None)
            except Exception:
                token_to_kv_pool = None
                token_to_kv_pool_lookup_error = True
                token_to_kv_pool_path = (
                    "model_runner.token_to_kv_pool_allocator.token_to_kv_pool"
                )
            else:
                if nested_token_to_kv_pool is not None:
                    token_to_kv_pool = nested_token_to_kv_pool
                    token_to_kv_pool_path = (
                        "model_runner.token_to_kv_pool_allocator.token_to_kv_pool"
                    )
                else:
                    token_to_kv_pool = allocator
                    token_to_kv_pool_path = "model_runner.token_to_kv_pool_allocator"

    if token_to_kv_pool is None and token_to_kv_pool_lookup_error is not True:
        try:
            allocator = getattr(model_runner, "kv_pool_allocator", None)
        except Exception:
            allocator = None
        if allocator is not None:
            try:
                token_to_kv_pool = getattr(allocator, "token_to_kv_pool", None)
            except Exception:
                token_to_kv_pool = None
                token_to_kv_pool_lookup_error = True
                token_to_kv_pool_path = (
                    "model_runner.kv_pool_allocator.token_to_kv_pool"
                )
            else:
                if token_to_kv_pool is not None:
                    token_to_kv_pool_path = (
                        "model_runner.kv_pool_allocator.token_to_kv_pool"
                    )

    if token_to_kv_pool is None and token_to_kv_pool_lookup_error is not True:
        try:
            memory_pool = getattr(model_runner, "memory_pool", None)
        except Exception:
            memory_pool = None
        if memory_pool is not None:
            try:
                token_to_kv_pool = getattr(memory_pool, "token_to_kv_pool", None)
            except Exception:
                token_to_kv_pool = None
                token_to_kv_pool_lookup_error = True
                token_to_kv_pool_path = "model_runner.memory_pool.token_to_kv_pool"
            else:
                if token_to_kv_pool is not None:
                    token_to_kv_pool_path = "model_runner.memory_pool.token_to_kv_pool"

    if (
        token_to_kv_pool is None
        and token_to_kv_pool_lookup_error is not True
        and forward_batch is not None
    ):
        try:
            token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        except Exception:
            token_to_kv_pool = None
            token_to_kv_pool_lookup_error = True
            token_to_kv_pool_path = "forward_batch.token_to_kv_pool"
        else:
            if token_to_kv_pool is not None:
                token_to_kv_pool_path = "forward_batch.token_to_kv_pool"

    req_to_token_resolution_bridge_enabled = (
        os.getenv("SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE") == "1"
    )
    req_to_token_resolution_results = None
    req_to_token_resolution_results_path = None
    req_to_token_resolution_bridge_state = "not_attempted"
    req_to_token_resolution_bridge_payload_count = 0
    req_to_token_resolution_bridge_valid_count = 0
    req_to_token_resolution_bridge_source_path = None
    req_to_token_resolution_bridge_blocked_reason = None

    if req_to_token_resolution_bridge_enabled:
        bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
            forward_batch=forward_batch,
            model_runner=model_runner,
            explicit_payloads=explicit_req_to_token_resolution_payloads,
            bridge_enabled=True,
        )
        bridge_result = bridge_results[0]
        req_to_token_resolution_bridge_state = str(
            bridge_result.get("bridge_state") or "unknown"
        )
        req_to_token_resolution_bridge_payload_count = int(
            bridge_result.get("payload_count") or 0
        )
        req_to_token_resolution_bridge_valid_count = int(
            bridge_result.get("valid_payload_count") or 0
        )
        req_to_token_resolution_bridge_source_path = bridge_result.get(
            "bridge_source_path"
        )
        req_to_token_resolution_bridge_blocked_reason = bridge_result.get(
            "blocked_reason"
        )
        if req_to_token_resolution_bridge_state == "bridged":
            req_to_token_resolution_results = bridge_result[
                "req_to_token_resolution_payloads"
            ]
            req_to_token_resolution_results_path = (
                req_to_token_resolution_bridge_source_path
            )
    else:
        for owner, path in (
            (forward_batch, "relaykv_req_to_token_resolution_results"),
            (forward_batch, "req_to_token_resolution_results"),
            (model_runner, "relaykv_req_to_token_resolution_results"),
            (model_runner, "req_to_token_resolution_results"),
        ):
            if owner is None:
                continue
            try:
                value = getattr(owner, path, None)
            except Exception:
                continue
            if value is not None:
                req_to_token_resolution_results = value
                req_to_token_resolution_results_path = (
                    f"forward_batch.{path}"
                    if owner is forward_batch
                    else f"model_runner.{path}"
                )
                break

    request_id = (
        getattr(forward_batch, "request_id", None) if forward_batch is not None else None
    )
    layer_id = (
        getattr(forward_batch, "layer_id", None) if forward_batch is not None else None
    )
    batch_id = (
        getattr(forward_batch, "batch_id", None) if forward_batch is not None else None
    )
    kv_head_group = (
        getattr(forward_batch, "kv_head_group", None)
        if forward_batch is not None
        else None
    )
    kv_class = (
        getattr(forward_batch, "kv_class", None) if forward_batch is not None else None
    )

    if not isinstance(req_to_token_resolution_results, (list, tuple)):
        blocked_source_payload = {
            "event_type": "relaykv_req_to_token_resolution_result",
            "request_id": request_id,
            "layer_id": layer_id,
            "batch_id": batch_id,
            "kv_head_group": kv_head_group,
            "kv_class": kv_class,
            "engine_name": "sglang",
            "adapter_name": "sglang",
            "engine_request_id": request_id,
            "logical_sequence_id": request_id,
            "decision_state": "SHADOW_ONLY",
            "position_check_state": "not_checked_metadata_only",
            "attention_mask_mode": "unknown",
            "rope_position_consistency": "not_checked",
            "adapter_metadata": {
                "req_to_token_resolution_results_path": (
                    req_to_token_resolution_results_path
                ),
                "req_to_token_resolution_bridge_enabled": (
                    req_to_token_resolution_bridge_enabled
                ),
                "req_to_token_resolution_bridge_state": (
                    req_to_token_resolution_bridge_state
                ),
                "req_to_token_resolution_bridge_payload_count": (
                    req_to_token_resolution_bridge_payload_count
                ),
                "req_to_token_resolution_bridge_source_path": (
                    req_to_token_resolution_bridge_source_path
                ),
                "req_to_token_resolution_bridge_blocked_reason": (
                    req_to_token_resolution_bridge_blocked_reason
                ),
            },
            "engine_block_ref": {},
            "full_kv_req_to_token_spans": None,
            "kv_pool_read": False,
            "kv_snapshot": False,
            "tensor_read": False,
            "attention_comparison_executed": False,
            "attention_override": False,
            "runtime_writeback": False,
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "source_mutated": False,
        }
        payloads = [
            _blocked_live_token_to_kv_pool_index_read_result_for_smoke(
                blocked_source_payload,
                blocking_reasons=[
                    req_to_token_resolution_bridge_blocked_reason
                    if req_to_token_resolution_bridge_enabled
                    and req_to_token_resolution_bridge_blocked_reason is not None
                    else "req_to_token_resolution_results_missing"
                ],
                warning_reasons=[
                    "guarded_live_token_to_kv_pool_index_read",
                    "bounded_index_lookup_only",
                    "no_kv_pool_read",
                    (
                        "req_to_token_resolution_bridge_blocked"
                        if req_to_token_resolution_bridge_enabled
                        else "missing_req_to_token_resolution_results"
                    ),
                ],
                source_path=token_to_kv_pool_path,
                token_to_kv_pool_type=_token_to_kv_pool_object_type_for_smoke(
                    token_to_kv_pool
                ),
                token_to_kv_pool_shape=_token_to_kv_pool_object_shape_for_smoke(
                    token_to_kv_pool
                ),
                read_token_to_kv_pool_index=True,
                max_tokens_per_request=256,
                max_total_tokens=1024,
            )
        ]
    else:
        payloads = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
            req_to_token_resolution_results,
            token_to_kv_pool_object=token_to_kv_pool,
            read_token_to_kv_pool_index=True,
            max_tokens_per_request=256,
            max_total_tokens=1024,
            source_path=token_to_kv_pool_path,
        )

    for payload in payloads:
        payload["hook_path"] = token_to_kv_pool_path
        payload["hook_source"] = (
            "model_runner_live_token_to_kv_pool_index_read_hook"
        )
        adapter_metadata = payload.get("adapter_metadata")
        if isinstance(adapter_metadata, dict):
            adapter_metadata["req_to_token_resolution_results_path"] = (
                req_to_token_resolution_results_path
            )
            adapter_metadata["req_to_token_resolution_bridge_enabled"] = (
                req_to_token_resolution_bridge_enabled
            )
            adapter_metadata["req_to_token_resolution_bridge_state"] = (
                req_to_token_resolution_bridge_state
            )
            adapter_metadata["req_to_token_resolution_bridge_payload_count"] = (
                req_to_token_resolution_bridge_payload_count
            )
            adapter_metadata["req_to_token_resolution_bridge_source_path"] = (
                req_to_token_resolution_bridge_source_path
            )
            adapter_metadata["req_to_token_resolution_bridge_blocked_reason"] = (
                req_to_token_resolution_bridge_blocked_reason
            )

    summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        payloads
    )
    summary["hook_path_counts"] = {
        "none": 0,
        "model_runner.token_to_kv_pool": 0,
        "model_runner.token_to_kv_pool_allocator.token_to_kv_pool": 0,
        "model_runner.token_to_kv_pool_allocator": 0,
        "model_runner.kv_pool_allocator.token_to_kv_pool": 0,
        "model_runner.memory_pool.token_to_kv_pool": 0,
        "forward_batch.token_to_kv_pool": 0,
    }
    hook_key = token_to_kv_pool_path or "none"
    if hook_key not in summary["hook_path_counts"]:
        summary["hook_path_counts"][hook_key] = 0
    summary["hook_path_counts"][hook_key] += len(payloads)
    summary["req_to_token_resolution_results_path"] = req_to_token_resolution_results_path
    summary["req_to_token_resolution_bridge_enabled"] = (
        req_to_token_resolution_bridge_enabled
    )
    summary["req_to_token_resolution_bridge_state"] = (
        req_to_token_resolution_bridge_state
    )
    summary["req_to_token_resolution_bridge_payload_count"] = (
        req_to_token_resolution_bridge_payload_count
    )
    summary["req_to_token_resolution_bridge_valid_count"] = (
        req_to_token_resolution_bridge_valid_count
    )
    summary["req_to_token_resolution_bridge_source_path"] = (
        req_to_token_resolution_bridge_source_path
    )
    summary["req_to_token_resolution_bridge_blocked_reason"] = (
        req_to_token_resolution_bridge_blocked_reason
    )

    return {
        "payloads": payloads,
        "summary": summary,
        "token_to_kv_pool_path": token_to_kv_pool_path,
        "req_to_token_resolution_results_path": req_to_token_resolution_results_path,
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
