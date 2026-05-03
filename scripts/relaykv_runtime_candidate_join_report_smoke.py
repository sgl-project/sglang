from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    assess_relaykv_readonly_materialization_readiness_for_smoke,
    build_relaykv_readonly_runtime_candidate_join_report_for_smoke,
)


def _runtime_summary() -> dict[str, Any]:
    return {
        "total_payloads": 4,
        "per_request_counts": {"rid-a": 2, "rid-b": 2},
        "per_layer_counts": {"0": 2, "14": 2},
        "per_batch_counts": {"join-batch-a": 4},
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _host_backup_candidate_summary() -> dict[str, Any]:
    return {
        "total_candidate_events": 3,
        "applied_candidate_count": 3,
        "fallback_candidate_count": 0,
        "host_backup_copy_candidate_count": 3,
        "host_backup_copy_executed_count": 3,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _join_summary(*, join_granularity: str = "per_event") -> dict[str, Any]:
    return {
        "join_granularity": join_granularity,
        "total_runtime_payloads": 4,
        "total_host_backup_candidate_events": 3,
        "joined_count": 2,
        "unmatched_runtime_count": 2,
        "unmatched_candidate_count": 1,
        "per_request_join_counts": {"rid-a": 1, "rid-b": 1},
        "per_layer_join_counts": {"0": 1, "14": 1},
        "req_pool_idx_joined_count": 2,
        "req_pool_idx_missing_count": 0,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _policy_dry_run_summary() -> dict[str, Any]:
    return {
        "total_events": 3,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _assert_safety_zero(report: dict[str, Any]) -> None:
    for key in (
        "source_mutated_true_count",
        "attention_override_true_count",
        "kv_cache_mutation_true_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
    ):
        if report[key] != 0:
            raise AssertionError(report)


def _assert_pass_report() -> dict[str, Any]:
    runtime_summary = _runtime_summary()
    candidate_summary = _host_backup_candidate_summary()
    join_summary = _join_summary()
    runtime_before = copy.deepcopy(runtime_summary)
    candidate_before = copy.deepcopy(candidate_summary)
    join_before = copy.deepcopy(join_summary)

    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        runtime_summary,
        candidate_summary,
        join_summary,
    )

    if runtime_summary != runtime_before:
        raise AssertionError("runtime summary was mutated")
    if candidate_summary != candidate_before:
        raise AssertionError("candidate summary was mutated")
    if join_summary != join_before:
        raise AssertionError("join summary was mutated")

    expected = {
        "report_type": "relaykv_readonly_runtime_candidate_join_report",
        "report_generated_from_readonly_inputs": True,
        "overall_safety_status": "pass",
        "total_runtime_payloads": 4,
        "total_host_backup_candidate_events": 3,
        "joined_count": 2,
        "unmatched_runtime_count": 2,
        "unmatched_candidate_count": 1,
        "join_granularity": "per_event",
        "req_pool_idx_joined_count": 2,
        "req_pool_idx_missing_count": 0,
    }
    for key, value in expected.items():
        if report[key] != value:
            raise AssertionError(report)
    _assert_safety_zero(report)
    if report["missing_field_counts"]["missing_field_count"] != 0:
        raise AssertionError(report)
    if report["report_warning_counts"]["missing_field_warning_count"] != 0:
        raise AssertionError(report)
    if report["policy_dry_run_included"] is not False:
        raise AssertionError(report)
    if report["policy_dry_run_summary"] is not None:
        raise AssertionError(report)
    return report


def _assert_policy_dry_run_report() -> dict[str, Any]:
    runtime_summary = _runtime_summary()
    candidate_summary = _host_backup_candidate_summary()
    join_summary = _join_summary()
    policy_summary = _policy_dry_run_summary()
    policy_before = copy.deepcopy(policy_summary)
    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        runtime_summary,
        candidate_summary,
        join_summary,
        policy_dry_run_summary=policy_summary,
    )
    if policy_summary != policy_before:
        raise AssertionError("policy dry-run summary was mutated")
    if report["policy_dry_run_included"] is not True:
        raise AssertionError(report)
    if report["policy_dry_run_summary"] != policy_summary:
        raise AssertionError(report)
    if report["policy_dry_run_total_events"] != 3:
        raise AssertionError(report)
    if report["policy_dry_run_selected_event_count"] != 0:
        raise AssertionError(report)
    if report["overall_safety_status"] != "pass":
        raise AssertionError(report)
    for key in (
        "source_mutated_true_count",
        "attention_override_true_count",
        "kv_cache_mutation_true_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
    ):
        if report[key] != 0:
            raise AssertionError(report)
    readiness = assess_relaykv_readonly_materialization_readiness_for_smoke(report)
    if readiness["ready_for_materialization"] is not True:
        raise AssertionError(readiness)
    return report


def _assert_policy_dry_run_fail_report() -> dict[str, Any]:
    policy_summary = _policy_dry_run_summary()
    policy_summary["kv_cache_mutation_true_count"] = 1
    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        _runtime_summary(),
        _host_backup_candidate_summary(),
        _join_summary(),
        policy_dry_run_summary=policy_summary,
    )
    if report["policy_dry_run_included"] is not True:
        raise AssertionError(report)
    if report["policy_dry_run_total_events"] != 3:
        raise AssertionError(report)
    if report["overall_safety_status"] != "fail":
        raise AssertionError(report)
    if report["kv_cache_mutation_true_count"] != 1:
        raise AssertionError(report)
    return report


def _assert_fail_report() -> dict[str, Any]:
    runtime_summary = _runtime_summary()
    candidate_summary = _host_backup_candidate_summary()
    join_summary = _join_summary()
    join_summary["kv_cache_mutation_true_count"] = 1
    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        runtime_summary,
        candidate_summary,
        join_summary,
    )
    if report["overall_safety_status"] != "fail":
        raise AssertionError(report)
    if report["kv_cache_mutation_true_count"] != 1:
        raise AssertionError(report)
    return report


def _assert_summary_only_unjoinable_report() -> dict[str, Any]:
    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        _runtime_summary(),
        _host_backup_candidate_summary(),
        _join_summary(join_granularity="summary_only_unjoinable"),
    )
    if report["join_granularity"] != "summary_only_unjoinable":
        raise AssertionError(report)
    if report["overall_safety_status"] != "pass":
        raise AssertionError(report)
    _assert_safety_zero(report)
    return report


def _assert_missing_fields_report() -> dict[str, Any]:
    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        {},
        {},
        {"join_granularity": "summary_only_unjoinable"},
    )
    if report["overall_safety_status"] != "pass":
        raise AssertionError(report)
    if report["missing_field_counts"]["missing_field_count"] == 0:
        raise AssertionError(report)
    if report["report_warning_counts"]["missing_field_warning_count"] == 0:
        raise AssertionError(report)
    return report


def main() -> None:
    result = {
        "pass_report": _assert_pass_report(),
        "policy_dry_run_report": _assert_policy_dry_run_report(),
        "policy_dry_run_fail_report": _assert_policy_dry_run_fail_report(),
        "fail_report": _assert_fail_report(),
        "summary_only_unjoinable_report": _assert_summary_only_unjoinable_report(),
        "missing_fields_report": _assert_missing_fields_report(),
    }
    print("relaykv_runtime_candidate_join_report_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
