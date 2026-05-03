from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    assess_relaykv_actual_host_backup_copy_readiness_for_smoke,
    assess_relaykv_host_backup_copy_readiness_for_smoke,
    assess_relaykv_readonly_attention_readiness_for_smoke,
    assess_relaykv_readonly_materialization_readiness_for_smoke,
    build_relaykv_candidate_event_materialization_results_for_smoke,
    build_relaykv_host_backup_copy_boundary_results_for_smoke,
    build_relaykv_host_backup_copy_requests_for_smoke,
    build_relaykv_policy_dry_run_events_for_smoke,
    build_relaykv_readonly_runtime_candidate_join_report_for_smoke,
    join_runtime_observation_with_host_backup_candidates_for_smoke,
    summarize_relaykv_host_backup_copy_boundary_results_for_smoke,
    summarize_relaykv_host_backup_copy_requests_for_smoke,
    summarize_host_backup_copy_candidates_for_smoke,
    summarize_relaykv_materialization_results_for_smoke,
    summarize_relaykv_policy_dry_run_events_for_smoke,
)
from sglang.srt.relaykv.observation import summarize_runtime_observation_payloads


def _runtime_payload(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
) -> dict[str, Any]:
    return {
        "event_type": "runtime_observation_readonly_metadata_candidate",
        "batch_id": "readonly-diagnostic-batch-a",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "req_pool_index": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "phase": "forward",
        "runtime_policy_state": "runtime_observation",
        "source": "forward_batch_readonly_runtime_observation_metadata",
        "source_mutated": False,
        "attention_override": False,
        "kv_cache_mutation": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
    }


def _candidate_event(
    request_id: str,
    req_pool_idx: int,
    layer_idx: int,
) -> dict[str, Any]:
    selected_block_ids = [req_pool_idx, req_pool_idx + 1]
    return {
        "runtime_policy_state": "applied_candidate",
        "batch_id": "readonly-diagnostic-batch-a",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "req_pool_index": req_pool_idx,
        "layer_idx": layer_idx,
        "selected_block_ids": selected_block_ids,
        "candidate_block_ids": [*selected_block_ids, req_pool_idx + 2],
        "snapshot_created": True,
        "host_backup_copy_candidate": True,
        "host_backup_copy_executed": True,
        "copy_equal": True,
        "source_mutated": False,
        "attention_override": False,
        "kv_cache_mutation": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
    }


def _runtime_payloads() -> list[dict[str, Any]]:
    return [
        _runtime_payload("rid-a", 10, 512, 0),
        _runtime_payload("rid-a", 10, 512, 14),
        _runtime_payload("rid-b", 11, 1024, 0),
        _runtime_payload("rid-b", 11, 1024, 14),
    ]


def _candidate_events() -> list[dict[str, Any]]:
    return [
        _candidate_event("rid-a", 10, 0),
        _candidate_event("rid-a", 10, 14),
        _candidate_event("rid-b", 11, 0),
        _candidate_event("rid-c", 12, 0),
    ]


def _block_metadata() -> dict[str, dict[str, list[int]]]:
    return {
        "rid-a": {
            "anchor_block_ids": [0],
            "recent_block_ids": [7, 8],
            "candidate_block_ids": [1, 2, 3, 4],
        },
        "rid-b": {
            "anchor_block_ids": [0, 1],
            "recent_block_ids": [15],
            "candidate_block_ids": [9, 10, 11, 12],
        },
    }


def _policy_config() -> dict[str, Any]:
    return {
        "kv_budget_tokens": 1024,
        "recent_tokens": 256,
        "anchor_tokens": 128,
        "transient_tokens": 64,
        "retrieval_top_k": 2,
        "layer_budget_policy": "uniform",
    }


def _assert_safety_zero(summary: dict[str, Any]) -> None:
    for key in (
        "source_mutated_true_count",
        "attention_override_true_count",
        "kv_cache_mutation_true_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
        "host_backup_copy_executed_count",
        "kv_pool_read_count",
        "kv_snapshot_count",
    ):
        if summary.get(key, 0) != 0:
            raise AssertionError(summary)


def _assert_readonly_flow() -> dict[str, Any]:
    runtime_payloads = _runtime_payloads()
    candidate_events = _candidate_events()
    block_metadata = _block_metadata()
    policy_config = _policy_config()
    runtime_before = copy.deepcopy(runtime_payloads)
    candidate_before = copy.deepcopy(candidate_events)
    block_before = copy.deepcopy(block_metadata)
    policy_before = copy.deepcopy(policy_config)

    runtime_summary = summarize_runtime_observation_payloads(runtime_payloads)
    if runtime_summary["total_payloads"] != 4:
        raise AssertionError(runtime_summary)

    candidate_summary = summarize_host_backup_copy_candidates_for_smoke(candidate_events)
    if candidate_summary["total_candidate_events"] != 4:
        raise AssertionError(candidate_summary)

    join_summary = join_runtime_observation_with_host_backup_candidates_for_smoke(
        runtime_payloads,
        candidate_events,
    )
    expected_join = {
        "joined_count": 3,
        "unmatched_runtime_count": 1,
        "unmatched_candidate_count": 1,
        "req_pool_idx_joined_count": 3,
    }
    for key, value in expected_join.items():
        if join_summary[key] != value:
            raise AssertionError(join_summary)

    dry_run_events = build_relaykv_policy_dry_run_events_for_smoke(
        runtime_payloads,
        block_metadata,
        policy_config,
    )
    dry_run_summary = summarize_relaykv_policy_dry_run_events_for_smoke(dry_run_events)
    if dry_run_summary["total_events"] != 4:
        raise AssertionError(dry_run_summary)
    if dry_run_summary["per_request_counts"] != {"rid-a": 2, "rid-b": 2}:
        raise AssertionError(dry_run_summary)
    if dry_run_summary["per_layer_counts"] != {"0": 2, "14": 2}:
        raise AssertionError(dry_run_summary)
    _assert_safety_zero(dry_run_summary)

    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        runtime_summary,
        candidate_summary,
        join_summary,
        policy_dry_run_summary=dry_run_summary,
    )
    materialization_readiness = (
        assess_relaykv_readonly_materialization_readiness_for_smoke(report)
    )
    if materialization_readiness["ready_for_materialization"] is not True:
        raise AssertionError(materialization_readiness)

    materialization_results = (
        build_relaykv_candidate_event_materialization_results_for_smoke(
            candidate_events,
            materialization_readiness,
        )
    )
    materialization_summary = summarize_relaykv_materialization_results_for_smoke(
        materialization_results
    )
    if materialization_summary["total_materialization_results"] != 4:
        raise AssertionError(materialization_summary)
    if materialization_summary["candidate_event_materialized_count"] != 4:
        raise AssertionError(materialization_summary)
    if materialization_summary["materialized_kv_count"] <= 0:
        raise AssertionError(materialization_summary)
    _assert_safety_zero(materialization_summary)

    copy_boundary_readiness = assess_relaykv_host_backup_copy_readiness_for_smoke(
        build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
            runtime_summary,
            candidate_summary,
            join_summary,
            policy_dry_run_summary=dry_run_summary,
            materialization_summary=materialization_summary,
        )
    )
    if copy_boundary_readiness["ready_for_host_backup_copy_boundary"] is not True:
        raise AssertionError(copy_boundary_readiness)

    host_backup_copy_requests = build_relaykv_host_backup_copy_requests_for_smoke(
        materialization_results,
        copy_boundary_readiness,
    )
    host_backup_copy_request_summary = (
        summarize_relaykv_host_backup_copy_requests_for_smoke(host_backup_copy_requests)
    )
    if host_backup_copy_request_summary["request_ready_count"] != 4:
        raise AssertionError(host_backup_copy_request_summary)
    if host_backup_copy_request_summary["blocked_count"] != 0:
        raise AssertionError(host_backup_copy_request_summary)
    _assert_safety_zero(host_backup_copy_request_summary)

    host_backup_copy_boundary_results = (
        build_relaykv_host_backup_copy_boundary_results_for_smoke(
            host_backup_copy_requests
        )
    )
    host_backup_copy_boundary_result_summary = (
        summarize_relaykv_host_backup_copy_boundary_results_for_smoke(
            host_backup_copy_boundary_results
        )
    )
    if host_backup_copy_boundary_result_summary["boundary_noop_count"] != 4:
        raise AssertionError(host_backup_copy_boundary_result_summary)
    if host_backup_copy_boundary_result_summary["blocked_count"] != 0:
        raise AssertionError(host_backup_copy_boundary_result_summary)
    _assert_safety_zero(host_backup_copy_boundary_result_summary)

    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        runtime_summary,
        candidate_summary,
        join_summary,
        policy_dry_run_summary=dry_run_summary,
        materialization_summary=materialization_summary,
        host_backup_copy_request_summary=host_backup_copy_request_summary,
        host_backup_copy_boundary_result_summary=(
            host_backup_copy_boundary_result_summary
        ),
    )
    if report["report_type"] != "relaykv_readonly_runtime_candidate_join_report":
        raise AssertionError(report)
    if report["policy_dry_run_included"] is not True:
        raise AssertionError(report)
    if report["policy_dry_run_total_events"] != 4:
        raise AssertionError(report)
    if report["materialization_summary_included"] is not True:
        raise AssertionError(report)
    if report["materialization_candidate_event_count"] != 4:
        raise AssertionError(report)
    if report["materialized_kv_count"] <= 0:
        raise AssertionError(report)
    if report["host_backup_copy_request_summary_included"] is not True:
        raise AssertionError(report)
    if report["host_backup_copy_request_ready_count"] != 4:
        raise AssertionError(report)
    if report["host_backup_copy_boundary_result_summary_included"] is not True:
        raise AssertionError(report)
    if report["host_backup_copy_boundary_noop_count"] != 4:
        raise AssertionError(report)
    if report["overall_safety_status"] != "pass":
        raise AssertionError(report)
    _assert_safety_zero(report)

    attention_readiness = assess_relaykv_readonly_attention_readiness_for_smoke(
        report
    )
    if attention_readiness["ready_for_attention_connection"] is not True:
        raise AssertionError(attention_readiness)
    if (
        attention_readiness["readiness_state"]
        != "ready_for_attention_connection_metadata_only"
    ):
        raise AssertionError(attention_readiness)

    actual_copy_readiness = assess_relaykv_actual_host_backup_copy_readiness_for_smoke(
        report
    )
    if actual_copy_readiness["ready_for_actual_host_backup_copy"] is not True:
        raise AssertionError(actual_copy_readiness)
    if (
        actual_copy_readiness["readiness_state"]
        != "ready_for_actual_host_backup_copy_smoke_boundary_complete"
    ):
        raise AssertionError(actual_copy_readiness)

    if runtime_payloads != runtime_before:
        raise AssertionError("runtime payloads were mutated")
    if candidate_events != candidate_before:
        raise AssertionError("candidate events were mutated")
    if block_metadata != block_before:
        raise AssertionError("block metadata was mutated")
    if policy_config != policy_before:
        raise AssertionError("policy config was mutated")

    return {
        "runtime_summary": runtime_summary,
        "candidate_summary": candidate_summary,
        "join_summary": join_summary,
        "dry_run_summary": dry_run_summary,
        "materialization_readiness": materialization_readiness,
        "materialization_summary": materialization_summary,
        "copy_boundary_readiness": copy_boundary_readiness,
        "host_backup_copy_request_summary": host_backup_copy_request_summary,
        "host_backup_copy_boundary_result_summary": (
            host_backup_copy_boundary_result_summary
        ),
        "report": report,
        "attention_readiness": attention_readiness,
        "actual_copy_readiness": actual_copy_readiness,
    }


def _assert_fail_propagation() -> dict[str, Any]:
    flow = _assert_readonly_flow()
    dry_run_summary = copy.deepcopy(flow["dry_run_summary"])
    dry_run_summary["runtime_writeback_true_count"] = 1
    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        flow["runtime_summary"],
        flow["candidate_summary"],
        flow["join_summary"],
        policy_dry_run_summary=dry_run_summary,
    )
    if report["overall_safety_status"] != "fail":
        raise AssertionError(report)
    if report["runtime_writeback_true_count"] != 1:
        raise AssertionError(report)
    return {"report": report}


def _assert_readiness_cases() -> dict[str, Any]:
    flow = _assert_readonly_flow()
    report = flow["report"]
    report_before = copy.deepcopy(report)
    readiness = assess_relaykv_readonly_materialization_readiness_for_smoke(report)
    if report != report_before:
        raise AssertionError("readiness helper mutated report")
    if readiness["ready_for_materialization"] is not True:
        raise AssertionError(readiness)
    if (
        readiness["readiness_state"]
        != "ready_for_safe_materialization_dry_run_complete"
    ):
        raise AssertionError(readiness)
    if readiness["blocking_reasons"] != []:
        raise AssertionError(readiness)

    safety_report = copy.deepcopy(report)
    safety_report["kv_cache_mutation_true_count"] = 1
    safety_readiness = assess_relaykv_readonly_materialization_readiness_for_smoke(
        safety_report
    )
    if safety_readiness["ready_for_materialization"] is not False:
        raise AssertionError(safety_readiness)
    if "safety_counter_nonzero" not in safety_readiness["blocking_reasons"]:
        raise AssertionError(safety_readiness)

    summary_only_report = copy.deepcopy(report)
    summary_only_report["join_granularity"] = "summary_only_unjoinable"
    summary_only_readiness = assess_relaykv_readonly_materialization_readiness_for_smoke(
        summary_only_report
    )
    if summary_only_readiness["ready_for_materialization"] is not False:
        raise AssertionError(summary_only_readiness)
    if "summary_only_unjoinable" not in summary_only_readiness["blocking_reasons"]:
        raise AssertionError(summary_only_readiness)

    missing_policy_report = copy.deepcopy(report)
    missing_policy_report["policy_dry_run_included"] = False
    missing_policy_report["policy_dry_run_total_events"] = 0
    missing_policy_readiness = (
        assess_relaykv_readonly_materialization_readiness_for_smoke(
            missing_policy_report
        )
    )
    if missing_policy_readiness["ready_for_materialization"] is not False:
        raise AssertionError(missing_policy_readiness)
    if "policy_dry_run_missing" not in missing_policy_readiness["blocking_reasons"]:
        raise AssertionError(missing_policy_readiness)

    missing_req_pool_report = copy.deepcopy(report)
    missing_req_pool_report["req_pool_idx_missing_count"] = 1
    missing_req_pool_readiness = (
        assess_relaykv_readonly_materialization_readiness_for_smoke(
            missing_req_pool_report
        )
    )
    if missing_req_pool_readiness["ready_for_materialization"] is not False:
        raise AssertionError(missing_req_pool_readiness)
    if "req_pool_idx_missing" not in missing_req_pool_readiness["blocking_reasons"]:
        raise AssertionError(missing_req_pool_readiness)

    no_join_report = copy.deepcopy(report)
    no_join_report["joined_count"] = 0
    no_join_readiness = assess_relaykv_readonly_materialization_readiness_for_smoke(
        no_join_report
    )
    if no_join_readiness["ready_for_materialization"] is not False:
        raise AssertionError(no_join_readiness)
    if "no_joined_events" not in no_join_readiness["blocking_reasons"]:
        raise AssertionError(no_join_readiness)

    return {
        "ready": readiness,
        "safety_counter_nonzero": safety_readiness,
        "summary_only_unjoinable": summary_only_readiness,
        "policy_dry_run_missing": missing_policy_readiness,
        "req_pool_idx_missing": missing_req_pool_readiness,
        "no_joined_events": no_join_readiness,
    }


def _assert_attention_readiness_cases() -> dict[str, Any]:
    flow = _assert_readonly_flow()
    report = flow["report"]
    report_before = copy.deepcopy(report)
    ready = assess_relaykv_readonly_attention_readiness_for_smoke(report)
    if report != report_before:
        raise AssertionError("attention readiness helper mutated report")
    if ready["ready_for_attention_connection"] is not True:
        raise AssertionError(ready)
    if ready["readiness_state"] != "ready_for_attention_connection_metadata_only":
        raise AssertionError(ready)

    missing_materialization_report = copy.deepcopy(report)
    missing_materialization_report["materialization_summary_included"] = False
    missing_materialization_report["materialization_total_results"] = 0
    missing_materialization_report["materialization_result_count"] = 0
    missing_materialization_report["materialized_kv_count"] = 0
    missing_materialization = assess_relaykv_readonly_attention_readiness_for_smoke(
        missing_materialization_report
    )
    if missing_materialization["ready_for_attention_connection"] is not False:
        raise AssertionError(missing_materialization)
    if (
        "materialization_summary_missing"
        not in missing_materialization["blocking_reasons"]
    ):
        raise AssertionError(missing_materialization)

    guarded_noop_report = copy.deepcopy(report)
    guarded_noop_report["materialization_guarded_noop_count"] = 1
    guarded_noop = assess_relaykv_readonly_attention_readiness_for_smoke(
        guarded_noop_report
    )
    if guarded_noop["ready_for_attention_connection"] is not False:
        raise AssertionError(guarded_noop)
    if "guarded_noop_present" not in guarded_noop["blocking_reasons"]:
        raise AssertionError(guarded_noop)

    blocked_report = copy.deepcopy(report)
    blocked_report["materialization_blocked_count"] = 1
    blocked = assess_relaykv_readonly_attention_readiness_for_smoke(blocked_report)
    if blocked["ready_for_attention_connection"] is not False:
        raise AssertionError(blocked)
    if "materialization_blocked" not in blocked["blocking_reasons"]:
        raise AssertionError(blocked)

    error_report = copy.deepcopy(report)
    error_report["materialization_error_count"] = 1
    error = assess_relaykv_readonly_attention_readiness_for_smoke(error_report)
    if error["ready_for_attention_connection"] is not False:
        raise AssertionError(error)
    if "materialization_error" not in error["blocking_reasons"]:
        raise AssertionError(error)

    host_backup_report = copy.deepcopy(report)
    host_backup_report["overall_safety_status"] = "fail"
    host_backup_report["host_backup_copy_executed_count"] = 1
    host_backup = assess_relaykv_readonly_attention_readiness_for_smoke(
        host_backup_report
    )
    if host_backup["ready_for_attention_connection"] is not False:
        raise AssertionError(host_backup)
    if "host_backup_copy_executed" not in host_backup["blocking_reasons"]:
        raise AssertionError(host_backup)
    if "overall_safety_not_pass" not in host_backup["blocking_reasons"]:
        raise AssertionError(host_backup)

    kv_pool_report = copy.deepcopy(report)
    kv_pool_report["overall_safety_status"] = "fail"
    kv_pool_report["kv_pool_read_count"] = 1
    kv_pool = assess_relaykv_readonly_attention_readiness_for_smoke(kv_pool_report)
    if kv_pool["ready_for_attention_connection"] is not False:
        raise AssertionError(kv_pool)
    if "kv_pool_read" not in kv_pool["blocking_reasons"]:
        raise AssertionError(kv_pool)

    no_kv_report = copy.deepcopy(report)
    no_kv_report["materialized_kv_count"] = 0
    no_kv = assess_relaykv_readonly_attention_readiness_for_smoke(no_kv_report)
    if no_kv["ready_for_attention_connection"] is not False:
        raise AssertionError(no_kv)
    if "no_materialized_kv" not in no_kv["blocking_reasons"]:
        raise AssertionError(no_kv)

    return {
        "ready": ready,
        "materialization_summary_missing": missing_materialization,
        "guarded_noop_present": guarded_noop,
        "materialization_blocked": blocked,
        "materialization_error": error,
        "host_backup_copy_executed": host_backup,
        "kv_pool_read": kv_pool,
        "no_materialized_kv": no_kv,
    }


def _assert_actual_copy_readiness_cases() -> dict[str, Any]:
    flow = _assert_readonly_flow()
    report = flow["report"]
    report_before = copy.deepcopy(report)
    ready = assess_relaykv_actual_host_backup_copy_readiness_for_smoke(report)
    if report != report_before:
        raise AssertionError("actual copy readiness helper mutated report")
    if ready["ready_for_actual_host_backup_copy"] is not True:
        raise AssertionError(ready)
    if (
        ready["readiness_state"]
        != "ready_for_actual_host_backup_copy_smoke_boundary_complete"
    ):
        raise AssertionError(ready)

    cases = {
        "copy_request_summary_missing": {
            "host_backup_copy_request_summary_included": False,
            "host_backup_copy_request_ready_count": 0,
            "host_backup_copy_request_blocked_count": 0,
        },
        "no_copy_requests_ready": {
            "host_backup_copy_request_ready_count": 0,
        },
        "copy_request_blocked": {
            "host_backup_copy_request_blocked_count": 1,
        },
        "boundary_result_summary_missing": {
            "host_backup_copy_boundary_result_summary_included": False,
            "host_backup_copy_boundary_noop_count": 0,
            "host_backup_copy_boundary_blocked_count": 0,
            "host_backup_copy_boundary_error_count": 0,
        },
        "no_boundary_noop_results": {
            "host_backup_copy_boundary_noop_count": 0,
        },
        "boundary_result_blocked": {
            "host_backup_copy_boundary_blocked_count": 1,
        },
        "boundary_result_error": {
            "host_backup_copy_boundary_error_count": 1,
        },
        "host_backup_copy_already_executed": {
            "overall_safety_status": "fail",
            "host_backup_copy_executed_count": 1,
        },
        "kv_pool_read_observed": {
            "overall_safety_status": "fail",
            "kv_pool_read_count": 1,
        },
        "kv_snapshot_observed": {
            "overall_safety_status": "fail",
            "kv_snapshot_count": 1,
        },
    }
    observed: dict[str, dict[str, Any]] = {"ready": ready}
    for reason, updates in cases.items():
        failed_report = copy.deepcopy(report)
        failed_report.update(updates)
        readiness = assess_relaykv_actual_host_backup_copy_readiness_for_smoke(
            failed_report
        )
        if readiness["ready_for_actual_host_backup_copy"] is not False:
            raise AssertionError(readiness)
        if reason not in readiness["blocking_reasons"]:
            raise AssertionError(readiness)
        observed[reason] = readiness
    return observed


def _assert_summary_only_candidate_flow() -> dict[str, Any]:
    runtime_payloads = _runtime_payloads()
    runtime_summary = summarize_runtime_observation_payloads(runtime_payloads)
    candidate_summary = summarize_host_backup_copy_candidates_for_smoke(
        _candidate_events()
    )
    join_summary = join_runtime_observation_with_host_backup_candidates_for_smoke(
        runtime_summary,
        candidate_summary,
    )
    if join_summary["join_granularity"] != "summary_only_unjoinable":
        raise AssertionError(join_summary)
    report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        runtime_summary,
        candidate_summary,
        join_summary,
    )
    if report["join_granularity"] != "summary_only_unjoinable":
        raise AssertionError(report)
    if report["overall_safety_status"] != "pass":
        raise AssertionError(report)
    return {"join_summary": join_summary, "report": report}


def main() -> None:
    result = {
        "readonly_flow": _assert_readonly_flow(),
        "fail_propagation": _assert_fail_propagation(),
        "readiness_cases": _assert_readiness_cases(),
        "attention_readiness_cases": _assert_attention_readiness_cases(),
        "actual_copy_readiness_cases": _assert_actual_copy_readiness_cases(),
        "summary_only_candidate_flow": _assert_summary_only_candidate_flow(),
    }
    print("relaykv_readonly_diagnostic_flow_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
