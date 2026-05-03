from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_policy_dry_run_events_for_smoke,
    build_relaykv_readonly_runtime_candidate_join_report_for_smoke,
    join_runtime_observation_with_host_backup_candidates_for_smoke,
    summarize_host_backup_copy_candidates_for_smoke,
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
    return {
        "runtime_policy_state": "applied_candidate",
        "batch_id": "readonly-diagnostic-batch-a",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "req_pool_index": req_pool_idx,
        "layer_idx": layer_idx,
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
    ):
        if summary[key] != 0:
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
    if report["report_type"] != "relaykv_readonly_runtime_candidate_join_report":
        raise AssertionError(report)
    if report["policy_dry_run_included"] is not True:
        raise AssertionError(report)
    if report["policy_dry_run_total_events"] != 4:
        raise AssertionError(report)
    if report["overall_safety_status"] != "pass":
        raise AssertionError(report)
    _assert_safety_zero(report)

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
        "report": report,
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
        "summary_only_candidate_flow": _assert_summary_only_candidate_flow(),
    }
    print("relaykv_readonly_diagnostic_flow_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
