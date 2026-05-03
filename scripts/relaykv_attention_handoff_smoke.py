from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    assess_relaykv_attention_handoff_readiness_for_smoke,
    build_relaykv_actual_host_backup_copy_report_for_smoke,
    build_relaykv_actual_host_backup_copy_results_for_smoke,
    build_relaykv_attention_handoff_candidates_for_smoke,
    build_relaykv_readonly_runtime_candidate_join_report_for_smoke,
    summarize_relaykv_actual_host_backup_copy_results_for_smoke,
    summarize_relaykv_attention_handoff_candidates_for_smoke,
)


class _PoisonTensorLike:
    def __init__(self) -> None:
        self.cpu_called = False
        self.item_called = False
        self.tolist_called = False
        self.iter_called = False
        self.len_called = False
        self.getitem_called = False
        self.shape = (1,)
        self.device = "cuda:0"
        self.dtype = "torch.int64"

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PoisonTensorLike":
        return self

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def __iter__(self):
        self.iter_called = True
        raise AssertionError("__iter__() must not be called")

    def __len__(self) -> int:
        self.len_called = True
        raise AssertionError("__len__() must not be called")

    def __getitem__(self, index: int) -> None:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

    @property
    def forbidden_access_called(self) -> bool:
        return (
            self.cpu_called
            or self.item_called
            or self.tolist_called
            or self.iter_called
            or self.len_called
            or self.getitem_called
        )


def _runtime_summary() -> dict[str, Any]:
    return {
        "total_payloads": 2,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _candidate_summary() -> dict[str, Any]:
    return {
        "total_candidate_events": 2,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _join_summary() -> dict[str, Any]:
    return {
        "join_granularity": "per_event",
        "total_runtime_payloads": 2,
        "total_host_backup_candidate_events": 2,
        "joined_count": 2,
        "unmatched_runtime_count": 0,
        "unmatched_candidate_count": 0,
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
        "total_events": 2,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _materialization_summary() -> dict[str, Any]:
    return {
        "total_materialization_results": 2,
        "materialized_result_count": 2,
        "candidate_event_materialized_count": 2,
        "fake_materialized_count": 0,
        "guarded_noop_count": 0,
        "host_backup_copy_materialized_count": 0,
        "blocked_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "materialized_kv_count": 4,
        "materialized_token_count": 0,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "host_backup_copy_executed_count": 0,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
    }


def _copy_request_summary() -> dict[str, Any]:
    return {
        "total_copy_requests": 2,
        "request_ready_count": 2,
        "blocked_count": 0,
        "materialized_kv_count": 4,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "host_backup_copy_executed_count": 0,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
    }


def _boundary_summary() -> dict[str, Any]:
    return {
        "total_boundary_results": 2,
        "boundary_noop_count": 2,
        "blocked_count": 0,
        "error_count": 0,
        "materialized_kv_count": 4,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "host_backup_copy_executed_count": 0,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
    }


def _actual_copy_readiness() -> dict[str, Any]:
    return {
        "ready_for_actual_host_backup_copy": True,
        "readiness_state": "ready_for_actual_host_backup_copy_smoke_boundary_complete",
        "blocking_reasons": [],
    }


def _build_actual_copy_report() -> dict[str, Any]:
    readonly_report = build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        _runtime_summary(),
        _candidate_summary(),
        _join_summary(),
        policy_dry_run_summary=_policy_dry_run_summary(),
        materialization_summary=_materialization_summary(),
        host_backup_copy_request_summary=_copy_request_summary(),
        host_backup_copy_boundary_result_summary=_boundary_summary(),
    )
    actual_copy_requests = [
        {
            "event_type": "relaykv_host_backup_copy_request",
            "request_id": "rid-a",
            "req_pool_idx": 10,
            "seq_len": 512,
            "layer_id": 0,
            "selected_block_ids": [1, 2],
            "materialized_block_ids": [1, 2],
            "retrieved_block_ids": [1, 2],
            "candidate_block_ids": [1, 2, 3, 4],
            "anchor_block_ids": [0],
            "recent_block_ids": [7, 8],
            "materialized_kv_count": 2,
            "materialized_token_count": 0,
            "copy_state": "request_ready",
        },
        {
            "event_type": "relaykv_host_backup_copy_request",
            "request_id": "rid-b",
            "req_pool_idx": 11,
            "seq_len": 1024,
            "layer_id": 14,
            "selected_block_ids": [9, 10],
            "materialized_block_ids": [9, 10],
            "retrieved_block_ids": [9, 10],
            "candidate_block_ids": [9, 10, 11, 12],
            "anchor_block_ids": [0, 1],
            "recent_block_ids": [15],
            "materialized_kv_count": 2,
            "materialized_token_count": 0,
            "copy_state": "request_ready",
        },
    ]
    actual_copy_results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        actual_copy_requests,
        _actual_copy_readiness(),
        execute_copy=True,
    )
    actual_copy_summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(
        actual_copy_results
    )
    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        readonly_report,
        actual_copy_summary,
    )
    return report


def _actual_copy_results(poison: _PoisonTensorLike | None = None) -> list[dict[str, Any]]:
    results = [
        {
            "event_type": "relaykv_materialization_result",
            "materialization_state": "host_backup_copy_materialized",
            "materialization_mode": "host_backup_copy",
            "copy_state": "copy_executed",
            "request_id": "rid-a",
            "req_pool_idx": 10,
            "seq_len": 512,
            "layer_id": 0,
            "materialized_block_ids": [1, 2],
            "retrieved_block_ids": [1, 2],
            "candidate_block_ids": [1, 2, 3, 4],
            "anchor_block_ids": [0],
            "recent_block_ids": [7, 8],
            "materialized_kv_count": 2,
            "materialized_token_count": 0,
        },
        {
            "event_type": "relaykv_materialization_result",
            "materialization_state": "host_backup_copy_materialized",
            "materialization_mode": "host_backup_copy",
            "copy_state": "copy_executed",
            "request_id": "rid-b",
            "req_pool_idx": 11,
            "seq_len": 1024,
            "layer_id": 14,
            "materialized_block_ids": [9, 10],
            "retrieved_block_ids": [9, 10],
            "candidate_block_ids": [9, 10, 11, 12],
            "anchor_block_ids": [0, 1],
            "recent_block_ids": [15],
            "materialized_kv_count": 2,
            "materialized_token_count": 0,
        },
    ]
    if poison is not None:
        results[0]["unrelated_tensor_like"] = poison
    return results


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    report = _build_actual_copy_report()
    report_before = copy.deepcopy(report)
    actual_copy_results = _actual_copy_results(poison)
    results_before = copy.deepcopy(actual_copy_results)

    readiness = assess_relaykv_attention_handoff_readiness_for_smoke(report)
    if readiness["ready_for_attention_handoff"] is not True:
        raise AssertionError(readiness)
    if readiness["readiness_state"] != "ready_for_attention_handoff_metadata_only":
        raise AssertionError(readiness)

    candidates = build_relaykv_attention_handoff_candidates_for_smoke(
        actual_copy_results,
        readiness,
    )
    if report != report_before:
        raise AssertionError("actual copy report was mutated")
    if actual_copy_results != results_before:
        raise AssertionError("actual copy results were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(candidates) != 2:
        raise AssertionError(candidates)
    for candidate in candidates:
        if candidate["handoff_state"] != "handoff_ready":
            raise AssertionError(candidate)
        if candidate["attention_target_backend"] != "unconnected":
            raise AssertionError(candidate)
        if candidate["attention_connection_attempted"] is not False:
            raise AssertionError(candidate)
        if candidate["attention_override"] is not False:
            raise AssertionError(candidate)
        if candidate["kv_pool_read"] is not False or candidate["kv_snapshot"] is not False:
            raise AssertionError(candidate)
    summary = summarize_relaykv_attention_handoff_candidates_for_smoke(candidates)
    if summary["handoff_ready_count"] != 2:
        raise AssertionError(summary)
    if summary["blocked_count"] != 0:
        raise AssertionError(summary)
    if summary["working_kv_block_count"] != 4:
        raise AssertionError(summary)
    if summary["attention_connection_attempted_count"] != 0:
        raise AssertionError(summary)
    expected_zero = (
        "attention_override_true_count",
        "attention_override_noop_count",
        "kv_pool_read_count",
        "kv_snapshot_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
        "kv_cache_mutation_true_count",
        "source_mutated_true_count",
    )
    for key in expected_zero:
        if summary[key] != 0:
            raise AssertionError(summary)
    return {"readiness": readiness, "summary": summary}


def _assert_blocked_candidate_cases() -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []

    base_results = _actual_copy_results()
    candidates = build_relaykv_attention_handoff_candidates_for_smoke(base_results, None)
    if any(c["handoff_state"] != "blocked" for c in candidates):
        raise AssertionError(candidates)
    if any("attention_handoff_readiness_not_provided" not in c["blocking_reasons"] for c in candidates):
        raise AssertionError(candidates)
    blocked.append(summarize_relaykv_attention_handoff_candidates_for_smoke(candidates))

    readiness_blocked = {
        "ready_for_attention_handoff": False,
        "readiness_state": "blocked_kv_pool_read_observed",
        "blocking_reasons": ["blocked_kv_pool_read_observed"],
    }
    candidates = build_relaykv_attention_handoff_candidates_for_smoke(
        base_results,
        readiness_blocked,
    )
    if any("blocked_kv_pool_read_observed" not in c["blocking_reasons"] for c in candidates):
        raise AssertionError(candidates)
    blocked.append(summarize_relaykv_attention_handoff_candidates_for_smoke(candidates))

    wrong_event = copy.deepcopy(base_results)
    wrong_event[0]["event_type"] = "wrong"
    candidates = build_relaykv_attention_handoff_candidates_for_smoke(
        wrong_event,
        {"ready_for_attention_handoff": True, "readiness_state": "ready_for_attention_handoff_metadata_only", "blocking_reasons": []},
    )
    if "not_materialization_result" not in candidates[0]["blocking_reasons"]:
        raise AssertionError(candidates[0])

    wrong_state = copy.deepcopy(base_results)
    wrong_state[0]["materialization_state"] = "blocked"
    candidates = build_relaykv_attention_handoff_candidates_for_smoke(
        wrong_state,
        {"ready_for_attention_handoff": True, "readiness_state": "ready_for_attention_handoff_metadata_only", "blocking_reasons": []},
    )
    if "not_host_backup_copy_materialized" not in candidates[0]["blocking_reasons"]:
        raise AssertionError(candidates[0])

    wrong_copy = copy.deepcopy(base_results)
    wrong_copy[0]["copy_state"] = "blocked"
    candidates = build_relaykv_attention_handoff_candidates_for_smoke(
        wrong_copy,
        {"ready_for_attention_handoff": True, "readiness_state": "ready_for_attention_handoff_metadata_only", "blocking_reasons": []},
    )
    if "copy_not_executed" not in candidates[0]["blocking_reasons"]:
        raise AssertionError(candidates[0])

    empty_blocks = copy.deepcopy(base_results)
    empty_blocks[0]["materialized_block_ids"] = []
    candidates = build_relaykv_attention_handoff_candidates_for_smoke(
        empty_blocks,
        {"ready_for_attention_handoff": True, "readiness_state": "ready_for_attention_handoff_metadata_only", "blocking_reasons": []},
    )
    if "no_materialized_blocks" not in candidates[0]["blocking_reasons"]:
        raise AssertionError(candidates[0])

    return blocked


def _assert_readiness_fail_cases() -> list[dict[str, Any]]:
    report = _build_actual_copy_report()
    outputs: list[dict[str, Any]] = []

    mutated = dict(report)
    mutated["report_type"] = "not_actual_copy_report"
    readiness = assess_relaykv_attention_handoff_readiness_for_smoke(mutated)
    if readiness["readiness_state"] != "blocked_not_actual_copy_report":
        raise AssertionError(readiness)
    outputs.append(readiness)

    mutated = dict(report)
    mutated["actual_copy_safety_status"] = "fail"
    readiness = assess_relaykv_attention_handoff_readiness_for_smoke(mutated)
    if readiness["readiness_state"] != "blocked_actual_copy_safety_not_pass":
        raise AssertionError(readiness)
    outputs.append(readiness)

    for key, state in (
        ("actual_host_backup_copy_summary_included", "blocked_actual_copy_summary_missing"),
        ("actual_host_backup_copy_materialized_count", "blocked_no_actual_copy_materialized"),
        ("actual_host_backup_copy_executed_count", "blocked_actual_copy_not_executed"),
        ("actual_host_backup_copy_kv_pool_read_count", "blocked_kv_pool_read_observed"),
        ("actual_host_backup_copy_kv_snapshot_count", "blocked_kv_snapshot_observed"),
        ("attention_override_true_count", "blocked_attention_override_observed"),
        ("runtime_writeback_true_count", "blocked_runtime_writeback_observed"),
        ("scheduler_policy_noop_false_count", "blocked_scheduler_mutation_observed"),
    ):
        mutated = dict(report)
        if key == "actual_host_backup_copy_summary_included":
            mutated[key] = False
        elif key in (
            "actual_host_backup_copy_materialized_count",
            "actual_host_backup_copy_executed_count",
        ):
            mutated[key] = 0
        else:
            mutated[key] = 1
        readiness = assess_relaykv_attention_handoff_readiness_for_smoke(mutated)
        if readiness["readiness_state"] != state:
            raise AssertionError(readiness)
        outputs.append(readiness)

    return outputs


def main() -> None:
    pass_flow = _assert_pass_flow()
    blocked_candidate_cases = _assert_blocked_candidate_cases()
    readiness_fail_cases = _assert_readiness_fail_cases()
    print(
        json.dumps(
            {
                "pass_flow": {
                    "readiness_state": pass_flow["readiness"]["readiness_state"],
                    "handoff_ready_count": pass_flow["summary"]["handoff_ready_count"],
                    "working_kv_block_count": pass_flow["summary"]["working_kv_block_count"],
                },
                "blocked_candidate_case_count": len(blocked_candidate_cases),
                "readiness_fail_case_count": len(readiness_fail_cases),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
