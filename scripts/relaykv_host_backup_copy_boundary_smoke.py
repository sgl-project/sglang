from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    assess_relaykv_host_backup_copy_readiness_for_smoke,
    build_relaykv_host_backup_copy_requests_for_smoke,
    summarize_relaykv_host_backup_copy_requests_for_smoke,
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


def _ready_report() -> dict[str, Any]:
    return {
        "report_generated_from_readonly_inputs": True,
        "overall_safety_status": "pass",
        "policy_dry_run_included": True,
        "policy_dry_run_total_events": 2,
        "materialization_summary_included": True,
        "materialization_total_results": 2,
        "materialization_result_count": 2,
        "materialized_kv_count": 4,
        "materialization_candidate_event_count": 2,
        "materialization_fake_count": 0,
        "materialization_guarded_noop_count": 0,
        "materialization_blocked_count": 0,
        "materialization_error_count": 0,
        "host_backup_copy_executed_count": 0,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }


def _materialization_result(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
    materialized_block_ids: list[int],
    candidate_block_ids: list[int],
    *,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event_type": "relaykv_materialization_result",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "materialization_state": "candidate_event_materialized",
        "materialization_mode": "candidate_event",
        "selected_block_ids": list(materialized_block_ids),
        "materialized_block_ids": list(materialized_block_ids),
        "retrieved_block_ids": list(materialized_block_ids),
        "candidate_block_ids": candidate_block_ids,
        "anchor_block_ids": [],
        "recent_block_ids": [],
        "materialized_kv_count": len(materialized_block_ids),
        "materialized_token_count": 0,
        "source": "host_backup_candidate_event_materialization",
    }
    if poison is not None:
        result["unrelated_tensor_like"] = poison
    return result


def _materialization_results(
    poison: _PoisonTensorLike | None = None,
) -> list[dict[str, Any]]:
    return [
        _materialization_result(
            "rid-a",
            10,
            512,
            0,
            [1, 2],
            [1, 2, 3, 4],
            poison=poison,
        ),
        _materialization_result(
            "rid-b",
            11,
            1024,
            14,
            [9, 10],
            [9, 10, 11, 12],
        ),
    ]


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
        if summary[key] != 0:
            raise AssertionError(summary)


def _assert_full_pass_flow() -> dict[str, Any]:
    report = _ready_report()
    report_before = copy.deepcopy(report)
    readiness = assess_relaykv_host_backup_copy_readiness_for_smoke(report)
    if report != report_before:
        raise AssertionError("report was mutated")
    if readiness["ready_for_host_backup_copy_boundary"] is not True:
        raise AssertionError(readiness)
    if readiness["readiness_state"] != "ready_for_host_backup_copy_boundary_smoke":
        raise AssertionError(readiness)

    poison = _PoisonTensorLike()
    results = _materialization_results(poison)
    results_before = copy.deepcopy(results)
    requests = build_relaykv_host_backup_copy_requests_for_smoke(results, readiness)
    if results != results_before:
        raise AssertionError("materialization results were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(requests) != 2:
        raise AssertionError(requests)
    for request in requests:
        if request["copy_state"] != "request_ready":
            raise AssertionError(request)
        if request["copy_destination"] != "materialization_result_only":
            raise AssertionError(request)
        if request["copy_guard_state"] != "pre_attention_no_runtime_writeback":
            raise AssertionError(request)
        if request["host_backup_copy_executed"] is not False:
            raise AssertionError(request)
        if request["kv_pool_read"] is not False:
            raise AssertionError(request)
        if request["kv_snapshot"] is not False:
            raise AssertionError(request)
    summary = summarize_relaykv_host_backup_copy_requests_for_smoke(requests)
    if summary["total_copy_requests"] != 2:
        raise AssertionError(summary)
    if summary["request_ready_count"] != 2:
        raise AssertionError(summary)
    if summary["blocked_count"] != 0:
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 4:
        raise AssertionError(summary)
    if summary["per_request_counts"] != {"rid-a": 1, "rid-b": 1}:
        raise AssertionError(summary)
    if summary["per_layer_counts"] != {"0": 1, "14": 1}:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"readiness": readiness, "requests": requests, "summary": summary}


def _assert_readiness_fail_cases() -> dict[str, Any]:
    cases = {
        "materialization_summary_missing": (
            {"materialization_summary_included": False},
            "materialization_summary_missing",
        ),
        "candidate_event_materialization_missing": (
            {"materialization_candidate_event_count": 0},
            "candidate_event_materialization_missing",
        ),
        "guarded_noop_present": (
            {"materialization_guarded_noop_count": 1},
            "guarded_noop_present",
        ),
        "materialization_blocked": (
            {"materialization_blocked_count": 1},
            "materialization_blocked",
        ),
        "materialization_error": (
            {"materialization_error_count": 1},
            "materialization_error",
        ),
        "host_backup_copy_already_executed": (
            {"overall_safety_status": "fail", "host_backup_copy_executed_count": 1},
            "host_backup_copy_already_executed",
        ),
        "kv_pool_read_observed": (
            {"overall_safety_status": "fail", "kv_pool_read_count": 1},
            "kv_pool_read_observed",
        ),
        "kv_snapshot_observed": (
            {"overall_safety_status": "fail", "kv_snapshot_count": 1},
            "kv_snapshot_observed",
        ),
        "safety_counter_nonzero": (
            {"overall_safety_status": "fail", "runtime_writeback_true_count": 1},
            "safety_counter_nonzero",
        ),
    }
    observed: dict[str, dict[str, Any]] = {}
    for name, (updates, expected_reason) in cases.items():
        report = _ready_report()
        report.update(updates)
        readiness = assess_relaykv_host_backup_copy_readiness_for_smoke(report)
        if readiness["ready_for_host_backup_copy_boundary"] is not False:
            raise AssertionError(readiness)
        if expected_reason not in readiness["blocking_reasons"]:
            raise AssertionError(readiness)
        observed[name] = readiness
    return observed


def _assert_requests_blocked_by_readiness() -> dict[str, Any]:
    readiness = {
        "ready_for_host_backup_copy_boundary": False,
        "readiness_state": "blocked_kv_pool_read_observed",
        "blocking_reasons": ["blocked_kv_pool_read_observed"],
    }
    requests = build_relaykv_host_backup_copy_requests_for_smoke(
        _materialization_results(),
        readiness,
    )
    for request in requests:
        if request["copy_state"] != "blocked":
            raise AssertionError(request)
        if "blocked_kv_pool_read_observed" not in request["blocking_reasons"]:
            raise AssertionError(request)
    summary = summarize_relaykv_host_backup_copy_requests_for_smoke(requests)
    if summary["blocked_count"] != len(requests):
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"requests": requests, "summary": summary}


def _assert_non_candidate_event_blocks() -> dict[str, Any]:
    result = _materialization_result("rid-noop", 12, 128, 0, [3], [3, 4])
    result["materialization_state"] = "guarded_noop"
    result["materialization_mode"] = "noop_guarded"
    requests = build_relaykv_host_backup_copy_requests_for_smoke(
        [result],
        assess_relaykv_host_backup_copy_readiness_for_smoke(_ready_report()),
    )
    request = requests[0]
    if request["copy_state"] != "blocked":
        raise AssertionError(request)
    if "not_candidate_event_materialized" not in request["blocking_reasons"]:
        raise AssertionError(request)
    summary = summarize_relaykv_host_backup_copy_requests_for_smoke(requests)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"requests": requests, "summary": summary}


def _assert_missing_readiness_warns() -> dict[str, Any]:
    requests = build_relaykv_host_backup_copy_requests_for_smoke(
        _materialization_results()
    )
    for request in requests:
        if request["copy_state"] != "request_ready":
            raise AssertionError(request)
        if "copy_readiness_not_provided" not in request["warning_reasons"]:
            raise AssertionError(request)
    summary = summarize_relaykv_host_backup_copy_requests_for_smoke(requests)
    if summary["request_ready_count"] != 2:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"requests": requests, "summary": summary}


def _assert_empty_materialized_ids_block() -> dict[str, Any]:
    result = _materialization_result("rid-empty", 13, 64, 0, [], [1, 2])
    requests = build_relaykv_host_backup_copy_requests_for_smoke(
        [result],
        assess_relaykv_host_backup_copy_readiness_for_smoke(_ready_report()),
    )
    request = requests[0]
    if request["copy_state"] != "blocked":
        raise AssertionError(request)
    if "no_materialized_blocks" not in request["blocking_reasons"]:
        raise AssertionError(request)
    summary = summarize_relaykv_host_backup_copy_requests_for_smoke(requests)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"requests": requests, "summary": summary}


def main() -> None:
    result = {
        "full_pass_flow": _assert_full_pass_flow(),
        "readiness_fail_cases": _assert_readiness_fail_cases(),
        "requests_blocked_by_readiness": _assert_requests_blocked_by_readiness(),
        "non_candidate_event_blocks": _assert_non_candidate_event_blocks(),
        "missing_readiness_warns": _assert_missing_readiness_warns(),
        "empty_materialized_ids_block": _assert_empty_materialized_ids_block(),
    }
    print("relaykv_host_backup_copy_boundary_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
