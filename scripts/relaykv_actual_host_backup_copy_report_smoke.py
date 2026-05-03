from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    assess_relaykv_attention_connection_readiness_for_smoke,
    build_relaykv_actual_host_backup_copy_report_for_smoke,
    build_relaykv_readonly_runtime_candidate_join_report_for_smoke,
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


def _host_backup_copy_request_summary() -> dict[str, Any]:
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


def _host_backup_copy_boundary_result_summary() -> dict[str, Any]:
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


def _actual_copy_summary(poison: _PoisonTensorLike | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "summary_type": "relaykv_actual_host_backup_copy_result_summary",
        "total_copy_results": 2,
        "host_backup_copy_materialized_count": 2,
        "blocked_count": 0,
        "error_count": 0,
        "materialized_kv_count": 4,
        "materialized_token_count": 0,
        "per_request_counts": {"rid-a": 1, "rid-b": 1},
        "per_layer_counts": {"0": 1, "14": 1},
        "per_copy_state_counts": {"copy_executed": 2},
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "host_backup_copy_executed_count": 2,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
    }
    if poison is not None:
        summary["unrelated_tensor_like"] = poison
    return summary


def _build_readonly_report(
    *,
    boundary_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_relaykv_readonly_runtime_candidate_join_report_for_smoke(
        _runtime_summary(),
        _candidate_summary(),
        _join_summary(),
        policy_dry_run_summary=_policy_dry_run_summary(),
        materialization_summary=_materialization_summary(),
        host_backup_copy_request_summary=_host_backup_copy_request_summary(),
        host_backup_copy_boundary_result_summary=(
            boundary_summary
            if boundary_summary is not None
            else _host_backup_copy_boundary_result_summary()
        ),
    )


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    readonly_report = _build_readonly_report()
    actual_summary = _actual_copy_summary(poison)
    readonly_before = copy.deepcopy(readonly_report)
    summary_before = copy.deepcopy(actual_summary)

    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        readonly_report,
        actual_summary,
    )

    if readonly_report != readonly_before:
        raise AssertionError("readonly report was mutated")
    if actual_summary != summary_before:
        raise AssertionError("actual copy summary was mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")

    if report["report_type"] != "relaykv_actual_host_backup_copy_report":
        raise AssertionError(report)
    if report["actual_copy_report_generated_from_isolated_smoke_inputs"] is not True:
        raise AssertionError(report)
    if report["actual_copy_safety_status"] != "pass":
        raise AssertionError(report)
    if report["actual_host_backup_copy_summary_included"] is not True:
        raise AssertionError(report)
    if report["actual_host_backup_copy_materialized_count"] != 2:
        raise AssertionError(report)
    if report["actual_host_backup_copy_executed_count"] != 2:
        raise AssertionError(report)
    if report["actual_host_backup_copy_kv_pool_read_count"] != 0:
        raise AssertionError(report)
    if report["actual_host_backup_copy_kv_snapshot_count"] != 0:
        raise AssertionError(report)

    readiness = assess_relaykv_attention_connection_readiness_for_smoke(report)
    if readiness["ready_for_attention_connection"] is not True:
        raise AssertionError(readiness)
    if (
        readiness["readiness_state"]
        != "ready_for_attention_connection_design_only"
    ):
        raise AssertionError(readiness)
    return {"report": report, "readiness": readiness}


def _assert_readonly_report_still_fails_on_copy_executed() -> dict[str, Any]:
    boundary_summary = _host_backup_copy_boundary_result_summary()
    boundary_summary["host_backup_copy_executed_count"] = 1
    report = _build_readonly_report(boundary_summary=boundary_summary)
    if report["overall_safety_status"] != "fail":
        raise AssertionError(report)
    if report["host_backup_copy_executed_count"] != 1:
        raise AssertionError(report)
    return report


def _assert_fail_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    readonly_fail = _build_readonly_report()
    readonly_fail["overall_safety_status"] = "fail"
    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        readonly_fail,
        _actual_copy_summary(),
    )
    if report["actual_copy_safety_status"] != "fail":
        raise AssertionError(report)
    if "readonly_report_overall_safety_not_pass" not in report["actual_copy_safety_reasons"]:
        raise AssertionError(report)
    cases.append(report)

    missing_summary_report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        _build_readonly_report(),
        None,
    )
    if missing_summary_report["actual_copy_safety_status"] != "fail":
        raise AssertionError(missing_summary_report)
    if "actual_host_backup_copy_summary_missing" not in missing_summary_report["actual_copy_safety_reasons"]:
        raise AssertionError(missing_summary_report)
    cases.append(missing_summary_report)

    def _assert_blocked(summary: dict[str, Any], expected_state: str) -> dict[str, Any]:
        report = build_relaykv_actual_host_backup_copy_report_for_smoke(
            _build_readonly_report(),
            summary,
        )
        readiness = assess_relaykv_attention_connection_readiness_for_smoke(report)
        if readiness["ready_for_attention_connection"] is not False:
            raise AssertionError(readiness)
        if readiness["readiness_state"] != expected_state:
            raise AssertionError(readiness)
        return report

    summary = _actual_copy_summary()
    summary["host_backup_copy_materialized_count"] = 0
    _assert_blocked(summary, "blocked_no_actual_copy_materialized")

    summary = _actual_copy_summary()
    summary["host_backup_copy_executed_count"] = 0
    _assert_blocked(summary, "blocked_actual_copy_not_executed")

    summary = _actual_copy_summary()
    summary["blocked_count"] = 1
    _assert_blocked(summary, "blocked_actual_copy_blocked")

    summary = _actual_copy_summary()
    summary["error_count"] = 1
    _assert_blocked(summary, "blocked_actual_copy_error")

    summary = _actual_copy_summary()
    summary["kv_pool_read_count"] = 1
    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        _build_readonly_report(),
        summary,
    )
    if report["actual_copy_safety_status"] != "fail":
        raise AssertionError(report)
    readiness = assess_relaykv_attention_connection_readiness_for_smoke(report)
    if readiness["readiness_state"] not in (
        "blocked_actual_copy_safety_not_pass",
        "blocked_multiple_reasons",
    ):
        raise AssertionError(readiness)

    summary = _actual_copy_summary()
    summary["kv_snapshot_count"] = 1
    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        _build_readonly_report(),
        summary,
    )
    if report["actual_copy_safety_status"] != "fail":
        raise AssertionError(report)

    summary = _actual_copy_summary()
    summary["attention_override_true_count"] = 1
    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        _build_readonly_report(),
        summary,
    )
    if report["actual_copy_safety_status"] != "fail":
        raise AssertionError(report)

    summary = _actual_copy_summary()
    summary["runtime_writeback_true_count"] = 1
    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        _build_readonly_report(),
        summary,
    )
    if report["actual_copy_safety_status"] != "fail":
        raise AssertionError(report)

    summary = _actual_copy_summary()
    summary["scheduler_policy_noop_false_count"] = 1
    report = build_relaykv_actual_host_backup_copy_report_for_smoke(
        _build_readonly_report(),
        summary,
    )
    if report["actual_copy_safety_status"] != "fail":
        raise AssertionError(report)

    return cases


def main() -> None:
    pass_flow = _assert_pass_flow()
    readonly_fail = _assert_readonly_report_still_fails_on_copy_executed()
    fail_cases = _assert_fail_cases()

    print(
        json.dumps(
            {
                "actual_copy_report_pass": {
                    "report_type": pass_flow["report"]["report_type"],
                    "actual_copy_safety_status": pass_flow["report"][
                        "actual_copy_safety_status"
                    ],
                    "actual_host_backup_copy_materialized_count": pass_flow["report"][
                        "actual_host_backup_copy_materialized_count"
                    ],
                    "actual_host_backup_copy_executed_count": pass_flow["report"][
                        "actual_host_backup_copy_executed_count"
                    ],
                },
                "attention_connection_readiness_pass": {
                    "ready_for_attention_connection": pass_flow["readiness"][
                        "ready_for_attention_connection"
                    ],
                    "readiness_state": pass_flow["readiness"]["readiness_state"],
                },
                "readonly_report_fail_on_copy_executed": {
                    "overall_safety_status": readonly_fail["overall_safety_status"],
                    "host_backup_copy_executed_count": readonly_fail[
                        "host_backup_copy_executed_count"
                    ],
                },
                "fail_case_count": len(fail_cases),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
