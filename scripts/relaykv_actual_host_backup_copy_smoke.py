from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_actual_host_backup_copy_results_for_smoke,
    summarize_relaykv_actual_host_backup_copy_results_for_smoke,
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


def _actual_copy_readiness_ready() -> dict[str, Any]:
    return {
        "ready_for_actual_host_backup_copy": True,
        "readiness_state": "ready_for_actual_host_backup_copy_smoke_boundary_complete",
        "blocking_reasons": [],
    }


def _actual_copy_readiness_blocked() -> dict[str, Any]:
    return {
        "ready_for_actual_host_backup_copy": False,
        "readiness_state": "blocked_kv_pool_read_observed",
        "blocking_reasons": ["blocked_kv_pool_read_observed"],
    }


def _copy_request(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
    materialized_block_ids: list[int],
    candidate_block_ids: list[int],
    *,
    copy_state: str = "request_ready",
    event_type: str = "relaykv_host_backup_copy_request",
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "event_type": event_type,
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "selected_block_ids": list(materialized_block_ids),
        "materialized_block_ids": list(materialized_block_ids),
        "retrieved_block_ids": list(materialized_block_ids),
        "candidate_block_ids": list(candidate_block_ids),
        "anchor_block_ids": [],
        "recent_block_ids": [],
        "materialized_kv_count": len(materialized_block_ids),
        "materialized_token_count": 0,
        "copy_state": copy_state,
        "copy_mode": "host_backup_copy_boundary",
        "source": "candidate_event_materialization_to_host_backup_copy_request",
    }
    if poison is not None:
        request["unrelated_tensor_like"] = poison
    return request


def _copy_requests(poison: _PoisonTensorLike | None = None) -> list[dict[str, Any]]:
    return [
        _copy_request(
            "rid-a",
            10,
            512,
            0,
            [1, 2],
            [1, 2, 3, 4],
            poison=poison,
        ),
        _copy_request(
            "rid-b",
            11,
            1024,
            14,
            [9, 10],
            [9, 10, 11, 12],
        ),
    ]


def _assert_safety(summary: dict[str, Any], *, executed_count: int) -> None:
    expected = {
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
        "host_backup_copy_executed_count": executed_count,
        "kv_pool_read_count": 0,
        "kv_snapshot_count": 0,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise AssertionError(summary)


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    requests = _copy_requests(poison)
    requests_before = copy.deepcopy(requests)
    readiness = _actual_copy_readiness_ready()
    readiness_before = copy.deepcopy(readiness)
    results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        requests,
        readiness,
        execute_copy=True,
    )
    if requests != requests_before:
        raise AssertionError("copy requests were mutated")
    if readiness != readiness_before:
        raise AssertionError("readiness was mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(results) != 2:
        raise AssertionError(results)
    for result in results:
        if result["materialization_state"] != "host_backup_copy_materialized":
            raise AssertionError(result)
        if result["materialization_mode"] != "host_backup_copy":
            raise AssertionError(result)
        if result["copy_state"] != "copy_executed":
            raise AssertionError(result)
        if result["copy_mode"] != "host_backup_copy_isolated_smoke":
            raise AssertionError(result)
        if result["host_backup_copy_executed"] is not True:
            raise AssertionError(result)
        if "isolated_smoke_no_runtime_connection" not in result["warning_reasons"]:
            raise AssertionError(result)
        if result["kv_pool_read"] is not False or result["kv_snapshot"] is not False:
            raise AssertionError(result)
    summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
    if summary["host_backup_copy_materialized_count"] != 2:
        raise AssertionError(summary)
    if summary["blocked_count"] != 0:
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 4:
        raise AssertionError(summary)
    if summary["per_request_counts"] != {"rid-a": 1, "rid-b": 1}:
        raise AssertionError(summary)
    if summary["per_layer_counts"] != {"0": 1, "14": 1}:
        raise AssertionError(summary)
    if summary["per_copy_state_counts"] != {"copy_executed": 2}:
        raise AssertionError(summary)
    _assert_safety(summary, executed_count=2)
    return {"results": results, "summary": summary}


def _assert_readiness_missing_blocks() -> dict[str, Any]:
    results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        _copy_requests(),
        None,
        execute_copy=True,
    )
    for result in results:
        if result["copy_state"] != "blocked":
            raise AssertionError(result)
        if "actual_copy_readiness_not_provided" not in result["blocking_reasons"]:
            raise AssertionError(result)
        if result["host_backup_copy_executed"] is not False:
            raise AssertionError(result)
    summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
    if summary["blocked_count"] != 2:
        raise AssertionError(summary)
    _assert_safety(summary, executed_count=0)
    return {"results": results, "summary": summary}


def _assert_readiness_blocked() -> dict[str, Any]:
    results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        _copy_requests(),
        _actual_copy_readiness_blocked(),
        execute_copy=True,
    )
    for result in results:
        if result["copy_state"] != "blocked":
            raise AssertionError(result)
        if "blocked_kv_pool_read_observed" not in result["blocking_reasons"]:
            raise AssertionError(result)
    summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
    if summary["blocked_count"] != 2:
        raise AssertionError(summary)
    _assert_safety(summary, executed_count=0)
    return {"results": results, "summary": summary}


def _assert_execute_copy_false_blocks() -> dict[str, Any]:
    results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        _copy_requests(),
        _actual_copy_readiness_ready(),
        execute_copy=False,
    )
    for result in results:
        if result["copy_state"] != "blocked":
            raise AssertionError(result)
        if "execute_copy_required_for_actual_copy_smoke" not in result["blocking_reasons"]:
            raise AssertionError(result)
    summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
    if summary["blocked_count"] != 2:
        raise AssertionError(summary)
    _assert_safety(summary, executed_count=0)
    return {"results": results, "summary": summary}


def _assert_request_not_ready_blocks() -> dict[str, Any]:
    request = _copy_request("rid-blocked", 12, 128, 0, [3], [3, 4], copy_state="blocked")
    results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        [request],
        _actual_copy_readiness_ready(),
        execute_copy=True,
    )
    result = results[0]
    if result["copy_state"] != "blocked":
        raise AssertionError(result)
    if "copy_request_not_ready" not in result["blocking_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety(summary, executed_count=0)
    return {"results": results, "summary": summary}


def _assert_wrong_event_type_blocks() -> dict[str, Any]:
    request = _copy_request(
        "rid-wrong",
        13,
        64,
        0,
        [4],
        [4, 5],
        event_type="not_relaykv_host_backup_copy_request",
    )
    results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        [request],
        _actual_copy_readiness_ready(),
        execute_copy=True,
    )
    result = results[0]
    if result["copy_state"] != "blocked":
        raise AssertionError(result)
    if "not_host_backup_copy_request" not in result["blocking_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety(summary, executed_count=0)
    return {"results": results, "summary": summary}


def _assert_empty_materialized_blocks() -> dict[str, Any]:
    request = _copy_request("rid-empty", 14, 32, 0, [], [1, 2])
    results = build_relaykv_actual_host_backup_copy_results_for_smoke(
        [request],
        _actual_copy_readiness_ready(),
        execute_copy=True,
    )
    result = results[0]
    if result["copy_state"] != "blocked":
        raise AssertionError(result)
    if "no_materialized_blocks" not in result["blocking_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_actual_host_backup_copy_results_for_smoke(results)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety(summary, executed_count=0)
    return {"results": results, "summary": summary}


def main() -> None:
    result = {
        "pass_flow": _assert_pass_flow(),
        "readiness_missing_blocks": _assert_readiness_missing_blocks(),
        "readiness_blocked": _assert_readiness_blocked(),
        "execute_copy_false_blocks": _assert_execute_copy_false_blocks(),
        "request_not_ready_blocks": _assert_request_not_ready_blocks(),
        "wrong_event_type_blocks": _assert_wrong_event_type_blocks(),
        "empty_materialized_blocks": _assert_empty_materialized_blocks(),
    }
    print("relaykv_actual_host_backup_copy_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
