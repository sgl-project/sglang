from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_host_backup_copy_boundary_results_for_smoke,
    summarize_relaykv_host_backup_copy_boundary_results_for_smoke,
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
        "copy_source": "host_backup_candidate",
        "copy_destination": "materialization_result_only",
        "copy_guard_state": "pre_attention_no_runtime_writeback",
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
    poison = _PoisonTensorLike()
    requests = _copy_requests(poison)
    requests_before = copy.deepcopy(requests)
    results = build_relaykv_host_backup_copy_boundary_results_for_smoke(requests)
    if requests != requests_before:
        raise AssertionError("copy requests were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(results) != 2:
        raise AssertionError(results)
    for result in results:
        if result["event_type"] != "relaykv_host_backup_copy_boundary_result":
            raise AssertionError(result)
        if result["materialization_state"] != "host_backup_copy_boundary_noop":
            raise AssertionError(result)
        if result["materialization_mode"] != "host_backup_copy_boundary":
            raise AssertionError(result)
        if result["copy_state"] != "boundary_noop":
            raise AssertionError(result)
        if "execute_copy_false_boundary_noop" not in result["warning_reasons"]:
            raise AssertionError(result)
        if result["blocking_reasons"]:
            raise AssertionError(result)
        if result["host_backup_copy_executed"] is not False:
            raise AssertionError(result)
        if result["kv_pool_read"] is not False:
            raise AssertionError(result)
        if result["kv_snapshot"] is not False:
            raise AssertionError(result)
    summary = summarize_relaykv_host_backup_copy_boundary_results_for_smoke(results)
    if summary["total_boundary_results"] != 2:
        raise AssertionError(summary)
    if summary["boundary_noop_count"] != 2:
        raise AssertionError(summary)
    if summary["blocked_count"] != 0:
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 4:
        raise AssertionError(summary)
    if summary["per_request_counts"] != {"rid-a": 1, "rid-b": 1}:
        raise AssertionError(summary)
    if summary["per_layer_counts"] != {"0": 1, "14": 1}:
        raise AssertionError(summary)
    if summary["per_copy_state_counts"] != {"boundary_noop": 2}:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_execute_copy_true_blocks() -> dict[str, Any]:
    results = build_relaykv_host_backup_copy_boundary_results_for_smoke(
        _copy_requests(),
        execute_copy=True,
    )
    for result in results:
        if result["copy_state"] != "blocked":
            raise AssertionError(result)
        if "execute_copy_not_allowed_in_smoke" not in result["blocking_reasons"]:
            raise AssertionError(result)
        if result["host_backup_copy_executed"] is not False:
            raise AssertionError(result)
    summary = summarize_relaykv_host_backup_copy_boundary_results_for_smoke(results)
    if summary["blocked_count"] != 2:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_blocked_request_remains_blocked() -> dict[str, Any]:
    request = _copy_request("rid-blocked", 12, 128, 0, [3], [3, 4])
    request["copy_state"] = "blocked"
    results = build_relaykv_host_backup_copy_boundary_results_for_smoke([request])
    result = results[0]
    if result["copy_state"] != "blocked":
        raise AssertionError(result)
    if "copy_request_not_ready" not in result["blocking_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_host_backup_copy_boundary_results_for_smoke(results)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
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
    results = build_relaykv_host_backup_copy_boundary_results_for_smoke([request])
    result = results[0]
    if result["copy_state"] != "blocked":
        raise AssertionError(result)
    if "not_host_backup_copy_request" not in result["blocking_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_host_backup_copy_boundary_results_for_smoke(results)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_empty_materialized_ids_block() -> dict[str, Any]:
    request = _copy_request("rid-empty", 14, 32, 0, [], [1, 2])
    results = build_relaykv_host_backup_copy_boundary_results_for_smoke([request])
    result = results[0]
    if result["copy_state"] != "blocked":
        raise AssertionError(result)
    if "no_materialized_blocks" not in result["blocking_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_host_backup_copy_boundary_results_for_smoke(results)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_missing_optional_fields_tolerated() -> dict[str, Any]:
    request = {
        "event_type": "relaykv_host_backup_copy_request",
        "request_id": "rid-minimal",
        "req_pool_idx": 15,
        "seq_len": 16,
        "layer_id": 0,
        "materialized_block_ids": [6],
        "materialized_kv_count": 1,
        "materialized_token_count": 0,
        "copy_state": "request_ready",
        "copy_mode": "host_backup_copy_boundary",
    }
    results = build_relaykv_host_backup_copy_boundary_results_for_smoke([request])
    result = results[0]
    if result["copy_state"] != "boundary_noop":
        raise AssertionError(result)
    if result["selected_block_ids"] != []:
        raise AssertionError(result)
    if result["retrieved_block_ids"] != []:
        raise AssertionError(result)
    if result["candidate_block_ids"] != []:
        raise AssertionError(result)
    summary = summarize_relaykv_host_backup_copy_boundary_results_for_smoke(results)
    if summary["boundary_noop_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def main() -> None:
    result = {
        "full_pass_flow": _assert_full_pass_flow(),
        "execute_copy_true_blocks": _assert_execute_copy_true_blocks(),
        "blocked_request_remains_blocked": _assert_blocked_request_remains_blocked(),
        "wrong_event_type_blocks": _assert_wrong_event_type_blocks(),
        "empty_materialized_ids_block": _assert_empty_materialized_ids_block(),
        "missing_optional_fields_tolerated": _assert_missing_optional_fields_tolerated(),
    }
    print("relaykv_host_backup_copy_boundary_result_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
