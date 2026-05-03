from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_fake_materialization_results_for_smoke,
    summarize_relaykv_materialization_results_for_smoke,
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


def _policy_event(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
    selected_block_ids: list[int],
    anchor_block_ids: list[int],
    recent_block_ids: list[int],
    candidate_block_ids: list[int],
    *,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_type": "relaykv_policy_dry_run",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "selected_block_ids": selected_block_ids,
        "anchor_block_ids": anchor_block_ids,
        "recent_block_ids": recent_block_ids,
        "candidate_block_ids": candidate_block_ids,
    }
    if poison is not None:
        event["unrelated_tensor_like"] = poison
    return event


def _policy_events(poison: _PoisonTensorLike | None = None) -> list[dict[str, Any]]:
    return [
        _policy_event(
            "rid-a",
            10,
            512,
            0,
            [1, 2],
            [0],
            [7, 8],
            [1, 2, 3, 4],
            poison=poison,
        ),
        _policy_event(
            "rid-b",
            11,
            1024,
            14,
            [9, 10],
            [0, 1],
            [15],
            [9, 10, 11, 12],
        ),
    ]


def _ready() -> dict[str, Any]:
    return {"ready_for_materialization": True, "blocking_reasons": []}


def _blocked() -> dict[str, Any]:
    return {
        "ready_for_materialization": False,
        "blocking_reasons": ["blocked_req_pool_idx_missing"],
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
        if summary[key] != 0:
            raise AssertionError(summary)


def _assert_result_safety(result: dict[str, Any]) -> None:
    for key in (
        "source_mutated",
        "attention_override",
        "kv_cache_mutation",
        "runtime_writeback",
        "host_backup_copy_executed",
        "kv_pool_read",
        "kv_snapshot",
    ):
        if result[key] is not False:
            raise AssertionError(result)
    if result["scheduler_policy_noop"] is not True:
        raise AssertionError(result)


def _assert_normal_fake_materialization() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    events = _policy_events(poison)
    events_before = copy.deepcopy(events)
    results = build_relaykv_fake_materialization_results_for_smoke(events, _ready())
    if events != events_before:
        raise AssertionError("policy dry-run events were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(results) != 2:
        raise AssertionError(results)
    for result, event in zip(results, events):
        _assert_result_safety(result)
        if result["event_type"] != "relaykv_materialization_result":
            raise AssertionError(result)
        if result["materialization_state"] != "fake_materialized":
            raise AssertionError(result)
        if result["materialization_mode"] != "fake":
            raise AssertionError(result)
        if result["materialized_block_ids"] != event["selected_block_ids"]:
            raise AssertionError(result)
        if result["retrieved_block_ids"] != event["selected_block_ids"]:
            raise AssertionError(result)
        if result["materialized_kv_count"] != len(event["selected_block_ids"]):
            raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["total_materialization_results"] != 2:
        raise AssertionError(summary)
    if summary["fake_materialized_count"] != 2:
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 4:
        raise AssertionError(summary)
    if summary["per_request_counts"] != {"rid-a": 1, "rid-b": 1}:
        raise AssertionError(summary)
    if summary["per_layer_counts"] != {"0": 1, "14": 1}:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_empty_selected_skips() -> dict[str, Any]:
    events = [
        _policy_event("rid-empty", 12, 128, 0, [], [], [], [3, 4]),
    ]
    results = build_relaykv_fake_materialization_results_for_smoke(events, _ready())
    result = results[0]
    if result["materialization_state"] != "skipped":
        raise AssertionError(result)
    if "no_selected_blocks" not in result["warning_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["skipped_count"] != 1:
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 0:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_readiness_blocked() -> dict[str, Any]:
    events = _policy_events()
    results = build_relaykv_fake_materialization_results_for_smoke(events, _blocked())
    for result, event in zip(results, events):
        if result["materialization_state"] != "blocked":
            raise AssertionError(result)
        if result["materialized_block_ids"] != []:
            raise AssertionError(result)
        if result["retrieved_block_ids"] != []:
            raise AssertionError(result)
        if result["skipped_block_ids"] != event["selected_block_ids"]:
            raise AssertionError(result)
        if "blocked_req_pool_idx_missing" not in result["blocking_reasons"]:
            raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["blocked_count"] != len(events):
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 0:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_missing_readiness_warns() -> dict[str, Any]:
    results = build_relaykv_fake_materialization_results_for_smoke(_policy_events())
    for result in results:
        if result["materialization_state"] != "fake_materialized":
            raise AssertionError(result)
        if "readiness_not_provided" not in result["warning_reasons"]:
            raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["fake_materialized_count"] != 2:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def main() -> None:
    result = {
        "normal_fake_materialization": _assert_normal_fake_materialization(),
        "empty_selected_skips": _assert_empty_selected_skips(),
        "readiness_blocked": _assert_readiness_blocked(),
        "missing_readiness_warns": _assert_missing_readiness_warns(),
    }
    print("relaykv_fake_materialization_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
