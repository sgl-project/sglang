from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_candidate_event_materialization_results_for_smoke,
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


def _candidate_events(poison: _PoisonTensorLike | None = None) -> list[dict[str, Any]]:
    event_a: dict[str, Any] = {
        "request_id": "rid-a",
        "req_pool_idx": 10,
        "seq_len": 512,
        "layer_id": 0,
        "selected_block_ids": [1, 2],
        "candidate_block_ids": [1, 2, 3, 4],
        "anchor_block_ids": [0],
        "recent_block_ids": [7, 8],
    }
    if poison is not None:
        event_a["unrelated_tensor_like"] = poison
    return [
        event_a,
        {
            "request_id": "rid-b",
            "req_pool_index": 11,
            "seq_len": 1024,
            "layer_idx": 14,
            "materialized_block_ids": [9, 10],
            "candidate_block_ids": [9, 10, 11, 12],
            "anchor_block_ids": [0, 1],
            "recent_block_ids": [15],
        },
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


def _assert_normal_candidate_event_materialization() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    events = _candidate_events(poison)
    events_before = copy.deepcopy(events)
    results = build_relaykv_candidate_event_materialization_results_for_smoke(
        events, _ready()
    )
    if events != events_before:
        raise AssertionError("host backup candidate events were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(results) != 2:
        raise AssertionError(results)
    for result in results:
        _assert_result_safety(result)
        if result["event_type"] != "relaykv_materialization_result":
            raise AssertionError(result)
        if result["materialization_state"] != "candidate_event_materialized":
            raise AssertionError(result)
        if result["materialization_mode"] != "candidate_event":
            raise AssertionError(result)
        if result["skipped_block_ids"] != []:
            raise AssertionError(result)
        if result["retrieved_block_ids"] != result["materialized_block_ids"]:
            raise AssertionError(result)
        if "candidate_event_metadata_only_no_kv_copy" not in result["warning_reasons"]:
            raise AssertionError(result)
    if results[0]["materialized_block_ids"] != [1, 2]:
        raise AssertionError(results[0])
    if results[1]["materialized_block_ids"] != [9, 10]:
        raise AssertionError(results[1])
    if results[1]["req_pool_idx"] != 11 or results[1]["layer_id"] != 14:
        raise AssertionError(results[1])
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["total_materialization_results"] != 2:
        raise AssertionError(summary)
    if summary["candidate_event_materialized_count"] != 2:
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 4:
        raise AssertionError(summary)
    if summary["per_request_counts"] != {"rid-a": 1, "rid-b": 1}:
        raise AssertionError(summary)
    if summary["per_layer_counts"] != {"0": 1, "14": 1}:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_explicit_retrieved_ids() -> dict[str, Any]:
    events = [
        {
            "request_id": "rid-retrieved",
            "layer_id": 0,
            "materialized_block_ids": [20, 21],
            "retrieved_block_ids": [21],
        }
    ]
    results = build_relaykv_candidate_event_materialization_results_for_smoke(
        events, _ready()
    )
    result = results[0]
    if result["materialized_block_ids"] != [20, 21]:
        raise AssertionError(result)
    if result["retrieved_block_ids"] != [21]:
        raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["candidate_event_materialized_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_alias_mapping() -> dict[str, Any]:
    events = [
        {"request_id": "rid-block", "layer_id": 0, "block_id": 33},
        {"request_id": "rid-candidate", "layer_id": 1, "candidate_block_id": 44},
        {"request_id": "rid-blocks", "layer_id": 2, "block_ids": [55, 56]},
        {"request_id": "rid-copied", "layer_id": 3, "copied_block_ids": [60, 61]},
    ]
    results = build_relaykv_candidate_event_materialization_results_for_smoke(
        events, _ready()
    )
    expected = [[33], [44], [55, 56], [60, 61]]
    for result, block_ids in zip(results, expected):
        if result["selected_block_ids"] != block_ids:
            raise AssertionError(results)
        if result["materialized_block_ids"] != block_ids:
            raise AssertionError(results)
        if result["retrieved_block_ids"] != block_ids:
            raise AssertionError(results)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["candidate_event_materialized_count"] != 4:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_empty_selected_skips() -> dict[str, Any]:
    events = [{"request_id": "rid-empty", "req_pool_idx": 12, "layer_id": 0}]
    results = build_relaykv_candidate_event_materialization_results_for_smoke(
        events, _ready()
    )
    result = results[0]
    if result["materialization_state"] != "skipped":
        raise AssertionError(result)
    if result["materialized_block_ids"] != []:
        raise AssertionError(result)
    if result["retrieved_block_ids"] != []:
        raise AssertionError(result)
    if result["skipped_block_ids"] != []:
        raise AssertionError(result)
    if "no_selected_blocks" not in result["warning_reasons"]:
        raise AssertionError(result)
    if "candidate_event_metadata_only_no_kv_copy" not in result["warning_reasons"]:
        raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["skipped_count"] != 1:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_readiness_blocked() -> dict[str, Any]:
    events = _candidate_events()
    results = build_relaykv_candidate_event_materialization_results_for_smoke(
        events, _blocked()
    )
    for result in results:
        if result["materialization_state"] != "blocked":
            raise AssertionError(result)
        if result["materialized_block_ids"] != []:
            raise AssertionError(result)
        if result["retrieved_block_ids"] != []:
            raise AssertionError(result)
        if result["skipped_block_ids"] != result["selected_block_ids"]:
            raise AssertionError(result)
        if "blocked_req_pool_idx_missing" not in result["blocking_reasons"]:
            raise AssertionError(result)
        if "candidate_event_metadata_only_no_kv_copy" not in result["warning_reasons"]:
            raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["blocked_count"] != len(events):
        raise AssertionError(summary)
    if summary["materialized_kv_count"] != 0:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def _assert_missing_readiness_warns() -> dict[str, Any]:
    results = build_relaykv_candidate_event_materialization_results_for_smoke(
        _candidate_events()
    )
    for result in results:
        if result["materialization_state"] != "candidate_event_materialized":
            raise AssertionError(result)
        if "readiness_not_provided" not in result["warning_reasons"]:
            raise AssertionError(result)
        if "candidate_event_metadata_only_no_kv_copy" not in result["warning_reasons"]:
            raise AssertionError(result)
    summary = summarize_relaykv_materialization_results_for_smoke(results)
    if summary["candidate_event_materialized_count"] != 2:
        raise AssertionError(summary)
    _assert_safety_zero(summary)
    return {"results": results, "summary": summary}


def main() -> None:
    result = {
        "normal_candidate_event_materialization": (
            _assert_normal_candidate_event_materialization()
        ),
        "explicit_retrieved_ids": _assert_explicit_retrieved_ids(),
        "alias_mapping": _assert_alias_mapping(),
        "empty_selected_skips": _assert_empty_selected_skips(),
        "readiness_blocked": _assert_readiness_blocked(),
        "missing_readiness_warns": _assert_missing_readiness_warns(),
    }
    print("relaykv_candidate_event_materialization_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
