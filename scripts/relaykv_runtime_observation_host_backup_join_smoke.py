from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    join_runtime_observation_with_host_backup_candidates_for_smoke,
    summarize_host_backup_copy_candidates_for_smoke,
)
from sglang.srt.relaykv.observation import summarize_runtime_observation_payloads


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


def _runtime_payload(
    request_id: str,
    req_pool_idx: int | None,
    layer_id: int,
    *,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_type": "runtime_observation_readonly_metadata_candidate",
        "batch_id": "join-batch-a",
        "request_id": request_id,
        "request_index": 0 if request_id == "rid-a" else 1,
        "request_index_in_batch": 0 if request_id == "rid-a" else 1,
        "req_pool_idx": req_pool_idx,
        "req_pool_index": req_pool_idx,
        "seq_len": 128 if request_id == "rid-a" else 256,
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
    if poison is not None:
        payload["unrelated_tensor_like"] = poison
    return payload


def _candidate_event(
    request_id: str,
    req_pool_idx: int | None,
    layer_idx: int,
    *,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "runtime_policy_state": "applied_candidate",
        "batch_id": "join-batch-a",
        "request_id": request_id,
        "request_index": 0 if request_id == "rid-a" else 1,
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
    if poison is not None:
        event["unrelated_tensor_like"] = poison
    return event


def _assert_safety(summary: dict[str, Any]) -> None:
    for key in (
        "source_mutated_true_count",
        "attention_override_true_count",
        "kv_cache_mutation_true_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
    ):
        if summary[key] != 0:
            raise AssertionError(summary)


def _assert_event_join() -> dict[str, Any]:
    runtime_poison = _PoisonTensorLike()
    candidate_poison = _PoisonTensorLike()
    runtime_payloads = [
        _runtime_payload("rid-a", 10, 0, poison=runtime_poison),
        _runtime_payload("rid-a", 10, 14),
        _runtime_payload("rid-b", 11, 0),
        _runtime_payload("rid-b", 11, 14),
    ]
    candidate_events = [
        _candidate_event("rid-a", 10, 0, poison=candidate_poison),
        _candidate_event("rid-b", 11, 14),
        _candidate_event("rid-c", 12, 0),
    ]
    runtime_before = copy.deepcopy(runtime_payloads)
    candidate_before = copy.deepcopy(candidate_events)

    summary = join_runtime_observation_with_host_backup_candidates_for_smoke(
        runtime_payloads,
        candidate_events,
    )
    if runtime_payloads != runtime_before:
        raise AssertionError("runtime payloads were mutated")
    if candidate_events != candidate_before:
        raise AssertionError("candidate events were mutated")
    if runtime_poison.forbidden_access_called or candidate_poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")

    expected = {
        "join_granularity": "event",
        "total_runtime_payloads": 4,
        "total_host_backup_candidate_events": 3,
        "joined_count": 2,
        "unmatched_runtime_count": 2,
        "unmatched_candidate_count": 1,
        "per_request_join_counts": {"rid-a": 1, "rid-b": 1},
        "per_layer_join_counts": {"0": 1, "14": 1},
        "req_pool_idx_joined_count": 2,
        "req_pool_idx_missing_count": 0,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise AssertionError(summary)
    _assert_safety(summary)
    return summary


def _assert_missing_req_pool_idx() -> dict[str, Any]:
    summary = join_runtime_observation_with_host_backup_candidates_for_smoke(
        [_runtime_payload("rid-a", None, 0)],
        [_candidate_event("rid-a", None, 0)],
    )
    if summary["joined_count"] != 0:
        raise AssertionError(summary)
    if summary["unmatched_runtime_count"] != 1:
        raise AssertionError(summary)
    if summary["unmatched_candidate_count"] != 1:
        raise AssertionError(summary)
    if summary["req_pool_idx_missing_count"] != 2:
        raise AssertionError(summary)
    _assert_safety(summary)
    return summary


def _assert_summary_only_unjoinable() -> dict[str, Any]:
    runtime_payloads = [
        _runtime_payload("rid-a", 10, 0),
        _runtime_payload("rid-a", 10, 14),
    ]
    candidate_events = [
        _candidate_event("rid-a", 10, 0),
        _candidate_event("rid-b", 11, 14),
    ]
    runtime_summary = summarize_runtime_observation_payloads(runtime_payloads)
    candidate_summary = summarize_host_backup_copy_candidates_for_smoke(candidate_events)
    summary = join_runtime_observation_with_host_backup_candidates_for_smoke(
        runtime_summary,
        candidate_summary,
    )
    if summary["join_granularity"] != "summary_only_unjoinable":
        raise AssertionError(summary)
    if summary["total_runtime_payloads"] != 2:
        raise AssertionError(summary)
    if summary["total_host_backup_candidate_events"] != 2:
        raise AssertionError(summary)
    if summary["joined_count"] != 0:
        raise AssertionError(summary)
    _assert_safety(summary)
    return summary


def main() -> None:
    event_join = _assert_event_join()
    missing_req_pool_idx = _assert_missing_req_pool_idx()
    summary_only = _assert_summary_only_unjoinable()
    result = {
        "event_join": event_join,
        "missing_req_pool_idx": missing_req_pool_idx,
        "summary_only": summary_only,
    }
    print("relaykv_runtime_observation_host_backup_join_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
