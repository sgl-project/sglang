from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_policy_dry_run_events_for_smoke,
    summarize_relaykv_policy_dry_run_events_for_smoke,
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


def _runtime_payloads(poison: _PoisonTensorLike | None = None) -> list[dict[str, Any]]:
    payloads = [
        {
            "request_id": "rid-a",
            "req_pool_idx": 10,
            "seq_len": 512,
            "layer_id": 0,
        },
        {
            "request_id": "rid-a",
            "req_pool_idx": 10,
            "seq_len": 512,
            "layer_id": 14,
        },
        {
            "request_id": "rid-b",
            "req_pool_idx": 11,
            "seq_len": 1024,
            "layer_id": 0,
        },
    ]
    if poison is not None:
        payloads[0]["unrelated_tensor_like"] = poison
    return payloads


def _block_metadata(poison: _PoisonTensorLike | None = None) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {
        "rid-a": {
            "anchor_block_ids": [0],
            "recent_block_ids": [7, 8],
            "candidate_block_ids": [1, 2, 3, 4],
        },
        "rid-b": {
            "anchor_block_ids": [0, 1],
            "recent_block_ids": [15],
            "candidate_block_ids": [9, 10, 11],
        },
    }
    if poison is not None:
        metadata["rid-a"]["unrelated_tensor_like"] = poison
    return metadata


def _policy_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "kv_budget_tokens": 1024,
        "recent_tokens": 256,
        "anchor_tokens": 128,
        "transient_tokens": 64,
        "retrieval_top_k": 2,
        "layer_budget_policy": "uniform",
    }
    config.update(overrides)
    return config


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


def _assert_event_safety(events: list[dict[str, Any]]) -> None:
    for event in events:
        if event["event_type"] != "relaykv_policy_dry_run":
            raise AssertionError(event)
        if event["policy_state"] != "dry_run":
            raise AssertionError(event)
        if event["source"] != "readonly_metadata_policy_dry_run":
            raise AssertionError(event)
        for key in (
            "source_mutated",
            "attention_override",
            "kv_cache_mutation",
            "runtime_writeback",
        ):
            if event[key] is not False:
                raise AssertionError(event)
        if event["scheduler_policy_noop"] is not True:
            raise AssertionError(event)


def _assert_main_dry_run() -> dict[str, Any]:
    runtime_poison = _PoisonTensorLike()
    block_poison = _PoisonTensorLike()
    runtime_payloads = _runtime_payloads(runtime_poison)
    block_metadata = _block_metadata(block_poison)
    policy_config = _policy_config()
    runtime_before = copy.deepcopy(runtime_payloads)
    block_before = copy.deepcopy(block_metadata)
    policy_before = copy.deepcopy(policy_config)

    events = build_relaykv_policy_dry_run_events_for_smoke(
        runtime_payloads,
        block_metadata,
        policy_config,
    )
    if runtime_payloads != runtime_before:
        raise AssertionError("runtime payloads were mutated")
    if block_metadata != block_before:
        raise AssertionError("block metadata was mutated")
    if policy_config != policy_before:
        raise AssertionError("policy config was mutated")
    if runtime_poison.forbidden_access_called or block_poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")

    _assert_event_safety(events)
    if len(events) != 3:
        raise AssertionError(events)
    for event in events:
        if event["retrieval_budget_tokens"] != 576:
            raise AssertionError(event)
        for kv_class in ("RECENT", "ANCHOR", "RETRIEVED", "COLD_CANDIDATE"):
            if kv_class not in event["kv_classes_present"]:
                raise AssertionError(event)
    rid_a_events = [event for event in events if event["request_id"] == "rid-a"]
    rid_b_events = [event for event in events if event["request_id"] == "rid-b"]
    if any(event["selected_block_ids"] != [1, 2] for event in rid_a_events):
        raise AssertionError(events)
    if any(event["selected_block_ids"] != [9, 10] for event in rid_b_events):
        raise AssertionError(events)

    summary = summarize_relaykv_policy_dry_run_events_for_smoke(events)
    if summary["total_events"] != 3:
        raise AssertionError(summary)
    if summary["event_type_counts"] != {"relaykv_policy_dry_run": 3}:
        raise AssertionError(summary)
    if summary["per_request_counts"] != {"rid-a": 2, "rid-b": 1}:
        raise AssertionError(summary)
    if summary["per_layer_counts"] != {"0": 2, "14": 1}:
        raise AssertionError(summary)
    _assert_safety(summary)
    return {"events": events, "summary": summary}


def _assert_top_k_larger_than_candidates() -> dict[str, Any]:
    events = build_relaykv_policy_dry_run_events_for_smoke(
        [
            {
                "request_id": "rid-c",
                "req_pool_idx": 12,
                "seq_len": 128,
                "layer_id": 0,
            }
        ],
        {"rid-c": {"candidate_block_ids": [5], "anchor_block_ids": [], "recent_block_ids": []}},
        _policy_config(retrieval_top_k=4),
    )
    if events[0]["selected_block_ids"] != [5]:
        raise AssertionError(events)
    if "COLD_CANDIDATE" in events[0]["kv_classes_present"]:
        raise AssertionError(events)
    return {"events": events}


def _assert_negative_retrieval_budget_clamps() -> dict[str, Any]:
    events = build_relaykv_policy_dry_run_events_for_smoke(
        [
            {
                "request_id": "rid-a",
                "req_pool_idx": 10,
                "seq_len": 512,
                "layer_id": 0,
            }
        ],
        _block_metadata(),
        _policy_config(
            kv_budget_tokens=128,
            recent_tokens=256,
            anchor_tokens=128,
            transient_tokens=64,
        ),
    )
    if events[0]["retrieval_budget_tokens"] != 0:
        raise AssertionError(events)
    return {"events": events}


def _assert_missing_block_metadata() -> dict[str, Any]:
    events = build_relaykv_policy_dry_run_events_for_smoke(
        [
            {
                "request_id": "rid-missing",
                "req_pool_idx": 13,
                "seq_len": 64,
                "layer_id": 0,
            }
        ],
        {},
        _policy_config(),
    )
    event = events[0]
    for key in (
        "candidate_block_ids",
        "selected_block_ids",
        "anchor_block_ids",
        "recent_block_ids",
        "kv_classes_present",
    ):
        if event[key] != []:
            raise AssertionError(event)
    return {"events": events}


def main() -> None:
    result = {
        "main_dry_run": _assert_main_dry_run(),
        "top_k_larger_than_candidates": _assert_top_k_larger_than_candidates(),
        "negative_retrieval_budget_clamps": _assert_negative_retrieval_budget_clamps(),
        "missing_block_metadata": _assert_missing_block_metadata(),
    }
    print("relaykv_policy_dry_run_smoke=pass")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
