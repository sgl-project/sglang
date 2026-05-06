from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke,
    summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    summarize_relaykv_runtime_req_to_token_resolution_payloads_for_smoke,
)


class _PoisonObject:
    def __init__(self) -> None:
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False
        self.iter_called = False
        self.len_called = False
        self.repr_called = False
        self.getitem_called = False

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PoisonObject":
        return self

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def numpy(self) -> None:
        self.numpy_called = True
        raise AssertionError("numpy() must not be called")

    def __iter__(self):
        self.iter_called = True
        raise AssertionError("__iter__() must not be called")

    def __len__(self) -> int:
        self.len_called = True
        raise AssertionError("__len__() must not be called")

    def __repr__(self) -> str:
        self.repr_called = True
        raise AssertionError("__repr__() must not be called")

    def __getitem__(self, index: Any) -> Any:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

    @property
    def touched(self) -> bool:
        return any(
            (
                self.cpu_called,
                self.tolist_called,
                self.item_called,
                self.numpy_called,
                self.iter_called,
                self.len_called,
                self.repr_called,
                self.getitem_called,
            )
        )


def _runtime_observation_payload(request_id: str = "req-runtime") -> dict[str, Any]:
    return {
        "event_type": "relaykv_runtime_observation_result",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": f"engine-{request_id}",
        "logical_sequence_id": f"seq-{request_id}",
        "request_id": request_id,
        "logical_block_id": 901,
        "token_span": [0, 3],
        "layer_id": 14,
        "kv_head_group": 2,
        "kv_class": "FULL",
        "fallback_reason": None,
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": {
            "seq_len": 3,
            "runtime_observation_marker": request_id,
        },
        "engine_block_ref": {
            "cache_position": 9,
            "req_pool_idx": 7,
        },
        "kv_pool_read": False,
        "kv_snapshot": False,
        "tensor_read": False,
        "attention_comparison_executed": False,
        "attention_override": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
    }


def _kv_index_resolution_plan(request_id: str = "req-runtime") -> dict[str, Any]:
    return {
        "event_type": "relaykv_kv_index_resolution_plan",
        "resolution_state": "block_span_resolved",
        "request_id": request_id,
        "req_pool_idx": 7,
        "layer_id": 14,
        "relaykv_plan_marker": f"plan-{request_id}",
    }


def _assert_zero_safety(summary: dict[str, Any]) -> None:
    for key in (
        "req_to_token_read_count",
        "actual_req_to_token_pool_read_count",
        "kv_pool_read_count",
        "kv_snapshot_count",
        "tensor_read_count",
        "attention_comparison_executed_count",
        "attention_override_true_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
        "kv_cache_mutation_true_count",
        "source_mutated_true_count",
    ):
        if summary.get(key) != 0:
            raise AssertionError((key, summary.get(key)))


def _build(
    *,
    runtime_observation_payloads: Any = None,
    kv_index_resolution_plans: Any = None,
    explicit_req_to_token_entries: Any = None,
    production_enabled: bool,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        kv_index_resolution_plans=kv_index_resolution_plans,
        explicit_req_to_token_entries=explicit_req_to_token_entries,
        production_enabled=production_enabled,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
    )
    summary = summarize_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        results,
        production_enabled=production_enabled,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
    )
    if summary["event_type"] != (
        "relaykv_runtime_req_to_token_resolution_payload_production_summary"
    ):
        raise AssertionError(summary)
    for key in (
        "req_to_token_read_count",
        "actual_req_to_token_pool_read_count",
        "kv_pool_read_count",
        "tensor_read_count",
        "source_mutated_true_count",
    ):
        if summary[key] != 0:
            raise AssertionError((key, summary[key]))
    return results, summary


def _assert_production_off_blocked() -> None:
    results, summary = _build(
        explicit_req_to_token_entries=[10, 11, 12],
        production_enabled=False,
    )
    if len(results) != 1:
        raise AssertionError(results)
    if results[0]["resolution_state"] != "blocked":
        raise AssertionError(results[0])
    if results[0]["blocking_reasons"] != [
        "runtime_req_to_token_payload_production_not_enabled"
    ]:
        raise AssertionError(results[0])
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)


def _assert_explicit_entries_resolve() -> None:
    entries = [10, 11, 12]
    before = copy.deepcopy(entries)
    results, summary = _build(
        explicit_req_to_token_entries=entries,
        production_enabled=True,
    )
    if entries != before:
        raise AssertionError("explicit entries mutated")
    result = results[0]
    if result["resolution_state"] != "req_to_token_resolved":
        raise AssertionError(result)
    span = result["full_kv_req_to_token_spans"][0]
    if span["req_to_token_entries"] != entries:
        raise AssertionError(span)
    if span["entry_count"] != len(entries):
        raise AssertionError(span)
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)


def _assert_runtime_metadata_and_plan_preserved() -> None:
    runtime_payload = _runtime_observation_payload()
    plan = _kv_index_resolution_plan()
    before_payload = copy.deepcopy(runtime_payload)
    before_plan = copy.deepcopy(plan)
    results, _ = _build(
        runtime_observation_payloads=[runtime_payload],
        kv_index_resolution_plans=[plan],
        explicit_req_to_token_entries=[10, 11, 12],
        production_enabled=True,
    )
    if runtime_payload != before_payload:
        raise AssertionError("runtime observation payload mutated")
    if plan != before_plan:
        raise AssertionError("kv index resolution plan mutated")
    result = results[0]
    if result["engine_request_id"] != "engine-req-runtime":
        raise AssertionError(result)
    if result["logical_sequence_id"] != "seq-req-runtime":
        raise AssertionError(result)
    if result["logical_block_id"] != 901:
        raise AssertionError(result)
    if result["token_span"] != [0, 3]:
        raise AssertionError(result)
    if result["layer_id"] != 14 or result["kv_head_group"] != 2:
        raise AssertionError(result)
    if result["kv_class"] != "FULL":
        raise AssertionError(result)
    if result["adapter_metadata"]["runtime_observation_marker"] != "req-runtime":
        raise AssertionError(result)
    if (
        result["adapter_metadata"]["kv_index_resolution_plan_metadata"][
            "relaykv_plan_marker"
        ]
        != "plan-req-runtime"
    ):
        raise AssertionError(result)


def _assert_blocked_cases() -> None:
    blocked_expectations = [
        (
            "max_tokens_per_request_exceeded",
            {
                "explicit_req_to_token_entries": [10, 11, 12],
                "max_tokens_per_request": 2,
            },
        ),
        (
            "max_total_tokens_exceeded",
            {
                "runtime_observation_payloads": [
                    _runtime_observation_payload("req-a"),
                    _runtime_observation_payload("req-b"),
                ],
                "explicit_req_to_token_entries": [10, 11, 12],
                "max_total_tokens": 5,
            },
        ),
        (
            "explicit_req_to_token_entry_not_int",
            {"explicit_req_to_token_entries": [10, "bad", 12]},
        ),
        (
            "explicit_req_to_token_entries_missing",
            {"explicit_req_to_token_entries": None},
        ),
        (
            "source_payload_invalid",
            {
                "runtime_observation_payloads": ["bad-source"],
                "explicit_req_to_token_entries": [10, 11, 12],
            },
        ),
        (
            "explicit_req_to_token_entries_not_list",
            {"explicit_req_to_token_entries": {"bad": 1}},
        ),
    ]
    for reason, kwargs in blocked_expectations:
        results, summary = _build(production_enabled=True, **kwargs)
        if results[0]["resolution_state"] != "blocked":
            raise AssertionError(results[0])
        if reason not in results[0]["blocking_reasons"]:
            raise AssertionError((reason, results[0]))
        if summary["blocked_count"] != 1:
            raise AssertionError(summary)


def _assert_poison_untouched() -> None:
    poison = _PoisonObject()
    runtime_payload = _runtime_observation_payload("req-poison")
    runtime_payload["poison"] = poison
    before = copy.deepcopy(runtime_payload)
    results, _ = _build(
        runtime_observation_payloads=[runtime_payload],
        explicit_req_to_token_entries=[10, 11, 12],
        production_enabled=True,
    )
    if poison.touched:
        raise AssertionError("poison object touched")
    if runtime_payload != before:
        raise AssertionError("runtime payload mutated")
    if results[0]["source_mutated"] is not False:
        raise AssertionError(results[0])


def _assert_bridge_and_live_read() -> None:
    produced_results, _ = _build(
        runtime_observation_payloads=[_runtime_observation_payload("req-live")],
        explicit_req_to_token_entries=[10, 11, 12],
        production_enabled=True,
    )
    bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        explicit_payloads=produced_results,
        bridge_enabled=True,
    )
    bridge_summary = summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        bridge_results
    )
    _assert_zero_safety(bridge_summary)
    bridge_result = bridge_results[0]
    if bridge_result["bridge_state"] != "bridged":
        raise AssertionError(bridge_result)
    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        bridge_result["req_to_token_resolution_payloads"],
        token_to_kv_pool_object={10: 100, 11: 101, 12: 102},
        read_token_to_kv_pool_index=True,
        source_path="runtime.production.bridge",
    )
    live_summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        live_results
    )
    if live_summary["token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(live_summary)
    if live_summary["actual_token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(live_summary)
    if live_summary["live_token_to_kv_pool_index_read_count"] <= 0:
        raise AssertionError(live_summary)
    _assert_zero_safety(live_summary)


def main() -> None:
    _assert_production_off_blocked()
    _assert_explicit_entries_resolve()
    _assert_runtime_metadata_and_plan_preserved()
    _assert_blocked_cases()
    _assert_poison_untouched()
    _assert_bridge_and_live_read()
    print("relaykv_runtime_req_to_token_resolution_payload_production_smoke=pass")
    print(
        "relaykv_runtime_req_to_token_resolution_payload_production_summary="
        + json.dumps(
            {
                "production_off": "blocked",
                "explicit_entries": "resolved",
                "runtime_metadata": "preserved",
                "blocked_cases": "validated",
                "bridge_to_live_read": "resolved",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
