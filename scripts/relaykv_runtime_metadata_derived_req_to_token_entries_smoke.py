from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    build_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke,
    build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke,
    summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    summarize_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke,
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


def _runtime_observation_payload(
    request_id: str = "req-a",
    *,
    token_span: list[int] | None = None,
    seq_len: int | None = 6,
) -> dict[str, Any]:
    adapter_metadata: dict[str, Any] = {"runtime_observation_marker": request_id}
    if seq_len is not None:
        adapter_metadata["seq_len"] = seq_len
    payload = {
        "event_type": "relaykv_runtime_observation_result",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": f"engine-{request_id}",
        "logical_sequence_id": f"seq-{request_id}",
        "request_id": request_id,
        "logical_block_id": 101,
        "layer_id": 14,
        "kv_head_group": 2,
        "kv_class": "FULL",
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": adapter_metadata,
        "engine_block_ref": {"req_pool_idx": 7},
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
    if token_span is not None:
        payload["token_span"] = token_span
    return payload


def _kv_index_resolution_plan(request_id: str = "req-a") -> dict[str, Any]:
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
        "token_to_kv_pool_read_count",
        "actual_token_to_kv_pool_read_count",
        "live_token_to_kv_pool_index_read_count",
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


def _assert_zero_bridge_safety(summary: dict[str, Any]) -> None:
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


def _derive(
    *,
    runtime_observation_payloads: Any = None,
    kv_index_resolution_plans: Any = None,
    production_enabled: bool,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = build_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        kv_index_resolution_plans=kv_index_resolution_plans,
        production_enabled=production_enabled,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
    )
    summary = summarize_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
        results,
        production_enabled=production_enabled,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
    )
    if summary["event_type"] != (
        "relaykv_runtime_metadata_derived_req_to_token_entries_summary"
    ):
        raise AssertionError(summary)
    _assert_zero_safety(summary)
    return results, summary


def _assert_production_off_blocked() -> None:
    results, summary = _derive(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 3])],
        production_enabled=False,
    )
    if results[0]["derivation_state"] != "blocked":
        raise AssertionError(results[0])
    if results[0]["blocked_reason"] != "runtime_metadata_derivation_not_enabled":
        raise AssertionError(results[0])
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)


def _assert_token_span_derives() -> None:
    payload = _runtime_observation_payload("req-span", token_span=[2, 5], seq_len=20)
    before = copy.deepcopy(payload)
    results, summary = _derive(
        runtime_observation_payloads=[payload],
        kv_index_resolution_plans=[_kv_index_resolution_plan("req-span")],
        production_enabled=True,
    )
    if payload != before:
        raise AssertionError("runtime observation payload mutated")
    result = results[0]
    if result["derivation_state"] != "derived":
        raise AssertionError(result)
    if result["derived_req_to_token_entries"] != [2, 3, 4]:
        raise AssertionError(result)
    if result["derived_entry_count"] != 3:
        raise AssertionError(result)
    if result["synthetic_metadata_only"] is not True:
        raise AssertionError(result)
    if result["adapter_metadata"]["runtime_metadata_derivation_source_path"] != (
        "runtime_observation_payloads"
    ):
        raise AssertionError(result)
    if summary["derived_count"] != 1 or summary["total_derived_entries"] != 3:
        raise AssertionError(summary)


def _assert_seq_len_fallback_derives() -> None:
    results, _ = _derive(
        runtime_observation_payloads=[_runtime_observation_payload("req-seq", token_span=None, seq_len=4)],
        production_enabled=True,
    )
    result = results[0]
    if result["derivation_state"] != "derived":
        raise AssertionError(result)
    if result["derived_req_to_token_entries"] != [0, 1, 2, 3]:
        raise AssertionError(result)
    if result["token_span"] != [0, 4]:
        raise AssertionError(result)


def _assert_blocked_cases() -> None:
    blocked_cases = [
        (
            "max_tokens_per_request_exceeded",
            {
                "runtime_observation_payloads": [
                    _runtime_observation_payload("req-max-pr", token_span=[0, 5], seq_len=10)
                ],
                "max_tokens_per_request": 4,
            },
        ),
        (
            "max_total_tokens_exceeded",
            {
                "runtime_observation_payloads": [
                    _runtime_observation_payload("req-total-a", token_span=[0, 3], seq_len=10),
                    _runtime_observation_payload("req-total-b", token_span=[0, 3], seq_len=10),
                ],
                "max_total_tokens": 5,
            },
        ),
        (
            "runtime_observation_payloads_missing",
            {"runtime_observation_payloads": None},
        ),
        (
            "invalid_token_span",
            {
                "runtime_observation_payloads": [
                    _runtime_observation_payload("req-bad-span", token_span=[4, 2], seq_len=10)
                ]
            },
        ),
        (
            "invalid_seq_len",
            {
                "runtime_observation_payloads": [
                    _runtime_observation_payload("req-bad-seq", token_span=None, seq_len=0)
                ]
            },
        ),
    ]
    for expected_reason, kwargs in blocked_cases:
        results, summary = _derive(production_enabled=True, **kwargs)
        matching_result = None
        for result in results:
            if result["blocked_reason"] == expected_reason:
                matching_result = result
                break
        if matching_result is None:
            raise AssertionError((expected_reason, results))
        if matching_result["derivation_state"] != "blocked":
            raise AssertionError(matching_result)
        if summary["blocked_count"] <= 0:
            raise AssertionError(summary)


def _assert_poison_untouched() -> None:
    poison = _PoisonObject()
    payload = _runtime_observation_payload("req-poison", token_span=[0, 3], seq_len=10)
    payload["poison"] = poison
    before = copy.deepcopy(payload)
    results, _ = _derive(
        runtime_observation_payloads=[payload],
        production_enabled=True,
    )
    if poison.touched:
        raise AssertionError("poison object touched")
    if payload != before:
        raise AssertionError("runtime observation payload mutated")
    if results[0]["source_mutated"] is not False:
        raise AssertionError(results[0])


def _assert_full_chain() -> None:
    derived_results, derived_summary = _derive(
        runtime_observation_payloads=[_runtime_observation_payload("req-chain", token_span=[10, 13], seq_len=20)],
        kv_index_resolution_plans=[_kv_index_resolution_plan("req-chain")],
        production_enabled=True,
    )
    if derived_summary["derived_count"] != 1:
        raise AssertionError(derived_summary)
    derived_entries = derived_results[0]["derived_req_to_token_entries"]
    payload_results = build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        runtime_observation_payloads=[_runtime_observation_payload("req-chain", token_span=[10, 13], seq_len=20)],
        kv_index_resolution_plans=[_kv_index_resolution_plan("req-chain")],
        explicit_req_to_token_entries=derived_entries,
        production_enabled=True,
    )
    payload_summary = summarize_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        payload_results,
        production_enabled=True,
        max_tokens_per_request=256,
        max_total_tokens=1024,
    )
    _assert_zero_safety(payload_summary)
    bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        explicit_payloads=payload_results,
        bridge_enabled=True,
    )
    bridge_summary = summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        bridge_results
    )
    _assert_zero_bridge_safety(bridge_summary)
    if bridge_results[0]["bridge_state"] != "bridged":
        raise AssertionError(bridge_results[0])
    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        bridge_results[0]["req_to_token_resolution_payloads"],
        token_to_kv_pool_object={10: 100, 11: 101, 12: 102},
        read_token_to_kv_pool_index=True,
        source_path="runtime.metadata_derived.chain",
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
        if live_summary[key] != 0:
            raise AssertionError((key, live_summary[key]))


def main() -> None:
    _assert_production_off_blocked()
    _assert_token_span_derives()
    _assert_seq_len_fallback_derives()
    _assert_blocked_cases()
    _assert_poison_untouched()
    _assert_full_chain()
    print("relaykv_runtime_metadata_derived_req_to_token_entries_smoke=pass")
    print(
        "relaykv_runtime_metadata_derived_req_to_token_entries_summary="
        + json.dumps(
            {
                "production_off": "blocked",
                "token_span": "derived",
                "seq_len_fallback": "derived",
                "blocked_cases": "validated",
                "chain": "derived_to_producer_to_bridge_to_live_read",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
