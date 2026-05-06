from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    build_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke,
    build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    build_relaykv_req_to_token_resolution_payloads_from_real_pool_read_for_smoke,
    summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    summarize_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke,
    summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
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
            )
        )


class _ControlledReqToTokenPool:
    def __init__(self, rows: dict[Any, Any]) -> None:
        self.rows = dict(rows)
        self.calls: list[tuple[Any, Any]] = []

    def __getitem__(self, req_pool_idx: Any) -> Any:
        row = self.rows[req_pool_idx]
        parent = self

        class _Row:
            def __getitem__(self, token_position: Any) -> Any:
                parent.calls.append((req_pool_idx, token_position))
                return row[token_position]

        return _Row()


def _runtime_observation_payload(
    request_id: str = "req-a",
    *,
    req_pool_idx: Any = 7,
    token_span: list[int] | None = None,
    seq_len: int | None = 4,
    poison: _PoisonObject | None = None,
) -> dict[str, Any]:
    adapter_metadata: dict[str, Any] = {"runtime_observation_marker": request_id}
    if seq_len is not None:
        adapter_metadata["seq_len"] = seq_len
    if poison is not None:
        adapter_metadata["poison"] = poison
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
        "engine_block_ref": {"req_pool_idx": req_pool_idx},
        "source_mutated": False,
    }
    if token_span is not None:
        payload["token_span"] = token_span
    if req_pool_idx is not None:
        payload["req_pool_idx"] = req_pool_idx
    return payload


def _assert_zero(summary: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if summary.get(key) != 0:
            raise AssertionError((key, summary.get(key)))


def _assert_read_summary_shape(summary: dict[str, Any], *, enabled: bool) -> None:
    if summary["event_type"] != "relaykv_real_req_to_token_pool_bounded_read_summary":
        raise AssertionError(summary)
    if summary["read_enabled"] is not enabled:
        raise AssertionError(summary)
    _assert_zero(
        summary,
        (
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
        ),
    )


def _read(
    *,
    runtime_observation_payloads: Any = None,
    req_to_token_pool_object: Any = None,
    read_req_to_token_pool: bool,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
    source_path: str = "fake.req_to_token_pool",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = build_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        req_to_token_pool_object=req_to_token_pool_object,
        read_req_to_token_pool=read_req_to_token_pool,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
        source_path=source_path,
    )
    summary = summarize_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
        results,
        read_enabled=read_req_to_token_pool,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
    )
    _assert_read_summary_shape(summary, enabled=read_req_to_token_pool)
    return results, summary


def _assert_read_disabled_blocked() -> None:
    results, summary = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 3])],
        req_to_token_pool_object={7: [10, 11, 12]},
        read_req_to_token_pool=False,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_read_not_enabled":
        raise AssertionError(results[0])
    if summary["req_to_token_read_count"] != 0:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_read_count"] != 0:
        raise AssertionError(summary)


def _assert_missing_inputs_blocked() -> None:
    results, _ = _read(read_req_to_token_pool=True)
    if results[0]["blocked_reason"] != "runtime_observation_payloads_missing":
        raise AssertionError(results[0])

    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 2])],
        read_req_to_token_pool=True,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_object_missing":
        raise AssertionError(results[0])

    results, _ = _read(
        runtime_observation_payloads=[
            _runtime_observation_payload(req_pool_idx=None, token_span=[0, 2])
        ],
        req_to_token_pool_object={7: [1, 2, 3]},
        read_req_to_token_pool=True,
    )
    if results[0]["blocked_reason"] != "req_pool_idx_missing":
        raise AssertionError(results[0])


def _assert_token_span_resolves() -> dict[str, Any]:
    poison = _PoisonObject()
    payload = _runtime_observation_payload("req-span", token_span=[1, 4], poison=poison)
    before = copy.deepcopy(payload)
    results, summary = _read(
        runtime_observation_payloads=[payload],
        req_to_token_pool_object={7: [100, 101, 102, 103, 104]},
        read_req_to_token_pool=True,
    )
    if payload != before:
        raise AssertionError("runtime observation payload mutated")
    if poison.touched:
        raise AssertionError("poison object was touched")
    result = results[0]
    if result["read_state"] != "req_to_token_pool_resolved":
        raise AssertionError(result)
    if result["req_to_token_index_preview"] != [101, 102, 103]:
        raise AssertionError(result)
    if result["req_to_token_index_count"] != 3:
        raise AssertionError(result)
    if result["req_to_token_index_checksum"] != (1 * 101 + 2 * 102 + 3 * 103):
        raise AssertionError(result)
    if summary["req_to_token_read_count"] != 3:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_read_count"] != 3:
        raise AssertionError(summary)
    return result


def _assert_seq_len_fallback_resolves() -> None:
    controlled = _ControlledReqToTokenPool({5: [10, 11, 12, 13]})
    results, summary = _read(
        runtime_observation_payloads=[
            _runtime_observation_payload(
                "req-seq",
                req_pool_idx=5,
                token_span=None,
                seq_len=4,
            )
        ],
        req_to_token_pool_object=controlled,
        read_req_to_token_pool=True,
    )
    if results[0]["read_state"] != "req_to_token_pool_resolved":
        raise AssertionError(results[0])
    if results[0]["token_span"] != [0, 4]:
        raise AssertionError(results[0])
    if controlled.calls != [(5, 0), (5, 1), (5, 2), (5, 3)]:
        raise AssertionError(controlled.calls)
    if summary["req_to_token_read_count"] != 4:
        raise AssertionError(summary)


def _assert_limits_and_errors() -> None:
    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 5])],
        req_to_token_pool_object={7: [1, 2, 3, 4, 5]},
        read_req_to_token_pool=True,
        max_tokens_per_request=4,
    )
    if results[0]["blocked_reason"] != "max_tokens_per_request_exceeded":
        raise AssertionError(results[0])

    results, _ = _read(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-a", req_pool_idx=1, token_span=[0, 3]),
            _runtime_observation_payload("req-b", req_pool_idx=2, token_span=[0, 3]),
        ],
        req_to_token_pool_object={1: [1, 2, 3], 2: [4, 5, 6]},
        read_req_to_token_pool=True,
        max_total_tokens=5,
    )
    if results[1]["blocked_reason"] != "max_total_tokens_exceeded":
        raise AssertionError(results)

    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 3])],
        req_to_token_pool_object={7: [1, "bad", 3]},
        read_req_to_token_pool=True,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(results[0])

    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 3])],
        req_to_token_pool_object={7: [1]},
        read_req_to_token_pool=True,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_index_error":
        raise AssertionError(results[0])


def _assert_conversion_and_chain() -> None:
    results, summary = _read(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-chain", req_pool_idx=9, token_span=[0, 3])
        ],
        req_to_token_pool_object={9: [21, 22, 23]},
        read_req_to_token_pool=True,
    )
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)
    converted = build_relaykv_req_to_token_resolution_payloads_from_real_pool_read_for_smoke(
        results
    )
    payload = converted[0]
    if payload["event_type"] != "relaykv_req_to_token_resolution_result":
        raise AssertionError(payload)
    if payload["resolution_state"] != "req_to_token_resolved":
        raise AssertionError(payload)
    span = payload["full_kv_req_to_token_spans"][0]
    if span["req_to_token_entries"] != [21, 22, 23]:
        raise AssertionError(span)

    bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        explicit_payloads=converted,
        bridge_enabled=True,
    )
    bridge_summary = summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        bridge_results
    )
    _assert_zero(
        bridge_summary,
        (
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
        ),
    )
    if bridge_summary["bridged_count"] <= 0:
        raise AssertionError(bridge_summary)

    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        bridge_results[0]["req_to_token_resolution_payloads"],
        token_to_kv_pool_object={21: 301, 22: 302, 23: 303},
        read_token_to_kv_pool_index=True,
        source_path="fake.token_to_kv_pool",
    )
    live_summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        live_results
    )
    if live_summary["physical_kv_index_resolved_count"] <= 0:
        raise AssertionError(live_summary)
    if live_summary["token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(live_summary)


def main() -> int:
    _assert_read_disabled_blocked()
    _assert_missing_inputs_blocked()
    _assert_token_span_resolves()
    _assert_seq_len_fallback_resolves()
    _assert_limits_and_errors()
    _assert_conversion_and_chain()

    print("relaykv_real_req_to_token_pool_bounded_read_smoke=pass")
    print(
        json.dumps(
            {
                "conversion_chain": "resolved",
                "disabled": "blocked",
                "limits_and_errors": "validated",
                "seq_len_fallback": "resolved",
                "token_span": "resolved",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
