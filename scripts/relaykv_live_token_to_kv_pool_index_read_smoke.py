from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    RelayKVBlockMeta,
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
)


class _PoisonObject:
    def __init__(self) -> None:
        self.cpu_called = False
        self.item_called = False
        self.tolist_called = False
        self.numpy_called = False
        self.iter_called = False
        self.len_called = False
        self.repr_called = False

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PoisonObject":
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
                self.item_called,
                self.tolist_called,
                self.numpy_called,
                self.iter_called,
                self.len_called,
                self.repr_called,
            )
        )


class _ControlledIndexable:
    def __init__(self, values: dict[Any, Any], *, shape: tuple[int, ...] | None = None):
        self._values = dict(values)
        self.shape = shape
        self.getitem_calls: list[Any] = []

    def __getitem__(self, index: Any) -> Any:
        self.getitem_calls.append(index)
        if index in self._values:
            return self._values[index]
        index_as_str = str(index)
        if index_as_str in self._values:
            return self._values[index_as_str]
        return None


class _UnsupportedObject:
    pass


class _RaisingIndexObject:
    def __getitem__(self, index: Any) -> Any:
        raise RuntimeError(f"bad index {index}")


def _checksum(values: list[int]) -> int:
    return sum((index + 1) * value for index, value in enumerate(values)) % 1000000007


def _req_to_token_span(
    block_id: int,
    token_start: int,
    token_end: int,
    req_to_token_entries: list[Any],
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "token_start": token_start,
        "token_end": token_end,
        "token_count": token_end - token_start,
        "req_to_token_entries": list(req_to_token_entries),
        "entry_count": len(req_to_token_entries),
        "resolution_source": "synthetic_req_to_token_table",
    }


def _resolution_result(
    request_id: str,
    *,
    layer_id: int = 14,
    kv_head_group: int = 2,
    spans: list[dict[str, Any]] | None = None,
    resolution_state: str = "req_to_token_resolved",
    event_type: str = "relaykv_req_to_token_resolution_result",
    kv_pool_read: bool = False,
    kv_snapshot: bool = False,
    tensor_read: bool = False,
    attention_override: bool = False,
    attention_comparison_executed: bool = False,
    runtime_writeback: bool = False,
    scheduler_policy_noop: bool = True,
    kv_cache_mutation: bool = False,
    source_mutated: bool = False,
) -> dict[str, Any]:
    resolved_spans = (
        spans
        if spans is not None
        else [_req_to_token_span(101, 0, 3, [10, 11, 12])]
    )
    total_tokens = sum(span["entry_count"] for span in resolved_spans)
    return {
        "event_type": event_type,
        "resolution_state": resolution_state,
        "resolution_mode": "readonly_synthetic_table",
        "source": "kv_index_resolution_plan_to_req_to_token_resolution_result",
        "request_id": request_id,
        "req_pool_idx": 7,
        "seq_len": 256,
        "layer_id": layer_id,
        "kv_head_group": kv_head_group,
        "kv_class": "FULL",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": f"engine-{request_id}",
        "logical_sequence_id": f"seq-{request_id}",
        "logical_block_id": resolved_spans[0]["block_id"] if resolved_spans else None,
        "token_span": [resolved_spans[0]["token_start"], resolved_spans[0]["token_end"]],
        "decision_state": "req_to_token_resolved",
        "fallback_reason": None,
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": {
            "req_pool_idx": 7,
            "existing_metadata": True,
            "reserved_block_meta": RelayKVBlockMeta().to_dict(),
        },
        "engine_block_ref": {
            "req_pool_idx": 7,
            "cache_position": 9,
            "token_to_kv_pool_index": 777,
        },
        "full_kv_req_to_token_spans": list(resolved_spans),
        "resolved_block_count": len(resolved_spans),
        "resolved_token_count": total_tokens if resolution_state == "req_to_token_resolved" else 0,
        "req_to_token_entry_count": total_tokens if resolution_state == "req_to_token_resolved" else 0,
        "req_to_token_read": True,
        "req_to_token_read_count": total_tokens,
        "actual_req_to_token_pool_read": False,
        "actual_req_to_token_pool_read_count": 0,
        "token_to_kv_pool_read": False,
        "token_to_kv_pool_read_count": 0,
        "actual_token_to_kv_pool_read": False,
        "actual_token_to_kv_pool_read_count": 0,
        "live_token_to_kv_pool_index_read": False,
        "live_token_to_kv_pool_index_read_count": 0,
        "kv_pool_read": kv_pool_read,
        "kv_snapshot": kv_snapshot,
        "tensor_read": tensor_read,
        "attention_comparison_executed": attention_comparison_executed,
        "attention_override": attention_override,
        "runtime_writeback": runtime_writeback,
        "scheduler_policy_noop": scheduler_policy_noop,
        "kv_cache_mutation": kv_cache_mutation,
        "source_mutated": source_mutated,
        "blocking_reasons": [],
        "warning_reasons": ["readonly_req_to_token_resolution"],
    }


def _assert_result_schema(
    result: dict[str, Any],
    *,
    expected_request_id: str,
    expected_indexes: list[int],
    expected_type: str,
    expected_source_path: str,
    expected_shape: list[int] | None,
    expected_truncated: bool,
) -> None:
    if result["event_type"] != "relaykv_live_token_to_kv_pool_index_read_result":
        raise AssertionError(result)
    if result["resolution_state"] != "physical_kv_index_resolved":
        raise AssertionError(result)
    if result["adapter_mode"] != "live_token_to_kv_pool_bounded_index_read":
        raise AssertionError(result)
    if result["engine_name"] != "sglang" or result["adapter_name"] != "sglang":
        raise AssertionError(result)
    if result["decision_state"] != "SHADOW_ONLY":
        raise AssertionError(result)
    if result["engine_request_id"] != f"engine-{expected_request_id}":
        raise AssertionError(result)
    if result["logical_sequence_id"] != f"seq-{expected_request_id}":
        raise AssertionError(result)
    if result["position_check_state"] != "not_checked_metadata_only":
        raise AssertionError(result)
    if result["attention_mask_mode"] != "unknown":
        raise AssertionError(result)
    if result["rope_position_consistency"] != "not_checked":
        raise AssertionError(result)
    if result["blocking_reasons"] != []:
        raise AssertionError(result)
    if "physical_kv_indexes" in result:
        raise AssertionError(result)

    engine_block_ref = result["engine_block_ref"]
    if engine_block_ref["token_to_kv_pool_index"] is not None:
        raise AssertionError(result)
    if engine_block_ref["cache_position"] is not None:
        raise AssertionError(result)
    if engine_block_ref["physical_kv_index_preview"] != expected_indexes[:4]:
        raise AssertionError(result)
    if engine_block_ref["physical_kv_index_count"] != len(expected_indexes):
        raise AssertionError(result)
    if engine_block_ref["physical_kv_index_checksum"] != _checksum(expected_indexes):
        raise AssertionError(result)

    metadata = result["adapter_metadata"]
    if metadata["token_to_kv_pool_source_path"] != expected_source_path:
        raise AssertionError(result)
    if metadata["token_to_kv_pool_type"] != expected_type:
        raise AssertionError(result)
    if metadata["token_to_kv_pool_shape"] != expected_shape:
        raise AssertionError(result)
    if metadata["live_index_read_enabled"] is not True:
        raise AssertionError(result)
    if metadata["truncated_preview"] is not expected_truncated:
        raise AssertionError(result)
    if metadata["reserved_block_meta"] != RelayKVBlockMeta().to_dict():
        raise AssertionError(result)


def _assert_pass_flows() -> dict[str, Any]:
    poison = _PoisonObject()
    base_inputs = [
        _resolution_result(
            "req-dict-int",
            spans=[_req_to_token_span(101, 0, 5, [10, 11, 12, 13, 14])],
        ),
        _resolution_result(
            "req-dict-str",
            layer_id=15,
            spans=[_req_to_token_span(102, 0, 2, [20, 21])],
        ),
    ]
    dict_object = {
        10: 100,
        11: 101,
        12: 102,
        13: 103,
        14: 104,
        "20": 200,
        "21": 201,
        999: poison,
    }
    before_inputs = copy.deepcopy(base_inputs)
    before_dict_object = copy.deepcopy(dict_object)
    results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        base_inputs,
        token_to_kv_pool_object=dict_object,
        read_token_to_kv_pool_index=True,
        max_preview_entries=4,
        source_path="runtime.token_to_kv_pool",
    )
    if base_inputs != before_inputs:
        raise AssertionError("base inputs mutated")
    if dict_object != before_dict_object:
        raise AssertionError("dict object mutated")
    if poison.touched:
        raise AssertionError("poison object touched")

    expected = {
        "req-dict-int": [100, 101, 102, 103, 104],
        "req-dict-str": [200, 201],
    }
    for result in results:
        _assert_result_schema(
            result,
            expected_request_id=result["request_id"],
            expected_indexes=expected[result["request_id"]],
            expected_type="dict",
            expected_source_path="runtime.token_to_kv_pool",
            expected_shape=None,
            expected_truncated=(result["request_id"] == "req-dict-int"),
        )

    list_result = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        [_resolution_result("req-list", spans=[_req_to_token_span(201, 0, 2, [2, 3])])],
        token_to_kv_pool_object=[None, None, 302, 303, None],
        read_token_to_kv_pool_index=True,
        max_preview_entries=4,
        source_path="runtime.list_token_to_kv_pool",
    )[0]
    _assert_result_schema(
        list_result,
        expected_request_id="req-list",
        expected_indexes=[302, 303],
        expected_type="list",
        expected_source_path="runtime.list_token_to_kv_pool",
        expected_shape=None,
        expected_truncated=False,
    )

    tuple_result = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        [_resolution_result("req-tuple", spans=[_req_to_token_span(202, 0, 2, [4, 5])])],
        token_to_kv_pool_object=(None, None, None, None, 404, 405),
        read_token_to_kv_pool_index=True,
        max_preview_entries=4,
        source_path="runtime.tuple_token_to_kv_pool",
    )[0]
    _assert_result_schema(
        tuple_result,
        expected_request_id="req-tuple",
        expected_indexes=[404, 405],
        expected_type="tuple",
        expected_source_path="runtime.tuple_token_to_kv_pool",
        expected_shape=None,
        expected_truncated=False,
    )

    controlled = _ControlledIndexable({30: 500, 31: 501}, shape=(64,))
    controlled_result = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        [_resolution_result("req-controlled", spans=[_req_to_token_span(203, 0, 2, [30, 31])])],
        token_to_kv_pool_object=controlled,
        read_token_to_kv_pool_index=True,
        max_preview_entries=4,
        source_path="runtime.controlled_token_to_kv_pool",
    )[0]
    if controlled.getitem_calls != [30, 31]:
        raise AssertionError(controlled.getitem_calls)
    _assert_result_schema(
        controlled_result,
        expected_request_id="req-controlled",
        expected_indexes=[500, 501],
        expected_type="_ControlledIndexable",
        expected_source_path="runtime.controlled_token_to_kv_pool",
        expected_shape=[64],
        expected_truncated=False,
    )

    summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        results + [list_result, tuple_result, controlled_result]
    )
    if summary["physical_kv_index_resolved_count"] != 5:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] != 13:
        raise AssertionError(summary)
    if summary["actual_token_to_kv_pool_read_count"] != 13:
        raise AssertionError(summary)
    if summary["live_token_to_kv_pool_index_read_count"] != 13:
        raise AssertionError(summary)
    if summary["physical_kv_index_count"] != 13:
        raise AssertionError(summary)
    if summary["physical_kv_index_preview_count"] != 12:
        raise AssertionError(summary)
    if summary["truncated_preview_count"] != 1:
        raise AssertionError(summary)
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
        if summary[key] != 0:
            raise AssertionError((key, summary[key]))

    return {"summary": summary}


def _assert_blocked_cases() -> dict[str, list[str]]:
    base = _resolution_result("req-base")
    cases = [
        ("live_index_read_not_enabled", base, {10: 100, 11: 101, 12: 102}, False, {}),
        ("not_req_to_token_resolution_result", dict(base, event_type="wrong_event"), {10: 100, 11: 101, 12: 102}, True, {}),
        (
            "req_to_token_resolution_not_resolved",
            _resolution_result("req-not-resolved", resolution_state="blocked"),
            {10: 100, 11: 101, 12: 102},
            True,
            {},
        ),
        ("token_to_kv_pool_object_missing", base, None, True, {}),
        ("token_to_kv_pool_object_not_indexable", base, _UnsupportedObject(), True, {}),
        ("req_to_token_entries_missing", dict(base, full_kv_req_to_token_spans=[{"block_id": 101}]), {10: 100}, True, {}),
        (
            "req_to_token_entry_not_int",
            _resolution_result("req-bad-entry", spans=[_req_to_token_span(101, 0, 2, [10, "bad"])]),
            {10: 100},
            True,
            {},
        ),
        (
            "token_to_kv_pool_entry_missing",
            _resolution_result("req-missing-index", spans=[_req_to_token_span(101, 0, 2, [10, 88])]),
            {10: 100},
            True,
            {},
        ),
        (
            "token_to_kv_pool_entry_not_int",
            _resolution_result("req-bad-physical", spans=[_req_to_token_span(101, 0, 2, [10, 11])]),
            {10: 100, 11: "bad"},
            True,
            {},
        ),
        (
            "max_tokens_per_request_exceeded",
            _resolution_result("req-max-per-request", spans=[_req_to_token_span(101, 0, 5, [10, 11, 12, 13, 14])]),
            {10: 100, 11: 101, 12: 102, 13: 103, 14: 104},
            True,
            {"max_tokens_per_request": 4},
        ),
        (
            "max_total_tokens_exceeded",
            [
                _resolution_result("req-total-1", spans=[_req_to_token_span(101, 0, 3, [10, 11, 12])]),
                _resolution_result("req-total-2", spans=[_req_to_token_span(102, 0, 3, [20, 21, 22])]),
            ],
            {10: 100, 11: 101, 12: 102, 20: 200, 21: 201, 22: 202},
            True,
            {"max_total_tokens": 5},
        ),
        (
            "token_to_kv_pool_index_read_failed",
            _resolution_result("req-read-failed", spans=[_req_to_token_span(101, 0, 1, [10])]),
            _RaisingIndexObject(),
            True,
            {},
        ),
        ("kv_pool_read_not_allowed", _resolution_result("req-kv-read", kv_pool_read=True), {10: 100, 11: 101, 12: 102}, True, {}),
        ("kv_snapshot_not_allowed", _resolution_result("req-kv-snapshot", kv_snapshot=True), {10: 100, 11: 101, 12: 102}, True, {}),
        ("tensor_read_not_allowed", _resolution_result("req-tensor-read", tensor_read=True), {10: 100, 11: 101, 12: 102}, True, {}),
        (
            "attention_override_true_not_allowed",
            _resolution_result("req-attn-override", attention_override=True),
            {10: 100, 11: 101, 12: 102},
            True,
            {},
        ),
        (
            "attention_comparison_executed_not_allowed",
            _resolution_result("req-attn-comp", attention_comparison_executed=True),
            {10: 100, 11: 101, 12: 102},
            True,
            {},
        ),
        (
            "runtime_writeback_not_allowed",
            _resolution_result("req-runtime-writeback", runtime_writeback=True),
            {10: 100, 11: 101, 12: 102},
            True,
            {},
        ),
        (
            "scheduler_mutation_not_allowed",
            _resolution_result("req-scheduler", scheduler_policy_noop=False),
            {10: 100, 11: 101, 12: 102},
            True,
            {},
        ),
        (
            "source_mutation_not_allowed",
            _resolution_result("req-source-mutated", source_mutated=True),
            {10: 100, 11: 101, 12: 102},
            True,
            {},
        ),
    ]

    observed: dict[str, list[str]] = {}
    for expected_reason, result_or_results, token_to_kv_pool_object, read_flag, extra in cases:
        results_input = (
            result_or_results
            if isinstance(result_or_results, list)
            else [result_or_results]
        )
        results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
            results_input,
            token_to_kv_pool_object=token_to_kv_pool_object,
            read_token_to_kv_pool_index=read_flag,
            source_path="runtime.blocked_case",
            **extra,
        )
        target = results[-1]
        if target["resolution_state"] != "blocked":
            raise AssertionError((expected_reason, target))
        if expected_reason not in target["blocking_reasons"]:
            raise AssertionError((expected_reason, target["blocking_reasons"]))
        if target["decision_state"] != "SHADOW_ONLY":
            raise AssertionError(target)
        if target["engine_block_ref"]["token_to_kv_pool_index"] is not None:
            raise AssertionError(target)
        observed[expected_reason] = list(target["blocking_reasons"])

    return observed


def main() -> None:
    outputs = {
        "pass_flow": _assert_pass_flows(),
        "blocked": _assert_blocked_cases(),
    }
    print("relaykv_live_token_to_kv_pool_index_read_smoke=pass")
    print(
        "relaykv_live_token_to_kv_pool_index_read_summary="
        + json.dumps(outputs, sort_keys=True)
    )


if __name__ == "__main__":
    main()
