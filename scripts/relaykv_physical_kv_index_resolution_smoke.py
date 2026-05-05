from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_physical_kv_index_resolution_results_for_smoke,
    summarize_relaykv_physical_kv_index_resolution_results_for_smoke,
)


class _PoisonObject:
    def __init__(self) -> None:
        self.cpu_called = False
        self.item_called = False
        self.tolist_called = False
        self.numpy_called = False
        self.iter_called = False
        self.len_called = False
        self.getitem_called = False
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

    def __getitem__(self, index: int) -> None:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

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
                self.getitem_called,
                self.repr_called,
            )
        )


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
    req_pool_idx: Any,
    *,
    resolution_state: str = "req_to_token_resolved",
    layer_id: int = 14,
    spans: list[dict[str, Any]] | None = None,
    blocking_reasons: list[str] | None = None,
    kv_pool_read: bool = False,
    tensor_read: bool = False,
    attention_override: bool = False,
    runtime_writeback: bool = False,
    scheduler_policy_noop: bool = True,
) -> dict[str, Any]:
    resolved_spans = (
        spans
        if spans is not None
        else [
            _req_to_token_span(101, 0, 4, [10, 11, 12, 13]),
        ]
    )
    total_tokens = sum(span["entry_count"] for span in resolved_spans)
    return {
        "event_type": "relaykv_req_to_token_resolution_result",
        "resolution_state": resolution_state,
        "resolution_mode": "readonly_synthetic_table",
        "source": "kv_index_resolution_plan_to_req_to_token_resolution_result",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": 256,
        "layer_id": layer_id,
        "kv_head_group": "kvh-0",
        "decision_state": resolution_state,
        "fallback_reason": None,
        "logical_block_id": resolved_spans[0]["block_id"] if resolved_spans else None,
        "token_span": [
            resolved_spans[0]["token_start"],
            resolved_spans[0]["token_end"],
        ]
        if resolved_spans
        else None,
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": request_id,
        "logical_sequence_id": request_id,
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": {
            "request_id": request_id,
            "req_pool_idx": req_pool_idx,
            "seq_len": 256,
            "pool_source_path": "synthetic.req_to_token_resolution",
            "req_to_token_source": "synthetic_req_to_token_table",
            "req_to_token_backing_type": "dict",
        },
        "engine_block_ref": {
            "req_pool_idx": req_pool_idx,
            "cache_position": 7,
            "req_to_token_entries_preview": list(
                resolved_spans[0]["req_to_token_entries"][:4]
            )
            if resolved_spans
            else [],
            "token_to_kv_pool_index": None,
        },
        "relaykv_working_req_to_token_spans": list(resolved_spans[:1]),
        "full_kv_req_to_token_spans": list(resolved_spans),
        "resolved_block_count": len(resolved_spans),
        "resolved_token_count": total_tokens if resolution_state == "req_to_token_resolved" else 0,
        "req_to_token_entry_count": total_tokens if resolution_state == "req_to_token_resolved" else 0,
        "req_to_token_read": resolution_state == "req_to_token_resolved",
        "req_to_token_read_count": total_tokens if resolution_state == "req_to_token_resolved" else 0,
        "token_to_kv_pool_read": False,
        "token_to_kv_pool_read_count": 0,
        "kv_pool_read": kv_pool_read,
        "kv_snapshot": False,
        "tensor_read": tensor_read,
        "attention_comparison_executed": False,
        "attention_override": attention_override,
        "runtime_writeback": runtime_writeback,
        "scheduler_policy_noop": scheduler_policy_noop,
        "kv_cache_mutation": False,
        "source_mutated": False,
        "blocking_reasons": list(blocking_reasons or []),
        "warning_reasons": [
            "readonly_req_to_token_resolution",
            "no_token_to_kv_pool_read",
        ],
    }


def _assert_pass_flow() -> dict[str, Any]:
    result_a = _resolution_result(
        "req-a",
        7,
        spans=[_req_to_token_span(101, 0, 5, [10, 11, 12, 13, 14])],
    )
    result_b = _resolution_result(
        "req-b",
        8,
        spans=[_req_to_token_span(102, 5, 7, [20, 21])],
        layer_id=15,
    )
    poison = _PoisonObject()
    table = {
        10: 100,
        11: 101,
        12: 102,
        13: 103,
        14: 104,
        "20": 200,
        "21": 201,
        9999: poison,
    }
    inputs = [result_a, result_b]
    before_inputs = copy.deepcopy(inputs)
    before_table = copy.deepcopy(table)

    results = build_relaykv_physical_kv_index_resolution_results_for_smoke(
        inputs,
        token_to_kv_pool_table=table,
        read_token_to_kv_pool=True,
        max_preview_entries=4,
    )

    if inputs != before_inputs:
        raise AssertionError("req_to_token resolution inputs were mutated")
    if table != before_table:
        raise AssertionError("token_to_kv_pool table was mutated")
    if poison.touched:
        raise AssertionError("poison object was touched")
    if len(results) != 2:
        raise AssertionError(results)

    expected_indexes = {
        "req-a": [100, 101, 102, 103, 104],
        "req-b": [200, 201],
    }

    for result in results:
        if result["event_type"] != "relaykv_physical_kv_index_resolution_result":
            raise AssertionError(result)
        if result["resolution_state"] != "physical_kv_index_resolved":
            raise AssertionError(result)
        if result["engine_name"] != "sglang" or result["adapter_name"] != "sglang":
            raise AssertionError(result)
        if result["decision_state"] != "req_to_token_resolved":
            raise AssertionError(result)
        if result["fallback_reason"] is not None:
            raise AssertionError(result)
        if result["position_check_state"] != "not_checked_metadata_only":
            raise AssertionError(result)
        if result["attention_mask_mode"] != "unknown":
            raise AssertionError(result)
        if result["rope_position_consistency"] != "not_checked":
            raise AssertionError(result)
        if result["token_to_kv_pool_read"] is not True:
            raise AssertionError(result)
        if result["kv_pool_read"] is not False or result["tensor_read"] is not False:
            raise AssertionError(result)
        if result["attention_override"] is not False:
            raise AssertionError(result)
        if result["runtime_writeback"] is not False:
            raise AssertionError(result)
        if result["scheduler_policy_noop"] is not True:
            raise AssertionError(result)
        if result["blocking_reasons"] != []:
            raise AssertionError(result)
        if "token_to_kv_pool_index" in result:
            raise AssertionError(result)
        engine_block_ref = result["engine_block_ref"]
        if not isinstance(engine_block_ref, dict):
            raise AssertionError(result)
        if engine_block_ref["cache_position"] != 7:
            raise AssertionError(result)
        if result["adapter_metadata"]["req_pool_idx"] != result["req_pool_idx"]:
            raise AssertionError(result)

        indexes = expected_indexes[result["request_id"]]
        if engine_block_ref["physical_kv_index_preview"] != indexes[:4]:
            raise AssertionError(result)
        if engine_block_ref["physical_kv_index_count"] != len(indexes):
            raise AssertionError(result)
        if engine_block_ref["physical_kv_index_checksum"] != _checksum(indexes):
            raise AssertionError(result)
        if result["token_to_kv_pool_read_count"] != len(indexes):
            raise AssertionError(result)

    if results[0]["engine_block_ref"]["token_to_kv_pool_index"] is not None:
        raise AssertionError(results[0])
    if results[0]["truncated_physical_kv_index_preview"] is not True:
        raise AssertionError(results[0])
    if results[1]["engine_block_ref"]["token_to_kv_pool_index"] is not None:
        raise AssertionError(results[1])
    if results[1]["truncated_physical_kv_index_preview"] is not False:
        raise AssertionError(results[1])

    summary = summarize_relaykv_physical_kv_index_resolution_results_for_smoke(results)
    if summary["physical_kv_index_resolved_count"] != 2:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] != 7:
        raise AssertionError(summary)
    if summary["physical_kv_index_count"] != 7:
        raise AssertionError(summary)
    if summary["physical_kv_index_preview_count"] != 6:
        raise AssertionError(summary)
    if summary["truncated_physical_kv_index_preview_count"] != 1:
        raise AssertionError(summary)
    for key in (
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

    return {"summary": summary, "results": results}


def _assert_list_and_tuple_lookup() -> dict[str, Any]:
    list_table: list[Any] = [None] * 8
    list_table[2] = 302
    list_table[3] = 303
    tuple_table = tuple(
        None if index not in {4, 5} else 400 + index for index in range(8)
    )

    list_result = build_relaykv_physical_kv_index_resolution_results_for_smoke(
        [
            _resolution_result(
                "req-list",
                9,
                spans=[_req_to_token_span(201, 0, 2, [2, 3])],
            )
        ],
        token_to_kv_pool_table=list_table,
        read_token_to_kv_pool=True,
    )[0]
    tuple_result = build_relaykv_physical_kv_index_resolution_results_for_smoke(
        [
            _resolution_result(
                "req-tuple",
                10,
                spans=[_req_to_token_span(202, 2, 4, [4, 5])],
                layer_id=16,
            )
        ],
        token_to_kv_pool_table=tuple_table,
        read_token_to_kv_pool=True,
    )[0]

    if list_result["engine_block_ref"]["physical_kv_index_preview"] != [302, 303]:
        raise AssertionError(list_result)
    if tuple_result["engine_block_ref"]["physical_kv_index_preview"] != [404, 405]:
        raise AssertionError(tuple_result)

    return {"list_result": list_result, "tuple_result": tuple_result}


def _assert_blocked_cases() -> dict[str, Any]:
    base = _resolution_result("req-blocked", 11)
    poison = _PoisonObject()

    cases = [
        (
            "not_req_to_token_resolution_result",
            dict(base, event_type="wrong_event"),
            {10: 100, 11: 101, 12: 102, 13: 103},
            True,
            "not_req_to_token_resolution_result",
        ),
        (
            "req_to_token_resolution_not_resolved",
            _resolution_result(
                "req-blocked-state",
                11,
                resolution_state="blocked",
                blocking_reasons=["req_to_token_table_missing"],
            ),
            {10: 100, 11: 101, 12: 102, 13: 103},
            True,
            "req_to_token_resolution_not_resolved",
        ),
        (
            "read_token_to_kv_pool_not_enabled",
            base,
            {10: 100, 11: 101, 12: 102, 13: 103},
            False,
            "read_token_to_kv_pool_not_enabled",
        ),
        (
            "token_to_kv_pool_table_missing",
            base,
            None,
            True,
            "token_to_kv_pool_table_missing",
        ),
        (
            "token_to_kv_pool_table_not_indexable",
            base,
            poison,
            True,
            "token_to_kv_pool_table_not_indexable",
        ),
        (
            "req_to_token_entries_missing",
            dict(base, full_kv_req_to_token_spans=[{"block_id": 101}]),
            {10: 100},
            True,
            "req_to_token_entries_missing",
        ),
        (
            "req_to_token_entry_not_int",
            _resolution_result(
                "req-bad-entry",
                11,
                spans=[_req_to_token_span(101, 0, 2, [10, "bad"])],
            ),
            {10: 100},
            True,
            "req_to_token_entry_not_int",
        ),
        (
            "token_to_kv_pool_entry_missing",
            _resolution_result(
                "req-missing-physical",
                11,
                spans=[_req_to_token_span(101, 0, 2, [10, 88])],
            ),
            {10: 100},
            True,
            "token_to_kv_pool_entry_missing",
        ),
        (
            "token_to_kv_pool_entry_not_int",
            _resolution_result(
                "req-bad-physical",
                11,
                spans=[_req_to_token_span(101, 0, 2, [10, 11])],
            ),
            {10: 100, 11: "bad"},
            True,
            "token_to_kv_pool_entry_not_int",
        ),
        (
            "kv_pool_read_not_allowed",
            _resolution_result("req-kv-read", 11, kv_pool_read=True),
            {10: 100, 11: 101, 12: 102, 13: 103},
            True,
            "kv_pool_read_not_allowed",
        ),
        (
            "tensor_read_not_allowed",
            _resolution_result("req-tensor-read", 11, tensor_read=True),
            {10: 100, 11: 101, 12: 102, 13: 103},
            True,
            "tensor_read_not_allowed",
        ),
        (
            "attention_override_true_not_allowed",
            _resolution_result("req-attn-override", 11, attention_override=True),
            {10: 100, 11: 101, 12: 102, 13: 103},
            True,
            "attention_override_true_not_allowed",
        ),
        (
            "runtime_writeback_not_allowed",
            _resolution_result("req-runtime-writeback", 11, runtime_writeback=True),
            {10: 100, 11: 101, 12: 102, 13: 103},
            True,
            "runtime_writeback_not_allowed",
        ),
        (
            "scheduler_mutation_not_allowed",
            _resolution_result("req-scheduler", 11, scheduler_policy_noop=False),
            {10: 100, 11: 101, 12: 102, 13: 103},
            True,
            "scheduler_mutation_not_allowed",
        ),
        (
            "max_tokens_per_request_exceeded",
            _resolution_result(
                "req-budget-per-request",
                11,
                spans=[_req_to_token_span(101, 0, 5, [10, 11, 12, 13, 14])],
            ),
            {10: 100, 11: 101, 12: 102, 13: 103, 14: 104},
            True,
            "max_tokens_per_request_exceeded",
        ),
    ]

    observed: dict[str, list[str]] = {}
    for case_name, result, table, read_flag, expected_reason in cases:
        kwargs: dict[str, Any] = {
            "token_to_kv_pool_table": table,
            "read_token_to_kv_pool": read_flag,
        }
        if case_name == "max_tokens_per_request_exceeded":
            kwargs["max_tokens_per_request"] = 4
        outputs = build_relaykv_physical_kv_index_resolution_results_for_smoke(
            [result],
            **kwargs,
        )
        if len(outputs) != 1:
            raise AssertionError(outputs)
        output = outputs[0]
        if output["resolution_state"] != "blocked":
            raise AssertionError(output)
        if expected_reason not in output["blocking_reasons"]:
            raise AssertionError((case_name, output))
        observed[case_name] = list(output["blocking_reasons"])

    total_limit_outputs = build_relaykv_physical_kv_index_resolution_results_for_smoke(
        [
            _resolution_result(
                "req-total-a",
                12,
                spans=[_req_to_token_span(301, 0, 3, [30, 31, 32])],
            ),
            _resolution_result(
                "req-total-b",
                13,
                spans=[_req_to_token_span(302, 3, 6, [33, 34, 35])],
            ),
        ],
        token_to_kv_pool_table={
            30: 300,
            31: 301,
            32: 302,
            33: 303,
            34: 304,
            35: 305,
        },
        read_token_to_kv_pool=True,
        max_total_tokens=5,
    )
    if total_limit_outputs[0]["resolution_state"] != "physical_kv_index_resolved":
        raise AssertionError(total_limit_outputs)
    if "max_total_tokens_exceeded" not in total_limit_outputs[1]["blocking_reasons"]:
        raise AssertionError(total_limit_outputs[1])

    if poison.touched:
        raise AssertionError("non-indexable poison object was touched")

    return {"blocked_reasons": observed}


def main() -> None:
    pass_flow = _assert_pass_flow()
    list_and_tuple = _assert_list_and_tuple_lookup()
    blocked = _assert_blocked_cases()
    print("relaykv_physical_kv_index_resolution_smoke=pass")
    print(
        "relaykv_physical_kv_index_resolution_summary="
        + json.dumps(pass_flow["summary"], sort_keys=True)
    )
    print(
        "relaykv_physical_kv_index_resolution_blocked="
        + json.dumps(blocked["blocked_reasons"], sort_keys=True)
    )
    print(
        "relaykv_physical_kv_index_resolution_variants="
        + json.dumps(
            {
                "list_preview": list_and_tuple["list_result"]["engine_block_ref"][
                    "physical_kv_index_preview"
                ],
                "tuple_preview": list_and_tuple["tuple_result"]["engine_block_ref"][
                    "physical_kv_index_preview"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
