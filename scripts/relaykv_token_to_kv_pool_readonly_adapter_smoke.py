from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke,
    summarize_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke,
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


class FakeTokenToKvPool:
    def __init__(self, token_to_kv_pool: Any, unrelated: Any = None) -> None:
        self._token_to_kv_pool = token_to_kv_pool
        self._unrelated = unrelated
        self.token_to_kv_pool_access_count = 0

    @property
    def token_to_kv_pool(self) -> Any:
        self.token_to_kv_pool_access_count += 1
        return self._token_to_kv_pool

    @property
    def unrelated(self) -> Any:
        raise AssertionError("unrelated field must not be accessed")


class _MissingTokenToKvPool:
    def __init__(self) -> None:
        self.other = "missing"


class _FailingTokenToKvPoolAttr:
    @property
    def token_to_kv_pool(self) -> Any:
        raise RuntimeError("attr access failed")


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
        else [_req_to_token_span(101, 0, 4, [10, 11, 12, 13])]
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
            "cache_position": 9,
            "req_to_token_entries_preview": list(
                resolved_spans[0]["req_to_token_entries"][:4]
            ),
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
    poison = _PoisonObject()
    inputs = [
        _resolution_result(
            "req-a",
            7,
            spans=[_req_to_token_span(101, 0, 5, [10, 11, 12, 13, 14])],
        ),
        _resolution_result(
            "req-b",
            8,
            spans=[_req_to_token_span(102, 5, 7, [20, 21])],
            layer_id=15,
        ),
    ]
    backing = {
        10: 100,
        11: 101,
        12: 102,
        13: 103,
        14: 104,
        "20": 200,
        "21": 201,
        999: poison,
    }
    pool = FakeTokenToKvPool(backing, unrelated=poison)
    before_inputs = copy.deepcopy(inputs)
    before_backing = copy.deepcopy(backing)

    payloads = build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
        inputs,
        token_to_kv_pool_pool=pool,
        read_token_to_kv_pool=True,
        max_preview_entries=4,
    )
    if inputs != before_inputs:
        raise AssertionError("resolution inputs were mutated")
    if backing != before_backing:
        raise AssertionError("backing was mutated")
    if poison.touched:
        raise AssertionError("poison object was touched")
    if pool.token_to_kv_pool_access_count != 1:
        raise AssertionError(pool.token_to_kv_pool_access_count)
    if len(payloads) != 2:
        raise AssertionError(payloads)

    expected_indexes = {
        "req-a": [100, 101, 102, 103, 104],
        "req-b": [200, 201],
    }

    for payload in payloads:
        if payload["event_type"] != "relaykv_token_to_kv_pool_readonly_adapter_payload":
            raise AssertionError(payload)
        if payload["adapter_state"] != "adapter_payload_ready":
            raise AssertionError(payload)
        if payload["adapter_mode"] != "fake_actual_token_to_kv_pool_readonly":
            raise AssertionError(payload)
        if payload["engine_name"] != "sglang" or payload["adapter_name"] != "sglang":
            raise AssertionError(payload)
        if payload["token_to_kv_pool_read"] is not True:
            raise AssertionError(payload)
        if payload["actual_token_to_kv_pool_read"] is not True:
            raise AssertionError(payload)
        if payload["kv_pool_read"] is not False or payload["tensor_read"] is not False:
            raise AssertionError(payload)
        if payload["attention_override"] is not False:
            raise AssertionError(payload)
        if payload["runtime_writeback"] is not False:
            raise AssertionError(payload)
        if payload["scheduler_policy_noop"] is not True:
            raise AssertionError(payload)
        if payload["position_check_state"] != "not_checked_metadata_only":
            raise AssertionError(payload)
        if payload["attention_mask_mode"] != "unknown":
            raise AssertionError(payload)
        if payload["rope_position_consistency"] != "not_checked":
            raise AssertionError(payload)
        if payload["blocking_reasons"] != []:
            raise AssertionError(payload)
        if "physical_kv_indexes" in payload:
            raise AssertionError(payload)

        engine_block_ref = payload["engine_block_ref"]
        if engine_block_ref["cache_position"] != 9:
            raise AssertionError(payload)
        indexes = expected_indexes[payload["request_id"]]
        if engine_block_ref["token_to_kv_pool_index_preview"] != indexes[:4]:
            raise AssertionError(payload)
        if engine_block_ref["physical_kv_index_preview"] != indexes[:4]:
            raise AssertionError(payload)
        if engine_block_ref["physical_kv_index_count"] != len(indexes):
            raise AssertionError(payload)
        if engine_block_ref["physical_kv_index_checksum"] != _checksum(indexes):
            raise AssertionError(payload)
        if payload["entry_checksum"] != _checksum(indexes):
            raise AssertionError(payload)
        if payload["adapter_metadata"]["pool_source_path"] != (
            "token_to_kv_pool_pool.token_to_kv_pool"
        ):
            raise AssertionError(payload)
        if payload["adapter_metadata"]["token_to_kv_pool_backing_type"] != "dict":
            raise AssertionError(payload)

    summary = summarize_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
        payloads
    )
    if summary["adapter_payload_ready_count"] != 2:
        raise AssertionError(summary)
    if summary["actual_token_to_kv_pool_read_count"] != 7:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] != 7:
        raise AssertionError(summary)
    if summary["physical_kv_index_count"] != 7:
        raise AssertionError(summary)
    if summary["preview_entry_count"] != 6:
        raise AssertionError(summary)
    if summary["truncated_preview_count"] != 1:
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

    return {"summary": summary}


def _assert_list_and_tuple_lookup() -> dict[str, Any]:
    list_pool = FakeTokenToKvPool([None, None, 302, 303, None])
    tuple_pool = FakeTokenToKvPool(tuple([None, None, None, None, 404, 405]))

    list_payload = build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
        [_resolution_result("req-list", 9, spans=[_req_to_token_span(201, 0, 2, [2, 3])])],
        token_to_kv_pool_pool=list_pool,
        read_token_to_kv_pool=True,
    )[0]
    tuple_payload = build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
        [_resolution_result("req-tuple", 10, spans=[_req_to_token_span(202, 0, 2, [4, 5])])],
        token_to_kv_pool_pool=tuple_pool,
        read_token_to_kv_pool=True,
    )[0]

    if list_payload["engine_block_ref"]["physical_kv_index_preview"] != [302, 303]:
        raise AssertionError(list_payload)
    if tuple_payload["engine_block_ref"]["physical_kv_index_preview"] != [404, 405]:
        raise AssertionError(tuple_payload)

    return {"list": list_payload, "tuple": tuple_payload}


def _assert_blocked_cases() -> dict[str, Any]:
    poison = _PoisonObject()
    base = _resolution_result("req-blocked", 11)

    cases = [
        (
            "not_req_to_token_resolution_result",
            dict(base, event_type="wrong_event"),
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
            True,
            "not_req_to_token_resolution_result",
        ),
        (
            "req_to_token_resolution_not_resolved",
            _resolution_result(
                "req-not-resolved",
                11,
                resolution_state="blocked",
                blocking_reasons=["req_to_token_table_missing"],
            ),
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
            True,
            "req_to_token_resolution_not_resolved",
        ),
        (
            "read_token_to_kv_pool_not_enabled",
            base,
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
            False,
            "read_token_to_kv_pool_not_enabled",
        ),
        (
            "token_to_kv_pool_object_missing",
            base,
            None,
            True,
            "token_to_kv_pool_object_missing",
        ),
        (
            "token_to_kv_pool_attr_missing",
            base,
            _MissingTokenToKvPool(),
            True,
            "token_to_kv_pool_attr_missing",
        ),
        (
            "token_to_kv_pool_attr_access_failed",
            base,
            _FailingTokenToKvPoolAttr(),
            True,
            "token_to_kv_pool_attr_access_failed",
        ),
        (
            "token_to_kv_pool_backing_not_indexable",
            base,
            FakeTokenToKvPool(poison),
            True,
            "token_to_kv_pool_backing_not_indexable",
        ),
        (
            "req_to_token_entries_missing",
            dict(base, full_kv_req_to_token_spans=[{"block_id": 101}]),
            FakeTokenToKvPool({10: 100}),
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
            FakeTokenToKvPool({10: 100}),
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
            FakeTokenToKvPool({10: 100}),
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
            FakeTokenToKvPool({10: 100, 11: "bad", 99: poison}),
            True,
            "token_to_kv_pool_entry_not_int",
        ),
        (
            "kv_pool_read_not_allowed",
            _resolution_result("req-kv-read", 11, kv_pool_read=True),
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
            True,
            "kv_pool_read_not_allowed",
        ),
        (
            "tensor_read_not_allowed",
            _resolution_result("req-tensor-read", 11, tensor_read=True),
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
            True,
            "tensor_read_not_allowed",
        ),
        (
            "attention_override_true_not_allowed",
            _resolution_result("req-attn-override", 11, attention_override=True),
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
            True,
            "attention_override_true_not_allowed",
        ),
        (
            "runtime_writeback_not_allowed",
            _resolution_result("req-runtime-writeback", 11, runtime_writeback=True),
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
            True,
            "runtime_writeback_not_allowed",
        ),
        (
            "scheduler_mutation_not_allowed",
            _resolution_result("req-scheduler", 11, scheduler_policy_noop=False),
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103}),
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
            FakeTokenToKvPool({10: 100, 11: 101, 12: 102, 13: 103, 14: 104}),
            True,
            "max_tokens_per_request_exceeded",
        ),
    ]

    observed: dict[str, list[str]] = {}
    for case_name, result, pool, read_flag, expected_reason in cases:
        kwargs: dict[str, Any] = {
            "token_to_kv_pool_pool": pool,
            "read_token_to_kv_pool": read_flag,
        }
        if case_name == "max_tokens_per_request_exceeded":
            kwargs["max_tokens_per_request"] = 4
        payload = build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
            [result],
            **kwargs,
        )[0]
        if payload["adapter_state"] != "blocked":
            raise AssertionError(payload)
        if expected_reason not in payload["blocking_reasons"]:
            raise AssertionError((case_name, payload))
        observed[case_name] = list(payload["blocking_reasons"])

    total_limit_payloads = build_relaykv_token_to_kv_pool_readonly_adapter_payloads_for_smoke(
        [
            _resolution_result(
                "req-total-a",
                12,
                spans=[_req_to_token_span(301, 0, 3, [30, 31, 32])],
            ),
            _resolution_result(
                "req-total-b",
                13,
                spans=[_req_to_token_span(302, 0, 3, [33, 34, 35])],
            ),
        ],
        token_to_kv_pool_pool=FakeTokenToKvPool(
            {30: 300, 31: 301, 32: 302, 33: 303, 34: 304, 35: 305}
        ),
        read_token_to_kv_pool=True,
        max_total_tokens=5,
    )
    if total_limit_payloads[0]["adapter_state"] != "adapter_payload_ready":
        raise AssertionError(total_limit_payloads)
    if "max_total_tokens_exceeded" not in total_limit_payloads[1]["blocking_reasons"]:
        raise AssertionError(total_limit_payloads[1])
    if poison.touched:
        raise AssertionError("poison object was touched during blocked cases")

    return {"blocked_reasons": observed}


def main() -> None:
    pass_flow = _assert_pass_flow()
    variants = _assert_list_and_tuple_lookup()
    blocked = _assert_blocked_cases()
    print("relaykv_token_to_kv_pool_readonly_adapter_smoke=pass")
    print(
        "relaykv_token_to_kv_pool_readonly_adapter_summary="
        + json.dumps(pass_flow["summary"], sort_keys=True)
    )
    print(
        "relaykv_token_to_kv_pool_readonly_adapter_blocked="
        + json.dumps(blocked["blocked_reasons"], sort_keys=True)
    )
    print(
        "relaykv_token_to_kv_pool_readonly_adapter_variants="
        + json.dumps(
            {
                "list_preview": variants["list"]["engine_block_ref"][
                    "physical_kv_index_preview"
                ],
                "tuple_preview": variants["tuple"]["engine_block_ref"][
                    "physical_kv_index_preview"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
