from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke,
    summarize_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke,
)


class _PoisonMethodObject:
    def __init__(self) -> None:
        self.item_called = False
        self.tolist_called = False
        self.cpu_called = False
        self.numpy_called = False
        self.repr_called = False

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def numpy(self) -> None:
        self.numpy_called = True
        raise AssertionError("numpy() must not be called")

    def __repr__(self) -> str:
        self.repr_called = True
        raise AssertionError("__repr__() must not be called")

    @property
    def touched(self) -> bool:
        return any(
            (
                self.item_called,
                self.tolist_called,
                self.cpu_called,
                self.numpy_called,
                self.repr_called,
            )
        )


class _ScalarLikeValue:
    shape = ()
    dtype = "bf16"
    device = "cuda:0"
    ndim = 0


class _OneElementTensorLikeValue:
    shape = (1,)
    dtype = "int32"
    device = "cuda:1"
    ndim = 1


class _NestedValue:
    def __init__(self, child: Any) -> None:
        self.shape = (2, 1)
        self.dtype = "nested"
        self.device = "meta"
        self.child = child


class _LenOneValue:
    dtype = "opaque"

    def __len__(self) -> int:
        return 1


class _ControlledReqToTokenPool:
    def __init__(self, rows: dict[Any, Any]) -> None:
        self.rows = copy.deepcopy(rows)
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


def _inspect(
    *,
    runtime_observation_payloads: Any = None,
    req_to_token_pool_object: Any = None,
    enabled: bool,
    max_tokens_per_request: int = 8,
    max_total_tokens: int = 16,
    source_path: str = "fake.req_to_token_pool",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = build_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        req_to_token_pool_object=req_to_token_pool_object,
        inspect_req_to_token_pool_value_shape=enabled,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
        source_path=source_path,
    )
    summary = summarize_relaykv_req_to_token_pool_value_shape_inspection_results_for_smoke(
        results,
        inspection_enabled=enabled,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
    )
    if summary["event_type"] != "relaykv_req_to_token_pool_value_shape_inspection_summary":
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
    return results, summary


def _assert_no_value_leak(result: dict[str, Any]) -> None:
    payload = json.dumps(result, sort_keys=True)
    for forbidden_key in ("req_to_token_index_preview", "req_to_token_entries"):
        if forbidden_key in payload:
            raise AssertionError(result)
    for observation in result["value_shape_observations"]:
        if "value" in observation:
            raise AssertionError(observation)
        if "token_position" in observation:
            raise AssertionError(observation)


def _assert_disabled_blocked() -> None:
    results, summary = _inspect(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 2])],
        req_to_token_pool_object={7: [1, 2]},
        enabled=False,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_value_shape_inspection_not_enabled":
        raise AssertionError(results[0])
    if summary["req_to_token_read_count"] != 0:
        raise AssertionError(summary)


def _assert_missing_inputs_blocked() -> None:
    results, _ = _inspect(enabled=True)
    if results[0]["blocked_reason"] != "runtime_observation_payloads_missing":
        raise AssertionError(results[0])

    results, _ = _inspect(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 2])],
        enabled=True,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_object_missing":
        raise AssertionError(results[0])


def _assert_int_value_inspected() -> None:
    payload = _runtime_observation_payload("req-int", token_span=[0, 2])
    before = copy.deepcopy(payload)
    results, summary = _inspect(
        runtime_observation_payloads=[payload],
        req_to_token_pool_object={7: [11, 12]},
        enabled=True,
    )
    if payload != before:
        raise AssertionError("runtime observation payload mutated")
    result = results[0]
    if result["inspection_state"] != "inspected":
        raise AssertionError(result)
    if result["probed_value_count"] != 2:
        raise AssertionError(result)
    observation = result["value_shape_observations"][0]
    if observation["value_is_int"] is not True:
        raise AssertionError(observation)
    if observation["value_type_name"] != "int":
        raise AssertionError(observation)
    if summary["req_to_token_read_count"] != 2:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_read_count"] != 2:
        raise AssertionError(summary)
    _assert_no_value_leak(result)


def _assert_scalar_and_tensor_like_inspected() -> None:
    results, summary = _inspect(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-scalar", req_pool_idx=1, token_span=[0, 1]),
            _runtime_observation_payload("req-tensor", req_pool_idx=2, token_span=[0, 1]),
        ],
        req_to_token_pool_object={
            1: [_ScalarLikeValue()],
            2: [_OneElementTensorLikeValue()],
        },
        enabled=True,
    )
    scalar_obs = results[0]["value_shape_observations"][0]
    if scalar_obs["value_shape"] != []:
        raise AssertionError(scalar_obs)
    if scalar_obs["value_is_scalar_like"] is not True:
        raise AssertionError(scalar_obs)
    if scalar_obs["value_is_one_element_like"] is not True:
        raise AssertionError(scalar_obs)

    tensor_obs = results[1]["value_shape_observations"][0]
    if tensor_obs["value_shape"] != [1]:
        raise AssertionError(tensor_obs)
    if tensor_obs["value_is_tensor_like"] is not True:
        raise AssertionError(tensor_obs)
    if tensor_obs["value_dtype"] != "int32":
        raise AssertionError(tensor_obs)
    if tensor_obs["value_device"] != "cuda:1":
        raise AssertionError(tensor_obs)
    if summary["observed_one_element_like_count"] < 2:
        raise AssertionError(summary)


def _assert_list_tuple_and_nested_nonrecursive() -> None:
    poison = _PoisonMethodObject()
    nested = _NestedValue(poison)
    len_one = _LenOneValue()
    results, _ = _inspect(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-list", req_pool_idx=3, token_span=[0, 1]),
            _runtime_observation_payload("req-tuple", req_pool_idx=4, token_span=[0, 1]),
            _runtime_observation_payload("req-nested", req_pool_idx=5, token_span=[0, 2]),
        ],
        req_to_token_pool_object={
            3: [[1, 2, 3]],
            4: [(4,)],
            5: [nested, len_one],
        },
        enabled=True,
    )
    list_obs = results[0]["value_shape_observations"][0]
    if list_obs["value_is_list"] is not True or list_obs["value_has_len"] is not True:
        raise AssertionError(list_obs)

    tuple_obs = results[1]["value_shape_observations"][0]
    if tuple_obs["value_is_tuple"] is not True:
        raise AssertionError(tuple_obs)

    nested_obs = results[2]["value_shape_observations"][0]
    if nested_obs["value_shape"] != [2, 1]:
        raise AssertionError(nested_obs)
    if poison.touched:
        raise AssertionError("nested child poison object was touched")
    len_one_obs = results[2]["value_shape_observations"][1]
    if len_one_obs["value_has_len"] is not True or len_one_obs["value_len"] != 1:
        raise AssertionError(len_one_obs)
    if len_one_obs["value_is_one_element_like"] is not True:
        raise AssertionError(len_one_obs)


def _assert_poison_not_touched() -> None:
    poison = _PoisonMethodObject()
    results, _ = _inspect(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-poison", req_pool_idx=6, token_span=[0, 1])
        ],
        req_to_token_pool_object={6: [poison]},
        enabled=True,
    )
    if poison.touched:
        raise AssertionError("poison object was touched")
    observation = results[0]["value_shape_observations"][0]
    if observation["value_type_name"] != "_PoisonMethodObject":
        raise AssertionError(observation)
    _assert_no_value_leak(results[0])


def _assert_limits_enforced() -> None:
    results, _ = _inspect(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 9])],
        req_to_token_pool_object={7: list(range(9))},
        enabled=True,
        max_tokens_per_request=8,
    )
    if results[0]["blocked_reason"] != "max_tokens_per_request_exceeded":
        raise AssertionError(results[0])

    results, _ = _inspect(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-a", req_pool_idx=1, token_span=[0, 8]),
            _runtime_observation_payload("req-b", req_pool_idx=2, token_span=[0, 8]),
        ],
        req_to_token_pool_object={1: list(range(8)), 2: list(range(8))},
        enabled=True,
        max_total_tokens=12,
    )
    if results[1]["blocked_reason"] != "max_total_tokens_exceeded":
        raise AssertionError(results)


def main() -> int:
    _assert_disabled_blocked()
    _assert_missing_inputs_blocked()
    _assert_int_value_inspected()
    _assert_scalar_and_tensor_like_inspected()
    _assert_list_tuple_and_nested_nonrecursive()
    _assert_poison_not_touched()
    _assert_limits_enforced()

    print("relaykv_req_to_token_pool_value_shape_inspection_smoke=pass")
    print(
        json.dumps(
            {
                "disabled_and_missing": "blocked",
                "int": "inspected",
                "limits": "enforced",
                "list_tuple_nested": "metadata_only",
                "poison": "untouched",
                "scalar_tensor_like": "inspected_without_conversion",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
