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


class _BaseTensorLike:
    def __init__(self, *, shape: Any, dtype: str = "torch.int32", device: str = "cuda:0") -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.item_called = 0
        self.cpu_called = 0
        self.tolist_called = 0
        self.numpy_called = 0
        self.repr_called = 0

    def cpu(self) -> None:
        self.cpu_called += 1
        raise AssertionError("cpu() must not be called")

    def tolist(self) -> None:
        self.tolist_called += 1
        raise AssertionError("tolist() must not be called")

    def numpy(self) -> None:
        self.numpy_called += 1
        raise AssertionError("numpy() must not be called")

    def __repr__(self) -> str:
        self.repr_called += 1
        raise AssertionError("__repr__() must not be called")

    @property
    def forbidden_touched(self) -> bool:
        return any((self.cpu_called, self.tolist_called, self.numpy_called, self.repr_called))


class _ScalarIntTensorLike(_BaseTensorLike):
    def __init__(self, value: int) -> None:
        super().__init__(shape=[])
        self._value = value

    def item(self) -> int:
        self.item_called += 1
        return self._value


class _OneElementIntTensorLike(_BaseTensorLike):
    ndim = 1

    def __init__(self, value: int) -> None:
        super().__init__(shape=(1,))
        self._value = value

    def item(self) -> int:
        self.item_called += 1
        return self._value


class _VectorTensorLike(_BaseTensorLike):
    ndim = 1

    def __init__(self) -> None:
        super().__init__(shape=(2,))

    def item(self) -> int:
        self.item_called += 1
        raise AssertionError("item() must not be called for multi-element tensor")


class _BoolItemTensorLike(_BaseTensorLike):
    def __init__(self) -> None:
        super().__init__(shape=[])

    def item(self) -> bool:
        self.item_called += 1
        return True


class _FloatItemTensorLike(_BaseTensorLike):
    def __init__(self) -> None:
        super().__init__(shape=[])

    def item(self) -> float:
        self.item_called += 1
        return 1.5


class _RaisingItemTensorLike(_BaseTensorLike):
    def __init__(self) -> None:
        super().__init__(shape=[])

    def item(self) -> int:
        self.item_called += 1
        raise RuntimeError("item failed")


class _PoisonTensorLike(_BaseTensorLike):
    def __init__(self) -> None:
        super().__init__(shape=[])

    def item(self) -> int:
        self.item_called += 1
        return 42


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


def _read(
    *,
    runtime_observation_payloads: Any = None,
    req_to_token_pool_object: Any = None,
    read_req_to_token_pool: bool = True,
    allow_scalar_tensor_item_conversion: bool = False,
    max_tokens_per_request: int = 256,
    max_total_tokens: int = 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = build_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        req_to_token_pool_object=req_to_token_pool_object,
        read_req_to_token_pool=read_req_to_token_pool,
        allow_scalar_tensor_item_conversion=allow_scalar_tensor_item_conversion,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
        source_path="fake.req_to_token_pool",
    )
    summary = summarize_relaykv_real_req_to_token_pool_bounded_read_results_for_smoke(
        results,
        read_enabled=read_req_to_token_pool,
        allow_scalar_tensor_item_conversion=allow_scalar_tensor_item_conversion,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
    )
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


def _assert_default_off_blocks_and_no_item() -> None:
    tensor = _ScalarIntTensorLike(11)
    results, summary = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [tensor]},
        allow_scalar_tensor_item_conversion=False,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(results[0])
    if tensor.item_called != 0:
        raise AssertionError(tensor.item_called)
    if summary["scalar_tensor_item_conversion_attempted_count"] != 0:
        raise AssertionError(summary)
    if summary["scalar_tensor_item_conversion_succeeded_count"] != 0:
        raise AssertionError(summary)


def _assert_zero_dim_tensor_resolves() -> None:
    tensor = _ScalarIntTensorLike(21)
    results, summary = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [tensor]},
        allow_scalar_tensor_item_conversion=True,
    )
    result = results[0]
    if result["read_state"] != "req_to_token_pool_resolved":
        raise AssertionError(result)
    if tensor.item_called != 1:
        raise AssertionError(tensor.item_called)
    if result["req_to_token_index_preview"] != [21]:
        raise AssertionError(result)
    if result["req_to_token_index_count"] != 1:
        raise AssertionError(result)
    if result["scalar_tensor_item_conversion_succeeded"] is not True:
        raise AssertionError(result)
    if summary["scalar_tensor_item_conversion_succeeded_count"] <= 0:
        raise AssertionError(summary)


def _assert_one_element_tensor_resolves() -> None:
    tensor = _OneElementIntTensorLike(31)
    results, summary = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [tensor]},
        allow_scalar_tensor_item_conversion=True,
    )
    if results[0]["read_state"] != "req_to_token_pool_resolved":
        raise AssertionError(results[0])
    if tensor.item_called != 1:
        raise AssertionError(tensor.item_called)
    if results[0]["scalar_tensor_item_conversion_shape"] != [1]:
        raise AssertionError(results[0])
    if summary["resolved_count"] <= 0:
        raise AssertionError(summary)


def _assert_vector_tensor_blocks_without_item() -> None:
    tensor = _VectorTensorLike()
    results, summary = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [tensor]},
        allow_scalar_tensor_item_conversion=True,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(results[0])
    if (
        results[0]["scalar_tensor_item_conversion_blocked_reason"]
        != "scalar_tensor_item_conversion_shape_not_scalar_or_one_element"
    ):
        raise AssertionError(results[0])
    if tensor.item_called != 0:
        raise AssertionError(tensor.item_called)
    if summary["scalar_tensor_item_conversion_blocked_count"] <= 0:
        raise AssertionError(summary)


def _assert_bool_float_and_raise_block() -> None:
    bool_tensor = _BoolItemTensorLike()
    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [bool_tensor]},
        allow_scalar_tensor_item_conversion=True,
    )
    if (
        results[0]["scalar_tensor_item_conversion_blocked_reason"]
        != "scalar_tensor_item_conversion_item_not_int"
    ):
        raise AssertionError(results[0])

    float_tensor = _FloatItemTensorLike()
    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [float_tensor]},
        allow_scalar_tensor_item_conversion=True,
    )
    if (
        results[0]["scalar_tensor_item_conversion_blocked_reason"]
        != "scalar_tensor_item_conversion_item_not_int"
    ):
        raise AssertionError(results[0])

    raising_tensor = _RaisingItemTensorLike()
    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [raising_tensor]},
        allow_scalar_tensor_item_conversion=True,
    )
    if (
        results[0]["scalar_tensor_item_conversion_blocked_reason"]
        != "scalar_tensor_item_conversion_item_failed"
    ):
        raise AssertionError(results[0])


def _assert_poison_forbidden_methods_not_touched() -> None:
    tensor = _PoisonTensorLike()
    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [tensor]},
        allow_scalar_tensor_item_conversion=True,
    )
    if results[0]["read_state"] != "req_to_token_pool_resolved":
        raise AssertionError(results[0])
    if tensor.item_called != 1:
        raise AssertionError(tensor.item_called)
    if tensor.forbidden_touched:
        raise AssertionError("forbidden tensor methods were touched")


def _assert_python_int_and_non_tensor_paths() -> None:
    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 2])],
        req_to_token_pool_object={7: [101, 102]},
        allow_scalar_tensor_item_conversion=True,
    )
    if results[0]["read_state"] != "req_to_token_pool_resolved":
        raise AssertionError(results[0])
    if results[0]["scalar_tensor_item_conversion_attempted"] is not False:
        raise AssertionError(results[0])

    results, _ = _read(
        runtime_observation_payloads=[_runtime_observation_payload(token_span=[0, 1])],
        req_to_token_pool_object={7: [[1, 2, 3]]},
        allow_scalar_tensor_item_conversion=True,
    )
    if results[0]["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(results[0])


def _assert_limits_and_chain() -> None:
    tensor_a = _OneElementIntTensorLike(41)
    tensor_b = _OneElementIntTensorLike(42)
    results, _ = _read(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-a", req_pool_idx=1, token_span=[0, 3]),
            _runtime_observation_payload("req-b", req_pool_idx=2, token_span=[0, 3]),
        ],
        req_to_token_pool_object={
            1: [tensor_a, tensor_a, tensor_a],
            2: [tensor_b, tensor_b, tensor_b],
        },
        allow_scalar_tensor_item_conversion=True,
        max_total_tokens=5,
    )
    if results[1]["blocked_reason"] != "max_total_tokens_exceeded":
        raise AssertionError(results)

    resolved_results, summary = _read(
        runtime_observation_payloads=[
            _runtime_observation_payload("req-chain", req_pool_idx=9, token_span=[0, 3])
        ],
        req_to_token_pool_object={9: [_OneElementIntTensorLike(51), _OneElementIntTensorLike(52), _OneElementIntTensorLike(53)]},
        allow_scalar_tensor_item_conversion=True,
    )
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)
    converted = build_relaykv_req_to_token_resolution_payloads_from_real_pool_read_for_smoke(
        resolved_results
    )
    payload = converted[0]
    if payload["resolution_state"] != "req_to_token_resolved":
        raise AssertionError(payload)
    bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        explicit_payloads=converted,
        bridge_enabled=True,
    )
    bridge_summary = summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        bridge_results
    )
    if bridge_summary["bridged_count"] <= 0:
        raise AssertionError(bridge_summary)
    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        bridge_results[0]["req_to_token_resolution_payloads"],
        token_to_kv_pool_object={51: 301, 52: 302, 53: 303},
        read_token_to_kv_pool_index=True,
        source_path="fake.token_to_kv_pool",
    )
    live_summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        live_results
    )
    if live_summary["physical_kv_index_resolved_count"] <= 0:
        raise AssertionError(live_summary)


def main() -> int:
    _assert_default_off_blocks_and_no_item()
    _assert_zero_dim_tensor_resolves()
    _assert_one_element_tensor_resolves()
    _assert_vector_tensor_blocks_without_item()
    _assert_bool_float_and_raise_block()
    _assert_poison_forbidden_methods_not_touched()
    _assert_python_int_and_non_tensor_paths()
    _assert_limits_and_chain()

    print("relaykv_req_to_token_pool_scalar_tensor_conversion_smoke=pass")
    print(
        json.dumps(
            {
                "bool_float_raise": "blocked",
                "default_off": "unchanged",
                "int_path": "resolved_without_conversion",
                "one_element": "resolved",
                "poison": "item_only_no_forbidden_methods",
                "vector": "blocked_without_item",
                "zero_dim": "resolved",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
