from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke,
    run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke,
)


class FakeTokenToKvPoolTable:
    def __init__(self, shape: tuple[int, int], device: str, dtype: str) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.len_called = False
        self.iter_called = False
        self.getitem_called = False
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False
        self.repr_called = False

    def __deepcopy__(self, memo: dict[int, Any]) -> "FakeTokenToKvPoolTable":
        return self

    def __len__(self) -> int:
        self.len_called = True
        raise AssertionError("__len__() must not be called")

    def __iter__(self):
        self.iter_called = True
        raise AssertionError("__iter__() must not be called")

    def __getitem__(self, index: int) -> None:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

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

    def __repr__(self) -> str:
        self.repr_called = True
        raise AssertionError("__repr__() must not be called")

    @property
    def forbidden_read_called(self) -> bool:
        return any(
            (
                self.len_called,
                self.iter_called,
                self.getitem_called,
                self.cpu_called,
                self.tolist_called,
                self.item_called,
                self.numpy_called,
                self.repr_called,
            )
        )


class _AttrFailureObject:
    @property
    def shape(self) -> Any:
        raise RuntimeError("shape access failed")


class FakeModelRunner:
    def __init__(self, token_to_kv_pool: Any = None) -> None:
        if token_to_kv_pool is not None:
            self.token_to_kv_pool = token_to_kv_pool


class FakeAllocator:
    def __init__(self, token_to_kv_pool: Any = None) -> None:
        if token_to_kv_pool is not None:
            self.token_to_kv_pool = token_to_kv_pool


class FakeAllocatorOnly:
    def __init__(self, shape: tuple[int, int], device: str, dtype: str) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype


class FakeMemoryPool:
    def __init__(self, token_to_kv_pool: Any = None) -> None:
        if token_to_kv_pool is not None:
            self.token_to_kv_pool = token_to_kv_pool


class FakeForwardBatch:
    def __init__(
        self,
        token_to_kv_pool: Any = None,
        *,
        request_id: str = "req-a",
        layer_id: int = 14,
        batch_id: str = "batch-a",
        kv_head_group: str = "kvh-0",
        kv_class: str = "UNKNOWN",
    ) -> None:
        if token_to_kv_pool is not None:
            self.token_to_kv_pool = token_to_kv_pool
        self.request_id = request_id
        self.layer_id = layer_id
        self.batch_id = batch_id
        self.kv_head_group = kv_head_group
        self.kv_class = kv_class


def _assert_zero_safety_counts(summary: dict[str, Any]) -> None:
    for key in (
        "token_to_kv_pool_read_count",
        "actual_token_to_kv_pool_read_count",
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


def _assert_metadata_payload(
    result: dict[str, Any],
    *,
    expected_path: str,
    expected_type: str,
) -> dict[str, Any]:
    payloads = result["payloads"]
    summary = result["summary"]
    if len(payloads) != 1:
        raise AssertionError(payloads)
    payload = payloads[0]
    if payload["inspection_state"] != "metadata_observed":
        raise AssertionError(payload)
    if payload["hook_path"] != expected_path:
        raise AssertionError(payload)
    if (
        payload["hook_source"]
        != "model_runner_token_to_kv_pool_runtime_inspection_hook"
    ):
        raise AssertionError(payload)
    if payload["event_type"] != "relaykv_token_to_kv_pool_runtime_inspection_payload":
        raise AssertionError(payload)
    if payload["inspection_mode"] != "runtime_metadata_only":
        raise AssertionError(payload)
    if payload["engine_name"] != "sglang" or payload["adapter_name"] != "sglang":
        raise AssertionError(payload)
    if payload["position_check_state"] != "not_checked_metadata_only":
        raise AssertionError(payload)
    if payload["attention_mask_mode"] != "unknown":
        raise AssertionError(payload)
    if payload["rope_position_consistency"] != "not_checked":
        raise AssertionError(payload)
    if payload["kv_class"] != "UNKNOWN":
        raise AssertionError(payload)
    if payload["decision_state"] not in {"SHADOW_ONLY", "metadata_observed"}:
        raise AssertionError(payload)
    if payload["fallback_reason"] is not None:
        raise AssertionError(payload)
    if payload["token_to_kv_pool_type"] != expected_type:
        raise AssertionError(payload)
    if payload["token_to_kv_pool_shape"] != (16, 1024):
        raise AssertionError(payload)
    if payload["token_to_kv_pool_device"] != "cuda:0":
        raise AssertionError(payload)
    if payload["token_to_kv_pool_dtype"] != "torch.int32":
        raise AssertionError(payload)
    if payload["adapter_metadata"]["token_to_kv_pool_type"] != expected_type:
        raise AssertionError(payload)
    if payload["engine_block_ref"]["token_to_kv_pool_index"] is not None:
        raise AssertionError(payload)
    if payload["engine_block_ref"]["physical_kv_index"] is not None:
        raise AssertionError(payload)
    if payload["engine_block_ref"]["cache_position"] is not None:
        raise AssertionError(payload)
    if summary["metadata_observed_count"] != 1:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_attr_present_count"] != 1:
        raise AssertionError(summary)
    if summary["actual_token_to_kv_pool_inspection_count"] != 1:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_attr_observed_count"] != 1:
        raise AssertionError(summary)
    _assert_zero_safety_counts(summary)
    return payload


def _assert_direct_object_metadata() -> dict[str, Any]:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    before_state = copy.deepcopy(table.__dict__)
    payloads = build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
        runtime_like={"request_id": "req-direct", "layer_id": 13},
        token_to_kv_pool=table,
        inspect_token_to_kv_pool=True,
    )
    if table.__dict__ != before_state:
        raise AssertionError("direct object mutated")
    payload = payloads[0]
    if payload["inspection_state"] != "metadata_observed":
        raise AssertionError(payload)
    if payload["token_to_kv_pool_type"] != "FakeTokenToKvPoolTable":
        raise AssertionError(payload)
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return payload


def _assert_runner_paths() -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    direct_table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    direct_runner = FakeModelRunner(token_to_kv_pool=direct_table)
    outputs.append(
        _assert_metadata_payload(
            run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
                direct_runner
            ),
            expected_path="model_runner.token_to_kv_pool",
            expected_type="FakeTokenToKvPoolTable",
        )
    )
    if direct_table.forbidden_read_called:
        raise AssertionError("direct runner table values were read")

    nested_table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    nested_runner = FakeModelRunner()
    nested_runner.token_to_kv_pool_allocator = FakeAllocator(
        token_to_kv_pool=nested_table
    )
    outputs.append(
        _assert_metadata_payload(
            run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
                nested_runner
            ),
            expected_path="model_runner.token_to_kv_pool_allocator.token_to_kv_pool",
            expected_type="FakeTokenToKvPoolTable",
        )
    )
    if nested_table.forbidden_read_called:
        raise AssertionError("nested allocator table values were read")

    allocator_only = FakeAllocatorOnly((16, 1024), "cuda:0", "torch.int32")
    allocator_only_runner = FakeModelRunner()
    allocator_only_runner.token_to_kv_pool_allocator = allocator_only
    outputs.append(
        _assert_metadata_payload(
            run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
                allocator_only_runner
            ),
            expected_path="model_runner.token_to_kv_pool_allocator",
            expected_type="FakeAllocatorOnly",
        )
    )

    memory_table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    memory_runner = FakeModelRunner()
    memory_runner.memory_pool = FakeMemoryPool(token_to_kv_pool=memory_table)
    outputs.append(
        _assert_metadata_payload(
            run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
                memory_runner
            ),
            expected_path="model_runner.memory_pool.token_to_kv_pool",
            expected_type="FakeTokenToKvPoolTable",
        )
    )
    if memory_table.forbidden_read_called:
        raise AssertionError("memory pool table values were read")

    kv_alloc_table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    kv_alloc_runner = FakeModelRunner()
    kv_alloc_runner.kv_pool_allocator = FakeAllocator(token_to_kv_pool=kv_alloc_table)
    outputs.append(
        _assert_metadata_payload(
            run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
                kv_alloc_runner
            ),
            expected_path="model_runner.kv_pool_allocator.token_to_kv_pool",
            expected_type="FakeTokenToKvPoolTable",
        )
    )
    if kv_alloc_table.forbidden_read_called:
        raise AssertionError("kv allocator table values were read")

    forward_table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    forward_runner = FakeModelRunner()
    forward_batch = FakeForwardBatch(token_to_kv_pool=forward_table)
    before_runner = copy.deepcopy(forward_runner.__dict__)
    before_forward = copy.deepcopy(forward_batch.__dict__)
    outputs.append(
        _assert_metadata_payload(
            run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
                forward_runner,
                forward_batch=forward_batch,
            ),
            expected_path="forward_batch.token_to_kv_pool",
            expected_type="FakeTokenToKvPoolTable",
        )
    )
    if forward_runner.__dict__ != before_runner:
        raise AssertionError("runner mutated")
    if forward_batch.__dict__ != before_forward:
        raise AssertionError("forward batch mutated")
    if forward_table.forbidden_read_called:
        raise AssertionError("forward batch table values were read")

    return outputs


def _assert_blocked_cases() -> dict[str, Any]:
    disabled_payload = build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
        runtime_like={},
        token_to_kv_pool=FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32"),
        inspect_token_to_kv_pool=False,
    )[0]
    if "inspection_not_enabled" not in disabled_payload["blocking_reasons"]:
        raise AssertionError(disabled_payload)

    missing_result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
        FakeModelRunner()
    )
    missing_payload = missing_result["payloads"][0]
    if "token_to_kv_pool_missing" not in missing_payload["blocking_reasons"]:
        raise AssertionError(missing_payload)
    if missing_payload["hook_path"] is not None:
        raise AssertionError(missing_payload)
    _assert_zero_safety_counts(missing_result["summary"])

    failing_payload = build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
        runtime_like={},
        token_to_kv_pool=_AttrFailureObject(),
        inspect_token_to_kv_pool=True,
    )[0]
    if "token_to_kv_pool_attr_access_failed" not in failing_payload["blocking_reasons"]:
        raise AssertionError(failing_payload)

    blocked_runtime_payload = build_relaykv_token_to_kv_pool_runtime_inspection_payloads_for_smoke(
        runtime_like={
            "token_to_kv_pool_read": True,
            "token_to_kv_pool_index_read": True,
            "kv_pool_read": True,
            "kv_snapshot": True,
            "tensor_read": True,
            "attention_override": True,
            "attention_comparison_executed": True,
            "runtime_writeback": True,
            "scheduler_policy_noop": False,
            "source_mutated": True,
        },
        token_to_kv_pool=FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32"),
        inspect_token_to_kv_pool=True,
    )[0]
    for reason in (
        "token_to_kv_pool_value_read_not_allowed",
        "token_to_kv_pool_index_read_not_allowed",
        "kv_pool_read_not_allowed",
        "kv_snapshot_not_allowed",
        "tensor_read_not_allowed",
        "attention_override_true_not_allowed",
        "attention_comparison_executed_not_allowed",
        "runtime_writeback_not_allowed",
        "scheduler_mutation_not_allowed",
        "source_mutation_not_allowed",
    ):
        if reason not in blocked_runtime_payload["blocking_reasons"]:
            raise AssertionError((reason, blocked_runtime_payload))

    return {
        "disabled": disabled_payload["blocking_reasons"],
        "missing": missing_payload["blocking_reasons"],
        "failing": failing_payload["blocking_reasons"],
        "runtime_blocked": blocked_runtime_payload["blocking_reasons"],
    }


def main() -> None:
    direct = _assert_direct_object_metadata()
    paths = _assert_runner_paths()
    blocked = _assert_blocked_cases()
    print("relaykv_token_to_kv_pool_runtime_inspection_smoke=pass")
    print(
        "relaykv_token_to_kv_pool_runtime_inspection_paths="
        + json.dumps(
            {
                "direct_request_id": direct["request_id"],
                "observed_paths": [payload["hook_path"] for payload in paths],
                "blocked": blocked,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
