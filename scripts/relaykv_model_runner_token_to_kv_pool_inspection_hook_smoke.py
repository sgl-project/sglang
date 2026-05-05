from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
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


class PoisonField:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("poison unrelated field must not be accessed")


class FakeModelRunner:
    pass


class FakeAllocator:
    pass


class FakeMemoryPool:
    pass


class FakeForwardBatch:
    def __init__(
        self,
        *,
        token_to_kv_pool: Any | None = None,
        request_id: str = "req-a",
        layer_id: int = 14,
        batch_id: str = "batch-a",
    ) -> None:
        if token_to_kv_pool is not None:
            self.token_to_kv_pool = token_to_kv_pool
        self.request_id = request_id
        self.layer_id = layer_id
        self.batch_id = batch_id


class FailingPropertyRunner:
    @property
    def token_to_kv_pool(self) -> Any:
        raise RuntimeError("direct attr access failed")


class FailingAllocator:
    @property
    def token_to_kv_pool(self) -> Any:
        raise RuntimeError("nested attr access failed")


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


def _assert_metadata_result(
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
    if payload["adapter_metadata"]["token_to_kv_pool_source_path"] != expected_path:
        raise AssertionError(payload)
    if payload["token_to_kv_pool_type"] != expected_type:
        raise AssertionError(payload)
    if payload["engine_name"] != "sglang" or payload["adapter_name"] != "sglang":
        raise AssertionError(payload)
    if payload["decision_state"] != "SHADOW_ONLY":
        raise AssertionError(payload)
    if payload["position_check_state"] != "not_checked_metadata_only":
        raise AssertionError(payload)
    if payload["attention_mask_mode"] != "unknown":
        raise AssertionError(payload)
    if payload["rope_position_consistency"] != "not_checked":
        raise AssertionError(payload)
    if payload["engine_block_ref"]["token_to_kv_pool_index"] is not None:
        raise AssertionError(payload)
    if payload["engine_block_ref"]["physical_kv_index"] is not None:
        raise AssertionError(payload)
    if payload["engine_block_ref"]["cache_position"] is not None:
        raise AssertionError(payload)
    if summary["metadata_observed_count"] != 1:
        raise AssertionError(summary)
    _assert_zero_safety_counts(summary)
    return payload


def _assert_direct_path() -> str:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    runner = FakeModelRunner()
    runner.token_to_kv_pool = table
    runner.unrelated = PoisonField()
    before_runner = {
        "keys": tuple(sorted(runner.__dict__.keys())),
        "token_to_kv_pool_id": id(runner.__dict__["token_to_kv_pool"]),
        "unrelated_id": id(runner.__dict__["unrelated"]),
    }
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(runner)
    after_runner = {
        "keys": tuple(sorted(runner.__dict__.keys())),
        "token_to_kv_pool_id": id(runner.__dict__["token_to_kv_pool"]),
        "unrelated_id": id(runner.__dict__["unrelated"]),
    }
    if after_runner != before_runner:
        raise AssertionError("runner mutated")
    _assert_metadata_result(
        result,
        expected_path="model_runner.token_to_kv_pool",
        expected_type="FakeTokenToKvPoolTable",
    )
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return result["token_to_kv_pool_path"]


def _assert_nested_allocator_path() -> str:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    allocator = FakeAllocator()
    allocator.token_to_kv_pool = table
    allocator.unrelated = PoisonField()
    runner = FakeModelRunner()
    runner.token_to_kv_pool_allocator = allocator
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(runner)
    _assert_metadata_result(
        result,
        expected_path="model_runner.token_to_kv_pool_allocator.token_to_kv_pool",
        expected_type="FakeTokenToKvPoolTable",
    )
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return result["token_to_kv_pool_path"]


def _assert_allocator_object_fallback_path() -> str:
    allocator = FakeAllocator()
    allocator.shape = (16, 1024)
    allocator.device = "cuda:0"
    allocator.dtype = "torch.int32"
    allocator.unrelated = PoisonField()
    runner = FakeModelRunner()
    runner.token_to_kv_pool_allocator = allocator
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(runner)
    _assert_metadata_result(
        result,
        expected_path="model_runner.token_to_kv_pool_allocator",
        expected_type="FakeAllocator",
    )
    return result["token_to_kv_pool_path"]


def _assert_kv_pool_allocator_path() -> str:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    allocator = FakeAllocator()
    allocator.token_to_kv_pool = table
    allocator.unrelated = PoisonField()
    runner = FakeModelRunner()
    runner.kv_pool_allocator = allocator
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(runner)
    _assert_metadata_result(
        result,
        expected_path="model_runner.kv_pool_allocator.token_to_kv_pool",
        expected_type="FakeTokenToKvPoolTable",
    )
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return result["token_to_kv_pool_path"]


def _assert_memory_pool_path() -> str:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    memory_pool = FakeMemoryPool()
    memory_pool.token_to_kv_pool = table
    memory_pool.unrelated = PoisonField()
    runner = FakeModelRunner()
    runner.memory_pool = memory_pool
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(runner)
    _assert_metadata_result(
        result,
        expected_path="model_runner.memory_pool.token_to_kv_pool",
        expected_type="FakeTokenToKvPoolTable",
    )
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return result["token_to_kv_pool_path"]


def _assert_forward_batch_path() -> str:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    runner = FakeModelRunner()
    forward_batch = FakeForwardBatch(token_to_kv_pool=table)
    before_runner = {"keys": tuple(sorted(runner.__dict__.keys()))}
    before_batch = {
        "keys": tuple(sorted(forward_batch.__dict__.keys())),
        "token_to_kv_pool_id": id(forward_batch.__dict__["token_to_kv_pool"]),
        "request_id": forward_batch.request_id,
        "layer_id": forward_batch.layer_id,
        "batch_id": forward_batch.batch_id,
    }
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
        runner,
        forward_batch=forward_batch,
    )
    after_runner = {"keys": tuple(sorted(runner.__dict__.keys()))}
    after_batch = {
        "keys": tuple(sorted(forward_batch.__dict__.keys())),
        "token_to_kv_pool_id": id(forward_batch.__dict__["token_to_kv_pool"]),
        "request_id": forward_batch.request_id,
        "layer_id": forward_batch.layer_id,
        "batch_id": forward_batch.batch_id,
    }
    if after_runner != before_runner:
        raise AssertionError("runner mutated")
    if after_batch != before_batch:
        raise AssertionError("forward batch mutated")
    _assert_metadata_result(
        result,
        expected_path="forward_batch.token_to_kv_pool",
        expected_type="FakeTokenToKvPoolTable",
    )
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return result["token_to_kv_pool_path"]


def _assert_precedence_order() -> str:
    direct = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    nested = FakeTokenToKvPoolTable((16, 1024), "cuda:1", "torch.int64")
    memory = FakeTokenToKvPoolTable((16, 1024), "cuda:2", "torch.int16")
    runner = FakeModelRunner()
    runner.token_to_kv_pool = direct
    runner.token_to_kv_pool_allocator = FakeAllocator()
    runner.token_to_kv_pool_allocator.token_to_kv_pool = nested
    runner.memory_pool = FakeMemoryPool()
    runner.memory_pool.token_to_kv_pool = memory
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(runner)
    payload = _assert_metadata_result(
        result,
        expected_path="model_runner.token_to_kv_pool",
        expected_type="FakeTokenToKvPoolTable",
    )
    if payload["token_to_kv_pool_device"] != "cuda:0":
        raise AssertionError(payload)
    if direct.forbidden_read_called or nested.forbidden_read_called or memory.forbidden_read_called:
        raise AssertionError("values were read")
    return result["token_to_kv_pool_path"]


def _assert_missing_blocked() -> list[str]:
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
        FakeModelRunner()
    )
    payload = result["payloads"][0]
    if payload["inspection_state"] != "blocked":
        raise AssertionError(payload)
    if payload["blocking_reasons"] != ["token_to_kv_pool_missing"]:
        raise AssertionError(payload)
    if payload["adapter_metadata"]["token_to_kv_pool_source_path"] is not None:
        raise AssertionError(payload)
    _assert_zero_safety_counts(result["summary"])
    return payload["blocking_reasons"]


def _assert_attr_failure_blocked() -> list[str]:
    result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
        FailingPropertyRunner()
    )
    payload = result["payloads"][0]
    if payload["inspection_state"] != "blocked":
        raise AssertionError(payload)
    if payload["blocking_reasons"] != ["token_to_kv_pool_attr_access_failed"]:
        raise AssertionError(payload)
    if payload["adapter_metadata"]["token_to_kv_pool_source_path"] != (
        "model_runner.token_to_kv_pool"
    ):
        raise AssertionError(payload)
    _assert_zero_safety_counts(result["summary"])

    runner = FakeModelRunner()
    runner.token_to_kv_pool_allocator = FailingAllocator()
    nested_result = run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke(
        runner
    )
    nested_payload = nested_result["payloads"][0]
    if nested_payload["blocking_reasons"] != ["token_to_kv_pool_attr_access_failed"]:
        raise AssertionError(nested_payload)
    if nested_payload["adapter_metadata"]["token_to_kv_pool_source_path"] != (
        "model_runner.token_to_kv_pool_allocator.token_to_kv_pool"
    ):
        raise AssertionError(nested_payload)
    _assert_zero_safety_counts(nested_result["summary"])
    return nested_payload["blocking_reasons"]


def main() -> None:
    observed_paths = [
        _assert_direct_path(),
        _assert_nested_allocator_path(),
        _assert_allocator_object_fallback_path(),
        _assert_kv_pool_allocator_path(),
        _assert_memory_pool_path(),
        _assert_forward_batch_path(),
        _assert_precedence_order(),
    ]
    blocked = {
        "missing": _assert_missing_blocked(),
        "attr_failure": _assert_attr_failure_blocked(),
    }
    print("relaykv_model_runner_token_to_kv_pool_inspection_hook_smoke=pass")
    print(
        "relaykv_model_runner_token_to_kv_pool_inspection_hook_paths="
        + json.dumps(
            {
                "observed_paths": observed_paths,
                "blocked": blocked,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
