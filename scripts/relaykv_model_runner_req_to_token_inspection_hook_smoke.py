from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    run_model_runner_req_to_token_runtime_inspection_hook_for_smoke,
)


class FakeReqToTokenTable:
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

    def __deepcopy__(self, memo: dict[int, Any]) -> "FakeReqToTokenTable":
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

    @property
    def forbidden_read_called(self) -> bool:
        return (
            self.len_called
            or self.iter_called
            or self.getitem_called
            or self.cpu_called
            or self.tolist_called
            or self.item_called
            or self.numpy_called
        )


class FakeReqToTokenPool:
    def __init__(self, req_to_token: Any) -> None:
        self._req_to_token = req_to_token
        self.req_to_token_access_count = 0

    @property
    def req_to_token(self) -> Any:
        self.req_to_token_access_count += 1
        return self._req_to_token


class FakeModelRunner:
    def __init__(self, req_to_token_pool: Any = None) -> None:
        self.req_to_token_pool = req_to_token_pool


class FakeAllocator:
    def __init__(self, req_to_token_pool: Any = None) -> None:
        self.req_to_token_pool = req_to_token_pool


class FakeModelRunnerWithAllocator:
    def __init__(self, allocator: Any) -> None:
        self.token_to_kv_pool_allocator = allocator


class FakeForwardBatch:
    def __init__(
        self,
        req_to_token_pool: Any = None,
        *,
        request_id: str = "req-a",
        layer_id: int = 14,
        batch_id: str = "hook-batch-a",
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.request_id = request_id
        self.layer_id = layer_id
        self.batch_id = batch_id


def _assert_zero_safety_counts(summary: dict[str, Any]) -> None:
    for key in (
        "req_to_token_read_count",
        "actual_req_to_token_pool_read_count",
        "token_to_kv_pool_read_count",
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
            raise AssertionError(summary)


def _assert_metadata_payload(
    result: dict[str, Any],
    *,
    expected_path: str,
    expected_type: str = "FakeReqToTokenTable",
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
        != "model_runner_req_to_token_runtime_inspection_hook"
    ):
        raise AssertionError(payload)
    if payload["req_to_token_type"] != expected_type:
        raise AssertionError(payload)
    if payload["req_to_token_shape"] != (16, 1024):
        raise AssertionError(payload)
    if payload["req_to_token_device"] != "cuda:0":
        raise AssertionError(payload)
    if payload["req_to_token_dtype"] != "torch.int32":
        raise AssertionError(payload)
    if summary["metadata_observed_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_attr_present_count"] != 1:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_inspection_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_attr_observed_count"] != 1:
        raise AssertionError(summary)
    _assert_zero_safety_counts(summary)
    return payload


def _assert_direct_runner_path() -> dict[str, Any]:
    table = FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
    pool = FakeReqToTokenPool(table)
    runner = FakeModelRunner(req_to_token_pool=pool)
    result = run_model_runner_req_to_token_runtime_inspection_hook_for_smoke(runner)
    payload = _assert_metadata_payload(
        result,
        expected_path="model_runner.req_to_token_pool",
    )
    if pool.req_to_token_access_count != 1:
        raise AssertionError(pool.req_to_token_access_count)
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return payload


def _assert_nested_allocator_path() -> dict[str, Any]:
    table = FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
    pool = FakeReqToTokenPool(table)
    runner = FakeModelRunnerWithAllocator(FakeAllocator(req_to_token_pool=pool))
    result = run_model_runner_req_to_token_runtime_inspection_hook_for_smoke(runner)
    payload = _assert_metadata_payload(
        result,
        expected_path=(
            "model_runner.token_to_kv_pool_allocator.req_to_token_pool"
        ),
    )
    if pool.req_to_token_access_count != 1:
        raise AssertionError(pool.req_to_token_access_count)
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return payload


def _assert_forward_batch_path() -> dict[str, Any]:
    table = FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
    pool = FakeReqToTokenPool(table)
    runner = FakeModelRunner()
    forward_batch = FakeForwardBatch(req_to_token_pool=pool)
    before_runner = copy.deepcopy(runner.__dict__)
    before_forward_batch_fields = {
        "request_id": forward_batch.request_id,
        "layer_id": forward_batch.layer_id,
        "batch_id": forward_batch.batch_id,
        "req_to_token_pool_id": id(forward_batch.req_to_token_pool),
    }
    result = run_model_runner_req_to_token_runtime_inspection_hook_for_smoke(
        runner,
        forward_batch=forward_batch,
    )
    if runner.__dict__ != before_runner:
        raise AssertionError("runner mutated")
    after_forward_batch_fields = {
        "request_id": forward_batch.request_id,
        "layer_id": forward_batch.layer_id,
        "batch_id": forward_batch.batch_id,
        "req_to_token_pool_id": id(forward_batch.req_to_token_pool),
    }
    if after_forward_batch_fields != before_forward_batch_fields:
        raise AssertionError("forward batch mutated")
    payload = _assert_metadata_payload(
        result,
        expected_path="forward_batch.req_to_token_pool",
    )
    if pool.req_to_token_access_count != 1:
        raise AssertionError(pool.req_to_token_access_count)
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    return payload


def _assert_missing_pool_blocked() -> dict[str, Any]:
    result = run_model_runner_req_to_token_runtime_inspection_hook_for_smoke(
        FakeModelRunner()
    )
    payload = result["payloads"][0]
    summary = result["summary"]
    if "req_to_token_pool_missing" not in payload["blocking_reasons"]:
        raise AssertionError(payload)
    if payload["hook_path"] is not None:
        raise AssertionError(payload)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["metadata_observed_count"] != 0:
        raise AssertionError(summary)
    _assert_zero_safety_counts(summary)
    return payload


def main() -> None:
    direct = _assert_direct_runner_path()
    nested = _assert_nested_allocator_path()
    forward_batch = _assert_forward_batch_path()
    missing = _assert_missing_pool_blocked()
    print(
        json.dumps(
            {
                "direct_path": direct["hook_path"],
                "nested_path": nested["hook_path"],
                "forward_batch_path": forward_batch["hook_path"],
                "missing_blocking_reasons": missing["blocking_reasons"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
