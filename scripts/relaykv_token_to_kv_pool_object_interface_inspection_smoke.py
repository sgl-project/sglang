from __future__ import annotations

import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_token_to_kv_pool_object_interface_inspection_results_for_smoke,
    run_model_runner_token_to_kv_pool_object_interface_inspection_hook_for_smoke,
    summarize_relaykv_token_to_kv_pool_object_interface_inspection_results_for_smoke,
)


class _TensorLike:
    def __init__(self, *, shape: Any = (), dtype: Any = "torch.int32", device: Any = "cuda:0") -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.item_called = False
        self.cpu_called = False
        self.tolist_called = False
        self.numpy_called = False
        self.repr_called = False

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def numpy(self) -> None:
        self.numpy_called = True
        raise AssertionError("numpy() must not be called")

    def __repr__(self) -> str:
        self.repr_called = True
        raise AssertionError("__repr__() must not be called")


class _AllocatorLike:
    def __init__(self, nested: Any) -> None:
        self.token_to_kv_pool = nested


class _AttrContainer:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _LenRaises:
    def __len__(self) -> int:
        raise TypeError("len not supported")


class _Poison:
    def __init__(self) -> None:
        self.repr_called = False
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False

    def __getattr__(self, name: str) -> Any:
        if name in {"shape", "dtype", "device", "ndim", "size"}:
            raise AssertionError(f"unexpected attr access: {name}")
        raise AttributeError(name)

    def __repr__(self) -> str:
        self.repr_called = True
        raise AssertionError("__repr__() must not be called")

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


class _MethodPresenceOnly:
    def get(self) -> None:
        raise AssertionError("methods must not be called")

    def lookup(self) -> None:
        raise AssertionError("methods must not be called")

    def read(self) -> None:
        raise AssertionError("methods must not be called")


class _FakeModelRunner:
    pass


class _FakeForwardBatch:
    pass


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


def _inspect(
    *,
    token_to_kv_pool_object: Any,
    enabled: bool,
    source_path: str | None = "model_runner.token_to_kv_pool",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = build_relaykv_token_to_kv_pool_object_interface_inspection_results_for_smoke(
        token_to_kv_pool_object=token_to_kv_pool_object,
        inspect_token_to_kv_pool_object_interface=enabled,
        source_path=source_path,
    )
    summary = summarize_relaykv_token_to_kv_pool_object_interface_inspection_results_for_smoke(
        results
    )
    _assert_zero_safety(summary)
    return results, summary


def main() -> int:
    disabled_results, disabled_summary = _inspect(
        token_to_kv_pool_object={"a": 1},
        enabled=False,
    )
    if disabled_summary["blocked_count"] != 1:
        raise AssertionError(disabled_summary)
    if disabled_results[0]["blocked_reason"] != (
        "token_to_kv_pool_object_interface_inspection_not_enabled"
    ):
        raise AssertionError(disabled_results[0])

    missing_results, missing_summary = _inspect(
        token_to_kv_pool_object=None,
        enabled=True,
    )
    if missing_summary["blocked_count"] != 1:
        raise AssertionError(missing_summary)
    if missing_results[0]["blocked_reason"] != "token_to_kv_pool_object_missing":
        raise AssertionError(missing_results[0])

    simple_results, simple_summary = _inspect(
        token_to_kv_pool_object={1: 10, 2: 11},
        enabled=True,
    )
    if simple_summary["inspected_count"] != 1:
        raise AssertionError(simple_summary)
    if simple_results[0]["object_has_getitem"] is not True:
        raise AssertionError(simple_results[0])

    tensor_like = _TensorLike(shape=(), dtype="torch.int32", device="cuda:0")
    tensor_results, tensor_summary = _inspect(
        token_to_kv_pool_object=tensor_like,
        enabled=True,
    )
    if tensor_summary["observed_shape_counts"].get("()") != 1:
        raise AssertionError(tensor_summary)
    if tensor_summary["observed_dtype_counts"].get("torch.int32") != 1:
        raise AssertionError(tensor_summary)
    if tensor_summary["observed_device_counts"].get("cuda:0") != 1:
        raise AssertionError(tensor_summary)
    if any(
        (
            tensor_like.item_called,
            tensor_like.cpu_called,
            tensor_like.tolist_called,
            tensor_like.numpy_called,
            tensor_like.repr_called,
        )
    ):
        raise AssertionError("forbidden tensor-like method called")

    nested_tensor = _TensorLike(shape=(8,), dtype="torch.int64", device="cuda:1")
    allocator_results, allocator_summary = _inspect(
        token_to_kv_pool_object=_AllocatorLike(nested_tensor),
        enabled=True,
    )
    if allocator_summary["known_attr_presence_counts"].get("token_to_kv_pool") != 1:
        raise AssertionError(allocator_summary)
    if "model_runner.token_to_kv_pool.token_to_kv_pool" not in allocator_results[0]["candidate_next_source_paths"]:
        raise AssertionError(allocator_results[0])

    data_container = _AttrContainer(
        indices=[1, 2, 3],
        data=_TensorLike(shape=(4,), dtype="torch.int16", device="cuda:2"),
        storage={"a": 1},
        tensor=_TensorLike(shape=(2,), dtype="torch.int8", device="cuda:3"),
    )
    data_results, data_summary = _inspect(
        token_to_kv_pool_object=data_container,
        enabled=True,
    )
    for expected in (
        "indices",
        "data",
        "storage",
        "tensor",
    ):
        if expected not in data_summary["candidate_indexable_attr_names"] and expected not in data_summary["candidate_tensor_like_attr_names"]:
            raise AssertionError((expected, data_summary))
    expected_paths = {
        "model_runner.token_to_kv_pool.indices",
        "model_runner.token_to_kv_pool.data",
        "model_runner.token_to_kv_pool.storage",
        "model_runner.token_to_kv_pool.tensor",
    }
    if not expected_paths.issubset(set(data_summary["candidate_next_source_paths"])):
        raise AssertionError(data_summary)

    poison = _Poison()
    poison_results, poison_summary = _inspect(
        token_to_kv_pool_object=poison,
        enabled=True,
    )
    if poison_summary["inspected_count"] != 1:
        raise AssertionError(poison_summary)
    if any(
        (
            poison.repr_called,
            poison.cpu_called,
            poison.tolist_called,
            poison.item_called,
            poison.numpy_called,
        )
    ):
        raise AssertionError("poison object was touched via forbidden method")
    if poison_results[0]["known_attr_summaries"] != []:
        raise AssertionError(poison_results[0])

    method_results, method_summary = _inspect(
        token_to_kv_pool_object=_MethodPresenceOnly(),
        enabled=True,
    )
    for expected_method in ("get", "lookup", "read"):
        if method_summary["known_method_presence_counts"].get(expected_method) != 1:
            raise AssertionError(method_summary)

    len_results, _ = _inspect(
        token_to_kv_pool_object=_LenRaises(),
        enabled=True,
    )
    if len_results[0]["object_has_len"] is not True:
        raise AssertionError(len_results[0])
    if len_results[0]["object_len"] is not None:
        raise AssertionError(len_results[0])

    wrapper_runner = _FakeModelRunner()
    wrapper_runner.token_to_kv_pool = _AllocatorLike(_TensorLike(shape=(1,), dtype="torch.int32"))
    wrapper_forward = _FakeForwardBatch()
    wrapper_result = run_model_runner_token_to_kv_pool_object_interface_inspection_hook_for_smoke(
        wrapper_runner,
        wrapper_forward,
    )
    wrapper_summary = wrapper_result["summary"]
    _assert_zero_safety(wrapper_summary)
    if wrapper_summary["inspected_count"] != 1:
        raise AssertionError(wrapper_summary)
    if wrapper_result["token_to_kv_pool_path"] != "model_runner.token_to_kv_pool":
        raise AssertionError(wrapper_result)
    if "model_runner.token_to_kv_pool.token_to_kv_pool" not in wrapper_summary["candidate_next_source_paths"]:
        raise AssertionError(wrapper_summary)

    output = {
        "disabled": disabled_results[0]["blocked_reason"],
        "missing": missing_results[0]["blocked_reason"],
        "simple_type": simple_results[0]["object_type_name"],
        "tensor_shape_counts": tensor_summary["observed_shape_counts"],
        "allocator_paths": allocator_results[0]["candidate_next_source_paths"],
        "data_candidate_paths": data_summary["candidate_next_source_paths"],
        "method_presence": method_summary["known_method_presence_counts"],
        "wrapper_path": wrapper_result["token_to_kv_pool_path"],
    }
    print("relaykv_token_to_kv_pool_object_interface_inspection_smoke=pass")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
