from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import sys
import types
from typing import Any

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

from sglang.srt.relaykv.metrics import (
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
)


class PoisonField:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("poison unrelated field must not be accessed")


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


class ScalarIntTensorLike(_BaseTensorLike):
    def __init__(self, value: int) -> None:
        super().__init__(shape=[])
        self._value = value

    def item(self) -> int:
        self.item_called += 1
        return self._value


class VectorTensorLike(_BaseTensorLike):
    ndim = 1

    def __init__(self) -> None:
        super().__init__(shape=(2,))

    def item(self) -> int:
        self.item_called += 1
        raise AssertionError("item() must not be called for vector tensor")


class BoolItemTensorLike(_BaseTensorLike):
    def __init__(self) -> None:
        super().__init__(shape=[])

    def item(self) -> bool:
        self.item_called += 1
        return True


class RaisingItemTensorLike(_BaseTensorLike):
    def __init__(self) -> None:
        super().__init__(shape=[])

    def item(self) -> int:
        self.item_called += 1
        raise RuntimeError("item failed")


class PoisonTensorLike(_BaseTensorLike):
    def __init__(self) -> None:
        super().__init__(shape=[])

    def item(self) -> int:
        self.item_called += 1
        return 41


class FakeReqToTokenPool:
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


class FakeModelOutput:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel
        self.can_run_graph = False
        self.expert_distribution_metrics = None
        self.routed_experts_output = None


class FakeForwardBatch:
    def __init__(
        self,
        *,
        runtime_observation_metadata: Any = None,
        runtime_observation_payloads: Any = None,
        req_to_token_pool: Any = None,
    ) -> None:
        self.request_id = "req-a"
        self.layer_id = 14
        self.batch_id = "batch-a"
        self.kv_head_group = 2
        self.kv_class = "FULL"
        if runtime_observation_metadata is not None:
            self.relaykv_runtime_observation_metadata = runtime_observation_metadata
        if runtime_observation_payloads is not None:
            self.relaykv_runtime_observation_payloads = runtime_observation_payloads
        if req_to_token_pool is not None:
            self.req_to_token_pool = req_to_token_pool


class SlotForwardBatch:
    __slots__ = ("request_id", "layer_id", "batch_id", "kv_head_group", "kv_class")

    def __init__(self) -> None:
        self.request_id = "req-a"
        self.layer_id = 14
        self.batch_id = "batch-a"
        self.kv_head_group = 2
        self.kv_class = "FULL"


class _Recorder:
    def with_forward_pass(self, *args: Any, **kwargs: Any):
        return contextlib.nullcontext({})


class _ExpertsCapturer:
    def on_forward_end(self, **kwargs: Any) -> None:
        return None


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _ImportCounter:
    def __init__(self) -> None:
        self.count = 0
        self.original_import = builtins.__import__

    def __call__(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "sglang.srt.relaykv.metrics":
            self.count += 1
        return self.original_import(name, globals, locals, fromlist, level)


def _runtime_observation_metadata(
    request_id: str = "req-a",
    *,
    req_pool_idx: Any = 7,
    token_span: list[int] | None = None,
    seq_len: int | None = 4,
) -> dict[str, Any]:
    payload = {
        "request_id": request_id,
        "rid": request_id,
        "logical_sequence_id": f"seq-{request_id}",
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": 14,
        "kv_head_group": 2,
        "logical_block_id": 101,
        "adapter_metadata": {"runtime_observation_metadata_marker": request_id},
        "engine_block_ref": {"req_pool_idx": req_pool_idx},
    }
    if token_span is not None:
        payload["token_span"] = token_span
    return payload


def _load_model_runner_module():
    os.environ["HOME"] = "/tmp"
    os.environ["XDG_CACHE_HOME"] = "/tmp"
    from sglang.srt.model_executor import model_runner as model_runner_module

    return model_runner_module


def _install_module_stubs(model_runner_module) -> None:
    model_runner_module.get_global_expert_distribution_recorder = lambda: _Recorder()
    model_runner_module.get_global_experts_capturer = lambda: _ExpertsCapturer()
    model_runner_module.ElasticEPStateManager = types.SimpleNamespace(
        instance=lambda: None
    )
    model_runner_module.dumper = types.SimpleNamespace(may_enable=False, step=lambda: None)


def _make_fake_runner(
    *,
    runtime_observation_metadata: Any = None,
    runtime_observation_payloads: Any = None,
    req_to_token_pool: Any = None,
    sentinel: str = "sentinel-output",
) -> Any:
    runner = types.SimpleNamespace()
    runner.forward_pass_id = 0
    runner.msprobe_debugger = None
    runner.server_args = types.SimpleNamespace(disable_overlap_schedule=False)
    runner.graph_runner = types.SimpleNamespace(bs=None)
    runner.eplb_manager = None
    runner.gpu_id = None
    runner.dp_size = None
    if runtime_observation_metadata is not None:
        runner.relaykv_runtime_observation_metadata = runtime_observation_metadata
    if runtime_observation_payloads is not None:
        runner.relaykv_runtime_observation_payloads = runtime_observation_payloads
    if req_to_token_pool is not None:
        runner.req_to_token_pool = req_to_token_pool
    runner.unrelated = PoisonField()

    def _forward_raw(*args: Any, **kwargs: Any) -> FakeModelOutput:
        return FakeModelOutput(sentinel)

    runner._forward_raw = _forward_raw
    return runner


@contextlib.contextmanager
def _patched_import_counter():
    counter = _ImportCounter()
    previous_import = builtins.__import__
    builtins.__import__ = counter
    try:
        yield counter
    finally:
        builtins.__import__ = previous_import


@contextlib.contextmanager
def _patched_hook_exception():
    metrics_module = sys.modules["sglang.srt.relaykv.metrics"]
    original = metrics_module.run_model_runner_real_req_to_token_pool_bounded_read_hook_for_smoke
    metrics_module.run_model_runner_real_req_to_token_pool_bounded_read_hook_for_smoke = (
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hook failure"))
    )
    try:
        yield
    finally:
        metrics_module.run_model_runner_real_req_to_token_pool_bounded_read_hook_for_smoke = (
            original
        )


def _set_env(key: str, value: str | None) -> str | None:
    previous = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return previous


def _restore_env(key: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def _run_forward(
    runner: Any,
    forward_batch: Any,
    *,
    bounded_env: str | None,
    scalar_env: str | None,
) -> tuple[FakeModelOutput, list[str], int]:
    model_runner_module = _load_model_runner_module()
    _install_module_stubs(model_runner_module)
    logger = model_runner_module.logger
    handler = _ListHandler()
    previous_level = logger.level
    previous_values = {
        "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_BOUNDED_READ": _set_env(
            "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_BOUNDED_READ",
            bounded_env,
        ),
        "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_SCALAR_ITEM_CONVERSION": _set_env(
            "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_SCALAR_ITEM_CONVERSION",
            scalar_env,
        ),
        "SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ": _set_env(
            "SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ",
            None,
        ),
        "SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE": _set_env(
            "SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE",
            None,
        ),
    }
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    with _patched_import_counter() as counter:
        try:
            output = model_runner_module.ModelRunner.forward(runner, forward_batch)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            for key, previous in previous_values.items():
                _restore_env(key, previous)
    return output, handler.messages, counter.count


def _extract_summary(messages: list[str]) -> dict[str, Any]:
    prefix = "relaykv_real_req_to_token_pool_bounded_read_summary="
    for message in messages:
        if message.startswith(prefix):
            return json.loads(message[len(prefix) :])
    raise AssertionError(messages)


def _assert_zero_forbidden_safety_counts(summary: dict[str, Any]) -> None:
    for key in (
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
        if summary[key] != 0:
            raise AssertionError((key, summary[key]))


def _assert_all_env_off() -> None:
    tensor = ScalarIntTensorLike(21)
    pool = FakeReqToTokenPool({7: [tensor]})
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
        req_to_token_pool=pool,
    )
    output, messages, import_count = _run_forward(
        runner,
        FakeForwardBatch(
            runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
            req_to_token_pool=pool,
        ),
        bounded_env="0",
        scalar_env=None,
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count != 0:
        raise AssertionError(import_count)
    if any("relaykv_real_req_to_token_pool_bounded_read_summary=" in msg for msg in messages):
        raise AssertionError(messages)
    if tensor.item_called != 0:
        raise AssertionError(tensor.item_called)


def _assert_bounded_on_scalar_off_blocks() -> None:
    tensor = ScalarIntTensorLike(22)
    pool = FakeReqToTokenPool({7: [tensor]})
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
        req_to_token_pool=pool,
    )
    output, messages, _ = _run_forward(
        runner,
        SlotForwardBatch(),
        bounded_env="1",
        scalar_env=None,
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    _assert_zero_forbidden_safety_counts(summary)
    if summary["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(summary)
    if summary["scalar_tensor_item_conversion_enabled"] is not False:
        raise AssertionError(summary)
    if tensor.item_called != 0:
        raise AssertionError(tensor.item_called)
    if summary["req_to_token_payload_attached"] is not False:
        raise AssertionError(summary)


def _assert_both_flags_on_resolve() -> None:
    tensor = ScalarIntTensorLike(31)
    pool = FakeReqToTokenPool({7: [tensor]})
    metadata = [_runtime_observation_metadata(token_span=[0, 1])]
    runner = _make_fake_runner(
        runtime_observation_metadata=metadata,
        req_to_token_pool=pool,
    )
    output, messages, _ = _run_forward(
        runner,
        SlotForwardBatch(),
        bounded_env="1",
        scalar_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    _assert_zero_forbidden_safety_counts(summary)
    if tensor.item_called != 1:
        raise AssertionError(tensor.item_called)
    if summary["resolved_count"] <= 0:
        raise AssertionError(summary)
    if summary["scalar_tensor_item_conversion_enabled"] is not True:
        raise AssertionError(summary)
    if summary["scalar_tensor_item_conversion_succeeded_count"] <= 0:
        raise AssertionError(summary)
    if summary["req_to_token_payload_attached"] is not True:
        raise AssertionError(summary)
    if summary["payload_count"] <= 0:
        raise AssertionError(summary)
    attached = runner.relaykv_req_to_token_resolution_payloads
    bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        explicit_payloads=attached,
        bridge_enabled=True,
    )
    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        bridge_results[0]["req_to_token_resolution_payloads"],
        token_to_kv_pool_object={31: 301},
        read_token_to_kv_pool_index=True,
        source_path="fake.token_to_kv_pool",
    )
    live_summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        live_results
    )
    if live_summary["physical_kv_index_resolved_count"] <= 0:
        raise AssertionError(live_summary)


def _assert_scalar_on_bounded_off_does_nothing() -> None:
    tensor = ScalarIntTensorLike(23)
    pool = FakeReqToTokenPool({7: [tensor]})
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
        req_to_token_pool=pool,
    )
    output, messages, import_count = _run_forward(
        runner,
        SlotForwardBatch(),
        bounded_env=None,
        scalar_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count != 0:
        raise AssertionError(import_count)
    if messages:
        for msg in messages:
            if "relaykv_real_req_to_token_pool_bounded_read_summary=" in msg:
                raise AssertionError(messages)
    if tensor.item_called != 0:
        raise AssertionError(tensor.item_called)


def _assert_vector_bool_raise_and_poison() -> None:
    vector = VectorTensorLike()
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
        req_to_token_pool=FakeReqToTokenPool({7: [vector]}),
    )
    output, messages, _ = _run_forward(
        runner,
        SlotForwardBatch(),
        bounded_env="1",
        scalar_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(summary)
    if vector.item_called != 0:
        raise AssertionError(vector.item_called)
    if summary["req_to_token_payload_attached"] is not False:
        raise AssertionError(summary)

    bool_tensor = BoolItemTensorLike()
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
        req_to_token_pool=FakeReqToTokenPool({7: [bool_tensor]}),
    )
    _, messages, _ = _run_forward(
        runner,
        SlotForwardBatch(),
        bounded_env="1",
        scalar_env="1",
    )
    summary = _extract_summary(messages)
    if summary["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(summary)
    if summary["req_to_token_payload_attached"] is not False:
        raise AssertionError(summary)

    raising_tensor = RaisingItemTensorLike()
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
        req_to_token_pool=FakeReqToTokenPool({7: [raising_tensor]}),
    )
    output, messages, _ = _run_forward(
        runner,
        SlotForwardBatch(),
        bounded_env="1",
        scalar_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_reason"] != "req_to_token_pool_value_not_int":
        raise AssertionError(summary)

    poison = PoisonTensorLike()
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 1])],
        req_to_token_pool=FakeReqToTokenPool({7: [poison]}),
    )
    output, messages, _ = _run_forward(
        runner,
        SlotForwardBatch(),
        bounded_env="1",
        scalar_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["resolved_count"] <= 0:
        raise AssertionError(summary)
    if poison.item_called != 1:
        raise AssertionError(poison.item_called)
    if poison.forbidden_touched:
        raise AssertionError("forbidden methods touched")


def _assert_hook_exception_swallowed() -> None:
    runner = _make_fake_runner()
    model_runner_module = _load_model_runner_module()
    _install_module_stubs(model_runner_module)
    logger = model_runner_module.logger
    handler = _ListHandler()
    previous_level = logger.level
    previous_envs = {
        "bounded": _set_env("SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_BOUNDED_READ", "1"),
        "scalar": _set_env(
            "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_SCALAR_ITEM_CONVERSION", "1"
        ),
    }
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with _patched_hook_exception():
            output = model_runner_module.ModelRunner.forward(runner, SlotForwardBatch())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        _restore_env(
            "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_BOUNDED_READ",
            previous_envs["bounded"],
        )
        _restore_env(
            "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_SCALAR_ITEM_CONVERSION",
            previous_envs["scalar"],
        )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)


def main() -> int:
    _assert_all_env_off()
    _assert_bounded_on_scalar_off_blocks()
    _assert_both_flags_on_resolve()
    _assert_scalar_on_bounded_off_does_nothing()
    _assert_vector_bool_raise_and_poison()
    _assert_hook_exception_swallowed()

    print("relaykv_fake_model_runner_real_req_to_token_pool_scalar_conversion_smoke=pass")
    print(
        json.dumps(
            {
                "all_off": "no_hook_no_log_no_attr_write",
                "bounded_on_scalar_off": "blocked_item_not_called",
                "bounded_on_scalar_on": "resolved_attached_live_compatible",
                "poison": "item_only_no_forbidden_methods",
                "scalar_on_bounded_off": "no_hook_no_item",
                "vector_bool_raise": "blocked_forward_unchanged",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
