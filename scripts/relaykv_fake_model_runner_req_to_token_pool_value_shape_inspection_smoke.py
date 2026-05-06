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


class PoisonField:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("poison unrelated field must not be accessed")


class PoisonReadValue:
    def __init__(self) -> None:
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False
        self.iter_called = False
        self.len_called = False
        self.repr_called = False

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
                self.tolist_called,
                self.item_called,
                self.numpy_called,
                self.iter_called,
                self.len_called,
                self.repr_called,
            )
        )


class PoisonInspectionValue:
    def __init__(self) -> None:
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False
        self.repr_called = False

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
    def touched(self) -> bool:
        return any(
            (
                self.cpu_called,
                self.tolist_called,
                self.item_called,
                self.numpy_called,
                self.repr_called,
            )
        )


class ScalarLikeValue:
    shape = ()
    dtype = "bf16"
    device = "cuda:0"
    ndim = 0


class OneElementTensorLikeValue:
    shape = (1,)
    dtype = "int32"
    device = "cuda:1"
    ndim = 1


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


class ErrorReqToTokenPool:
    def __getitem__(self, req_pool_idx: Any) -> Any:
        raise RuntimeError("pool read failed")


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
    original = (
        metrics_module.run_model_runner_req_to_token_pool_value_shape_inspection_hook_for_smoke
    )
    metrics_module.run_model_runner_req_to_token_pool_value_shape_inspection_hook_for_smoke = (
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hook failure"))
    )
    try:
        yield
    finally:
        metrics_module.run_model_runner_req_to_token_pool_value_shape_inspection_hook_for_smoke = (
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
    env_value: str | None,
) -> tuple[FakeModelOutput, list[str], int]:
    model_runner_module = _load_model_runner_module()
    _install_module_stubs(model_runner_module)
    logger = model_runner_module.logger
    handler = _ListHandler()
    previous_level = logger.level
    previous_values = {
        "SGLANG_RELAYKV_REQ_TO_TOKEN_POOL_VALUE_SHAPE_INSPECTION": _set_env(
            "SGLANG_RELAYKV_REQ_TO_TOKEN_POOL_VALUE_SHAPE_INSPECTION",
            env_value,
        ),
        "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_BOUNDED_READ": _set_env(
            "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_BOUNDED_READ",
            None,
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
    prefix = "relaykv_req_to_token_pool_value_shape_inspection_summary="
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


def _assert_env_off_no_hook() -> None:
    pool = FakeReqToTokenPool({7: [21, 22, 23]})
    runner = _make_fake_runner(
        runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 3])],
        req_to_token_pool=pool,
    )
    output, messages, import_count = _run_forward(
        runner,
        FakeForwardBatch(
            runtime_observation_metadata=[_runtime_observation_metadata(token_span=[0, 3])],
            req_to_token_pool=pool,
        ),
        env_value="0",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count != 0:
        raise AssertionError(import_count)
    if any(
        "relaykv_req_to_token_pool_value_shape_inspection_summary=" in msg
        for msg in messages
    ):
        raise AssertionError(messages)
    if hasattr(runner, "relaykv_req_to_token_resolution_payloads"):
        raise AssertionError("runner attr write should not happen")


def _assert_missing_observation_blocked() -> None:
    pool = FakeReqToTokenPool({7: [21, 22, 23]})
    runner = _make_fake_runner(req_to_token_pool=pool)
    forward_batch = FakeForwardBatch(req_to_token_pool=pool)
    output, messages, _ = _run_forward(runner, forward_batch, env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    _assert_zero_forbidden_safety_counts(summary)
    if summary["blocked_reason"] != "runtime_observation_payloads_missing":
        raise AssertionError(summary)
    if summary["req_to_token_read_count"] != 0:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_read_count"] != 0:
        raise AssertionError(summary)


def _assert_missing_pool_blocked() -> None:
    metadata = [_runtime_observation_metadata(token_span=[0, 3])]
    runner = _make_fake_runner(runtime_observation_metadata=metadata)
    forward_batch = FakeForwardBatch(runtime_observation_metadata=metadata)
    output, messages, _ = _run_forward(runner, forward_batch, env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    _assert_zero_forbidden_safety_counts(summary)
    if summary["blocked_reason"] != "req_to_token_pool_object_missing":
        raise AssertionError(summary)
    if summary["req_to_token_read_count"] != 0:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_read_count"] != 0:
        raise AssertionError(summary)


def _assert_forward_batch_int_inspected() -> None:
    metadata = [_runtime_observation_metadata(token_span=[0, 3])]
    pool = FakeReqToTokenPool({7: [21, 22, 23]})
    runner = _make_fake_runner()
    forward_batch = FakeForwardBatch(
        runtime_observation_metadata=metadata,
        req_to_token_pool=pool,
    )
    output, messages, _ = _run_forward(runner, forward_batch, env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    _assert_zero_forbidden_safety_counts(summary)
    if summary["runtime_observation_source_bridge_state"] != "bridged":
        raise AssertionError(summary)
    if summary["inspected_count"] <= 0:
        raise AssertionError(summary)
    if int(summary["observed_type_counts"].get("int", 0)) <= 0:
        raise AssertionError(summary)
    if summary["req_to_token_read_count"] <= 0:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_read_count"] <= 0:
        raise AssertionError(summary)
    if hasattr(forward_batch, "relaykv_req_to_token_resolution_payloads"):
        raise AssertionError("no payload attach expected")


def _assert_model_runner_scalar_like_inspected() -> None:
    metadata = [_runtime_observation_metadata(token_span=[0, 1])]
    pool = FakeReqToTokenPool({7: [ScalarLikeValue()]})
    runner = _make_fake_runner(
        runtime_observation_metadata=metadata,
        req_to_token_pool=pool,
    )
    output, messages, _ = _run_forward(runner, SlotForwardBatch(), env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["inspected_count"] <= 0:
        raise AssertionError(summary)
    if summary["observed_scalar_like_count"] <= 0 and summary["observed_one_element_like_count"] <= 0:
        raise AssertionError(summary)
    if hasattr(runner, "relaykv_req_to_token_resolution_payloads"):
        raise AssertionError("no payload attach expected")


def _assert_one_element_shape_inspected() -> None:
    metadata = [_runtime_observation_metadata(token_span=[0, 1])]
    pool = FakeReqToTokenPool({7: [OneElementTensorLikeValue()]})
    runner = _make_fake_runner(
        runtime_observation_metadata=metadata,
        req_to_token_pool=pool,
    )
    output, messages, _ = _run_forward(runner, SlotForwardBatch(), env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if int(summary["observed_shape_counts"].get("[1]", 0)) <= 0:
        raise AssertionError(summary)
    if summary["observed_one_element_like_count"] <= 0:
        raise AssertionError(summary)


def _assert_index_error_blocked() -> None:
    metadata = [_runtime_observation_metadata(token_span=[0, 3])]
    runner = _make_fake_runner(
        runtime_observation_metadata=metadata,
        req_to_token_pool=ErrorReqToTokenPool(),
    )
    output, messages, _ = _run_forward(runner, SlotForwardBatch(), env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_reason"] != "req_to_token_pool_index_error":
        raise AssertionError(summary)


def _assert_poison_not_touched() -> None:
    metadata = [_runtime_observation_metadata(token_span=[0, 1])]
    poison = PoisonInspectionValue()
    runner = _make_fake_runner(
        runtime_observation_metadata=metadata,
        req_to_token_pool=FakeReqToTokenPool({7: [poison]}),
    )
    output, messages, _ = _run_forward(runner, SlotForwardBatch(), env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["inspected_count"] <= 0:
        raise AssertionError(summary)
    if poison.touched:
        raise AssertionError("poison object was touched")


def _assert_hook_exception_swallowed() -> None:
    runner = _make_fake_runner()
    model_runner_module = _load_model_runner_module()
    _install_module_stubs(model_runner_module)
    logger = model_runner_module.logger
    handler = _ListHandler()
    previous_level = logger.level
    previous_env = _set_env("SGLANG_RELAYKV_REQ_TO_TOKEN_POOL_VALUE_SHAPE_INSPECTION", "1")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with _patched_hook_exception():
            output = model_runner_module.ModelRunner.forward(runner, SlotForwardBatch())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        _restore_env("SGLANG_RELAYKV_REQ_TO_TOKEN_POOL_VALUE_SHAPE_INSPECTION", previous_env)
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)


def main() -> int:
    _assert_env_off_no_hook()
    _assert_missing_observation_blocked()
    _assert_missing_pool_blocked()
    _assert_forward_batch_int_inspected()
    _assert_model_runner_scalar_like_inspected()
    _assert_one_element_shape_inspected()
    _assert_index_error_blocked()
    _assert_poison_not_touched()
    _assert_hook_exception_swallowed()

    print("relaykv_fake_model_runner_req_to_token_pool_value_shape_inspection_smoke=pass")
    print(
        json.dumps(
            {
                "env_off": "no_lazy_import_no_hook_no_log_no_attr_write",
                "forward_batch_int": "inspected_summary_only",
                "hook_exception": "swallowed_forward_unchanged",
                "missing_observation_pool": "clean_blocked",
                "model_runner_scalar": "scalar_like_inspected",
                "one_element": "shape_recorded_without_conversion",
                "poison": "untouched_no_attr_write",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
