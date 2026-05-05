from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import sys
import types
from typing import Any


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
        return any(
            (
                self.len_called,
                self.iter_called,
                self.getitem_called,
                self.cpu_called,
                self.tolist_called,
                self.item_called,
                self.numpy_called,
            )
        )


class FakeModelOutput:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel
        self.can_run_graph = False
        self.expert_distribution_metrics = None
        self.routed_experts_output = None


class FakeForwardBatch:
    def __init__(self, token_to_kv_pool: Any = None) -> None:
        if token_to_kv_pool is not None:
            self.token_to_kv_pool = token_to_kv_pool
        self.request_id = "req-a"
        self.layer_id = 14
        self.batch_id = "batch-a"


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
    token_to_kv_pool: Any = None,
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
    if token_to_kv_pool is not None:
        runner.token_to_kv_pool = token_to_kv_pool

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
    original = metrics_module.run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke
    metrics_module.run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke = (
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hook failure"))
    )
    try:
        yield
    finally:
        metrics_module.run_model_runner_token_to_kv_pool_runtime_inspection_hook_for_smoke = original


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
    previous_env = os.environ.get("SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    with _patched_import_counter() as counter:
        try:
            if env_value is None:
                os.environ.pop("SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION", None)
            else:
                os.environ["SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION"] = env_value
            output = model_runner_module.ModelRunner.forward(runner, forward_batch)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            if previous_env is None:
                os.environ.pop("SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION", None)
            else:
                os.environ["SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INSPECTION"] = previous_env
    return output, handler.messages, counter.count


def _extract_summary(messages: list[str]) -> dict[str, Any]:
    prefix = "relaykv_token_to_kv_pool_runtime_inspection_summary="
    for message in messages:
        if message.startswith(prefix):
            return json.loads(message[len(prefix) :])
    raise AssertionError(messages)


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
            raise AssertionError(summary)


def _assert_env_off_no_hook() -> None:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    runner = _make_fake_runner(token_to_kv_pool=table)
    output, messages, import_count = _run_forward(
        runner,
        FakeForwardBatch(),
        env_value="0",
    )
    if not isinstance(output, FakeModelOutput) or output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if runner.forward_pass_id != 1:
        raise AssertionError(runner.forward_pass_id)
    if import_count != 0:
        raise AssertionError(import_count)
    if any(
        "relaykv_token_to_kv_pool_runtime_inspection_summary=" in message
        for message in messages
    ):
        raise AssertionError(messages)
    if table.forbidden_read_called:
        raise AssertionError("token_to_kv_pool values were read")


def _assert_env_on_metadata_hook() -> None:
    table = FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32")
    runner = _make_fake_runner(token_to_kv_pool=table)
    output, messages, import_count = _run_forward(
        runner,
        FakeForwardBatch(),
        env_value="1",
    )
    if not isinstance(output, FakeModelOutput) or output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count < 1:
        raise AssertionError(import_count)
    summary = _extract_summary(messages)
    if summary["metadata_observed_count"] != 1:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_attr_present_count"] != 1:
        raise AssertionError(summary)
    if summary["actual_token_to_kv_pool_inspection_count"] != 1:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_attr_observed_count"] != 1:
        raise AssertionError(summary)
    _assert_zero_safety_counts(summary)
    if table.forbidden_read_called:
        raise AssertionError("token_to_kv_pool values were read")


def _assert_missing_pool_blocked() -> None:
    runner = _make_fake_runner()
    output, messages, import_count = _run_forward(
        runner,
        FakeForwardBatch(),
        env_value="1",
    )
    if not isinstance(output, FakeModelOutput) or output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count < 1:
        raise AssertionError(import_count)
    summary = _extract_summary(messages)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["metadata_observed_count"] != 0:
        raise AssertionError(summary)
    _assert_zero_safety_counts(summary)


def _assert_hook_exception_swallowed() -> None:
    _load_model_runner_module()
    runner = _make_fake_runner(token_to_kv_pool=FakeTokenToKvPoolTable((16, 1024), "cuda:0", "torch.int32"))
    with _patched_hook_exception():
        output, messages, import_count = _run_forward(
            runner,
            FakeForwardBatch(),
            env_value="1",
        )
    if not isinstance(output, FakeModelOutput) or output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count < 1:
        raise AssertionError(import_count)
    if any(
        "relaykv_token_to_kv_pool_runtime_inspection_summary=" in message
        for message in messages
    ):
        raise AssertionError(messages)


def main() -> None:
    _assert_env_off_no_hook()
    _assert_env_on_metadata_hook()
    _assert_missing_pool_blocked()
    _assert_hook_exception_swallowed()
    print(
        json.dumps(
            {
                "env_off": "no_lazy_import_no_hook_no_log",
                "env_on": "metadata_summary_emitted",
                "missing_pool": "blocked_no_crash",
                "hook_exception": "swallowed_no_crash",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
