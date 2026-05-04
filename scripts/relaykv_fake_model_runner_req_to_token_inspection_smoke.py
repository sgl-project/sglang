from __future__ import annotations

import contextlib
import json
import logging
import os
import types
from typing import Any

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


class FakeModelOutput:
    def __init__(self) -> None:
        self.can_run_graph = False
        self.expert_distribution_metrics = None
        self.routed_experts_output = None


class FakeForwardBatch:
    def __init__(self, req_to_token_pool: Any = None) -> None:
        self.req_to_token_pool = req_to_token_pool


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


def _make_fake_runner(req_to_token_pool: Any = None) -> Any:
    runner = types.SimpleNamespace()
    runner.forward_pass_id = 0
    runner.msprobe_debugger = None
    runner.server_args = types.SimpleNamespace(disable_overlap_schedule=False)
    runner.graph_runner = types.SimpleNamespace(bs=None)
    runner.eplb_manager = None
    runner.gpu_id = None
    runner.dp_size = None

    def _forward_raw(*args: Any, **kwargs: Any) -> FakeModelOutput:
        return FakeModelOutput()

    runner._forward_raw = _forward_raw
    if req_to_token_pool is not None:
        runner.req_to_token_pool = req_to_token_pool
    return runner


def _run_forward(
    runner: Any,
    forward_batch: Any,
    *,
    env_value: str | None,
) -> tuple[FakeModelOutput, list[str]]:
    model_runner_module = _load_model_runner_module()
    _install_module_stubs(model_runner_module)
    logger = model_runner_module.logger
    handler = _ListHandler()
    previous_level = logger.level
    previous_env = os.environ.get("SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        if env_value is None:
            os.environ.pop("SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION", None)
        else:
            os.environ["SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION"] = env_value
        output = model_runner_module.ModelRunner.forward(runner, forward_batch)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        if previous_env is None:
            os.environ.pop("SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION", None)
        else:
            os.environ["SGLANG_RELAYKV_REQ_TO_TOKEN_INSPECTION"] = previous_env
    return output, handler.messages


def _extract_summary(messages: list[str]) -> dict[str, Any]:
    for message in messages:
        prefix = "relaykv_req_to_token_runtime_inspection="
        if message.startswith(prefix):
            return json.loads(message[len(prefix) :])
    raise AssertionError(messages)


def _assert_env_off_no_hook() -> None:
    table = FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
    pool = FakeReqToTokenPool(table)
    runner = _make_fake_runner(req_to_token_pool=pool)
    output, messages = _run_forward(runner, FakeForwardBatch(), env_value="0")
    if not isinstance(output, FakeModelOutput):
        raise AssertionError(output)
    if runner.forward_pass_id != 1:
        raise AssertionError(runner.forward_pass_id)
    if pool.req_to_token_access_count != 0:
        raise AssertionError(pool.req_to_token_access_count)
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    if any("relaykv_req_to_token_runtime_inspection=" in msg for msg in messages):
        raise AssertionError(messages)


def _assert_env_on_metadata_hook() -> None:
    table = FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
    pool = FakeReqToTokenPool(table)
    runner = _make_fake_runner(req_to_token_pool=pool)
    output, messages = _run_forward(runner, FakeForwardBatch(), env_value="1")
    if not isinstance(output, FakeModelOutput):
        raise AssertionError(output)
    if runner.forward_pass_id != 1:
        raise AssertionError(runner.forward_pass_id)
    if pool.req_to_token_access_count != 1:
        raise AssertionError(pool.req_to_token_access_count)
    if table.forbidden_read_called:
        raise AssertionError("table values were read")
    summary = _extract_summary(messages)
    if summary["metadata_observed_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_attr_present_count"] != 1:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_inspection_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_attr_observed_count"] != 1:
        raise AssertionError(summary)
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


def _assert_missing_pool_blocked() -> None:
    runner = _make_fake_runner()
    output, messages = _run_forward(runner, FakeForwardBatch(), env_value="1")
    if not isinstance(output, FakeModelOutput):
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["metadata_observed_count"] != 0:
        raise AssertionError(summary)
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


def main() -> None:
    _assert_env_off_no_hook()
    _assert_env_on_metadata_hook()
    _assert_missing_pool_blocked()
    print(
        json.dumps(
            {
                "env_off": "no_hook_no_log",
                "env_on": "metadata_summary_emitted",
                "missing_pool": "blocked_no_crash",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
