from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import sys
import types
from typing import Any


class PoisonField:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("poison unrelated field must not be accessed")


class PoisonTokenToKvPoolTable:
    def __init__(self, values: dict[Any, Any]) -> None:
        self._values = dict(values)
        self.getitem_calls: list[Any] = []
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False
        self.iter_called = False
        self.len_called = False
        self.repr_called = False

    def __getitem__(self, index: Any) -> Any:
        self.getitem_calls.append(index)
        if index in self._values:
            return self._values[index]
        index_as_str = str(index)
        if index_as_str in self._values:
            return self._values[index_as_str]
        return None

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
    def forbidden_method_called(self) -> bool:
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
        req_to_token_resolution_results: list[dict[str, Any]] | None = None,
        token_to_kv_pool: Any = None,
    ) -> None:
        self.request_id = "req-a"
        self.layer_id = 14
        self.batch_id = "batch-a"
        self.kv_head_group = 2
        self.kv_class = "FULL"
        if req_to_token_resolution_results is not None:
            self.relaykv_req_to_token_resolution_results = req_to_token_resolution_results
        if token_to_kv_pool is not None:
            self.token_to_kv_pool = token_to_kv_pool


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


def _req_to_token_span(
    block_id: int,
    token_start: int,
    token_end: int,
    req_to_token_entries: list[int],
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "token_start": token_start,
        "token_end": token_end,
        "token_count": token_end - token_start,
        "req_to_token_entries": list(req_to_token_entries),
        "entry_count": len(req_to_token_entries),
    }


def _req_to_token_resolution_result() -> dict[str, Any]:
    spans = [_req_to_token_span(101, 0, 3, [10, 11, 12])]
    return {
        "event_type": "relaykv_req_to_token_resolution_result",
        "resolution_state": "req_to_token_resolved",
        "request_id": "req-a",
        "layer_id": 14,
        "kv_head_group": 2,
        "kv_class": "FULL",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": "engine-req-a",
        "logical_sequence_id": "seq-req-a",
        "decision_state": "req_to_token_resolved",
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": {"req_pool_idx": 7},
        "engine_block_ref": {"req_pool_idx": 7, "cache_position": 9},
        "full_kv_req_to_token_spans": spans,
        "kv_pool_read": False,
        "kv_snapshot": False,
        "tensor_read": False,
        "attention_comparison_executed": False,
        "attention_override": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
    }


def _make_fake_runner(
    *,
    token_to_kv_pool: Any = None,
    req_to_token_resolution_results: list[dict[str, Any]] | None = None,
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
    if req_to_token_resolution_results is not None:
        runner.relaykv_req_to_token_resolution_results = req_to_token_resolution_results
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
    original = metrics_module.run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke
    metrics_module.run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke = (
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hook failure"))
    )
    try:
        yield
    finally:
        metrics_module.run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke = (
            original
        )


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
    previous_env = os.environ.get("SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    with _patched_import_counter() as counter:
        try:
            if env_value is None:
                os.environ.pop("SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ", None)
            else:
                os.environ["SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ"] = env_value
            output = model_runner_module.ModelRunner.forward(runner, forward_batch)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            if previous_env is None:
                os.environ.pop("SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ", None)
            else:
                os.environ["SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ"] = previous_env
    return output, handler.messages, counter.count


def _extract_summary(messages: list[str]) -> dict[str, Any]:
    prefix = "relaykv_live_token_to_kv_pool_index_read_summary="
    for message in messages:
        if message.startswith(prefix):
            return json.loads(message[len(prefix) :])
    raise AssertionError(messages)


def _assert_zero_forbidden_safety_counts(summary: dict[str, Any]) -> None:
    for key in (
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


def _assert_env_off_no_hook() -> None:
    table = PoisonTokenToKvPoolTable({10: 100, 11: 101, 12: 102})
    runner = _make_fake_runner(
        token_to_kv_pool=table,
        req_to_token_resolution_results=[_req_to_token_resolution_result()],
    )
    output, messages, import_count = _run_forward(
        runner,
        FakeForwardBatch(req_to_token_resolution_results=[_req_to_token_resolution_result()]),
        env_value="0",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if runner.forward_pass_id != 1:
        raise AssertionError(runner.forward_pass_id)
    if import_count != 0:
        raise AssertionError(import_count)
    if any("relaykv_live_token_to_kv_pool_index_read_summary=" in msg for msg in messages):
        raise AssertionError(messages)
    if table.getitem_calls:
        raise AssertionError(table.getitem_calls)
    if table.forbidden_method_called:
        raise AssertionError("forbidden method called")


def _assert_env_on_with_payloads() -> None:
    resolution_results = [_req_to_token_resolution_result()]
    table = PoisonTokenToKvPoolTable({10: 100, 11: 101, 12: 102})
    runner = _make_fake_runner(
        token_to_kv_pool=table,
        req_to_token_resolution_results=resolution_results,
    )
    output, messages, import_count = _run_forward(
        runner,
        FakeForwardBatch(req_to_token_resolution_results=resolution_results),
        env_value="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count < 1:
        raise AssertionError(import_count)
    summary = _extract_summary(messages)
    if summary["physical_kv_index_resolved_count"] != 1:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] != 3:
        raise AssertionError(summary)
    if summary["actual_token_to_kv_pool_read_count"] != 3:
        raise AssertionError(summary)
    if summary["live_token_to_kv_pool_index_read_count"] != 3:
        raise AssertionError(summary)
    _assert_zero_forbidden_safety_counts(summary)
    if table.getitem_calls != [10, 11, 12]:
        raise AssertionError(table.getitem_calls)
    if table.forbidden_method_called:
        raise AssertionError("forbidden method called")


def _assert_env_on_without_payloads_blocked() -> None:
    table = PoisonTokenToKvPoolTable({10: 100, 11: 101, 12: 102})
    runner = _make_fake_runner(token_to_kv_pool=table)
    output, messages, _ = _run_forward(runner, FakeForwardBatch(), env_value="1")
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["physical_kv_index_resolved_count"] != 0:
        raise AssertionError(summary)
    _assert_zero_forbidden_safety_counts(summary)
    if table.getitem_calls:
        raise AssertionError(table.getitem_calls)


def _assert_missing_token_to_kv_pool_blocked() -> None:
    resolution_results = [_req_to_token_resolution_result()]
    runner = _make_fake_runner(req_to_token_resolution_results=resolution_results)
    output, messages, _ = _run_forward(
        runner,
        FakeForwardBatch(req_to_token_resolution_results=resolution_results),
        env_value="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    _assert_zero_forbidden_safety_counts(summary)


def _assert_hook_exception_swallowed() -> None:
    _load_model_runner_module()
    resolution_results = [_req_to_token_resolution_result()]
    runner = _make_fake_runner(
        token_to_kv_pool=PoisonTokenToKvPoolTable({10: 100}),
        req_to_token_resolution_results=resolution_results,
    )
    with _patched_hook_exception():
        output, messages, import_count = _run_forward(
            runner,
            FakeForwardBatch(req_to_token_resolution_results=resolution_results),
            env_value="1",
        )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count < 1:
        raise AssertionError(import_count)
    if any("relaykv_live_token_to_kv_pool_index_read_summary=" in msg for msg in messages):
        raise AssertionError(messages)


def main() -> None:
    _assert_env_off_no_hook()
    _assert_env_on_with_payloads()
    _assert_env_on_without_payloads_blocked()
    _assert_missing_token_to_kv_pool_blocked()
    _assert_hook_exception_swallowed()
    print(
        json.dumps(
            {
                "env_off": "no_lazy_import_no_hook_no_log",
                "env_on_with_payloads": "summary_emitted_forward_unchanged",
                "env_on_without_payloads": "blocked_summary_no_crash",
                "missing_token_to_kv_pool": "blocked_summary_no_crash",
                "hook_exception": "swallowed_forward_unchanged",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
