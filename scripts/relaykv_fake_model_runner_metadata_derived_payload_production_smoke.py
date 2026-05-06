from __future__ import annotations

import builtins
import contextlib
import copy
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
    run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke,
)


class _PoisonObject:
    def __init__(self) -> None:
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False
        self.iter_called = False
        self.len_called = False
        self.repr_called = False
        self.getitem_called = False

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PoisonObject":
        return self

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

    def __getitem__(self, index: Any) -> Any:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

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
                self.getitem_called,
            )
        )


class _PoisonTokenToKvPoolTable:
    def __init__(self, values: dict[Any, Any]) -> None:
        self._values = dict(values)
        self.getitem_calls: list[Any] = []
        self.poison = _PoisonObject()

    def __getitem__(self, index: Any) -> Any:
        self.getitem_calls.append(index)
        if index in self._values:
            return self._values[index]
        index_as_str = str(index)
        if index_as_str in self._values:
            return self._values[index_as_str]
        return None


class _FakeModelOutput:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel
        self.can_run_graph = False
        self.expert_distribution_metrics = None
        self.routed_experts_output = None


class _FakeForwardBatch:
    def __init__(self, runtime_payload: dict[str, Any] | None = None) -> None:
        self.request_id = "req-a"
        self.layer_id = 14
        self.batch_id = "batch-a"
        self.kv_head_group = 2
        self.kv_class = "FULL"
        if runtime_payload is not None:
            self.relaykv_runtime_observation_payloads = [runtime_payload]
        self.relaykv_kv_index_resolution_plans = [_kv_index_resolution_plan()]
        self.unrelated = _PoisonObject()


class _SlotForwardBatch:
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


def _runtime_observation_payload(
    *,
    token_span: list[int] | None,
    seq_len: int | None,
    poison: _PoisonObject | None = None,
) -> dict[str, Any]:
    adapter_metadata: dict[str, Any] = {"runtime_observation_marker": "req-a"}
    if seq_len is not None:
        adapter_metadata["seq_len"] = seq_len
    payload = {
        "event_type": "relaykv_runtime_observation_result",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": "engine-req-a",
        "logical_sequence_id": "seq-req-a",
        "request_id": "req-a",
        "logical_block_id": 101,
        "layer_id": 14,
        "kv_head_group": 2,
        "kv_class": "FULL",
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": adapter_metadata,
        "engine_block_ref": {"req_pool_idx": 7},
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
    if token_span is not None:
        payload["token_span"] = token_span
    if poison is not None:
        payload["poison"] = poison
    return payload


def _kv_index_resolution_plan() -> dict[str, Any]:
    return {
        "event_type": "relaykv_kv_index_resolution_plan",
        "resolution_state": "block_span_resolved",
        "request_id": "req-a",
        "req_pool_idx": 7,
        "layer_id": 14,
        "relaykv_plan_marker": "plan-req-a",
    }


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
    explicit_entries: Any = None,
    runtime_observation_payloads: Any = None,
    kv_index_resolution_plans: Any = None,
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
    if explicit_entries is not None:
        runner.relaykv_explicit_req_to_token_entries_for_smoke = explicit_entries
    if runtime_observation_payloads is not None:
        runner.relaykv_runtime_observation_payloads = runtime_observation_payloads
    if kv_index_resolution_plans is not None:
        runner.relaykv_kv_index_resolution_plans = kv_index_resolution_plans
    runner.unrelated = _PoisonObject()

    def _forward_raw(*args: Any, **kwargs: Any) -> _FakeModelOutput:
        return _FakeModelOutput(sentinel)

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
    production_env: str | None,
    metadata_env: str | None,
) -> tuple[_FakeModelOutput, list[str], int]:
    model_runner_module = _load_model_runner_module()
    _install_module_stubs(model_runner_module)
    logger = model_runner_module.logger
    handler = _ListHandler()
    previous_level = logger.level
    previous_values = {
        "SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION": _set_env(
            "SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION",
            production_env,
        ),
        "SGLANG_RELAYKV_RUNTIME_METADATA_DERIVED_REQ_TO_TOKEN_ENTRIES": _set_env(
            "SGLANG_RELAYKV_RUNTIME_METADATA_DERIVED_REQ_TO_TOKEN_ENTRIES",
            metadata_env,
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
    prefix = "relaykv_runtime_req_to_token_payload_production_summary="
    for message in messages:
        if message.startswith(prefix):
            return json.loads(message[len(prefix) :])
    raise AssertionError(messages)


def _assert_zero_producer_safety(summary: dict[str, Any]) -> None:
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
        if summary[key] != 0:
            raise AssertionError((key, summary[key]))


def _run_bridge_live_read(
    runner: Any,
    forward_batch: Any,
    token_to_kv_pool: _PoisonTokenToKvPoolTable,
) -> dict[str, Any]:
    previous = _set_env("SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE", "1")
    try:
        runner.token_to_kv_pool = token_to_kv_pool
        return run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke(
            runner,
            forward_batch=forward_batch,
        )
    finally:
        _restore_env("SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE", previous)


def _assert_metadata_env_off_unchanged_blocked() -> None:
    table = _PoisonTokenToKvPoolTable({10: 100, 11: 101, 12: 102})
    runner = _make_fake_runner(token_to_kv_pool=table)
    payload = _runtime_observation_payload(token_span=[10, 13], seq_len=20)
    forward_batch = _FakeForwardBatch(payload)
    output, messages, _ = _run_forward(
        runner,
        forward_batch,
        production_env="1",
        metadata_env="0",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["metadata_derived_entries_enabled"] is not False:
        raise AssertionError(summary)
    if summary["metadata_derived_entries_state"] != "not_attempted":
        raise AssertionError(summary)
    if summary["req_to_token_entry_source"] != "none":
        raise AssertionError(summary)
    if summary["payload_attached"] is not False:
        raise AssertionError(summary)
    _assert_zero_producer_safety(summary)
    if hasattr(forward_batch, "relaykv_req_to_token_resolution_payloads"):
        raise AssertionError("unexpected payload attachment")
    if table.getitem_calls:
        raise AssertionError(table.getitem_calls)


def _assert_metadata_token_span_derives_and_resolves() -> None:
    table = _PoisonTokenToKvPoolTable({10: 100, 11: 101, 12: 102})
    poison = _PoisonObject()
    payload = _runtime_observation_payload(
        token_span=[10, 13],
        seq_len=20,
        poison=poison,
    )
    before_payload = copy.deepcopy(payload)
    runner = _make_fake_runner(token_to_kv_pool=table)
    forward_batch = _FakeForwardBatch(payload)
    before_plans = copy.deepcopy(forward_batch.relaykv_kv_index_resolution_plans)
    output, messages, import_count = _run_forward(
        runner,
        forward_batch,
        production_env="1",
        metadata_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    if import_count < 1:
        raise AssertionError(import_count)
    summary = _extract_summary(messages)
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_entry_source"] != "metadata_derived":
        raise AssertionError(summary)
    if summary["metadata_derived_entries_enabled"] is not True:
        raise AssertionError(summary)
    if summary["metadata_derived_entries_state"] != "derived":
        raise AssertionError(summary)
    if summary["metadata_derived_entry_count"] != 3:
        raise AssertionError(summary)
    if summary["payload_attached"] is not True:
        raise AssertionError(summary)
    _assert_zero_producer_safety(summary)
    if payload != before_payload:
        raise AssertionError("runtime payload mutated")
    if forward_batch.relaykv_kv_index_resolution_plans != before_plans:
        raise AssertionError("kv plans mutated")
    if poison.touched or forward_batch.unrelated.touched or runner.unrelated.touched:
        raise AssertionError("poison object touched")
    attached = forward_batch.relaykv_req_to_token_resolution_payloads
    entries = attached[0]["full_kv_req_to_token_spans"][0]["req_to_token_entries"]
    if entries != [10, 11, 12]:
        raise AssertionError(entries)
    if table.getitem_calls:
        raise AssertionError(table.getitem_calls)

    live_result = _run_bridge_live_read(runner, forward_batch, table)
    live_summary = live_result["summary"]
    if live_summary["physical_kv_index_resolved_count"] != 1:
        raise AssertionError(live_summary)
    if table.getitem_calls != [10, 11, 12]:
        raise AssertionError(table.getitem_calls)


def _assert_metadata_seq_len_fallback_derives() -> None:
    table = _PoisonTokenToKvPoolTable({0: 200, 1: 201, 2: 202})
    payload = _runtime_observation_payload(token_span=None, seq_len=3)
    runner = _make_fake_runner(token_to_kv_pool=table)
    forward_batch = _FakeForwardBatch(payload)
    output, messages, _ = _run_forward(
        runner,
        forward_batch,
        production_env="1",
        metadata_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_entry_source"] != "metadata_derived":
        raise AssertionError(summary)
    attached = forward_batch.relaykv_req_to_token_resolution_payloads
    entries = attached[0]["full_kv_req_to_token_spans"][0]["req_to_token_entries"]
    if entries != [0, 1, 2]:
        raise AssertionError(entries)
    if table.getitem_calls:
        raise AssertionError(table.getitem_calls)


def _assert_explicit_entries_win() -> None:
    table = _PoisonTokenToKvPoolTable({30: 300, 31: 301, 32: 302})
    payload = _runtime_observation_payload(token_span=[10, 13], seq_len=20)
    runner = _make_fake_runner(token_to_kv_pool=table)
    forward_batch = _FakeForwardBatch(payload)
    forward_batch.relaykv_explicit_req_to_token_entries_for_smoke = [30, 31, 32]
    output, messages, _ = _run_forward(
        runner,
        forward_batch,
        production_env="1",
        metadata_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["req_to_token_entry_source"] != "explicit":
        raise AssertionError(summary)
    if summary["explicit_req_to_token_entries_present"] is not True:
        raise AssertionError(summary)
    if summary["metadata_derived_entries_state"] != "not_attempted":
        raise AssertionError(summary)
    attached = forward_batch.relaykv_req_to_token_resolution_payloads
    entries = attached[0]["full_kv_req_to_token_spans"][0]["req_to_token_entries"]
    if entries != [30, 31, 32]:
        raise AssertionError(entries)


def _assert_invalid_and_missing_metadata_blocked() -> None:
    runner = _make_fake_runner()
    invalid_payload = _runtime_observation_payload(token_span=[5, 2], seq_len=None)
    forward_batch = _FakeForwardBatch(invalid_payload)
    output, messages, _ = _run_forward(
        runner,
        forward_batch,
        production_env="1",
        metadata_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["metadata_derived_entries_state"] != "blocked":
        raise AssertionError(summary)
    if summary["metadata_derived_blocked_reason"] != "invalid_token_span":
        raise AssertionError(summary)
    if summary["payload_attached"] is not False:
        raise AssertionError(summary)

    runner = _make_fake_runner()
    forward_batch = _SlotForwardBatch()
    output, messages, _ = _run_forward(
        runner,
        forward_batch,
        production_env="1",
        metadata_env="1",
    )
    if output.sentinel != "sentinel-output":
        raise AssertionError(output)
    summary = _extract_summary(messages)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["metadata_derived_entries_state"] != "blocked":
        raise AssertionError(summary)
    if summary["metadata_derived_blocked_reason"] != "runtime_observation_payloads_missing":
        raise AssertionError(summary)
    if summary["req_to_token_entry_source"] != "none":
        raise AssertionError(summary)
    if summary["payload_attached"] is not False:
        raise AssertionError(summary)


def main() -> None:
    _assert_metadata_env_off_unchanged_blocked()
    _assert_metadata_token_span_derives_and_resolves()
    _assert_metadata_seq_len_fallback_derives()
    _assert_explicit_entries_win()
    _assert_invalid_and_missing_metadata_blocked()
    print("relaykv_fake_model_runner_metadata_derived_payload_production_smoke=pass")
    print(
        json.dumps(
            {
                "metadata_off": "clean_blocked_unchanged",
                "token_span": "metadata_derived_attached_bridge_live_resolved",
                "seq_len_fallback": "metadata_derived_attached",
                "explicit_priority": "explicit_wins",
                "invalid_missing": "clean_blocked",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
