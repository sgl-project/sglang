from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import sys
import types
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke,
)


class _Poison:
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


class _ControlledTokenToKvPool:
    def __init__(self, values: dict[Any, Any]) -> None:
        self._values = dict(values)
        self.getitem_calls: list[Any] = []
        self.poison = _Poison()

    def __getitem__(self, index: Any) -> Any:
        self.getitem_calls.append(index)
        if index in self._values:
            return self._values[index]
        if str(index) in self._values:
            return self._values[str(index)]
        return None


class _IndexErrorPool:
    def __init__(self) -> None:
        self.getitem_calls: list[Any] = []

    def __getitem__(self, index: Any) -> Any:
        self.getitem_calls.append(index)
        raise KeyError(index)


class _PoisonField:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("poison field must not be accessed")


class _FakeModelOutput:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel
        self.can_run_graph = False
        self.expert_distribution_metrics = None
        self.routed_experts_output = None


class _FakeForwardBatch:
    def __init__(self, *, payloads: Any = None, token_to_kv_pool: Any = None) -> None:
        self.request_id = "req-forward"
        self.layer_id = 14
        self.batch_id = "batch-forward"
        self.kv_head_group = 2
        self.kv_class = "FULL"
        self.unrelated = _PoisonField()
        if payloads is not None:
            self.relaykv_req_to_token_resolution_payloads = payloads
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


def _valid_payload(entries: list[int] | tuple[int, ...] = (10, 11, 12)) -> dict[str, Any]:
    token_entries = list(entries)
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
        "engine_block_ref": {"req_pool_idx": 7},
        "full_kv_req_to_token_spans": [
            {
                "block_id": 101,
                "token_start": 0,
                "token_end": len(token_entries),
                "token_count": len(token_entries),
                "req_to_token_entries": token_entries,
                "entry_count": len(token_entries),
            }
        ],
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


def _invalid_payload() -> dict[str, Any]:
    return {"event_type": "wrong", "resolution_state": "blocked"}


def _assert_zero_forbidden_live(summary: dict[str, Any]) -> None:
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
        if summary.get(key) != 0:
            raise AssertionError((key, summary.get(key)))


def _set_bridge_env(enabled: bool) -> str | None:
    key = "SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE"
    previous = os.environ.get(key)
    if enabled:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    return previous


def _restore_bridge_env(previous: str | None) -> None:
    key = "SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE"
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def _direct_read(
    *,
    payloads: list[dict[str, Any]],
    token_to_kv_pool_object: Any,
    max_tokens_per_request: int = 8,
    max_total_tokens: int = 32,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        payloads,
        token_to_kv_pool_object=token_to_kv_pool_object,
        read_token_to_kv_pool_index=True,
        max_tokens_per_request=max_tokens_per_request,
        max_total_tokens=max_total_tokens,
        source_path="direct.token_to_kv_pool",
    )
    resolved_count = sum(
        1 for result in results if result.get("resolution_state") == "physical_kv_index_resolved"
    )
    blocked_count = sum(1 for result in results if result.get("resolution_state") == "blocked")
    token_read_count = sum(int(result.get("token_to_kv_pool_read_count") or 0) for result in results)
    actual_token_read_count = sum(
        int(result.get("actual_token_to_kv_pool_read_count") or 0) for result in results
    )
    live_read_count = sum(
        int(result.get("live_token_to_kv_pool_index_read_count") or 0) for result in results
    )
    blocked_reason = None
    for result in results:
        reasons = result.get("blocking_reasons")
        if isinstance(reasons, list) and reasons:
            blocked_reason = reasons[0]
            break
    summary = {
        "resolved_count": resolved_count,
        "blocked_count": blocked_count,
        "token_to_kv_pool_read_count": token_read_count,
        "actual_token_to_kv_pool_read_count": actual_token_read_count,
        "live_token_to_kv_pool_index_read_count": live_read_count,
        "blocked_reason": blocked_reason,
    }
    _assert_zero_forbidden_live(
        {
            **summary,
            "req_to_token_read_count": 0,
            "actual_req_to_token_pool_read_count": 0,
            "kv_pool_read_count": 0,
            "kv_snapshot_count": 0,
            "tensor_read_count": 0,
            "attention_comparison_executed_count": 0,
            "attention_override_true_count": 0,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation_true_count": 0,
            "source_mutated_true_count": 0,
        }
    )
    return summary, results


def _base_runner(token_to_kv_pool: Any = None) -> Any:
    runner = types.SimpleNamespace()
    runner.forward_pass_id = 0
    runner.msprobe_debugger = None
    runner.server_args = types.SimpleNamespace(disable_overlap_schedule=False)
    runner.graph_runner = types.SimpleNamespace(bs=None)
    runner.eplb_manager = None
    runner.gpu_id = None
    runner.dp_size = None
    runner.unrelated = _PoisonField()
    if token_to_kv_pool is not None:
        runner.token_to_kv_pool = token_to_kv_pool

    def _forward_raw(*args: Any, **kwargs: Any) -> _FakeModelOutput:
        return _FakeModelOutput("sentinel-output")

    runner._forward_raw = _forward_raw
    return runner


def _load_model_runner_module():
    os.environ["HOME"] = "/tmp"
    os.environ["XDG_CACHE_HOME"] = "/tmp"
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")
    os.environ.setdefault("FLASHINFER_JIT_CACHE_DIR", "/tmp/relaykv_flashinfer_jit")
    from sglang.srt.model_executor import model_runner as model_runner_module

    return model_runner_module


def _install_module_stubs(model_runner_module) -> None:
    model_runner_module.get_global_expert_distribution_recorder = lambda: _Recorder()
    model_runner_module.get_global_experts_capturer = lambda: _ExpertsCapturer()
    model_runner_module.ElasticEPStateManager = types.SimpleNamespace(
        instance=lambda: None
    )
    model_runner_module.dumper = types.SimpleNamespace(may_enable=False, step=lambda: None)


@contextlib.contextmanager
def _patched_import_counter():
    counter = _ImportCounter()
    previous_import = builtins.__import__
    builtins.__import__ = counter
    try:
        yield counter
    finally:
        builtins.__import__ = previous_import


def _run_forward(
    runner: Any,
    forward_batch: Any,
    *,
    env_value: str | None,
) -> tuple[_FakeModelOutput, list[str], int]:
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


def _can_run_forward_path() -> bool:
    try:
        _load_model_runner_module()
    except OSError as exc:
        if "flashinfer" in str(exc) or "Read-only file system" in str(exc):
            return False
        raise
    return True


def _extract_summary(messages: list[str]) -> dict[str, Any]:
    prefix = "relaykv_live_token_to_kv_pool_index_read_summary="
    for message in messages:
        if message.startswith(prefix):
            return json.loads(message[len(prefix) :])
    raise AssertionError(messages)


def _assert_bridge_disabled_blocked() -> None:
    runner = _base_runner(token_to_kv_pool=_ControlledTokenToKvPool({10: 100}))
    forward_batch = _FakeForwardBatch(payloads=[_valid_payload()])
    result = _run_wrapper(bridge_enabled=False, model_runner=runner, forward_batch=forward_batch)
    summary = result["summary"]
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] != 0:
        raise AssertionError(summary)


def _run_wrapper(
    *,
    bridge_enabled: bool,
    model_runner: Any,
    forward_batch: Any,
    explicit_payloads: Any = None,
) -> dict[str, Any]:
    previous = _set_bridge_env(bridge_enabled)
    try:
        return run_model_runner_live_token_to_kv_pool_index_read_hook_for_smoke(
            model_runner,
            forward_batch=forward_batch,
            explicit_req_to_token_resolution_payloads=explicit_payloads,
        )
    finally:
        _restore_bridge_env(previous)


def _assert_bridge_enabled_missing_payload_blocked() -> None:
    pool = _ControlledTokenToKvPool({10: 100})
    runner = _base_runner(token_to_kv_pool=pool)
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=_FakeForwardBatch(),
    )
    summary = result["summary"]
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] != 0:
        raise AssertionError(summary)
    if pool.getitem_calls:
        raise AssertionError(pool.getitem_calls)


def _assert_bridge_enabled_missing_source_blocked() -> None:
    runner = _base_runner()
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=_FakeForwardBatch(payloads=[_valid_payload()]),
    )
    summary = result["summary"]
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["blocked_reason"] != "token_to_kv_pool_object_missing_after_bridged_req_to_token_payload":
        raise AssertionError(summary)


def _assert_direct_sources_resolve() -> None:
    summary, results = _direct_read(
        payloads=[_valid_payload()],
        token_to_kv_pool_object={10: 100, 11: 101, 12: 102},
    )
    if summary["resolved_count"] != 1 or summary["token_to_kv_pool_read_count"] != 3:
        raise AssertionError((summary, results))
    resolved = results[0]
    if resolved.get("physical_kv_index_preview_count") != 3:
        raise AssertionError(resolved)

    summary, _ = _direct_read(
        payloads=[_valid_payload()],
        token_to_kv_pool_object=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 101, 102],
    )
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)

    summary, _ = _direct_read(
        payloads=[_valid_payload()],
        token_to_kv_pool_object=tuple([0] * 10 + [100, 101, 102]),
    )
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)


def _assert_controlled_object_and_limits() -> None:
    pool = _ControlledTokenToKvPool({10: 100, 11: 101, 12: 102})
    summary, _ = _direct_read(payloads=[_valid_payload()], token_to_kv_pool_object=pool)
    if summary["resolved_count"] != 1:
        raise AssertionError(summary)
    if pool.getitem_calls != [10, 11, 12]:
        raise AssertionError(pool.getitem_calls)
    if pool.poison.touched:
        raise AssertionError("poison touched")

    summary, _ = _direct_read(
        payloads=[_valid_payload(entries=list(range(20)))],
        token_to_kv_pool_object={index: index + 100 for index in range(20)},
        max_tokens_per_request=8,
    )
    if summary["blocked_reason"] != "max_tokens_per_request_exceeded":
        raise AssertionError(summary)

    summary, _ = _direct_read(
        payloads=[_valid_payload(entries=list(range(10))), _valid_payload(entries=list(range(10)))],
        token_to_kv_pool_object={index: index + 100 for index in range(10)},
        max_tokens_per_request=16,
        max_total_tokens=12,
    )
    if summary["blocked_reason"] != "max_total_tokens_exceeded":
        raise AssertionError(summary)


def _assert_error_and_non_int_blocked() -> None:
    summary, _ = _direct_read(
        payloads=[_valid_payload()],
        token_to_kv_pool_object=_IndexErrorPool(),
    )
    if summary["blocked_reason"] != "token_to_kv_pool_index_read_failed":
        raise AssertionError(summary)

    summary, _ = _direct_read(
        payloads=[_valid_payload()],
        token_to_kv_pool_object={10: 100, 11: "bad", 12: 102},
    )
    if summary["blocked_reason"] != "token_to_kv_pool_entry_not_int":
        raise AssertionError(summary)


def _assert_wrapper_and_forward_path() -> None:
    pool = _ControlledTokenToKvPool({10: 100, 11: 101, 12: 102})
    runner = _base_runner(token_to_kv_pool=pool)
    forward_batch = _FakeForwardBatch(payloads=[_valid_payload()])
    wrapper_result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=forward_batch,
    )
    summary = wrapper_result["summary"]
    if summary["req_to_token_resolution_bridge_state"] != "bridged":
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_valid_count"] <= 0:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(summary)
    if summary["actual_token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(summary)
    if summary["live_token_to_kv_pool_index_read_count"] <= 0:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_source_path"] != "model_runner.token_to_kv_pool":
        raise AssertionError(summary)
    _assert_zero_forbidden_live(summary)

    if _can_run_forward_path():
        output, messages, import_count = _run_forward(
            runner,
            forward_batch,
            env_value="1",
        )
        if output.sentinel != "sentinel-output":
            raise AssertionError(output)
        if import_count < 1:
            raise AssertionError(import_count)
        summary = _extract_summary(messages)
        if summary["physical_kv_index_resolved_count"] != 1:
            raise AssertionError(summary)
        if summary["req_to_token_resolution_bridge_state"] != "bridged":
            raise AssertionError(summary)
        if summary["token_to_kv_pool_read_count"] <= 0:
            raise AssertionError(summary)

        output, messages, import_count = _run_forward(
            _base_runner(token_to_kv_pool=_ControlledTokenToKvPool({10: 100})),
            _FakeForwardBatch(payloads=[_valid_payload()]),
            env_value="0",
        )
        if output.sentinel != "sentinel-output":
            raise AssertionError(output)
        if import_count != 0:
            raise AssertionError(import_count)
        if any(
            "relaykv_live_token_to_kv_pool_index_read_summary=" in message
            for message in messages
        ):
            raise AssertionError(messages)


def main() -> None:
    _assert_bridge_disabled_blocked()
    _assert_bridge_enabled_missing_payload_blocked()
    _assert_bridge_enabled_missing_source_blocked()
    _assert_direct_sources_resolve()
    _assert_controlled_object_and_limits()
    _assert_error_and_non_int_blocked()
    _assert_wrapper_and_forward_path()
    print("relaykv_fake_model_runner_bridge_payload_live_token_to_kv_pool_read_smoke=pass")
    print(
        json.dumps(
            {
                "bridge_disabled": "blocked",
                "missing_payload": "blocked",
                "missing_source": "blocked_after_bridged_payload",
                "dict_list_tuple": "resolved",
                "controlled_object": "resolved_bounded",
                "limits": "enforced",
                "index_error_non_int": "blocked",
                "wrapper_forward": (
                    "resolved_forward_unchanged"
                    if _can_run_forward_path()
                    else "wrapper_resolved_forward_path_not_testable_locally"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
