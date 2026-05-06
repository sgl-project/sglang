from __future__ import annotations

import copy
import json
import os
from typing import Any

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


class _TokenToKvPool:
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


class _Box:
    pass


def _valid_payload(request_id: str = "req-a") -> dict[str, Any]:
    return {
        "event_type": "relaykv_req_to_token_resolution_result",
        "resolution_state": "req_to_token_resolved",
        "request_id": request_id,
        "layer_id": 14,
        "kv_head_group": 2,
        "kv_class": "FULL",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": f"engine-{request_id}",
        "logical_sequence_id": f"seq-{request_id}",
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
                "token_end": 3,
                "token_count": 3,
                "req_to_token_entries": [10, 11, 12],
                "entry_count": 3,
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
    return {"event_type": "wrong_event", "resolution_state": "blocked"}


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


def _assert_zero_forbidden_safety(summary: dict[str, Any]) -> None:
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


def _run_wrapper(
    *,
    bridge_enabled: bool,
    forward_batch: Any = None,
    model_runner: Any = None,
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


def _base_runner_and_batch() -> tuple[Any, Any, _TokenToKvPool]:
    token_to_kv_pool = _TokenToKvPool({10: 100, 11: 101, 12: 102})
    runner = _Box()
    runner.token_to_kv_pool = token_to_kv_pool
    runner.unrelated = _PoisonObject()
    forward_batch = _Box()
    forward_batch.request_id = "req-a"
    forward_batch.layer_id = 14
    forward_batch.batch_id = "batch-a"
    forward_batch.kv_head_group = 2
    forward_batch.kv_class = "FULL"
    forward_batch.unrelated = _PoisonObject()
    return runner, forward_batch, token_to_kv_pool


def _assert_bridge_off_unchanged_blocked() -> None:
    runner, forward_batch, token_to_kv_pool = _base_runner_and_batch()
    result = _run_wrapper(
        bridge_enabled=False,
        model_runner=runner,
        forward_batch=forward_batch,
    )
    summary = result["summary"]
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_enabled"] is not False:
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_state"] != "not_attempted":
        raise AssertionError(summary)
    if token_to_kv_pool.getitem_calls:
        raise AssertionError(token_to_kv_pool.getitem_calls)
    _assert_zero_forbidden_safety(summary)


def _assert_explicit_bridge_resolves() -> None:
    runner, forward_batch, token_to_kv_pool = _base_runner_and_batch()
    explicit_payloads = [_valid_payload("req-explicit")]
    before = copy.deepcopy(explicit_payloads)
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=forward_batch,
        explicit_payloads=explicit_payloads,
    )
    if explicit_payloads != before:
        raise AssertionError("explicit payloads mutated")
    summary = result["summary"]
    if summary["physical_kv_index_resolved_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_state"] != "bridged":
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_source_path"] != "explicit_payloads":
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_valid_count"] <= 0:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(summary)
    if token_to_kv_pool.getitem_calls != [10, 11, 12]:
        raise AssertionError(token_to_kv_pool.getitem_calls)
    _assert_zero_forbidden_safety(summary)


def _assert_forward_batch_bridge_resolves() -> None:
    runner, forward_batch, token_to_kv_pool = _base_runner_and_batch()
    forward_batch.relaykv_req_to_token_resolution_payloads = [_valid_payload("req-fb")]
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=forward_batch,
    )
    summary = result["summary"]
    if summary["req_to_token_resolution_bridge_source_path"] != (
        "forward_batch.relaykv_req_to_token_resolution_payloads"
    ):
        raise AssertionError(summary)
    if summary["physical_kv_index_resolved_count"] != 1:
        raise AssertionError(summary)
    if token_to_kv_pool.getitem_calls != [10, 11, 12]:
        raise AssertionError(token_to_kv_pool.getitem_calls)
    _assert_zero_forbidden_safety(summary)


def _assert_model_runner_bridge_resolves() -> None:
    runner, forward_batch, token_to_kv_pool = _base_runner_and_batch()
    runner.relaykv_req_to_token_resolution_results = [_valid_payload("req-runner")]
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=forward_batch,
    )
    summary = result["summary"]
    if summary["req_to_token_resolution_bridge_source_path"] != (
        "model_runner.relaykv_req_to_token_resolution_results"
    ):
        raise AssertionError(summary)
    if summary["physical_kv_index_resolved_count"] != 1:
        raise AssertionError(summary)
    if token_to_kv_pool.getitem_calls != [10, 11, 12]:
        raise AssertionError(token_to_kv_pool.getitem_calls)
    _assert_zero_forbidden_safety(summary)


def _assert_invalid_and_missing_block_cleanly() -> None:
    runner, forward_batch, token_to_kv_pool = _base_runner_and_batch()
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=forward_batch,
        explicit_payloads=[_invalid_payload()],
    )
    summary = result["summary"]
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_blocked_reason"] != "bridge_payload_invalid":
        raise AssertionError(summary)
    if token_to_kv_pool.getitem_calls:
        raise AssertionError(token_to_kv_pool.getitem_calls)

    runner, forward_batch, token_to_kv_pool = _base_runner_and_batch()
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=forward_batch,
    )
    summary = result["summary"]
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_resolution_bridge_blocked_reason"] != "bridge_source_missing":
        raise AssertionError(summary)
    if token_to_kv_pool.getitem_calls:
        raise AssertionError(token_to_kv_pool.getitem_calls)
    _assert_zero_forbidden_safety(summary)


def _assert_priority_and_no_mutation() -> None:
    runner, forward_batch, token_to_kv_pool = _base_runner_and_batch()
    runner.relaykv_req_to_token_resolution_results = [_invalid_payload()]
    forward_batch.relaykv_req_to_token_resolution_results = [_invalid_payload()]
    explicit_payloads = [_valid_payload("req-priority")]
    before_runner = copy.deepcopy(runner.relaykv_req_to_token_resolution_results)
    before_forward_batch = copy.deepcopy(forward_batch.relaykv_req_to_token_resolution_results)
    result = _run_wrapper(
        bridge_enabled=True,
        model_runner=runner,
        forward_batch=forward_batch,
        explicit_payloads=explicit_payloads,
    )
    summary = result["summary"]
    if summary["req_to_token_resolution_bridge_source_path"] != "explicit_payloads":
        raise AssertionError(summary)
    if runner.relaykv_req_to_token_resolution_results != before_runner:
        raise AssertionError("runner payloads mutated")
    if forward_batch.relaykv_req_to_token_resolution_results != before_forward_batch:
        raise AssertionError("forward_batch payloads mutated")
    if runner.unrelated.touched or forward_batch.unrelated.touched:
        raise AssertionError("poison object touched")
    if token_to_kv_pool.poison.touched:
        raise AssertionError("token_to_kv_pool poison touched")


def main() -> None:
    _assert_bridge_off_unchanged_blocked()
    _assert_explicit_bridge_resolves()
    _assert_forward_batch_bridge_resolves()
    _assert_model_runner_bridge_resolves()
    _assert_invalid_and_missing_block_cleanly()
    _assert_priority_and_no_mutation()
    print("relaykv_model_runner_req_to_token_bridge_live_index_read_smoke=pass")
    print(
        "relaykv_model_runner_req_to_token_bridge_live_index_read_summary="
        + json.dumps(
            {
                "bridge_off": "blocked_unchanged",
                "explicit": "bridged_and_resolved",
                "forward_batch": "bridged_and_resolved",
                "model_runner": "bridged_and_resolved",
                "invalid_missing": "clean_blocked",
                "priority": "explicit_wins",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
