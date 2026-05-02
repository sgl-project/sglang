from __future__ import annotations

import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

from sglang.srt.relaykv.observation import (
    run_model_runner_forward_observation_hook,
)


@dataclass(frozen=True)
class _FakeForwardBatchListLike:
    rids: list[str]
    req_pool_indices: list[int]
    seq_lens: list[int]

    @property
    def batch_size(self) -> int:
        return len(self.rids)


@dataclass(frozen=True)
class _FakeForwardBatchTensorLike:
    rids: list[str]
    req_pool_indices: Any
    seq_lens: list[int] | Any

    @property
    def batch_size(self) -> int:
        return len(self.rids)


class _PoisonTensorLike:
    def __init__(self) -> None:
        self.cpu_called = False
        self.item_called = False
        self.tolist_called = False

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    @property
    def sync_called(self) -> bool:
        return self.cpu_called or self.item_called or self.tolist_called


class _ExplodingForwardBatch:
    @property
    def rids(self) -> list[str]:
        raise RuntimeError("synthetic hook failure")


def _fake_forward_with_observation_hook(
    *,
    forward_batch: Any,
    env_value: str | None,
    raise_after_hook: bool = False,
) -> dict[str, Any]:
    forward_pass_id = 0
    forward_pass_id += 1
    try:
        hook_result = run_model_runner_forward_observation_hook(
            forward_batch=forward_batch,
            forward_pass_id=forward_pass_id,
            env_value=env_value,
        )
    except Exception as exc:
        hook_result = {
            "enabled": env_value == "1",
            "skipped": True,
            "skip_reason": type(exc).__name__,
            "summary": None,
        }

    if raise_after_hook:
        raise AssertionError("fake forward body should not fail")

    return {
        "forward_completed": True,
        "forward_pass_id": forward_pass_id,
        "hook_result": hook_result,
        "result": "fake_forward_output",
    }


def _assert_forward_completed(result: dict[str, Any]) -> None:
    if result["forward_completed"] is not True:
        raise AssertionError(result)
    if result["result"] != "fake_forward_output":
        raise AssertionError(result)


def _assert_env_off() -> dict[str, Any]:
    result = _fake_forward_with_observation_hook(
        forward_batch=_FakeForwardBatchListLike(
            rids=["rid-a"],
            req_pool_indices=[0],
            seq_lens=[128],
        ),
        env_value="0",
    )
    _assert_forward_completed(result)
    hook_result = result["hook_result"]
    if hook_result != {
        "enabled": False,
        "skipped": True,
        "skip_reason": "env_disabled",
        "summary": None,
    }:
        raise AssertionError(result)
    return result


def _assert_env_on_payloads() -> dict[str, Any]:
    result = _fake_forward_with_observation_hook(
        forward_batch=_FakeForwardBatchListLike(
            rids=["rid-a", "rid-b"],
            req_pool_indices=[0, 1],
            seq_lens=[128, 256],
        ),
        env_value="1",
    )
    _assert_forward_completed(result)
    hook_result = result["hook_result"]
    summary = hook_result["summary"]
    if hook_result["enabled"] is not True or hook_result["skipped"] is not False:
        raise AssertionError(result)
    if summary["total_payloads"] != 2:
        raise AssertionError(result)
    for key in (
        "source_mutated_true_count",
        "attention_override_true_count",
        "kv_cache_mutation_true_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
    ):
        if summary[key] != 0:
            raise AssertionError(result)
    return result


def _assert_tensor_like_skip() -> dict[str, Any]:
    tensor_like = _PoisonTensorLike()
    result = _fake_forward_with_observation_hook(
        forward_batch=_FakeForwardBatchTensorLike(
            rids=["rid-a"],
            req_pool_indices=tensor_like,
            seq_lens=[128],
        ),
        env_value="1",
    )
    _assert_forward_completed(result)
    if tensor_like.sync_called:
        raise AssertionError("sync-like method was called")
    hook_result = result["hook_result"]
    if hook_result["enabled"] is not True or hook_result["skipped"] is not True:
        raise AssertionError(result)
    if hook_result["skip_reason"] != "TypeError":
        raise AssertionError(result)
    return result


def _assert_hook_exception_skip() -> dict[str, Any]:
    result = _fake_forward_with_observation_hook(
        forward_batch=_ExplodingForwardBatch(),
        env_value="1",
    )
    _assert_forward_completed(result)
    hook_result = result["hook_result"]
    if hook_result["enabled"] is not True or hook_result["skipped"] is not True:
        raise AssertionError(result)
    if hook_result["skip_reason"] != "RuntimeError":
        raise AssertionError(result)
    return result


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = {
        "env_off": _assert_env_off(),
        "env_on": _assert_env_on_payloads(),
        "tensor_like": _assert_tensor_like_skip(),
        "exception": _assert_hook_exception_skip(),
    }
    print("relaykv_fake_model_runner_forward_observation_smoke: ok")
    print(
        "relaykv_fake_model_runner_forward_observation_results="
        + json.dumps(results, sort_keys=True)
    )


if __name__ == "__main__":
    main()
