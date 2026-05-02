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
class _FakeForwardBatchLike:
    rids: list[str] | Any
    req_pool_indices: list[int] | Any
    seq_lens: list[int] | Any

    @property
    def batch_size(self) -> int:
        return len(self.rids)


class _SyncForbiddenTensorLike:
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


class _ExplodingBatch:
    @property
    def rids(self) -> list[str]:
        raise RuntimeError("synthetic hook failure")


def _assert_off() -> dict[str, Any]:
    result = run_model_runner_forward_observation_hook(
        forward_batch=_FakeForwardBatchLike(
            rids=["rid-a"],
            req_pool_indices=[0],
            seq_lens=[128],
        ),
        forward_pass_id=1,
        env_value="0",
    )
    if result != {
        "enabled": False,
        "skipped": True,
        "skip_reason": "env_disabled",
        "summary": None,
    }:
        raise AssertionError(result)
    return result


def _assert_on_payloads() -> dict[str, Any]:
    result = run_model_runner_forward_observation_hook(
        forward_batch=_FakeForwardBatchLike(
            rids=["rid-a", "rid-b"],
            req_pool_indices=[0, 1],
            seq_lens=[128, 256],
        ),
        forward_pass_id=2,
        env_value="1",
    )
    summary = result["summary"]
    if result["enabled"] is not True or result["skipped"] is not False:
        raise AssertionError(result)
    if summary["total_payloads"] != 2:
        raise AssertionError(result)
    if summary["per_request_counts"] != {"rid-a": 1, "rid-b": 1}:
        raise AssertionError(result)
    if summary["per_layer_counts"] != {"0": 2}:
        raise AssertionError(result)
    if summary["per_batch_counts"] != {"forward-2": 2}:
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
    tensor_like = _SyncForbiddenTensorLike()
    result = run_model_runner_forward_observation_hook(
        forward_batch=_FakeForwardBatchLike(
            rids=["rid-a"],
            req_pool_indices=tensor_like,
            seq_lens=[128],
        ),
        forward_pass_id=3,
        env_value="1",
    )
    if tensor_like.sync_called:
        raise AssertionError("sync-like method was called")
    if result["enabled"] is not True or result["skipped"] is not True:
        raise AssertionError(result)
    if result["skip_reason"] != "TypeError":
        raise AssertionError(result)
    return result


def _assert_exception_does_not_escape() -> dict[str, Any]:
    result = run_model_runner_forward_observation_hook(
        forward_batch=_ExplodingBatch(),
        forward_pass_id=4,
        env_value="1",
    )
    if result["enabled"] is not True or result["skipped"] is not True:
        raise AssertionError(result)
    if result["skip_reason"] != "RuntimeError":
        raise AssertionError(result)
    return result


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = {
        "off": _assert_off(),
        "on": _assert_on_payloads(),
        "tensor_like_skip": _assert_tensor_like_skip(),
        "exception_skip": _assert_exception_does_not_escape(),
    }
    print("relaykv_model_runner_observation_hook_smoke: ok")
    print(
        "relaykv_model_runner_observation_hook_smoke_results="
        + json.dumps(results, sort_keys=True)
    )


if __name__ == "__main__":
    main()
