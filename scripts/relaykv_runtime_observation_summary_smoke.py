from __future__ import annotations

import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

from sglang.srt.relaykv.observation import (
    build_runtime_observation_payloads,
    log_runtime_observation_summary,
    summarize_runtime_observation_payloads,
)


@dataclass(frozen=True)
class _FakeForwardBatchLike:
    rids: list[str]
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


def _assert_summary(summary: dict[str, Any]) -> None:
    if summary["total_payloads"] != 9:
        raise AssertionError(summary)
    if summary["per_request_counts"] != {"rid-a": 3, "rid-b": 3, "rid-c": 3}:
        raise AssertionError(summary)
    if summary["per_layer_counts"] != {"0": 3, "1": 3, "2": 3}:
        raise AssertionError(summary)
    if summary["per_batch_counts"] != {"obs-batch-a": 9}:
        raise AssertionError(summary)
    if summary["source_mutated_true_count"] != 0:
        raise AssertionError(summary)
    if summary["attention_override_true_count"] != 0:
        raise AssertionError(summary)
    if summary["kv_cache_mutation_true_count"] != 0:
        raise AssertionError(summary)
    if summary["runtime_writeback_true_count"] != 0:
        raise AssertionError(summary)
    if summary["scheduler_policy_noop_false_count"] != 0:
        raise AssertionError(summary)


def _assert_skip_for_tensor_like(field_name: str) -> str:
    tensor_like = _SyncForbiddenTensorLike()
    batch = _FakeForwardBatchLike(
        rids=["rid-a", "rid-b", "rid-c"],
        req_pool_indices=[0, 1, 2],
        seq_lens=[128, 256, 384],
    )
    if field_name == "req_pool_indices":
        batch = _FakeForwardBatchLike(
            rids=batch.rids,
            req_pool_indices=tensor_like,
            seq_lens=batch.seq_lens,
        )
    elif field_name == "seq_lens":
        batch = _FakeForwardBatchLike(
            rids=batch.rids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=tensor_like,
        )
    else:
        raise AssertionError(field_name)

    try:
        build_runtime_observation_payloads(
            batch=batch,
            layer_ids=[0],
            batch_id="obs-batch-skip",
            phase="decode",
            runtime_policy_state="shadow_observation",
        )
    except TypeError as exc:
        if tensor_like.sync_called:
            raise AssertionError("sync-like method was called") from exc
        return type(exc).__name__

    raise AssertionError("tensor-like payload build should fail safely")


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    batch = _FakeForwardBatchLike(
        rids=["rid-a", "rid-b", "rid-c"],
        req_pool_indices=[0, 1, 2],
        seq_lens=[128, 256, 384],
    )
    payloads = build_runtime_observation_payloads(
        batch=batch,
        layer_ids=[0, 1, 2],
        batch_id="obs-batch-a",
        phase="decode",
        runtime_policy_state="shadow_observation",
    )
    summary = summarize_runtime_observation_payloads(payloads)
    _assert_summary(summary)
    skip_results = {
        "req_pool_indices": _assert_skip_for_tensor_like("req_pool_indices"),
        "seq_lens": _assert_skip_for_tensor_like("seq_lens"),
    }
    log_runtime_observation_summary(
        summary,
        prefix="relaykv_runtime_observation_summary_smoke_log",
    )

    print("relaykv_runtime_observation_summary_smoke: ok")
    print(
        "relaykv_runtime_observation_summary_smoke_summary="
        + json.dumps(summary, sort_keys=True)
    )
    print(
        "relaykv_runtime_observation_summary_smoke_skip="
        + json.dumps(skip_results, sort_keys=True)
    )


if __name__ == "__main__":
    main()
