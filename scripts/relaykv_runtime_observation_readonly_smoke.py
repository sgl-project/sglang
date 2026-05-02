from __future__ import annotations

import json
import logging
import os
import signal
from dataclasses import dataclass, replace
from typing import Any

# flashinfer.jit.env reads FLASHINFER_WORKSPACE_BASE at import time. Set it
# before any SGLang imports so smoke-only JIT logs go to a writable /tmp path.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

import torch

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.metrics import (
    log_host_backup_copy_candidate_summary,
    policy_event_payload,
    summarize_host_backup_copy_candidates_for_smoke,
)
from sglang.srt.relaykv.memory import run_host_backup_copy_candidate_for_smoke


class _FakeMHATokenToKVPool:
    def __init__(self, *, layers: int = 3) -> None:
        self.start_layer = 0
        self.dtype = torch.float16
        self.device = "cpu"
        self.k_buffer = []
        self.v_buffer = []
        for layer_idx in range(layers):
            offset = layer_idx * 4096
            self.k_buffer.append(
                (torch.arange(12 * 2 * 8, dtype=self.dtype) + offset).reshape(
                    12, 2, 8
                )
            )
            self.v_buffer.append(
                (torch.arange(12 * 2 * 8, dtype=self.dtype) + offset + 1024).reshape(
                    12, 2, 8
                )
            )


class _FakeTokenToKVPoolAllocator:
    def __init__(self) -> None:
        self._kvcache = _FakeMHATokenToKVPool()


@dataclass(frozen=True)
class _FakeRuntimeObservationBatch:
    batch_id: str
    rids: list[str]
    req_pool_indices: list[int]
    seq_lens: list[int]
    layer_ids: list[int]

    @property
    def batch_size(self) -> int:
        return len(self.rids)


@dataclass(frozen=True)
class _ObservationCase:
    request_id: str
    layer_idx: int
    runtime_policy_state: str


def _relaykv_config() -> RelayKVConfig:
    return RelayKVConfig(
        enabled=True,
        mode="shadow",
        kv_working_budget_tokens=2048,
        recent_window=512,
        anchor_blocks=2,
        budget_block_size=128,
        retrieval_top_k=4,
    )


def _make_candidate_event(
    *,
    batch: _FakeRuntimeObservationBatch,
    case: _ObservationCase,
    allocator: _FakeTokenToKVPoolAllocator,
) -> dict[str, object]:
    request_index = batch.rids.index(case.request_id)
    seq_len = batch.seq_lens[request_index]
    plan = make_shadow_plan(
        seq_len=seq_len,
        config=_relaykv_config(),
        page_size=1,
        request_id=case.request_id,
        kv_bytes_per_token=28672,
    )
    plan = replace(plan, runtime_policy_state=case.runtime_policy_state)
    payload = policy_event_payload(
        plan,
        extra={
            "runtime_policy_state": case.runtime_policy_state,
            "batch_id": batch.batch_id,
            "batch_size": batch.batch_size,
            "batch_seq_lens": list(batch.seq_lens),
            "batch_req_pool_indices": list(batch.req_pool_indices),
            "batch_layer_ids": list(batch.layer_ids),
            "request_index": request_index,
            "request_pool_idx": batch.req_pool_indices[request_index],
            "layer_idx": case.layer_idx,
            "phase": "runtime_observation_readonly",
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "attention_override": False,
            "runtime_writeback": False,
            "host_backup_copy": False,
        },
    )
    return run_host_backup_copy_candidate_for_smoke(
        plan=plan,
        event_payload=payload,
        token_to_kv_pool_allocator=allocator,
        token_indices=[2, 3, 4, 5],
        layer_idx=case.layer_idx,
    )


def _assert_summary(summary: dict[str, Any]) -> None:
    if summary["total_candidate_events"] != 6:
        raise AssertionError(summary)
    if summary["applied_candidate_count"] != 4:
        raise AssertionError(summary)
    if summary["fallback_candidate_count"] != 2:
        raise AssertionError(summary)
    if (
        summary["host_backup_copy_executed_count"]
        != summary["applied_candidate_count"]
    ):
        raise AssertionError(summary)
    if (
        summary["fallback_candidate_noop_guard_count"]
        != summary["fallback_candidate_count"]
    ):
        raise AssertionError(summary)
    if summary["host_backup_copy_executed_count"] != 4:
        raise AssertionError(summary)
    if summary["fallback_candidate_noop_guard_count"] != 2:
        raise AssertionError(summary)
    if set(summary["per_layer_counts"]) != {"0", "1", "2"}:
        raise AssertionError(summary)
    if set(summary["per_request_counts"]) != {"rid-a", "rid-b", "rid-c"}:
        raise AssertionError(summary)
    if set(summary["per_batch_counts"]) != {"obs-batch-a"}:
        raise AssertionError(summary)
    if summary["per_batch_counts"]["obs-batch-a"]["total_candidate_events"] != 6:
        raise AssertionError(summary)
    if summary["per_request_counts"]["rid-a"]["total_candidate_events"] != 2:
        raise AssertionError(summary)
    if summary["per_request_counts"]["rid-b"]["host_backup_copy_executed_count"] != 2:
        raise AssertionError(summary)
    if summary["per_request_counts"]["rid-c"]["fallback_candidate_count"] != 1:
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
    if summary["skipped_reason_counts"] != {"fallback_candidate_noop_guard": 2}:
        raise AssertionError(summary)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    batch = _FakeRuntimeObservationBatch(
        batch_id="obs-batch-a",
        rids=["rid-a", "rid-b", "rid-c"],
        req_pool_indices=[0, 1, 2],
        seq_lens=[128, 256, 384],
        layer_ids=[0, 1, 2],
    )
    cases = [
        _ObservationCase("rid-a", 0, "applied_candidate"),
        _ObservationCase("rid-a", 1, "fallback_candidate"),
        _ObservationCase("rid-b", 0, "applied_candidate"),
        _ObservationCase("rid-b", 2, "applied_candidate"),
        _ObservationCase("rid-c", 1, "fallback_candidate"),
        _ObservationCase("rid-c", 2, "applied_candidate"),
    ]
    allocator = _FakeTokenToKVPoolAllocator()
    candidate_events = [
        _make_candidate_event(batch=batch, case=case, allocator=allocator)
        for case in cases
    ]
    summary = summarize_host_backup_copy_candidates_for_smoke(candidate_events)
    _assert_summary(summary)
    log_host_backup_copy_candidate_summary(
        candidate_events,
        prefix="relaykv_runtime_observation_readonly_summary_log",
    )

    print("relaykv_runtime_observation_readonly_smoke: ok")
    print(
        "relaykv_runtime_observation_readonly_batch="
        + json.dumps(
            {
                "batch_id": batch.batch_id,
                "batch_size": batch.batch_size,
                "rids": batch.rids,
                "req_pool_indices": batch.req_pool_indices,
                "seq_lens": batch.seq_lens,
                "layer_ids": batch.layer_ids,
            },
            sort_keys=True,
        )
    )
    print(
        "relaykv_runtime_observation_readonly_summary="
        + json.dumps(summary, sort_keys=True)
    )


if __name__ == "__main__":
    main()
