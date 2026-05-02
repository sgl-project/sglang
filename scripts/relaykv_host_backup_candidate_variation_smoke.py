from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from typing import Any

# flashinfer.jit.env reads FLASHINFER_WORKSPACE_BASE at import time. Set it
# before any SGLang imports so smoke-only JIT logs go to a writable /tmp path.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

import torch

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.metrics import (
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
            layer_offset = layer_idx * 4096
            self.k_buffer.append(
                (
                    torch.arange(10 * 2 * 8, dtype=self.dtype)
                    + layer_offset
                ).reshape(10, 2, 8)
            )
            self.v_buffer.append(
                (
                    torch.arange(10 * 2 * 8, dtype=self.dtype)
                    + layer_offset
                    + 1024
                ).reshape(10, 2, 8)
            )


class _FakeTokenToKVPoolAllocator:
    def __init__(self) -> None:
        self._kvcache = _FakeMHATokenToKVPool()


@dataclass(frozen=True)
class _VariationEvent:
    request_id: str
    seq_len: int
    layer_idx: int
    batch_id: str
    request_index: int
    phase: str
    token_to_kv_pool_allocator: Any


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


def _run_variation_event(event: _VariationEvent) -> dict[str, object]:
    plan = make_shadow_plan(
        seq_len=event.seq_len,
        config=_relaykv_config(),
        page_size=1,
        request_id=event.request_id,
        kv_bytes_per_token=28672,
    )
    payload = policy_event_payload(
        plan,
        extra={
            "batch_id": event.batch_id,
            "phase": event.phase,
            "request_index": event.request_index,
            "layer_idx": event.layer_idx,
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
        token_to_kv_pool_allocator=event.token_to_kv_pool_allocator,
        token_indices=[2, 3, 4, 5],
        layer_idx=event.layer_idx,
    )


def _assert_summary(summary: dict[str, Any]) -> None:
    if summary["total_candidate_events"] != 11:
        raise AssertionError(summary)
    if summary["applied_candidate_count"] != 5:
        raise AssertionError(summary)
    if summary["fallback_candidate_count"] != 5:
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
    if summary["snapshot_created_count"] != summary["applied_candidate_count"]:
        raise AssertionError(summary)
    if (
        summary["host_backup_copy_candidate_count"]
        != summary["applied_candidate_count"]
    ):
        raise AssertionError(summary)
    if summary["host_backup_copy_skipped_count"] != 6:
        raise AssertionError(summary)
    if summary["copy_equal_true_count"] != summary["applied_candidate_count"]:
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
    if set(summary["per_layer_counts"]) != {"0", "1", "2"}:
        raise AssertionError(summary)
    if summary["per_layer_counts"]["0"]["total_candidate_events"] != 6:
        raise AssertionError(summary)
    if summary["per_layer_counts"]["1"]["total_candidate_events"] != 3:
        raise AssertionError(summary)
    if summary["per_layer_counts"]["2"]["total_candidate_events"] != 2:
        raise AssertionError(summary)
    for request_id in (
        "rid-single-layered",
        "rid-multi-a",
        "rid-multi-b",
        "rid-mm-a",
        "rid-mm-b",
        "rid-fallback-only-a",
        "rid-fallback-only-b",
        "rid-skipped-no-kv",
    ):
        if request_id not in summary["per_request_counts"]:
            raise AssertionError(summary)
    if summary["per_batch_counts"]["single_request_multi_layer"][
        "host_backup_copy_executed_count"
    ] != 2:
        raise AssertionError(summary)
    if summary["per_batch_counts"]["multi_request_single_layer"][
        "total_candidate_events"
    ] != 2:
        raise AssertionError(summary)
    if summary["per_batch_counts"]["multi_request_multi_layer"][
        "total_candidate_events"
    ] != 4:
        raise AssertionError(summary)
    if summary["per_batch_counts"]["fallback_only_noop"][
        "fallback_candidate_noop_guard_count"
    ] != 2:
        raise AssertionError(summary)
    if summary["per_batch_counts"]["skipped_candidate_no_kv"][
        "host_backup_copy_skipped_count"
    ] != 1:
        raise AssertionError(summary)
    if summary["skipped_reason_counts"] != {
        "fallback_candidate_noop_guard": 5,
        "kv_cache_object_not_found": 1,
    }:
        raise AssertionError(summary)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    allocator = _FakeTokenToKVPoolAllocator()
    skipped_allocator = None
    events = [
        # single request / multi layer
        _VariationEvent(
            "rid-single-layered",
            2535,
            0,
            "single_request_multi_layer",
            0,
            "prefill",
            allocator,
        ),
        _VariationEvent(
            "rid-single-layered",
            2535,
            1,
            "single_request_multi_layer",
            0,
            "prefill",
            allocator,
        ),
        # multi request / single layer
        _VariationEvent(
            "rid-multi-a",
            2535,
            0,
            "multi_request_single_layer",
            0,
            "prefill",
            allocator,
        ),
        _VariationEvent(
            "rid-multi-b",
            8192,
            0,
            "multi_request_single_layer",
            1,
            "prefill",
            allocator,
        ),
        # multi request / multi layer
        _VariationEvent(
            "rid-mm-a",
            2535,
            1,
            "multi_request_multi_layer",
            0,
            "decode",
            allocator,
        ),
        _VariationEvent(
            "rid-mm-a",
            2535,
            2,
            "multi_request_multi_layer",
            0,
            "decode",
            allocator,
        ),
        _VariationEvent(
            "rid-mm-b",
            8192,
            1,
            "multi_request_multi_layer",
            1,
            "decode",
            allocator,
        ),
        _VariationEvent(
            "rid-mm-b",
            8192,
            2,
            "multi_request_multi_layer",
            1,
            "decode",
            allocator,
        ),
        # fallback-only no-op
        _VariationEvent(
            "rid-fallback-only-a",
            8192,
            0,
            "fallback_only_noop",
            0,
            "prefill",
            allocator,
        ),
        _VariationEvent(
            "rid-fallback-only-b",
            8192,
            0,
            "fallback_only_noop",
            1,
            "prefill",
            allocator,
        ),
        # malformed/skipped: no KV object, handled by existing skipped_reason path
        _VariationEvent(
            "rid-skipped-no-kv",
            1024,
            0,
            "skipped_candidate_no_kv",
            0,
            "prefill",
            skipped_allocator,
        ),
    ]
    candidate_events = [_run_variation_event(event) for event in events]
    summary = summarize_host_backup_copy_candidates_for_smoke(candidate_events)
    _assert_summary(summary)

    print("relaykv_host_backup_candidate_variation_smoke: ok")
    print(
        "relaykv_host_backup_candidate_variation_summary="
        + json.dumps(summary, sort_keys=True)
    )


if __name__ == "__main__":
    main()
