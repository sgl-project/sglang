from __future__ import annotations

import io
import json
import logging
import signal
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.metrics import (
    log_host_backup_copy_candidate_summary,
    log_policy_event,
    log_policy_summary,
    log_shadow_plan,
    policy_event_payload,
    summarize_candidate_events,
    summarize_host_backup_copy_candidates_for_smoke,
    summarize_policy_events,
)
from sglang.srt.relaykv.memory import run_host_backup_copy_candidate_for_smoke


@dataclass(frozen=True)
class _FakeRuntimeRequest:
    rid: str
    seq_len: int
    step_idx: int


class _FakeMHATokenToKVPool:
    def __init__(self) -> None:
        self.start_layer = 0
        self.dtype = torch.float16
        self.device = "cpu"
        self.k_buffer = [
            torch.arange(8 * 2 * 8, dtype=self.dtype).reshape(8, 2, 8),
        ]
        self.v_buffer = [
            (torch.arange(8 * 2 * 8, dtype=self.dtype) + 1024).reshape(8, 2, 8),
        ]


class _FakeTokenToKVPoolAllocator:
    def __init__(self) -> None:
        self._kvcache = _FakeMHATokenToKVPool()


def _runtime_policy_pass(
    *,
    requests: list[_FakeRuntimeRequest],
    config: RelayKVConfig,
    phase: str,
    kv_bytes_per_token: Optional[int],
    token_to_kv_pool_allocator: object,
) -> tuple[list[object], list[dict[str, object]]]:
    plans = []
    candidate_events = []
    for request_index, req in enumerate(requests):
        plan = make_shadow_plan(
            seq_len=req.seq_len,
            config=config,
            page_size=1,
            request_id=req.rid,
            kv_bytes_per_token=kv_bytes_per_token,
        )
        plans.append(plan)
        log_shadow_plan(
            plan,
            prefix=f"relaykv_shadow_plan_{phase}",
            extra={
                "phase": phase,
                "request_index": request_index,
                "step_idx": req.step_idx,
            },
        )
        if plan.runtime_policy_state in (
            "applied_candidate",
            "fallback_candidate",
        ):
            extra = {
                "phase": phase,
                "request_index": request_index,
                "step_idx": req.step_idx,
                "scheduler_policy_noop": True,
                "kv_cache_mutation": False,
                "attention_override": False,
                "host_backup_copy": False,
            }
            event = policy_event_payload(plan, extra=extra)
            event = run_host_backup_copy_candidate_for_smoke(
                plan=plan,
                event_payload=event,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                token_indices=[2, 3, 4, 5],
                layer_idx=0,
            )
            candidate_events.append(event)
            log_policy_event(
                plan,
                prefix=f"relaykv_runtime_policy_event_{phase}",
                extra=event,
            )
    log_policy_summary(plans, prefix=f"relaykv_runtime_policy_summary_{phase}")
    return plans, candidate_events


def _capture_relaykv_logs() -> tuple[io.StringIO, logging.Handler]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    relaykv_logger = logging.getLogger("sglang.srt.relaykv.metrics")
    relaykv_logger.setLevel(logging.INFO)
    relaykv_logger.addHandler(handler)
    relaykv_logger.propagate = False
    return stream, handler


def _release_relaykv_logs(handler: logging.Handler) -> None:
    relaykv_logger = logging.getLogger("sglang.srt.relaykv.metrics")
    relaykv_logger.removeHandler(handler)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    stream, handler = _capture_relaykv_logs()
    try:
        token_to_kv_pool_allocator = _FakeTokenToKVPoolAllocator()
        pressure_config = RelayKVConfig(
            enabled=True,
            mode="shadow",
            kv_working_budget_tokens=2048,
            recent_window=512,
            anchor_blocks=2,
            budget_block_size=128,
            retrieval_top_k=4,
        )
        pressure_requests = [
            _FakeRuntimeRequest("rid-off-a", 1024, 0),
            _FakeRuntimeRequest("rid-applied-a", 2535, 1),
            _FakeRuntimeRequest("rid-fallback-a", 4096, 2),
            _FakeRuntimeRequest("rid-off-b", 1536, 3),
            _FakeRuntimeRequest("rid-fallback-b", 8192, 4),
        ]
        plans, candidate_events = _runtime_policy_pass(
            requests=pressure_requests,
            config=pressure_config,
            phase="prefill",
            kv_bytes_per_token=28672,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

        shadow_config = RelayKVConfig(
            enabled=True,
            mode="shadow",
            available_kv_budget_mib=512.0,
            recent_window=256,
            anchor_blocks=1,
            budget_block_size=128,
            retrieval_top_k=2,
        )
        shadow_plans, shadow_candidate_events = _runtime_policy_pass(
            requests=[_FakeRuntimeRequest("rid-shadow-a", 2048, 5)],
            config=shadow_config,
            phase="decode",
            kv_bytes_per_token=None,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        plans.extend(shadow_plans)
        candidate_events.extend(shadow_candidate_events)
        log_host_backup_copy_candidate_summary(candidate_events)
    finally:
        _release_relaykv_logs(handler)

    policy_summary = summarize_policy_events(plans)
    candidate_summary = summarize_candidate_events(candidate_events)
    copy_candidate_summary = summarize_host_backup_copy_candidates_for_smoke(
        candidate_events
    )
    logs = stream.getvalue().splitlines()
    runtime_event_logs = [
        line for line in logs if line.startswith("relaykv_runtime_policy_event_")
    ]
    copy_summary_logs = [
        line
        for line in logs
        if line.startswith("relaykv_host_backup_copy_candidate_summary=")
    ]

    expected_state_counts = {
        "off": 2,
        "shadow": 1,
        "applied_candidate": 1,
        "fallback_candidate": 2,
    }
    if policy_summary["policy_state_counts"] != expected_state_counts:
        raise AssertionError(policy_summary)
    if candidate_summary["candidate_event_counts"] != {
        "applied_candidate": 1,
        "fallback_candidate": 2,
    }:
        raise AssertionError(candidate_summary)
    noop_counts = candidate_summary["noop_guard_counts"]
    if noop_counts["fallback_candidate_noop_guard_true"] != 2:
        raise AssertionError(candidate_summary)
    if noop_counts["applied_candidate_log_only_true"] != 1:
        raise AssertionError(candidate_summary)
    if noop_counts["dry_copy_candidate_true"] != 1:
        raise AssertionError(candidate_summary)
    for key in (
        "scheduler_policy_noop_true",
        "kv_cache_mutation_false",
        "attention_override_false",
        "runtime_writeback_false",
    ):
        if noop_counts[key] != 3:
            raise AssertionError(candidate_summary)
    if noop_counts["host_backup_copy_false"] != 3:
        raise AssertionError(candidate_summary)
    if noop_counts["host_backup_copy_candidate_true"] != 1:
        raise AssertionError(candidate_summary)
    if noop_counts["host_backup_copy_candidate_false"] != 2:
        raise AssertionError(candidate_summary)
    if noop_counts["host_backup_copy_executed_true"] != 1:
        raise AssertionError(candidate_summary)
    if noop_counts["host_backup_copy_executed_false"] != 2:
        raise AssertionError(candidate_summary)
    if noop_counts["snapshot_created_true"] != 1:
        raise AssertionError(candidate_summary)
    if noop_counts["snapshot_created_false"] != 2:
        raise AssertionError(candidate_summary)
    if len(runtime_event_logs) != 3:
        raise AssertionError(runtime_event_logs)
    if len(copy_summary_logs) != 1:
        raise AssertionError(copy_summary_logs)
    applied_events = [
        event
        for event in candidate_events
        if event["runtime_policy_state"] == "applied_candidate"
    ]
    fallback_events = [
        event
        for event in candidate_events
        if event["runtime_policy_state"] == "fallback_candidate"
    ]
    if len(applied_events) != 1 or len(fallback_events) != 2:
        raise AssertionError(candidate_events)
    applied_event = applied_events[0]
    if applied_event["snapshot_created"] is not True:
        raise AssertionError(applied_event)
    if applied_event["host_backup_copy_candidate"] is not True:
        raise AssertionError(applied_event)
    if applied_event["host_backup_copy_executed"] is not True:
        raise AssertionError(applied_event)
    if applied_event["copy_equal"] is not True:
        raise AssertionError(applied_event)
    if applied_event["source_mutated"] is not False:
        raise AssertionError(applied_event)
    if applied_event["runtime_writeback"] is not False:
        raise AssertionError(applied_event)
    for fallback_event in fallback_events:
        if fallback_event["snapshot_created"] is not False:
            raise AssertionError(fallback_event)
        if fallback_event["host_backup_copy_candidate"] is not False:
            raise AssertionError(fallback_event)
        if fallback_event["host_backup_copy_executed"] is not False:
            raise AssertionError(fallback_event)
        if fallback_event["fallback_candidate_noop_guard"] is not True:
            raise AssertionError(fallback_event)
        if fallback_event["runtime_writeback"] is not False:
            raise AssertionError(fallback_event)
    if copy_candidate_summary["total_candidate_events"] != 3:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["applied_candidate_count"] != 1:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["fallback_candidate_count"] != 2:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["snapshot_created_count"] != 1:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["snapshot_skipped_count"] != 2:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["host_backup_copy_candidate_count"] != 1:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["host_backup_copy_executed_count"] != 1:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["host_backup_copy_skipped_count"] != 2:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["copy_equal_true_count"] != 1:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["copy_equal_false_count"] != 2:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["source_mutated_true_count"] != 0:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["source_mutated_false_count"] != 3:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["fallback_candidate_noop_guard_count"] != 2:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["attention_override_true_count"] != 0:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["attention_override_false_count"] != 3:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["kv_cache_mutation_true_count"] != 0:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["kv_cache_mutation_false_count"] != 3:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["runtime_writeback_true_count"] != 0:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["runtime_writeback_false_count"] != 3:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["scheduler_policy_noop_false_count"] != 0:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["scheduler_policy_noop_true_count"] != 3:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["skipped_reason_counts"] != {
        "fallback_candidate_noop_guard": 2,
    }:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["policy_state_counts"] != {
        "applied_candidate": 1,
        "fallback_candidate": 2,
    }:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["per_layer_counts"]["0"][
        "total_candidate_events"
    ] != 3:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["per_request_counts"]["rid-applied-a"][
        "host_backup_copy_executed_count"
    ] != 1:
        raise AssertionError(copy_candidate_summary)
    if copy_candidate_summary["per_request_counts"]["rid-fallback-a"][
        "fallback_candidate_noop_guard_count"
    ] != 1:
        raise AssertionError(copy_candidate_summary)

    print("relaykv_runtime_policy_smoke: ok")
    print("relaykv_runtime_policy_event_example=" + runtime_event_logs[0])
    print("relaykv_host_backup_copy_candidate_summary_log=" + copy_summary_logs[0])
    print("relaykv_policy_summary=" + json.dumps(policy_summary, sort_keys=True))
    print(
        "relaykv_candidate_event_summary="
        + json.dumps(candidate_summary, sort_keys=True)
    )
    print(
        "relaykv_host_backup_copy_candidate_summary="
        + json.dumps(copy_candidate_summary, sort_keys=True)
    )


if __name__ == "__main__":
    main()
