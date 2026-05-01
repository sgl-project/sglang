from __future__ import annotations

import json

import torch

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.memory import (
    estimate_host_backup_shadow_for_plan,
    estimate_kv_memory_for_plan,
    estimate_kv_memory_from_metadata,
    observe_kv_layout_for_host_backup,
    observe_request_kv_pool_mapping,
    validate_shadow_log_schema,
)
from sglang.srt.relaykv.metrics import policy_event_payload, summarize_policy_events


class _FakeModelConfig:
    num_hidden_layers = 28
    num_key_value_heads = 2
    num_attention_heads = 16
    hidden_size = 2048
    head_dim = 128


class _FakeKVCache:
    dtype = torch.bfloat16
    device = "cuda:0"

    def __init__(self):
        self.k_buffer = [torch.zeros((2536, 2, 128), dtype=torch.bfloat16)]
        self.v_buffer = [torch.zeros((2536, 2, 128), dtype=torch.bfloat16)]


class _FakeAllocator:
    page_size = 1

    def __init__(self):
        self._kvcache = _FakeKVCache()

    def get_kvcache(self):
        return self._kvcache


class _FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.arange(0, 32, dtype=torch.int32).view(4, 8)


def _assert_close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-3:
        raise AssertionError(f"expected {expected}, got {actual}")


def _assert_budget_plan(
    *,
    available_mib: float,
    expected_working_tokens: int,
    recent_window: int,
    anchor_blocks: int,
    budget_block_size: int,
    retrieval_top_k: int,
    expected_anchor_tokens: int,
    expected_retrieval_tokens: int,
    expected_retrieval_blocks: int,
    expected_top_k_effective: int,
    expected_overflow: bool,
    expected_reason: str,
) -> None:
    plan = make_shadow_plan(
        100000,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            available_kv_budget_mib=available_mib,
            recent_window=recent_window,
            anchor_blocks=anchor_blocks,
            budget_block_size=budget_block_size,
            retrieval_top_k=retrieval_top_k,
        ),
        kv_bytes_per_token=28672,
    )
    if plan.kv_working_budget_tokens != expected_working_tokens:
        raise AssertionError(plan)
    if plan.kv_working_budget_source != "estimated_from_available_kv_budget_mib":
        raise AssertionError(plan)
    if plan.recent_window_tokens != recent_window:
        raise AssertionError(plan)
    if plan.budget_block_size != budget_block_size:
        raise AssertionError(plan)
    if plan.anchor_blocks != anchor_blocks:
        raise AssertionError(plan)
    if plan.anchor_budget_tokens != expected_anchor_tokens:
        raise AssertionError(plan)
    if plan.retrieval_budget_tokens != expected_retrieval_tokens:
        raise AssertionError(plan)
    if plan.retrieval_block_budget != expected_retrieval_blocks:
        raise AssertionError(plan)
    if plan.retrieval_top_k_effective != expected_top_k_effective:
        raise AssertionError(plan)
    if plan.budget_overflow is not expected_overflow:
        raise AssertionError(plan)
    if plan.budget_policy_reason != expected_reason:
        raise AssertionError(plan)


def _assert_small_budget_plan() -> None:
    plan = make_shadow_plan(
        100000,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            kv_working_budget_tokens=1024,
            recent_window=768,
            anchor_blocks=4,
            budget_block_size=128,
            retrieval_top_k=8,
        ),
        kv_bytes_per_token=28672,
    )
    if plan.anchor_budget_tokens != 256:
        raise AssertionError(plan)
    if plan.retrieval_budget_tokens != 0:
        raise AssertionError(plan)
    if plan.retrieval_block_budget != 0:
        raise AssertionError(plan)
    if plan.retrieval_top_k_effective != 0:
        raise AssertionError(plan)
    if plan.budget_overflow is not True:
        raise AssertionError(plan)
    if plan.budget_policy_reason != "anchor_budget_clipped_after_recent_window":
        raise AssertionError(plan)


def main() -> None:
    plan = make_shadow_plan(
        2535,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            resident_budget_tokens=1024,
            recent_window=768,
            anchor_pages=4,
        ),
    )
    estimate = estimate_kv_memory_from_metadata(
        seq_len=plan.seq_len,
        planned_resident_tokens=plan.planned_resident_tokens,
        planned_cold_tokens=plan.planned_cold_tokens,
        num_layers=28,
        num_key_value_heads=2,
        head_dim=128,
        kv_dtype_bytes=2,
    )

    if estimate.num_layers != 28:
        raise AssertionError(estimate)
    if estimate.kv_bytes_per_token != 28672:
        raise AssertionError(estimate)
    _assert_close(estimate.planned_resident_kv_mib, 28.0)
    _assert_close(estimate.planned_cold_kv_mib, 41.316)
    if plan.runtime_policy_state != "fallback_candidate":
        raise AssertionError(plan)
    if plan.full_kv_fits is not False:
        raise AssertionError(plan)
    if plan.available_kv_budget_tokens != 1024:
        raise AssertionError(plan)
    if plan.estimated_full_kv_tokens != 2535:
        raise AssertionError(plan)
    if plan.budget_pressure is not True:
        raise AssertionError(plan)
    if plan.risk_level != "high":
        raise AssertionError(plan)
    if "low_coverage_ratio" not in plan.policy_reason:
        raise AssertionError(plan)
    if plan.recent_window != 768:
        raise AssertionError(plan)
    if plan.prompt_risk != "unknown":
        raise AssertionError(plan)

    estimate_from_model = estimate_kv_memory_for_plan(
        plan,
        model_config=_FakeModelConfig(),
        kv_dtype="torch.bfloat16",
    )
    host_backup_estimate = estimate_host_backup_shadow_for_plan(
        plan,
        memory_estimate=estimate_from_model,
        host_backup_shadow=True,
        host_backup_max_mib=0.0,
        host_backup_dry_copy=True,
    )
    if host_backup_estimate.host_backup_candidate_tokens != 1511:
        raise AssertionError(host_backup_estimate)
    _assert_close(host_backup_estimate.host_backup_candidate_kv_mib, 41.316)
    if host_backup_estimate.host_backup_budget_ok is not True:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_would_copy is not False:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.resident_anchor_ranges != [[0, 4]]:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.resident_recent_ranges != [[1767, 2535]]:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.cold_candidate_ranges != [[4, 1767]]:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_copy_target_ranges != [[4, 1767]]:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_copy_target_tokens != 1763:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_copy_target_reason != "metadata_only_no_tensor_copy_range_token_mismatch":
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_dry_copy_candidate is not False:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_dry_copy_guard_ok is not False:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_dry_copy_would_run is not False:
        raise AssertionError(host_backup_estimate)
    if (
        host_backup_estimate.host_backup_dry_copy_reason
        != "runtime_policy_fallback_candidate_noop"
    ):
        raise AssertionError(host_backup_estimate)
    layout_observation = observe_kv_layout_for_host_backup(
        token_to_kv_pool_allocator=_FakeAllocator(),
        req_to_token_pool=object(),
        request_id="smoke-rid",
        seq_len=plan.seq_len,
        copy_target_ranges=host_backup_estimate.host_backup_copy_target_ranges,
    )
    if layout_observation.kv_layout_observed is not True:
        raise AssertionError(layout_observation)
    if layout_observation.kv_layout_k_shape != [2536, 2, 128]:
        raise AssertionError(layout_observation)
    if layout_observation.kv_layout_v_shape != [2536, 2, 128]:
        raise AssertionError(layout_observation)
    if layout_observation.kv_layout_dtype != "torch.bfloat16":
        raise AssertionError(layout_observation)
    if layout_observation.kv_layout_device != "cpu":
        raise AssertionError(layout_observation)
    if layout_observation.kv_layout_range_mapping_supported is not True:
        raise AssertionError(layout_observation)
    mapping_observation = observe_request_kv_pool_mapping(
        req_to_token_pool=_FakeReqToTokenPool(),
        request_pool_idx=1,
        seq_len=8,
        observed_token_count=8,
        cold_candidate_ranges=[[2, 6]],
    )
    if mapping_observation.kv_pool_mapping_observed is not True:
        raise AssertionError(mapping_observation)
    if mapping_observation.kv_pool_mapping_shape != [4, 8]:
        raise AssertionError(mapping_observation)
    if mapping_observation.request_pool_indices_count != 8:
        raise AssertionError(mapping_observation)
    if mapping_observation.request_pool_indices_preview_head != [8, 9, 10, 11, 12, 13, 14, 15]:
        raise AssertionError(mapping_observation)
    if mapping_observation.cold_range_pool_indices_preview != [10, 11, 12, 13]:
        raise AssertionError(mapping_observation)
    if mapping_observation.cold_range_pool_indices_count != 4:
        raise AssertionError(mapping_observation)
    if mapping_observation.cold_range_pool_mapping_supported is not True:
        raise AssertionError(mapping_observation)
    if mapping_observation.mapping_valid_count != 4:
        raise AssertionError(mapping_observation)
    if mapping_observation.mapping_zero_count != 0:
        raise AssertionError(mapping_observation)
    if mapping_observation.mapping_invalid_count != 0:
        raise AssertionError(mapping_observation)
    if mapping_observation.mapping_ready_for_copy is not True:
        raise AssertionError(mapping_observation)
    if mapping_observation.prefill_pending_tokens != 0:
        raise AssertionError(mapping_observation)
    if mapping_observation.prefill_complete_for_request is not True:
        raise AssertionError(mapping_observation)
    if mapping_observation.host_backup_dry_copy_final_guard_ok is not True:
        raise AssertionError(mapping_observation)
    if (
        mapping_observation.host_backup_dry_copy_final_guard_reason
        != "ready_for_execution_metadata_only"
    ):
        raise AssertionError(mapping_observation)
    payload = plan.to_log_dict()
    payload.update(estimate_from_model.to_log_dict())
    payload.update({"host_backup_planned": True, **host_backup_estimate.to_log_dict()})
    payload.update(layout_observation.to_log_dict())
    payload.update(mapping_observation.to_log_dict())
    validate_shadow_log_schema(payload)

    _assert_budget_plan(
        available_mib=512.0,
        expected_working_tokens=18724,
        recent_window=768,
        anchor_blocks=4,
        budget_block_size=128,
        retrieval_top_k=8,
        expected_anchor_tokens=512,
        expected_retrieval_tokens=17444,
        expected_retrieval_blocks=136,
        expected_top_k_effective=8,
        expected_overflow=False,
        expected_reason="estimated_from_available_kv_budget_mib",
    )
    _assert_budget_plan(
        available_mib=1024.0,
        expected_working_tokens=37449,
        recent_window=768,
        anchor_blocks=4,
        budget_block_size=128,
        retrieval_top_k=8,
        expected_anchor_tokens=512,
        expected_retrieval_tokens=36169,
        expected_retrieval_blocks=282,
        expected_top_k_effective=8,
        expected_overflow=False,
        expected_reason="estimated_from_available_kv_budget_mib",
    )
    _assert_budget_plan(
        available_mib=2048.0,
        expected_working_tokens=74898,
        recent_window=768,
        anchor_blocks=4,
        budget_block_size=128,
        retrieval_top_k=8,
        expected_anchor_tokens=512,
        expected_retrieval_tokens=73618,
        expected_retrieval_blocks=575,
        expected_top_k_effective=8,
        expected_overflow=False,
        expected_reason="estimated_from_available_kv_budget_mib",
    )
    _assert_small_budget_plan()

    explicit_plan = make_shadow_plan(
        4096,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            available_kv_budget_mib=512.0,
            kv_working_budget_tokens=2048,
            recent_window=1024,
            anchor_blocks=64,
            budget_block_size=128,
            retrieval_top_k=2048,
        ),
        kv_bytes_per_token=28672,
    )
    if explicit_plan.kv_working_budget_tokens != 2048:
        raise AssertionError(explicit_plan)
    if explicit_plan.kv_working_budget_source != "explicit_working_budget_tokens":
        raise AssertionError(explicit_plan)
    if explicit_plan.anchor_blocks != 64:
        raise AssertionError(explicit_plan)
    if explicit_plan.anchor_budget_tokens != 1024:
        raise AssertionError(explicit_plan)
    if explicit_plan.retrieval_budget_tokens != 0:
        raise AssertionError(explicit_plan)
    if explicit_plan.retrieval_top_k_effective != 0:
        raise AssertionError(explicit_plan)
    if explicit_plan.runtime_policy_state != "fallback_candidate":
        raise AssertionError(explicit_plan)
    if explicit_plan.risk_level != "high":
        raise AssertionError(explicit_plan)

    full_fit_plan = make_shadow_plan(
        1024,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            kv_working_budget_tokens=2048,
            recent_window=512,
            anchor_blocks=2,
            budget_block_size=128,
            retrieval_top_k=4,
        ),
    )
    if full_fit_plan.runtime_policy_state != "off":
        raise AssertionError(full_fit_plan)
    if full_fit_plan.full_kv_fits is not True:
        raise AssertionError(full_fit_plan)
    if full_fit_plan.budget_pressure is not False:
        raise AssertionError(full_fit_plan)
    if full_fit_plan.policy_reason != "full_kv_fits":
        raise AssertionError(full_fit_plan)

    shadow_plan = make_shadow_plan(
        1024,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            available_kv_budget_mib=512.0,
            recent_window=256,
            anchor_blocks=1,
            budget_block_size=128,
            retrieval_top_k=2,
        ),
    )
    if shadow_plan.runtime_policy_state != "shadow":
        raise AssertionError(shadow_plan)

    applied_plan = make_shadow_plan(
        4096,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            kv_working_budget_tokens=3072,
            recent_window=512,
            anchor_blocks=2,
            budget_block_size=128,
            retrieval_top_k=4,
        ),
    )
    if applied_plan.runtime_policy_state != "applied_candidate":
        raise AssertionError(applied_plan)
    if applied_plan.risk_level != "normal":
        raise AssertionError(applied_plan)
    applied_memory_estimate = estimate_kv_memory_for_plan(
        applied_plan,
        model_config=_FakeModelConfig(),
        kv_dtype="torch.bfloat16",
    )
    applied_host_backup_estimate = estimate_host_backup_shadow_for_plan(
        applied_plan,
        memory_estimate=applied_memory_estimate,
        host_backup_shadow=True,
        host_backup_max_mib=0.0,
        host_backup_dry_copy=True,
    )
    if applied_host_backup_estimate.host_backup_dry_copy_candidate is not True:
        raise AssertionError(applied_host_backup_estimate)
    if applied_host_backup_estimate.host_backup_dry_copy_guard_ok is not True:
        raise AssertionError(applied_host_backup_estimate)
    if applied_host_backup_estimate.host_backup_dry_copy_would_run is not True:
        raise AssertionError(applied_host_backup_estimate)
    if (
        applied_host_backup_estimate.host_backup_dry_copy_reason
        != "dry_copy_candidate_metadata_only_no_tensor_copy"
    ):
        raise AssertionError(applied_host_backup_estimate)

    late_layer_plan = make_shadow_plan(
        4096,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            kv_working_budget_tokens=4096,
            recent_window=1024,
            anchor_blocks=4,
            budget_block_size=128,
            retrieval_top_k=4,
        ),
        layer_idx=27,
        total_layers=28,
    )
    if late_layer_plan.runtime_policy_state != "off":
        raise AssertionError(late_layer_plan)
    if late_layer_plan.risk_level != "high":
        raise AssertionError(late_layer_plan)
    if late_layer_plan.layer_idx != 27:
        raise AssertionError(late_layer_plan)

    summary = summarize_policy_events(
        [full_fit_plan, shadow_plan, applied_plan, plan, explicit_plan, late_layer_plan]
    )
    if summary["relaykv_policy_off_count"] != 2:
        raise AssertionError(summary)
    if summary["relaykv_policy_shadow_count"] != 1:
        raise AssertionError(summary)
    if summary["relaykv_policy_applied_candidate_count"] != 1:
        raise AssertionError(summary)
    if summary["relaykv_policy_fallback_candidate_count"] != 2:
        raise AssertionError(summary)
    fallback_event = policy_event_payload(
        plan,
        extra={
            "phase": "prefill",
            "request_index": 0,
            "scheduler_policy_noop": True,
            "kv_cache_mutation": False,
            "attention_override": False,
            "host_backup_copy": False,
        },
    )
    if fallback_event["runtime_policy_action"] != "log_only_noop":
        raise AssertionError(fallback_event)
    if fallback_event["dry_copy_candidate"] is not False:
        raise AssertionError(fallback_event)
    if fallback_event["fallback_candidate_noop_guard"] is not True:
        raise AssertionError(fallback_event)
    if fallback_event["host_backup_copy"] is not False:
        raise AssertionError(fallback_event)
    if fallback_event["host_backup_copy_executed"] is not False:
        raise AssertionError(fallback_event)
    if fallback_event["host_backup_copy_skipped_reason"] != "fallback_candidate_noop_guard":
        raise AssertionError(fallback_event)
    if fallback_event["kv_cache_mutation"] is not False:
        raise AssertionError(fallback_event)
    applied_event = policy_event_payload(applied_plan)
    if applied_event["applied_candidate_log_only"] is not True:
        raise AssertionError(applied_event)
    if applied_event["dry_copy_candidate"] is not True:
        raise AssertionError(applied_event)
    if applied_event["runtime_policy_action"] != "dry_copy_candidate_log_only":
        raise AssertionError(applied_event)
    if applied_event["host_backup_copy"] is not False:
        raise AssertionError(applied_event)
    if applied_event["host_backup_copy_executed"] is not False:
        raise AssertionError(applied_event)

    print("relaykv_memory_smoke: ok")
    print(payload)
    print("relaykv_runtime_policy_event=" + json.dumps(fallback_event, sort_keys=True))
    print("relaykv_policy_summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
