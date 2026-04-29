from __future__ import annotations

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
    if host_backup_estimate.host_backup_dry_copy_guard_ok is not True:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_dry_copy_would_run is not True:
        raise AssertionError(host_backup_estimate)
    if host_backup_estimate.host_backup_dry_copy_reason != "guard_only_no_tensor_copy":
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
    payload = plan.to_log_dict()
    payload.update(estimate_from_model.to_log_dict())
    payload.update({"host_backup_planned": True, **host_backup_estimate.to_log_dict()})
    payload.update(layout_observation.to_log_dict())
    payload.update(mapping_observation.to_log_dict())
    validate_shadow_log_schema(payload)

    print("relaykv_memory_smoke: ok")
    print(payload)


if __name__ == "__main__":
    main()
