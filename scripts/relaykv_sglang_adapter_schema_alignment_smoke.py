from __future__ import annotations

import copy
import json

from sglang.srt.relaykv.metrics import (
    normalize_relaykv_sglang_adapter_schema_for_smoke,
    summarize_relaykv_sglang_adapter_schema_alignment_for_smoke,
)


def _phase_4_8_payloads() -> list[dict[str, object]]:
    return [
        {
            "event_type": "relaykv_kv_index_resolution_plan",
            "resolution_state": "block_span_resolved",
            "resolution_mode": "metadata_only",
            "request_id": "req-4.8.1",
            "req_pool_idx": 7,
            "seq_len": 256,
            "layer_id": 14,
            "kv_class": "FULL",
            "relaykv_working_kv_block_ids": [11],
            "full_kv_block_ids": [10, 11],
            "relaykv_working_block_spans": [
                {
                    "block_id": 11,
                    "token_start": 64,
                    "token_end": 128,
                    "token_count": 64,
                }
            ],
            "full_kv_block_spans": [
                {
                    "block_id": 10,
                    "token_start": 0,
                    "token_end": 64,
                    "token_count": 64,
                },
                {
                    "block_id": 11,
                    "token_start": 64,
                    "token_end": 128,
                    "token_count": 64,
                },
            ],
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read": False,
            "kv_pool_read_count": 0,
            "kv_snapshot": False,
            "kv_snapshot_count": 0,
            "tensor_read": False,
            "tensor_read_count": 0,
            "attention_comparison_executed": False,
            "attention_comparison_executed_count": 0,
            "attention_override": False,
            "attention_override_true_count": 0,
            "runtime_writeback": False,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop": True,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation": False,
            "kv_cache_mutation_true_count": 0,
            "source_mutated": False,
            "source_mutated_true_count": 0,
        },
        {
            "event_type": "relaykv_req_to_token_resolution_result",
            "resolution_state": "blocked",
            "resolution_mode": "readonly_synthetic_table",
            "request_id": "req-4.8.2",
            "req_pool_idx": 8,
            "seq_len": 192,
            "layer_id": 15,
            "full_kv_req_to_token_spans": [
                {
                    "block_id": 21,
                    "token_start": 0,
                    "token_end": 64,
                    "token_count": 64,
                }
            ],
            "blocking_reasons": ["req_to_token_table_missing"],
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read": False,
            "kv_pool_read_count": 0,
            "kv_snapshot": False,
            "kv_snapshot_count": 0,
            "tensor_read": False,
            "tensor_read_count": 0,
            "attention_comparison_executed": False,
            "attention_comparison_executed_count": 0,
            "attention_override": False,
            "attention_override_true_count": 0,
            "runtime_writeback": False,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop": True,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation": False,
            "kv_cache_mutation_true_count": 0,
            "source_mutated": False,
            "source_mutated_true_count": 0,
        },
        {
            "event_type": "relaykv_req_to_token_readonly_adapter_payload",
            "adapter_state": "adapter_payload_ready",
            "adapter_mode": "readonly_bounded_preview",
            "request_id": "req-4.8.3",
            "req_pool_idx": 9,
            "seq_len": 320,
            "layer_id": 16,
            "req_to_token_source": "synthetic_req_to_token_table",
            "req_to_token_backing_type": "dict",
            "req_to_token_entries_preview": [1000, 1001, 1002, 1003],
            "preview_entries": [1000, 1001, 1002, 1003],
            "requested_block_count": 1,
            "requested_token_count": 64,
            "read_token_count": 64,
            "preview_entry_count": 4,
            "token_span": [0, 64],
            "logical_block_id": 31,
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read": False,
            "kv_pool_read_count": 0,
            "kv_snapshot": False,
            "kv_snapshot_count": 0,
            "tensor_read": False,
            "tensor_read_count": 0,
            "attention_comparison_executed": False,
            "attention_comparison_executed_count": 0,
            "attention_override": False,
            "attention_override_true_count": 0,
            "runtime_writeback": False,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop": True,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation": False,
            "kv_cache_mutation_true_count": 0,
            "source_mutated": False,
            "source_mutated_true_count": 0,
        },
        {
            "event_type": "relaykv_req_to_token_readonly_adapter_payload",
            "adapter_state": "adapter_payload_ready",
            "adapter_mode": "actual_pool_readonly_bounded_preview",
            "request_id": "req-4.8.4",
            "req_pool_idx": 10,
            "seq_len": 384,
            "layer_id": 17,
            "pool_source_path": "model_runner.req_to_token_pool.req_to_token",
            "req_to_token_source": "actual_pool_attr",
            "req_to_token_backing_type": "list",
            "req_to_token_shape": (12, 384),
            "req_to_token_device": "cuda:0",
            "req_to_token_dtype": "torch.int64",
            "cache_position": None,
            "logical_block_id": 41,
            "token_span": [64, 128],
            "token_to_kv_pool_read": False,
            "token_to_kv_pool_read_count": 0,
            "kv_pool_read": False,
            "kv_pool_read_count": 0,
            "kv_snapshot": False,
            "kv_snapshot_count": 0,
            "tensor_read": False,
            "tensor_read_count": 0,
            "attention_comparison_executed": False,
            "attention_comparison_executed_count": 0,
            "attention_override": False,
            "attention_override_true_count": 0,
            "runtime_writeback": False,
            "runtime_writeback_true_count": 0,
            "scheduler_policy_noop": True,
            "scheduler_policy_noop_false_count": 0,
            "kv_cache_mutation": False,
            "kv_cache_mutation_true_count": 0,
            "source_mutated": False,
            "source_mutated_true_count": 0,
        },
    ]


def _assert_normalized_payloads() -> dict[str, object]:
    payloads = _phase_4_8_payloads()
    before_payloads = copy.deepcopy(payloads)

    normalized_payloads = [
        normalize_relaykv_sglang_adapter_schema_for_smoke(payload)
        for payload in payloads
    ]

    if payloads != before_payloads:
        raise AssertionError("input payloads were mutated")
    if len(normalized_payloads) != 4:
        raise AssertionError(normalized_payloads)

    for original, normalized in zip(payloads, normalized_payloads):
        if normalized is original:
            raise AssertionError("normalized payload reused input dict")
        for key, value in original.items():
            if normalized.get(key) != value:
                raise AssertionError((key, normalized.get(key), value))
        if normalized["engine_name"] != "sglang":
            raise AssertionError(normalized)
        if normalized["adapter_name"] != "sglang":
            raise AssertionError(normalized)
        if normalized["engine_request_id"] != original["request_id"]:
            raise AssertionError(normalized)
        if normalized["logical_sequence_id"] != original["request_id"]:
            raise AssertionError(normalized)
        if normalized["position_check_state"] != "not_checked_metadata_only":
            raise AssertionError(normalized)
        if normalized["attention_mask_mode"] != "unknown":
            raise AssertionError(normalized)
        if normalized["rope_position_consistency"] != "not_checked":
            raise AssertionError(normalized)
        if normalized["decision_state"] is None:
            raise AssertionError(normalized)
        if normalized["adapter_metadata"]["req_pool_idx"] != original["req_pool_idx"]:
            raise AssertionError(normalized)
        engine_block_ref = normalized["engine_block_ref"]
        if not isinstance(engine_block_ref, dict):
            raise AssertionError(normalized)
        if engine_block_ref["req_pool_idx"] != original["req_pool_idx"]:
            raise AssertionError(normalized)
        if engine_block_ref["token_to_kv_pool_index"] is not None:
            raise AssertionError(normalized)

    if normalized_payloads[0]["logical_block_id"] != 11:
        raise AssertionError(normalized_payloads[0])
    if normalized_payloads[0]["token_span"] != [64, 128]:
        raise AssertionError(normalized_payloads[0])

    if normalized_payloads[1]["logical_block_id"] != 21:
        raise AssertionError(normalized_payloads[1])
    if normalized_payloads[1]["token_span"] != [0, 64]:
        raise AssertionError(normalized_payloads[1])
    if normalized_payloads[1]["fallback_reason"] != "req_to_token_table_missing":
        raise AssertionError(normalized_payloads[1])

    if normalized_payloads[2]["engine_block_ref"]["req_to_token_entries_preview"] != [
        1000,
        1001,
        1002,
        1003,
    ]:
        raise AssertionError(normalized_payloads[2])

    if normalized_payloads[3]["adapter_metadata"]["pool_source_path"] != (
        "model_runner.req_to_token_pool.req_to_token"
    ):
        raise AssertionError(normalized_payloads[3])
    if normalized_payloads[3]["engine_block_ref"]["cache_position"] is not None:
        raise AssertionError(normalized_payloads[3])

    summary = summarize_relaykv_sglang_adapter_schema_alignment_for_smoke(payloads)
    if payloads != before_payloads:
        raise AssertionError("summary mutated input payloads")
    if summary["summary_type"] != "relaykv_sglang_adapter_schema_alignment_summary":
        raise AssertionError(summary)
    if summary["total_payload_count"] != 4:
        raise AssertionError(summary)
    if summary["token_span_present_count"] != 4:
        raise AssertionError(summary)
    if summary["logical_block_id_present_count"] != 4:
        raise AssertionError(summary)
    if summary["adapter_metadata_req_pool_idx_count"] != 4:
        raise AssertionError(summary)
    if summary["engine_block_ref_req_pool_idx_count"] != 4:
        raise AssertionError(summary)
    if summary["token_to_kv_pool_index_none_count"] != 4:
        raise AssertionError(summary)
    if summary["position_check_default_count"] != 4:
        raise AssertionError(summary)
    if summary["per_decision_state_counts"]["adapter_payload_ready"] != 2:
        raise AssertionError(summary)
    if summary["per_decision_state_counts"]["block_span_resolved"] != 1:
        raise AssertionError(summary)
    if summary["per_decision_state_counts"]["blocked"] != 1:
        raise AssertionError(summary)

    for key in (
        "token_to_kv_pool_read_count",
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

    return {
        "normalized_payloads": normalized_payloads,
        "summary": summary,
    }


def main() -> None:
    outputs = _assert_normalized_payloads()
    print("relaykv_sglang_adapter_schema_alignment_smoke=pass")
    print(
        "relaykv_sglang_adapter_schema_alignment_summary="
        + json.dumps(outputs["summary"], sort_keys=True)
    )


if __name__ == "__main__":
    main()
