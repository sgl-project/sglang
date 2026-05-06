from __future__ import annotations

import json
from dataclasses import asdict

from relaykv_sglang_adapter_schema_alignment_smoke import _assert_normalized_payloads
from sglang.srt.relaykv.metrics import (
    RelayKVBlockMeta,
    RelayKVGroupMeta,
    RelayKVPolicyDecision,
)


def _assert_block_meta_defaults_and_reserved_fields() -> None:
    default_meta = RelayKVBlockMeta()
    if asdict(default_meta) != {
        "query_block_score": None,
        "middle_layer_query_block_score": None,
        "retrieval_criticality_rank": None,
        "gather_anchor_score": None,
        "aggregate_retrieval_score": None,
        "massive_qk_score": None,
        "working_set_stability_score": None,
        "last_retrieved_step": None,
        "retrieval_reuse_count": None,
    }:
        raise AssertionError(default_meta)

    reserved_meta = RelayKVBlockMeta(
        query_block_score=0.25,
        middle_layer_query_block_score=0.5,
        retrieval_criticality_rank=3,
        gather_anchor_score=0.75,
        aggregate_retrieval_score=1.25,
        massive_qk_score=2.5,
        working_set_stability_score=0.9,
        last_retrieved_step=12,
        retrieval_reuse_count=4,
    )
    payload = reserved_meta.to_dict()
    if payload["retrieval_criticality_rank"] != 3:
        raise AssertionError(payload)
    json.dumps(payload, sort_keys=True)


def _assert_group_meta_defaults_and_reserved_fields() -> None:
    default_group = RelayKVGroupMeta(layer_id=7, kv_head_group=2)
    if default_group.to_dict() != {
        "layer_id": 7,
        "kv_head_group": 2,
        "retrieval_head_score": None,
        "query_dependent_group_score": None,
        "group_budget_bonus": None,
    }:
        raise AssertionError(default_group)

    reserved_group = RelayKVGroupMeta(
        layer_id=11,
        kv_head_group=5,
        retrieval_head_score=0.42,
        query_dependent_group_score=0.61,
        group_budget_bonus=2,
    )
    payload = asdict(reserved_group)
    if payload["group_budget_bonus"] != 2:
        raise AssertionError(payload)
    json.dumps(payload, sort_keys=True)


def _assert_policy_decision_defaults_and_mutable_isolation() -> None:
    first = RelayKVPolicyDecision()
    second = RelayKVPolicyDecision()

    if first.temporal_reuse_enabled:
        raise AssertionError(first)
    if first.reused_block_ids or first.newly_retrieved_block_ids:
        raise AssertionError(first)
    if first.selection_reason_counts:
        raise AssertionError(first)
    if first.selection_stability_ratio is not None:
        raise AssertionError(first)

    first.reused_block_ids.append(10)
    first.newly_retrieved_block_ids.append(11)
    first.selection_reason_counts["reserved_only"] = 1

    if second.reused_block_ids or second.newly_retrieved_block_ids:
        raise AssertionError((first, second))
    if second.selection_reason_counts:
        raise AssertionError((first, second))

    reserved_policy = RelayKVPolicyDecision(
        temporal_reuse_enabled=True,
        reused_block_ids=[101, 102],
        newly_retrieved_block_ids=[103],
        selection_stability_ratio=0.5,
        selection_reason_counts={"reuse": 2, "retrieve": 1},
    )
    payload = reserved_policy.to_dict()
    if payload["selection_reason_counts"]["reuse"] != 2:
        raise AssertionError(payload)
    json.dumps(payload, sort_keys=True)


def main() -> None:
    _assert_block_meta_defaults_and_reserved_fields()
    _assert_group_meta_defaults_and_reserved_fields()
    _assert_policy_decision_defaults_and_mutable_isolation()
    _assert_normalized_payloads()
    print("relaykv_reserved_retrieval_metadata_fields_smoke=pass")


if __name__ == "__main__":
    main()
