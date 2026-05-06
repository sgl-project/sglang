from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
)


class _PoisonObject:
    def __init__(self) -> None:
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False
        self.iter_called = False
        self.len_called = False
        self.repr_called = False

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PoisonObject":
        return self

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def numpy(self) -> None:
        self.numpy_called = True
        raise AssertionError("numpy() must not be called")

    def __iter__(self):
        self.iter_called = True
        raise AssertionError("__iter__() must not be called")

    def __len__(self) -> int:
        self.len_called = True
        raise AssertionError("__len__() must not be called")

    def __repr__(self) -> str:
        self.repr_called = True
        raise AssertionError("__repr__() must not be called")

    @property
    def touched(self) -> bool:
        return any(
            (
                self.cpu_called,
                self.tolist_called,
                self.item_called,
                self.numpy_called,
                self.iter_called,
                self.len_called,
                self.repr_called,
            )
        )


class _Box:
    pass


def _req_to_token_span(
    block_id: int,
    token_start: int,
    token_end: int,
    req_to_token_entries: list[Any],
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "token_start": token_start,
        "token_end": token_end,
        "token_count": token_end - token_start,
        "req_to_token_entries": list(req_to_token_entries),
        "entry_count": len(req_to_token_entries),
    }


def _valid_payload(
    request_id: str = "req-a",
    *,
    spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_spans = spans if spans is not None else [_req_to_token_span(101, 0, 3, [10, 11, 12])]
    return {
        "event_type": "relaykv_req_to_token_resolution_result",
        "resolution_state": "req_to_token_resolved",
        "request_id": request_id,
        "layer_id": 14,
        "kv_head_group": 2,
        "kv_class": "FULL",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": f"engine-{request_id}",
        "logical_sequence_id": f"seq-{request_id}",
        "decision_state": "req_to_token_resolved",
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": {"req_pool_idx": 7},
        "engine_block_ref": {"req_pool_idx": 7},
        "full_kv_req_to_token_spans": list(resolved_spans),
        "kv_pool_read": False,
        "kv_snapshot": False,
        "tensor_read": False,
        "attention_comparison_executed": False,
        "attention_override": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
    }


def _invalid_payload() -> dict[str, Any]:
    return {
        "event_type": "wrong_event",
        "resolution_state": "blocked",
        "full_kv_req_to_token_spans": [{"block_id": 1}],
    }


def _assert_zero_safety(summary: dict[str, Any]) -> None:
    for key in (
        "req_to_token_read_count",
        "actual_req_to_token_pool_read_count",
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


def _bridge_result(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
    explicit_payloads: Any = None,
    bridge_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        forward_batch=forward_batch,
        model_runner=model_runner,
        explicit_payloads=explicit_payloads,
        bridge_enabled=bridge_enabled,
    )
    if len(results) != 1:
        raise AssertionError(results)
    summary = summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(results)
    _assert_zero_safety(summary)
    return results[0], summary


def _assert_bridge_off_blocked() -> None:
    payload, summary = _bridge_result(
        explicit_payloads=[_valid_payload()],
        bridge_enabled=False,
    )
    if payload["bridge_state"] != "blocked":
        raise AssertionError(payload)
    if payload["blocked_reason"] != "bridge_not_enabled":
        raise AssertionError(payload)
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)


def _assert_explicit_payload_bridged() -> None:
    inputs = [_valid_payload("req-explicit")]
    before = copy.deepcopy(inputs)
    payload, summary = _bridge_result(explicit_payloads=inputs, bridge_enabled=True)
    if inputs != before:
        raise AssertionError("explicit payloads mutated")
    if payload["bridge_state"] != "bridged":
        raise AssertionError(payload)
    if payload["bridge_source_path"] != "explicit_payloads":
        raise AssertionError(payload)
    if payload["valid_payload_count"] != 1 or payload["blocked_payload_count"] != 0:
        raise AssertionError(payload)
    if summary["bridged_count"] != 1:
        raise AssertionError(summary)


def _assert_attr_sources_bridged() -> None:
    for attr_name, owner_name in (
        ("relaykv_req_to_token_resolution_results", "forward_batch"),
        ("relaykv_req_to_token_resolution_payloads", "forward_batch"),
        ("relaykv_req_to_token_resolution_results", "model_runner"),
        ("relaykv_req_to_token_resolution_payloads", "model_runner"),
    ):
        owner = _Box()
        setattr(owner, attr_name, [_valid_payload(f"{owner_name}-{attr_name}")])
        kwargs = {"bridge_enabled": True}
        if owner_name == "forward_batch":
            kwargs["forward_batch"] = owner
        else:
            kwargs["model_runner"] = owner
        payload, _ = _bridge_result(**kwargs)
        if payload["bridge_state"] != "bridged":
            raise AssertionError(payload)
        if payload["bridge_source_path"] != f"{owner_name}.{attr_name}":
            raise AssertionError(payload)


def _assert_source_priority() -> None:
    forward_batch = _Box()
    forward_batch.relaykv_req_to_token_resolution_results = [_invalid_payload()]
    payload, _ = _bridge_result(
        forward_batch=forward_batch,
        explicit_payloads=[_valid_payload("req-priority")],
        bridge_enabled=True,
    )
    if payload["bridge_source_path"] != "explicit_payloads":
        raise AssertionError(payload)
    if payload["valid_payload_count"] != 1:
        raise AssertionError(payload)


def _assert_missing_empty_invalid_blocked() -> None:
    payload, _ = _bridge_result(bridge_enabled=True)
    if payload["blocked_reason"] != "bridge_source_missing":
        raise AssertionError(payload)

    payload, _ = _bridge_result(explicit_payloads=[], bridge_enabled=True)
    if payload["blocked_reason"] != "bridge_source_empty":
        raise AssertionError(payload)

    payload, _ = _bridge_result(explicit_payloads=_invalid_payload(), bridge_enabled=True)
    if payload["blocked_reason"] != "bridge_source_not_list_or_tuple":
        raise AssertionError(payload)

    payload, _ = _bridge_result(explicit_payloads=[_invalid_payload()], bridge_enabled=True)
    if payload["blocked_reason"] != "bridge_payload_invalid":
        raise AssertionError(payload)


def _assert_mixed_valid_invalid() -> None:
    payload, summary = _bridge_result(
        explicit_payloads=[_valid_payload("req-valid"), _invalid_payload()],
        bridge_enabled=True,
    )
    if payload["bridge_state"] != "bridged":
        raise AssertionError(payload)
    if payload["valid_payload_count"] != 1 or payload["blocked_payload_count"] != 1:
        raise AssertionError(payload)
    if summary["blocked_payload_count"] != 1:
        raise AssertionError(summary)


def _assert_tuple_accepted() -> None:
    payload, _ = _bridge_result(
        explicit_payloads=(_valid_payload("req-tuple"),),
        bridge_enabled=True,
    )
    if payload["bridge_state"] != "bridged":
        raise AssertionError(payload)


def _assert_poison_untouched() -> None:
    poison = _PoisonObject()
    forward_batch = _Box()
    forward_batch.unrelated = poison
    forward_batch.relaykv_req_to_token_resolution_results = [_valid_payload("req-poison")]
    before = copy.deepcopy(forward_batch.relaykv_req_to_token_resolution_results)
    payload, _ = _bridge_result(forward_batch=forward_batch, bridge_enabled=True)
    if poison.touched:
        raise AssertionError("poison object touched")
    if forward_batch.relaykv_req_to_token_resolution_results != before:
        raise AssertionError("source object mutated")
    if payload["source_mutated"] is not False:
        raise AssertionError(payload)


def _assert_bridge_feeds_live_index_read() -> None:
    bridge_payload, _ = _bridge_result(
        explicit_payloads=[_valid_payload("req-live")],
        bridge_enabled=True,
    )
    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        bridge_payload["req_to_token_resolution_payloads"],
        token_to_kv_pool_object={10: 100, 11: 101, 12: 102},
        read_token_to_kv_pool_index=True,
        source_path="bridge.explicit_payloads",
    )
    if len(live_results) != 1:
        raise AssertionError(live_results)
    live_result = live_results[0]
    if live_result["resolution_state"] != "physical_kv_index_resolved":
        raise AssertionError(live_result)
    live_summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        live_results
    )
    if live_summary["token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(live_summary)
    if live_summary["actual_token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(live_summary)
    if live_summary["live_token_to_kv_pool_index_read_count"] <= 0:
        raise AssertionError(live_summary)
    _assert_zero_safety(live_summary)


def main() -> None:
    _assert_bridge_off_blocked()
    _assert_explicit_payload_bridged()
    _assert_attr_sources_bridged()
    _assert_source_priority()
    _assert_missing_empty_invalid_blocked()
    _assert_mixed_valid_invalid()
    _assert_tuple_accepted()
    _assert_poison_untouched()
    _assert_bridge_feeds_live_index_read()
    print("relaykv_req_to_token_resolution_payload_bridge_smoke=pass")
    print(
        "relaykv_req_to_token_resolution_payload_bridge_summary="
        + json.dumps(
            {
                "bridge_off": "blocked",
                "explicit": "bridged",
                "attribute_sources": "bridged",
                "priority": "explicit_wins",
                "missing_empty_invalid": "blocked",
                "mixed": "valid_preserved_invalid_counted",
                "bridge_to_live_read": "resolved",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
