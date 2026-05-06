from __future__ import annotations

import copy
import json
import os
import types
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    build_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke,
    build_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke,
    build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke,
    run_model_runner_runtime_req_to_token_payload_production_hook_for_smoke,
    summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke,
    summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke,
    summarize_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke,
    summarize_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke,
    summarize_relaykv_runtime_req_to_token_resolution_payloads_for_smoke,
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
        self.getitem_called = False

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

    def __getitem__(self, index: Any) -> Any:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

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
                self.getitem_called,
            )
        )


def _runtime_observation_payload(
    request_id: str = "req-a",
    *,
    token_span: list[int] | None = None,
    seq_len: int | None = 4,
    poison: _PoisonObject | None = None,
) -> dict[str, Any]:
    adapter_metadata: dict[str, Any] = {"runtime_observation_marker": request_id}
    if seq_len is not None:
        adapter_metadata["seq_len"] = seq_len
    if poison is not None:
        adapter_metadata["poison"] = poison
    payload = {
        "event_type": "relaykv_runtime_observation_result",
        "engine_name": "sglang",
        "adapter_name": "sglang",
        "engine_request_id": f"engine-{request_id}",
        "logical_sequence_id": f"seq-{request_id}",
        "request_id": request_id,
        "logical_block_id": 101,
        "layer_id": 14,
        "kv_head_group": 2,
        "kv_class": "FULL",
        "position_check_state": "not_checked_metadata_only",
        "attention_mask_mode": "unknown",
        "rope_position_consistency": "not_checked",
        "adapter_metadata": adapter_metadata,
        "engine_block_ref": {"req_pool_idx": 7},
        "source_mutated": False,
    }
    if token_span is not None:
        payload["token_span"] = token_span
    return payload


def _runtime_observation_metadata(
    request_id: str = "req-a",
    *,
    token_span: list[int] | None = None,
    seq_len: int | None = 4,
    layer_id: int | None = 14,
    layer_ids: list[int] | None = None,
    poison: _PoisonObject | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "request_id": request_id,
        "rid": request_id,
        "logical_sequence_id": f"seq-{request_id}",
        "req_pool_idx": 7,
        "seq_len": seq_len,
        "kv_head_group": 2,
        "logical_block_id": 101,
        "adapter_metadata": {"runtime_observation_metadata_marker": request_id},
        "engine_block_ref": {"req_pool_idx": 7},
    }
    if token_span is not None:
        metadata["token_span"] = token_span
    if layer_id is not None:
        metadata["layer_id"] = layer_id
    if layer_ids is not None:
        metadata["layer_ids"] = list(layer_ids)
    if poison is not None:
        metadata["adapter_metadata"]["poison"] = poison
    return metadata


def _assert_zero(summary: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if summary.get(key) != 0:
            raise AssertionError((key, summary.get(key)))


def _assert_bridge_zero(summary: dict[str, Any]) -> None:
    _assert_zero(
        summary,
        (
            "req_to_token_read_count",
            "actual_req_to_token_pool_read_count",
            "token_to_kv_pool_read_count",
            "actual_token_to_kv_pool_read_count",
            "kv_pool_read_count",
            "kv_snapshot_count",
            "tensor_read_count",
            "attention_comparison_executed_count",
            "attention_override_true_count",
            "runtime_writeback_true_count",
            "scheduler_policy_noop_false_count",
            "kv_cache_mutation_true_count",
            "source_mutated_true_count",
        ),
    )


def _assert_producer_zero(summary: dict[str, Any]) -> None:
    _assert_zero(
        summary,
        (
            "req_to_token_read_count",
            "actual_req_to_token_pool_read_count",
            "token_to_kv_pool_read_count",
            "actual_token_to_kv_pool_read_count",
            "live_token_to_kv_pool_index_read_count",
            "kv_pool_read_count",
            "kv_snapshot_count",
            "tensor_read_count",
            "attention_comparison_executed_count",
            "attention_override_true_count",
            "runtime_writeback_true_count",
            "scheduler_policy_noop_false_count",
            "kv_cache_mutation_true_count",
            "source_mutated_true_count",
        ),
    )


def _bridge(
    *,
    forward_batch: Any = None,
    model_runner: Any = None,
    explicit_runtime_observation_payloads: Any = None,
    bridge_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = build_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
        forward_batch=forward_batch,
        model_runner=model_runner,
        explicit_runtime_observation_payloads=explicit_runtime_observation_payloads,
        bridge_enabled=bridge_enabled,
    )
    if len(results) != 1:
        raise AssertionError(results)
    result = results[0]
    summary = summarize_relaykv_runtime_observation_metadata_source_bridge_payloads_for_smoke(
        results
    )
    _assert_bridge_zero(summary)
    return result, summary


def _derive_produce_bridge_live(
    runtime_observation_payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    derived = build_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        production_enabled=True,
    )
    derived_summary = summarize_relaykv_runtime_metadata_derived_req_to_token_entries_for_smoke(
        derived,
        production_enabled=True,
        max_tokens_per_request=256,
        max_total_tokens=1024,
    )
    _assert_producer_zero(derived_summary)
    if derived_summary["derived_count"] <= 0:
        raise AssertionError(derived_summary)

    derived_entries = derived[0]["derived_req_to_token_entries"]
    payloads = build_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        runtime_observation_payloads=runtime_observation_payloads,
        explicit_req_to_token_entries=derived_entries,
        production_enabled=True,
    )
    payload_summary = summarize_relaykv_runtime_req_to_token_resolution_payloads_for_smoke(
        payloads,
        production_enabled=True,
        max_tokens_per_request=256,
        max_total_tokens=1024,
    )
    _assert_producer_zero(payload_summary)
    if payload_summary["resolved_count"] <= 0:
        raise AssertionError(payload_summary)

    bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        explicit_payloads=payloads,
        bridge_enabled=True,
    )
    bridge_summary = summarize_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        bridge_results
    )
    _assert_zero(
        bridge_summary,
        (
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
        ),
    )
    req_payloads = bridge_results[0]["req_to_token_resolution_payloads"]

    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        req_payloads,
        token_to_kv_pool_object={0: 200, 1: 201, 2: 202, 3: 203, 4: 204, 5: 205},
        read_token_to_kv_pool_index=True,
        source_path="fake.token_to_kv_pool",
    )
    live_summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        live_results
    )
    if live_summary["physical_kv_index_resolved_count"] <= 0:
        raise AssertionError(live_summary)
    if live_summary["token_to_kv_pool_read_count"] <= 0:
        raise AssertionError(live_summary)
    return derived_summary, payload_summary, bridge_summary, live_summary


def _assert_bridge_disabled_blocked() -> None:
    result, summary = _bridge(
        explicit_runtime_observation_payloads=[_runtime_observation_payload()],
        bridge_enabled=False,
    )
    if result["bridge_state"] != "blocked":
        raise AssertionError(result)
    if result["blocked_reason"] != "bridge_not_enabled":
        raise AssertionError(result)
    if summary["bridge_state"] != "blocked":
        raise AssertionError(summary)


def _assert_source_priority_and_paths() -> None:
    explicit_payload = _runtime_observation_payload("req-explicit", token_span=[0, 2])
    fb = types.SimpleNamespace(
        relaykv_runtime_observation_payloads=[
            _runtime_observation_payload("req-fb-payload", token_span=[0, 3])
        ],
        relaykv_runtime_observation_metadata=[
            _runtime_observation_metadata("req-fb-meta", token_span=[0, 4])
        ],
    )
    mr = types.SimpleNamespace(
        relaykv_runtime_observation_payloads=[
            _runtime_observation_payload("req-mr-payload", token_span=[0, 5])
        ],
        relaykv_runtime_observation_metadata=[
            _runtime_observation_metadata("req-mr-meta", token_span=[0, 6])
        ],
    )

    result, _ = _bridge(
        forward_batch=fb,
        model_runner=mr,
        explicit_runtime_observation_payloads=[explicit_payload],
        bridge_enabled=True,
    )
    if result["bridge_source_path"] != "explicit_runtime_observation_payloads":
        raise AssertionError(result)
    if result["runtime_observation_payloads"][0]["engine_request_id"] != "engine-req-explicit":
        raise AssertionError(result)

    result, _ = _bridge(forward_batch=fb, model_runner=mr, bridge_enabled=True)
    if result["bridge_source_path"] != "forward_batch.relaykv_runtime_observation_payloads":
        raise AssertionError(result)

    delattr(fb, "relaykv_runtime_observation_payloads")
    result, _ = _bridge(forward_batch=fb, model_runner=mr, bridge_enabled=True)
    if result["bridge_source_path"] != "forward_batch.relaykv_runtime_observation_metadata":
        raise AssertionError(result)

    delattr(fb, "relaykv_runtime_observation_metadata")
    result, _ = _bridge(forward_batch=fb, model_runner=mr, bridge_enabled=True)
    if result["bridge_source_path"] != "model_runner.relaykv_runtime_observation_payloads":
        raise AssertionError(result)

    delattr(mr, "relaykv_runtime_observation_payloads")
    result, _ = _bridge(forward_batch=fb, model_runner=mr, bridge_enabled=True)
    if result["bridge_source_path"] != "model_runner.relaykv_runtime_observation_metadata":
        raise AssertionError(result)


def _assert_metadata_paths_and_mixed_inputs() -> None:
    poison = _PoisonObject()
    metadata = _runtime_observation_metadata(
        "req-meta",
        token_span=[1, 4],
        layer_ids=[14, 15],
        layer_id=None,
        poison=poison,
    )
    before = copy.deepcopy(metadata)
    result, summary = _bridge(
        forward_batch=types.SimpleNamespace(
            relaykv_runtime_observation_metadata=[metadata, {"bad": "value"}]
        ),
        bridge_enabled=True,
    )
    if metadata != before:
        raise AssertionError("metadata source mutated")
    if poison.touched:
        raise AssertionError("poison object was touched")
    if result["bridge_state"] != "bridged":
        raise AssertionError(result)
    if result["valid_payload_count"] != 2:
        raise AssertionError(result)
    if result["blocked_payload_count"] != 1:
        raise AssertionError(result)
    if summary["blocked_payload_count"] != 1:
        raise AssertionError(summary)
    for payload in result["runtime_observation_payloads"]:
        if payload["adapter_metadata"]["runtime_observation_source_bridge"] is not True:
            raise AssertionError(payload)


def _assert_missing_and_invalid_blocked() -> None:
    result, _ = _bridge(bridge_enabled=True)
    if result["bridge_state"] != "blocked":
        raise AssertionError(result)
    if result["blocked_reason"] != "runtime_observation_source_missing":
        raise AssertionError(result)

    result, _ = _bridge(
        explicit_runtime_observation_payloads={"bad": "not-a-listable-source"},
        bridge_enabled=True,
    )
    if result["bridge_state"] != "blocked":
        raise AssertionError(result)

    result, _ = _bridge(
        explicit_runtime_observation_payloads=[{"request_id": None, "seq_len": 4}],
        bridge_enabled=True,
    )
    if result["bridge_state"] != "blocked":
        raise AssertionError(result)
    if result["blocked_reason"] != "runtime_observation_source_invalid":
        raise AssertionError(result)


def _assert_chain_from_bridged_metadata() -> None:
    result, _ = _bridge(
        explicit_runtime_observation_payloads=[
            _runtime_observation_metadata("req-chain", token_span=[0, 3])
        ],
        bridge_enabled=True,
    )
    derived_summary, payload_summary, bridge_summary, live_summary = (
        _derive_produce_bridge_live(result["runtime_observation_payloads"])
    )
    if derived_summary["derived_count"] <= 0:
        raise AssertionError(derived_summary)
    if payload_summary["resolved_count"] <= 0:
        raise AssertionError(payload_summary)
    if bridge_summary["bridged_count"] <= 0:
        raise AssertionError(bridge_summary)
    if live_summary["physical_kv_index_resolved_count"] <= 0:
        raise AssertionError(live_summary)


def _assert_wrapper_metadata_only_path() -> None:
    previous = os.environ.get(
        "SGLANG_RELAYKV_RUNTIME_METADATA_DERIVED_REQ_TO_TOKEN_ENTRIES"
    )
    os.environ["SGLANG_RELAYKV_RUNTIME_METADATA_DERIVED_REQ_TO_TOKEN_ENTRIES"] = "1"
    try:
        forward_batch = types.SimpleNamespace(
            relaykv_runtime_observation_metadata=[
                _runtime_observation_metadata("req-wrapper", token_span=[0, 3])
            ]
        )
        runner = types.SimpleNamespace()
        result = run_model_runner_runtime_req_to_token_payload_production_hook_for_smoke(
            runner,
            forward_batch=forward_batch,
        )
    finally:
        if previous is None:
            os.environ.pop(
                "SGLANG_RELAYKV_RUNTIME_METADATA_DERIVED_REQ_TO_TOKEN_ENTRIES",
                None,
            )
        else:
            os.environ[
                "SGLANG_RELAYKV_RUNTIME_METADATA_DERIVED_REQ_TO_TOKEN_ENTRIES"
            ] = previous

    summary = result["summary"]
    _assert_producer_zero(summary)
    if summary["payload_attached"] is not True:
        raise AssertionError(summary)
    if summary["req_to_token_entry_source"] != "metadata_derived":
        raise AssertionError(summary)
    if summary["runtime_observation_source_bridge_state"] != "bridged":
        raise AssertionError(summary)
    if summary["runtime_observation_source_bridge_valid_count"] <= 0:
        raise AssertionError(summary)
    attached = getattr(forward_batch, "relaykv_req_to_token_resolution_payloads", None)
    if not isinstance(attached, list) or not attached:
        raise AssertionError(attached)

    bridge_results = build_relaykv_req_to_token_resolution_bridge_payloads_for_smoke(
        explicit_payloads=attached,
        bridge_enabled=True,
    )
    live_results = build_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        bridge_results[0]["req_to_token_resolution_payloads"],
        token_to_kv_pool_object={0: 300, 1: 301, 2: 302},
        read_token_to_kv_pool_index=True,
        source_path="fake.wrapper.token_to_kv_pool",
    )
    live_summary = summarize_relaykv_live_token_to_kv_pool_index_read_results_for_smoke(
        live_results
    )
    if live_summary["physical_kv_index_resolved_count"] <= 0:
        raise AssertionError(live_summary)


def main() -> int:
    _assert_bridge_disabled_blocked()
    _assert_source_priority_and_paths()
    _assert_metadata_paths_and_mixed_inputs()
    _assert_missing_and_invalid_blocked()
    _assert_chain_from_bridged_metadata()
    _assert_wrapper_metadata_only_path()

    print("relaykv_runtime_observation_metadata_source_bridge_smoke=pass")
    print(
        json.dumps(
            {
                "bridge_disabled": "blocked",
                "metadata_paths": "bridged",
                "mixed": "valid_preserved_invalid_counted",
                "priority": (
                    "explicit>forward_batch_payloads>forward_batch_metadata>"
                    "model_runner_payloads>model_runner_metadata"
                ),
                "wrapper_metadata_only": "metadata_derived_attached_and_live_resolved",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
