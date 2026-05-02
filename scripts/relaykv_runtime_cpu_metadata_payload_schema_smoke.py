from __future__ import annotations

import json
import os
import signal
from typing import Any

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

from sglang.srt.relaykv.observation import (
    build_runtime_observation_cpu_metadata_payloads,
    summarize_runtime_observation_payloads,
)


class _PoisonTensorLike:
    def __init__(self) -> None:
        self.cpu_called = False
        self.item_called = False
        self.tolist_called = False
        self.iter_called = False
        self.len_called = False
        self.getitem_called = False
        self.shape = (2,)
        self.device = "cuda:0"
        self.dtype = "torch.int64"

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def __iter__(self):
        self.iter_called = True
        raise AssertionError("__iter__() must not be called")

    def __len__(self) -> int:
        self.len_called = True
        raise AssertionError("__len__() must not be called")

    def __getitem__(self, index: int) -> None:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

    @property
    def forbidden_access_called(self) -> bool:
        return (
            self.cpu_called
            or self.item_called
            or self.tolist_called
            or self.iter_called
            or self.len_called
            or self.getitem_called
        )


def _assert_safety_flags(payload: dict[str, Any]) -> None:
    expected = {
        "source_mutated": False,
        "attention_override": False,
        "kv_cache_mutation": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
    }
    for key, value in expected.items():
        if payload[key] is not value:
            raise AssertionError(payload)


def _assert_expected_payloads(payloads: list[dict[str, Any]]) -> None:
    if len(payloads) != 4:
        raise AssertionError(payloads)

    expected = [
        ("rid-a", 0, 10, 128, 0),
        ("rid-a", 0, 10, 128, 1),
        ("rid-b", 1, 11, 256, 0),
        ("rid-b", 1, 11, 256, 1),
    ]
    for payload, (rid, request_index, req_pool_idx, seq_len, layer_id) in zip(
        payloads, expected
    ):
        if payload["event_type"] != "runtime_observation_cpu_metadata_candidate":
            raise AssertionError(payload)
        if payload["source"] != "cpu_metadata":
            raise AssertionError(payload)
        if payload["batch_id"] != "cpu-meta-batch-a":
            raise AssertionError(payload)
        if payload["request_id"] != rid:
            raise AssertionError(payload)
        if payload["request_index_in_batch"] != request_index:
            raise AssertionError(payload)
        if payload["request_index"] != request_index:
            raise AssertionError(payload)
        if payload["req_pool_idx"] != req_pool_idx:
            raise AssertionError(payload)
        if payload["req_pool_index"] != req_pool_idx:
            raise AssertionError(payload)
        if payload["seq_len"] != seq_len:
            raise AssertionError(payload)
        if payload["layer_id"] != layer_id:
            raise AssertionError(payload)
        if payload["phase"] != "decode":
            raise AssertionError(payload)
        if payload["runtime_policy_state"] != "cpu_metadata_schema":
            raise AssertionError(payload)
        if payload["extend_seq_len"] != seq_len - 8:
            raise AssertionError(payload)
        if payload["extend_prefix_len"] != 8:
            raise AssertionError(payload)
        _assert_safety_flags(payload)


def _assert_with_optional_extend_metadata() -> dict[str, Any]:
    payloads = build_runtime_observation_cpu_metadata_payloads(
        rids=["rid-a", "rid-b"],
        req_pool_indices_cpu=[10, 11],
        seq_lens_cpu=[128, 256],
        extend_seq_lens_cpu=[120, 248],
        extend_prefix_lens_cpu=[8, 8],
        layer_ids=[0, 1],
        batch_id="cpu-meta-batch-a",
        phase="decode",
        runtime_policy_state="cpu_metadata_schema",
    )
    _assert_expected_payloads(payloads)
    summary = summarize_runtime_observation_payloads(payloads)
    expected_summary = {
        "total_payloads": 4,
        "per_request_counts": {"rid-a": 2, "rid-b": 2},
        "per_layer_counts": {"0": 2, "1": 2},
        "per_batch_counts": {"cpu-meta-batch-a": 4},
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }
    if summary != expected_summary:
        raise AssertionError(summary)
    return {"payloads": payloads, "summary": summary}


def _assert_without_optional_extend_metadata() -> dict[str, Any]:
    payloads = build_runtime_observation_cpu_metadata_payloads(
        rids=["rid-a", "rid-b"],
        req_pool_indices_cpu=[10, 11],
        seq_lens_cpu=[128, 256],
        layer_ids=[0, 1],
        batch_id="cpu-meta-batch-a",
        phase="decode",
        runtime_policy_state="cpu_metadata_schema",
    )
    if len(payloads) != 4:
        raise AssertionError(payloads)
    for payload in payloads:
        if "extend_seq_len" in payload:
            raise AssertionError(payload)
        if "extend_prefix_len" in payload:
            raise AssertionError(payload)
        _assert_safety_flags(payload)
    return {"payloads": payloads}


def _assert_length_mismatch_rejected() -> dict[str, str]:
    cases = {
        "req_pool_indices_cpu": {
            "rids": ["rid-a", "rid-b"],
            "req_pool_indices_cpu": [10],
            "seq_lens_cpu": [128, 256],
        },
        "seq_lens_cpu": {
            "rids": ["rid-a", "rid-b"],
            "req_pool_indices_cpu": [10, 11],
            "seq_lens_cpu": [128],
        },
        "extend_seq_lens_cpu": {
            "rids": ["rid-a", "rid-b"],
            "req_pool_indices_cpu": [10, 11],
            "seq_lens_cpu": [128, 256],
            "extend_seq_lens_cpu": [120],
        },
        "extend_prefix_lens_cpu": {
            "rids": ["rid-a", "rid-b"],
            "req_pool_indices_cpu": [10, 11],
            "seq_lens_cpu": [128, 256],
            "extend_prefix_lens_cpu": [8],
        },
    }
    results: dict[str, str] = {}
    for case_name, kwargs in cases.items():
        try:
            build_runtime_observation_cpu_metadata_payloads(
                **kwargs,
                layer_ids=[0, 1],
                batch_id="cpu-meta-batch-a",
                phase="decode",
                runtime_policy_state="cpu_metadata_schema",
            )
        except ValueError as exc:
            results[case_name] = type(exc).__name__
        else:
            raise AssertionError(f"{case_name} was not rejected")
    return results


def _assert_tensor_like_rejected_without_forbidden_access() -> dict[str, str]:
    tensor_like = _PoisonTensorLike()
    try:
        build_runtime_observation_cpu_metadata_payloads(
            rids=["rid-a", "rid-b"],
            req_pool_indices_cpu=tensor_like,
            seq_lens_cpu=[128, 256],
            layer_ids=[0, 1],
            batch_id="cpu-meta-batch-a",
            phase="decode",
            runtime_policy_state="cpu_metadata_schema",
        )
    except TypeError as exc:
        if tensor_like.forbidden_access_called:
            raise AssertionError("forbidden tensor-like access was called") from exc
        return {"req_pool_indices_cpu": type(exc).__name__}
    raise AssertionError("tensor-like req_pool_indices_cpu was not rejected")


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    results = {
        "with_optional_extend_metadata": _assert_with_optional_extend_metadata(),
        "without_optional_extend_metadata": _assert_without_optional_extend_metadata(),
        "length_mismatch_rejected": _assert_length_mismatch_rejected(),
        "tensor_like_rejected": _assert_tensor_like_rejected_without_forbidden_access(),
    }
    print("relaykv_runtime_cpu_metadata_payload_schema_smoke: ok")
    print(
        "relaykv_runtime_cpu_metadata_payload_schema_results="
        + json.dumps(results, sort_keys=True)
    )


if __name__ == "__main__":
    main()
