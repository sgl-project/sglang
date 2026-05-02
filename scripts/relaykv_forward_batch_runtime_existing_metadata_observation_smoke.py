from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

import torch

from sglang.srt.relaykv.observation import (
    build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata,
    run_model_runner_forward_observation_hook,
    summarize_runtime_observation_payloads,
)


@dataclass(frozen=True)
class _FakeForwardBatchRuntimeExistingMetadata:
    rids: Any
    req_pool_indices: Any = None
    seq_lens: Any = None
    seq_lens_cpu: Any = None
    extend_seq_lens_cpu: Any = None
    extend_prefix_lens_cpu: Any = None


class _PoisonTensorLike:
    def __init__(self, *, device: str = "cuda:0", dtype: str = "torch.int64") -> None:
        self.cpu_called = False
        self.item_called = False
        self.tolist_called = False
        self.iter_called = False
        self.len_called = False
        self.getitem_called = False
        self.shape = (2,)
        self.device = device
        self.dtype = dtype

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


def _assert_expected_payloads(
    payloads: list[dict[str, Any]],
    *,
    expected_value_source: str,
) -> None:
    if len(payloads) != 4:
        raise AssertionError(payloads)

    expected = [
        ("rid-a", 0, 128, 16, 112, 0),
        ("rid-a", 0, 128, 16, 112, 14),
        ("rid-b", 1, 256, 32, 224, 0),
        ("rid-b", 1, 256, 32, 224, 14),
    ]
    for payload, (rid, request_index, seq_len, extend_seq, extend_prefix, layer_id) in zip(
        payloads, expected
    ):
        if (
            payload["event_type"]
            != "runtime_observation_forward_batch_existing_metadata_candidate"
        ):
            raise AssertionError(payload)
        if payload["source"] != "forward_batch_existing_cpu_metadata_runtime_observation":
            raise AssertionError(payload)
        if payload["batch_id"] != "runtime-existing-meta-batch-a":
            raise AssertionError(payload)
        if payload["request_id"] != rid:
            raise AssertionError(payload)
        if payload["request_index_in_batch"] != request_index:
            raise AssertionError(payload)
        if payload["request_index"] != request_index:
            raise AssertionError(payload)
        if payload["req_pool_idx"] is not None:
            raise AssertionError(payload)
        if payload["req_pool_index"] is not None:
            raise AssertionError(payload)
        if payload["seq_len"] != seq_len:
            raise AssertionError(payload)
        if payload["seq_lens_cpu_value_source"] != expected_value_source:
            raise AssertionError(payload)
        if payload["extend_seq_len"] != extend_seq:
            raise AssertionError(payload)
        if payload["extend_prefix_len"] != extend_prefix:
            raise AssertionError(payload)
        if payload["layer_id"] != layer_id:
            raise AssertionError(payload)
        if payload["phase"] != "forward":
            raise AssertionError(payload)
        if payload["runtime_policy_state"] != "runtime_observation":
            raise AssertionError(payload)
        _assert_safety_flags(payload)


def _assert_cpu_tensor_seq_lens_payloads() -> dict[str, Any]:
    payloads = (
        build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata(
            forward_batch=_FakeForwardBatchRuntimeExistingMetadata(
                rids=["rid-a", "rid-b"],
                seq_lens_cpu=torch.tensor([128, 256], dtype=torch.int64),
                extend_seq_lens_cpu=[16, 32],
                extend_prefix_lens_cpu=[112, 224],
            ),
            layer_ids=[0, 14],
            batch_id="runtime-existing-meta-batch-a",
            phase="forward",
            runtime_policy_state="runtime_observation",
        )
    )
    _assert_expected_payloads(
        payloads,
        expected_value_source="cpu_tensor_observation_only",
    )
    summary = summarize_runtime_observation_payloads(payloads)
    if summary != {
        "total_payloads": 4,
        "per_request_counts": {"rid-a": 2, "rid-b": 2},
        "per_layer_counts": {"0": 2, "14": 2},
        "per_batch_counts": {"runtime-existing-meta-batch-a": 4},
        "source_mutated_true_count": 0,
        "attention_override_true_count": 0,
        "kv_cache_mutation_true_count": 0,
        "runtime_writeback_true_count": 0,
        "scheduler_policy_noop_false_count": 0,
    }:
        raise AssertionError(summary)
    return {"payloads": payloads, "summary": summary}


def _assert_list_tuple_seq_lens_payloads_without_extend() -> dict[str, Any]:
    payloads = (
        build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata(
            forward_batch=_FakeForwardBatchRuntimeExistingMetadata(
                rids=["rid-a", "rid-b"],
                seq_lens_cpu=[128, 256],
            ),
            layer_ids=[0, 14],
            batch_id="runtime-existing-meta-batch-a",
            phase="forward",
            runtime_policy_state="runtime_observation",
        )
    )
    if len(payloads) != 4:
        raise AssertionError(payloads)
    for payload in payloads:
        if payload["seq_lens_cpu_value_source"] != "list_tuple_observation_only":
            raise AssertionError(payload)
        if payload["req_pool_idx"] is not None or payload["req_pool_index"] is not None:
            raise AssertionError(payload)
        if "extend_seq_len" in payload or "extend_prefix_len" in payload:
            raise AssertionError(payload)
        _assert_safety_flags(payload)
    return {"payloads": payloads}


def _assert_gpu_tensor_like_rejected_without_forbidden_access() -> dict[str, str]:
    gpu_tensor_like = _PoisonTensorLike(device="cuda:0")
    try:
        build_runtime_observation_payload_candidates_from_forward_batch_runtime_existing_metadata(
            forward_batch=_FakeForwardBatchRuntimeExistingMetadata(
                rids=["rid-a", "rid-b"],
                seq_lens_cpu=gpu_tensor_like,
            ),
            layer_ids=[0, 14],
            batch_id="runtime-existing-meta-batch-a",
            phase="forward",
            runtime_policy_state="runtime_observation",
        )
    except TypeError as exc:
        if gpu_tensor_like.forbidden_access_called:
            raise AssertionError("forbidden GPU tensor-like access was called") from exc
        return {"seq_lens_cpu": type(exc).__name__}
    raise AssertionError("GPU tensor-like seq_lens_cpu was not rejected")


def _assert_req_pool_indices_and_seq_lens_not_read_by_fallback() -> dict[str, Any]:
    req_pool_indices = _PoisonTensorLike(device="cuda:0")
    seq_lens = _PoisonTensorLike(device="cuda:0")
    result = run_model_runner_forward_observation_hook(
        forward_batch=_FakeForwardBatchRuntimeExistingMetadata(
            rids=["rid-a", "rid-b"],
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor([128, 256], dtype=torch.int64),
            extend_seq_lens_cpu=[16, 32],
            extend_prefix_lens_cpu=[112, 224],
        ),
        forward_pass_id=5,
        env_value="1",
    )
    if req_pool_indices.forbidden_access_called or seq_lens.forbidden_access_called:
        raise AssertionError("req_pool_indices or seq_lens values were read")
    if result["enabled"] is not True or result["skipped"] is not False:
        raise AssertionError(result)
    summary = result["summary"]
    if summary["total_payloads"] != 2:
        raise AssertionError(result)
    if summary["source"] != "forward_batch_existing_cpu_metadata_runtime_observation":
        raise AssertionError(result)
    if summary["seq_lens_cpu_value_source"] != "cpu_tensor_observation_only":
        raise AssertionError(result)
    if summary["req_pool_idx_none"] is not True:
        raise AssertionError(result)
    for key in (
        "source_mutated_true_count",
        "attention_override_true_count",
        "kv_cache_mutation_true_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
    ):
        if summary[key] != 0:
            raise AssertionError(result)
    return result


def _assert_env_off_does_not_read_metadata() -> dict[str, Any]:
    req_pool_indices = _PoisonTensorLike(device="cuda:0")
    seq_lens = _PoisonTensorLike(device="cuda:0")
    seq_lens_cpu = _PoisonTensorLike(device="cpu")
    result = run_model_runner_forward_observation_hook(
        forward_batch=_FakeForwardBatchRuntimeExistingMetadata(
            rids=["rid-a", "rid-b"],
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
        ),
        forward_pass_id=6,
        env_value="0",
    )
    if (
        req_pool_indices.forbidden_access_called
        or seq_lens.forbidden_access_called
        or seq_lens_cpu.forbidden_access_called
    ):
        raise AssertionError("metadata was read while env was off")
    if result["skip_reason"] != "env_disabled":
        raise AssertionError(result)
    return result


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    results = {
        "cpu_tensor_seq_lens_payloads": _assert_cpu_tensor_seq_lens_payloads(),
        "list_tuple_seq_lens_payloads_without_extend": (
            _assert_list_tuple_seq_lens_payloads_without_extend()
        ),
        "gpu_tensor_like_rejected": (
            _assert_gpu_tensor_like_rejected_without_forbidden_access()
        ),
        "hook_fallback_payloads": (
            _assert_req_pool_indices_and_seq_lens_not_read_by_fallback()
        ),
        "env_off": _assert_env_off_does_not_read_metadata(),
    }
    print("relaykv_forward_batch_runtime_existing_metadata_observation_smoke: ok")
    print(
        "relaykv_forward_batch_runtime_existing_metadata_observation_results="
        + json.dumps(results, sort_keys=True)
    )


if __name__ == "__main__":
    main()
