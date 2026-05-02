from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

from sglang.srt.relaykv.observation import (
    build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata,
    summarize_runtime_observation_payloads,
)


@dataclass(frozen=True)
class _FakeForwardBatchExistingMetadata:
    rids: Any
    seq_lens_cpu: Any
    extend_seq_lens_cpu: Any = None
    extend_prefix_lens_cpu: Any = None


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
        if payload["source"] != "forward_batch_existing_cpu_metadata":
            raise AssertionError(payload)
        if payload["batch_id"] != "forward-existing-meta-batch-a":
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
        if payload["extend_seq_len"] != extend_seq:
            raise AssertionError(payload)
        if payload["extend_prefix_len"] != extend_prefix:
            raise AssertionError(payload)
        if payload["layer_id"] != layer_id:
            raise AssertionError(payload)
        if payload["phase"] != "decode":
            raise AssertionError(payload)
        if payload["runtime_policy_state"] != "forward_batch_existing_metadata_schema":
            raise AssertionError(payload)
        _assert_safety_flags(payload)


def _assert_with_optional_extend_metadata() -> dict[str, Any]:
    payloads = build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata(
        forward_batch=_FakeForwardBatchExistingMetadata(
            rids=["rid-a", "rid-b"],
            seq_lens_cpu=[128, 256],
            extend_seq_lens_cpu=[16, 32],
            extend_prefix_lens_cpu=[112, 224],
        ),
        layer_ids=[0, 14],
        batch_id="forward-existing-meta-batch-a",
        phase="decode",
        runtime_policy_state="forward_batch_existing_metadata_schema",
    )
    _assert_expected_payloads(payloads)
    summary = summarize_runtime_observation_payloads(payloads)
    expected_summary = {
        "total_payloads": 4,
        "per_request_counts": {"rid-a": 2, "rid-b": 2},
        "per_layer_counts": {"0": 2, "14": 2},
        "per_batch_counts": {"forward-existing-meta-batch-a": 4},
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
    payloads = build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata(
        forward_batch=_FakeForwardBatchExistingMetadata(
            rids=["rid-a", "rid-b"],
            seq_lens_cpu=[128, 256],
        ),
        layer_ids=[0, 14],
        batch_id="forward-existing-meta-batch-a",
        phase="decode",
        runtime_policy_state="forward_batch_existing_metadata_schema",
    )
    if len(payloads) != 4:
        raise AssertionError(payloads)
    for payload in payloads:
        if payload["req_pool_idx"] is not None:
            raise AssertionError(payload)
        if payload["req_pool_index"] is not None:
            raise AssertionError(payload)
        if "extend_seq_len" in payload:
            raise AssertionError(payload)
        if "extend_prefix_len" in payload:
            raise AssertionError(payload)
        _assert_safety_flags(payload)
    return {"payloads": payloads}


def _assert_length_mismatch_rejected() -> dict[str, str]:
    cases = {
        "seq_lens_cpu": _FakeForwardBatchExistingMetadata(
            rids=["rid-a", "rid-b"],
            seq_lens_cpu=[128],
        ),
        "extend_seq_lens_cpu": _FakeForwardBatchExistingMetadata(
            rids=["rid-a", "rid-b"],
            seq_lens_cpu=[128, 256],
            extend_seq_lens_cpu=[16],
        ),
        "extend_prefix_lens_cpu": _FakeForwardBatchExistingMetadata(
            rids=["rid-a", "rid-b"],
            seq_lens_cpu=[128, 256],
            extend_prefix_lens_cpu=[112],
        ),
    }
    results: dict[str, str] = {}
    for case_name, forward_batch in cases.items():
        try:
            build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata(
                forward_batch=forward_batch,
                layer_ids=[0, 14],
                batch_id="forward-existing-meta-batch-a",
                phase="decode",
                runtime_policy_state="forward_batch_existing_metadata_schema",
            )
        except ValueError as exc:
            results[case_name] = type(exc).__name__
        else:
            raise AssertionError(f"{case_name} was not rejected")
    return results


def _assert_tensor_like_rejected_without_forbidden_access() -> dict[str, str]:
    tensor_like = _PoisonTensorLike()
    try:
        build_runtime_observation_payload_candidates_from_forward_batch_existing_metadata(
            forward_batch=_FakeForwardBatchExistingMetadata(
                rids=["rid-a", "rid-b"],
                seq_lens_cpu=tensor_like,
            ),
            layer_ids=[0, 14],
            batch_id="forward-existing-meta-batch-a",
            phase="decode",
            runtime_policy_state="forward_batch_existing_metadata_schema",
        )
    except TypeError as exc:
        if tensor_like.forbidden_access_called:
            raise AssertionError("forbidden tensor-like access was called") from exc
        return {"seq_lens_cpu": type(exc).__name__}
    raise AssertionError("tensor-like seq_lens_cpu was not rejected")


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    results = {
        "with_optional_extend_metadata": _assert_with_optional_extend_metadata(),
        "without_optional_extend_metadata": _assert_without_optional_extend_metadata(),
        "length_mismatch_rejected": _assert_length_mismatch_rejected(),
        "tensor_like_rejected": _assert_tensor_like_rejected_without_forbidden_access(),
    }
    print("relaykv_forward_batch_existing_metadata_payload_candidate_smoke: ok")
    print(
        "relaykv_forward_batch_existing_metadata_payload_candidate_results="
        + json.dumps(results, sort_keys=True)
    )


if __name__ == "__main__":
    main()
