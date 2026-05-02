from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

import torch

from sglang.srt.relaykv.observation import (
    build_runtime_observation_payload_candidates_from_forward_batch_readonly_metadata,
    run_model_runner_forward_observation_hook,
)


@dataclass(frozen=True)
class _FakeForwardBatchReadonlyMetadata:
    rids: Any
    req_pool_indices: Any
    seq_lens: Any
    seq_lens_cpu: Any
    relaykv_runtime_observation_metadata: Any = None
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


class _ExplodingReadonlyMetadataBatch:
    @property
    def relaykv_runtime_observation_metadata(self) -> Any:
        raise AssertionError("metadata must not be read while env is off")


def _readonly_metadata() -> list[dict[str, Any]]:
    return [
        {
            "request_id": "rid-a",
            "rid": "rid-a",
            "request_index_in_batch": 0,
            "req_pool_idx": 10,
            "seq_len": 128,
            "extend_seq_len": 16,
            "extend_prefix_len": 112,
            "phase": "decode",
            "source": "forward_batch_readonly_runtime_observation_metadata",
        },
        {
            "request_id": "rid-b",
            "rid": "rid-b",
            "request_index_in_batch": 1,
            "req_pool_idx": 11,
            "seq_len": 256,
            "extend_seq_len": 32,
            "extend_prefix_len": 224,
            "phase": "decode",
            "source": "forward_batch_readonly_runtime_observation_metadata",
        },
    ]


def _assert_payloads(payloads: list[dict[str, Any]]) -> None:
    expected = [
        ("rid-a", 0, 10, 128, 16, 112, 0),
        ("rid-a", 0, 10, 128, 16, 112, 14),
        ("rid-b", 1, 11, 256, 32, 224, 0),
        ("rid-b", 1, 11, 256, 32, 224, 14),
    ]
    if len(payloads) != 4:
        raise AssertionError(payloads)
    for payload, (rid, index, req_pool_idx, seq_len, extend_seq, extend_prefix, layer) in zip(
        payloads, expected
    ):
        if payload["event_type"] != "runtime_observation_readonly_metadata_candidate":
            raise AssertionError(payload)
        if payload["source"] != "forward_batch_readonly_runtime_observation_metadata":
            raise AssertionError(payload)
        if payload["request_id"] != rid:
            raise AssertionError(payload)
        if payload["request_index_in_batch"] != index:
            raise AssertionError(payload)
        if payload["request_index"] != index:
            raise AssertionError(payload)
        if payload["req_pool_idx"] != req_pool_idx:
            raise AssertionError(payload)
        if payload["req_pool_index"] != req_pool_idx:
            raise AssertionError(payload)
        if payload["seq_len"] != seq_len:
            raise AssertionError(payload)
        if payload["extend_seq_len"] != extend_seq:
            raise AssertionError(payload)
        if payload["extend_prefix_len"] != extend_prefix:
            raise AssertionError(payload)
        if payload["layer_id"] != layer:
            raise AssertionError(payload)
        for key in (
            "source_mutated",
            "attention_override",
            "kv_cache_mutation",
            "runtime_writeback",
        ):
            if payload[key] is not False:
                raise AssertionError(payload)
        if payload["scheduler_policy_noop"] is not True:
            raise AssertionError(payload)


def _assert_direct_readonly_metadata_builder() -> dict[str, Any]:
    payloads = build_runtime_observation_payload_candidates_from_forward_batch_readonly_metadata(
        forward_batch=_FakeForwardBatchReadonlyMetadata(
            rids=["rid-a", "rid-b"],
            req_pool_indices=_PoisonTensorLike(),
            seq_lens=_PoisonTensorLike(),
            seq_lens_cpu=torch.tensor([128, 256], dtype=torch.int64),
            relaykv_runtime_observation_metadata=_readonly_metadata(),
        ),
        layer_ids=[0, 14],
        batch_id="readonly-meta-batch-a",
        phase="forward",
        runtime_policy_state="runtime_observation",
    )
    _assert_payloads(payloads)
    return {"payloads": payloads}


def _assert_hook_prefers_readonly_metadata() -> dict[str, Any]:
    req_pool_indices = _PoisonTensorLike()
    seq_lens = _PoisonTensorLike()
    result = run_model_runner_forward_observation_hook(
        forward_batch=_FakeForwardBatchReadonlyMetadata(
            rids=["rid-a", "rid-b"],
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor([128, 256], dtype=torch.int64),
            relaykv_runtime_observation_metadata=_readonly_metadata(),
            extend_seq_lens_cpu=[16, 32],
            extend_prefix_lens_cpu=[112, 224],
        ),
        forward_pass_id=7,
        env_value="1",
    )
    if req_pool_indices.forbidden_access_called or seq_lens.forbidden_access_called:
        raise AssertionError("GPU tensor-like values were read")
    if result["enabled"] is not True or result["skipped"] is not False:
        raise AssertionError(result)
    summary = result["summary"]
    if summary["source"] != "forward_batch_readonly_runtime_observation_metadata":
        raise AssertionError(result)
    if summary["req_pool_idx_none"] is not False:
        raise AssertionError(result)
    if summary["total_payloads"] != 2:
        raise AssertionError(result)
    if summary.get("seq_lens_cpu_value_source"):
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


def _assert_missing_metadata_falls_back_to_existing_metadata() -> dict[str, Any]:
    result = run_model_runner_forward_observation_hook(
        forward_batch=_FakeForwardBatchReadonlyMetadata(
            rids=["rid-a", "rid-b"],
            req_pool_indices=_PoisonTensorLike(),
            seq_lens=_PoisonTensorLike(),
            seq_lens_cpu=torch.tensor([128, 256], dtype=torch.int64),
            relaykv_runtime_observation_metadata=None,
            extend_seq_lens_cpu=[16, 32],
            extend_prefix_lens_cpu=[112, 224],
        ),
        forward_pass_id=8,
        env_value="1",
    )
    summary = result["summary"]
    if summary["source"] != "forward_batch_existing_cpu_metadata_runtime_observation":
        raise AssertionError(result)
    if summary["req_pool_idx_none"] is not True:
        raise AssertionError(result)
    if result["readonly_metadata_skip_reason"] != "TypeError":
        raise AssertionError(result)
    return result


def _assert_env_off_does_not_read_metadata() -> dict[str, Any]:
    result = run_model_runner_forward_observation_hook(
        forward_batch=_ExplodingReadonlyMetadataBatch(),
        forward_pass_id=9,
        env_value="0",
    )
    if result["skip_reason"] != "env_disabled":
        raise AssertionError(result)
    return result


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    results = {
        "direct_readonly_metadata_builder": _assert_direct_readonly_metadata_builder(),
        "hook_prefers_readonly_metadata": _assert_hook_prefers_readonly_metadata(),
        "missing_metadata_fallback": _assert_missing_metadata_falls_back_to_existing_metadata(),
        "env_off": _assert_env_off_does_not_read_metadata(),
    }
    print("relaykv_forward_batch_readonly_metadata_observation_smoke: ok")
    print(
        "relaykv_forward_batch_readonly_metadata_observation_results="
        + json.dumps(results, sort_keys=True)
    )


if __name__ == "__main__":
    main()
