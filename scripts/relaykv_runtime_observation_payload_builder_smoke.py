from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from typing import Any

# Keep the smoke import environment consistent with other RelayKV smoke scripts.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

from sglang.srt.relaykv.observation import build_runtime_observation_payloads


@dataclass(frozen=True)
class _FakeForwardBatchLike:
    rids: list[str]
    req_pool_indices: list[int]
    seq_lens: list[int]

    @property
    def batch_size(self) -> int:
        return len(self.rids)


def _assert_payloads(payloads: list[dict[str, Any]]) -> None:
    expected_rids = ["rid-a", "rid-b", "rid-c"]
    expected_req_pool_indices = [0, 1, 2]
    expected_seq_lens = [128, 256, 384]
    expected_layer_ids = [0, 1, 2]

    if len(payloads) != 9:
        raise AssertionError(payloads)

    expected = []
    for request_index, rid in enumerate(expected_rids):
        for layer_id in expected_layer_ids:
            expected.append(
                {
                    "batch_id": "obs-batch-a",
                    "request_id": rid,
                    "request_index": request_index,
                    "req_pool_index": expected_req_pool_indices[request_index],
                    "seq_len": expected_seq_lens[request_index],
                    "layer_id": layer_id,
                }
            )

    for payload, expected_fields in zip(payloads, expected):
        for key, value in expected_fields.items():
            if payload[key] != value:
                raise AssertionError(payload)

        if payload["event_type"] != "runtime_observation":
            raise AssertionError(payload)
        if payload["phase"] != "decode":
            raise AssertionError(payload)
        if payload["runtime_policy_state"] != "shadow_observation":
            raise AssertionError(payload)
        if payload["source_mutated"] is not False:
            raise AssertionError(payload)
        if payload["attention_override"] is not False:
            raise AssertionError(payload)
        if payload["kv_cache_mutation"] is not False:
            raise AssertionError(payload)
        if payload["runtime_writeback"] is not False:
            raise AssertionError(payload)
        if payload["scheduler_policy_noop"] is not True:
            raise AssertionError(payload)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    batch = _FakeForwardBatchLike(
        rids=["rid-a", "rid-b", "rid-c"],
        req_pool_indices=[0, 1, 2],
        seq_lens=[128, 256, 384],
    )
    payloads = build_runtime_observation_payloads(
        batch=batch,
        layer_ids=[0, 1, 2],
        batch_id="obs-batch-a",
        phase="decode",
        runtime_policy_state="shadow_observation",
    )
    _assert_payloads(payloads)
    print("relaykv_runtime_observation_payload_builder_smoke: ok")
    print(
        "relaykv_runtime_observation_payload_builder_summary="
        + json.dumps(
            {
                "total_payloads": len(payloads),
                "requests": sorted({payload["request_id"] for payload in payloads}),
                "layers": sorted({payload["layer_id"] for payload in payloads}),
                "batch_ids": sorted({payload["batch_id"] for payload in payloads}),
                "source_mutated_true_count": sum(
                    payload["source_mutated"] is True for payload in payloads
                ),
                "attention_override_true_count": sum(
                    payload["attention_override"] is True for payload in payloads
                ),
                "kv_cache_mutation_true_count": sum(
                    payload["kv_cache_mutation"] is True for payload in payloads
                ),
                "runtime_writeback_true_count": sum(
                    payload["runtime_writeback"] is True for payload in payloads
                ),
                "scheduler_policy_noop_false_count": sum(
                    payload["scheduler_policy_noop"] is False for payload in payloads
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
