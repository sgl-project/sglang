# SPDX-License-Identifier: Apache-2.0

from scripts.bench_sensenova_u1_interleave import (
    RequestInvocation,
    _determinism_summary,
)


def _request(logical_request_id: str, contract_digest: str) -> dict:
    return {
        "logical_request_id": logical_request_id,
        "contract": {
            "contract_sha256": contract_digest,
            "image_sha256": f"image-{contract_digest}",
            "pre_image_token_ids": [1],
            "post_image_token_ids": [2],
        },
    }


def _result(bs1_digest: str, bs8_digest: str) -> dict:
    logical_request_id = "measurement-repeat-0-ordinal-0"
    return {
        "levels": {
            "1": {
                "raw_repeats": [
                    {"requests": [_request(logical_request_id, bs1_digest)]}
                ]
            },
            "8": {
                "raw_repeats": [
                    {"requests": [_request(logical_request_id, bs8_digest)]}
                ]
            },
        }
    }


def test_u1_benchmark_maps_same_logical_request_across_concurrency() -> None:
    bs1 = RequestInvocation(1, "measurement", 3, 0)
    bs8 = RequestInvocation(8, "measurement", 3, 0)

    assert bs1.logical_request_id == bs8.logical_request_id


def test_u1_benchmark_rejects_cross_concurrency_contract_drift() -> None:
    invariant = _determinism_summary(_result("same", "same"), (1, 8))
    drifted = _determinism_summary(_result("bs1", "bs8"), (1, 8))

    assert invariant["logical_request_batch_size_invariant"]
    assert not drifted["logical_request_batch_size_invariant"]
