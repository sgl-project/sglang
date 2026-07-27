"""GPU parity tests for BAGEL's FlashInfer attention fallback."""

import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.bagel_transformer import (
    _run_flashinfer_varlen_attention,
    _sdpa_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="BAGEL uses FlashInfer fallback only on Blackwell or newer",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 128


def _make_tensors(
    query_length: int,
    key_length: int,
    query_heads: int,
    key_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic BF16 GQA tensors for one request."""
    torch.manual_seed(query_length + key_length + query_heads)
    query = torch.randn(
        query_length,
        query_heads,
        HEAD_DIM,
        device=DEVICE,
        dtype=DTYPE,
    )
    key = torch.randn(
        key_length,
        key_heads,
        HEAD_DIM,
        device=DEVICE,
        dtype=DTYPE,
    )
    return query, key, torch.randn_like(key)


@pytest.mark.parametrize(
    ("query_length", "key_length", "query_heads", "key_heads", "causal"),
    [
        (128, 128, 28, 4, True),
        (1, 128, 28, 4, True),
        (256, 384, 28, 4, False),
        (256, 384, 14, 2, False),
    ],
)
def test_bagel_flashinfer_matches_sdpa(
    query_length: int,
    key_length: int,
    query_heads: int,
    key_heads: int,
    causal: bool,
) -> None:
    query, key, value = _make_tensors(
        query_length,
        key_length,
        query_heads,
        key_heads,
    )

    actual = _run_flashinfer_varlen_attention(
        query,
        key,
        value,
        [query_length],
        [key_length],
        causal=causal,
    )
    expected = _sdpa_attention(query, key, value, causal)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_bagel_flashinfer_keeps_dynamic_requests_isolated(causal: bool) -> None:
    query_sizes = [64, 97]
    key_sizes = [96, 128]
    requests = [
        _make_tensors(query_length, key_length, 28, 4)
        for query_length, key_length in zip(query_sizes, key_sizes)
    ]
    query = torch.cat([request[0] for request in requests], dim=0)
    key = torch.cat([request[1] for request in requests], dim=0)
    value = torch.cat([request[2] for request in requests], dim=0)

    actual = _run_flashinfer_varlen_attention(
        query,
        key,
        value,
        query_sizes,
        key_sizes,
        causal=causal,
    )
    expected = torch.cat(
        [
            _sdpa_attention(request_query, request_key, request_value, causal)
            for request_query, request_key, request_value in requests
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
