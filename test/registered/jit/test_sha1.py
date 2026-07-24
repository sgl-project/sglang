"""Correctness tests for CUDA JIT SHA-1 (matches hashlib.sha1)."""

from __future__ import annotations

import hashlib

import pytest
import torch

from sglang.kernels.jit.sha1 import sha1_bytes_cuda, sha1_prefix_data_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _ref_sha1(data: bytes) -> bytes:
    return hashlib.sha1(data).digest()


@pytest.mark.parametrize(
    "n",
    [
        0,
        1,
        63,
        64,
        65,
        127,
        128,
        1023,
        1024,
        4096,
        4097,
        1 << 16,
        (1 << 16) + 17,
    ],
)
def test_sha1_bytes_matches_hashlib(n: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if n == 0:
        data = torch.empty(0, dtype=torch.uint8, device="cuda")
        raw = b""
    else:
        # Deterministic pattern
        raw = bytes((i * 17 + 3) & 0xFF for i in range(n))
        data = torch.tensor(list(raw), dtype=torch.uint8, device="cuda")
    got = sha1_bytes_cuda(data)
    assert got == _ref_sha1(raw), f"mismatch at n={n}"


@pytest.mark.parametrize("n", [0, 1, 64, 1000, 10000])
@pytest.mark.parametrize("prefix", [b"", b"hello", b"(1, 2, 3)torch.float16"])
def test_sha1_prefix_data_matches_hashlib(n: int, prefix: bytes):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    raw = bytes((i * 31) & 0xFF for i in range(n))
    data = (
        torch.empty(0, dtype=torch.uint8, device="cuda")
        if n == 0
        else torch.tensor(list(raw), dtype=torch.uint8, device="cuda")
    )
    got = sha1_prefix_data_cuda(prefix, data)
    assert got == _ref_sha1(prefix + raw)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
