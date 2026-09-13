"""FP4 schedule metadata and page padding must follow live replay inputs."""

import itertools
import sys

import pytest
import torch
import triton

from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
    _prefill_schedule_prep_kernel,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)


def _reference(ends, parallel_units, smax):
    chunks = [(max(length, 0) + 255) // 256 for length in ends]
    # Exhaustive ascending search is independent of the kernel's binary search.
    safe = next(
        (
            s
            for s in range(1, smax + 1)
            if sum((c + s - 1) // s for c in chunks) <= parallel_units
        ),
        max(max(chunks), 1),
    )
    counts = [(chunk + safe - 1) // safe for chunk in chunks]
    incl = list(itertools.accumulate(counts))
    excl = [prefix - count for prefix, count in zip(incl, counts)]
    return (
        chunks
        + incl
        + excl
        + list(range(len(ends)))
        + [0] * len(ends)
        + [safe, sum(counts)]
    )


@pytest.mark.parametrize(
    "tokens,parallel_units,smax",
    [(1, 1, 1), (17, 17, 2), (257, 512, 1024), (4096, 4096, 1024)],
)
def test_prefill_schedule_replay(tokens, parallel_units, smax):
    # The 17-row case cannot fit within smax, despite P >= T as in production.
    lengths = [
        256 * (4 + i % 11) if smax == 2 else (i * 7919) % 262145 for i in range(tokens)
    ]
    lengths[0] = -17
    updated = [
        256 * (2 + i % 5) if value <= 0 else value // 2
        for i, value in enumerate(lengths)
    ]
    ends = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    storage = torch.full((5 * tokens + 6,), -777, dtype=torch.int32, device="cuda")
    body = storage[2:-2]
    outputs = [body[i * tokens : (i + 1) * tokens] for i in range(5)]
    rows, width = tokens + 2, 257 if tokens == 257 else 5
    padded_width = (width + 3) // 4 * 4 + 4
    source = torch.arange(rows * (width + 7), dtype=torch.int32, device="cuda").view(
        rows, width + 7
    )
    destination = torch.full(
        (rows, padded_width + 11), -777, dtype=torch.int32, device="cuda"
    )
    page, padded = source[:, :width], destination[:, :padded_width]
    expected_page = torch.full(destination.shape, -777, dtype=torch.int32)
    expected_page[:, :padded_width] = 0
    expected_page[:, :width] = page.cpu()

    def launch():
        tiles = triton.cdiv(padded_width, 256)
        _prefill_schedule_prep_kernel[(1 + rows * tiles,)](
            ends,
            *outputs,
            body[5 * tokens :],
            page,
            padded,
            page.stride(0),
            padded.stride(0),
            tokens,
            parallel_units,
            smax,
            width,
            padded_width,
            tiles,
            BLOCK_K=256,
            BLOCK_T=max(16, triton.next_power_of_2(tokens)),
            PT_BLOCK=256,
        )

    launch()  # Compile before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for values in (lengths, updated, [0] * tokens, lengths):
        ends.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
        graph.replay()
        expected = [-777] * 2 + _reference(values, parallel_units, smax) + [-777] * 2
        torch.testing.assert_close(
            storage.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
        )
        # Check every row, including rows beyond T and untouched stride gaps.
        torch.testing.assert_close(destination.cpu(), expected_page, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
