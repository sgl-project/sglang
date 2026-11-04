"""Expand geometry caching and graph lifetime."""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.lora.common.lora_b import (
    grouped_lora_b,
    slice_geometry,
)
from sglang.kernels.ops.lora.common.route_view import RouteViewKind
from sglang.kernels.ops.lora.common.routing import build_route
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SLOTS, RANK = 4, 32
CONFIG = {
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}
TOL = dict(rtol=2e-2, atol=2e-2)


def _site(offsets, tokens=41, seed=3):
    """A ranked expand over ``offsets`` with a torch reference."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    token_slots = torch.tensor(
        [i % (SLOTS + 1) - 1 for i in range(tokens)], dtype=torch.int32, device=DEVICE
    )
    ranks = torch.tensor([32, 16, 8, 32], dtype=torch.int32, device=DEVICE)
    scalings = torch.tensor([1.5, 0.5, 2.0, 1.0], dtype=torch.float32, device=DEVICE)
    slices = len(offsets) - 1
    bridge = (torch.randn(tokens, slices * RANK, generator=g, device=DEVICE) * 0.5).to(
        DTYPE
    )
    weight = (
        torch.randn(SLOTS, offsets[-1], RANK, generator=g, device=DEVICE) * 0.05
    ).to(DTYPE)
    base = torch.randn(tokens, offsets[-1], generator=g, device=DEVICE, dtype=DTYPE)
    route = build_route(
        token_slots, max_loras=SLOTS, block_size=16, view=RouteViewKind.ALIGNED
    )
    expected = base.float().clone()
    for t in range(tokens):
        s = int(token_slots[t])
        if s < 0:
            continue
        r = int(ranks[s])
        for k in range(slices):
            # slice k of a rank-r slot sits at bridge columns [k*r, (k+1)*r)
            lo, hi = offsets[k], offsets[k + 1]
            expected[t, lo:hi] += float(scalings[s]) * (
                weight[s, lo:hi, :r].float() @ bridge[t, k * r : (k + 1) * r].float()
            )
    return (
        dict(
            bridge=bridge,
            weight=weight,
            base=base,
            route=route,
            ranks=ranks,
            scalings=scalings,
        ),
        expected,
    )


def _expand(site, geometry, out):
    grouped_lora_b(
        site["bridge"],
        site["weight"],
        out,
        site["route"],
        geometry=geometry,
        config=CONFIG,
        add_inplace=True,
        zero_sentinel=False,
        lora_ranks=site["ranks"],
        scalings=site["scalings"],
    )


def test_cache_normalizes_device_and_keeps_distinct_destination_columns():
    rows = (0, 96, 192)
    same = slice_geometry(rows, 64, DEVICE)
    indexed = torch.device("cuda", torch.cuda.current_device())
    assert slice_geometry(rows, 64, indexed, out_offsets=rows[:-1]) is same
    gapped = slice_geometry(rows, 64, indexed, out_offsets=(0, 128))
    assert gapped is not same
    assert gapped.out_offsets.tolist() == [0, 128] and same.out_offsets.tolist() == [
        0,
        96,
    ]


def test_a_captured_graph_keeps_its_tables_when_other_geometries_appear():
    offsets = (0, 96, 224, 304)
    site, expected = _site(offsets, seed=11)
    geometry = slice_geometry(offsets, 64, DEVICE)
    out = site["base"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _expand(site, geometry, out)  # warm-up compiles the kernel
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    out.copy_(site["base"])
    with torch.cuda.graph(graph, stream=stream):
        _expand(site, geometry, out)
    del geometry  # Only the cache retains the tables after capture.
    for i in range(3):
        slice_geometry(
            (0, 128 + 16 * (i + 1), 256 + 32 * (i + 1)), 64, DEVICE
        )  # new entries after capture
        out.copy_(site["base"])
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, **TOL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
