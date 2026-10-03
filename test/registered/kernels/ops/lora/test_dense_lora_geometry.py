"""Expand geometry tables are keyed by rows, columns, tile width, and device.
Tables are built outside capture, shared across streams, and retained for graph lifetime.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.kernels.lora_b import (  # noqa: E402
    SliceGeometry,
    grouped_lora_b,
    slice_geometry,
)
from sglang.srt.lora.kernels.routing import build_route  # noqa: E402
from sglang.srt.lora.route_view import RouteViewKind  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

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
    return dict(
        bridge=bridge,
        weight=weight,
        base=base,
        route=route,
        ranks=ranks,
        scalings=scalings,
    ), expected


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


def test_same_geometry_is_one_entry_across_device_spellings():
    a = slice_geometry((0, 96, 192), 64, torch.device("cuda"))
    b = slice_geometry(
        (0, 96, 192), 64, torch.device("cuda", torch.cuda.current_device())
    )
    assert a is b
    assert isinstance(a, SliceGeometry)
    assert (
        a.num_slices,
        a.num_column_tiles,
        a.uniform_width,
        a.out_stride,
        a.full_tiles,
    ) == (2, 4, 96, 96, False)


def test_distinct_destination_columns_are_distinct_entries():
    rows = (0, 96, 192)
    same = slice_geometry(rows, 64, DEVICE)
    gapped = slice_geometry(rows, 64, DEVICE, out_offsets=(0, 128))
    assert same is not gapped
    assert gapped.out_offsets.tolist() == [0, 128] and same.out_offsets.tolist() == [
        0,
        96,
    ]
    assert gapped.out_stride == 128 and same.out_stride == 96
    irregular = slice_geometry((0, 96, 192, 288), 64, DEVICE, out_offsets=(0, 96, 288))
    assert irregular.out_stride == 0  # the kernels read the table instead


def test_the_two_tables_share_one_cache_line():
    # The expand reads the row table and then its slice's column; one
    # allocation, columns at a 16-byte boundary, keeps the second read a hit.
    for rows, columns in (
        ((0, 512, 640, 768), None),
        ((0, 96, 192, 288), (0, 96, 288)),
        ((0, 64 * 5 + 8), None),
        ((0, 256, 512, 768, 1024), None),
    ):
        g = slice_geometry(rows, 64, DEVICE, out_offsets=columns)
        assert g.slice_offsets.tolist() == list(rows)
        assert g.out_offsets.tolist() == list(columns or rows[:-1])
        gap = g.out_offsets.data_ptr() - g.slice_offsets.data_ptr()
        assert gap == -(-len(rows) // 4) * 16
        assert gap + 4 * g.num_slices <= 128


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_an_index_less_device_means_the_current_device():
    offsets = (0, 64 * 3 + 16)  # a key no other test builds
    with torch.cuda.device(0):
        a = slice_geometry(offsets, 64, torch.device("cuda"))
    with torch.cuda.device(1):
        b = slice_geometry(offsets, 64, torch.device("cuda"))
    assert a is not b
    assert (a.slice_offsets.device.index, b.slice_offsets.device.index) == (0, 1)
    with torch.cuda.device(1):
        assert slice_geometry(offsets, 64, torch.device("cuda")) is b
    assert slice_geometry(offsets, 64, torch.device("cuda", 1)) is b
    assert slice_geometry(offsets, 64, torch.device("cuda", 0)) is a


def test_non_cuda_device_is_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        slice_geometry((0, 64), 64, torch.device("cpu"))


def test_first_use_under_capture_raises_before_any_allocation():
    offsets = (0, 64 * 7 + 8)  # a key no other test builds
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    marker = torch.zeros(1, device=DEVICE)
    with torch.cuda.graph(graph, stream=stream):
        marker.add_(1)  # the capture holds real work before the miss
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            slice_geometry(offsets, 64, DEVICE)
    slice_geometry(offsets, 64, DEVICE)  # eager afterwards is fine


@pytest.mark.parametrize("built_on", ["caller", "side"])
def test_tables_built_on_one_stream_are_read_on_another(built_on):
    """The helper copies its tables with blocking copies, so a consumer stream
    needs no wait on the stream that built them."""
    offsets = (0, 128, 256)
    side = torch.cuda.Stream()
    site, expected = _site(offsets, seed=7 + (built_on == "side"))
    torch.cuda.synchronize()  # the inputs are complete; only the tables' publication is under test
    # A fresh key for this test: a tile width nobody else uses (a power of two, as tl.arange needs).
    tile = 32 if built_on == "caller" else 128
    config = {**CONFIG, "BLOCK_SIZE_N": tile}
    build_stream = torch.cuda.current_stream() if built_on == "caller" else side
    use_stream = side if built_on == "caller" else torch.cuda.current_stream()
    with torch.cuda.stream(build_stream):
        geometry = slice_geometry(offsets, tile, site["weight"].device)
    with torch.cuda.stream(use_stream):  # no wait on build_stream
        out = site["base"].clone()
        grouped_lora_b(
            site["bridge"],
            site["weight"],
            out,
            site["route"],
            geometry=geometry,
            config=config,
            add_inplace=True,
            zero_sentinel=False,
            lora_ranks=site["ranks"],
            scalings=site["scalings"],
        )
    torch.cuda.current_stream().wait_stream(use_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, **TOL)


def test_a_captured_graph_keeps_its_tables_when_other_geometries_appear():
    offsets = (0, 128, 256)
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
