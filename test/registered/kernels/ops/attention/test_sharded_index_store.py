# SPDX-License-Identifier: Apache-2.0
"""Byte-exact dual-store checks against two ordinary CUDA indexer stores.

The act_quant check also exercises the unfused quantize-plus-store route. CUDA
and Triton FP8 tie rounding can differ by one representable value, as in the
ordinary fused-store tests; dual destinations must always match each other.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="The sharded indexer store requires NVIDIA CUDA",
)

PAGE_SIZE = 64
PAGE_BYTES = PAGE_SIZE * 132
SHARDS = [(world, rank) for world in (2, 4, 8) for rank in range(world)]


def _ordinary_reference(key, scratch, rows, cache, loc, world, rank):
    from sglang.kernels.ops.attention.fused_store_index_cache import (
        fused_store_index_k_cache,
    )

    if loc.numel():
        fused_store_index_k_cache(key, scratch, rows, PAGE_SIZE)
    owned = torch.nonzero((loc // PAGE_SIZE) % world == rank).flatten()
    if owned.numel():
        local = loc[owned] // (world * PAGE_SIZE) * PAGE_SIZE + loc[owned] % PAGE_SIZE
        fused_store_index_k_cache(key.index_select(0, owned), cache, local, PAGE_SIZE)


def _case(n, world, rank, strided=False):
    pages = max(1, (n + PAGE_SIZE - 1) // PAGE_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(142 + n + world + rank)
    # Fragmented logical pages, in a different order from their scratch pages.
    order = torch.randperm(pages, generator=generator, device="cuda") * 3 + world
    loc = (
        (order[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE, device="cuda"))
        .flatten()[:n]
        .long()
    )
    rows = torch.arange(n, device="cuda", dtype=torch.int64) + PAGE_SIZE
    key = torch.randn(
        (n, 256 if strided else 128),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if strided:
        key = key[:, ::2]
        loc = torch.stack((loc, loc), dim=1)[:, 0]
        rows = torch.stack((rows, rows), dim=1)[:, 0]
    if n == 255:
        key[::3] = 0
        key[1::3] *= 1e-8
        key[2::3] *= 1e8
    scratch = torch.full(
        (pages + 2, PAGE_BYTES), 0xA5, dtype=torch.uint8, device="cuda"
    )
    cache = torch.full(
        ((pages * 3 + world) // world + 2, PAGE_BYTES),
        0x5A,
        dtype=torch.uint8,
        device="cuda",
    )
    return key, scratch, rows, cache, loc


@pytest.mark.parametrize("world,rank", SHARDS)
@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 255, 1023, 16384])
@pytest.mark.parametrize("strided", [False, True])
def test_dual_store_matches_ordinary_stores(world, rank, n, strided):
    from sglang.kernels.ops.attention.fused_store_index_cache import (
        fused_store_sharded_index_k_cache,
    )

    key, scratch, rows, cache, loc = _case(n, world, rank, strided)
    ref_scratch, ref_cache = scratch.clone(), cache.clone()
    _ordinary_reference(key, ref_scratch, rows, ref_cache, loc, world, rank)
    fused_store_sharded_index_k_cache(
        key, scratch, rows, cache, loc, world, rank, PAGE_SIZE
    )
    torch.cuda.synchronize()
    # Compare whole allocations: includes untouched pages, holes and tail guards.
    torch.testing.assert_close(scratch, ref_scratch, rtol=0, atol=0)
    torch.testing.assert_close(cache, ref_cache, rtol=0, atol=0)


@pytest.mark.parametrize("world,rank", SHARDS)
def test_reserved_page_and_scratch_trash_rows_are_written(world, rank):
    from sglang.kernels.ops.attention.fused_store_index_cache import (
        fused_store_sharded_index_k_cache,
    )

    key, scratch, _, cache, _ = _case(PAGE_SIZE, world, rank)
    # Page zero is owned by rank zero, not skipped. Unplanned/padded logical
    # pages are mapped to a valid scratch trash page by the pool beforehand.
    loc = torch.arange(PAGE_SIZE, dtype=torch.int64, device="cuda")
    rows = loc + (scratch.shape[0] - 1) * PAGE_SIZE
    ref_scratch, ref_cache = scratch.clone(), cache.clone()
    _ordinary_reference(key, ref_scratch, rows, ref_cache, loc, world, rank)
    fused_store_sharded_index_k_cache(
        key, scratch, rows, cache, loc, world, rank, PAGE_SIZE
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(scratch, ref_scratch, rtol=0, atol=0)
    torch.testing.assert_close(cache, ref_cache, rtol=0, atol=0)


def test_graph_replay_reads_updated_inputs_and_locations():
    from sglang.kernels.ops.attention.fused_store_index_cache import (
        fused_store_sharded_index_k_cache,
    )

    world, rank = 4, 2
    key, scratch, rows, cache, loc = _case(255, world, rank)

    def run():
        fused_store_sharded_index_k_cache(
            key, scratch, rows, cache, loc, world, rank, PAGE_SIZE
        )

    # Compile and warm on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        key.mul_(-0.5)
        loc.copy_(loc.roll(1))
        rows.copy_(rows.roll(-1))
        scratch.fill_(0xA5)
        cache.fill_(0x5A)
        ref_scratch, ref_cache = scratch.clone(), cache.clone()
        _ordinary_reference(key, ref_scratch, rows, ref_cache, loc, world, rank)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(scratch, ref_scratch, rtol=0, atol=0)
        torch.testing.assert_close(cache, ref_cache, rtol=0, atol=0)


def _read_rows(buf, rows):
    page = rows // PAGE_SIZE
    offset = rows % PAGE_SIZE
    flat = buf.flatten()
    key_offsets = page[:, None] * PAGE_BYTES + offset[:, None] * 128
    key_bytes = flat[key_offsets + torch.arange(128, device=buf.device)]
    scale_offsets = page * (PAGE_BYTES // 4) + 128 * PAGE_SIZE // 4 + offset
    scales = flat.view(torch.float32)[scale_offsets]
    return key_bytes, scales


@pytest.mark.parametrize("world", [2, 4, 8])
def test_matches_unfused_quantize_and_store_semantics(world):
    from sglang.kernels.ops.attention.dsa.index_buf_accessor import SetKAndS
    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant
    from sglang.kernels.ops.attention.fused_store_index_cache import (
        fused_store_sharded_index_k_cache,
    )

    rank = world - 1
    key, scratch, rows, cache, loc = _case(257, world, rank)
    key[0].zero_()
    key[0, 0] = 2.0
    key[0, 1] = 2**-15  # subnormal FP8 rounding boundary
    ref_scratch, ref_cache = scratch.clone(), cache.clone()
    k, scale = act_quant(key, 128, None)
    pool = SimpleNamespace(index_page_size=PAGE_SIZE)
    SetKAndS.execute(
        pool=pool, buf=ref_scratch, loc=rows, index_k=k, index_k_scale=scale
    )
    owned = torch.nonzero((loc // PAGE_SIZE) % world == rank).flatten()
    local = loc[owned] // (world * PAGE_SIZE) * PAGE_SIZE + loc[owned] % PAGE_SIZE
    if owned.numel():
        SetKAndS.execute(
            pool=pool,
            buf=ref_cache,
            loc=local,
            index_k=k.index_select(0, owned),
            index_k_scale=scale.index_select(0, owned),
        )
    fused_store_sharded_index_k_cache(
        key, scratch, rows, cache, loc, world, rank, PAGE_SIZE
    )
    torch.cuda.synchronize()
    for out, ref, addresses in (
        (scratch, ref_scratch, rows),
        (cache, ref_cache, local),
    ):
        out_k, out_s = _read_rows(out, addresses)
        ref_k, ref_s = _read_rows(ref, addresses)
        torch.testing.assert_close(out_s, ref_s, rtol=1e-6, atol=1e-10)
        out_codes, ref_codes = out_k.to(torch.int16), ref_k.to(torch.int16)
        out_codes = torch.where(out_codes < 128, out_codes, 128 - out_codes)
        ref_codes = torch.where(ref_codes < 128, ref_codes, 128 - ref_codes)
        torch.testing.assert_close(out_codes, ref_codes, rtol=0, atol=1)
        assert (out_codes != ref_codes).float().mean().item() < 0.01


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, *args]))
