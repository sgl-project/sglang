"""Bit-exactness of the page_first_direct HiCache transfers behind ``direct_kernel``.

Device layers are separate allocations reached through a pointer table, and the
host pool is a cudaHostRegister-ed [page, layer, page_bytes] buffer, as in
MLATokenToKVPoolHost / DSAIndexerPoolHost. Indices are token slots, whole pages.
Every case compares whole buffers, so it also checks that bytes outside the
transferred pages are untouched.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache_direct import (
    make_hicache_direct_tables,
    transfer_hicache_direct,
)
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    _cuda_host_unregister,
    alloc_with_host_register,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda(),
    reason="HiCache direct_kernel transfers are CUDA-only.",
)

DEVICE = "cuda"
PAGE_SIZE = 64
# GLM-5.2 fp8 MLA KV page (raw 576 B rows) and DSA indexer page (128 B key + 4 B scale).
KV_PAGE_BYTES = PAGE_SIZE * 576
INDEXER_PAGE_BYTES = PAGE_SIZE * 132
# Host layers per rank: a PP4 stage (19/20), a PP4 last stage with the packed MTP
# layer (21), and DEP4 with the packed MTP layer (79).
LAYER_COUNTS = [19, 21, 79]


def _registered_host(num_pages: int, num_layers: int, page_bytes: int) -> torch.Tensor:
    host = alloc_with_host_register(
        (num_pages, num_layers, page_bytes),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
        allocator=HostTensorAllocator(),
    )
    host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8))
    return host


def _token_indices(pages: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(PAGE_SIZE, dtype=torch.int64, device=pages.device)
    return (pages[:, None] * PAGE_SIZE + offsets).reshape(-1)


class _Pools:
    def __init__(self, *, num_layers, num_pages, page_bytes, seed):
        gen = torch.Generator().manual_seed(seed)
        num_device_pages = 2 * num_pages + 3
        num_host_pages = 2 * num_pages + 5
        # Separate allocations, like the per-layer device KV and indexer buffers.
        self.device_layers = [
            torch.randint(0, 256, (num_device_pages, page_bytes), dtype=torch.uint8).to(
                DEVICE
            )
            for _ in range(num_layers)
        ]
        layer_ptrs = torch.tensor(
            [t.data_ptr() for t in self.device_layers],
            dtype=torch.uint64,
            device=DEVICE,
        )
        self.host = _registered_host(num_host_pages, num_layers, page_bytes)
        self.tables = make_hicache_direct_tables(self.host, layer_ptrs)
        # Device page 0 is the padding page.
        self.device_pages = (
            torch.randperm(num_device_pages - 1, generator=gen)[:num_pages] + 1
        )
        self.host_pages = torch.randperm(num_host_pages, generator=gen)[:num_pages]
        self.device_indices = _token_indices(self.device_pages).to(DEVICE)
        self.host_indices = _token_indices(self.host_pages).to(DEVICE)

    def transfer(self, tables, *, load):
        transfer_hicache_direct(
            tables,
            self.host_indices,
            self.device_indices,
            page_size=PAGE_SIZE,
            load=load,
        )
        torch.cuda.synchronize()

    def release(self):
        _cuda_host_unregister(self.host)


@pytest.fixture
def make_pools():
    created = []

    def make(**kwargs):
        pools = _Pools(**kwargs)
        created.append(pools)
        return pools

    yield make
    torch.cuda.synchronize()
    for pools in created:
        pools.release()


@pytest.mark.parametrize("num_layers", LAYER_COUNTS)
@pytest.mark.parametrize("num_pages", [1, 17])
@pytest.mark.parametrize("page_bytes", [KV_PAGE_BYTES, INDEXER_PAGE_BYTES])
def test_write_back(make_pools, page_bytes, num_pages, num_layers):
    pools = make_pools(
        num_layers=num_layers,
        num_pages=num_pages,
        page_bytes=page_bytes,
        seed=num_layers,
    )
    expected = pools.host.clone()
    for layer, device_layer in enumerate(pools.device_layers):
        expected[pools.host_pages, layer] = device_layer[
            pools.device_pages.to(DEVICE)
        ].cpu()
    pools.transfer(pools.tables, load=False)
    assert torch.equal(pools.host, expected)


@pytest.mark.parametrize("num_layers", LAYER_COUNTS)
@pytest.mark.parametrize("num_pages", [1, 17])
@pytest.mark.parametrize("page_bytes", [KV_PAGE_BYTES, INDEXER_PAGE_BYTES])
def test_load_one_layer(make_pools, page_bytes, num_pages, num_layers):
    pools = make_pools(
        num_layers=num_layers,
        num_pages=num_pages,
        page_bytes=page_bytes,
        seed=num_layers,
    )
    layer_tables = pools.tables.split_layers()
    for layer in (0, num_layers - 1):
        expected = [t.clone() for t in pools.device_layers]
        expected[layer][pools.device_pages.to(DEVICE)] = pools.host[
            pools.host_pages, layer
        ].to(DEVICE)
        pools.transfer(layer_tables[layer], load=True)
        for got, want in zip(pools.device_layers, expected):
            assert torch.equal(got, want)


# 656 B rows are the fp8 FlashMLA layout; 64 x 528 B pages are an odd number of
# 128 B lines, so they exercise the ragged segment split.
@pytest.mark.parametrize("page_bytes", [PAGE_SIZE * 656, PAGE_SIZE * 528])
def test_other_page_sizes_round_trip(make_pools, page_bytes):
    pools = make_pools(num_layers=3, num_pages=5, page_bytes=page_bytes, seed=0)
    originals = [layer.clone() for layer in pools.device_layers]
    pools.transfer(pools.tables, load=False)
    for layer in pools.device_layers:
        layer.zero_()
    pools.transfer(pools.tables, load=True)
    device_pages = pools.device_pages.to(DEVICE)
    for layer, original in zip(pools.device_layers, originals):
        assert torch.equal(layer[device_pages], original[device_pages])


def test_rejects_partial_pages(make_pools):
    """Indices that are not whole pages must not be copied."""
    pools = make_pools(num_layers=3, num_pages=2, page_bytes=KV_PAGE_BYTES, seed=0)
    snapshot = pools.host.clone()
    with pytest.raises(Exception, match="whole pages"):
        transfer_hicache_direct(
            pools.tables,
            pools.host_indices[:-1],
            pools.device_indices[:-1],
            page_size=PAGE_SIZE,
            load=False,
        )
    torch.cuda.synchronize()
    assert torch.equal(pools.host, snapshot)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
