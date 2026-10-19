"""HiCache copy kernels on a KV-sharded stripe.

Emulates the N ranks of a shard group on one GPU: each rank's device pool is a
stock pool holding only its owned pages at local rows, and each rank's host
pool is striped like it (pool_host/page_interleave.py). A chain is backed up
with every rank narrowing the transfer to its own pages through the
controller's index path, evicted, re-allocated with the host pages' owners,
and loaded back layer by layer with the real kernels of each host layout.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.page_interleave import (
    page_interleave_host_pool_cls,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

N = 4
PS = 16
PAGES_PER_RANK = 32
LAYERS = 3
DTYPE = torch.bfloat16


def _shard(pool, rank):
    """Give a stock pool the placement passed to its striped host pool.

    Its buffers already have the per-rank geometry.
    """
    pool.placement = PageInterleavePlacement(
        PageShardSpec(
            shard_rank=rank,
            shard_size=N,
            page_size=PS,
            max_prefix_tokens=N * PS,
            chunk_tokens=PS,
        )
    )
    pool.shard_rank = rank
    pool.shard_size = N
    return pool


def _mha_pool(rank):
    return _shard(
        MHATokenToKVPool(
            size=PAGES_PER_RANK * PS,
            page_size=PS,
            dtype=DTYPE,
            head_num=2,
            head_dim=64,
            layer_num=LAYERS,
            device="cuda",
            enable_memory_saver=False,
        ),
        rank,
    )


def _mla_pool(rank):
    return _shard(
        MLATokenToKVPool(
            size=PAGES_PER_RANK * PS,
            page_size=PS,
            dtype=DTYPE,
            kv_lora_rank=128,
            qk_rope_head_dim=64,
            layer_num=LAYERS,
            device="cuda",
            enable_memory_saver=False,
        ),
        rank,
    )


def _buffers(pool):
    if isinstance(pool, MLATokenToKVPool):
        return list(pool.kv_buffer)
    return list(pool.k_buffer) + list(pool.v_buffer)


class TestStripedHostRoundTrip(CustomTestCase):
    def _controller(self, io_backend, layout):
        stub = SimpleNamespace(
            io_backend=io_backend,
            device="cuda",
            mem_pool_host=SimpleNamespace(layout=layout),
        )
        stub.move_indices = lambda h, d: HiCacheController.move_indices(stub, h, d)
        return stub

    def _round_trip(self, make_pool, host_cls, layout, io_backend):
        pools = [make_pool(r) for r in range(N)]
        hosts = []
        for pool in pools:
            kwargs = (
                {"target_layer_num": pool.layer_num}
                if isinstance(pool, MLATokenToKVPool)
                else {}
            )
            hosts.append(
                page_interleave_host_pool_cls(host_cls)(
                    pool.get_device_pool_infos()[0].buffer_info,
                    2.0,
                    0,
                    PS,
                    layout,
                    pin_memory=True,
                    device="cpu",
                    device_capacity_tokens=pool.size,
                    start_layer=pool.start_layer,
                    end_layer=pool.end_layer,
                    placement=pool.placement,
                    **kwargs,
                )
            )
        for h in hosts:
            # Unpin now: a leaked registration fails the next case's (rc 712).
            self.addCleanup(h.destroy)
        controller = self._controller(io_backend, layout)
        allocator = PageInterleavePoolAllocator(
            size=PAGES_PER_RANK * PS,
            physical_page_size=PS,
            shard_size=N,
            dtype=DTYPE,
            device="cuda",
            kvcache=None,
            need_sort=False,
        )
        placement = pools[0].placement

        # Seed: every rank holds its own pages of one rotated chain.
        chain = allocator.alloc_cyclic(3, 9 * PS)
        owners = placement.owner_of(chain)
        expected = [
            torch.randn((chain.numel(),) + buf.shape[1:], device="cuda").to(DTYPE)
            for buf in _buffers(pools[0])
        ]
        for r in range(N):
            rows = placement.local_index(chain[owners == r])
            for buf, want in zip(_buffers(pools[r]), expected):
                buf[rows] = want[owners == r]

        # Backup: mirrored host slots, each rank copies only its stripe.
        start_class = int(chain[0]) // PS % N
        host = hosts[0].alloc(chain.numel(), start_class=start_class)
        for h in hosts[1:]:
            self.assertTrue(
                torch.equal(h.alloc(chain.numel(), start_class=start_class), host)
            )
        for r in range(N):
            h_rows, d_rows = HybridCacheController._move_pool_indices(
                controller,
                hosts[r],
                host,
                chain,
                write_back_jit=hosts[r].can_use_write_back_jit,
            )
            hosts[r].backup_from_device_all_layer(pools[r], h_rows, d_rows, io_backend)
        torch.cuda.synchronize()

        # Evict, churn, and load back onto pages of the host pages' owners.
        allocator.free(chain)
        allocator.alloc_cyclic(0, 5 * PS)
        for pool in pools:
            for buf in _buffers(pool):
                buf.zero_()
        reload = allocator.alloc_cyclic(int(host[0]) // PS % N, host.numel())
        self.assertFalse(torch.equal(reload, chain))
        for r in range(N):
            h_rows, d_rows = HybridCacheController._move_pool_indices(
                controller, hosts[r], host, reload, write_back_jit=False
            )
            for layer in range(LAYERS):
                hosts[r].load_to_device_per_layer(
                    pools[r], h_rows, d_rows, layer, io_backend
                )
        torch.cuda.synchronize()

        owners = placement.owner_of(reload)
        self.assertTrue(torch.equal(owners, placement.owner_of(chain)))
        for r in range(N):
            rows = placement.local_index(reload[owners == r])
            for buf, want in zip(_buffers(pools[r]), expected):
                torch.testing.assert_close(buf[rows], want[owners == r])

    def test_mha_layouts(self):
        for layout, io_backend in (
            ("layer_first", "kernel"),
            ("page_first", "kernel"),
            ("page_head", "kernel"),
            ("layer_first", "direct"),
        ):
            with self.subTest(layout=layout, io_backend=io_backend):
                self._round_trip(_mha_pool, MHATokenToKVPoolHost, layout, io_backend)
                self.doCleanups()

    def test_mla_layouts(self):
        for layout, io_backend in (
            ("layer_first", "kernel"),
            ("page_first", "kernel"),
        ):
            with self.subTest(layout=layout, io_backend=io_backend):
                self._round_trip(_mla_pool, MLATokenToKVPoolHost, layout, io_backend)
                self.doCleanups()


if __name__ == "__main__":
    unittest.main()
