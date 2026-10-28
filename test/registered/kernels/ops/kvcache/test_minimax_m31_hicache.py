"""Exercise M3.1 NVFP4 HiCache through the asynchronous L2 transfer engine."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")
# backend-specific: the HIP build of the NVFP4 host pool's page-row transfer kernel.
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")


def _indices(pages):
    values = torch.cat([torch.arange(page * 128, (page + 1) * 128) for page in pages])
    # Controllers can pass strided index views.
    backing = torch.full((values.numel() * 2,), -1, device="cuda", dtype=torch.int64)
    backing[::2] = values.to("cuda")
    return backing[::2]


@pytest.mark.parametrize("layout", ["layer_first", "page_first", "page_first_direct"])
@pytest.mark.parametrize(
    "backend",
    [
        "kernel",
        pytest.param(
            "direct",
            marks=pytest.mark.skipif(
                torch.version.hip is not None,
                reason="the ROCm sgl_kernel build has no direct range-copy op",
            ),
        ),
    ],
)
def test_nvfp4_async_hicache_round_trip(layout, backend):
    layer_num = 2
    pool = MiniMaxM3Nvfp4KVPool(
        size=768,
        page_size=128,
        dtype=torch.float8_e4m3fn,
        head_num=2,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=list(range(layer_num)),
        disable_value_sparse_layer_ids=list(range(layer_num)),
        device="cuda",
        start_layer=0,
        end_layer=layer_num,
    )
    bytes_per_token = sum(pool.get_kv_size_bytes()) // pool.main_pool.total_slots
    memory = SimpleNamespace(
        hicache_size=1024 * bytes_per_token / 1e9,
        hicache_ratio=2,
        hicache_io_backend=backend,
        hicache_mem_layout=layout,
        hicache_write_policy="write_through",
    )
    with (
        patch.object(assembler, "get_memory", return_value=memory),
        patch.object(
            assembler, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
        ),
        patch.object(assembler, "_get_allocator_type", return_value="default"),
        patch.object(assembler, "HybridCacheController"),
    ):
        group, _ = assembler.build_minimax_sparse_hicache_stack(
            params=CacheInitParams(False, None, None, 128),
            sparse_pool=pool,
            load_cache_event=None,
            storage_backend=None,
        )
    main = group.get_pool(PoolName.KV)
    index = group.get_pool(PoolName.INDEXER)
    try:
        source, target, host_indices = (
            _indices([1, 3]),
            _indices([4, 5]),
            _indices([2, 6]),
        )
        slabs = [pool.main_pool.slab, pool.index_k_pool.slab]
        for slab in slabs:
            slab.random_(0, 256)
        expected = [slab[:, [1, 3]].clone() for slab in slabs]
        engine = L2TransferEngine(backend)
        transfers = [
            L2Transfer(
                entry.host_pool,
                entry.device_pool,
                # Match the controller's per-pool staging index placement.
                host_indices.cpu()
                if entry.host_pool.can_use_write_back_jit
                else host_indices,
                source,
                entry.layer_mapper,
            )
            for entry in group.entries
        ]
        engine.submit_device_to_host(transfers).finish_event.synchronize()
        for slab in slabs:
            slab.zero_()
        transfers = [
            t._replace(host_indices=host_indices, device_indices=target)
            for t in transfers
        ]
        layers_done = []
        with (
            patch.object(
                main, "_transfer_page_indices", wraps=main._transfer_page_indices
            ) as main_prepare,
            patch.object(
                index, "_transfer_page_indices", wraps=index._transfer_page_indices
            ) as index_prepare,
        ):
            completion = engine.submit_host_to_device(
                transfers,
                transfer_layer_id_max=layer_num,
                on_layer_done=layers_done.append,
            )
            completion.finish_event.synchronize()
            assert main_prepare.call_count == index_prepare.call_count == 1
        assert layers_done == list(range(layer_num))
        for before, slab in zip(expected, slabs):
            assert torch.equal(before, slab[:, [4, 5]])
            assert torch.count_nonzero(slab[:, [0, 1, 2, 3, 6]]) == 0
    finally:
        torch.cuda.synchronize()
        for entry in group.entries:
            entry.host_pool.destroy()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
