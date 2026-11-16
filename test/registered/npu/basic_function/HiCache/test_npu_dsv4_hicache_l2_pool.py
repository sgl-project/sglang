"""NPU-specific DSV4 HiCache pool assembly contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=10, suite="base-b-test-1-npu-a3")


class TestDSV4PoolAssembly(CustomTestCase):
    def test_npu_indexer_registers_k_and_scale_as_separate_host_pools(self):
        k_buffer = [torch.empty((2, 32, 1, 4), dtype=torch.int8)]
        scale_buffer = [torch.empty((2, 32, 1, 1), dtype=torch.float16)]
        c4_buffer = [torch.empty((2, 32, 1, 8), dtype=torch.int8)]
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
            NPUDeepSeekV4IndexerPool,
            NPUDeepSeekV4SingleKVPool,
        )

        indexer_pool = NPUDeepSeekV4IndexerPool.__new__(NPUDeepSeekV4IndexerPool)
        indexer_pool._kernel_page_size = 32
        indexer_pool.index_k_buffer = k_buffer
        indexer_pool.index_scale_buffer = scale_buffer
        c4_pool = NPUDeepSeekV4SingleKVPool.__new__(NPUDeepSeekV4SingleKVPool)
        c4_pool.store_dtype = torch.bfloat16
        c4_pool.kernel_page_size = 32
        c4_pool.kv_buffer = c4_buffer

        kvcache = SimpleNamespace(
            _unified_kv=True,
            size=256,
            swa_size=0,
            swa_page_size=256,
            c4_kv_pool=c4_pool,
            c4_indexer_kv_pool=indexer_pool,
            start_layer=5,
            stage_layer_ids=(5,),
            get_device_pool_infos=lambda: tuple(
                DevicePoolInfo(
                    pool_name=name,
                    indices_from_pool=PoolName.KV,
                    layer_ids=(5,),
                    buffer_info=region,
                )
                for name, region in (
                    (PoolName.DEEPSEEK_V4_C4, c4_pool.paged_buffer_info()),
                    (
                        PoolName.DEEPSEEK_V4_C4_INDEXER,
                        indexer_pool.indexer_buffer_info().key,
                    ),
                    (
                        PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
                        indexer_pool.indexer_buffer_info().scale,
                    ),
                )
            ),
        )
        params = SimpleNamespace(
            page_size=128,
            token_to_kv_pool_allocator=SimpleNamespace(size_full=256),
            mtp_draft_device_pools=[],
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        memory = SimpleNamespace(
            hicache_mem_layout="page_first",
            hicache_write_policy="write_through",
            hicache_io_backend="kernel_ascend",
            hicache_host_memory_mode="cache",
        )
        logical_pool = SimpleNamespace(
            layout=memory.hicache_mem_layout,
            page_size=128,
            device="cpu",
            size=256,
            logical_size=256,
            can_use_write_back_jit=False,
        )

        def make_host_pool(**kwargs):
            return SimpleNamespace(
                **kwargs,
                page_size=kwargs["slot_page_size"],
                device="cpu",
                size=kwargs["num_host_pages"] * kwargs["slot_page_size"],
                logical_size=kwargs["num_host_pages"] * kwargs["slot_page_size"],
                can_use_write_back_jit=False,
            )

        controller = object()
        with (
            patch.object(assembler, "get_memory", return_value=memory),
            patch.object(
                assembler, "reserve_dsv4_paged_host_memory"
            ) as reserve_host_memory,
            patch.object(
                assembler, "_deepseek_v4_num_host_pages", return_value=(2, 0, 0)
            ),
            patch.object(assembler, "_get_allocator_type", return_value="default"),
            patch.object(assembler, "LogicalHostPool", return_value=logical_pool),
            patch.object(
                assembler,
                "DeepSeekV4PagedHostPool",
                side_effect=make_host_pool,
            ),
            patch.object(assembler, "HybridCacheController", return_value=controller),
        ):
            group, built_controller = assembler.build_deepseek_v4_hicache_stack(
                params=params,
                kvcache=kvcache,
                load_cache_event=None,
                storage_backend=None,
            )

        reserve_host_memory.assert_called_once()
        k_host = group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER].host_pool
        scale_host = group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE].host_pool
        self.assertIs(k_host.device_buffers[0], k_buffer[0])
        self.assertIs(scale_host.device_buffers[0], scale_buffer[0])
        self.assertEqual(indexer_pool.indexer_buffer_info().key.page_slots, 32)
        self.assertEqual((k_host.item_bytes, scale_host.item_bytes), (128, 64))
        self.assertEqual((k_host.slot_page_size, scale_host.slot_page_size), (128, 128))
        self.assertIs(built_controller, controller)


if __name__ == "__main__":
    unittest.main()
