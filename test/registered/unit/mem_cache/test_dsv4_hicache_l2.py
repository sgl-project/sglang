"""Focused CPU contracts for DeepSeek V4 NPU HiCache L2."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4DecodeCapacity(CustomTestCase):
    def test_decode_capacity_uses_exact_c128_page_demand(self):
        allocator = SimpleNamespace(
            page_size=128,
            c128_attn_allocator=SimpleNamespace(page_size=16),
            evict_to_free_tokens=MagicMock(),
            ensure_c128_capacity=MagicMock(return_value=True),
            full_swa_available_size=MagicMock(return_value=384),
        )
        allocator.c128_num_pages_needed = MethodType(
            DSV4NPUTokenToKVPoolAllocator.c128_num_pages_needed, allocator
        )
        requests = [
            SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=length))
            for length in (127, 2175, 2047)
        ]

        result = DSV4NPUTokenToKVPoolAllocator.check_decode_capacity(
            allocator,
            num_tokens=384,
            tree_cache=MagicMock(),
            requests=requests,
            spec_algorithm=SimpleNamespace(is_some=lambda: False),
        )

        self.assertTrue(result)
        # 127 -> 128 and 2175 -> 2176 open new physical C128 pages;
        # 2047 -> 2048 only fills the current page.
        allocator.ensure_c128_capacity.assert_called_once()
        self.assertEqual(allocator.ensure_c128_capacity.call_args.args[1], 2)
        allocator.evict_to_free_tokens.assert_called_once()

    def test_c128_capacity_evicts_full_leaves_until_enough_pages_exist(self):
        available_pages = 1
        c128_page_size = 16

        def available_size():
            return available_pages * c128_page_size

        def evict(_params):
            nonlocal available_pages
            available_pages += 1
            return SimpleNamespace(num_tokens_evicted=128)

        allocator = SimpleNamespace(
            page_size=128,
            c128_attn_allocator=SimpleNamespace(
                page_size=c128_page_size,
                available_size=available_size,
            ),
        )
        tree_cache = SimpleNamespace(
            is_chunk_cache=lambda: False,
            evict=MagicMock(side_effect=evict),
        )

        result = DSV4NPUTokenToKVPoolAllocator.ensure_c128_capacity(
            allocator, tree_cache, num_pages=3
        )

        self.assertTrue(result)
        self.assertEqual(tree_cache.evict.call_count, 2)
        self.assertEqual(
            [call.args[0].num_tokens for call in tree_cache.evict.call_args_list],
            [128, 128],
        )


class TestDSV4HostViews(CustomTestCase):
    @staticmethod
    def _layout_buffer(raw, layout):
        if layout == "layer_first":
            return [raw[:, layer, :].clone() for layer in range(raw.shape[1])]
        if layout == "page_first":
            return raw.clone()
        return raw.unsqueeze(2).clone()

    @staticmethod
    def _selected_layer(buffer, layout, layer):
        if layout == "layer_first":
            return buffer[layer]
        if layout == "page_first":
            return buffer[:, layer, :]
        return buffer[:, layer, 0, :]

    def test_paged_and_state_host_views_support_every_layout(self):
        pages, layers, page_slots, last_dim = 2, 2, 2, 3
        item_bytes = page_slots * last_dim
        raw = torch.arange(pages * layers * item_bytes, dtype=torch.uint8).reshape(
            pages, layers, item_bytes
        )

        state_dtype = torch.float16
        ring_size = 2
        state_last_dim = 3
        state_page_bytes = ring_size * state_last_dim * state_dtype.itemsize
        state_raw = torch.arange(
            pages * layers * state_page_bytes, dtype=torch.uint8
        ).reshape(pages, layers, state_page_bytes)

        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(pool="paged", layout=layout):
                host = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
                host.pool_name = "c4"
                host.layout = layout
                host.num_host_pages = pages
                host.slot_page_size = page_slots
                host.device_buffers = [
                    torch.empty((4, page_slots, 1, last_dim), dtype=torch.int8)
                    for _ in range(layers)
                ]
                host.kv_buffer = self._layout_buffer(raw, layout)

                view = host._host_page_view(1)
                expected = self._selected_layer(host.kv_buffer, layout, 1).view(
                    torch.int8
                )
                self.assertEqual(view.shape, (pages, 1, page_slots, 1, last_dim))
                self.assertTrue(
                    torch.equal(view.reshape(pages, -1), expected.reshape(pages, -1))
                )

            with self.subTest(pool="state", layout=layout):
                host = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
                host.pool_name = "c4_state"
                host.layout = layout
                host.num_host_pages = pages
                host.ring_size = ring_size
                host.state_page_bytes = state_page_bytes
                host.state_pools = [
                    SimpleNamespace(
                        kv_score_buffer=SimpleNamespace(
                            kv_score=torch.empty(1, dtype=state_dtype)
                        )
                    )
                    for _ in range(layers)
                ]
                host.kv_buffer = self._layout_buffer(state_raw, layout)

                view = host._state_host_page_view(1)
                expected = self._selected_layer(host.kv_buffer, layout, 1).view(
                    state_dtype
                )
                self.assertEqual(view.shape, (pages, 1, ring_size, 1, state_last_dim))
                self.assertTrue(
                    torch.equal(view.reshape(pages, -1), expected.reshape(pages, -1))
                )


class TestC128L2Ownership(CustomTestCase):
    def test_load_back_retains_and_attaches_each_c128_page(self):
        allocator = SimpleNamespace(
            c128_attn_allocator=SimpleNamespace(page_size=4),
            retain_c128_pages=MagicMock(),
        )
        nodes = {
            node_id: SimpleNamespace(
                component_data={
                    ComponentType.C128: SimpleNamespace(
                        host_value=torch.arange(4), value=None
                    )
                }
            )
            for node_id in (1, 2)
        }
        tree_core = SimpleNamespace(
            node_by_id=lambda node_id: nodes[node_id],
            set_component_device_value=MagicMock(),
        )
        component = C128SidecarComponent.__new__(C128SidecarComponent)
        component.cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        component.tree_core = tree_core
        transfer = PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C128,
            device_indices=torch.arange(28, 36),
            nodes_to_load=[1, 2],
        )

        component.commit_hicache_transfer(
            node=None,
            phase=CacheTransferPhase.LOAD_BACK,
            transfers=[transfer],
            cache_actions=[],
        )

        retained = [
            call.args[0].tolist() for call in allocator.retain_c128_pages.call_args_list
        ]
        attached = [
            (call.args[0], call.args[2].tolist())
            for call in tree_core.set_component_device_value.call_args_list
        ]
        self.assertEqual(retained, [[7], [8]])
        self.assertEqual(attached, [(1, [7]), (2, [8])])


class TestDSV4PoolAssembly(CustomTestCase):
    def test_npu_indexer_registers_k_and_scale_as_separate_host_pools(self):
        k_buffer = [torch.empty((2, 32, 1, 4), dtype=torch.int8)]
        scale_buffer = [torch.empty((2, 32, 1, 1), dtype=torch.float16)]
        c4_buffer = [torch.empty((2, 32, 1, 8), dtype=torch.int8)]
        indexer_pool = SimpleNamespace(
            has_npu_storage=True,
            index_k_buffer=k_buffer,
            index_scale_buffer=scale_buffer,
        )
        kvcache = SimpleNamespace(
            _unified_kv=True,
            size=256,
            swa_size=0,
            swa_page_size=256,
            c4_kv_pool=SimpleNamespace(kernel_page_size=32),
            c4_indexer_kv_pool=indexer_pool,
            unified_region_buffers=lambda ratio: (c4_buffer, 256),
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
        mappings = assembler._DeepSeekV4LayerMappings(
            transfer_layer_num=1,
            full={0: 0},
            swa={},
            c4={0: 0},
            c128={},
            c4_state={},
            c4_state_global_layers=[],
        )
        memory = SimpleNamespace(
            hicache_mem_layout="page_first_direct",
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
                layer_mappings=mappings,
            )

        k_host = group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER].host_pool
        scale_host = group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE].host_pool
        self.assertIs(k_host.device_buffers, k_buffer)
        self.assertIs(scale_host.device_buffers, scale_buffer)
        self.assertEqual((k_host.item_bytes, scale_host.item_bytes), (128, 64))
        self.assertEqual((k_host.slot_page_size, scale_host.slot_page_size), (32, 32))
        self.assertIs(built_controller, controller)


if __name__ == "__main__":
    unittest.main()
