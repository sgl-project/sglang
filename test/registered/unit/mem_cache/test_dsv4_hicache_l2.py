"""Focused CPU contracts for DeepSeek V4 NPU HiCache L2."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_buffer_info import DSV4StateBufferInfo
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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

    @staticmethod
    def _state_pool(dtype, width, ring_size=2):
        pool = CompressStatePool.__new__(CompressStatePool)
        pool.request_scoped = False
        pool.ring_size = ring_size
        pool.kv_score_buffer = KVAndScore(torch.zeros((8, width), dtype=dtype))
        return pool

    def test_state_binding_preserves_storage_and_per_layer_dtype(self):
        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pools = [
                    self._state_pool(torch.float16, 4),
                    self._state_pool(torch.float32, 2),
                ]
                infos = [pool.checkpoint_buffer_info() for pool in pools]
                host = DeepSeekV4StateHostPool(
                    pool_name="c4",
                    state_infos=infos,
                    num_host_pages=3,
                    swa_page_size=4,
                    layout=layout,
                    pin_memory=False,
                )
                # Rebinding a pool cannot change an already assembled transfer.
                for pool in pools:
                    pool.kv_score_buffer = KVAndScore(torch.zeros((8, 8)))
                for layer, (dtype, width) in enumerate(
                    ((torch.float16, 4), (torch.float32, 2))
                ):
                    view = host._state_host_page_view(layer)
                    self.assertEqual(view.dtype, dtype)
                    self.assertEqual(view.shape, (3, 1, 2, 1, width))
                    self.assertIs(host.device_page_views[layer], infos[layer].pages)
                    host.device_page_views[layer][1].fill_(17)
                    self.assertTrue((pools[layer].kv_score_buffer.kv_score == 0).all())
                pages = host._to_page_indices(torch.tensor([8, 9, 10, 11]))
                self.assertEqual(pages.tolist(), [2])
                self.assertEqual(host._ring_op_indices(pages).tolist(), [4, 5])

    def test_state_host_rejects_incompatible_ring_groups(self):
        first = self._state_pool(torch.float16, 4).checkpoint_buffer_info()
        for second in (
            self._state_pool(torch.float32, 1, ring_size=4),
            self._state_pool(torch.float16, 6),
        ):
            with self.subTest(
                ring=second.ring_size, shape=second.kv_score_buffer.shape
            ):
                with self.assertRaisesRegex(
                    ValueError, "share ring size and slot bytes"
                ):
                    DeepSeekV4StateHostPool(
                        pool_name="c4",
                        state_infos=[first, second.checkpoint_buffer_info()],
                        num_host_pages=3,
                        swa_page_size=4,
                        pin_memory=False,
                    )

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
                host.state_infos = [
                    DSV4StateBufferInfo(
                        pages=torch.empty((4, state_page_bytes), dtype=torch.uint8),
                        ring_size=ring_size,
                        dtype=state_dtype,
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
    def test_eviction_hooks_accept_cache_actions(self):
        pages = torch.tensor([7, 8])
        data = SimpleNamespace(value=pages, host_value=None)
        node = SimpleNamespace(component_data={ComponentType.C128: data})
        component = C128SidecarComponent.__new__(C128SidecarComponent)
        component.tree_core = SimpleNamespace(
            component_evictable_size_={ComponentType.C128: 2}
        )
        device_frees = {ComponentType.C128: []}
        host_frees = {ComponentType.C128: []}
        actions = []

        self.assertEqual(
            component.evict_component(
                node, device_frees, host_frees, cache_actions=actions
            ),
            (0, 0),
        )
        self.assertIsNone(data.value)
        self.assertIs(device_frees[ComponentType.C128][0], pages)
        self.assertEqual(
            component.tree_core.component_evictable_size_[ComponentType.C128], 0
        )

        component.is_evict_device_ongoing = False
        component.evict_device_start(1)
        try:
            self.assertIsNone(
                component.evict_device_next_node(
                    {}, device_frees, host_frees, cache_actions=actions
                )
            )
        finally:
            component.evict_device_end()
        self.assertEqual(actions, [])

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


if __name__ == "__main__":
    unittest.main()
