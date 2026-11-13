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
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.deepseek_v4_buffer_info import DSV4StateBufferInfo
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.srt.mem_cache.pool_host import npu_host_memory
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

    def test_host_allocations_round_trip_storage_pages(self):
        def allocate(dims, *, dtype, device, **kwargs):
            return torch.empty(dims, dtype=dtype, device=device)

        for npu in (False, True):
            for layout in ("layer_first", "page_first"):
                with (
                    self.subTest(npu=npu, layout=layout),
                    patch.object(memory_pool_host, "_is_npu", npu),
                    patch.object(memory_pool_host, "_is_xpu", False),
                    patch.object(
                        memory_pool_host, "ALLOC_MEMORY_FUNCS", {"cpu": allocate}
                    ),
                    patch.object(
                        memory_pool_host,
                        "get_allocator_from_storage",
                        return_value=None,
                    ),
                    patch.object(
                        memory_pool_host,
                        "host_memory_preflight",
                        side_effect=lambda lengths, **_: (sum(lengths), 2**30),
                    ),
                    patch.object(
                        memory_pool_host, "make_kernel_ptr_table", return_value=None
                    ),
                    patch.object(
                        DeepSeekV4PagedHostPool, "_init_write_back_staging_buffers"
                    ),
                    patch.object(
                        DeepSeekV4StateHostPool, "_init_write_back_staging_buffers"
                    ),
                ):
                    paged = DeepSeekV4PagedHostPool(
                        "c4",
                        [
                            torch.empty((4, 2, 1, 3), dtype=torch.uint8)
                            for _ in range(2)
                        ],
                        item_bytes=6,
                        num_host_pages=4,
                        slot_page_size=2,
                        layout=layout,
                        pin_memory=False,
                    )
                    state = DeepSeekV4StateHostPool(
                        "state",
                        [
                            self._state_pool(torch.float32, 3).checkpoint_buffer_info()
                            for _ in range(2)
                        ],
                        num_host_pages=4,
                        swa_page_size=2,
                        layout=layout,
                        pin_memory=False,
                    )
                    for pool in (paged, state):
                        width = pool.size_per_token
                        data = torch.arange(2 * width, dtype=torch.uint8)
                        pool.set_from_flat_data_page(2, data)
                        self.assertTrue(torch.equal(pool.get_data_page(2), data))
                        pointers, lengths = pool.get_page_buffer_meta(
                            torch.tensor([2, 3])
                        )
                        self.assertEqual(sum(lengths), data.numel())
                        self.assertEqual(
                            len(pointers), 2 if layout == "layer_first" else 1
                        )
                        if layout == "layer_first":
                            self.assertIsInstance(pool.kv_buffer, list)
                            self.assertEqual(len(pool.kv_buffer), 2)
                            self.assertEqual(len(pool.data_refs), 2)
                            for layer, buffer in enumerate(pool.kv_buffer):
                                self.assertTrue(buffer.is_contiguous())
                                self.assertEqual(tuple(buffer.shape), (4, width))
                                self.assertIs(pool.data_refs[layer], buffer)

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
                    self.assertEqual(host.state_infos[layer].dtype, dtype)
                    if layout != "page_first_direct":
                        view = host._state_host_page_view(layer)
                        self.assertEqual(view.dtype, dtype)
                        shape = (
                            (1, 3, 2, 1, width)
                            if layout == "layer_first"
                            else (3, 1, 2, 1, width)
                        )
                        self.assertEqual(view.shape, shape)
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

    def test_paged_and_state_host_views_support_ascend_layouts(self):
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

        for layout in ("page_first", "layer_first"):
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
                shape = (
                    (1, pages, page_slots, 1, last_dim)
                    if layout == "layer_first"
                    else (pages, 1, page_slots, 1, last_dim)
                )
                self.assertEqual(view.shape, shape)
                self.assertTrue(
                    torch.equal(view.reshape(pages, -1), expected.reshape(pages, -1))
                )

        for layout in ("page_first", "layer_first"):
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
                if layout == "layer_first":
                    shape = (1, pages, ring_size, 1, state_last_dim)
                else:
                    shape = (pages, 1, ring_size, 1, state_last_dim)
                self.assertEqual(view.shape, shape)
                self.assertTrue(
                    torch.equal(view.reshape(pages, -1), expected.reshape(pages, -1))
                )

    def test_paged_ascend_copy_uses_matching_split_operator(self):
        for layout, backup_name, load_name in (
            (
                "page_first",
                "transfer_kv_all_layer_kv_split_lf_pf",
                "transfer_kv_per_layer_kv_split_pf_lf",
            ),
            (
                "layer_first",
                "transfer_kv_all_layer_kv_split",
                "transfer_kv_per_layer_kv_split",
            ),
        ):
            with self.subTest(layout=layout):
                host = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
                host.layout = layout
                host.layer_num = 2
                host.num_host_pages = 4
                host.slot_page_size = 4
                host.page_aligned_only = True
                host.device_buffers = [
                    torch.empty((4, 2, 1, 3), dtype=torch.uint8) for _ in range(2)
                ]
                raw = torch.empty((4, 2, 6), dtype=torch.uint8)
                host.kv_buffer = self._layout_buffer(raw, layout)
                device_indices = torch.arange(8, 12)
                host_indices = torch.arange(4, 8)
                with (
                    patch.object(memory_pool_host, backup_name, create=True) as backup,
                    patch.object(memory_pool_host, load_name, create=True) as load,
                ):
                    host.backup_from_device_all_layer(
                        None, host_indices, device_indices, "kernel_ascend"
                    )
                    host.load_to_device_per_layer(
                        None, host_indices, device_indices, 1, "kernel_ascend"
                    )
                self.assertEqual(backup.call_count, 2)
                self.assertEqual(load.call_count, 1)
                if layout == "page_first":
                    self.assertEqual(backup.call_args.kwargs["host_layer_start"], 0)
                self.assertEqual(backup.call_args.kwargs["page_size"], 2)
                self.assertEqual(load.call_args.kwargs["page_size"], 2)
                self.assertEqual(load.call_args.kwargs["device_layer_id"], 0)
                self.assertEqual(load.call_args.kwargs["host_layer_id"], 0)
                self.assertEqual(
                    load.call_args.kwargs["device_k"].shape, (1, 4, 2, 1, 3)
                )
                expected_host_shape = (
                    (4, 1, 2, 1, 3) if layout == "page_first" else (1, 4, 2, 1, 3)
                )
                self.assertEqual(
                    load.call_args.kwargs["host_k"].shape, expected_host_shape
                )
                self.assertTrue(
                    torch.equal(
                        load.call_args.kwargs["device_indices"], torch.tensor([4, 5])
                    )
                )
                self.assertTrue(
                    torch.equal(
                        load.call_args.kwargs["host_indices"], torch.tensor([2, 3])
                    )
                )

    def test_state_ascend_copy_uses_matching_split_operator(self):
        for layout, backup_name, load_name in (
            (
                "page_first",
                "transfer_kv_all_layer_kv_split_lf_pf",
                "transfer_kv_per_layer_kv_split_pf_lf",
            ),
            (
                "layer_first",
                "transfer_kv_all_layer_kv_split",
                "transfer_kv_per_layer_kv_split",
            ),
        ):
            with self.subTest(layout=layout):
                host = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
                host.layout = layout
                host.layer_num = 2
                host.num_host_pages = 4
                host.swa_page_size = 4
                host.ring_size = 2
                host.state_page_bytes = 12
                host.device_page_views = [
                    torch.empty((4, 12), dtype=torch.uint8) for _ in range(2)
                ]
                host.state_infos = [
                    DSV4StateBufferInfo(
                        pages=pages, ring_size=host.ring_size, dtype=torch.float16
                    )
                    for pages in host.device_page_views
                ]
                raw = torch.empty((4, 2, 12), dtype=torch.uint8)
                host.kv_buffer = self._layout_buffer(raw, layout)
                device_indices = torch.arange(8, 12)
                host_indices = torch.arange(4, 8)
                with (
                    patch.object(memory_pool_host, backup_name, create=True) as backup,
                    patch.object(memory_pool_host, load_name, create=True) as load,
                ):
                    host.backup_from_device_all_layer(
                        None, host_indices, device_indices, "kernel_ascend"
                    )
                    host.load_to_device_per_layer(
                        None, host_indices, device_indices, 1, "kernel_ascend"
                    )
                self.assertEqual(backup.call_count, 2)
                self.assertEqual(load.call_count, 1)
                if layout == "page_first":
                    self.assertEqual(backup.call_args.kwargs["host_layer_start"], 0)
                self.assertEqual(load.call_args.kwargs["page_size"], 2)
                self.assertEqual(
                    load.call_args.kwargs["device_k"].shape, (1, 4, 2, 1, 3)
                )
                self.assertEqual(load.call_args.kwargs["device_k"].dtype, torch.float16)
                expected_host_shape = (
                    (4, 1, 2, 1, 3) if layout == "page_first" else (1, 4, 2, 1, 3)
                )
                self.assertEqual(
                    load.call_args.kwargs["host_k"].shape, expected_host_shape
                )
                self.assertTrue(
                    torch.equal(
                        load.call_args.kwargs["device_indices"], torch.tensor([4, 5])
                    )
                )
                self.assertTrue(
                    torch.equal(
                        load.call_args.kwargs["host_indices"], torch.tensor([2, 3])
                    )
                )

    @staticmethod
    def _paged_pool_info(num_layers, page_bytes):
        return SimpleNamespace(
            layer_ids=tuple(range(num_layers)),
            buffer_info=SimpleNamespace(
                buffers=(object(),) * num_layers, page_bytes=page_bytes
            ),
        )

    def test_dsv4_acl_capacity_declares_paged_pool_bytes(self):
        infos = {PoolName.SWA: self._paged_pool_info(2, 6)}
        backend = MagicMock()
        with (
            patch.object(npu_host_memory, "is_npu", return_value=True),
            patch.object(
                npu_host_memory,
                "get_memory",
                return_value=SimpleNamespace(hicache_mem_layout="page_first"),
            ),
            patch.object(
                npu_host_memory, "get_host_memory_backend", return_value=backend
            ),
            patch.object(
                torch, "npu", SimpleNamespace(current_device=lambda: 0), create=True
            ),
        ):
            npu_host_memory.reserve_dsv4_paged_host_memory(
                infos=infos,
                num_host_pages=4,
                swa_num_host_pages=4,
                c128_num_host_pages=0,
                num_draft_swa_layers=0,
            )
        backend.ensure_capacity.assert_called_once_with(48, 0)

    def test_capacity_declares_actual_paged_and_state_allocations(self):
        def checkpoint_info(num_layers, ring_bytes):
            state = SimpleNamespace(pages=SimpleNamespace(shape=(1, ring_bytes)))
            return SimpleNamespace(
                buffer_info=SimpleNamespace(states=(state,) * num_layers)
            )

        infos = {
            PoolName.SWA: self._paged_pool_info(2, 6),
            PoolName.DEEPSEEK_V4_C4: self._paged_pool_info(2, 4),
            PoolName.DEEPSEEK_V4_C4_INDEXER: self._paged_pool_info(1, 3),
            PoolName.DEEPSEEK_V4_C4_ROPE: self._paged_pool_info(2, 2),
            PoolName.DEEPSEEK_V4_C4_STATE: checkpoint_info(2, 8),
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE: checkpoint_info(2, 16),
            PoolName.DEEPSEEK_V4_C128: self._paged_pool_info(1, 5),
            PoolName.DEEPSEEK_V4_C128_ROPE: self._paged_pool_info(1, 7),
        }
        expected_blocks = {
            "layer_first": [24, 24, 24, 16, 16, 12, 8, 8, 32, 32, 64, 64, 10, 14],
            "page_first": [72, 32, 12, 16, 64, 128, 10, 14],
        }
        for backend_name in ("acl", "memfabric"):
            for layout, blocks in expected_blocks.items():
                backend = MagicMock(name=backend_name)
                with (
                    self.subTest(backend=backend_name, layout=layout),
                    patch.object(npu_host_memory, "is_npu", return_value=True),
                    patch.object(
                        npu_host_memory,
                        "get_memory",
                        return_value=SimpleNamespace(hicache_mem_layout=layout),
                    ),
                    patch.object(
                        npu_host_memory, "get_host_memory_backend", return_value=backend
                    ),
                    patch.object(
                        torch,
                        "npu",
                        SimpleNamespace(current_device=lambda: 0),
                        create=True,
                    ),
                ):
                    npu_host_memory.reserve_dsv4_paged_host_memory(
                        infos=infos,
                        num_host_pages=4,
                        swa_num_host_pages=4,
                        c128_num_host_pages=2,
                        num_draft_swa_layers=1,
                    )
                backend.ensure_capacity.assert_called_once_with(sum(blocks), 0)

    def test_non_npu_skips_host_memory_reservation(self):
        with (
            patch.object(npu_host_memory, "is_npu", return_value=False),
            patch.object(npu_host_memory, "get_memory") as get_memory,
            patch.object(npu_host_memory, "get_host_memory_backend") as get_backend,
        ):
            npu_host_memory.reserve_dsv4_paged_host_memory(
                infos={},
                num_host_pages=0,
                swa_num_host_pages=0,
                c128_num_host_pages=0,
                num_draft_swa_layers=0,
            )
        get_memory.assert_not_called()
        get_backend.assert_not_called()


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


class TestDSV4PoolAssembly(CustomTestCase):
    def test_npu_indexer_registers_k_and_scale_as_separate_host_pools(self):
        k_buffer = [torch.empty((2, 32, 1, 4), dtype=torch.int8)]
        scale_buffer = [torch.empty((2, 32, 1, 1), dtype=torch.float16)]
        c4_buffer = [torch.empty((2, 32, 1, 8), dtype=torch.int8)]
        # Only the unavailable hardware package is stubbed; exercise the real
        # NPU provider, including its native page geometry and scale dtype.
        import sys
        from types import ModuleType

        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
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
