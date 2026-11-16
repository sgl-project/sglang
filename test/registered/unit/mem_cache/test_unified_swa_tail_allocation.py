import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.token_demand import demand_windows
from sglang.test.unified_allocator_fixtures import build_test_unified_pools

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class _AllocationOnlySWAPool(BaseSWAKVPool):
    full_kv_pool = None
    swa_kv_pool = None

    def __init__(self):
        pass

    def register_mapping(self, mapping):
        self.full_to_swa_index_mapping = mapping

    def translate_loc_from_full_to_swa(self, indices):
        return self.full_to_swa_index_mapping[indices]

    def get_state_buf_infos(self):
        return [], [], []

    def get_key_buffer(self, *args, **kwargs):
        raise AssertionError("Allocation must not access KV contents")

    get_value_buffer = get_key_buffer
    get_kv_buffer = get_key_buffer
    set_kv_buffer = get_key_buffer


class TestUnifiedSWATailAllocation(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", enable_unified_memory=True), role="tokenizer"
        )

    def test_flexkv_restore_allocates_window_tail_after_eviction(self):
        """A host restore must bind only its trailing SWA page, including after
        eviction. The unified and separate-buffer allocators accept different
        tail contracts; either mismatch raises before slots can be returned.
        """
        # The external transfer package is not needed to allocate destinations.
        connector = ModuleType("flexkv.integration.sglang.connector")
        connector.FlexKVConnector = object
        connector.FlexKVHostReleaseShim = object
        module_name = "_flexkv_hybrid_allocation_cuda"
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_hybrid_radix_cache.py"
        )
        spec = spec_from_file_location(module_name, path)
        module = module_from_spec(spec)
        with patch.dict(
            "sys.modules", {connector.__name__: connector, module_name: module}
        ):
            spec.loader.exec_module(module)
        cache_type = module.FlexKVHybridRadixCache

        for kind, prefix_len in (
            ("unified", 0),
            ("unified", 18),
            ("separate", 0),
            ("separate", 16),
        ):
            for pressure in (False, True):
                with self.subTest(kind=kind, prefix_len=prefix_len, pressure=pressure):
                    page_size = 16
                    bundle = build_test_unified_pools(
                        device="cuda",
                        kv_cache_dtype=torch.float16,
                        head_num=1,
                        head_dim=8,
                        swa_head_num=1,
                        swa_head_dim=8,
                        page_size=page_size,
                        start_layer=0,
                        end_layer=2,
                        full_attention_layer_ids=[0],
                        swa_attention_layer_ids=[1],
                        total_bytes=1 << 14,
                    )
                    allocator = (
                        bundle.token_to_kv_pool_allocator
                        if kind == "unified"
                        else SWATokenToKVPoolAllocator(
                            size=128,
                            size_swa=128,
                            page_size=page_size,
                            dtype=torch.float16,
                            device="cuda",
                            kvcache=_AllocationOnlySWAPool(),
                            need_sort=False,
                        )
                    )
                    full_available = allocator.full_available_size()
                    swa_available = allocator.swa_available_size()

                    def allocate(count):
                        if not count:
                            return torch.empty(0, dtype=torch.int64, device="cuda")
                        prefix_cpu = torch.tensor([0], dtype=torch.int64)
                        seq_cpu = torch.tensor([count], dtype=torch.int64)
                        return allocator.alloc_extend(
                            prefix_cpu.cuda(),
                            prefix_cpu,
                            seq_cpu.cuda(),
                            seq_cpu,
                            torch.tensor([-1], dtype=torch.int64, device="cuda"),
                            count,
                        )

                    prefix_capacity = -(-prefix_len // page_size) * page_size
                    prefix = allocate(prefix_capacity)[:prefix_len]
                    prefix_swa = allocator.translate_swa_indices_for_transfer(
                        prefix
                    ).clone()
                    blockers = []
                    if pressure:
                        while (block := allocate(page_size)) is not None:
                            blockers.append(block)
                        self.assertTrue(blockers)

                    def evict(_demand):
                        allocator.free(torch.cat(blockers))
                        blockers.clear()

                    cache = cache_type.__new__(cache_type)
                    cache.token_to_kv_pool_allocator = allocator
                    cache.device = torch.device("cuda")
                    cache.page_size = page_size
                    cache._inner_cache = SimpleNamespace(
                        supports_swa=lambda: True,
                        path_device_indices=lambda _node: prefix,
                        window_sub_pools=lambda: tuple(
                            (name, None) for name in allocator.window_pool_names()
                        ),
                        evict_for_alloc=evict,
                    )
                    req = SimpleNamespace(last_node=object(), prefix_len=prefix_len)
                    restored = cache._alloc_restore_slots(req, 64)
                    self.assertIsNotNone(restored)
                    self.assertEqual(restored.numel(), 64)
                    self.assertFalse(blockers)
                    torch.testing.assert_close(
                        allocator.translate_swa_indices_for_transfer(prefix), prefix_swa
                    )
                    swa = allocator.translate_swa_indices_for_transfer(restored)
                    self.assertTrue(bool((swa[-page_size:] > 0).all()))
                    new_pages = restored[
                        page_size - prefix_len % page_size : -2 * page_size
                    ]
                    if new_pages.numel():
                        self.assertTrue(
                            bool(
                                (
                                    allocator.translate_swa_indices_for_transfer(
                                        new_pages
                                    )
                                    == 0
                                ).all()
                            )
                        )
                    allocator.free(torch.cat((prefix, restored)))
                    self.assertEqual(allocator.full_available_size(), full_available)
                    self.assertEqual(allocator.swa_available_size(), swa_available)

    def test_extend_binds_only_new_tail_pages(self):
        """PD tail allocation must leave new FULL-only pages unbound in SWA,
        while preserving an existing partial page and binding the trailing KV."""
        for page_size in (4, 16):
            for prefix_len, seq_len, tail_len in (
                (page_size, 5 * page_size, 0),
                (page_size, 5 * page_size, page_size),
                (page_size, 5 * page_size, 2 * page_size),
                (page_size + 2, 5 * page_size, page_size),
                (page_size + 2, 5 * page_size, 4 * page_size - 2),
                (page_size + 2, 2 * page_size - 1, page_size - 3),
                (page_size + 2, 5 * page_size - 1, page_size),
                (0, None, 1),
            ):
                with self.subTest(
                    page_size=page_size,
                    prefix_len=prefix_len,
                    seq_len=seq_len,
                    tail_len=tail_len,
                ):
                    bundle = build_test_unified_pools(
                        device="cuda",
                        kv_cache_dtype=torch.float16,
                        head_num=1,
                        head_dim=8,
                        v_head_dim=8,
                        swa_head_num=1,
                        swa_head_dim=8,
                        swa_v_head_dim=8,
                        page_size=page_size,
                        start_layer=0,
                        end_layer=2,
                        full_attention_layer_ids=[0],
                        swa_attention_layer_ids=[1],
                        total_bytes=1 << 16,
                        enable_memory_saver=False,
                        need_sort=False,
                    )
                    allocator = bundle.token_to_kv_pool_allocator
                    if seq_len is None:
                        seq_len = allocator.available_size() + page_size
                        self.assertFalse(
                            allocator.admits(
                                KVDemand(
                                    seq_len, demand_windows(allocator, seq_len, seq_len)
                                )
                            )
                        )
                    prefix_capacity = -(-prefix_len // page_size) * page_size
                    prefix = allocator.alloc(prefix_capacity)[:prefix_len]
                    prefix_swa = allocator.translate_swa_indices_for_transfer(
                        prefix
                    ).clone()
                    prefix_cpu = torch.tensor([prefix_len], dtype=torch.int64)
                    seq_cpu = torch.tensor([seq_len], dtype=torch.int64)
                    extended = allocator.alloc_extend_swa_tail(
                        prefix_lens=prefix_cpu.cuda(),
                        prefix_lens_cpu=prefix_cpu,
                        seq_lens=seq_cpu.cuda(),
                        seq_lens_cpu=seq_cpu,
                        last_loc=(
                            prefix[-1:]
                            if prefix_len
                            else torch.tensor([-1], device="cuda")
                        ),
                        extend_num_tokens=seq_len - prefix_len,
                        window_tail_lens=dict.fromkeys(
                            allocator.window_pool_names(), tail_len
                        ),
                    )
                    self.assertIsNotNone(extended)
                    self.assertEqual(extended.numel(), seq_len - prefix_len)
                    tokens = torch.cat((prefix, extended))
                    full_phys = allocator.translate_kv_indices_for_transfer(tokens)
                    swa_phys = allocator.translate_swa_indices_for_transfer(tokens)
                    self.assertTrue(bool((full_phys > 0).all()))
                    self.assertTrue(torch.equal(swa_phys[:prefix_len], prefix_swa))
                    tail_start = seq_len - tail_len
                    new_pages = torch.unique(tokens[prefix_capacity:] // page_size)
                    tail_pages = torch.unique(tokens[tail_start:] // page_size)
                    full_only_pages = new_pages[~torch.isin(new_pages, tail_pages)]
                    self.assertTrue(
                        bool(
                            (allocator.swa_v2p_page_table[full_only_pages] == -1).all()
                        )
                    )
                    if tail_len:
                        pages = allocator.swa_v2p_page_table[
                            tokens[tail_start:] // page_size
                        ]
                        self.assertTrue(bool((pages > 0).all()))
                        expected = pages * page_size + tokens[tail_start:] % page_size
                        self.assertTrue(torch.equal(swa_phys[tail_start:], expected))
                    self.assertEqual(
                        allocator.swa_attn_allocator.allocated_count(),
                        prefix_capacity
                        + torch.isin(new_pages, tail_pages).sum().item() * page_size,
                    )
                    if prefix_len < prefix_capacity:
                        reused_tokens = min(prefix_capacity, seq_len) - prefix_len
                        self.assertTrue(
                            torch.equal(
                                extended[:reused_tokens],
                                prefix[-1]
                                + torch.arange(1, reused_tokens + 1, device="cuda"),
                            )
                        )
                    allocator.free(tokens)
                    self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
                    self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)


if __name__ == "__main__":
    unittest.main()
