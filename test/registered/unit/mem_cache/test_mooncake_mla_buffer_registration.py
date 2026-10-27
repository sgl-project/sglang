import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host import mla as mla_pool_host
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeMLABufferRegistration(unittest.TestCase):
    def make_pool(self, layout, *, indexer=False, scale=False, fp8=False):
        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.layout = layout
        pool.page_num = 3
        pool.page_size = 2
        pool.size = pool.page_num * pool.page_size
        pool.layer_num = 2
        pool.kv_lora_rank = 4
        pool.qk_rope_head_dim = 2
        pool.kv_cache_dim = 6
        pool.dtype = torch.float16
        pool.device = "cpu"
        pool.pin_memory = False
        pool.allocator = None
        pool.dsa_kv_cache_store_fp8 = fp8
        pool.device_pool = SimpleNamespace(
            device="cpu",
            index_head_dim=3 if indexer else None,
            num_indexer_layers=1,
            index_k_scale_buffer=object() if scale else None,
            dsa_kv_cache_store_fp8=fp8,
            kv_cache_dim=8,
        )
        pool.get_ksize_per_token = mock.Mock(return_value=1)

        def allocate(dims, *, dtype, **kwargs):
            return torch.empty(dims, dtype=dtype)

        with (
            mock.patch.dict(mla_pool_host.ALLOC_MEMORY_FUNCS, {"cpu": allocate}),
            mock.patch.object(mla_pool_host, "_is_npu", False),
        ):
            pool.kv_buffer = pool.init_kv_buffer()
        return pool

    def assert_registered(self, pool, buffers):
        store = MooncakeStore.__new__(MooncakeStore)
        store.store = mock.Mock()
        store.store.register_buffer.return_value = 0
        store.register_mem_pool_host(pool)
        expected = [
            mock.call(buffer.data_ptr(), buffer.numel() * buffer.element_size())
            for buffer in buffers
        ]
        self.assertEqual(store.store.register_buffer.call_args_list, expected)
        return [call.args for call in store.store.register_buffer.call_args_list]

    def test_split_buffers_cover_every_zero_copy_page(self):
        for indexer, scale, fp8 in (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        ):
            with self.subTest(indexer=indexer, scale=scale, fp8=fp8):
                pool = self.make_pool(
                    "page_first_kv_split", indexer=indexer, scale=scale, fp8=fp8
                )
                buffers = [pool.k_buffer, pool.v_buffer]
                if indexer:
                    buffers.append(pool.index_k_buffer)
                if scale:
                    buffers.append(pool.index_k_scale_buffer)
                self.assertIs(pool.kv_buffer, pool.k_buffer)
                registered = self.assert_registered(pool, buffers)
                ptrs, sizes = pool.get_page_buffer_meta(torch.arange(pool.size))
                self.assertEqual(len(ptrs), pool.page_num * (2 - fp8 + indexer + scale))
                self.assertEqual(len(ptrs), len(sizes))
                for ptr, size in zip(ptrs, sizes):
                    self.assertTrue(
                        any(
                            base <= ptr and ptr + size <= base + length
                            for base, length in registered
                        ),
                        f"Unregistered page range: {ptr=}, {size=}",
                    )

    def test_ordinary_layouts_register_only_kv_buffer(self):
        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = self.make_pool(layout)
                self.assert_registered(pool, [pool.kv_buffer])

    def test_split_layout_without_optional_buffer_attributes(self):
        pool = self.make_pool("page_first_kv_split")
        del pool.index_k_buffer
        del pool.index_k_scale_buffer
        self.assert_registered(pool, [pool.k_buffer, pool.v_buffer])

    def test_dummy_pool_has_no_buffers_to_register(self):
        for layout in ("page_first", "page_first_kv_split"):
            with self.subTest(layout=layout):
                pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
                pool.layout = layout
                pool.kv_buffer = None
                self.assertEqual(list(MooncakeStore._iter_host_pool_buffers(pool)), [])
                self.assert_registered(pool, [])


if __name__ == "__main__":
    unittest.main()
