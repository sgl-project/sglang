"""Flat-page round trip for AsymmetricMHATokenToKVPoolHost (head_dim != v_head_dim).

HiCache's `file` storage backend (and the other flat-page backends) moves one page
between the host pool and storage through get_data_page / get_dummy_flat_data_page /
set_from_flat_data_page. These used to raise NotImplementedError for asymmetric
K/V models (e.g. MiMo-V2: K 192 / V 128), so the L3 tier silently did nothing.

CPU-only: builds the host pool buffers directly and round-trips pages through the
real HiCacheFile backend, bit-exact, for both supported layouts.
"""

import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host.mha import AsymmetricMHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE, LAYERS, HEADS, K_DIM, V_DIM, PAGES = 16, 3, 2, 192, 128, 4


def _host_pool(layout: str, fill: bool) -> AsymmetricMHATokenToKVPoolHost:
    # Bypass __init__ (it needs a device pool and pinned memory); set only what
    # the flat-page methods read.
    pool = AsymmetricMHATokenToKVPoolHost.__new__(AsymmetricMHATokenToKVPoolHost)
    pool.layout, pool.page_size = layout, PAGE_SIZE
    pool.layer_num, pool.head_num = LAYERS, HEADS
    pool.head_dim, pool.v_head_dim = K_DIM, V_DIM
    pool.dtype, pool.device, pool.pin_memory = torch.bfloat16, "cpu", False
    if layout == "page_first":
        k_shape = (PAGES * PAGE_SIZE, LAYERS, HEADS, K_DIM)
        v_shape = (PAGES * PAGE_SIZE, LAYERS, HEADS, V_DIM)
    else:  # page_first_direct
        k_shape = (PAGES, LAYERS, PAGE_SIZE, HEADS, K_DIM)
        v_shape = (PAGES, LAYERS, PAGE_SIZE, HEADS, V_DIM)
    make = (
        (lambda s: torch.randn(s).to(torch.bfloat16))
        if fill
        else (lambda s: torch.zeros(s, dtype=torch.bfloat16))
    )
    pool.kv_buffer = (make(k_shape), make(v_shape))
    return pool


class TestAsymmetricMHAFlatPage(CustomTestCase):
    def test_round_trip_through_file_backend(self):
        storage_dir = tempfile.mkdtemp()
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="test",
            extra_config={"file_storage_path": storage_dir},
        )
        storage = HiCacheFile(config)
        flat_numel = PAGE_SIZE * LAYERS * HEADS * (K_DIM + V_DIM)
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                src, dst = _host_pool(layout, True), _host_pool(layout, False)
                for page in range(PAGES):
                    flat = src.get_data_page(page * PAGE_SIZE, flat=True)
                    self.assertEqual(flat.numel(), flat_numel)
                    self.assertTrue(storage.set(f"{layout}-{page}", flat))
                for page in range(PAGES):
                    dst_page = PAGES - 1 - page  # restore into another slot
                    buf = dst.get_dummy_flat_data_page()
                    self.assertEqual(buf.numel(), flat_numel)
                    self.assertIsNotNone(storage.get(f"{layout}-{page}", buf))
                    dst.set_from_flat_data_page(dst_page * PAGE_SIZE, buf)
                    src_k, src_v = src._page_slices(page * PAGE_SIZE)
                    dst_k, dst_v = dst._page_slices(dst_page * PAGE_SIZE)
                    self.assertTrue(torch.equal(src_k, dst_k))
                    self.assertTrue(torch.equal(src_v, dst_v))

    def test_wrong_size_page_is_rejected(self):
        pool = _host_pool("page_first", False)
        with self.assertRaises(ValueError):
            pool.set_from_flat_data_page(0, torch.zeros(5, dtype=torch.bfloat16))

    def test_non_flat_page_is_rejected(self):
        pool = _host_pool("page_first", True)
        with self.assertRaises(NotImplementedError):
            pool.get_data_page(0, flat=False)


if __name__ == "__main__":
    unittest.main()
