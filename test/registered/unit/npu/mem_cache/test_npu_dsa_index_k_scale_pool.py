"""
Unit tests for the MXFP8 index-k scale cache of NPUMLATokenToKVPool
(index_k_scale_buffer / get/set_index_k_scale_buffer), the per-token E8M0
block-scale storage consumed by quant_lightning_indexer (v2) quant_mode 3,
and the host-side mirror in MLATokenToKVPoolHost.
"""

import unittest

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")

HAVE_NPU = is_npu() and torch.npu.is_available()

# Matches the quant_lightning_indexer (v2) PA_BBND constraints enforced by
# _check_quant_lightning_indexer_constraints: page_size in [16, 1024], % 16.
PAGE_SIZE = 64
SIZE = 256  # -> SIZE // PAGE_SIZE + 1 == 5 paged slots (slot 0 is the pad slot)
KV_LORA_RANK = 128
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = KV_LORA_RANK + KV_LORA_RANK // 128 * 4 + QK_ROPE_HEAD_DIM * 2  # 260
INDEX_HEAD_DIM = 128
LAYER_NUM = 2


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestNPUIndexKScaleBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = NPUMLATokenToKVPool(
            size=SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="npu",
            enable_memory_saver=False,
            index_head_dim=INDEX_HEAD_DIM,
            kv_cache_dim=KV_CACHE_DIM,
        )

    def test_scale_buffer_layout(self):
        # One E8M0 byte per 32-element block: per-token tail (k_n, d/64, 2)
        # == (1, 2, 2), stored as raw uint8 bytes.
        buf = self.pool.index_k_scale_buffer
        self.assertEqual(buf.shape, (LAYER_NUM, SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, 2, 2))
        self.assertEqual(buf.dtype, torch.uint8)
        self.assertEqual(buf.device.type, "npu")

    def test_index_k_buffer_layout(self):
        buf = self.pool.index_k_buffer
        self.assertEqual(
            buf.shape, (LAYER_NUM, SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, INDEX_HEAD_DIM)
        )
        self.assertEqual(buf.dtype, torch.float8_e4m3fn)

    def test_hadamard_matrix(self):
        had = self.pool.indexer_hadamard_128
        self.assertEqual(had.shape, (128, 128))
        self.assertEqual(had.dtype, torch.bfloat16)
        self.assertEqual(had.device.type, "npu")

    def test_get_scale_buffer_is_e8m0_view(self):
        out = self.pool.get_index_k_scale_buffer(0)
        self.assertEqual(out.dtype, torch.float8_e8m0fnu)
        self.assertEqual(out.shape, (SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, 2, 2))
        # Same storage as the uint8 buffer (a view, not a copy).
        self.assertEqual(out.data_ptr(), self.pool.index_k_scale_buffer[0].data_ptr())

    def test_set_get_round_trip_across_pages(self):
        # Flat slot indices spanning pages 0, 1 and 2 (slot 0 is the pad slot).
        loc = torch.tensor([1, 31, 32, 63, 64, 191], dtype=torch.int64, device="npu")
        # torch.arange with a uint8 dtype is unsupported on the NPU
        # (aclnnArange); build it on CPU and move it over.
        scale_bytes = torch.arange(loc.numel() * 4, dtype=torch.uint8).to(
            "npu"
        ).view(loc.numel(), 1, 2, 2)
        scale = scale_bytes.view(torch.float8_e8m0fnu)

        self.pool.set_index_k_scale_buffer(0, loc, scale)
        torch.npu.synchronize()

        got = self.pool.get_index_k_scale_buffer(0).view(torch.uint8)
        for i, slot in enumerate(loc.tolist()):
            page, off = divmod(slot, PAGE_SIZE)
            # got[page, off, 0] is (2, 2); scale_bytes[i] is (1, 2, 2) —
            # drop the middle dim or torch.equal fails on the shape mismatch.
            self.assertTrue(
                torch.equal(got[page, off, 0], scale_bytes[i, 0]),
                f"slot {slot} (page {page}, offset {off}): "
                f"{got[page, off, 0].tolist()} != {scale_bytes[i, 0].tolist()}",
            )
        # Untouched slots stay zero.
        self.assertTrue(torch.all(got[0, 0, 0] == 0))
        self.assertTrue(torch.all(got[0, 10, 0] == 0))
        self.assertTrue(torch.all(got[3] == 0))

    def test_set_rejects_multibyte_scale(self):
        loc = torch.tensor([1, 2], dtype=torch.int64, device="npu")
        bad_scale = torch.zeros(2, 1, 1, dtype=torch.float32, device="npu")
        with self.assertRaises(AssertionError):
            self.pool.set_index_k_scale_buffer(0, loc, bad_scale)

    def test_layer_slot_isolation(self):
        loc = torch.tensor([1, 2], dtype=torch.int64, device="npu")
        scale_a = torch.full((2, 1, 2, 2), 0x7F, dtype=torch.uint8, device="npu")
        scale_b = torch.full((2, 1, 2, 2), 0x80, dtype=torch.uint8, device="npu")

        self.pool.set_index_k_scale_buffer(0, loc, scale_a.view(torch.float8_e8m0fnu))
        self.pool.set_index_k_scale_buffer(1, loc, scale_b.view(torch.float8_e8m0fnu))
        torch.npu.synchronize()

        got0 = self.pool.get_index_k_scale_buffer(0).view(torch.uint8)
        got1 = self.pool.get_index_k_scale_buffer(1).view(torch.uint8)
        for slot in loc.tolist():
            page, off = divmod(slot, PAGE_SIZE)
            self.assertTrue(torch.all(got0[page, off, 0] == 0x7F))
            self.assertTrue(torch.all(got1[page, off, 0] == 0x80))
        # No cross-layer contamination: the other layer's pattern bytes must
        # not appear anywhere in this layer's buffer (order-independent: the
        # round-trip test's arange(24) bytes never reach 0x7F/0x80).
        self.assertEqual(int((got0 == 0x80).sum()), 0)
        self.assertEqual(int((got1 == 0x7F).sum()), 0)


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestHostIndexKScaleMirror(unittest.TestCase):
    def test_mirror_matches_device_pool(self):
        device_pool = NPUMLATokenToKVPool(
            size=SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="npu",
            enable_memory_saver=False,
            index_head_dim=INDEX_HEAD_DIM,
            kv_cache_dim=KV_CACHE_DIM,
        )
        host = MLATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=PAGE_SIZE,
            # The host-side scale mirror is only allocated in the
            # page_first_kv_split branch, which is also the layout the NPU
            # production path always uses (npu/utils.py).
            layout="page_first_kv_split",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=device_pool.kv_cache_dim,
        )
        self.assertIsNotNone(host.index_k_scale_buffer)
        # The host mirror is page-first (page, layer, ...) — the same
        # ordering as the host index_k_buffer — because the NPU transfer op
        # transfer_kv_dim_exchange requires device dim0 == host dim1 (layer),
        # unlike the device pool, which is layer-first. The per-token MX
        # scale tail must match the device pool byte-for-byte.
        dev_shape = device_pool.index_k_scale_buffer.shape
        self.assertEqual(
            host.index_k_scale_buffer.shape,
            (dev_shape[1], dev_shape[0], *dev_shape[2:]),
        )
        self.assertEqual(host.index_k_scale_buffer.shape[2:], (PAGE_SIZE, 1, 2, 2))
        self.assertEqual(host.index_k_scale_buffer.dtype, torch.uint8)
        self.assertEqual(host.index_k_scale_buffer.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
