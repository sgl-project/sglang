"""
Unit tests for --indexer-kv-cache-dtype: the
quant_lightning_indexer (v2) quant-mode resolver
(resolve_indexer_quant_mode) and the per-mode NPU index-k pool storage
(quant_mode 1 = token-wise FP8 + FP32 scales, 3 = block-32 MXFP8 + E8M0,
5 = block-32 MXFP4 + E8M0), including the host-mirror mirror/shape
contract of MLATokenToKVPoolHost.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv_cache_dtype import (
    QUANT_MODE_MXFP4,
    QUANT_MODE_MXFP8,
    QUANT_MODE_TOKEN_FP8,
    resolve_indexer_quant_mode,
)
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")

HAVE_NPU = is_npu() and torch.npu.is_available()
HAVE_FP4X2 = hasattr(torch, "float4_e2m1fn_x2")

# Matches the quant_lightning_indexer (v2) PA_BBND constraints enforced by
# _check_quant_lightning_indexer_constraints: page_size in [16, 1024], % 16.
PAGE_SIZE = 64
SIZE = 256  # -> SIZE // PAGE_SIZE + 1 == 5 paged slots (slot 0 is the pad slot)
KV_LORA_RANK = 128
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = KV_LORA_RANK + KV_LORA_RANK // 128 * 4 + QK_ROPE_HEAD_DIM * 2  # 260
INDEX_HEAD_DIM = 128
LAYER_NUM = 2


class TestResolveIndexerQuantMode(unittest.TestCase):
    """CPU-safe resolver tests: --indexer-kv-cache-dtype + resolved main KV
    cache dtype -> quant_lightning_indexer (v2) quant_mode (or None for the
    legacy bf16 npu_lightning_indexer path)."""

    def test_none_inherits_mxfp8_for_fp8_main(self):
        self.assertEqual(
            resolve_indexer_quant_mode(None, torch.float8_e4m3fn),
            QUANT_MODE_MXFP8,
        )

    def test_none_inherits_mxfp8_for_fp8_fnuz_main(self):
        self.assertEqual(
            resolve_indexer_quant_mode(None, torch.float8_e4m3fnuz),
            QUANT_MODE_MXFP8,
        )

    def test_none_disables_quant_indexer_for_bf16_main(self):
        self.assertIsNone(resolve_indexer_quant_mode(None, torch.bfloat16))

    def test_explicit_token_fp8(self):
        self.assertEqual(
            resolve_indexer_quant_mode("fp8_e4m3", torch.float8_e4m3fn),
            QUANT_MODE_TOKEN_FP8,
        )

    def test_explicit_mxfp8(self):
        self.assertEqual(
            resolve_indexer_quant_mode("mxfp8", torch.float8_e4m3fn),
            QUANT_MODE_MXFP8,
        )

    @unittest.skipUnless(HAVE_FP4X2, "torch.float4_e2m1fn_x2 required")
    def test_explicit_mxfp4(self):
        self.assertEqual(
            resolve_indexer_quant_mode("fp4_e2m1", torch.float8_e4m3fn),
            QUANT_MODE_MXFP4,
        )

    @unittest.skipIf(HAVE_FP4X2, "torch.float4_e2m1fn_x2 is available")
    def test_mxfp4_requires_fp4_dtype_support(self):
        with self.assertRaises(ValueError):
            resolve_indexer_quant_mode("fp4_e2m1", torch.float8_e4m3fn)

    def test_explicit_values_require_fp8_main(self):
        for value in ("fp8_e4m3", "mxfp8", "fp4_e2m1"):
            with self.assertRaises(ValueError):
                resolve_indexer_quant_mode(value, torch.bfloat16)

    def test_invalid_value_rejected(self):
        with self.assertRaises(ValueError):
            resolve_indexer_quant_mode("auto", torch.float8_e4m3fn)


if HAVE_NPU:
    from sglang.srt.hardware_backend.npu.memory_pool_npu import (
        NPUMLATokenToKVPool,
    )
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    def _make_pool(indexer_quant_mode=None) -> NPUMLATokenToKVPool:
        return NPUMLATokenToKVPool(
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
            indexer_quant_mode=indexer_quant_mode,
        )

    def _make_host_pool(pool: NPUMLATokenToKVPool) -> MLATokenToKVPoolHost:
        # The host mirror is only allocated in the page_first_kv_split
        # branch, which is also the layout the NPU production path always
        # uses (npu/utils.py).
        return MLATokenToKVPoolHost(
            device_pool=pool,
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="page_first_kv_split",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=pool.kv_cache_dim,
        )


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestIndexerPoolTokenFp8Mode(unittest.TestCase):
    """quant_mode 1: token-wise FP8 E4M3 index-k with one FP32 scale per
    token-head (k_descale PA_BBND (block_num, block_size, k_n))."""

    @classmethod
    def setUpClass(cls):
        cls.pool = _make_pool(indexer_quant_mode=QUANT_MODE_TOKEN_FP8)

    def test_mode_attribute(self):
        self.assertEqual(self.pool.indexer_quant_mode, QUANT_MODE_TOKEN_FP8)

    def test_index_k_buffer_layout(self):
        # Same storage as mode 3: full-width FP8.
        buf = self.pool.index_k_buffer
        self.assertEqual(
            buf.shape, (LAYER_NUM, SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, INDEX_HEAD_DIM)
        )
        self.assertEqual(buf.dtype, torch.float8_e4m3fn)

    def test_scale_buffer_layout(self):
        # One FP32 per token-head: trailing (k_n,) == (1,).
        buf = self.pool.index_k_scale_buffer
        self.assertEqual(buf.shape, (LAYER_NUM, SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1))
        self.assertEqual(buf.dtype, torch.float32)

    def test_hadamard_matrix(self):
        had = self.pool.indexer_hadamard_128
        self.assertEqual(had.shape, (128, 128))
        self.assertEqual(had.dtype, torch.bfloat16)
        self.assertEqual(had.device.type, "npu")

    def test_get_scale_buffer_is_fp32_view(self):
        out = self.pool.get_index_k_scale_buffer(0)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1))
        # Same storage as the FP32 buffer (a view, not a copy).
        self.assertEqual(out.data_ptr(), self.pool.index_k_scale_buffer[0].data_ptr())

    def test_set_get_round_trip_across_pages(self):
        # Flat slot indices spanning pages 0, 1 and 2 (slot 0 is the pad slot).
        loc = torch.tensor([1, 31, 32, 63, 64, 191], dtype=torch.int64, device="npu")
        scale = torch.arange(
            loc.numel(), dtype=torch.float32
        ).to("npu").view(loc.numel(), 1, 1)

        self.pool.set_index_k_scale_buffer(0, loc, scale)
        torch.npu.synchronize()

        got = self.pool.get_index_k_scale_buffer(0)
        for i, slot in enumerate(loc.tolist()):
            page, off = divmod(slot, PAGE_SIZE)
            # got[page, off] is (1,) and scale[i] is (1, 1); compare
            # same-shaped 1-D slices (torch.equal does not broadcast).
            self.assertTrue(
                torch.equal(got[page, off], scale[i, 0]),
                f"slot {slot} (page {page}, offset {off}): "
                f"{got[page, off].tolist()} != {scale[i, 0].tolist()}",
            )
        # Untouched slots stay zero.
        self.assertTrue(torch.all(got[0, 0] == 0))
        self.assertTrue(torch.all(got[0, 10] == 0))
        self.assertTrue(torch.all(got[3] == 0))

    def test_set_get_index_k_round_trip(self):
        loc = torch.tensor([1, 32, 191], dtype=torch.int64, device="npu")
        k = torch.randn(
            loc.numel(), 1, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="npu"
        )
        k_fp8 = k.to(torch.float8_e4m3fn)
        self.pool.set_index_k_buffer(0, loc, k_fp8)
        torch.npu.synchronize()
        got = self.pool.get_index_k_buffer(0).view(torch.uint8)
        ref = k_fp8.view(torch.uint8)
        for i, slot in enumerate(loc.tolist()):
            page, off = divmod(slot, PAGE_SIZE)
            self.assertTrue(torch.equal(got[page, off], ref[i]))
        self.assertTrue(torch.all(got[0, 0] == 0))

    def test_host_mirror_matches_device_pool(self):
        host = _make_host_pool(self.pool)
        # Host index_k mirror: page-first (page, layer, page_size, k_n, d).
        self.assertEqual(
            host.index_k_buffer.shape,
            (SIZE // PAGE_SIZE + 1, LAYER_NUM, PAGE_SIZE, 1, INDEX_HEAD_DIM),
        )
        self.assertEqual(host.index_k_buffer.dtype, torch.float8_e4m3fn)
        # Host scale mirror: page-first with the FP32 tail shape[4:] == ().
        dev_shape = self.pool.index_k_scale_buffer.shape
        self.assertEqual(
            host.index_k_scale_buffer.shape,
            (dev_shape[1], dev_shape[0], *dev_shape[2:]),
        )
        self.assertEqual(host.index_k_scale_buffer.dtype, torch.float32)
        self.assertEqual(host.index_k_scale_buffer.device.type, "cpu")


@unittest.skipUnless(
    HAVE_NPU and HAVE_FP4X2, "Ascend NPU + torch.float4_e2m1fn_x2 required"
)
class TestIndexerPoolMxFP4Mode(unittest.TestCase):
    """quant_mode 5: block-32 MXFP4 (e2m1) index-k, 2 values packed per
    byte, with the same E8M0 (d/64, 2) scale tail as quant_mode 3."""

    @classmethod
    def setUpClass(cls):
        cls.pool = _make_pool(indexer_quant_mode=QUANT_MODE_MXFP4)

    def test_mode_attribute(self):
        self.assertEqual(self.pool.indexer_quant_mode, QUANT_MODE_MXFP4)

    def test_index_k_buffer_is_packed_bytes(self):
        # 2 e2m1 values per byte: half the logical width, raw uint8 storage.
        buf = self.pool.index_k_buffer
        self.assertEqual(
            buf.shape,
            (LAYER_NUM, SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, INDEX_HEAD_DIM // 2),
        )
        self.assertEqual(buf.dtype, torch.uint8)

    def test_get_index_k_buffer_is_fp4_view(self):
        out = self.pool.get_index_k_buffer(0)
        self.assertEqual(out.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(
            out.shape, (SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, INDEX_HEAD_DIM // 2)
        )
        # A view over the byte storage (no copy).
        self.assertEqual(
            out.view(torch.uint8).data_ptr(),
            self.pool.index_k_buffer[0].data_ptr(),
        )

    def test_scale_buffer_layout(self):
        # Same MX E8M0 tail as mode 3: 4 bytes per token.
        buf = self.pool.index_k_scale_buffer
        self.assertEqual(
            buf.shape, (LAYER_NUM, SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, 2, 2)
        )
        self.assertEqual(buf.dtype, torch.uint8)

    def test_get_scale_buffer_is_e8m0_view(self):
        out = self.pool.get_index_k_scale_buffer(0)
        self.assertEqual(out.dtype, torch.float8_e8m0fnu)
        self.assertEqual(out.shape, (SIZE // PAGE_SIZE + 1, PAGE_SIZE, 1, 2, 2))

    def test_set_get_index_k_round_trip_raw_bytes(self):
        loc = torch.tensor([1, 31, 64, 191], dtype=torch.int64, device="npu")
        raw = torch.arange(
            loc.numel() * (INDEX_HEAD_DIM // 2), dtype=torch.uint8
        ).to("npu").view(loc.numel(), 1, INDEX_HEAD_DIM // 2)
        fp4 = torch.zeros(
            loc.numel(),
            1,
            INDEX_HEAD_DIM // 2,
            dtype=torch.float4_e2m1fn_x2,
            device="npu",
        )
        fp4.view(torch.uint8).copy_(raw)

        self.pool.set_index_k_buffer(0, loc, fp4)
        torch.npu.synchronize()

        got = self.pool.get_index_k_buffer(0).view(torch.uint8)
        for i, slot in enumerate(loc.tolist()):
            page, off = divmod(slot, PAGE_SIZE)
            self.assertTrue(
                torch.equal(got[page, off], raw[i]),
                f"slot {slot} (page {page}, offset {off}): bytes differ",
            )
        # Untouched slots stay zero.
        self.assertTrue(torch.all(got[0, 0] == 0))
        self.assertTrue(torch.all(got[3] == 0))

    def test_host_mirror_fences_mxfp4(self):
        # The host mirror/transfer path assumes the pool storage dtype and
        # full index_head_dim width, which packed MXFP4 violates; the pool
        # must refuse to construct instead of corrupting pages silently.
        with self.assertRaises(ValueError):
            _make_host_pool(self.pool)


if __name__ == "__main__":
    unittest.main()
