"""Unit tests for the unified-KV HiSparse device pool.

These tests cover ``HiSparseUnifiedC4DevicePool`` — the HiSparse C4 device hot
pool used in unified-KV mode (ROCm). Unlike separate-KV, the compressed C4 KV
lives inside ``DeepSeekV4UnifiedKVPool``'s ``rows[swa_pages:]`` (bf16,
``head_dim`` wide); this pool binds per-C4-layer *views* into that region and
reuses the HiSparse index-mapping machinery.

Scope:
- view aliasing onto the unified compressed region (rows[swa_pages:]),
- host-mirror item geometry (bf16 head_dim row),
- the page_size==1 logical->hisparse-device allocation / rollback path,
  mirroring ``test_hisparse_unit.py``'s page_size==1 staging logic.

These are component-level checks of pool construction and index allocation; the
end-to-end write/read swap path is covered by integration runs rather than here.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# Small-scale config for fast CI.
QK_NOPE_HEAD_DIM = 8
QK_ROPE_HEAD_DIM = 4
HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
NUM_SLOTS = 4
SWA_RING_SIZE = 8
NUM_BLOCKS = 3
# K_PER_BLOCK[4] == 32 -> c4 compressed rows per layer.
C4_ROWS = NUM_BLOCKS * 32


def _fake_memory_saver_adapter():
    """Adapter whose .region(...) is a no-op context manager."""
    return SimpleNamespace(region=lambda *a, **k: nullcontext())


class TestHiSparseUnifiedPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA/ROCm is required for HiSparse pool tests.")
        if is_npu() or is_xpu():
            raise unittest.SkipTest("HiSparse tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            raise unittest.SkipTest("CUDA/ROCm not available.")

    def _build_unified_pool(self, stage_ratios):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4UnifiedKVPool,
        )

        return DeepSeekV4UnifiedKVPool(
            stage_ratios=stage_ratios,
            num_slots=NUM_SLOTS,
            num_blocks=NUM_BLOCKS,
            page_size=1,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            device="cuda",
            memory_saver_adapter=_fake_memory_saver_adapter(),
            custom_mem_pool=None,
            swa_ring_size=SWA_RING_SIZE,
        )

    def _build_hisparse_pool(self, unified_pool, stage_ratios, page_size=1):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            HiSparseUnifiedC4DevicePool,
        )

        c4_local_layer_ids = [i for i, r in enumerate(stage_ratios) if r == 4]
        return (
            HiSparseUnifiedC4DevicePool(
                unified_kv_pool=unified_pool,
                c4_local_layer_ids=c4_local_layer_ids,
                page_size=page_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            c4_local_layer_ids,
        )

    # ------------------------------------------------------------------
    # Construction / layer selection
    # ------------------------------------------------------------------
    def test_only_c4_layers_get_views(self):
        """Only ratio==4 layers are bound; c128/c1 layers are skipped."""
        stage_ratios = [4, 128, 4]
        unified = self._build_unified_pool(stage_ratios)
        pool, c4_ids = self._build_hisparse_pool(unified, stage_ratios)

        self.assertEqual(c4_ids, [0, 2])
        self.assertEqual(pool.layer_num, 2)
        self.assertEqual(len(pool.kv_buffer), 2)
        self.assertEqual(pool.size, C4_ROWS)
        for buf in pool.kv_buffer:
            self.assertEqual(tuple(buf.shape), (C4_ROWS, HEAD_DIM))
            self.assertEqual(buf.dtype, torch.bfloat16)

    def test_c4_compress_pages_shrinks_device_region(self):
        """c4_compress_pages shrinks the device C4 region; c128 layers unchanged."""
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4UnifiedKVPool,
        )

        stage_ratios = [4, 128, 4]
        # Shrink the device C4 region below the logical budget (NUM_BLOCKS*32).
        c4_device_rows = C4_ROWS // 2
        unified = DeepSeekV4UnifiedKVPool(
            stage_ratios=stage_ratios,
            num_slots=NUM_SLOTS,
            num_blocks=NUM_BLOCKS,
            page_size=1,
            c4_compress_pages=c4_device_rows,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            device="cuda",
            memory_saver_adapter=_fake_memory_saver_adapter(),
            custom_mem_pool=None,
            swa_ring_size=SWA_RING_SIZE,
        )
        swa_pages = unified.swa_pages

        # c4 layers (0, 2): swa_pages + shrunk device rows.
        for local_id in (0, 2):
            self.assertEqual(
                unified.kv_buffer[local_id].shape[0], swa_pages + c4_device_rows
            )
        # c128 layer (1): unchanged (num_blocks * 1).
        self.assertEqual(unified.kv_buffer[1].shape[0], swa_pages + NUM_BLOCKS * 1)

        # The HiSparse hot pool aliases the shrunk region -> its size follows.
        pool, _ = self._build_hisparse_pool(unified, stage_ratios)
        self.assertEqual(pool.size, c4_device_rows)

    # ------------------------------------------------------------------
    # View aliasing onto rows[swa_pages:]
    # ------------------------------------------------------------------
    def test_views_alias_unified_compressed_region(self):
        """Writes through the hot pool land in unified rows[swa_pages:]."""
        stage_ratios = [4, 4]
        unified = self._build_unified_pool(stage_ratios)
        pool, c4_ids = self._build_hisparse_pool(unified, stage_ratios)

        swa_pages = unified.swa_pages
        self.assertEqual(pool.swa_pages, swa_pages)

        for c, local_id in enumerate(c4_ids):
            # data_ptr of the view must equal the unified buffer's row swa_pages.
            expected_ptr = unified.kv_buffer[local_id][swa_pages].data_ptr()
            self.assertEqual(pool.kv_buffer[c].data_ptr(), expected_ptr)

            # Mutating through the hot pool view shows up in the unified buffer.
            pool.kv_buffer[c][5].fill_(1.25)
            self.assertTrue(
                torch.equal(
                    unified.kv_buffer[local_id][swa_pages + 5],
                    pool.kv_buffer[c][5],
                )
            )
            # SWA region (rows[:swa_pages]) is untouched.
            self.assertTrue(torch.all(unified.kv_buffer[local_id][:swa_pages] == 0))

    # ------------------------------------------------------------------
    # Host-mirror item geometry (bf16 head_dim row)
    # ------------------------------------------------------------------
    def test_host_mirror_geometry(self):
        stage_ratios = [4]
        unified = self._build_unified_pool(stage_ratios)
        for page_size in (1, 2):
            pool, _ = self._build_hisparse_pool(
                unified, stage_ratios, page_size=page_size
            )
            self.assertEqual(pool.kv_cache_total_dim, HEAD_DIM)
            self.assertEqual(pool.store_dtype, torch.bfloat16)
            self.assertEqual(pool.compress_ratio, 4)
            self.assertEqual(
                pool.bytes_per_page_padded,
                page_size * HEAD_DIM * torch.bfloat16.itemsize,
            )

    # ------------------------------------------------------------------
    # page_size==1 logical -> hisparse-device alloc / rollback
    # (mirrors HiSparseTokenToKVPoolAllocator.alloc staging logic)
    # ------------------------------------------------------------------
    def test_page_size_one_alloc_free_mapping(self):
        if not is_hip():
            self.skipTest("page_size==1 alloc path is ROCm-specific")

        from sglang.srt.mem_cache.allocator.paged import (
            PagedTokenToKVPoolAllocator,
        )

        stage_ratios = [4, 4]
        unified = self._build_unified_pool(stage_ratios)
        pool, _ = self._build_hisparse_pool(unified, stage_ratios, page_size=1)

        logical_size = 64
        mapping = torch.cat(
            [
                torch.zeros(
                    logical_size + pool.page_size, dtype=torch.int64, device="cuda"
                ),
                torch.tensor([-1], dtype=torch.int64, device="cuda"),
            ]
        )
        pool.register_mapping(mapping)

        hisparse_alloc = PagedTokenToKVPoolAllocator(
            size=pool.size,
            page_size=pool.page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=pool,
            need_sort=False,
        )
        initial_avail = hisparse_alloc.available_size()

        need_size = 16
        logical_indices = torch.arange(
            1, 1 + need_size, dtype=torch.int64, device="cuda"
        )
        hisparse_indices = hisparse_alloc.alloc(need_size)
        self.assertIsNotNone(hisparse_indices)
        self.assertEqual(len(hisparse_indices), need_size)

        # Staging logic: record logical -> hisparse device mapping.
        pool.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        self.assertLess(hisparse_alloc.available_size(), initial_avail)

        # Roundtrip translate returns what we stored (as int32).
        translated = pool.translate_loc_to_hisparse_device(logical_indices)
        self.assertTrue(torch.equal(translated.long(), hisparse_indices.long()))
        self.assertTrue(
            torch.all(pool.full_to_hisparse_device_index_mapping[logical_indices] > 0)
        )

        # Free + clear mapping -> sizes fully restored, mapping cleared.
        hisparse_alloc.is_not_in_free_group = True
        hisparse_alloc.free(hisparse_indices)
        pool.full_to_hisparse_device_index_mapping[logical_indices] = 0
        self.assertEqual(hisparse_alloc.available_size(), initial_avail)
        self.assertTrue(
            torch.all(pool.full_to_hisparse_device_index_mapping[logical_indices] == 0)
        )

    def test_alloc_oversubscribe_returns_none(self):
        if not is_hip():
            self.skipTest("page_size==1 alloc path is ROCm-specific")

        from sglang.srt.mem_cache.allocator.paged import (
            PagedTokenToKVPoolAllocator,
        )

        stage_ratios = [4]
        unified = self._build_unified_pool(stage_ratios)
        pool, _ = self._build_hisparse_pool(unified, stage_ratios, page_size=1)

        hisparse_alloc = PagedTokenToKVPoolAllocator(
            size=pool.size,
            page_size=pool.page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=pool,
            need_sort=False,
        )
        avail = hisparse_alloc.available_size()
        # Asking for more than capacity must fail gracefully (no leak).
        self.assertIsNone(hisparse_alloc.alloc(avail + 1))
        self.assertEqual(hisparse_alloc.available_size(), avail)


if __name__ == "__main__":
    unittest.main()
