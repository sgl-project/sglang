"""Unit tests for the DeepSeek-V4 NVFP4 sparse-MLA KV cache path (SM120/SM121).

Most of this runs on CPU and pins the invariants that the integration silently
depends on. Each one corresponds to a bug that reached a running server during
development, so they are regression tests rather than documentation:

  - ``_nvfp4_stage`` must never move a bucket's storage. A decode CUDA graph
    bakes in the staging address at capture time (captured at the largest
    ``cuda_graph_max_bs``), and a later prefill needs far more rows; a grow-in-
    place scheme frees the tensor the replayed graph still writes through.
  - ``_nvfp4_stage`` must bucket rather than key on the exact row count, or the
    cache grows without bound (prefill row counts follow each chunk's token
    count).
  - Its rows must be zeroed per call: entries the compress kernel skips must
    not be quantized into the pool as stale data.
  - ``nvfp4_cache_view`` must force uint8. ``get_key_buffer`` hands out an
    ``fp8_e4m3`` view under ``--kv-cache-dtype fp8_e4m3`` and the NVFP4 op
    rejects anything but uint8.
  - The platform gate must reject the flag on non-SM120 hardware instead of
    letting the pool allocate an NVFP4 layout no kernel can read.

The slot-addressing round-trip needs the kernels and therefore real SM120
hardware; it is skipped elsewhere. sglang CI has no SM120 runner, so this file
is not registered in run_suite.py.

Run: python3 test/srt/mem_cache/test_dsv4_nvfp4_cache.py
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4.compressor_v2 import CompressorBackendMixin
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    NVFP4_BYTES_PER_TOKEN,
    DeepSeekV4SingleKVPool,
    is_nvfp4_kv_cache,
)


def _is_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class TestNvfp4StageBuffer(unittest.TestCase):
    """The compressed-cache staging buffer, on CPU."""

    def setUp(self):
        self.mixin = CompressorBackendMixin()
        self.dev = torch.device("cpu")

    def test_bucket_storage_is_stable_when_a_larger_request_arrives(self):
        """A decode-sized bucket must survive a later prefill-sized request.

        This is the CUDA-graph hazard: the graph captured at bs=32 keeps writing
        through whatever pointer it saw, so growing that allocation corrupts it.
        """
        rows_small, _ = self.mixin._nvfp4_stage(4, 32, self.dev)
        ptr_small = rows_small.data_ptr()

        # A prefill chunk of 32768 tokens at compress_ratio 4 emits 8192 rows.
        self.mixin._nvfp4_stage(4, 8192, self.dev)

        rows_again, _ = self.mixin._nvfp4_stage(4, 32, self.dev)
        self.assertEqual(
            rows_again.data_ptr(),
            ptr_small,
            "staging storage moved; a replayed CUDA graph would write through a "
            "freed pointer",
        )

    def test_row_counts_are_bucketed_to_powers_of_two(self):
        """Distinct row counts must not each get their own tensor."""
        seen = set()
        for n in (33, 40, 50, 64):
            rows, _ = self.mixin._nvfp4_stage(4, n, self.dev)
            self.assertEqual(rows.shape, (n, 512))
            seen.add(rows.untyped_storage().data_ptr())
        # 33..64 all round up to 64, so one backing allocation serves them all.
        self.assertEqual(len(seen), 1)

        # ... and a count past the bucket does allocate a new one.
        rows, _ = self.mixin._nvfp4_stage(4, 65, self.dev)
        self.assertNotIn(rows.untyped_storage().data_ptr(), seen)

    def test_compress_ratios_do_not_share_a_buffer(self):
        rows_c4, _ = self.mixin._nvfp4_stage(4, 64, self.dev)
        rows_c128, _ = self.mixin._nvfp4_stage(128, 64, self.dev)
        self.assertNotEqual(rows_c4.data_ptr(), rows_c128.data_ptr())

    def test_rows_are_zeroed_on_every_call(self):
        rows, _ = self.mixin._nvfp4_stage(4, 16, self.dev)
        rows.fill_(1.0)
        rows_again, _ = self.mixin._nvfp4_stage(4, 16, self.dev)
        self.assertTrue(torch.equal(rows_again, torch.zeros_like(rows_again)))

    def test_out_loc_is_int64_identity(self):
        """The compress store kernel rejects int32 out_loc."""
        rows, loc = self.mixin._nvfp4_stage(4, 12, self.dev)
        self.assertEqual(loc.dtype, torch.int64)
        self.assertEqual(rows.shape[0], loc.shape[0])
        self.assertTrue(torch.equal(loc, torch.arange(12, dtype=torch.int64)))


class TestNvfp4CacheView(unittest.TestCase):
    """The single view both the append helper and the attention kernel use."""

    def _view(self, buf, page_size):
        pool = types.SimpleNamespace(nvfp4_page_size=page_size)
        return DeepSeekV4SingleKVPool.nvfp4_cache_view(pool, buf)

    def test_abi_is_384_bytes_per_token(self):
        self.assertEqual(NVFP4_BYTES_PER_TOKEN, 384)

    def test_view_shape_and_contiguity(self):
        num_rows, page_size = 8, 64
        buf = torch.zeros(
            num_rows, page_size * NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8
        )
        view = self._view(buf, page_size)
        self.assertEqual(view.shape, (num_rows, page_size, NVFP4_BYTES_PER_TOKEN))
        # The op derives the page split from these strides.
        self.assertEqual(
            view.stride(), (page_size * NVFP4_BYTES_PER_TOKEN, NVFP4_BYTES_PER_TOKEN, 1)
        )

    def test_view_forces_uint8_from_an_fp8_buffer(self):
        """--kv-cache-dtype fp8_e4m3 makes get_key_buffer hand out an fp8 view."""
        page_size = 64
        buf = torch.zeros(4, page_size * NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
        view = self._view(buf.view(torch.float8_e4m3fn), page_size)
        self.assertEqual(view.dtype, torch.uint8)
        self.assertEqual(view.shape, (4, page_size, NVFP4_BYTES_PER_TOKEN))

    def test_extra_cache_page_sizes(self):
        """c4 pools carry page 64 and c128 pools page 2; the op accepts both."""
        for page_size in (2, 64):
            buf = torch.zeros(6, page_size * NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8)
            view = self._view(buf, page_size)
            self.assertEqual(view.shape[1], page_size)
            self.assertEqual(view.numel(), buf.numel())


class TestPlatformGate(unittest.TestCase):
    def tearDown(self):
        is_nvfp4_kv_cache.cache_clear()

    def _gate(self, fmt, sm120):
        is_nvfp4_kv_cache.cache_clear()
        with mock.patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.envs"
        ) as envs, mock.patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.is_sm120_supported",
            return_value=sm120,
        ):
            envs.SGLANG_SM120_KV_CACHE_FORMAT.get.return_value = fmt
            return is_nvfp4_kv_cache()

    def test_default_is_fp8(self):
        self.assertFalse(self._gate("fp8", sm120=True))

    def test_fp8_is_allowed_off_sm120(self):
        self.assertFalse(self._gate("fp8", sm120=False))

    def test_nvfp4_on_sm120(self):
        self.assertTrue(self._gate("nvfp4", sm120=True))

    def test_nvfp4_off_sm120_raises(self):
        with self.assertRaisesRegex(ValueError, "SM120"):
            self._gate("nvfp4", sm120=False)


@unittest.skipUnless(_is_sm120(), "NVFP4 sparse MLA requires SM120/SM121")
class TestSlotAddressingRoundTrip(unittest.TestCase):
    """Append at arbitrary flat slots, read them back through the attention op.

    Writers regroup the flat slot space into 64-token pages and readers index it
    by the same flat ids; this pins that the two agree, which is what makes the
    pool's page regrouping safe.
    """

    def test_round_trip_matches_bf16_reference(self):
        from flashinfer.mla import nvfp4_quantize_append_sparse_mla_cache
        from flashinfer.mla._sparse_mla_nvfp4_sm120 import (
            _sparse_mla_nvfp4_sm120_paged_attention,
        )

        torch.manual_seed(0)
        dev, page, d = "cuda", 64, 512
        num_pages, heads, topk = 64, 64, 128
        num_slots = num_pages * page
        batch = 4

        kv = (torch.randn(num_slots, d, dtype=torch.bfloat16, device=dev) / 10).clamp(
            -1, 1
        )
        cache = torch.zeros(
            num_pages, page, NVFP4_BYTES_PER_TOKEN, dtype=torch.uint8, device=dev
        )

        # Two shuffled chunks, mirroring incremental prefill then decode stores.
        perm = torch.randperm(num_slots, device=dev).to(torch.int32)
        for chunk in (perm[: num_slots // 2], perm[num_slots // 2 :]):
            nvfp4_quantize_append_sparse_mla_cache(
                kv[chunk.long()].contiguous(), chunk, cache
            )

        q = (torch.randn(batch, heads, d, dtype=torch.bfloat16, device=dev) / 10).clamp(
            -1, 1
        )
        idx = torch.randint(0, num_slots, (batch, topk), dtype=torch.int32, device=dev)
        sm_scale = d**-0.5
        sink = torch.zeros(heads, dtype=torch.float32, device=dev)

        out = torch.empty(batch, heads, d, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(batch, heads, dtype=torch.float32, device=dev)
        splits = (topk + 63) // 64
        _sparse_mla_nvfp4_sm120_paged_attention(
            q,
            cache,
            idx,
            out,
            lse,
            sm_scale,
            topk_length=torch.full((batch,), topk, dtype=torch.int32, device=dev),
            attn_sink=sink,
            mid_out=torch.empty(
                batch, heads, splits, d, dtype=torch.bfloat16, device=dev
            ),
            mid_lse=torch.empty(batch, heads, splits, dtype=torch.float32, device=dev),
            use_prefill=False,
        )

        gathered = kv[idx.long()].float()
        scores = torch.einsum("bhd,bkd->bhk", q.float(), gathered) * sm_scale
        scores = torch.cat(
            [scores, sink.view(1, heads, 1).expand(batch, heads, 1)], dim=-1
        )
        ref = torch.einsum("bhk,bkd->bhd", scores.softmax(dim=-1)[..., :topk], gathered)

        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(1), ref.flatten(1), dim=-1
        )
        # NVFP4 is a 4-bit format; this bound catches addressing errors, not
        # quantization error (a slot mismatch collapses cosine to ~0).
        self.assertGreater(cos.min().item(), 0.97)


if __name__ == "__main__":
    unittest.main()
