"""A DeepSeek-V4 family KV pool in the V4.1 layouts hands the attention kernel
buffers that satisfy its host checks, and its fused writers round-trip."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import (
    KVLayout,
    is_valid_kv_layout_pair,
)
from sglang.srt.layers.attention.dsv4 import torch_quant as tq
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    resolve_compressed_kv_layout,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
FULL_SIZE = 4 * PAGE_SIZE
# The reader's page-stride unit, and the int32 budget of its TMA coordinates.
INT32_MAX = 2**31 - 1


def _sm100():
    return (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability()[0] >= 10
    )


def _make_pool(ratios, kv_source_layers, kv_layout, compressed_kv_layout=None, **sizes):
    return DeepSeekV4TokenToKVPool(
        max_num_reqs=16,
        swa_size=FULL_SIZE,
        c4_size=sizes.get("c4_size", 0),
        c128_size=sizes.get("c128_size", 0),
        c4_state_pool_size=sizes.get("c4_state_pool_size", 0),
        c128_state_pool_size=sizes.get("c128_state_pool_size", 0),
        page_size=PAGE_SIZE,
        swa_page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        indexer_head_dim=128,
        layer_num=len(ratios),
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=ratios,
        kv_source_layers=kv_source_layers,
        full_size=FULL_SIZE,
        kv_layout=kv_layout,
        compressed_kv_layout=compressed_kv_layout,
    )


def flashmla_view(buf: torch.Tensor, page_size: int, layout: KVLayout) -> torch.Tensor:
    """The ``(num_pages, page_size, 1, bytes_per_token)`` view the backend hands the kernel."""
    bpt = layout.bytes_per_token
    return buf[:, : page_size * bpt].view(buf.shape[0], page_size, 1, bpt)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestV41KVPool(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE_SIZE)
        )

    def assert_kernel_requirements(self, pool, layout):
        """What the decode kernel asserts on a paged cache: 16-byte base, contiguous
        rows, a page stride that is a multiple of its TMA row stride and an int32
        TMA row count."""
        self.assertIs(pool.kv_layout, layout)
        self.assertEqual(pool.get_bytes_per_token(), layout.bytes_per_token)
        self.assertEqual(pool.kv_cache_total_dim, layout.bytes_per_token)
        self.assertEqual(pool.bytes_per_page_padded, layout.page_bytes(pool.page_size))
        self.assertGreaterEqual(
            pool.bytes_per_page_padded, pool.page_size * layout.bytes_per_token
        )
        for buf in pool.kv_buffer:
            self.assertEqual(buf.dtype, torch.uint8)
            self.assertEqual(buf.data_ptr() % 16, 0)
            self.assertEqual(buf.shape[1], pool.bytes_per_page_padded)
            self.assertEqual(buf.stride(0) % layout.page_align, 0)
            view = flashmla_view(buf, pool.page_size, layout)
            self.assertEqual(view.shape[3], layout.bytes_per_token)
            self.assertEqual(view.stride(1), layout.bytes_per_token)
            self.assertEqual(view.stride(0), pool.bytes_per_page_padded)
            self.assertLess(view.stride(0), INT32_MAX)
            self.assertLessEqual(
                buf.shape[0] * (buf.stride(0) // layout.page_align), INT32_MAX
            )

    def test_default_layout_is_unchanged(self):
        pool = _make_pool([0, 0, 2, 1], [2, 3], KVLayout.V4)
        self.assertIs(pool.get_swa_key_layout(), KVLayout.V4)
        self.assertEqual(pool.get_swa_key_bytes_per_token(), 584)
        self.assertEqual(
            pool.swa_kv_pool.bytes_per_page_padded, -(-PAGE_SIZE * 584 // 576) * 576
        )
        for ratio in (1, 2):
            layer_id = pool.sources_by_ratio[ratio][0]
            self.assertIs(pool.get_extra_key_layout(layer_id), KVLayout.V4)
            self.assertEqual(pool.get_extra_key_bytes_per_token(layer_id), 584)

    def test_compressed_layout_resolution(self):
        self.assertIs(resolve_compressed_kv_layout(KVLayout.V4, 1), KVLayout.V4)
        self.assertIs(resolve_compressed_kv_layout(KVLayout.V41, 1), KVLayout.V41_FP4)
        self.assertIs(resolve_compressed_kv_layout(KVLayout.V41, 2), KVLayout.V41_FP4)
        self.assertIs(resolve_compressed_kv_layout(KVLayout.V41, 4), KVLayout.V41)
        self.assertIs(resolve_compressed_kv_layout(KVLayout.V41, 128), KVLayout.V41)
        self.assertIs(
            resolve_compressed_kv_layout(KVLayout.V41, 4, "fp4"), KVLayout.V41_FP4
        )
        self.assertIs(
            resolve_compressed_kv_layout(KVLayout.V41, 1, "fp8"), KVLayout.V41
        )
        with self.assertRaises(AssertionError):
            resolve_compressed_kv_layout(KVLayout.V4, 1, "fp4")
        # The pairs the kernel accepts.
        self.assertTrue(is_valid_kv_layout_pair(KVLayout.V41, KVLayout.V41_FP4))
        self.assertTrue(is_valid_kv_layout_pair(KVLayout.V41, KVLayout.V41))
        self.assertFalse(is_valid_kv_layout_pair(KVLayout.V41, KVLayout.V4))
        self.assertFalse(is_valid_kv_layout_pair(KVLayout.V4, KVLayout.V41_FP4))
        # Page strides of the production page sizes need no padding except c128's.
        self.assertEqual(KVLayout.V41.page_bytes(256), 256 * 528)
        self.assertEqual(KVLayout.V41_FP4.page_bytes(256), 256 * 288)
        self.assertEqual(KVLayout.V41.page_bytes(2), 1536)
        self.assertEqual(KVLayout.V41_FP4.page_bytes(2), 768)

    def test_v41_pool_buffers(self):
        for option, expect in (
            (None, KVLayout.V41_FP4),
            ("fp8", KVLayout.V41),
            ("fp4", KVLayout.V41_FP4),
        ):
            with self.subTest(compressed=option):
                pool = _make_pool([0, 0, 2, 1, 1], [2, 3], KVLayout.V41, option)
                self.assert_kernel_requirements(pool.swa_kv_pool, KVLayout.V41)
                self.assertEqual(pool.get_swa_key_bytes_per_token(), 528)
                for ratio in (1, 2):
                    layer_id = pool.sources_by_ratio[ratio][0]
                    self.assertIs(pool.get_extra_key_layout(layer_id), expect)
                    self.assertEqual(
                        pool.get_extra_key_bytes_per_token(layer_id),
                        expect.bytes_per_token,
                    )
                    self.assertTrue(is_valid_kv_layout_pair(pool.kv_layout, expect))
                    self.assert_kernel_requirements(pool.kv_pools[ratio], expect)
                    self.assertEqual(pool.kv_pools[ratio].page_size, PAGE_SIZE // ratio)
                # A pool of the fp4 layout cannot be the main cache.
                with self.assertRaises(AssertionError):
                    _make_pool([0], [], KVLayout.V41_FP4)

    def test_v41_pool_with_c4_c128(self):
        pool = _make_pool(
            [0, 4, 128],
            [],
            KVLayout.V41,
            c4_size=PAGE_SIZE,
            c128_size=PAGE_SIZE,
            c4_state_pool_size=16,
            c128_state_pool_size=16,
        )
        for ratio, expect in ((4, KVLayout.V41), (128, KVLayout.V41)):
            self.assert_kernel_requirements(pool.kv_pools[ratio], expect)
            self.assertEqual(pool.kv_pools[ratio].page_size, PAGE_SIZE // ratio)
        # The 2-token c128 page is the only production page that pads.
        self.assertEqual(pool.kv_pools[128].bytes_per_page_padded, 1536)

    @unittest.skipUnless(_sm100(), "the V4.1 store kernels are SM100 kernels")
    def test_fused_writers_round_trip(self):
        """SWA write (fp8) and compressed write with in-kernel RoPE (fp4) read back
        through the layout-aware dequant as the reference values."""
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

        pool = _make_pool([0, 0, 2, 1], [2, 3], KVLayout.V41)
        g = torch.Generator(device="cuda").manual_seed(3)
        n = 100
        # SWA: finished (normed, rotated) bf16 rows.
        x = torch.randn(n, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16)
        swa_loc = torch.randperm(FULL_SIZE, generator=g, device="cuda")[:n].to(
            torch.int32
        )
        pool.set_swa_key_buffer_radix_fused(layer_id=0, swa_loc=swa_loc, cache_k=x)
        got = dequantize_k_cache_paged(
            pool.get_swa_key_buffer_radix(0),
            swa_loc,
            pool.swa_page_size,
            layout=pool.get_swa_key_layout(),
        )
        ref = tq.dequantize_k_cache_v41(
            tq.quantize_k_cache_v41(x.view(1, n, HEAD_DIM)), n
        ).view(n, 1, HEAD_DIM)
        self.assertTrue(torch.equal(got, ref))
        # Compressed (fp4): the un-rotated latent plus its freqs; the cache holds
        # exactly fake_quant_compressed_kv(rope_tail(latent)).
        layer_id = pool.sources_by_ratio[1][0]
        latent = torch.randn(
            n, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16
        )
        angles = torch.randn(n, ROPE_DIM // 2, generator=g, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        loc = torch.randperm(FULL_SIZE, generator=g, device="cuda")[:n].to(torch.int64)
        pool.set_extra_key_buffer_fused(
            layer_id=layer_id, loc=loc, cache_k=latent, freqs_cis=freqs
        )
        got = dequantize_k_cache_paged(
            pool.get_extra_key_buffer(layer_id),
            loc,
            pool.get_extra_key_page_size(layer_id),
            layout=pool.get_extra_key_layout(layer_id),
        )
        self.assertTrue(
            torch.equal(
                got.squeeze(1),
                tq.fake_quant_compressed_kv(rope_tail(latent, freqs, ROPE_DIM)),
            )
        )
        # The (fp8 nope, bf16 rope) pack writer is the V4 layout only.
        with self.assertRaises(AssertionError):
            pool.set_swa_key_buffer(0, swa_loc, None)


if __name__ == "__main__":
    unittest.main()
