"""A DeepSeek-V4 KV pool in the V4.1 layouts hands the attention kernel buffers
that satisfy its host checks: alignment, page stride, and int32 TMA bounds."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import (
    KVLayout,
    is_valid_kv_layout_pair,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
FULL_SIZE = 4 * PAGE_SIZE
# The int32 budget of the decode kernel's TMA coordinates.
INT32_MAX = 2**31 - 1


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
    """The (num_pages, page_size, 1, bytes_per_token) view handed to the kernel."""
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

    def test_v41_pool_buffers(self):
        for option, expect in ((None, KVLayout.V41_FP4), ("fp8", KVLayout.V41)):
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


if __name__ == "__main__":
    unittest.main()
