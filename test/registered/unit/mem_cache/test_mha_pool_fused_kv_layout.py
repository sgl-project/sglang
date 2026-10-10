"""MHATokenToKVPool stores K and V as block-aligned halves of one allocation.

trtllm_mha's fmha_v2 prefill reaches V from k_cache's base pointer via int32
block offsets and falls back to a full-pool copy (with only a warning) when
the layout breaks. This pins the invariant that path relies on.

    python -m pytest test/registered/unit/mem_cache/test_mha_pool_fused_kv_layout.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEAD_NUM = 2
HEAD_DIM = 128
LAYER_NUM = 2


class TestMHAPoolFusedKVLayout(unittest.TestCase):
    def test_v_is_block_aligned_offset_from_k(self):
        for page_size in (16, 32, 64):
            with self.subTest(page_size=page_size):
                pool = MHATokenToKVPool(
                    size=page_size * 7,
                    page_size=page_size,
                    dtype=torch.bfloat16,
                    head_num=HEAD_NUM,
                    head_dim=HEAD_DIM,
                    layer_num=LAYER_NUM,
                    device="cpu",
                    enable_memory_saver=False,
                    enable_alt_stream=False,
                )
                for layer_id in range(LAYER_NUM):
                    k_buf, v_buf = pool.get_kv_buffer(layer_id)
                    # Same paged view and block size the fmha_v2 path computes.
                    k_cache = k_buf.view(-1, page_size, HEAD_NUM, HEAD_DIM)
                    block_bytes = k_cache.stride(0) * k_cache.element_size()
                    delta, rem = divmod(
                        v_buf.data_ptr() - k_buf.data_ptr(), block_bytes
                    )
                    self.assertEqual(rem, 0, f"layer {layer_id}")
                    self.assertGreater(delta, 0, f"layer {layer_id}")


if __name__ == "__main__":
    unittest.main()
