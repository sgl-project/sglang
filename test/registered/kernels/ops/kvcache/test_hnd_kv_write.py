"""HND cache writes preserve K/V with independently strided source layouts."""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import memory_pool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestHndKvWrite(unittest.TestCase):
    def test_source_strides(self):
        num_tokens, num_heads, head_dim, page_size = 5, 4, 32, 4

        def source(layout, dtype):
            shape = (num_tokens, num_heads, head_dim)
            if layout == "contiguous":
                return torch.randn(shape, dtype=dtype, device="cuda")
            if layout == "token":
                return torch.randn(
                    num_tokens * 2, num_heads, head_dim, dtype=dtype, device="cuda"
                )[::2]
            base = torch.randn(
                num_tokens, num_heads, head_dim * 2, dtype=dtype, device="cuda"
            )
            return base[..., :head_dim] if layout == "head" else base[..., ::2]

        layouts = ("contiguous", "token", "head", "dim")
        for dtype in (torch.bfloat16, torch.float16):
            with patch.dict(os.environ, {"SGLANG_USE_HND_KVCACHE": "1"}):
                pool = memory_pool.MHATokenToKVPool(
                    size=16,
                    page_size=page_size,
                    dtype=dtype,
                    head_num=num_heads,
                    head_dim=head_dim,
                    layer_num=1,
                    device="cuda",
                    enable_memory_saver=False,
                )
            self.assertTrue(pool.use_hnd)
            loc = torch.tensor([4, 7, 9, 14, 18], device="cuda")
            for k_layout, v_layout in itertools.product(layouts, repeat=2):
                with self.subTest(dtype=dtype, k=k_layout, v=v_layout):
                    k, v = source(k_layout, dtype), source(v_layout, dtype)
                    pool.k_buffer[0].zero_()
                    pool.v_buffer[0].zero_()
                    expected_k = torch.zeros_like(pool.k_buffer[0])
                    expected_v = torch.zeros_like(pool.v_buffer[0])
                    expected_k[loc // page_size, :, loc % page_size, :] = k
                    expected_v[loc // page_size, :, loc % page_size, :] = v
                    with patch.object(
                        memory_pool,
                        "launch_reshape_and_cache_flash",
                        wraps=memory_pool.launch_reshape_and_cache_flash,
                    ) as fused_write:
                        pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, k, v)
                    expected_calls = int(
                        k_layout in ("contiguous", "token")
                        and v_layout in ("contiguous", "token")
                    )
                    self.assertEqual(fused_write.call_count, expected_calls)
                    torch.testing.assert_close(
                        pool.k_buffer[0], expected_k, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        pool.v_buffer[0], expected_v, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
