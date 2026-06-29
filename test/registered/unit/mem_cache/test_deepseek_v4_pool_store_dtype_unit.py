import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-small")

# DSV4 byte-packed KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim
# BF16 (64*2) + nope FP8 scales + scale_pad == 584 bytes/token.
QK_NOPE_HEAD_DIM = 448
QK_ROPE_HEAD_DIM = 64


class TestDeepSeekV4PoolStoreDtype(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm is required for DeepSeekV4SingleKVPool.")
        if is_npu() or is_xpu():
            self.skipTest("DSV4 pool tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def _build_pool(self, dtype: torch.dtype) -> DeepSeekV4SingleKVPool:
        page_size = 64
        return DeepSeekV4SingleKVPool(
            page_size,
            page_size,
            dtype,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            1,
            "cuda",
            False,
        )

    def test_non_fp8_kv_dtype_does_not_break_byte_packed_pool(self):
        # Repro for #25118: under the AMD torch-fallback config the kv-cache
        # dtype is bf16 (not an fp8 enum), which previously left
        # store_dtype == bf16 and crashed create_buffer's uint8 assertion.
        pool = self._build_pool(torch.bfloat16)
        self.assertEqual(pool.store_dtype, torch.uint8)
        self.assertEqual(pool.kv_buffer[0].dtype, torch.uint8)

    def test_fp8_kv_dtype_still_uint8_backed(self):
        pool = self._build_pool(torch.float8_e4m3fn)
        self.assertEqual(pool.store_dtype, torch.uint8)
        self.assertEqual(pool.kv_buffer[0].dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
