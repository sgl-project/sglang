"""Raw FP8 allocation, fused paged write and PD buffer byte descriptors."""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available() and not is_hip(), "CUDA pool required")
class TestTileLangRawFp8Pool(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_raw_write_round_trip_and_buffer_lengths(self):
        for dim_nope, dim_rope in ((256, 0), (512, 0), (512, 64)):
            with self.subTest(dim_nope=dim_nope, dim_rope=dim_rope):
                pool = MLATokenToKVPool(
                    size=128,
                    page_size=64,
                    dtype=torch.float8_e4m3fn,
                    kv_lora_rank=dim_nope,
                    qk_rope_head_dim=dim_rope,
                    layer_num=1,
                    device="cuda",
                    enable_memory_saver=False,
                    use_dsa=True,
                )
                self.assertFalse(pool.dsa_kv_cache_store_fp8)
                positions = torch.tensor(
                    [1, 63, 64, 127], device="cuda", dtype=torch.int64
                )
                nope = torch.randn(4, 1, dim_nope, device="cuda", dtype=torch.bfloat16)
                rope = (
                    torch.randn(4, 1, dim_rope, device="cuda", dtype=torch.bfloat16)
                    if dim_rope
                    else None
                )
                with get_parallel().override(attn_dcp_size=1):
                    pool._write_mla_kv_buffer(pool.kv_buffer[0], positions, nope, rope)
                torch.cuda.synchronize()
                expected = nope if rope is None else torch.cat((nope, rope), dim=-1)
                actual = pool.get_key_buffer(0)[positions].float()
                torch.testing.assert_close(
                    actual, expected.to(torch.float8_e4m3fn).float(), atol=0, rtol=0
                )
                untouched = pool.get_key_buffer(0)[0].float()
                self.assertEqual(torch.count_nonzero(untouched).item(), 0)
                _, lengths, page_bytes = pool.get_contiguous_buf_infos()
                width = dim_nope + dim_rope
                self.assertEqual(lengths, [(128 + 64) * width])
                self.assertEqual(page_bytes, [64 * width])
                del pool


if __name__ == "__main__":
    unittest.main()
