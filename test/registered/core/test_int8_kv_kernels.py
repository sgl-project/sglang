import unittest

import torch

from sglang.srt.mem_cache.int8_kv_kernels import (
    dequant_int8_kv,
    gather_dequant_kv_from_pool,
    scatter_quant_kv_to_pool,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestInt8KVKernels(unittest.TestCase):
    def test_dequant_formula_cpu(self):
        q = torch.tensor(
            [[[-128, -1, 0, 127], [10, -10, 20, -20]]], dtype=torch.int8
        )
        scale = torch.tensor([[[0.5], [0.25]]], dtype=torch.float16)
        zp = torch.tensor([[[1.0], [-2.0]]], dtype=torch.float16)

        out = dequant_int8_kv(q, scale, zp, out_dtype=torch.float32)
        expected = q.to(torch.float32) * scale.to(torch.float32) + zp.to(
            torch.float32
        )

        torch.testing.assert_close(out, expected)

    def test_scatter_and_gather_roundtrip_cpu(self):
        cache_k = torch.tensor(
            [
                [[-1.0, -0.5, 0.0, 0.5], [0.1, 0.2, 0.3, 0.4]],
                [[2.0, 1.0, 0.0, -1.0], [-0.4, -0.2, 0.2, 0.4]],
            ],
            dtype=torch.float32,
        )
        cache_v = torch.tensor(
            [
                [[0.5, 0.25, 0.0], [1.0, 0.0, -1.0]],
                [[-2.0, -1.0, 0.0], [0.3, 0.6, 0.9]],
            ],
            dtype=torch.float32,
        )
        loc = torch.tensor([1, 3], dtype=torch.int64)

        k_pool = torch.zeros((5, 2, 4), dtype=torch.int8)
        v_pool = torch.zeros((5, 2, 3), dtype=torch.int8)
        k_scale = torch.zeros((5, 2, 1), dtype=torch.float16)
        k_zp = torch.zeros((5, 2, 1), dtype=torch.float16)
        v_scale = torch.zeros((5, 2, 1), dtype=torch.float16)
        v_zp = torch.zeros((5, 2, 1), dtype=torch.float16)

        scatter_quant_kv_to_pool(
            cache_k=cache_k,
            cache_v=cache_v,
            loc=loc,
            k_int8_pool=k_pool,
            v_int8_pool=v_pool,
            k_scale_pool=k_scale,
            k_zp_pool=k_zp,
            v_scale_pool=v_scale,
            v_zp_pool=v_zp,
        )

        gathered_k = gather_dequant_kv_from_pool(
            loc=loc,
            q_pool=k_pool,
            scale_pool=k_scale,
            zp_pool=k_zp,
            out_dtype=torch.float32,
        )
        gathered_v = gather_dequant_kv_from_pool(
            loc=loc,
            q_pool=v_pool,
            scale_pool=v_scale,
            zp_pool=v_zp,
            out_dtype=torch.float32,
        )

        self.assertEqual(tuple(gathered_k.shape), tuple(cache_k.shape))
        self.assertEqual(tuple(gathered_v.shape), tuple(cache_v.shape))
        torch.testing.assert_close(gathered_k, cache_k, atol=0.02, rtol=0.0)
        torch.testing.assert_close(gathered_v, cache_v, atol=0.02, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
