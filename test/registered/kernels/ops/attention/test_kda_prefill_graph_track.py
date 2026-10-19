"""Track copies of the CUDA-graph captured KDA extend against a torch reference.

Mode 1 rows copy the row's own pool slot, mode 2 rows copy their fp32
chunk-boundary snapshot, mode 0 rows leave their destination untouched. The
pool is checked in fp32 and bf16 (--mamba-ssm-dtype bfloat16).
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.kda_prefill_graph_track import (
    TRACK_CHUNK_STATE,
    TRACK_FINAL_STATE,
    TRACK_NONE,
    kda_track_ssm_state,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKdaTrackSsmState(CustomTestCase):
    def _check(self, pool_dtype):
        torch.manual_seed(0)
        slots, heads, dim = 12, 4, 128
        pool = torch.randn(slots, heads, dim, dim, device="cuda").to(pool_dtype)
        h_track = torch.randn(4, heads, dim, dim, device="cuda", dtype=torch.float32)
        mode = torch.tensor(
            [TRACK_FINAL_STATE, TRACK_CHUNK_STATE, TRACK_NONE, TRACK_CHUNK_STATE],
            device="cuda",
            dtype=torch.int32,
        )
        dst = torch.tensor([8, 9, 10, 11], device="cuda", dtype=torch.int64)
        cache_indices = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.int32)
        expected = pool.clone()
        expected[8] = pool[1]
        expected[9] = h_track[1].to(pool_dtype)
        expected[11] = h_track[3].to(pool_dtype)

        kda_track_ssm_state(pool, h_track, mode, dst, cache_indices)
        torch.cuda.synchronize()
        torch.testing.assert_close(pool, expected, rtol=0, atol=0)

    def test_fp32_pool(self):
        self._check(torch.float32)

    def test_bf16_pool(self):
        self._check(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
