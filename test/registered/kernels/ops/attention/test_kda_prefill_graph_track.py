"""``kda_track_ssm_state`` with a bf16 SSM pool (``--mamba-ssm-dtype bfloat16``):
mode 1 copies the row's final state slot, mode 2 its fp32 chunk snapshot."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.kda_prefill_graph_track import (
    TRACK_CHUNK_STATE,
    TRACK_FINAL_STATE,
    TRACK_NONE,
    kda_track_ssm_state,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


class TestKdaTrackSsmState(CustomTestCase):
    def test_pool_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                pool = torch.randn(8, 4, 32, 32, device="cuda").to(dtype)
                h_track = torch.randn(3, 4, 32, 32, device="cuda")
                mode = torch.tensor(
                    [TRACK_FINAL_STATE, TRACK_CHUNK_STATE, TRACK_NONE],
                    dtype=torch.int32,
                    device="cuda",
                )
                dst = torch.tensor([5, 6, 7], dtype=torch.int64, device="cuda")
                src = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
                expected = pool.clone()
                expected[5] = pool[1]
                expected[6] = h_track[1].to(dtype)

                kda_track_ssm_state(pool, h_track, mode, dst, src)

                torch.testing.assert_close(pool, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
