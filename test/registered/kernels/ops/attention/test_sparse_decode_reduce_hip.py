"""`gfx_sparse_split_reduce` ignores splits that pa_decode_sparse retired (m = -inf, part_acc unwritten)."""

import unittest

import torch

from sglang.kernels.ops.attention.sparse_decode_reduce_hip import (
    gfx_sparse_split_reduce,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip(), "HIP split reduce")
class TestSparseDecodeReduceHip(CustomTestCase):
    def test_retired_splits_do_not_change_the_output(self):
        torch.manual_seed(0)
        T, S, H, D, live = 3, 4, 16, 512, 2
        part_m = torch.randn(T, S, H, device="cuda")
        part_l = torch.rand(T, S, H, device="cuda") + 0.5
        part_acc = torch.randn(T, S, H, D, device="cuda")
        sink = torch.randn(H, device="cuda")
        want = gfx_sparse_split_reduce(
            part_acc[:, :live].contiguous(),
            part_m[:, :live].contiguous(),
            part_l[:, :live].contiguous(),
            sink,
        )
        part_m[:, live:], part_l[:, live:] = float("-inf"), 0.0
        part_acc[:, live:] = float("nan")
        got = gfx_sparse_split_reduce(part_acc, part_m, part_l, sink)
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
