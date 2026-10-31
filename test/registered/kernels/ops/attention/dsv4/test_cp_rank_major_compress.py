"""Compare CP rank-major input against logical-order compressor input.

Run with one CUDA GPU; the AllGather result is materialized locally:
    python test/registered/kernels/ops/attention/dsv4/test_cp_rank_major_compress.py
Set CP_SIZE=2/4/8 to select one layout (default: test all three).
This is a kernel correctness test, not a distributed/model quality evaluation.
"""

import os
import struct
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorPrefillPlan,
    RankMajorScores,
    compress_forward,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-h100")


def packed_plan(rows, width):
    return torch.tensor(list(b"".join(rows)), dtype=torch.uint8, device="cuda").reshape(
        -1, width
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "CUDA required"
)
class TestCPRankMajorCompress(unittest.TestCase):
    def test_output_and_state(self):
        for cp_size in map(int, os.environ.get("CP_SIZE", "2,4,8").split(",")):
            with self.subTest(cp_size=cp_size):
                self._check_output_and_state(cp_size)

    def _check_output_and_state(self, cp_size):
        torch.manual_seed(1031)
        for actual, padded in ((4096, 4096), (25001, 25008), (50000, 50000)):
            for ratio, dim, online in (
                (4, 128, False),
                (4, 512, False),
                (128, 512, False),
                (128, 512, True),
            ):
                width = (4 if ratio == 4 else 2) * dim
                x = torch.randn(padded, width, device="cuda")
                ranked = (
                    x.view(-1, cp_size, width)
                    .transpose(0, 1)
                    .reshape(padded, width)
                    .contiguous()
                )
                window = 8 if ratio == 4 else 128
                ape = torch.randn(window, dim, device="cuda") * 0.2
                if online:
                    compress = [
                        struct.pack("<IHHii", end, end - 1, 128, -1, -1)
                        for end in range(128, actual + 1, 128)
                    ]
                    write = (
                        [struct.pack("<IHHii", actual, actual - 1, actual % 128, 0, -1)]
                        if actual % 128
                        else []
                    )
                    plan = CompressorPrefillPlan(
                        ratio, packed_plan(compress, 16), packed_plan(write, 16)
                    )
                else:
                    compress = [
                        struct.pack("<IHHii", end, end - 1, max(window - end, 0), 0, 0)
                        for end in range(ratio, actual + 1, ratio)
                    ]
                    write = [
                        struct.pack("<Ii", actual - window + i, i)
                        for i in range(window)
                    ]
                    plan = CompressorPrefillPlan(
                        ratio, packed_plan(compress, 16), packed_plan(write, 8)
                    )
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(
                        actual=actual,
                        ratio=ratio,
                        dim=dim,
                        online=online,
                        state_dtype=dtype,
                    ):
                        shape = (1, 1, 1536) if online else (2, ratio, width)
                        ref_state = torch.zeros(shape, dtype=dtype, device="cuda")
                        state = ref_state.clone()
                        kwargs = dict(
                            head_dim=dim, compress_ratio=ratio, is_online=online
                        )
                        reference = compress_forward(ref_state, x, ape, plan, **kwargs)
                        result = compress_forward(
                            state, RankMajorScores(ranked, cp_size), ape, plan, **kwargs
                        )
                        torch.testing.assert_close(result, reference, atol=0, rtol=0)
                        torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
