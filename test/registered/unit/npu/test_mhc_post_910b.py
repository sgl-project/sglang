import unittest

import torch
from sglang.srt.hardware_backend.npu.dsv4.mhc_post import post_910b
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMHCPost(unittest.TestCase):
    def test_chunk_boundaries_and_comb_orientation(self):
        rng = torch.Generator().manual_seed(20)
        for rows in (0, 1, 512, 513, 2048):
            x = torch.randn(rows, 32, generator=rng).bfloat16()
            residual = torch.randn(rows, 4, 32, generator=rng).bfloat16()
            comb = torch.randn(rows, 4, 4, generator=rng)
            post = torch.randn(rows, 4, generator=rng)
            expected = (
                torch.einsum("rij,rih->rjh", comb, residual.float())
                + post[:, :, None] * x.float()[:, None]
            ).bfloat16()
            torch.testing.assert_close(
                post_910b(x, residual, post, comb), expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
