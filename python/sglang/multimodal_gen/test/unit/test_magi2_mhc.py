# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_mhc import sinkhorn_knopp

ITERS = 20
EPS = 1e-12


class TestSinkhornKnopp(unittest.TestCase):
    def test_output_is_doubly_stochastic(self):
        torch.manual_seed(0)
        for tokens, n in ((7, 2), (5, 4), (3, 8)):
            with self.subTest(tokens=tokens, n=n):
                out = sinkhorn_knopp(
                    torch.randn(tokens, n, n) * 2.0, num_iters=ITERS, eps=EPS
                )
                ones = torch.ones(tokens, n)
                self.assertTrue(torch.allclose(out.sum(dim=-1), ones, atol=1e-4))
                self.assertTrue(torch.allclose(out.sum(dim=-2), ones, atol=1e-4))
                self.assertTrue((out >= 0).all())


if __name__ == "__main__":
    unittest.main()
