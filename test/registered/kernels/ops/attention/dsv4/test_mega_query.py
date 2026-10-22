import unittest

import torch

from sglang.kernels.ops.attention.dsv4.mega_query import pack_query
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMegaQuery(CustomTestCase):
    def test_exact_interleaving_padding_and_strides(self):
        for heads, padded in ((8, 64), (16, 64), (64, 64), (64, 128), (128, 128)):
            for tokens in (0, 1, 3, 128):
                for strided in (False, True):
                    with self.subTest(
                        heads=heads, padded=padded, tokens=tokens, strided=strided
                    ):
                        source = torch.randn(
                            tokens * 2, heads, 1024, device="cuda", dtype=torch.bfloat16
                        )
                        q = (
                            source[::2, :, ::2]
                            if strided
                            else source[:tokens, :, :512].contiguous()
                        )
                        expected = q.new_zeros(tokens, 32, padded, 16)
                        expected[:, :, :heads] = q.reshape(
                            tokens, heads, 32, 16
                        ).transpose(1, 2)
                        actual = pack_query(q, padded)
                        torch.testing.assert_close(
                            actual,
                            expected.reshape(tokens, padded, 512),
                            rtol=0,
                            atol=0,
                        )
                        self.assertTrue(actual.is_contiguous())


if __name__ == "__main__":
    unittest.main()
