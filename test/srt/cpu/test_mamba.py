import unittest

import sgl_kernel
import torch
import torch.nn.functional as F
from torch.nn.functional import softplus
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


def torch_gdn_gating(A_log, a, dt_bias):
    return -A_log.float().exp() * softplus(a.float() + dt_bias)


class TestMambaAttention(CustomTestCase):
    def test_fused_gdn_gating(self):
        dims = [6, 32]
        for dim in dims:
            A_log = torch.rand(dim)
            a = torch.rand(1024, dim, dtype=torch.bfloat16)
            dt_bias = torch.rand(dim, dtype=torch.bfloat16)

            g = torch_gdn_gating(A_log, a, dt_bias)
            g_sgl = torch.ops.sgl_kernel.fused_gdn_gating_cpu(A_log, a, dt_bias)
            atol = rtol = precision[g.dtype]
            self.assertTrue(torch.allclose(g, g_sgl, atol=atol, rtol=rtol))


if __name__ == "__main__":
    unittest.main()
