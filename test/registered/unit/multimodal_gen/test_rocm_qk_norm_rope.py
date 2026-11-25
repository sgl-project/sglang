# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="base-b", runner_config="1-gpu-small-amd")

DIM = 5120
HEAD_DIM = 128
TOKENS = 16
EPS = 1e-6


def _norm_rope_reference(x, weight, cos, sin):
    """Across-heads RMSNorm, rounded to the input dtype, then interleaved RoPE."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    n = (xf * rstd * weight.float()).to(x.dtype).float()
    pairs = n.view(x.shape[0], DIM // 2, 2)
    n0, n1 = pairs[..., 0], pairs[..., 1]
    freq = torch.arange(DIM // 2, device=x.device) % (HEAD_DIM // 2)
    cs, sn = cos.float()[:, freq], sin.float()[:, freq]
    out = torch.stack((n0 * cs - n1 * sn, n1 * cs + n0 * sn), dim=-1)
    return out.reshape(x.shape).to(x.dtype)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip, "needs a ROCm GPU"
)
class TestRocmQkNormRope(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.diffusion.norm.rocm_qk_norm_rope_triton import (
            rocm_qk_norm_rope,
        )

        self.kernel = rocm_qk_norm_rope
        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        # q and k are views of one packed QKV buffer, like the Wan QKV GEMM output.
        self.qkv = torch.randn(TOKENS, 3 * DIM, device=dev, dtype=dt)
        self.wq = torch.randn(DIM, device=dev, dtype=dt)
        self.wk = torch.randn(DIM, device=dev, dtype=dt)
        angle = torch.rand(TOKENS, HEAD_DIM // 2, device=dev) * 6.28
        self.cos, self.sin = angle.cos().to(dt), angle.sin().to(dt)

    def test_qk_norm_rope_matches_reference(self):
        q, k = self.qkv[:, :DIM], self.qkv[:, DIM : 2 * DIM]
        ref_q = _norm_rope_reference(q, self.wq, self.cos, self.sin)
        ref_k = _norm_rope_reference(k, self.wk, self.cos, self.sin)
        out_q, out_k = self.kernel(
            q, k, self.wq, self.wk, self.cos, self.sin, EPS, HEAD_DIM
        )
        torch.testing.assert_close(out_q, ref_q, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(out_k, ref_k, atol=3e-2, rtol=3e-2)

    def test_inplace_writes_the_result_back_into_the_packed_buffer(self):
        q, k = self.qkv[:, :DIM], self.qkv[:, DIM : 2 * DIM]
        ref_q = _norm_rope_reference(q, self.wq, self.cos, self.sin)
        ref_k = _norm_rope_reference(k, self.wk, self.cos, self.sin)
        value = self.qkv[:, 2 * DIM :].clone()
        self.kernel(q, k, self.wq, self.wk, self.cos, self.sin, EPS, HEAD_DIM, True)
        torch.testing.assert_close(q, ref_q, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(k, ref_k, atol=3e-2, rtol=3e-2)
        self.assertTrue(torch.equal(self.qkv[:, 2 * DIM :], value))

    def test_unsupported_head_layout_falls_back_to_none(self):
        q, k = self.qkv[:, :DIM], self.qkv[:, DIM : 2 * DIM]
        out = self.kernel(q, k, self.wq, self.wk, self.cos, self.sin, EPS, 96)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
