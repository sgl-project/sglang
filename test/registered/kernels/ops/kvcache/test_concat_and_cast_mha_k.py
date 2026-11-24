"""concat_and_cast_mha_k_triton must accept head counts and nope/rope widths that
are not powers of two (GLM-5.x qk_nope_head_dim=192, ROCM-29011)."""

import unittest

import torch

from sglang.kernels.ops.kvcache.cache_ops import concat_and_cast_mha_k_triton
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestConcatAndCastMhaK(CustomTestCase):
    def _check(self, *, num_tokens, heads, nope_dim, rope_dim, out_dtype):
        torch.manual_seed(0)
        k_nope = torch.randn(num_tokens, heads, nope_dim, device="cuda")
        k_nope = k_nope.to(torch.bfloat16)
        k_rope = torch.randn(num_tokens, 1, rope_dim, device="cuda")
        k_rope = k_rope.to(torch.bfloat16)
        # Poison the output so a lane the kernel skips shows up as a mismatch.
        k = torch.full((num_tokens, heads, nope_dim + rope_dim), 7.0, device="cuda").to(
            out_dtype
        )

        concat_and_cast_mha_k_triton(k, k_nope, k_rope)

        expected = torch.cat(
            [k_nope, k_rope.expand(num_tokens, heads, rope_dim)], dim=-1
        ).to(out_dtype)
        torch.testing.assert_close(k.float(), expected.float(), atol=0, rtol=0)

    def test_power_of_two_extents(self):
        """DeepSeek-style 128 nope + 64 rope on the fast kernel stays exact."""
        self._check(
            num_tokens=5, heads=16, nope_dim=128, rope_dim=64, out_dtype=torch.bfloat16
        )

    def test_non_power_of_two_heads(self):
        """Kimi-K3 at TP8 gives 12 heads."""
        self._check(
            num_tokens=5, heads=12, nope_dim=128, rope_dim=64, out_dtype=torch.bfloat16
        )

    def test_non_power_of_two_nope_dim(self):
        """GLM-5.x has qk_nope_head_dim=192, which used to fail Triton's arange."""
        for heads in (16, 12):
            with self.subTest(heads=heads):
                self._check(
                    num_tokens=5,
                    heads=heads,
                    nope_dim=192,
                    rope_dim=64,
                    out_dtype=torch.bfloat16,
                )

    def test_non_power_of_two_nope_dim_fp8_out(self):
        """The aiter FP8 path casts while concatenating; padding must not change that."""
        fp8 = (
            torch.float8_e4m3fnuz
            if torch.version.hip
            and "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
            else torch.float8_e4m3fn
        )
        self._check(num_tokens=3, heads=16, nope_dim=192, rope_dim=64, out_dtype=fp8)


if __name__ == "__main__":
    unittest.main()
