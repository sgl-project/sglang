"""The SHUFFLE 5D KV writer's fused fp8 quantization: from bf16 K/V and
per-tensor scales it writes what an fp32 divide, a saturation to the fp8 range
and a cast would, for each way a scale can be given, and a cache of the
source's dtype is written as before."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=12, stage="jit-kernel-unit", runner_config="amd")

_HAS_GPU = torch.cuda.is_available()
NUM_TOKENS, NUM_HEADS, HEAD_SIZE, BLOCK_SIZE, NUM_BLOCKS = 37, 8, 128, 16, 8


def _caches(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = 16 // torch.empty((), dtype=dtype).element_size()
    shape_k = (NUM_BLOCKS, NUM_HEADS, HEAD_SIZE // x, BLOCK_SIZE, x)
    shape_v = (NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE // x, HEAD_SIZE, x)
    return (
        torch.zeros(shape_k, dtype=dtype, device="cuda"),
        torch.zeros(shape_v, dtype=dtype, device="cuda"),
    )


@unittest.skipUnless(_HAS_GPU, "Triton kernels require a GPU")
class TestShuffle5dFp8Write(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

        torch.manual_seed(0)
        self.fp8 = fp8_dtype
        self.fp8_max = torch.finfo(fp8_dtype).max
        shape = (NUM_TOKENS, NUM_HEADS, HEAD_SIZE)
        self.k = 4 * torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        self.v = 4 * torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        slots = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device="cuda")
        self.loc = slots[:NUM_TOKENS].to(torch.int64)

    def _written(self, k_scale, v_scale) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V read back from an fp8 cache the fused path wrote."""
        from sglang.kernels.ops.kvcache.shuffle_5d_cache import (
            launch_gather_shuffle_5d_to_linear,
            launch_reshape_and_cache_shuffle_5d,
        )

        k_cache, v_cache = _caches(torch.uint8)
        k_cache, v_cache = k_cache.view(self.fp8), v_cache.view(self.fp8)
        launch_reshape_and_cache_shuffle_5d(
            self.k, self.v, k_cache, v_cache, self.loc, k_scale=k_scale, v_scale=v_scale
        )
        return launch_gather_shuffle_5d_to_linear(k_cache, v_cache, self.loc)

    def _expected(self, scale: float, x: torch.Tensor) -> torch.Tensor:
        # A one-element divisor, not a Python float: dividing by a Python float
        # multiplies by its fp32 reciprocal, which differs from the kernel's
        # fp32 divide in the last bit for a scale such as 0.03.
        divisor = torch.tensor(scale, dtype=torch.float32, device=x.device)
        return (x.float() / divisor).clamp(-self.fp8_max, self.fp8_max).to(self.fp8)

    def _assert_matches(self, got: torch.Tensor, want: torch.Tensor) -> None:
        self.assertFalse(torch.isnan(got.float()).any())
        self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))

    def test_each_way_of_giving_the_scales_writes_the_fp32_divided_values(self):
        k_want, v_want = self._expected(0.05, self.k), self._expected(0.03, self.v)
        ways = (
            (0.05, 0.03),
            (torch.tensor([0.05], device="cuda"), torch.tensor([0.03], device="cuda")),
            (torch.tensor([0.05]), 0.03),
        )
        for k_scale, v_scale in ways:
            with self.subTest(k_scale=k_scale, v_scale=v_scale):
                k_got, v_got = self._written(k_scale, v_scale)
                self._assert_matches(k_got, k_want)
                self._assert_matches(v_got, v_want)

    def test_values_past_the_fp8_range_saturate(self):
        k_got, v_got = self._written(1e-4, 1e-4)

        self._assert_matches(k_got, self._expected(1e-4, self.k))
        self._assert_matches(v_got, self._expected(1e-4, self.v))
        self.assertEqual(k_got.float().abs().max().item(), self.fp8_max)

    def test_a_cache_of_the_source_dtype_is_written_as_given(self):
        from sglang.kernels.ops.kvcache.shuffle_5d_cache import (
            launch_gather_shuffle_5d_to_linear,
            launch_reshape_and_cache_shuffle_5d,
        )

        k_cache, v_cache = _caches(torch.bfloat16)
        launch_reshape_and_cache_shuffle_5d(self.k, self.v, k_cache, v_cache, self.loc)
        k_got, v_got = launch_gather_shuffle_5d_to_linear(k_cache, v_cache, self.loc)

        self.assertTrue(torch.equal(k_got, self.k) and torch.equal(v_got, self.v))


if __name__ == "__main__":
    unittest.main()
