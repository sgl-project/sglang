"""Regression test: Marlin MoE must be bitwise deterministic for identical inputs.

The kernel reduces split-K partial sums through a global lock array that assumes a
single resident block per SM. Launching more than one block per SM mixed the
partials nondeterministically and corrupted a few output rows silently; see
sgl-project/sglang#44999.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    marlin_moe_permute_scales,
    moe_awq_to_marlin_zero_points,
)
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 32
HIDDEN = 1024
INTERMEDIATE = 256
GROUP_SIZE = 128
NUM_BITS = 4
TOPK = 4
BLOCK_M = 16


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is required"
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        return "Marlin requires compute capability >= 8.0"
    return None


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestMarlinMoEDeterminism(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
            gptq_marlin_moe_repack,
        )

        cls.device = "cuda"
        torch.manual_seed(0)
        pack = 32 // NUM_BITS

        def rand(shape, dtype):
            if dtype == torch.int32:
                return torch.randint(
                    0, 2**31 - 1, shape, dtype=torch.int32, device=cls.device
                )
            return (
                torch.rand(shape, device=cls.device, dtype=torch.float32) * 0.05 + 0.01
            ).to(dtype)

        # AWQ-format tensors (packed along the reduction dim), then marlin-packed.
        w13_qweight = rand((NUM_EXPERTS, HIDDEN // pack, 2 * INTERMEDIATE), torch.int32)
        w2_qweight = rand((NUM_EXPERTS, INTERMEDIATE // pack, HIDDEN), torch.int32)
        w13_scales = rand(
            (NUM_EXPERTS, HIDDEN // GROUP_SIZE, 2 * INTERMEDIATE), torch.bfloat16
        )
        w2_scales = rand(
            (NUM_EXPERTS, INTERMEDIATE // GROUP_SIZE, HIDDEN), torch.bfloat16
        )
        w13_qzeros = rand(
            (NUM_EXPERTS, HIDDEN // GROUP_SIZE, 2 * INTERMEDIATE // pack), torch.int32
        )
        w2_qzeros = rand(
            (NUM_EXPERTS, INTERMEDIATE // GROUP_SIZE, HIDDEN // pack), torch.int32
        )

        empty_perm = torch.empty((NUM_EXPERTS, 0), dtype=torch.int32, device=cls.device)
        cls.w1 = gptq_marlin_moe_repack(
            w13_qweight, empty_perm, HIDDEN, 2 * INTERMEDIATE, NUM_BITS
        )
        cls.w2 = gptq_marlin_moe_repack(
            w2_qweight, empty_perm, INTERMEDIATE, HIDDEN, NUM_BITS
        )
        cls.w1_scale = marlin_moe_permute_scales(
            w13_scales, HIDDEN, 2 * INTERMEDIATE, GROUP_SIZE
        ).contiguous()
        cls.w2_scale = marlin_moe_permute_scales(
            w2_scales, INTERMEDIATE, HIDDEN, GROUP_SIZE
        ).contiguous()
        cls.w1_zeros = moe_awq_to_marlin_zero_points(
            w13_qzeros, HIDDEN // GROUP_SIZE, 2 * INTERMEDIATE, NUM_BITS
        )
        cls.w2_zeros = moe_awq_to_marlin_zero_points(
            w2_qzeros, INTERMEDIATE // GROUP_SIZE, HIDDEN, NUM_BITS
        )

    def _inputs(self, num_tokens, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        hidden = (
            torch.randn((num_tokens, HIDDEN), device=self.device, generator=gen) * 0.1
        ).to(torch.bfloat16)
        gating = torch.randn((num_tokens, NUM_EXPERTS), device=self.device, generator=gen)
        topk_weights, topk_ids = torch.topk(gating, TOPK, dim=-1)
        return hidden, gating, topk_weights.float(), topk_ids.to(torch.int32)

    def _run(self, inputs):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        hidden, gating, topk_weights, topk_ids = inputs
        return fused_marlin_moe(
            hidden,
            self.w1,
            self.w2,
            self.w1_scale,
            self.w2_scale,
            gating,
            topk_weights,
            topk_ids,
            w1_zeros=self.w1_zeros,
            w2_zeros=self.w2_zeros,
            num_bits=NUM_BITS,
        )

    def test_repeated_calls_are_bitwise_identical(self):
        """The same input must produce the same bytes, whatever runs in between."""
        probe = self._inputs(64, 12345)
        reference = self._run(probe).clone()
        torch.cuda.synchronize()

        for iteration in range(500):
            # Interleave a call with another batch size: that is the pattern that
            # exposed the per-SM occupancy race in #44999.
            other = self._inputs(int(torch.randint(1, 512, (1,))), 5000 + iteration)
            self._run(other)
            out = self._run(probe)
            self.assertTrue(
                torch.equal(out, reference),
                f"iteration {iteration}: output differs from the first call",
            )

    def test_single_token_decode_is_bitwise_identical(self):
        """Decode-shaped calls (M=1) must also be reproducible."""
        probe = self._inputs(1, 777)
        reference = self._run(probe).clone()
        torch.cuda.synchronize()

        for iteration in range(100):
            other = self._inputs(int(torch.randint(1, 512, (1,))), 9000 + iteration)
            self._run(other)
            self.assertTrue(torch.equal(self._run(probe), reference))


if __name__ == "__main__":
    unittest.main()