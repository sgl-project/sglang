"""moe_fused_mul_sum's output_add epilogue vs. a separate torch add.

The epilogue exists to drop one [num_tokens, hidden_size] elementwise-add kernel
per MoE layer (the shared-expert add). Folding it in is only a safe drop-in if
it changes no bits, so that is what this pins.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
except ImportError:  # triton is not installed on CPU-only runners
    moe_fused_mul_sum = None


@unittest.skipUnless(
    torch.cuda.is_available() and moe_fused_mul_sum is not None,
    "CUDA + triton required",
)
class TestMoeFusedMulSumOutputAdd(CustomTestCase):
    def _run(
        self,
        *,
        num_tokens,
        top_k,
        size,
        dtype,
        routed_scaling_factor=None,
        expert_map=None,
        is_ep=False,
        seed=0,
    ):
        torch.manual_seed(seed)
        device = "cuda"
        num_experts = max(top_k, 8)
        inputs = torch.randn(num_tokens, top_k, size, dtype=dtype, device=device)
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
        )
        if is_ep:
            # -1 marks an expert that does not live on this rank.
            topk_ids[:, 0] = -1
        output_add = torch.randn(num_tokens, size, dtype=dtype, device=device)

        kwargs = dict(
            inputs=inputs,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            routed_scaling_factor=routed_scaling_factor,
            is_ep=is_ep,
        )
        unfused = moe_fused_mul_sum(**kwargs) + output_add
        fused = moe_fused_mul_sum(**kwargs, output_add=output_add)
        return unfused, fused

    def test_bit_identical_across_shapes(self):
        for num_tokens in (1, 3, 8, 129, 1024):
            for top_k in (1, 2, 8):
                for size in (128, 4096):
                    with self.subTest(num_tokens=num_tokens, top_k=top_k, size=size):
                        unfused, fused = self._run(
                            num_tokens=num_tokens,
                            top_k=top_k,
                            size=size,
                            dtype=torch.bfloat16,
                        )
                        self.assertTrue(torch.equal(fused, unfused))

    def test_bit_identical_across_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                unfused, fused = self._run(
                    num_tokens=64, top_k=4, size=512, dtype=dtype
                )
                self.assertTrue(torch.equal(fused, unfused))

    def test_bit_identical_with_scaling_and_masking(self):
        expert_map = torch.full((8,), -1, dtype=torch.int32, device="cuda")
        expert_map[:4] = torch.arange(4, dtype=torch.int32, device="cuda")
        cases = {
            "routed_scaling": dict(routed_scaling_factor=2.5),
            "expert_map": dict(expert_map=expert_map),
            "is_ep": dict(is_ep=True),
            "is_ep_and_scaling": dict(is_ep=True, routed_scaling_factor=0.5),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                unfused, fused = self._run(
                    num_tokens=96, top_k=4, size=1024, dtype=torch.bfloat16, **kwargs
                )
                self.assertTrue(torch.equal(fused, unfused))

    def test_writes_into_a_preallocated_output(self):
        torch.manual_seed(7)
        inputs = torch.randn(32, 4, 256, dtype=torch.bfloat16, device="cuda")
        topk_weights = torch.rand(32, 4, dtype=torch.bfloat16, device="cuda")
        output_add = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda")

        outputs = torch.empty(32, 256, dtype=torch.bfloat16, device="cuda")
        moe_fused_mul_sum(
            inputs=inputs,
            topk_weights=topk_weights,
            outputs=outputs,
            output_add=output_add,
        )
        expected = (
            moe_fused_mul_sum(inputs=inputs, topk_weights=topk_weights) + output_add
        )
        self.assertTrue(torch.equal(outputs, expected))


if __name__ == "__main__":
    unittest.main()
