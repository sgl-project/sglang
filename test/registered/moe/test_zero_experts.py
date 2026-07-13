import unittest

import torch

from sglang.srt.layers.moe.ep_moe.kernels import zero_experts_compute_triton
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=16, stage="base-b", runner_config="1-gpu-small")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestZeroExpertsComputeTriton(CustomTestCase):
    def test_zero_expert_routes_keep_valid_ids(self):
        num_experts = 4
        hidden_states = torch.arange(
            2 * 512, dtype=torch.float32, device="cuda"
        ).reshape(2, 512)
        expert_indices = torch.tensor(
            [[0, 4, 1], [5, 2, 6]], dtype=torch.int64, device="cuda"
        )
        expert_scales = torch.tensor(
            [[0.1, 0.3, 0.2], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
            device="cuda",
        )
        original_indices = expert_indices.clone()
        original_scales = expert_scales.clone()

        output = zero_experts_compute_triton(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )
        torch.cuda.synchronize()

        zero_expert_mask = original_indices >= num_experts
        normal_expert_mask = ~zero_expert_mask
        self.assertTrue(torch.all(expert_indices >= 0).item())
        torch.testing.assert_close(
            expert_indices[normal_expert_mask],
            original_indices[normal_expert_mask],
        )
        torch.testing.assert_close(
            expert_indices[zero_expert_mask],
            torch.zeros_like(expert_indices[zero_expert_mask]),
        )
        torch.testing.assert_close(
            expert_scales[normal_expert_mask],
            original_scales[normal_expert_mask],
        )
        torch.testing.assert_close(
            expert_scales[zero_expert_mask],
            torch.zeros_like(expert_scales[zero_expert_mask]),
        )

        expected_scale = (
            original_scales * zero_expert_mask.to(original_scales.dtype)
        ).sum(dim=-1, keepdim=True)
        torch.testing.assert_close(output, hidden_states * expected_scale)


if __name__ == "__main__":
    unittest.main()
