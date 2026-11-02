"""EPLB moves of expert-major views whose per-domain slices live apart."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location_updater import (
    create_temp_buffers,
    update_expert_weights_single_layer,
)
from sglang.test.test_utils import CustomTestCase


def _localized(num_experts: int, rows: int, cols: int) -> torch.Tensor:
    # Padded slice stride, like `deep_gemm.localize` rounding each domain up
    domain_major = torch.zeros(2, num_experts + 1, rows, cols, dtype=torch.int8)[
        :, :num_experts
    ]
    return domain_major.copy_(torch.randint(-128, 128, domain_major.shape))


class TestLocalizedExpertMoves(CustomTestCase):
    def test_temp_buffers(self):
        localized = _localized(4, 3, 8).transpose(0, 1)
        scales = torch.zeros(4, 5, 3).transpose(1, 2)
        temp_localized, temp_scales = create_temp_buffers([localized, scales])
        self.assertTrue(temp_localized[0].is_contiguous())
        self.assertEqual(temp_scales.stride(), scales.stride())

    def test_permute_localized_view(self):
        new_map = [1, 0, 3, 2]
        # 8-byte rows take the int64 copy path, 5-byte rows the plain one
        for cols in (8, 5):
            domain_major = _localized(4, 3, cols)
            before = domain_major.clone()
            weights = [domain_major.transpose(0, 1)]
            update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=create_temp_buffers(weights),
                old_physical_to_logical_map=list(range(4)),
                new_physical_to_logical_map=new_map,
                num_local_physical_experts=4,
                num_gpu_per_node=1,
                rank=0,
                world_size=1,
            )
            self.assertTrue(torch.equal(domain_major, before[:, new_map]))


if __name__ == "__main__":
    unittest.main()
