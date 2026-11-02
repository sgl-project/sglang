"""EPLB moves of expert-major views whose per-domain slices live apart."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location_updater import (
    _split_noncontiguous_experts,
    update_expert_weights_single_layer,
)
from sglang.test.test_utils import CustomTestCase


class TestLocalizedExpertMoves(CustomTestCase):
    def test_unsplittable_experts_are_kept(self):
        # Contiguous experts, and MN-major scales whose dim-1 slices are strided
        for weight in (torch.zeros(4, 3, 5), torch.zeros(4, 5, 3).transpose(1, 2)):
            parts = _split_noncontiguous_experts(weight)
            self.assertEqual(len(parts), 1)
            self.assertIs(parts[0], weight)

    def test_permute_localized_view(self):
        num_domains, num_experts, rows, cols = 2, 4, 3, 5
        # Padded slice stride, like `deep_gemm.localize` rounding each domain up
        domain_major = torch.zeros(num_domains, num_experts + 1, rows, cols)[
            :, :num_experts
        ]
        domain_major.copy_(torch.randn(num_domains, num_experts, rows, cols))
        before = domain_major.clone()

        parts = _split_noncontiguous_experts(domain_major.transpose(0, 1))
        self.assertEqual(len(parts), num_domains)
        self.assertTrue(all(p[0].is_contiguous() for p in parts))

        new_map = [1, 0, 3, 2]
        update_expert_weights_single_layer(
            routed_experts_weights=parts,
            temp_buffers=[torch.empty_like(p) for p in parts],
            old_physical_to_logical_map=list(range(num_experts)),
            new_physical_to_logical_map=new_map,
            num_local_physical_experts=num_experts,
            num_gpu_per_node=1,
            rank=0,
            world_size=1,
        )
        self.assertTrue(torch.equal(domain_major, before[:, new_map]))


if __name__ == "__main__":
    unittest.main()
