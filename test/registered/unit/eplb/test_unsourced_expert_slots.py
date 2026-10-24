"""A p2p recv from a retired rank must not publish its unfilled buffer."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.eplb import expert_location_updater as elu
from sglang.test.test_utils import CustomTestCase


class TestUnsourcedSlotsAreNotPublished(CustomTestCase):
    def test_unsourced_slots_are_left_for_the_reload(self):
        # Logical 9 lives only on retired rank 3, and rank 0 wants it in slots 0 and 2;
        # slot 2 free-rides off slot 0's buffer, so both must keep their old contents.
        active = torch.tensor([True, True, True, False])
        weights = [torch.arange(3, dtype=torch.float32).reshape(3, 1) + 1234.5]
        before, missing = weights[0].clone(), []
        state = SimpleNamespace(active_ranks_cpu=active)
        with (
            patch.object(ElasticEPStateManager, "instance", lambda: state),
            patch.object(elu, "P2POp", SimpleNamespace),
        ):
            elu.update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=elu.create_temp_buffers(weights),
                old_physical_to_logical_map=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9],
                new_physical_to_logical_map=[9, 1, 9, 3, 4, 5, 6, 7, 8, 0, 2, 8],
                num_local_physical_experts=3,
                num_gpu_per_node=4,
                rank=0,
                world_size=4,
                missing_logical_experts_info=missing,
            )
        torch.testing.assert_close(weights[0], before)
        self.assertEqual(missing, [9])


if __name__ == "__main__":
    unittest.main()
