"""Real metadata CPU mirrors and captured tensor views survive a fixed-size swap."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.eplb.expert_swap import plan_expert_swap
from sglang.srt.eplb.expert_swap_transaction import _metadata_layout
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertSwapMetadata(CustomTestCase):
    def test_cpu_mirrors_cached_maps_and_old_views_observe_new_ids_in_place(self):
        source = '{"num_experts":8,"keep":[[7,1,5,3]]}'
        with (
            get_context().override_server_args(
                device="cpu",
                tp_size=1,
                ep_size=1,
                moe_expert_profile=source,
                moe_a2a_backend="none",
                ep_dispatch_algorithm="dynamic",
            ),
            patch.object(
                ModelConfigForExpertLocation,
                "from_model_config",
                return_value=ModelConfigForExpertLocation(1, 8),
            ),
        ):
            old = ExpertLocationMetadata.init_by_mapping(None, [[7, 1, 5, 3]], 0)
            info = ExpertLocationDispatchInfo(
                "dynamic",
                None,
                old.logical_to_all_physical_map[0],
                old.logical_to_all_physical_map_num_valid[0],
                4,
                rank_invariant=True,
            )
            cached = old.mega_gate_maps("dynamic")
            layout = _metadata_layout(old)
            plan = plan_expert_swap([[7, 1, 5, 3]], [[5, 6, 7, 0]], 8)
            target = ExpertLocationMetadata.init_by_mapping(
                None, plan.physical_to_logical, 0
            )
            old.update(target, [0])
            self.assertEqual(_metadata_layout(old), layout)
            self.assertEqual(old.physical_to_logical_map_cpu.tolist(), [[7, 6, 5, 0]])
            self.assertEqual(old.logical_to_all_physical(0, 1), [])
            self.assertEqual(old.logical_to_all_physical(0, 6), [1])
            self.assertEqual(
                info.partial_logical_to_all_physical_map_num_valid[1].item(), 0
            )
            self.assertEqual(
                info.partial_logical_to_all_physical_map_num_valid[6].item(), 1
            )
            self.assertEqual(cached[1][0, 1].item(), 0)
            self.assertEqual(cached[1][0, 6].item(), 1)
            result = topk_ids_logical_to_physical(torch.tensor([[6, 7, 5]]), info)
            self.assertEqual(result.tolist(), [[1, 0, 2]])


if __name__ == "__main__":
    unittest.main()
