"""A FusedMoE that opts out of expert location still follows the slot layout."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_ROUTED = 128
EP_SIZE = 4
NUM_LOCAL_ROUTED = NUM_ROUTED // EP_SIZE


class _OptedOutMoE:
    """The loader-facing surface of a draft FusedMoE with per-rank shared slots."""

    uses_expert_location = False
    layer_id = 0
    quant_method = None
    num_fused_shared_experts = 1
    moe_ep_size = EP_SIZE
    _expert_storage_rank = 1
    _has_fused_shared = True
    _num_global_routed = NUM_ROUTED
    _num_local_routed = NUM_LOCAL_ROUTED
    num_local_experts = NUM_LOCAL_ROUTED + 1

    def __init__(self):
        self.loaded = []

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return FusedMoE._map_global_expert_id_to_local_expert_id(self, expert_id)

    def _weight_loader_physical(self, **kwargs):
        return FusedMoE._weight_loader_physical(self, **kwargs)

    def _weight_loader_impl(
        self, *, param, loaded_weight, weight_name, shard_id, expert_id
    ):
        self.loaded.append(expert_id)


class TestFusedMoEExpertLocationOptOut(CustomTestCase):
    def _load(self, *, expert_id, require_global):
        moe = _OptedOutMoE()
        param = torch.nn.Parameter(torch.zeros(1))
        if require_global:
            param._sglang_require_global_experts = True
        with (
            patch.object(
                fused_moe_layer,
                "get_global_expert_location_metadata",
                side_effect=AssertionError("opted-out layer read the target map"),
            ),
            patch.object(
                fused_moe_layer, "uses_per_rank_fused_shared_slots", return_value=True
            ),
        ):
            FusedMoE.weight_loader(
                moe,
                param=param,
                loaded_weight=torch.zeros(1),
                weight_name="w13_input_scale",
                shard_id="w1",
                expert_id=expert_id,
            )
        return moe.loaded

    def test_global_shared_param_fills_every_ranks_shared_slot(self):
        """A global-sized param's fused shared expert lands on every rank's shared
        slot, not on routed slot ``num_routed`` as the checkpoint numbers it."""
        shared_id = NUM_ROUTED
        expected = [
            rank * (NUM_LOCAL_ROUTED + 1) + NUM_LOCAL_ROUTED for rank in range(EP_SIZE)
        ]
        self.assertEqual(self._load(expert_id=shared_id, require_global=True), expected)

    def test_routed_param_keeps_the_identity_placement(self):
        own = _OptedOutMoE._expert_storage_rank * NUM_LOCAL_ROUTED + 3
        self.assertEqual(self._load(expert_id=own, require_global=False), [3])
        self.assertEqual(
            self._load(expert_id=own + NUM_LOCAL_ROUTED, require_global=False), []
        )


if __name__ == "__main__":
    unittest.main()
