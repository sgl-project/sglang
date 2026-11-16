"""NPU routing coverage requiring the registered MoE operator API."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopK, fused_topk_torch_native
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.runtime_context import (
    get_context,
    get_model,
    get_parallel,
    get_resources,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=10, suite="base-b-test-1-npu-a3")


def _placement(num_layers=1):
    logical_to_physical = torch.tensor([[2, 0, 3, 1]], dtype=torch.int64).repeat(
        num_layers, 1
    )
    physical_to_logical = logical_to_physical.argsort(dim=-1)
    all_physical = torch.full((num_layers, 4, 4), -1, dtype=torch.int64)
    all_physical[:, :, 0] = logical_to_physical
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical,
        physical_to_logical_map_cpu=physical_to_logical.clone(),
        logical_to_all_physical_map=all_physical,
        logical_to_all_physical_map_cpu=all_physical.clone(),
        logical_to_all_physical_map_num_valid=torch.ones(
            (num_layers, 4), dtype=torch.int64
        ),
        ep_size=1,
        logical_to_rank_dispatch_physical_map=logical_to_physical,
    )


class TestNPURoutingInputs(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = get_context().override_server_args(
            ep_dispatch_algorithm="static", enable_eplb=True
        )
        config.install()
        self.addCleanup(config.restore)
        initialize_moe_config()
        topology = get_parallel().override(
            attn_dp_size=1, attn_cp_size=1, attn_tp_size=1, attn_tp_rank=0
        )
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        for name in ("_is_cuda", "_is_hip", "_use_aiter"):
            scope = patch.object(topk_module, name, False)
            scope.start()
            self.addCleanup(scope.stop)
        self.metadata = _placement()
        get_resources().expert_location_metadata = self.metadata
        self.addCleanup(setattr, get_resources(), "expert_location_metadata", None)
        self.hidden = torch.ones((4, 2))
        self.logits = torch.tensor([[9.0, 8.0, 1.0, 0.0]]).repeat(4, 1)

    def _router(self, **kwargs):
        shared_slots = kwargs.pop("num_fused_shared_experts", 0)
        layer_id = kwargs.pop("layer_id", 0)
        with get_model().override(
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=shared_slots)
        ):
            return TopK(
                top_k=2 - shared_slots,
                layer_id=layer_id,
                custom_routing_function=fused_topk_torch_native,
                **kwargs,
            )

    def test_npu_routes_keep_valid_ids_and_zero_only_padded_contributions(self):
        """NPU padding must stay inert without inventing invalid or duplicate routes."""
        router = self._router(num_fused_shared_experts=1)
        router.topk_config.scoring_func = "sqrtsoftplus"
        logical = torch.tensor([[0, 4], [1, 4], [2, 4], [3, 4]], dtype=torch.int32)
        weights = torch.tensor([[0.75, 0.25]]).repeat(4, 1)

        class Capture:
            def capture(self, *, layer_id, topk_indices):
                self.ids = topk_indices.clone()

        capture = Capture()
        get_resources().experts_capturer = capture
        self.addCleanup(setattr, get_resources(), "experts_capturer", None)
        _, info = router.routing_inputs()
        with patch(
            "sglang.srt.hardware_backend.npu.moe.topk.moe_gating_top_k",
            return_value=(weights, logical, None),
        ):
            output = fused_topk_npu(
                hidden_states=self.hidden,
                router_logits=self.logits,
                topk_config=router.topk_config,
                num_token_non_padded=torch.tensor(2),
                expert_location_dispatch_info=info,
                layer_id=0,
            )
        self.assertEqual(output.topk_ids.tolist(), [[2, 4], [0, 4], [3, 4], [1, 4]])
        torch.testing.assert_close(
            output.topk_weights[:2], torch.tensor([[0.75, 0.25]]).repeat(2, 1)
        )
        torch.testing.assert_close(output.topk_weights[2:], torch.zeros((2, 2)))
        self.assertEqual(capture.ids.tolist(), [[0], [1], [2], [3]])


if __name__ == "__main__":
    unittest.main()
