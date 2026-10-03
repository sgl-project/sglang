"""Deferred routing preserves logical IDs until all token shards are gathered."""

import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def route(hidden_states, gating_output, topk, renormalize):
    weights, ids = gating_output.softmax(-1).topk(topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights, ids.to(torch.int32)


def mask_padding(ids, num_token_non_padded):
    if num_token_non_padded is not None:
        ids[num_token_non_padded:] = -1


class TestTopKDeferredPostprocess(CustomTestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        self.capture = Mock()
        self.record = Mock()
        self.captured = []
        self.recorded = []
        self.capture.capture.side_effect = lambda **kwargs: self.captured.append(
            kwargs["topk_indices"].clone()
        )
        self.record.on_select_experts.side_effect = lambda **kwargs: (
            self.recorded.append(kwargs["topk_ids"].clone())
        )
        # Keep real postprocessing and placement selection, replacing only the
        # compiled wrapper and CUDA mask so these tests run on CPU CI.
        remap = topk._biased_grouped_topk_postprocess.__wrapped__
        for name, value in [
            ("_is_cuda", True),
            ("_is_hip", False),
            ("_use_aiter", False),
            ("_biased_grouped_topk_postprocess", remap),
            ("_mask_topk_ids_padded_region", mask_padding),
            ("has_per_rank_fused_shared_slots", lambda _: False),
            ("get_global_experts_capturer", lambda: self.capture),
            ("get_global_expert_distribution_recorder", lambda: self.record),
        ]:
            self.stack.enter_context(patch.object(topk, name, value))
        self.stack.enter_context(
            patch.object(
                topk.envs.SGLANG_SIMULATE_UNIFORM_EXPERTS, "get", return_value=False
            )
        )
        self.stack.enter_context(
            patch.object(
                topk.envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS, "get", return_value=False
            )
        )
        self.config = topk.TopKConfig(top_k=1, custom_routing_function=route)

    def test_gathered_routes_use_global_replica_indices_and_record_physical_ids(self):
        hidden = torch.ones(5, 2)
        logits = torch.tensor([[4.0, 1.0, 0.0]]).expand(5, -1)
        placement = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="dynamic",
            partial_logical_to_rank_dispatch_physical_map=None,
            partial_logical_to_all_physical_map=torch.tensor([[0, 3], [1, 4], [2, -1]]),
            partial_logical_to_all_physical_map_num_valid=torch.tensor([2, 2, 1]),
            num_physical_experts=5,
            rank_invariant=True,
        )
        shards = [
            topk.select_experts(
                hidden[rows],
                logits[rows],
                self.config,
                layer_id=2,
                expert_location_dispatch_info=placement,
                defer_postprocessing=True,
            )
            for rows in (slice(0, 3), slice(3, 5))
        ]
        self.capture.capture.assert_not_called()
        self.record.on_select_experts.assert_not_called()
        gathered = topk.StandardTopKOutput(
            torch.cat([shard.topk_weights for shard in shards]),
            torch.cat([shard.topk_ids for shard in shards]),
            None,
        )
        output = topk.postprocess_topk_output(
            gathered, self.config, 2, expert_location_dispatch_info=placement
        )
        expected_ids = torch.tensor([[0], [3], [0], [3], [0]], dtype=torch.int32)
        torch.testing.assert_close(output.topk_ids, expected_ids)
        torch.testing.assert_close(self.captured[0], torch.zeros_like(expected_ids))
        torch.testing.assert_close(self.recorded[0], expected_ids)
        self.assertEqual(self.capture.capture.call_args.kwargs["layer_id"], 2)

        ordinary = topk.select_experts(
            hidden,
            logits,
            self.config,
            layer_id=2,
            expert_location_dispatch_info=placement,
        )
        torch.testing.assert_close(output.topk_ids, ordinary.topk_ids)
        torch.testing.assert_close(output.topk_weights, ordinary.topk_weights)
        self.assertEqual(len(self.captured), 2)
        self.assertEqual(len(self.recorded), 2)

    def test_padding_is_applied_only_after_deferred_selection(self):
        hidden = torch.ones(3, 2)
        logits = torch.tensor([[0.0, 4.0, 1.0]]).expand(3, -1)
        output = topk.select_experts(
            hidden,
            logits,
            self.config,
            layer_id=2,
            num_token_non_padded=torch.tensor(1),
            defer_postprocessing=True,
        )
        torch.testing.assert_close(output.topk_ids, torch.ones(3, 1, dtype=torch.int32))
        output = topk.postprocess_topk_output(
            output, self.config, 2, num_token_non_padded=torch.tensor(1)
        )
        expected_ids = torch.tensor([[1], [-1], [-1]], dtype=torch.int32)
        torch.testing.assert_close(output.topk_ids, expected_ids)
        torch.testing.assert_close(self.recorded[0], expected_ids)

    def test_deferred_fused_router_does_not_emit_masked_or_packed_ids(self):
        config = topk.TopKConfig(
            top_k=1, scoring_func="sqrtsoftplus", fused_gate_packed_ids=True
        )
        logits = torch.ones(3, 4)
        kernel = Mock(
            return_value=(torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.int32))
        )
        with (
            patch.object(topk, "_is_cpu", False),
            patch.object(topk, "_is_xpu", False),
            patch.object(topk, "biased_topk_jit_kernel_impl", kernel),
        ):
            output = topk.select_experts(
                torch.ones(3, 2),
                logits,
                config,
                num_token_non_padded=torch.tensor(1),
                defer_postprocessing=True,
            )
        self.assertIs(type(output), topk.StandardTopKOutput)
        self.assertIsNone(kernel.call_args.kwargs["num_token_non_padded"])
        self.assertNotIn("packed_out", kernel.call_args.kwargs)
        self.capture.capture.assert_not_called()
        self.record.on_select_experts.assert_not_called()

    def test_postprocess_preserves_existing_packed_payload(self):
        packed = torch.tensor([[123], [456]], dtype=torch.int32)
        output = topk.StandardTopKOutputPacked(
            torch.ones(2, 1),
            torch.tensor([[1], [0]], dtype=torch.int32),
            torch.ones(2, 3),
            packed,
        )
        result = topk.postprocess_topk_output(output, self.config, 2)
        self.assertIs(type(result), topk.StandardTopKOutputPacked)
        self.assertIs(result.packed_topk_ids, packed)
        self.capture.capture.assert_called_once()
        self.record.on_select_experts.assert_called_once()

    def test_rank_padding_holes_are_excluded_after_physical_remapping(self):
        placement = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=torch.tensor([2, 0, 1]),
            partial_logical_to_all_physical_map=torch.tensor([[2], [0], [1]]),
            partial_logical_to_all_physical_map_num_valid=torch.ones(
                3, dtype=torch.long
            ),
            num_physical_experts=3,
        )
        output = topk.StandardTopKOutput(
            torch.ones(4, 1),
            torch.tensor([[0], [0], [1], [0]], dtype=torch.int32),
            None,
        )
        result = topk.postprocess_topk_output(
            output,
            self.config,
            2,
            expert_location_dispatch_info=placement,
            valid_token_mask=torch.tensor([[1], [0], [1], [0]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            result.topk_ids, torch.tensor([[2], [0], [0], [0]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            result.topk_weights, torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        )
        torch.testing.assert_close(
            self.recorded[0], torch.tensor([[2], [-1], [0], [-1]], dtype=torch.int32)
        )
        torch.testing.assert_close(self.captured[0], output.topk_ids)


if __name__ == "__main__":
    unittest.main()
