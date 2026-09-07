"""Tests for replicated shared experts around deferred reductions."""

import ast
import importlib
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.communicator import (
    CommunicateSummableTensorPairFn,
    LayerCommunicator,
)
from sglang.srt.layers.moe.utils import has_replicated_shared_expert
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestHasReplicatedSharedExpert(CustomTestCase):
    def test_both_attribute_spellings(self):
        for attr in ("shared_experts", "shared_expert"):
            mlp = SimpleNamespace(**{attr: object()}, _shared_expert_tp1=True)
            self.assertTrue(has_replicated_shared_expert(mlp), attr)

    def test_tp_sharded_shared_expert_is_not_replicated(self):
        mlp = SimpleNamespace(shared_experts=object(), _shared_expert_tp1=False)
        self.assertFalse(has_replicated_shared_expert(mlp))

    def test_no_shared_expert(self):
        self.assertFalse(
            has_replicated_shared_expert(
                SimpleNamespace(shared_experts=None, _shared_expert_tp1=True)
            )
        )
        self.assertFalse(has_replicated_shared_expert(SimpleNamespace()))


class TestDeferredAllReduceDoubleCounts(CustomTestCase):
    @staticmethod
    def _layer(tp_size, shared_inside_allreduce):
        rows = 4
        shared = torch.full((rows, 1), 7.0)
        routed = [torch.full((rows, 1), float(r)) for r in range(tp_size)]
        if shared_inside_allreduce:
            per_rank = [r + shared for r in routed]
            return torch.stack(per_rank).sum(0)
        return torch.stack(routed).sum(0) + shared

    def test_shared_output_inside_the_reduction_is_scaled_by_tp_size(self):
        tp_size = 8
        correct = self._layer(tp_size, shared_inside_allreduce=False)
        wrong = self._layer(tp_size, shared_inside_allreduce=True)
        self.assertTrue(torch.allclose(wrong - correct, torch.full((4, 1), 49.0)))
        self.assertFalse(torch.allclose(wrong, correct))


class TestDpReduceScatterSharedExpert(CustomTestCase):
    def _communicator(self):
        communicator = LayerCommunicator.__new__(LayerCommunicator)
        communicator.allow_reduce_scatter = True
        communicator._communicate_summable_tensor_pair_fn = (
            CommunicateSummableTensorPairFn._scatter_hidden_states
        )
        return communicator

    def test_variable_and_equal_chunk_paths_require_reduce_scatter(self):
        communicator = self._communicator()

        variable = SimpleNamespace(
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: False)
        )
        with patch(
            "sglang.srt.layers.communicator.should_use_dp_reduce_scatterv",
            return_value=True,
        ):
            self.assertTrue(communicator.should_use_dp_reduce_scatter(variable))

        equal = SimpleNamespace(
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: True)
        )
        with patch(
            "sglang.srt.layers.communicator.should_use_dp_reduce_scatterv",
            return_value=False,
        ):
            self.assertTrue(communicator.should_use_dp_reduce_scatter(equal))

        communicator._communicate_summable_tensor_pair_fn = (
            CommunicateSummableTensorPairFn._trivial
        )
        with patch(
            "sglang.srt.layers.communicator.should_use_dp_reduce_scatterv",
            return_value=True,
        ):
            self.assertFalse(communicator.should_use_dp_reduce_scatter(variable))

    def _check_combine(self, sizes):
        world_size = len(sizes)
        rows = sum(sizes)
        shared = torch.arange(1, rows + 1, dtype=torch.float32).view(rows, 1)
        routed = [torch.full((rows, 1), float(rank + 1)) for rank in range(world_size)]
        routed_sum = torch.stack(routed).sum(0)
        wrong = torch.stack([partial + shared for partial in routed]).sum(0)
        expected = routed_sum + shared

        for rank in range(world_size):
            with (
                patch(
                    "sglang.srt.layers.communicator.get_dp_global_num_tokens",
                    return_value=sizes,
                ),
                patch(
                    "sglang.srt.layers.communicator.get_parallel",
                    return_value=SimpleNamespace(attn_dp_rank=rank),
                ),
            ):
                local_shared = LayerCommunicator.get_dp_local_hidden_states(shared)
                local_routed = LayerCommunicator.get_dp_local_hidden_states(routed_sum)
                local_wrong = LayerCommunicator.get_dp_local_hidden_states(wrong)
                local_expected = LayerCommunicator.get_dp_local_hidden_states(expected)

            fixed = local_routed + local_shared
            self.assertTrue(torch.equal(fixed, local_expected))
            self.assertTrue(
                torch.equal(
                    local_wrong - fixed,
                    local_shared * (world_size - 1),
                )
            )

    def test_equal_chunk_shared_is_added_once(self):
        self._check_combine([3, 3, 3, 3])

    def test_variable_chunk_shared_is_added_once(self):
        self._check_combine([0, 1, 4, 2])

    def test_laguna_moe_can_skip_hoisted_shared_expert(self):
        laguna = importlib.import_module("sglang.srt.models.laguna")

        class Gate(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states.new_zeros((hidden_states.shape[0], 2))

        class TopK(torch.nn.Module):
            def forward(self, hidden_states, router_logits):
                return None

        class Experts(torch.nn.Module):
            def forward(self, hidden_states, topk_output):
                return hidden_states * 2

        class Shared(torch.nn.Module):
            def forward(self, hidden_states, forward_batch=None):
                return hidden_states * 10

        moe = laguna.LagunaMoE.__new__(laguna.LagunaMoE)
        torch.nn.Module.__init__(moe)
        moe.tp_size = 1
        moe.routed_scaling_factor = 1.0
        moe.router_logit_softcapping = 0.0
        moe._shared_expert_tp1 = True
        moe.gate = Gate()
        moe.topk = TopK()
        moe.experts = Experts()
        moe.shared_expert = Shared()

        hidden_states = torch.ones(3, 2)
        with_shared = moe(hidden_states)
        routed_only = moe(hidden_states, skip_shared_experts=True)
        self.assertTrue(torch.equal(with_shared, hidden_states * 12))
        self.assertTrue(torch.equal(routed_only, hidden_states * 2))


class TestFusionGatedOnReplicatedSharedExpert(CustomTestCase):
    MODULES = ("sglang.srt.models.deepseek_v2", "sglang.srt.models.laguna")

    def test_publication_is_gated(self):
        for name in self.MODULES:
            module = importlib.import_module(name)
            source = inspect.getsource(module)
            self.assertIn(
                "_shared_expert_tp1",
                source,
                f"{name} no longer owns a TP1 shared expert; drop it from MODULES",
            )
            for node in ast.walk(ast.parse(source)):
                if not (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "fuse_mlp_allreduce"
                ):
                    continue
                self.assertIn(
                    "has_replicated_shared_expert",
                    ast.dump(node.value),
                    f"{name}: fuse_mlp_allreduce lacks the shared-expert guard",
                )


class TestMandatorySharedExpertHoist(CustomTestCase):
    def _assignment(self, source, name):
        tree = ast.parse(textwrap.dedent(source))
        values = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        self.assertEqual(len(values), 1, name)
        return ast.dump(values[0])

    def test_eager_dp_collectives_require_the_hoist(self):
        module = importlib.import_module("sglang.srt.models.deepseek_v4")
        source = inspect.getsource(module.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync)

        mlp_reduce_scatter = self._assignment(source, "mlp_reduce_scatter")
        dp_hoist = self._assignment(source, "_dp_must_hoist_shared")
        do_shared_local = self._assignment(source, "_do_shared_local")

        self.assertIn("_use_dynamic_reduce_scatterv", mlp_reduce_scatter)
        self.assertIn("_use_dynamic_reduce_scatterv", dp_hoist)
        self.assertIn("_cp_must_hoist_shared", do_shared_local)
        self.assertIn("_dp_must_hoist_shared", do_shared_local)

        tree = ast.parse(textwrap.dedent(source))
        reduce_scatterv_conditions = [
            ast.dump(node.test)
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "_use_dynamic_reduce_scatterv" in ast.dump(node.test)
            and "_use_reduce_scatterv" in ast.dump(node.test)
        ]
        self.assertTrue(reduce_scatterv_conditions)

    def test_dp_tbo_hoist_is_not_perf_gated(self):
        module = importlib.import_module("sglang.srt.models.deepseek_v4")
        source = inspect.getsource(module.DeepseekV4DecoderLayer.op_gather_a)
        do_shared_local = self._assignment(source, "do_shared_local")

        self.assertIn("has_replicated_shared_expert", do_shared_local)
        self.assertNotIn("_SHARED_EXPERT_LOCAL", do_shared_local)

    def test_layer_communicator_dp_models_hoist_shared_expert(self):
        cases = (
            ("sglang.srt.models.deepseek_v2", "DeepseekV2DecoderLayer"),
            ("sglang.srt.models.laguna", "LagunaDecoderLayer"),
        )
        for module_name, class_name in cases:
            module = importlib.import_module(module_name)
            source = inspect.getsource(getattr(module, class_name).forward)
            for required in (
                "should_use_dp_reduce_scatter",
                "get_dp_local_hidden_states",
                "skip_shared_experts=True",
            ):
                self.assertIn(required, source, f"{class_name}: missing {required}")


if __name__ == "__main__":
    unittest.main()
