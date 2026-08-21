"""Tests for replicated shared experts around deferred reductions."""

import ast
import importlib
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import torch

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


if __name__ == "__main__":
    unittest.main()
