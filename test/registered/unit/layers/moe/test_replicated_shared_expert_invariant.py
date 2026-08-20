"""A replicated shared expert must never sit inside a tensor a collective SUMs.

A TP-sharded shared expert produces a genuine per-rank partial, so a
post-experts reduction over it is correct. A replicated (``tp_size=1``) one
holds the identical full value on every rank, so any group SUM scales it by the
group size -- silently, once per layer.

MoE modules encode this by adding the shared output *after* their own
post-experts all-reduce. That guard only covers the reduction they perform
themselves. ``should_skip_post_experts_all_reduce`` lets a caller defer,
replace, or absorb that reduction, and every such caller inherits the
invariant:

  - ``fuse_mlp_allreduce`` -- the next layer's residual+layernorm performs the
    all-reduce instead (covered here).
  - the prefill-CP combine -- ``dsa_cp_reduce_scatter_hidden_states`` sums
    across the CP group (covered in ``test/registered/cp/test_cp_strategy_unit.py``).
"""

import ast
import inspect
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.utils import has_replicated_shared_expert
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestHasReplicatedSharedExpert(CustomTestCase):
    def test_both_attribute_spellings(self):
        # DeepseekV2MoE and everything reusing it say "shared_experts";
        # LagunaMoE says "shared_expert".
        for attr in ("shared_experts", "shared_expert"):
            mlp = SimpleNamespace(**{attr: object()}, _shared_expert_tp1=True)
            self.assertTrue(has_replicated_shared_expert(mlp), attr)

    def test_tp_sharded_shared_expert_is_not_replicated(self):
        # Per-rank partials: the downstream SUM is exactly what reassembles them.
        mlp = SimpleNamespace(shared_experts=object(), _shared_expert_tp1=False)
        self.assertFalse(has_replicated_shared_expert(mlp))

    def test_no_shared_expert(self):
        # Fused into the routed kernel, or a dense (non-MoE) layer: the mlp has
        # no shared-expert submodule at all.
        self.assertFalse(
            has_replicated_shared_expert(
                SimpleNamespace(shared_experts=None, _shared_expert_tp1=True)
            )
        )
        self.assertFalse(has_replicated_shared_expert(SimpleNamespace()))


class TestDeferredAllReduceDoubleCounts(CustomTestCase):
    """Why declining the fusion is the fix, not a nicety."""

    @staticmethod
    def _layer(tp_size, shared_inside_allreduce):
        """One layer's post-experts reduction; returns the reduced output.

        ``routed`` are per-rank partials that legitimately sum to a total.
        ``shared`` is replicated: identical on every rank.
        """
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
        # The whole error is (tp_size - 1) extra copies of the shared output.
        self.assertTrue(torch.allclose(wrong - correct, torch.full((4, 1), 49.0)))
        self.assertFalse(torch.allclose(wrong, correct))


class TestFusionGatedOnReplicatedSharedExpert(CustomTestCase):
    """Ratchet: a decoder that both publishes ``fuse_mlp_allreduce`` and can own
    a replicated shared expert must gate the fusion on the invariant.

    Source-level rather than behavioural because reaching the real path needs a
    multi-GPU TP group; the failure mode this guards against is a new model (or
    a refactor) reintroducing an ungated ``fuse_mlp_allreduce`` publication.
    """

    # Model modules that publish fuse_mlp_allreduce AND whose MoE can set
    # _shared_expert_tp1. Keep in sync with `grep -l _shared_expert_tp1`.
    MODULES = ("sglang.srt.models.deepseek_v2", "sglang.srt.models.laguna")

    def test_publication_is_gated(self):
        import importlib

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
                    f"{name}: fuse_mlp_allreduce is published without checking "
                    "has_replicated_shared_expert(self.mlp). Fusing defers the "
                    "post-experts all-reduce to the next layer, past the point "
                    "where the MoE adds a replicated shared output -- which "
                    "then gets summed once per TP rank.",
                )


class TestMandatoryHoistNotGatedOnPerfEnvVar(CustomTestCase):
    """Ratchet: DSV4's shared-expert hoist must not depend only on
    ``SGLANG_DP_SHARED_EXPERT_LOCAL``.

    That variable is a perf PoC and defaults to FALSE (``get_bool_env_var``
    default ``"false"``). But the hoist is also the only thing standing between
    a replicated shared expert and a collective that SUMs it: under CP,
    ``dsa_cp_reduce_scatter_hidden_states``; on the DP path, reduce_scatterv and
    the equal-chunk reduce_scatter. Both replace the MoE-internal all-reduce
    that the TP1 shared expert was supposed to be added after, so gating the
    hoist on a perf switch makes the default configuration the corrupting one.

    Source-level rather than behavioural because reaching the real path needs a
    multi-GPU TP group.
    """

    def test_do_shared_local_includes_the_mandatory_terms(self):
        import importlib

        module = importlib.import_module("sglang.srt.models.deepseek_v4")
        source = inspect.getsource(module)
        found = False
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_do_shared_local"
            ):
                continue
            found = True
            dumped = ast.dump(node.value)
            for term in ("_cp_must_hoist_shared", "_dp_must_hoist_shared"):
                self.assertIn(
                    term,
                    dumped,
                    "_do_shared_local must include "
                    f"{term}: the hoist is mandatory for correctness whenever a "
                    "downstream collective sums the MoE output, and must not be "
                    "reachable only via _SHARED_EXPERT_LOCAL (a perf env var "
                    "that defaults to false).",
                )
        self.assertTrue(found, "_do_shared_local assignment not found in deepseek_v4")


if __name__ == "__main__":
    unittest.main()
