"""The framework consumes routing capabilities without owning policy state."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    resolve_shared_experts_config,
)
from sglang.srt.runtime_context import get_context, get_model, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _SharedModel:
    @staticmethod
    def shared_experts_fusion_disable_reason(config, quant_config):
        return config.disable_reason

    @staticmethod
    def num_fusable_shared_experts(config):
        return config.shared_count


class TestRoutingBoundary(CustomTestCase):
    def test_draft_build_excludes_target_replicas_and_restores_on_failure(self):
        from sglang.srt.runtime_context import get_exec

        with get_context().override_server_args(
            moe_load_balancer_algorithm="lplb+waterfill", ep_num_redundant_experts=8
        ):
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                with draft_model_build_scope():
                    self.assertIsNone(get_exec().moe.moe_load_balancer_algorithm)
                    self.assertEqual(get_exec().moe.ep_num_redundant_experts, 0)
                    raise RuntimeError("draft failed")
            self.assertEqual(
                get_exec().moe.moe_load_balancer_algorithm, "lplb+waterfill"
            )
            self.assertEqual(get_exec().moe.ep_num_redundant_experts, 8)

    def test_empty_target_calls_balancer_but_draft_keeps_native_slots(self):
        pipeline = SimpleNamespace(
            capabilities=SimpleNamespace(requires_post_topk_routing=True)
        )
        shared = SharedExpertsConfig(
            num_fused_shared_experts=1, routed_by_load_balancer=True
        )
        for router_type in (TopK, HashTopK):
            with (
                self.subTest(router=router_type.__name__),
                get_context().override_server_args(),
                get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
                patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.get_configured_moe_load_balancer_pipeline",
                    return_value=pipeline,
                ),
            ):

                def make_router():
                    if router_type is TopK:
                        return TopK(top_k=2, layer_id=7)
                    return HashTopK(topk=2, num_experts=8, vocab_size=4, layer_id=7)

                with get_model().override(shared_experts=shared):
                    target = make_router()
                with draft_model_build_scope():
                    draft_config = resolve_shared_experts_config(
                        _SharedModel,
                        SimpleNamespace(disable_reason=None, shared_count=1),
                        None,
                    )
                    with get_model().override(shared_experts=draft_config):
                        draft = make_router()
                self.assertFalse(draft_config.routed_by_load_balancer)
                self.assertEqual(draft_config.num_shared_topk_slots, 1)
                self.assertFalse(draft.mlb_requires_post_topk_routing)

                target.moe_load_balancer = object()
                result = StandardTopKOutput(
                    torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.int32), None
                )
                with patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                    return_value=result,
                ) as route:
                    self.assertIs(target.empty_topk_output(torch.device("cpu")), result)
                    draft.empty_topk_output(torch.device("cpu"))
                route.assert_called_once()
                self.assertEqual(route.call_args.kwargs["layer_id"], 7)
                self.assertEqual(
                    route.call_args.kwargs["topk_output"].topk_ids.shape, (0, 2)
                )

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"), "evenmoe is not installed"
    )
    def test_model_validation_uses_resolved_layout_and_evenmoe_requirements(self):
        from evenmoe import RoutingPipeline

        for name, redundant, count, disabled, valid in (
            ("waterfill", 0, 1, None, True),
            ("waterfill", 0, 0, None, False),
            ("waterfill", 0, 1, "checkpoint cannot fuse", False),
            ("waterfill", 0, 2, None, False),
            ("lplb", 0, 0, None, False),
            ("lplb", 2, 0, None, True),
        ):
            with (
                self.subTest(
                    name=name, redundant=redundant, count=count, disabled=disabled
                ),
                get_context().override_server_args(ep_num_redundant_experts=redundant),
                patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.get_configured_moe_load_balancer_pipeline",
                    return_value=RoutingPipeline.from_value(name),
                ),
            ):
                config = SimpleNamespace(disable_reason=disabled, shared_count=count)
                if valid:
                    layout = resolve_shared_experts_config(_SharedModel, config, None)
                    self.assertEqual(
                        layout.routed_by_load_balancer, name == "waterfill"
                    )
                else:
                    with self.assertRaises(ValueError):
                        resolve_shared_experts_config(_SharedModel, config, None)

    def test_unconfigured_path_and_draft_do_not_import_evenmoe(self):
        import sys

        for draft in (False, True):
            with (
                self.subTest(draft=draft),
                get_context().override_server_args(),
                patch.dict(sys.modules, {"evenmoe": None}),
            ):
                config = SimpleNamespace(disable_reason=None, shared_count=1)
                if draft:
                    with draft_model_build_scope():
                        result = resolve_shared_experts_config(
                            _SharedModel, config, None
                        )
                else:
                    result = resolve_shared_experts_config(_SharedModel, config, None)
                self.assertFalse(result.routed_by_load_balancer)
                self.assertEqual(result.num_shared_topk_slots, 1)


if __name__ == "__main__":
    unittest.main()
