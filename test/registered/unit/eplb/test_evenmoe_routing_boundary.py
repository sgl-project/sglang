"""The framework consumes routing capabilities without owning policy state."""

import importlib.util
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKConfig
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
    def test_draft_build_excludes_target_placement_and_restores_on_failure(self):
        from sglang.srt.runtime_context import get_exec

        with get_context().override_server_args(
            moe_load_balancer_algorithm="lplb+waterfill", ep_num_redundant_experts=8
        ):
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                with draft_model_build_scope():
                    self.assertIsNone(get_exec().moe.moe_load_balancer_algorithm)
                    # The draft keeps the native redundant slots, unused as on main.
                    self.assertEqual(get_exec().moe.ep_num_redundant_experts, 8)
                    raise RuntimeError("draft failed")
            self.assertEqual(
                get_exec().moe.moe_load_balancer_algorithm, "lplb+waterfill"
            )

    def test_empty_target_calls_balancer_but_draft_keeps_native_slots(self):
        pipeline = SimpleNamespace(
            capabilities=SimpleNamespace(requires_post_topk_routing=True)
        )
        shared = SharedExpertsConfig(
            num_fused_shared_experts=1, routed_by_load_balancer=True
        )
        mlb = object()
        for router_type in (TopK, HashTopK):
            with (
                self.subTest(router=router_type.__name__),
                get_context().override_server_args(
                    moe_load_balancer_algorithm="waterfill"
                ),
                get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
                get_context().resources.override(moe_load_balancer=mlb),
                # Parse the configured algorithm, so the draft scope's override applies.
                patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.get_moe_load_balancer_pipeline",
                    side_effect=lambda algorithm: pipeline if algorithm else None,
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
                self.assertIs(target.moe_load_balancer, mlb)
                self.assertIsNone(draft.moe_load_balancer)

                result = StandardTopKOutput(
                    torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.int32), None
                )
                with patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                    return_value=result,
                ) as route:
                    actual = target.empty_topk_output(torch.device("cpu"))
                    self.assertIs(actual.topk_ids, result.topk_ids)
                    self.assertIs(actual.topk_weights, result.topk_weights)
                    draft.empty_topk_output(torch.device("cpu"))
                route.assert_called_once()
                self.assertEqual(route.call_args.kwargs["layer_id"], 7)
                self.assertEqual(
                    route.call_args.kwargs["topk_output"].topk_ids.shape, (0, 2)
                )

    def test_routed_experts_capture_keeps_logical_ids_under_balancer(self):
        class Capture:
            def __init__(self):
                self.ids = None

            def capture(self, *, layer_id, topk_indices):
                self.ids = topk_indices.clone()

        logical = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        physical = StandardTopKOutput(weights, logical + 8, None)
        capture = Capture()
        with (
            get_context().resources.override(experts_capturer=capture),
            patch(
                "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                return_value=physical,
            ),
        ):
            ids, _, _ = topk_module._post_process_topk_ids(
                topk_ids=logical.clone(),
                topk_weights=weights.clone(),
                topk_config=TopKConfig(top_k=2),
                router_logits=torch.zeros((2, 4)),
                layer_id=0,
                moe_load_balancer=object(),
            )
        # Users get the model's logical experts; dispatch uses the physical ids.
        self.assertEqual(capture.ids.tolist(), logical.tolist())
        self.assertEqual(ids.tolist(), (logical + 8).tolist())

    def test_routing_router_requires_published_balancer(self):
        pipeline = SimpleNamespace(
            capabilities=SimpleNamespace(requires_post_topk_routing=True)
        )
        with (
            get_context().override_server_args(moe_load_balancer_algorithm="lplb"),
            patch(
                "sglang.srt.eplb.moe_load_balancer_glue.get_moe_load_balancer_pipeline",
                return_value=pipeline,
            ),
        ):
            with (
                get_context().resources.override(moe_load_balancer=None),
                self.assertRaisesRegex(RuntimeError, "no MoELoadBalancer"),
            ):
                TopK(top_k=2, layer_id=0)

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

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"), "evenmoe is not installed"
    )
    def test_model_load_checks_the_policy_layout_preconditions(self):
        from evenmoe import RoutingPipeline

        cases = (
            # (num_logical_experts, ep_size, num_redundant_experts, error)
            (4, 2, 2, None),
            (5, 2, 2, "split evenly"),
            (4, 2, 3, "split evenly"),
            (4, 2, 0, "redundant"),
            (4, 1, 1, "more than one rank"),
        )
        context = get_context()
        for num_logical_experts, ep_size, num_redundant, error in cases:
            with (
                self.subTest(
                    num_logical_experts=num_logical_experts,
                    ep_size=ep_size,
                    num_redundant=num_redundant,
                ),
                context.override_server_args(
                    tp_size=ep_size,
                    ep_size=ep_size,
                    ep_num_redundant_experts=num_redundant,
                ),
                context.resources.override(
                    mlb_model_info={"num_logical_experts": num_logical_experts}
                ),
                patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.get_configured_moe_load_balancer_pipeline",
                    return_value=RoutingPipeline.from_value("ultraep"),
                ),
            ):
                config = SimpleNamespace(disable_reason=None, shared_count=0)
                if error is None:
                    resolve_shared_experts_config(_SharedModel, config, None)
                else:
                    with self.assertRaisesRegex(ValueError, error):
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


class TestLoadBalancerBeforeSharedLayout(CustomTestCase):
    @contextmanager
    def _boundary_context(self, per_rank=True):
        with ExitStack() as stack:
            stack.enter_context(
                get_context().override_server_args(tp_size=2, ep_size=2)
            )
            stack.enter_context(
                get_parallel().override(
                    moe_ep_size=2, moe_ep_rank=1, tp_group=SimpleNamespace(world_size=2)
                )
            )
            stack.enter_context(
                patch.object(
                    topk_module,
                    "has_per_rank_fused_shared_slots",
                    return_value=per_rank,
                )
            )
            for name in ("_is_cuda", "_is_hip", "_is_npu", "_use_aiter"):
                stack.enter_context(patch.object(topk_module, name, False))
            stack.enter_context(
                patch(
                    "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
                    return_value=SimpleNamespace(num_physical_experts=8),
                )
            )
            # Any second logical-to-physical remap would corrupt the balancer's IDs.
            stack.enter_context(
                patch.object(
                    topk_module,
                    "topk_ids_logical_to_physical",
                    side_effect=AssertionError("physical IDs remapped twice"),
                )
            )
            yield

    def test_shared_columns_are_removed_before_routing_and_layout(self):
        for shared_count in (1, 2):
            for gate_appended in (False, True):
                for folded_scale in (False, True):
                    with (
                        self.subTest(
                            shared=shared_count,
                            appended=gate_appended,
                            folded=folded_scale,
                        ),
                        self._boundary_context(),
                    ):
                        ids = torch.tensor([[0, 3]], dtype=torch.int32)
                        weights = torch.tensor([[0.25, 0.75]])
                        if gate_appended:
                            ids = torch.cat(
                                [ids, torch.arange(4, 4 + shared_count)[None, :]], dim=1
                            )
                            weights = torch.cat(
                                [weights, torch.full((1, shared_count), 99.0)], dim=1
                            )
                        config = TopKConfig(
                            top_k=2 + shared_count,
                            shared_experts=SharedExpertsConfig(
                                num_fused_shared_experts=shared_count
                            ),
                            routed_scaling_factor=2.5,
                            apply_routed_scaling_factor_on_output=folded_scale,
                        )

                        def route(**kwargs):
                            raw = kwargs["topk_output"]
                            torch.testing.assert_close(
                                raw.topk_ids, torch.tensor([[0, 3]], dtype=ids.dtype)
                            )
                            torch.testing.assert_close(
                                raw.topk_weights, torch.tensor([[0.25, 0.75]])
                            )
                            return StandardTopKOutput(
                                raw.topk_weights,
                                raw.topk_ids.new_tensor([[5, 1]]),
                                raw.router_logits,
                            )

                        with patch(
                            "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                            side_effect=route,
                        ) as routed:
                            out_ids, out_weights, recorder_ids = (
                                topk_module._post_process_topk_ids(
                                    ids,
                                    weights,
                                    config,
                                    torch.zeros(1, 4),
                                    7,
                                    moe_load_balancer=object(),
                                )
                            )
                        routed.assert_called_once()
                        torch.testing.assert_close(
                            out_ids[:, :2], ids.new_tensor([[5 + shared_count, 1]])
                        )
                        torch.testing.assert_close(
                            out_ids[:, 2:],
                            torch.arange(
                                8 + shared_count, 8 + 2 * shared_count, dtype=ids.dtype
                            )[None, :],
                        )
                        torch.testing.assert_close(
                            out_weights[:, :2], torch.tensor([[0.25, 0.75]])
                        )
                        torch.testing.assert_close(
                            out_weights[:, 2:],
                            torch.full((1, shared_count), 1.0 if folded_scale else 0.4),
                        )
                        self.assertIsNone(recorder_ids)

    def test_global_shared_slot_follows_redundant_physical_experts(self):
        with self._boundary_context(per_rank=False):
            ids = torch.tensor([[0, 3, 4]], dtype=torch.int32)
            weights = torch.tensor([[0.25, 0.75, 0.4]])
            config = TopKConfig(
                top_k=3,
                shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
                fused_shared_experts_scaling_factor=0.5,
            )
            with patch(
                "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                return_value=StandardTopKOutput(
                    weights[:, :2], ids.new_tensor([[5, 1]]), None
                ),
            ):
                out_ids, out_weights, _ = topk_module._post_process_topk_ids(
                    ids,
                    weights,
                    config,
                    torch.zeros(1, 4),
                    7,
                    moe_load_balancer=object(),
                )
            torch.testing.assert_close(out_ids, ids.new_tensor([[5, 1, 8]]))
            torch.testing.assert_close(out_weights, torch.tensor([[0.25, 0.75, 0.2]]))

    def test_waterfill_output_is_not_expanded_twice(self):
        with self._boundary_context():
            config = TopKConfig(
                top_k=2,
                shared_experts=SharedExpertsConfig(
                    num_fused_shared_experts=1, routed_by_load_balancer=True
                ),
            )
            # has_per_rank_fused_shared_slots must still respect a zero native-slot count.
            with (
                patch.object(
                    topk_module,
                    "has_per_rank_fused_shared_slots",
                    side_effect=lambda count: count > 0,
                ),
                patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                    return_value=StandardTopKOutput(
                        torch.tensor([[0.25, 0.75, 0.4]]),
                        torch.tensor([[6, 1, 9]]),
                        None,
                    ),
                ) as route,
            ):
                out_ids, out_weights, _ = topk_module._post_process_topk_ids(
                    torch.tensor([[0, 3]]),
                    torch.tensor([[0.25, 0.75]]),
                    config,
                    torch.zeros(1, 4),
                    7,
                    moe_load_balancer=object(),
                )
            self.assertEqual(
                route.call_args.kwargs["topk_output"].topk_ids.shape, (1, 2)
            )
            torch.testing.assert_close(out_ids, torch.tensor([[6, 1, 9]]))
            self.assertEqual(out_weights.shape, (1, 3))

    def test_padding_is_masked_before_routing_and_after_shared_append(self):
        with self._boundary_context():
            config = TopKConfig(
                top_k=3, shared_experts=SharedExpertsConfig(num_fused_shared_experts=1)
            )

            def route(**kwargs):
                raw = kwargs["topk_output"]
                self.assertTrue(raw.topk_ids.is_contiguous())
                self.assertTrue(raw.topk_weights.is_contiguous())
                torch.testing.assert_close(
                    raw.topk_ids, torch.tensor([[0, 3], [-1, -1]])
                )
                return StandardTopKOutput(
                    raw.topk_weights,
                    torch.tensor([[5, 1], [-1, -1]]),
                    raw.router_logits,
                )

            with patch(
                "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                side_effect=route,
            ):
                ids, _, _ = topk_module._post_process_topk_ids(
                    torch.tensor([[0, 3, 4], [2, 1, 4]]),
                    torch.ones(2, 3),
                    config,
                    torch.zeros(2, 4),
                    7,
                    num_token_non_padded=torch.tensor(1),
                    moe_load_balancer=object(),
                )
            torch.testing.assert_close(ids, torch.tensor([[6, 1, 9], [-1, -1, -1]]))

    def test_topk_and_hash_forward_route_once_before_shared_layout(self):
        pipeline = SimpleNamespace(
            capabilities=SimpleNamespace(requires_post_topk_routing=True)
        )
        mlb = Mock()
        mlb.transform_topk_inputs.side_effect = lambda logits, bias: (logits, bias)
        for router_type in (TopK, HashTopK):
            with (
                self.subTest(router=router_type.__name__),
                self._boundary_context(),
                get_model().override(
                    shared_experts=SharedExpertsConfig(num_fused_shared_experts=1)
                ),
                get_context().resources.override(moe_load_balancer=mlb),
                patch(
                    "sglang.srt.eplb.moe_load_balancer_glue.get_configured_moe_load_balancer_pipeline",
                    return_value=pipeline,
                ),
            ):
                router = (
                    TopK(top_k=2, layer_id=7, routed_scaling_factor=2.5)
                    if router_type is TopK
                    else HashTopK(
                        topk=2,
                        num_experts=4,
                        vocab_size=4,
                        layer_id=7,
                        routed_scaling_factor=2.5,
                    )
                )

                def route(**kwargs):
                    raw = kwargs["topk_output"]
                    torch.testing.assert_close(
                        raw.topk_ids, torch.tensor([[0, 1]], dtype=raw.topk_ids.dtype)
                    )
                    return StandardTopKOutput(
                        raw.topk_weights,
                        raw.topk_ids.new_tensor([[5, 1]]),
                        raw.router_logits,
                    )

                with (
                    patch(
                        "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                        side_effect=route,
                    ) as routed,
                    patch(
                        "sglang.srt.layers.moe.hash_topk.envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get",
                        return_value=False,
                    ),
                ):
                    if router_type is TopK:
                        result = router.forward_native(
                            torch.zeros(1, 4), torch.tensor([[4.0, 3.0, 2.0, 1.0]])
                        )
                    else:
                        result = router(
                            torch.zeros(1, 4),
                            torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
                            torch.tensor([0]),
                        )
                routed.assert_called_once()
                torch.testing.assert_close(
                    result.topk_ids, result.topk_ids.new_tensor([[6, 1, 9]])
                )

    def test_empty_ranks_route_without_shared_columns(self):
        pipeline = SimpleNamespace(
            capabilities=SimpleNamespace(requires_post_topk_routing=True)
        )
        for router_type in (TopK, HashTopK):
            for per_rank in (False, True):
                with (
                    self.subTest(router=router_type.__name__, per_rank=per_rank),
                    self._boundary_context(per_rank=per_rank),
                    get_model().override(
                        shared_experts=SharedExpertsConfig(num_fused_shared_experts=1)
                    ),
                    get_context().resources.override(moe_load_balancer=object()),
                    patch(
                        "sglang.srt.eplb.moe_load_balancer_glue.get_configured_moe_load_balancer_pipeline",
                        return_value=pipeline,
                    ),
                ):
                    router = (
                        TopK(top_k=2, layer_id=7)
                        if router_type is TopK
                        else HashTopK(topk=2, num_experts=4, vocab_size=4, layer_id=7)
                    )
                    with patch(
                        "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
                        side_effect=lambda **kwargs: kwargs["topk_output"],
                    ) as route:
                        result = router.empty_topk_output(torch.device("cpu"))
                    route.assert_called_once()
                    self.assertEqual(
                        route.call_args.kwargs["topk_output"].topk_ids.shape, (0, 2)
                    )
                    self.assertEqual(result.topk_ids.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
