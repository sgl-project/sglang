"""EvenMoE bootstrap and placement commit integration with RuntimeContext."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.eplb.eplb_manager import (
    EPLBManager,
    update_expert_location_with_recovery,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEvenMoERuntimeContext(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_bootstrap_uses_runtime_context_before_metadata_exists(self):
        from evenmoe import MoELoadBalancer
        from evenmoe.adapters.sglang import to_load_balancer_kwargs

        from sglang.srt.model_executor.model_runner import ModelRunner

        context = get_context()
        model_config = object()
        with (
            context.override_server_args(
                moe_load_balancer_algorithm="static",
                tp_size=2,
                ep_size=2,
                ep_num_redundant_experts=2,
            ),
            context.parallel.override(moe_ep_rank=1),
            context.resources.override(mlb_model_info=None),
            patch.object(
                ModelConfigForExpertLocation,
                "from_model_config",
                return_value=ModelConfigForExpertLocation(1, 2, 1),
            ) as extract,
            patch.object(
                ExpertLocationMetadata,
                "_init_common",
                side_effect=AssertionError(
                    "EvenMoE must not call SGLang's layout helper"
                ),
            ),
        ):
            self.assertIsNone(context.resources.expert_location_metadata)
            mlb = ModelRunner._create_moe_load_balancer(
                SimpleNamespace(is_draft_worker=False, model_config=model_config)
            )
            self.assertEqual(
                context.resources.mlb_model_info,
                {"num_logical_experts": 2, "num_groups": 1},
            )
            kwargs = to_load_balancer_kwargs(context)
            self.assertEqual(kwargs["algorithm"], "static")
            self.assertEqual(kwargs["ep_size"], 2)
            self.assertEqual(kwargs["source_rank"], 1)
            self.assertEqual(kwargs["experts_per_rank"], 2)
            self.assertIs(type(mlb), MoELoadBalancer)
            extract.assert_called_once_with(model_config)

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_context_layout_matches_sglang_initial_geometry(self):
        from evenmoe.adapters.sglang.placement import _expert_layout

        context = get_context()
        cases = [
            dict(ep_size=2),
            dict(ep_size=4, elastic_ep_initial_size=2),
            dict(
                ep_size=2,
                elastic_ep_initial_size=2,
                ep_join_mode="scale",
                ep_join_rank_offset=4,
                tp_size=4,
            ),
        ]
        for fields in cases:
            with (
                self.subTest(**fields),
                context.override_server_args(ep_num_redundant_experts=2, **fields),
                context.resources.override(
                    mlb_model_info={"num_logical_experts": 6, "num_groups": None}
                ),
                patch.object(
                    ModelConfigForExpertLocation,
                    "from_model_config",
                    return_value=ModelConfigForExpertLocation(1, 6, None),
                ),
            ):
                expected = ExpertLocationMetadata._init_common(object())
                actual = _expert_layout(context)
                for field in (
                    "ep_size",
                    "num_physical_experts",
                    "num_local_physical_experts",
                ):
                    self.assertEqual(actual[field], expected[field])
                self.assertIsNone(actual["num_groups"])

    def test_model_runner_rejects_models_without_routed_experts(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        )
        with (
            patch.object(
                ModelConfigForExpertLocation, "from_model_config", return_value=None
            ),
            patch(
                "sglang.srt.model_executor.model_runner.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_load_balancer_algorithm="static")
                ),
            ),
            self.assertRaisesRegex(ValueError, "routed experts"),
        ):
            ModelRunner._create_moe_load_balancer(
                SimpleNamespace(is_draft_worker=False, model_config=model_config)
            )

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_model_runner_creates_core_through_context_adapter(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        context = SimpleNamespace(resources=SimpleNamespace(mlb_model_info=None))
        model_config, mlb = object(), object()
        kwargs = dict(algorithm="static", ep_size=2, source_rank=1, experts_per_rank=2)
        with (
            patch.object(
                ModelConfigForExpertLocation,
                "from_model_config",
                return_value=ModelConfigForExpertLocation(1, 2, 1),
            ) as extract,
            patch(
                "sglang.srt.model_executor.model_runner.get_context",
                return_value=context,
            ),
            patch(
                "sglang.srt.model_executor.model_runner.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_load_balancer_algorithm="static")
                ),
            ),
            patch(
                "evenmoe.adapters.sglang.to_load_balancer_kwargs",
                return_value=kwargs,
            ) as adapt,
            patch(
                "evenmoe.MoELoadBalancer.from_algorithm",
                return_value=mlb,
            ) as create,
        ):
            result = ModelRunner._create_moe_load_balancer(
                SimpleNamespace(is_draft_worker=False, model_config=model_config)
            )
        self.assertIs(result, mlb)
        extract.assert_called_once_with(model_config)
        adapt.assert_called_once_with(context)
        create.assert_called_once_with(**kwargs)

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_placement_planning_uses_context_adapter(self):
        import torch
        from evenmoe import MoELoadBalancer

        module = "sglang.srt.eplb.expert_location"
        mlb = Mock(spec=MoELoadBalancer)
        context, model_config = object(), object()
        counts = torch.tensor([[[10, 20]]])
        mapping = torch.tensor([[0, 1, 0, 1]])
        maps = SimpleNamespace(
            physical_to_logical_map=mapping,
            logical_to_all_physical_map=mapping,
            routing_metadata={0: {"rank_quota_prefix": torch.tensor([[1, 2]])}},
        )
        request = object()
        layout = dict(
            ep_size=2,
            num_physical_experts=4,
            model_config_for_expert_location=SimpleNamespace(num_groups=1),
        )
        with (
            patch(f"{module}.get_device", return_value=SimpleNamespace(device="cpu")),
            patch("sglang.srt.runtime_context.get_context", return_value=context),
            patch.object(ExpertLocationMetadata, "_init_common", return_value=layout),
            patch.object(ExpertLocationMetadata, "_init_raw") as initialize,
            patch(
                "sglang.srt.elastic_ep.elastic_ep.ElasticEPStateManager.instance",
                return_value=None,
            ),
            patch(
                "evenmoe.adapters.sglang.to_placement_request",
                return_value=request,
            ) as adapt_request,
            patch(
                "evenmoe.adapters.sglang.to_sglang_maps", return_value=maps
            ) as adapt_result,
        ):
            result = ExpertLocationMetadata.init_by_eplb(
                model_config,
                counts,
                use_flat_topology=True,
                moe_load_balancer=mlb,
            )
        adapt_request.assert_called_once_with(
            counts, context=context, num_nodes=1, active_ranks=None
        )
        mlb.plan_placement.assert_called_once_with(request)
        adapt_result.assert_called_once_with(mlb.plan_placement.return_value)
        self.assertIs(result, initialize.return_value)
        self.assertIs(initialize.call_args.kwargs["physical_to_logical_map"], mapping)
        self.assertIs(
            initialize.call_args.kwargs["mlb_routing_metadata"], maps.routing_metadata
        )

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_topk_modules_share_the_load_balancer_and_commit_their_layers(self):
        from sglang.srt.layers.moe.hash_topk import HashTopK
        from sglang.srt.layers.moe.topk import TopK
        from sglang.srt.model_executor.model_runner import ModelRunner

        topk = Mock(spec=TopK, layer_id=2)
        hash_topk = Mock(spec=HashTopK, layer_id=3)
        mlb = Mock()
        runner = SimpleNamespace(
            moe_load_balancer=mlb,
            model=SimpleNamespace(modules=lambda: [object(), topk, hash_topk]),
        )
        context = object()
        with (
            patch(
                "sglang.srt.model_executor.model_runner.get_context",
                return_value=context,
            ),
            patch("evenmoe.adapters.sglang.commit_placement") as commit,
        ):
            ModelRunner._prepare_moe_topk(runner)
        self.assertIs(topk.moe_load_balancer, mlb)
        self.assertIs(hash_topk.moe_load_balancer, mlb)
        commit.assert_called_once_with(mlb, context, [2, 3])

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_rebalance_commits_each_chunk_to_the_same_load_balancer(self):
        order = []
        mlb = Mock()
        updater = Mock()
        updater.update.side_effect = lambda *a, **kw: (
            order.append(("move", kw["update_layer_ids"])) or {}
        )
        manager = Mock(
            _rebalance_disabled_reason=None,
            _rebalance_layers_per_chunk=1,
            _moe_load_balancer=mlb,
            _ps=SimpleNamespace(tp_rank=0),
        )
        manager._compute_update_layer_ids_chunks.return_value = [[2], [3]]
        manager._check_rebalance_needed.return_value = True
        manager._get_expert_location_updater.return_value = updater
        recorder = Mock()
        recorder.dump_record.return_value = {
            "logical_count": object(),
            "average_utilization_rate_over_window": 0.5,
        }
        module = "sglang.srt.eplb.eplb_manager"
        with (
            patch(
                "evenmoe.adapters.sglang.commit_placement",
                side_effect=lambda core, context, ids: order.append(("commit", ids)),
            ) as commit,
            patch(f"{module}.ElasticEPStateManager.instance", return_value=None),
            patch(
                f"{module}.get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
            patch(
                f"{module}.get_parallel",
                return_value=SimpleNamespace(nnodes=1, tp_rank=0),
            ),
            patch(
                f"{module}.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_dispatch_algorithm=None)
                ),
            ),
        ):
            list(EPLBManager.rebalance(manager))
        self.assertEqual(
            order, [("move", [2]), ("commit", [2]), ("move", [3]), ("commit", [3])]
        )
        self.assertTrue(all(call.args[0] is mlb for call in commit.call_args_list))

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_glue_routes_once_and_records_current_resources(self):
        from sglang.srt.eplb import moe_load_balancer_glue as glue

        output = SimpleNamespace(
            topk_ids=object(),
            topk_weights=object(),
            router_logits=object(),
            recorded_physical_topk_ids=object(),
        )
        context = SimpleNamespace(
            resources=SimpleNamespace(
                expert_distribution_recorder=Mock(),
                experts_capturer=Mock(),
                mlb_placement_refresh=None,
            )
        )
        mlb = Mock()
        topk = SimpleNamespace(
            topk_ids=object(), topk_weights=object(), router_logits=object()
        )
        request = object()
        with (
            patch.object(glue, "get_context", return_value=context),
            patch(
                "evenmoe.adapters.sglang.to_routing_request",
                return_value=request,
            ) as adapt_request,
            patch(
                "evenmoe.adapters.sglang.to_sglang_routing_output",
                return_value=output,
            ) as adapt_result,
        ):
            result = glue.route_topk_with_mlb(
                moe_load_balancer=mlb,
                layer_id=3,
                topk_output=topk,
                num_tokens=2,
                num_token_non_padded=None,
                routed_scaling_factor=1.0,
            )
        adapt_request.assert_called_once_with(
            context=context,
            layer_id=3,
            logical_topk_ids=topk.topk_ids,
            topk_weights=topk.topk_weights,
            token_count=2,
            stage=None,
            routed_scaling_factor=1.0,
        )
        mlb.route_tokens.assert_called_once_with(request)
        adapt_result.assert_called_once_with(
            mlb.route_tokens.return_value,
            context=context,
            router_logits=topk.router_logits,
            routed_scaling_factor=1.0,
        )
        context.resources.expert_distribution_recorder.on_select_experts.assert_called_once_with(
            topk_ids=output.recorded_physical_topk_ids
        )
        context.resources.experts_capturer.capture.assert_called_once_with(
            layer_id=3, topk_indices=output.recorded_physical_topk_ids
        )
        self.assertIs(result.topk_ids, output.topk_ids)
        self.assertIs(result.topk_weights, output.topk_weights)
        self.assertIs(result.router_logits, output.router_logits)

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_commit_reads_current_resources_and_notifies_each_layer(self):
        from evenmoe.adapters.sglang import commit_placement

        mlb = Mock()
        context = SimpleNamespace(
            resources=SimpleNamespace(expert_location_metadata=object())
        )
        with patch(
            "evenmoe.adapters.sglang.placement.to_placement_snapshot",
            side_effect=lambda metadata, layer_id: SimpleNamespace(
                metadata=metadata, layer_id=layer_id
            ),
        ):
            commit_placement(mlb, context, [2, 3])
            snapshots = [
                call.args[0] for call in mlb.on_placement_committed.call_args_list
            ]
            self.assertEqual([snapshot.layer_id for snapshot in snapshots], [2, 3])
            self.assertTrue(
                all(
                    snapshot.metadata is context.resources.expert_location_metadata
                    for snapshot in snapshots
                )
            )
            context.resources = SimpleNamespace(expert_location_metadata=object())
            self.assertEqual(mlb.on_placement_committed.call_count, 2)
            commit_placement(mlb, context, [3])
            self.assertEqual(mlb.on_placement_committed.call_count, 3)
            self.assertIs(
                mlb.on_placement_committed.call_args.args[0].metadata,
                context.resources.expert_location_metadata,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_commit_skips_policies_without_placement_state(self):
        from evenmoe.adapters.sglang import commit_placement

        mlb = Mock()
        mlb.routing_capabilities.requires_placement_state = False
        commit_placement(mlb, object(), [2])
        mlb.on_placement_committed.assert_not_called()

    @unittest.skipUnless(
        importlib.util.find_spec("evenmoe"),
        "evenmoe is not installed",
    )
    def test_commit_rejects_missing_metadata_before_notifying_core(self):
        from evenmoe.adapters.sglang import commit_placement

        mlb = Mock()
        context = SimpleNamespace(
            resources=SimpleNamespace(expert_location_metadata=None)
        )
        with self.assertRaisesRegex(RuntimeError, "committed expert metadata"):
            commit_placement(mlb, context, [2])
        mlb.on_placement_committed.assert_not_called()

    def test_commit_follows_weight_update_and_recovery(self):
        order = []
        updater = Mock()
        updater.update.side_effect = lambda *a, **kw: order.append("move") or {0: [1]}
        backup = Mock(use_backup=True)
        backup.update_weights.side_effect = lambda *a: order.append("recover")
        model = SimpleNamespace(
            routed_experts_weights_of_layer={},
            generate_weight_name_filter=lambda missing: missing,
        )
        committed = Mock(side_effect=lambda layers: order.append("commit"))
        with patch(
            "sglang.srt.eplb.eplb_manager.get_parallel",
            return_value=SimpleNamespace(nnodes=1, tp_rank=0),
        ):
            update_expert_location_with_recovery(
                expert_location_updater=updater,
                model=model,
                new_expert_location_metadata=object(),
                update_layer_ids=[0],
                tp_rank=0,
                expert_backup_client=backup,
                update_weights_from_disk_callable=Mock(),
                ep_dispatch_algorithm=None,
                init_lplb_solvers_callable=Mock(),
                on_placement_committed=committed,
            )
        self.assertEqual(order, ["move", "recover", "commit"])
        committed.assert_called_once_with([0])

    def test_failed_recovery_does_not_commit(self):
        updater = Mock()
        updater.update.return_value = {0: [1]}
        backup = Mock(use_backup=True)
        backup.update_weights.side_effect = RuntimeError("recovery failed")
        committed = Mock()
        with (
            self.assertRaisesRegex(RuntimeError, "recovery failed"),
            patch(
                "sglang.srt.eplb.eplb_manager.get_parallel",
                return_value=SimpleNamespace(nnodes=1, tp_rank=0),
            ),
        ):
            update_expert_location_with_recovery(
                expert_location_updater=updater,
                model=SimpleNamespace(routed_experts_weights_of_layer={}),
                new_expert_location_metadata=object(),
                update_layer_ids=[0],
                tp_rank=0,
                expert_backup_client=backup,
                update_weights_from_disk_callable=Mock(),
                ep_dispatch_algorithm=None,
                init_lplb_solvers_callable=Mock(),
                on_placement_committed=committed,
            )
        committed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
