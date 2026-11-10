"""Coupled EvenMoE placement: the L1 policy re-plans each layer from live
traffic and its plan state must commit together with the expert weights."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("evenmoe") is None, reason="evenmoe is not installed"
)

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.runtime_context import get_context


def metadata(quota, layers=(0, 1)):
    """Two-layer live metadata, or a one-layer plan keyed by its target layer."""
    n = len(layers)
    return ExpertLocationMetadata(
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]] * n),
        physical_to_logical_map_cpu=torch.tensor([[0, 1, 0, 1]] * n),
        logical_to_all_physical_map=torch.tensor(
            [[[0, 2, -1, -1], [1, 3, -1, -1]]] * n
        ),
        logical_to_all_physical_map_cpu=torch.tensor(
            [[[0, 2, -1, -1], [1, 3, -1, -1]]] * n
        ),
        logical_to_all_physical_map_num_valid=torch.tensor([[2, 2]] * n),
        ep_size=2,
        logical_to_rank_dispatch_physical_map=None,
        mlb_routing_metadata={
            layer: {"rank_quota_prefix": q} for layer, q in zip(layers, quota)
        },
    )


def test_partial_commit_keeps_other_layer_quota():
    from evenmoe.adapters.sglang import to_placement_snapshot

    old = [torch.tensor([[1, 2], [1, 2]]) for _ in range(2)]
    new = [torch.tensor([[3, 4], [3, 4]]) for _ in range(2)]
    live = metadata(old)
    live.update(metadata(new), [1])
    assert to_placement_snapshot(live, 0).metadata["rank_quota_prefix"] is old[0]
    assert to_placement_snapshot(live, 1).metadata["rank_quota_prefix"] is new[1]


def test_single_layer_commit_moves_weights_before_publishing_metadata():
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater

    old = [torch.ones(2, 2) for _ in range(2)]
    new = torch.full((2, 2), 3)
    live = metadata(old)
    plan = metadata([new], layers=(1,))
    plan.physical_to_logical_map[0] = torch.tensor([1, 0, 1, 0])
    plan.physical_to_logical_map_cpu.copy_(plan.physical_to_logical_map)
    weights = [torch.zeros(2, 4), torch.ones(2, 1)]
    updater = ExpertLocationUpdater()
    buffers = []

    def move(**kwargs):
        if not buffers:
            # Weights move against the still-live placement; metadata lands after.
            assert live.mlb_routing_metadata[1]["rank_quota_prefix"] is old[1]
            assert kwargs["old_physical_to_logical_map"] == [0, 1, 0, 1]
            assert kwargs["new_physical_to_logical_map"] == [1, 0, 1, 0]
        buffers.append(kwargs["temp_buffers"])

    with (
        get_context().resources.override(expert_location_metadata=live),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.barrier") as initialize_group,
        patch(
            "sglang.srt.eplb.expert_location_updater.update_expert_weights_single_layer",
            side_effect=move,
        ),
    ):
        updater.update_layer(weights, plan, 1, nnodes=1, rank=0)
        updater.update_layer(weights, plan, 1, nnodes=1, rank=0)
    initialize_group.assert_called_once_with()
    assert len(buffers) == 2 and buffers[0] is buffers[1]  # one barrier, one scratch
    assert live.mlb_routing_metadata[1]["rank_quota_prefix"] is new
    assert live.mlb_routing_metadata[0]["rank_quota_prefix"] is old[0]
    torch.testing.assert_close(
        live.physical_to_logical_map[0], torch.tensor([0, 1, 0, 1])
    )
    torch.testing.assert_close(
        live.physical_to_logical_map[1], torch.tensor([1, 0, 1, 0])
    )
    torch.testing.assert_close(
        live.physical_to_logical_map_cpu, live.physical_to_logical_map
    )


def test_routing_glue_refreshes_placement_before_routing():
    from sglang.srt.eplb.moe_load_balancer_glue import route_topk_with_mlb

    events = []
    mlb = Mock()
    mlb.route_tokens.side_effect = lambda request: events.append("route")
    output = SimpleNamespace(
        topk_ids=torch.empty((0, 2), dtype=torch.int32),
        topk_weights=torch.empty((0, 2)),
        router_logits=None,
    )
    with (
        get_context().resources.override(
            mlb_placement_refresh=lambda layer, ids: events.append(
                ("refresh", layer, ids is output.topk_ids)
            ),
            expert_distribution_recorder=Mock(),
        ),
        patch("evenmoe.adapters.sglang.to_routing_request"),
        patch("evenmoe.adapters.sglang.to_sglang_routing_output"),
    ):
        route_topk_with_mlb(
            moe_load_balancer=mlb,
            layer_id=3,
            topk_output=output,
            num_tokens=0,
            num_token_non_padded=None,
            routed_scaling_factor=1.0,
        )
    assert events == [("refresh", 3, True), "route"]


@pytest.mark.parametrize("empty", [False, True])
def test_current_batch_refresh_includes_idle_rank_and_commits_before_routing(empty):
    from sglang.srt.eplb.layer_refresh import LayerPlacementRefresher

    context = get_context()
    live = metadata([torch.ones(2, 2), torch.ones(2, 2)])
    plan = metadata([torch.full((2, 2), 4)], layers=(1,))
    events = []
    from evenmoe import MoELoadBalancer
    from evenmoe.adapters.sglang.routing import _SGLangCollectives

    mlb = Mock()
    mlb.routing_capabilities.requires_placement_state = True
    # EvenMoE combines the counts over SGLang's MoE EP group, the real way.
    mlb.aggregate_logical_count.side_effect = MoELoadBalancer(
        routing_pipeline="ultraep", collectives=_SGLangCollectives(context)
    ).aggregate_logical_count
    mlb.plan_placement.return_value = SimpleNamespace(
        physical_to_logical_map=plan.physical_to_logical_map,
        logical_to_all_physical_map=plan.logical_to_all_physical_map,
        logical_to_physical_count=plan.logical_to_all_physical_map_num_valid,
        metadata={"rank_quota_prefix": torch.full((1, 2, 2), 4)},
    )

    def solve(request):
        assert request.policy.metadata == {"rank": 0, "num_nvl_ranks": 2}
        torch.testing.assert_close(
            request.stats.logical_count[:, 0],
            torch.tensor([[0, 0] if empty else [2, 1], [4, 2]], dtype=torch.int32),
        )
        events.append("solve")
        return mlb.plan_placement.return_value

    mlb.plan_placement.side_effect = solve
    mlb.on_placement_committed.side_effect = lambda _: events.append("publish")
    updater = Mock()

    def move(weights, layer_metadata, layer_id, **kwargs):
        assert layer_id == 1 and list(layer_metadata.mlb_routing_metadata) == [1]
        events.append("move")

    updater.update_layer.side_effect = move
    manager = LayerPlacementRefresher(
        moe_load_balancer=mlb,
        get_model=lambda: SimpleNamespace(routed_experts_weights_of_layer={1: []}),
        get_expert_location_updater=lambda: updater,
        refresh_interval=2,
    )
    manager._forward_batch = SimpleNamespace(
        global_forward_mode=SimpleNamespace(
            is_decode=lambda: False, is_idle=lambda: False
        ),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        is_extend_in_batch=True,
        original_global_num_tokens_cpu=[0 if empty else 2, 1024],
    )

    payloads = []

    def all_reduce(payload):
        # Rank 1 contributes [4, 2] to its own row of the flattened rank rows.
        payloads.append(payload.clone())
        peer = torch.zeros_like(payload)
        peer.view(2, -1)[1] = payload.new_tensor([4, 2])
        return payload + peer

    def init_raw(**kwargs):
        plan.mlb_routing_metadata = kwargs["mlb_routing_metadata"]
        return plan

    with (
        context.override_server_args(
            device="cpu",
            tp_size=2,
            ep_size=2,
            ep_num_redundant_experts=2,
            moe_load_balancer_algorithm="ultraep",
        ),
        context.resources.override(
            expert_location_metadata=live,
            mlb_model_info={"num_logical_experts": 2, "num_groups": 1},
        ),
        context.parallel.override(
            moe_ep_rank=0,
            tp_rank=0,
            moe_ep_group=SimpleNamespace(
                rank_in_group=0, world_size=2, all_reduce=all_reduce
            ),
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        patch.object(ExpertLocationMetadata, "_init_raw", side_effect=init_raw),
    ):
        ids = torch.tensor([[0, 1], [0, -1]])
        if empty:
            ids = ids[:0]
        manager.refresh_layer(1, ids)  # batch 1 of 2: not due yet
        manager.refresh_layer(1, ids)  # due
        assert len(payloads) == 1
        # One flat [ranks * experts] row per rank: a 2-D int tensor could reach
        # SGLang's FlashInfer all-reduce, which assumes [tokens, hidden].
        assert payloads[0].tolist() == ([0, 0] if empty else [2, 1]) + [0, 0]
        manager.refresh_layer(1, ids)
        manager.refresh_layer(1, ids)
        assert len(payloads) == 2
        # A decode-only pass (idle DP ranks included) neither refreshes nor counts.
        manager._forward_batch.global_forward_mode = SimpleNamespace(
            is_decode=lambda: True
        )
        manager._forward_batch.is_extend_in_batch = False
        before = dict(manager._refresh_gate._batches_since_refresh)
        manager.refresh_layer(1, ids)
        manager._forward_batch.global_forward_mode = None
        manager._forward_batch.forward_mode = SimpleNamespace(
            is_decode=lambda: False, is_idle=lambda: True
        )
        manager.refresh_layer(1, ids)
        assert manager._refresh_gate._batches_since_refresh == before
        assert len(payloads) == 2
    assert events == ["solve", "move", "publish"] * 2


def refresh_batch(non_padded=None, dp_counts=None, decode=False):
    mode = SimpleNamespace(is_decode=lambda: decode, is_idle=lambda: False)
    return SimpleNamespace(
        global_forward_mode=mode,
        forward_mode=mode,
        is_extend_in_batch=not decode,
        original_global_num_tokens_cpu=dp_counts,
        global_num_tokens_cpu=None,
        global_num_token_non_padded_cpu=non_padded,
    )


def test_refresh_decision_bootstraps_then_waits_for_representative_batches():
    from sglang.srt.eplb.layer_refresh import LayerPlacementRefresher

    manager = LayerPlacementRefresher(
        moe_load_balancer=None,
        get_model=None,
        get_expert_location_updater=None,
        refresh_interval=2,
    )

    def decide(batch):
        with manager.forward_scope(batch):
            return manager._should_refresh_layer(0)

    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        assert not decide(None)  # ModelRunner published no batch
        assert manager._forward_batch is None  # the scope always clears it
        assert not decide(refresh_batch(1024, decode=True))  # decode never counts
        assert manager._refresh_gate._batches_since_refresh == {}
        assert decide(refresh_batch(100))  # the first call always bootstraps
        assert decide(refresh_batch(1024))  # a thin bootstrap is due at once
        assert not decide(refresh_batch(1024))  # 1 of 2 batches: not due
        assert not decide(refresh_batch(100))  # due, but under 512 tokens
        assert decide(refresh_batch(dp_counts=[256, 256]))  # DP counts sum ranks
    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        assert not decide(refresh_batch(1024))


def test_rank_counts_retain_source_rank_and_sum_recorded_steps():
    from sglang.srt.eplb.expert_distribution import _StatAccumulator

    physical = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=torch.int32)
    expected = torch.tensor([[16, 20]], dtype=torch.int64)

    class Group:
        world_size = 2

        def all_gather(self, value, dim):
            torch.testing.assert_close(value, expected)
            return torch.cat((value, value * 2), dim=dim)

    accumulator = object.__new__(_StatAccumulator)
    accumulator._global_physical_count_of_buffered_step = SimpleNamespace(
        get_all=lambda: physical
    )
    accumulator._expert_location_metadata = SimpleNamespace(
        num_layers=1,
        num_logical_experts=2,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
    )
    accumulator._first_dump = False
    accumulator._rank = 0
    accumulator._get_global_average_utilization_rate = lambda: None
    with (
        get_context().parallel.override(moe_ep_group=Group()),
        patch(
            "sglang.srt.eplb.expert_distribution._keeps_per_source_rank_statistics",
            return_value=True,
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("rank counts must not be reduced"),
        ),
    ):
        output = accumulator.dump("object")
    torch.testing.assert_close(
        output["logical_count"], torch.stack((expected, expected * 2))
    )
    assert output["logical_count_layout"] == "rank_layer_expert"


def test_source_counts_are_gathered_before_dispatch_even_on_deepep_auto():
    from evenmoe import MoELoadBalancer

    from sglang.srt.eplb.expert_distribution import _SinglePassGatherer

    mlb = MoELoadBalancer.from_algorithm(
        "ultraep", ep_size=2, source_rank=0, experts_per_rank=2
    )
    with (
        get_context().override_server_args(
            moe_load_balancer_algorithm="ultraep",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
        ),
        get_context().resources.override(moe_load_balancer=mlb),
        patch(
            "sglang.srt.eplb.expert_distribution._SelectExpertsSinglePassGatherer"
        ) as select,
    ):
        result = _SinglePassGatherer.init_new(object(), rank=0)
    assert result is select.return_value


@pytest.mark.parametrize(
    "algorithm, per_rank", [("ultraep", True), ("lplb", False), (None, False)]
)
def test_recorder_keeps_source_ranks_when_the_placement_policy_asks(
    algorithm, per_rank
):
    from evenmoe import MoELoadBalancer

    from sglang.srt.eplb.expert_distribution import _keeps_per_source_rank_statistics

    mlb = (
        None
        if algorithm is None
        else MoELoadBalancer.from_algorithm(
            algorithm, ep_size=2, source_rank=0, experts_per_rank=2
        )
    )
    with get_context().resources.override(moe_load_balancer=mlb):
        assert _keeps_per_source_rank_statistics() is per_rank


@pytest.mark.parametrize(
    "data, message",
    [
        ('{"physical_to_logical_map": [[0, 1]]}', "master experts fixed"),
        ('{"logical_count": [[[1, 2]], [[3, 4]]]}', "per source EP rank"),
    ],
)
def test_offline_placement_rejects_missing_rank_statistics(data, message):
    from evenmoe import MoELoadBalancer

    from sglang.srt.eplb.expert_location import compute_initial_expert_location_metadata

    mlb = MoELoadBalancer.from_algorithm(
        "ultraep", ep_size=2, source_rank=0, experts_per_rank=2
    )
    with get_context().override_server_args(init_expert_location=data):
        with pytest.raises(ValueError, match=message):
            compute_initial_expert_location_metadata(
                object(), moe_ep_rank=0, moe_load_balancer=mlb
            )


@pytest.mark.parametrize("algorithm", ["ultraep", "lplb"])
def test_initial_layout_comes_from_the_placement_policy(algorithm):
    """UltraEP supplies its layout (each rank's masters, then its redundant slots
    filled round-robin with them); a policy without one keeps SGLang's trivial."""
    from evenmoe import MoELoadBalancer

    from sglang.srt.eplb.expert_location import compute_initial_expert_location_metadata

    mlb = MoELoadBalancer.from_algorithm(
        algorithm, ep_size=2, source_rank=0, experts_per_rank=3
    )
    common = dict(
        ep_size=2,
        model_config_for_expert_location=SimpleNamespace(
            num_layers=1, num_logical_experts=4
        ),
    )
    moe = SimpleNamespace(ep_num_redundant_experts=2, ep_join_mode=None)
    context = SimpleNamespace(
        resources=SimpleNamespace(
            mlb_model_info={"num_logical_experts": 4, "num_groups": None}
        ),
        parallel=SimpleNamespace(ep_size=2, nnodes=1, elastic_ep_initial_size=None),
        config_bag=lambda name: SimpleNamespace(moe=moe),
    )
    with (
        get_context().override_server_args(
            init_expert_location="trivial", device="cpu"
        ),
        patch.object(ExpertLocationMetadata, "_init_common", return_value=common),
        patch("sglang.srt.runtime_context.get_context", return_value=context),
        patch.object(ExpertLocationMetadata, "_init_raw") as initialize,
        patch.object(ExpertLocationMetadata, "init_trivial") as trivial,
    ):
        compute_initial_expert_location_metadata(object(), 0, mlb)
    if algorithm != "ultraep":
        initialize.assert_not_called()
        trivial.assert_called_once()
        return
    trivial.assert_not_called()
    kwargs = initialize.call_args.kwargs
    # Each rank's masters, then its redundant slots filled with its own masters:
    # the layout SGLang hand-built before EvenMoE owned it.
    assert kwargs["physical_to_logical_map"].tolist() == [[0, 1, 0, 2, 3, 2]]
    # Every replica is listed, master first, as a later plan publishes them.
    assert kwargs["logical_to_all_physical_map"][0, :, 0].tolist() == [0, 1, 3, 4]
    # Any routing state the policy publishes up front is committed on device.
    for state in kwargs["mlb_routing_metadata"].values():
        assert all(value.device.type == "cpu" for value in state.values())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pp_size": 2}, "PP=1"),
        ({"enable_eplb": False}, "recorded placement"),
        ({"disable_cuda_graph": False}, "disable-cuda-graph"),
        ({"enable_two_batch_overlap": True}, "microbatches"),
        ({"expert_distribution_recorder_mode": "per_token"}, "stat expert"),
    ],
)
def test_unsupported_coupled_configuration_fails_early(overrides, message):
    from sglang.srt.arg_groups.moe_hook import handle_moe_load_balancer

    values = dict(
        moe_load_balancer_algorithm="ultraep",
        ep_dispatch_algorithm=None,
        ep_num_redundant_experts=2,
        ep_size=2,
        elastic_ep_backend=None,
        pp_size=1,
        enable_eplb=True,
        init_expert_location="trivial",
        disable_cuda_graph=True,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        expert_distribution_recorder_mode=None,
    )
    values.update(overrides)
    with patch(
        "sglang.srt.arg_groups.moe_hook.resolving_view",
        return_value=SimpleNamespace(**values),
    ):
        with pytest.raises(ValueError, match=message):
            handle_moe_load_balancer(object())


@pytest.mark.parametrize("per_layer", [False, True])
def test_per_layer_refresh_replaces_the_periodic_rebalance(per_layer):
    from sglang.srt.eplb.layer_refresh import LayerPlacementRefresher
    from sglang.srt.model_executor import model_runner as model_runner_module

    runner = object.__new__(model_runner_module.ModelRunner)
    runner.is_draft_worker = False
    runner.model_config = object()
    runner.moe_load_balancer = SimpleNamespace(
        placement_policy="ultraep" if per_layer else None
    )
    context = get_context()
    with (
        context.override_server_args(enable_eplb=True),
        context.resources.override(mlb_placement_refresh=None),
        patch.object(model_runner_module, "EPLBManager") as manager_cls,
    ):
        runner.maybe_init_eplb_manager()
        refresh = context.resources.mlb_placement_refresh
    if per_layer:
        # No periodic rebalance runs, so no EPLBManager starts the recorder.
        manager_cls.assert_not_called()
        assert runner.eplb_manager is None
        assert isinstance(runner.layer_placement_refresher, LayerPlacementRefresher)
        assert refresh == runner.layer_placement_refresher.refresh_layer
    else:
        assert runner.eplb_manager is manager_cls.return_value
        assert runner.layer_placement_refresher is None
        assert refresh is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
