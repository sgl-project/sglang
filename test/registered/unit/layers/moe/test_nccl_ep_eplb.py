"""Fixed-topology EPLB with real relocation/Triton/Graph; native EP is replaced."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.eplb import metadata, tensor_addresses
from registered.unit.layers.moe.test_nccl_ep_graph_config import (  # noqa: F401
    ep_bindings,
    model_path,
    server_args,
)

from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import get_flags, get_resources
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("graph", [False, True])
def test_public_fixed_topology_eplb(model_path, monkeypatch, graph):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    args = server_args(
        model_path,
        tp_size=2,
        enable_eplb=True,
        ep_num_redundant_experts=2,
        enable_nccl_ep_cuda_graph=graph,
    )
    assert args.ep_size == 2
    assert args.ep_dispatch_algorithm == "static"
    assert args.expert_distribution_recorder_mode == "stat"
    assert args.enable_nccl_ep_cuda_graph == graph


@pytest.mark.parametrize(
    "change",
    [
        {"elastic_ep_initial_size": 2},
        {"enable_elastic_expert_backup": True},
    ],
)
def test_elastic_eplb_rejected_before_dispatch(model_path, monkeypatch, change):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    with pytest.raises(ValueError, match="NCCL EP EPLB.*fixed"):
        server_args(model_path, tp_size=2, enable_eplb=True, **change)


@pytest.mark.parametrize("change", ["ranks", "experts", "layers", "mapping"])
def test_topology_change_rejected_before_any_weight_write(monkeypatch, change):
    old = metadata([[0, 1]])
    new = metadata([[1, 0]])
    if change == "ranks":
        new.ep_size = 2
    elif change == "experts":
        new = metadata([[0, 1, 0]])
    elif change == "layers":
        new = metadata([[0, 1], [1, 0]])
    else:
        new.logical_to_rank_dispatch_physical_map = None
    weights = {0: [torch.tensor([[1.0], [2.0]], device="cuda")]}
    before = weights[0][0].clone()
    monkeypatch.setattr(get_resources(), "expert_location_metadata", old)
    with get_flags().moe.override(a2a_backend=MoeA2ABackend.NCCL_EP):
        with pytest.raises(ValueError, match="NCCL EP EPLB"):
            ExpertLocationUpdater().update(weights, new, [0], nnodes=1, rank=0)
    torch.testing.assert_close(weights[0][0], before, rtol=0, atol=0)
    torch.testing.assert_close(
        old.physical_to_logical_map, metadata([[0, 1]]).physical_to_logical_map
    )


def test_relocation_rejects_incomplete_or_overlapping_graph_submission(monkeypatch):
    from sglang.srt.eplb.nccl_ep import expert_update_session
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        NcclEpGraphResources,
    )

    owner = NcclEpGraphResources(8, shutdown=lambda: None)
    monkeypatch.setitem(get_resources().buffers, "nccl_ep_graph_resources", owner)
    old, new = metadata([[0, 1]]), metadata([[1, 0]])
    owner.borrower = object()
    with pytest.raises(RuntimeError, match="completed EP transaction"):
        with expert_update_session(old, new):
            pytest.fail("Updated an incomplete transaction")
    owner.borrower = None
    # An uninitialized owner is removed when its last session exits.
    monkeypatch.setitem(get_resources().buffers, "nccl_ep_graph_resources", owner)
    with owner.submission_session("replay"):
        with pytest.raises(RuntimeError, match="Overlapping"):
            with expert_update_session(old, new):
                pytest.fail("Updated during replay submission")


@pytest.mark.parametrize("mode", ["stat", "stat_approx"])
def test_received_counts_include_idle_sender_and_replay(monkeypatch, mode):
    from sglang.srt.eplb.expert_distribution import (
        _DeepepLowLatencySinglePassGatherer,
        _SinglePassGatherer,
    )

    meta = metadata([[0, 1, 2, 3, 0, 1]], ep_size=2)
    gatherer = _SinglePassGatherer.init_new(
        SimpleNamespace(
            moe_a2a_backend="nccl_ep", expert_distribution_recorder_mode=mode
        ),
        meta,
        rank=0,
    )
    assert isinstance(gatherer, _DeepepLowLatencySinglePassGatherer)
    counts = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    # No local TopK call: a rank can receive work while it has no input tokens.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gatherer.on_deepep_dispatch_low_latency(0, counts)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gatherer.on_deepep_dispatch_low_latency(0, counts)
    for values in ([0, 3, 7], [9, 0, 1], [0, 0, 0]):
        gatherer.reset()
        counts.copy_(torch.tensor(values, device="cuda"))
        graph.replay()
        actual = gatherer.collect()["global_physical_count"]
        torch.testing.assert_close(
            actual.cpu(), torch.tensor([values + [0, 0, 0]], dtype=actual.dtype)
        )


def test_relocation_preserves_graph_weights_routing_and_handle(monkeypatch):
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.sglang_graph import backend_for
    from nccl_ep_test.triton_compute import configure_compute, make_compute_fixture

    from sglang.srt.eplb.expert_distribution import ExpertDistributionRecorder
    from sglang.srt.eplb.expert_location_dispatch import (
        ExpertLocationDispatchInfo,
        topk_ids_logical_to_physical,
    )
    from sglang.srt.layers.moe.moe_runner.nccl_ep_triton import run_nccl_ep_triton
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 1)
    with dispatcher_environment(capacity=8) as environment:
        configure_compute(graph_enabled=True)
        old = metadata([[0, 1], [0, 1]])
        monkeypatch.setattr(get_resources(), "expert_location_metadata", old)
        recorder = ExpertDistributionRecorder.init_new(
            SimpleNamespace(
                moe_a2a_backend="nccl_ep",
                expert_distribution_recorder_mode="stat",
                expert_distribution_recorder_buffer_size=8,
                enable_expert_distribution_metrics=False,
                device="cuda",
            ),
            old,
            rank=0,
        )
        recorder.start_record()
        monkeypatch.setattr(get_resources(), "expert_distribution_recorder", recorder)
        with get_flags().moe.override(a2a_backend=MoeA2ABackend.NCCL_EP):
            dispatchers = [environment.dispatcher(layer_id=layer) for layer in range(2)]
            quants, configs, weights, infos = [], [], {}, []
            for layer in range(2):
                _, quant, config = make_compute_fixture(hidden=2048, experts=2)
                quants.append(quant)
                configs.append(replace(config, num_experts=2))
                weights[layer] = [
                    quant.w13_weight,
                    quant.w2_weight,
                    quant.w13_scale,
                    quant.w2_scale,
                ]
                infos.append(
                    ExpertLocationDispatchInfo(
                        "static",
                        old.logical_to_rank_dispatch_physical_map[layer],
                        old.logical_to_all_physical_map[layer],
                        old.logical_to_all_physical_map_num_valid[layer],
                        2,
                    )
                )
            addresses = tensor_addresses(old, weights)
            x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16) * 0.25
            logical_ids = torch.tensor([[0, 1]] * 8, device="cuda")
            factors = torch.tensor([[0.25, 0.75]] * 8, device="cuda")

            def forward(bucket):
                outputs = []
                for layer, dispatcher in enumerate(dispatchers):
                    ids = topk_ids_logical_to_physical(
                        logical_ids[:bucket], infos[layer]
                    )
                    with recorder.with_current_layer(layer):
                        dispatched = dispatcher.dispatch(
                            x[:bucket], StandardTopKOutput(factors[:bucket], ids, None)
                        )
                    outputs.append(
                        dispatcher.combine(
                            run_nccl_ep_triton(
                                dispatched, quants[layer], configs[layer]
                            )
                        )
                    )
                return outputs

            backend = backend_for(environment.coordinator, 8)
            capture_stream, update_stream = torch.cuda.Stream(), torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            updater = ExpertLocationUpdater()
            try:
                with torch.cuda.stream(capture_stream), backend.capture_session(
                    capture_stream
                ):
                    for bucket in (8, 4):
                        backend.capture_one(
                            ShapeKey(bucket), lambda bucket=bucket: forward(bucket)
                        )
                torch.cuda.current_stream().wait_stream(capture_stream)
                with backend.replay_session():
                    baseline = [
                        value.clone() for value in backend.replay(ShapeKey(8), None)
                    ]
                handle_count = environment.events.count("handle_create")
                updates = len(environment.updates)
                for layout in ([[1, 0], [1, 0]], [[0, 1], [0, 1]], [[1, 0], [0, 1]]):
                    new = metadata(layout)
                    update_stream.wait_stream(torch.cuda.current_stream())
                    # Rebalance one layer at a time, as EPLB chunking does.
                    for layer in range(2):
                        with torch.cuda.stream(update_stream):
                            assert (
                                updater.update(weights, new, [layer], nnodes=1, rank=0)
                                == {}
                            )
                        for bucket in (4, 8):
                            with backend.replay_session():
                                for (
                                    gatherer
                                ) in recorder._single_pass_gatherers.values():
                                    gatherer.reset()
                                actual = backend.replay(ShapeKey(bucket), None)
                                for output, expected in zip(actual, baseline):
                                    torch.testing.assert_close(
                                        output, expected[:bucket], rtol=0.02, atol=0.02
                                    )
                                for (
                                    gatherer
                                ) in recorder._single_pass_gatherers.values():
                                    counts = gatherer.collect()["global_physical_count"]
                                    torch.testing.assert_close(
                                        counts, torch.full_like(counts, bucket)
                                    )
                    assert tensor_addresses(old, weights) == addresses
                assert environment.events.count("handle_create") == handle_count == 1
                assert len(environment.updates) == updates
            finally:
                backend.cleanup()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
