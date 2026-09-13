"""Interleaved NCCL EP ownership and real staged MoE compute on one GPU."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.eplb import metadata
from nccl_ep_test.fake_ep import dispatcher_environment
from nccl_ep_test.moe_model import make_moe
from nccl_ep_test.sglang_graph import backend_for
from nccl_ep_test.shared_compute import make_shared_mlp
from nccl_ep_test.tbo_model import forward_tbo
from nccl_ep_test.triton_compute import configure_compute, make_compute_fixture

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_flags, get_parallel, get_resources
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_models(environment, monkeypatch, fused_scaling):
    from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher

    monkeypatch.setattr(
        "sglang.srt.layers.moe.fused_moe_triton.layer.get_tp_group",
        lambda: environment.coordinator,
    )
    monkeypatch.setattr(
        "sglang.srt.layers.linear.get_tp_group", lambda: environment.coordinator
    )
    ids = torch.tensor([[0, 1]] * 8, device="cuda")
    factors = torch.tensor([[0.25, 0.75]] * 8, device="cuda")

    class Router:
        def __call__(
            self, hidden_states, router_logits, *, num_token_non_padded, **kwargs
        ):
            from sglang.srt.eplb.expert_location_dispatch import (
                topk_ids_logical_to_physical,
            )

            n = len(hidden_states)
            physical = topk_ids_logical_to_physical(
                ids[:n], kwargs.get("expert_location_dispatch_info")
            )
            routes = torch.where(
                torch.arange(n, device="cuda")[:, None] < num_token_non_padded,
                physical,
                -1,
            )
            return StandardTopKOutput(
                factors[:n] * (2.5 if fused_scaling else 1), routes, None
            )

        def empty_topk_output(self, device, **kwargs):
            return StandardTopKOutput(factors[:0], ids[:0], None)

    models = []
    for layer in range(2):
        _, quant, config = make_compute_fixture(hidden=2048, experts=2)
        config = replace(config, num_experts=2, layer_id=layer)
        dispatcher = create_moe_dispatcher(config)
        model = make_moe(
            dispatcher,
            quant,
            config,
            make_shared_mlp(hidden=2048),
            Router(),
            fused_scaling=fused_scaling,
        )
        # The transport double has one rank; exercise the model's EP stages.
        model.ep_size = 2
        models.append(model)
    return models, ids, factors


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
@pytest.mark.parametrize("fused_scaling", [False, True])
@pytest.mark.parametrize("composed", [False, True])
def test_staged_tbo_matches_serial_with_two_groups_and_dynamic_graph(
    monkeypatch, mode, fused_scaling, composed
):
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        get_nccl_ep_graph_resources,
        nccl_ep_eager_session,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    with dispatcher_environment(capacity=8) as environment:
        configure_compute(
            graph_enabled=True, dispatch_algorithm="static" if composed else None
        )
        monkeypatch.setattr(
            get_resources(), "expert_location_metadata", metadata([[0, 1]] * 2)
        )
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 1)
        with get_flags().moe.override(tbo_enabled=True, sbo_enabled=composed):
            models, ids, factors = make_models(environment, monkeypatch, fused_scaling)
            for model in models:
                model._fuse_shared_experts_inside_sbo = composed
            updater = ExpertLocationUpdater()
            weight_tensors = {
                layer: [
                    getattr(model.experts.quant, name)
                    for name in ("w13_weight", "w2_weight", "w13_scale", "w2_scale")
                ]
                for layer, model in enumerate(models)
            }
            x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16) * 0.125
            counts = torch.tensor([4, 4], device="cuda")

            def forward(rows, split=None, padded=None):
                split = rows // 2 if split is None else split
                return forward_tbo(
                    models,
                    x[:rows],
                    split=split,
                    padded=padded or (split, rows - split),
                    counts=counts,
                    mode=mode,
                )

            def reference(rows, split, padded):
                # Serial model execution on each independently padded child.
                result = []
                for index, (start, end) in enumerate(((0, split), (split, rows))):
                    child = torch.zeros(
                        padded[index], 2048, device="cuda", dtype=x.dtype
                    )
                    child[: end - start].copy_(x[start:end])
                    for model in models:
                        child = model.forward_deepep(
                            child, SimpleNamespace(num_token_non_padded=counts[index])
                        )
                    result.append(child[: end - start].clone())
                return torch.cat(result)

            expected = reference(8, 4, (4, 4))
            torch.testing.assert_close(forward(8), expected, rtol=0, atol=0)
            assert len(environment.groups) == 2
            left, right = models[0].experts.dispatcher._inners
            assert left.buffer is not right.buffer
            assert models[1].experts.dispatcher._inners[0].buffer is left.buffer
            backend = backend_for(environment.coordinator, 8)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(stream), backend.capture_session(stream):
                    for rows in (8, 4):
                        backend.capture_one(
                            ShapeKey(rows), lambda rows=rows: forward(rows)
                        )
                torch.cuda.current_stream().wait_stream(stream)
                owner = get_nccl_ep_graph_resources()
                assert owner.lane(0).handle is not owner.lane(1).handle
                assert owner.lane(0).state.group is not owner.lane(1).state.group
                assert len(environment.groups) == 4
                updates = len(environment.updates)
                for rows, active in (
                    (8, (4, 0)),
                    (4, (1, 2)),
                    (8, (0, 0)),
                    (4, (2, 1)),
                ):
                    if composed:
                        updater.update(
                            weight_tensors,
                            metadata([[1, 0] if rows == 8 else [0, 1]] * 2),
                            [0, 1],
                            nnodes=1,
                            rank=0,
                        )
                    ids.copy_(ids.flip(1))
                    factors.copy_(factors.flip(1))
                    x.mul_(0.5)
                    counts.copy_(torch.tensor(active, device="cuda"))
                    with nccl_ep_eager_session():
                        expected = reference(rows, rows // 2, (rows // 2, rows // 2))
                    event_count = len(environment.events)
                    with backend.replay_session():
                        torch.testing.assert_close(
                            backend.replay(ShapeKey(rows), None),
                            expected,
                            rtol=0,
                            atol=0,
                        )
                    assert len(environment.events) == event_count
                assert len(environment.updates) == updates
                # Real zero-length and unequal/padded children use the runner's
                # eager ordering session, interleaved between Graph replays.
                for rows, split, padded in (
                    (1, 0, (0, 1)),
                    (5, 1, (2, 4)),
                    (0, 0, (0, 0)),
                ):
                    counts.copy_(torch.tensor([split, rows - split], device="cuda"))
                    with nccl_ep_eager_session():
                        expected = reference(rows, split, padded)
                        torch.testing.assert_close(
                            forward(rows, split, padded), expected, rtol=0, atol=0
                        )
                assert owner.borrower is None
            finally:
                backend.cleanup()


@pytest.mark.parametrize("graph", [False, True])
def test_tbo_second_lane_blocks_cleanup_and_eplb_before_mutation(monkeypatch, graph):

    from sglang.srt.eplb.nccl_ep import expert_update_session
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        NcclEpGraphResources,
    )

    with dispatcher_environment(capacity=8) as environment:
        lane0 = environment.dispatcher(layer_id=0, instance_id=0)
        lane1 = environment.dispatcher(layer_id=0, instance_id=1)
        same_lane = environment.dispatcher(layer_id=1, instance_id=1)
        x = torch.ones(4, 2048, device="cuda", dtype=torch.bfloat16)
        topk = StandardTopKOutput(
            torch.full((4, 2), 0.5, device="cuda"),
            torch.tensor([[0, 1]] * 4, device="cuda"),
            None,
        )
        owner = NcclEpGraphResources(8, shutdown=lambda: None)
        from contextlib import nullcontext

        with owner.capture_session() if graph else nullcontext():
            lane0.dispatch_a(x, topk)
            lane1.dispatch_a(x * 2, topk)
            with pytest.raises(RuntimeError, match="(serial|incomplete)"):
                same_lane.dispatch_a(x, topk)
            for lane in (lane0, lane1):
                out = lane.dispatch_b()
                from sglang.srt.layers.moe.token_dispatcher.deepep import (
                    DeepEPLLCombineInput,
                )

                # Dequantize identity experts, preserving the real output format.
                tokens = (
                    out.hidden_states.float().reshape(2, 8, -1, 128)
                    * out.hidden_states_scale[..., None]
                )
                lane.combine_a(
                    DeepEPLLCombineInput(
                        tokens.reshape(2, 8, 2048).bfloat16(),
                        out.topk_ids,
                        out.topk_weights,
                    )
                )
            lane0.combine_b()
            assert (
                (owner.borrower is lane1) if graph else (lane1.buffer.borrower is lane1)
            )
            with pytest.raises(RuntimeError, match="incomplete"):
                (owner.close if graph else NcclEpBuffer.destroy)()
            if not graph:
                with pytest.raises(RuntimeError, match="completed EP transaction"):
                    with expert_update_session(metadata([[0, 1]]), metadata([[1, 0]])):
                        pytest.fail("Migrated weights with lane 1 in flight")
            lane1.combine_b()
        if graph:
            torch.cuda.synchronize()
            owner.close()


def test_tbo_runner_updates_child_counts_for_idle_and_padded_replays(monkeypatch):
    from nccl_ep_test.runner_inputs import SyntheticDecodeRunner, input_batch

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        nccl_ep_eager_session,
    )
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    with dispatcher_environment(capacity=8) as environment:
        configure_compute(graph_enabled=True)
        monkeypatch.setattr(
            get_resources(), "expert_location_metadata", metadata([[0, 1]] * 2)
        )
        with get_flags().moe.override(tbo_enabled=True), get_parallel().override(
            attn_tp_size=1
        ):
            models, _, _ = make_models(environment, monkeypatch, False)

            def forward(batch):
                x = (
                    batch.input_ids[:, None].to(torch.bfloat16).expand(-1, 2048)
                    * 0.03125
                )
                out = forward_tbo(
                    models, x, mode=ForwardMode.DECODE, children=batch.tbo_children
                )
                return LogitsProcessorOutput(next_token_logits=out[:, :1])

            runner = SyntheticDecodeRunner(
                forward,
                environment.coordinator,
                buckets=(4, 8),
                tbo=True,
                backend_factory=lambda runner: FullCudaGraphBackend(
                    runner, nccl_ep_capacity=8
                ),
            )
            try:
                for tokens, wanted in (
                    ([1, 2, 3, 4, 5], [4, 1]),
                    ([], [0, 0]),
                    ([1, 2, 3], [2, 1]),
                ):
                    batch = input_batch(tokens)
                    batch.tbo_split_seq_index = len(tokens) // 2
                    actual = runner.execute(batch).next_token_logits
                    assert len(actual) == len(tokens)
                    torch.testing.assert_close(
                        runner.tbo_plugin._tbo_children_num_token_non_padded.cpu(),
                        torch.tensor(wanted, dtype=torch.int32),
                    )
                    assert torch.isfinite(actual).all()
                    with nccl_ep_eager_session():
                        expected = (
                            batch.input_ids[:, None].to(torch.bfloat16).expand(-1, 2048)
                            * 0.03125
                        )
                        for model in models:
                            expected = model.forward_deepep(expected, batch)
                    torch.testing.assert_close(
                        actual, expected[:, :1], rtol=0.01, atol=0.01
                    )
            finally:
                runner.backend.cleanup()


@pytest.mark.parametrize("nccl", [False, True])
@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
def test_tbo_global_admission_with_idle_peer_preserves_deepep_ll_gate(nccl, mode):
    from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
    from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend

    with get_flags().moe.override(
        tbo_enabled=True,
        tbo_token_distribution_threshold=0.25,
        a2a_backend=MoeA2ABackend.NCCL_EP if nccl else MoeA2ABackend.DEEPEP,
        deepep_mode=DeepEPMode.LOW_LATENCY,
    ):
        local = SimpleNamespace(
            forward_mode=mode,
            spec_info=None,
            batch_size=lambda: 3,
            extend_num_tokens=3,
            extend_lens=[1, 2],
            is_extend_in_batch=mode.is_extend(),
        )
        active, idle = TboDPAttentionPreparer(), TboDPAttentionPreparer()
        active_info = active.prepare_all_gather(local)
        idle_info = idle.prepare_all_gather(None)
        peers = torch.tensor([active_info, idle_info])
        active_split, active_mode = active.compute_output(peers)
        idle_split, idle_mode = idle.compute_output(peers)
        eligible = nccl or mode.is_decode()
        assert (active_split is not None) == eligible
        assert (idle_split == 0) == eligible
        assert active_mode == idle_mode == (mode if eligible else None)
