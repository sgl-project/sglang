"""NCCL EP SBO model hooks and real compute/Graph, without native EP."""

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
from nccl_ep_test.triton_compute import configure_compute, make_compute_fixture
from registered.unit.layers.moe.test_nccl_ep_graph_config import (  # noqa: F401
    ep_bindings,
    model_path,
    server_args,
)

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_flags, get_resources
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("runner", ["triton", "deep_gemm"])
def test_public_sbo_configuration(model_path, monkeypatch, graph, runner):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    args = server_args(
        model_path,
        tp_size=2,
        enable_single_batch_overlap=True,
        moe_runner_backend=runner,
        enable_nccl_ep_cuda_graph=graph,
    )
    assert args.enable_single_batch_overlap
    assert args.moe_a2a_backend == "nccl_ep"


@pytest.mark.parametrize("blackwell", [False, True])
@pytest.mark.parametrize(
    "runner", [MoeRunnerBackend.TRITON, MoeRunnerBackend.DEEP_GEMM]
)
def test_nccl_sbo_does_not_select_deepep_combine_signals(
    monkeypatch, blackwell, runner
):
    monkeypatch.setattr(
        "sglang.srt.batch_overlap.single_batch_overlap.is_blackwell", lambda: blackwell
    )
    with get_flags().moe.override(
        a2a_backend=MoeA2ABackend.NCCL_EP, runner_backend=runner, sbo_enabled=True
    ):
        assert SboFlags.enable_dispatch_shared_one_stream_overlap()
        assert not SboFlags.enable_combine_shared_two_stream_overlap()
        assert not SboFlags.enable_combine_down_gemm_two_stream_overlap()


@pytest.mark.parametrize("blackwell", [False, True])
@pytest.mark.parametrize(
    "runner", [MoeRunnerBackend.TRITON, MoeRunnerBackend.DEEP_GEMM]
)
def test_deepep_sbo_policy_is_preserved(monkeypatch, blackwell, runner):
    from sglang.srt.environ import envs

    monkeypatch.setattr(
        "sglang.srt.batch_overlap.single_batch_overlap.is_blackwell", lambda: blackwell
    )
    with (
        envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.override(False),
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.DEEPEP, runner_backend=runner, sbo_enabled=True
        ),
    ):
        assert SboFlags.enable_dispatch_shared_one_stream_overlap() == (not blackwell)
        assert SboFlags.enable_combine_shared_two_stream_overlap() == blackwell
        assert SboFlags.enable_combine_down_gemm_two_stream_overlap() == (
            runner.is_deep_gemm() and not blackwell
        )


def _model(dispatcher, fused_scaling):
    _, quant, config = make_compute_fixture(hidden=2048, experts=2)
    ids = torch.tensor([[0, 1]] * 8, device="cuda")
    factors = torch.tensor([[0.25, 0.75]] * 8, device="cuda")

    class Router:
        def __call__(self, x, logits, *, num_token_non_padded, **kwargs):
            from sglang.srt.eplb.expert_location_dispatch import (
                topk_ids_logical_to_physical,
            )

            physical = topk_ids_logical_to_physical(
                ids[: len(x)], kwargs.get("expert_location_dispatch_info")
            )
            routing = torch.where(
                torch.arange(len(x), device="cuda")[:, None] < num_token_non_padded,
                physical,
                -1,
            )
            weights = factors[: len(x)] * (2.5 if fused_scaling else 1)
            return StandardTopKOutput(weights, routing, None)

        def empty_topk_output(self, device, **kwargs):
            return StandardTopKOutput(factors[:0], ids[:0], None)

    model = make_moe(
        dispatcher,
        quant,
        replace(config, num_experts=2, layer_id=0),
        make_shared_mlp(hidden=2048),
        Router(),
        fused_scaling=fused_scaling,
    )
    return model, ids, factors


@pytest.mark.parametrize("fused_scaling", [False, True])
@pytest.mark.parametrize("rebalance", [False, True])
def test_sbo_hooks_graph_and_eager_match_serial_shared_experts(
    monkeypatch, fused_scaling, rebalance
):
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        nccl_ep_eager_session,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    with dispatcher_environment(capacity=8) as environment:
        monkeypatch.setattr(
            "sglang.srt.layers.linear.get_tp_group", lambda: environment.coordinator
        )
        configure_compute(
            graph_enabled=True, dispatch_algorithm="static" if rebalance else None
        )
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 1)
        monkeypatch.setattr(
            get_resources(), "expert_location_metadata", metadata([[0, 1]])
        )
        with get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NCCL_EP,
            runner_backend=MoeRunnerBackend.TRITON,
            sbo_enabled=True,
        ):
            dispatcher = environment.dispatcher(layer_id=0)
            model, ids, factors = _model(dispatcher, fused_scaling)
            observed_streams = []

            def shared_hook(module, args):
                observed_streams.append(torch.cuda.current_stream().cuda_stream)
                environment.events.append("shared")
                if model._fuse_shared_experts_inside_sbo:
                    assert dispatcher.handle.pending == "dispatch"

            hook = model.shared_experts.register_forward_pre_hook(shared_hook)
            x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16) * 0.25
            count = torch.tensor(8, device="cuda")
            batch = SimpleNamespace(num_token_non_padded=count)
            forward = lambda rows: model.forward_deepep(x[:rows], batch)
            baseline = forward(8).clone()
            model._fuse_shared_experts_inside_sbo = True
            start = len(environment.events)
            torch.testing.assert_close(forward(8), baseline, rtol=0, atol=0)
            operations = [
                event
                for event in environment.events[start:]
                if event in ("dispatch", "shared", "complete", "combine")
            ]
            assert operations == [
                "dispatch",
                "shared",
                "complete",
                "combine",
                "complete",
            ]
            assert all(
                stream != model.alt_stream.cuda_stream for stream in observed_streams
            )
            backend = backend_for(environment.coordinator, 8)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(stream), backend.capture_session(stream):
                    for bucket in (8, 4):
                        backend.capture_one(
                            ShapeKey(bucket), lambda bucket=bucket: forward(bucket)
                        )
                torch.cuda.current_stream().wait_stream(stream)
                updater = ExpertLocationUpdater()
                weights = {
                    0: [
                        getattr(model.experts.quant, name)
                        for name in ("w13_weight", "w2_weight", "w13_scale", "w2_scale")
                    ]
                }
                for bucket, active, reverse in (
                    (8, 8, False),
                    (4, 1, True),
                    (8, 0, False),
                    (4, 4, True),
                ):
                    if rebalance:
                        updater.update(
                            weights,
                            metadata([[1, 0] if reverse else [0, 1]]),
                            [0],
                            nnodes=1,
                            rank=0,
                        )
                    x.mul_(0.5)
                    ids.copy_(
                        torch.tensor([[1, 0] if reverse else [0, 1]] * 8, device="cuda")
                    )
                    factors.copy_(
                        torch.tensor(
                            [[0.75, 0.25] if reverse else [0.25, 0.75]] * 8,
                            device="cuda",
                        )
                    )
                    count.fill_(active)
                    with nccl_ep_eager_session():
                        model._fuse_shared_experts_inside_sbo = False
                        expected = forward(bucket).clone()
                        model._fuse_shared_experts_inside_sbo = True
                        torch.testing.assert_close(
                            forward(bucket), expected, rtol=0, atol=0
                        )
                    event_count = len(environment.events)
                    with backend.replay_session():
                        actual = backend.replay(ShapeKey(bucket), None)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert (
                        len(environment.events) == event_count
                    ), "Replay must bypass Python hooks"
                    assert len(dispatcher._dispatch_hooks.hook_dict) == 0
                    with nccl_ep_eager_session():
                        empty = forward(0)
                        assert empty.shape == (0, 2048)
                    assert dispatcher._stage.name == "INITIAL"
            finally:
                hook.remove()
                backend.cleanup()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
