"""Real shared MLP and model composition on one GPU, without native EP."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.shared_compute import make_shared_mlp, shared_reference
from nccl_ep_test.triton_compute import configure_compute

from sglang.srt.distributed import parallel_state
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def compute_context(monkeypatch):
    # No collective is permitted here: TP=1 shared MLP needs only the allocation
    # context's group identity. Matrix multiplication and checkpoint loading are real.
    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(world_size=1))
    with get_context().preserve_config():
        configure_compute()
        yield


def test_shared_fixture_does_not_require_global_tp_initialization(monkeypatch):
    monkeypatch.setattr(parallel_state, "_TP", None)
    monkeypatch.setattr(
        "sglang.srt.layers.linear.get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    mlp = make_shared_mlp()
    x = torch.zeros(1, 2048, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(mlp(x), x, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("hidden,intermediate", [(256, 128), (2048, 1408)])
def test_shared_weights_load_and_graph_matches_cpu(count, fp8, hidden, intermediate):
    mlp = make_shared_mlp(
        shared_experts=count, hidden=hidden, intermediate=intermediate, fp8=fp8
    )
    assert mlp.tp_size == 1
    assert not mlp.down_proj.reduce_results
    static_x = torch.randn(8, hidden, device="cuda", dtype=torch.bfloat16) * 0.25
    original_x = static_x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mlp(static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = mlp(static_x)
    for scale in (1, 0, 0.5):
        static_x.copy_(original_x * scale)
        eager = mlp(static_x)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        torch.testing.assert_close(
            captured.cpu().float(),
            shared_reference(mlp, static_x),
            rtol=0.02,
            atol=0.02,
        )
    empty = static_x[:0]
    assert mlp(empty).shape == empty.shape


@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("fused_scaling", [False, True])
def test_model_keeps_shared_mlp_serial_and_scales_only_routed_output(
    scale, fused_scaling
):
    mlp = make_shared_mlp(hidden=256, fp8=False)
    model = DeepseekV2MoE.__new__(DeepseekV2MoE)
    torch.nn.Module.__init__(model)
    model.shared_experts = mlp
    model._nccl_ep_serial_shared_experts = True
    model.alt_stream = torch.cuda.Stream()
    model._fuse_shared_experts_inside_sbo = False
    model.is_nextn = True
    model.num_fused_shared_experts = 0
    model.layer_id = 0
    model.routed_scaling_factor = scale
    model.gate = lambda x, **kwargs: torch.zeros(len(x), 2, device=x.device)
    model.topk = lambda x, logits, **kwargs: StandardTopKOutput(None, None, None)
    calls = []

    class Experts:
        should_fuse_routed_scaling_factor_in_topk = fused_scaling

        def __call__(self, hidden_states, topk_output):
            calls.append("experts")
            return hidden_states * (scale if fused_scaling else 1)

    model.experts = Experts()
    streams = []
    handle = mlp.register_forward_pre_hook(
        lambda module, args: streams.append(torch.cuda.current_stream().cuda_stream)
    )
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16) * 0.25
    batch = SimpleNamespace(num_token_non_padded=torch.tensor([4], device="cuda"))
    try:
        actual = model.forward_deepep(x, batch)
        routed = (x * scale).float() if fused_scaling else x.float() * scale
        expected = mlp(x).float() + routed
        torch.testing.assert_close(
            actual.float(), expected.bfloat16().float(), rtol=0, atol=0
        )
        assert calls == ["experts"]
        assert all(
            stream == torch.cuda.current_stream().cuda_stream for stream in streams
        )
        assert model.alt_stream.cuda_stream not in streams
        model.topk = SimpleNamespace(
            empty_topk_output=lambda *args, **kwargs: StandardTopKOutput(
                None, None, None
            )
        )
        before = len(streams)
        empty = model.forward_deepep(x[:0], batch)
        assert len(empty) == 0 and len(streams) == before
        assert calls == ["experts", "experts"], "Idle rank must still enter EP experts"
    finally:
        handle.remove()


def test_shared_and_routed_compute_through_dispatcher_and_graph():
    from dataclasses import replace

    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.oracle import RoutingBatch
    from nccl_ep_test.pair_followups import expected_output
    from nccl_ep_test.runner_inputs import SyntheticDecodeRunner, input_batch
    from nccl_ep_test.triton_compute import make_compute_fixture

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.moe.moe_runner.nccl_ep_triton import run_nccl_ep_triton
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    with dispatcher_environment(capacity=8) as environment:
        configure_compute(graph_enabled=True)
        dispatcher = environment.dispatcher(layer_id=0)
        _, quant, config = make_compute_fixture(hidden=2048, experts=2)
        config = replace(config, num_experts=2)
        model = DeepseekV2MoE.__new__(DeepseekV2MoE)
        torch.nn.Module.__init__(model)
        model.shared_experts = make_shared_mlp()
        model._nccl_ep_serial_shared_experts = True
        model.alt_stream = torch.cuda.Stream()
        model._fuse_shared_experts_inside_sbo = False
        model.is_nextn = True
        model.num_fused_shared_experts = 0
        model.layer_id = 0
        model.routed_scaling_factor = 2.5
        model.gate = lambda x, **kwargs: torch.zeros(len(x), 2, device=x.device)

        class Router:
            def __init__(self):
                self.ids = torch.tensor([0, 1], device="cuda")
                self.weights = torch.tensor([0.25, 0.75], device="cuda")

            def __call__(self, x, logits, *, num_token_non_padded, **kwargs):
                ids = self.ids.expand(len(x), 2)
                ids = torch.where(
                    torch.arange(len(x), device=x.device)[:, None]
                    < num_token_non_padded,
                    ids,
                    -1,
                )
                weights = self.weights.expand(len(x), 2)
                return StandardTopKOutput(weights, ids, None)

        class Experts:
            should_fuse_routed_scaling_factor_in_topk = False

            def __call__(self, hidden_states, topk_output):
                dispatched = dispatcher.dispatch(hidden_states, topk_output)
                return dispatcher.combine(run_nccl_ep_triton(dispatched, quant, config))

        model.topk, model.experts = Router(), Experts()

        def forward(batch):
            x = batch.input_ids[:, None].expand(-1, 2048).bfloat16() / 8
            return LogitsProcessorOutput(
                next_token_logits=model.forward_deepep(x, batch)
            )

        runner = SyntheticDecodeRunner(
            forward,
            environment.coordinator,
            buckets=(1, 8),
            backend_factory=lambda owner: FullCudaGraphBackend(
                owner, nccl_ep_capacity=8
            ),
        )
        try:
            for values in ([1, 2, 4], [], [4] * 8, [], [2]):
                incoming = input_batch(values)
                incoming.forward_mode = (
                    ForwardMode.DECODE if values else ForwardMode.IDLE
                )
                result = runner.execute(incoming).next_token_logits
                assert result.shape == (len(values), 2048)
                if values:
                    x = incoming.input_ids[:, None].expand(-1, 2048).bfloat16() / 8
                    topk = model.topk(
                        x, None, num_token_non_padded=incoming.num_token_non_padded
                    )
                    fixture = RoutingBatch(
                        (x.cpu(),),
                        (topk.topk_ids.cpu(),),
                        (topk.topk_weights.cpu(),),
                        2,
                    )
                    wanted = expected_output(
                        fixture,
                        0,
                        quant,
                        model.shared_experts,
                        2.5,
                        compute_backend="triton",
                    )
                    torch.testing.assert_close(
                        result.cpu().float(), wanted, rtol=0.02, atol=0.02
                    )
            assert environment.events.count("handle_create") == 1
        finally:
            runner.backend.cleanup()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
