"""Dense route cache lifetime across warmup, capture, and replay."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.lora.common.lora_a import grouped_lora_a
from sglang.kernels.ops.lora.moe.lora_b import run_lora_b
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend
from sglang.srt.lora.dense.plan import DensePlan, Overlap
from sglang.srt.lora.moe.plan import (
    BFamily,
    BSpec,
    RouteBuilderFamily,
    RouteRequirement,
    Site,
)
from sglang.srt.lora.moe.routing import build_moe_routes
from sglang.srt.lora.utils import Phase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _bind(runner, slots, *, graph_mode, prefill=False):
    runner.begin_batch(
        token_slots=slots,
        lora_ranks=torch.tensor([16, 8, 0, 4], device="cuda", dtype=torch.int32),
        scalings=torch.tensor([1.0, 0.5, 1.0, 2.0], device="cuda"),
        num_tokens=slots.numel(),
        phase=Phase.PREFILL if prefill else Phase.DECODE,
        graph_mode=graph_mode,
        is_prefill_graph=prefill,
    )


@pytest.mark.parametrize(
    "overlap,moe_first",
    # the two publication directions; NONE has no cross-stream traffic
    [(Overlap.A, False), (Overlap.AB_DELTA, True)],
    ids=["dense-first-a", "moe-first-ab_delta"],
)
def test_dense_moe_token_route_reuse_graph_replay(overlap, moe_first):
    backend = TritonV2LoRABackend(4, torch.device("cuda"))
    runner = backend.runner
    workspace = runner.workspace
    generator = torch.Generator(device="cuda").manual_seed(741)

    def rand(*shape):
        return (torch.randn(shape, device="cuda", generator=generator) * 0.1).bfloat16()

    a, b, base_weight = rand(4, 16, 128), rand(4, 128, 16), rand(128, 128)
    moe_a, moe_b = rand(4, 16, 128), rand(4, 3, 64, 16)
    plan = DensePlan(overlap=overlap)
    moe_plan = SimpleNamespace(
        route_requirements=lambda: {
            RouteRequirement.SHARED_TOKEN_PLAN,
            RouteRequirement.RAW_PER_EXPERT,
        },
        route_builder=RouteBuilderFamily.STANDARD,
    )
    captures = {}
    stream = torch.cuda.Stream()  # one stream for every bucket, as in a graph runner
    for tokens in (64, 8):
        x = rand(tokens, 128)
        slots = torch.arange(tokens, device="cuda", dtype=torch.int32) % 4
        _bind(runner, slots, graph_mode=True, prefill=True)
        runner.lora_ranks.fill_(16)
        runner.scalings.fill_(0.5)
        ranks, scalings = runner.lora_ranks, runner.scalings
        topk = (
            torch.arange(tokens * 2, device="cuda", dtype=torch.int32).view(tokens, 2)
            % 3
        )
        bridge = torch.empty(tokens, 16, device="cuda", dtype=x.dtype)
        delta = torch.empty(tokens * 2, 64, device="cuda", dtype=x.dtype)

        def moe_routes():
            return build_moe_routes(
                moe_plan,
                topk_ids=topk,
                token_lora_mapping=slots,
                num_local_experts=3,
                max_loras=4,
                block_size=16,
                workspace=workspace,
            )

        def forward():
            backend.reset_routing_cache()
            routes = moe_routes() if moe_first else None
            dense = runner.apply(
                x,
                lambda: x @ base_weight.T,
                plan,
                a=a,
                b=b,
                offsets=(0, 128),
            )
            if routes is None:
                routes = moe_routes()
            cached = runner.route(16)
            assert (
                routes.shared_token.sorted_pair_ids.data_ptr()
                == cached.sorted_pair_ids.data_ptr()
            )
            assert routes.shared_token.groups_per_slot == 1
            grouped_lora_a(
                dense, moe_a, bridge, routes.shared_token, config=plan.a_tiles
            )
            run_lora_b(
                BSpec(Site.GATE_UP, BFamily.PER_ROW),
                bridge=bridge,
                weight=moe_b.flatten(0, 1),
                destination=delta,
                routing=routes.raw_per_expert,
                destination_offsets=(0,),
                config=plan.b_tiles,
                pair_bridge=False,
            )
            again = runner.apply(
                x,
                lambda: x @ base_weight.T,
                plan,
                a=a,
                b=b,
                offsets=(0, 128),
            )
            return dense, delta, again, cached

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = forward()
        captures[tokens] = graph, x, slots, topk, ranks, scalings, outputs

    for iteration, tokens in enumerate((8, 64, 8, 64)):
        graph, x, slots, topk, ranks, scalings, (dense, delta, again, route) = captures[
            tokens
        ]
        x.copy_(rand(tokens, 128))
        slots.copy_(
            (torch.arange(tokens, device="cuda", dtype=torch.int32) + iteration) % 5 - 1
        )
        ranks.copy_(torch.tensor([16, 8, 0, 4], device="cuda", dtype=ranks.dtype))
        scalings.fill_(0.25 + iteration * 0.25)
        topk.copy_(
            (
                torch.arange(tokens * 2, device="cuda", dtype=torch.int32).view(
                    tokens, 2
                )
                + iteration
            )
            % 4
            - 1
        )
        if iteration == 2:
            slots.fill_(-1)
        route.sorted_pair_ids.fill_(-7)
        route.num_pairs_post_padded.zero_()
        graph.replay()
        torch.cuda.synchronize()
        expected = (x.float() @ base_weight.float().T).bfloat16().float()
        for token, slot in enumerate(slots.tolist()):
            if slot >= 0:
                rank = ranks[slot].item()
                low_rank = (
                    (x[token].float() @ a[slot, :rank].float().T).bfloat16().float()
                )
                expected[token] += scalings[slot] * (
                    low_rank @ b[slot, :, :rank].float().T
                )
        torch.testing.assert_close(dense.float(), expected, atol=0.012, rtol=0.04)
        torch.testing.assert_close(again, dense)
        expected_delta = torch.zeros_like(delta, dtype=torch.float32)
        for token, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            low_rank = (dense[token].float() @ moe_a[slot].float().T).bfloat16().float()
            for k, expert in enumerate(topk[token].tolist()):
                if expert >= 0:
                    expected_delta[token * 2 + k] = (
                        low_rank @ moe_b[slot, expert].float().T
                    )
        torch.testing.assert_close(delta.float(), expected_delta, atol=0.005, rtol=0.04)


def test_route_cache_visibility_follows_parallel_stream_joins():
    backend = TritonV2LoRABackend(4, torch.device("cuda"))
    runner, workspace = backend.runner, backend.runner.workspace
    slots = torch.arange(32, device="cuda", dtype=torch.int32) % 4
    _bind(runner, slots, graph_mode=True)
    route = runner.route(16)
    inherited = ("sorted", 16, 4)
    side_only, not_joined, after_done = ("side",), ("unjoined",), ("late",)
    side_stream = workspace.side_stream(slots.device)

    def unexpected_build():
        pytest.fail("the existing stream wait should make the route reusable")

    def side():
        assert workspace.route(slots, inherited, unexpected_build) is route
        workspace.route(slots, side_only, lambda: route)
        workspace.route(slots, not_joined, lambda: route)

    def compute():
        def unjoined_build():
            builds.append("unjoined")
            return route

        workspace.route(slots, not_joined, unjoined_build)
        # This work follows the outer done event, so its route cannot be
        # exposed by that event's eventual caller-side wait.
        with torch.cuda.stream(side_stream):
            workspace.route(slots, after_done, lambda: route)

    builds = []
    workspace.run_parallel(
        name="route-cache-test", device=slots.device, compute=compute, side=side
    )
    assert builds == ["unjoined"]
    assert workspace.route(slots, side_only, unexpected_build) is route

    def late_build():
        builds.append("late")
        return route

    workspace.route(slots, after_done, late_build)
    assert builds == ["unjoined", "late"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
