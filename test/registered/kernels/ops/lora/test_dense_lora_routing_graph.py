"""Dense route cache lifetime across warmup, capture, and replay."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.lora.common.lora_a import grouped_lora_a
from sglang.kernels.ops.lora.moe.lora_b import _per_row_lora_b
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend
from sglang.srt.lora.dense import runner as runner_module
from sglang.srt.lora.dense.plan import DensePlan, Overlap
from sglang.srt.lora.moe.plan import RouteBuilderFamily, RouteRequirement
from sglang.srt.lora.moe.routing import build_moe_routes
from sglang.srt.lora.utils import Phase
from sglang.srt.lora.workspace import LoraWorkspace
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


@pytest.mark.parametrize("graph_mode", [False, True])
def test_forward_reset_preserves_batch_metadata_and_workspace(graph_mode):
    backend = TritonV2LoRABackend(4, torch.device("cuda"))
    backend.validate_lora_targets(None, {"lm_head"})
    slots = torch.arange(32, device="cuda", dtype=torch.int32) % 4
    metadata = {}
    for runner in (backend.runner, backend.lm_head_runner):
        _bind(runner, slots, graph_mode=graph_mode)
        runner.route(16)
        metadata[runner] = (
            runner._token_slots,
            runner.token_slots,
            runner.lora_ranks,
            runner.scalings,
            runner.workspace._graph_mode,
            runner.num_tokens,
            runner.route(16).sorted_pair_ids.data_ptr(),
        )
    backend.batch_info = object()
    batch_info = backend.batch_info

    backend.reset_routing_cache()

    assert backend.batch_info is batch_info
    for runner, expected in metadata.items():
        assert runner.active
        assert not runner.workspace.routes
        assert runner._raw_route is None
        actual = (
            runner._token_slots,
            runner.token_slots,
            runner.lora_ranks,
            runner.scalings,
            runner.workspace._graph_mode,
            runner.num_tokens,
        )
        assert all(a is b for a, b in zip(actual[:4], expected[:4]))
        assert actual[4:] == expected[4:6]
        assert runner.route(16).sorted_pair_ids.data_ptr() == expected[-1]


def test_token_extent_change_resets_routes_without_rebinding_metadata():
    backend = TritonV2LoRABackend(4, torch.device("cuda"))
    runner = backend.runner
    slots = torch.arange(64, device="cuda", dtype=torch.int32) % 4
    _bind(runner, slots, graph_mode=True)
    route = runner.route(16)
    runner.set_num_tokens(64)
    assert runner.route(16) is route

    runner.set_num_tokens(8)
    assert runner._token_slots is slots
    assert runner.token_slots.numel() == runner.num_tokens == 8
    assert not runner.workspace.routes and runner._raw_route is None
    smaller = runner.route(16)
    assert smaller is not route
    runner.set_num_tokens(64)
    assert not runner.workspace.routes
    assert runner.token_slots.data_ptr() == slots.data_ptr()
    assert runner.token_slots.numel() == 64


@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill"])
def test_capture_rebuilds_warmup_route_once_and_old_graph_replays(monkeypatch, prefill):
    backend = TritonV2LoRABackend(4, torch.device("cuda"))
    runner = backend.runner
    slots = torch.arange(64, device="cuda", dtype=torch.int32) % 4
    build = runner_module.build_route
    builds = []

    def counted_build(*args, **kwargs):
        builds.append(torch.cuda.is_current_stream_capturing())
        return build(*args, **kwargs)

    monkeypatch.setattr(runner_module, "build_route", counted_build)
    captures = {}
    stream = torch.cuda.Stream()
    for tokens in (64, 8):
        _bind(runner, slots[:tokens], graph_mode=True, prefill=prefill)

        def forward():
            backend.reset_routing_cache()
            route = runner.route(16)
            assert runner.route(16) is route
            return route

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        count = len(builds)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            route = forward()
        assert builds[count:] == [True]
        captures[tokens] = (graph, route)

    for iteration, tokens in enumerate((8, 64, 8, 64)):
        current = ((torch.arange(64, device="cuda") + iteration) % 5 - 1).int()
        if iteration == 2:
            current.fill_(-1)
        slots.copy_(current)
        graph, route = captures[tokens]
        route.sorted_pair_ids.fill_(-7)
        route.block_bucket_ids.fill_(-7)
        route.num_pairs_post_padded.zero_()
        count = len(builds)
        graph.replay()
        torch.cuda.synchronize()
        assert len(builds) == count
        ids = route.sorted_pair_ids[: route.num_pairs_post_padded.item()]
        bucket_ids = route.block_bucket_ids.repeat_interleave(16)[: ids.numel()]
        in_range = (ids >= 0) & (ids < tokens)
        # No-adapter tokens retain their IDs in the ignored bucket -1.
        mask = in_range & (bucket_ids >= 0) & (bucket_ids < 4)
        valid = ids[mask].long()
        expected = torch.where(current[:tokens] >= 0)[0]
        torch.testing.assert_close(valid.sort().values, expected)
        torch.testing.assert_close(bucket_ids[mask], current[ids[mask].long()])
        assert (bucket_ids[in_range & ~mask] == -1).all()
        assert (current[ids[in_range & ~mask].long()] == -1).all()


@pytest.mark.parametrize("overlap", list(Overlap))
@pytest.mark.parametrize("moe_first", [False, True], ids=["dense-first", "moe-first"])
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
            _per_row_lora_b(
                bridge,
                moe_b.flatten(0, 1),
                delta,
                routes.raw_per_expert,
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

        forward()
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
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


def test_moe_only_forward_reset_rebuilds_each_block_size_during_capture():
    backend = BaseLoRABackend(4, torch.device("cuda"))
    workspace = backend.lora_workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
    slots = torch.arange(64, device="cuda", dtype=torch.int32) % 4
    topk = torch.zeros((64, 2), device="cuda", dtype=torch.int32)
    plan = SimpleNamespace(
        route_requirements=lambda: {RouteRequirement.SHARED_TOKEN_PLAN},
        route_builder=RouteBuilderFamily.STANDARD,
    )

    def forward():
        backend.reset_routing_cache()
        return [
            build_moe_routes(
                plan,
                topk_ids=topk,
                token_lora_mapping=slots,
                num_local_experts=3,
                max_loras=4,
                block_size=block,
                workspace=workspace,
            ).shared_token
            for block in (16, 32)
        ]

    warmup = forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routes = forward()
    assert all(route is not old for route, old in zip(routes, warmup))
    assert routes[0].sorted_pair_ids.data_ptr() != routes[1].sorted_pair_ids.data_ptr()
    for step in range(3):
        slots.copy_((torch.arange(64, device="cuda") + step).int() % 5 - 1)
        graph.replay()
        torch.cuda.synchronize()
        for route in routes:
            ids = route.sorted_pair_ids[: route.num_pairs_post_padded.item()]
            bucket_ids = route.block_bucket_ids.repeat_interleave(route.block_size)[
                : ids.numel()
            ]
            valid = (ids >= 0) & (ids < 64) & (bucket_ids >= 0)
            torch.testing.assert_close(
                ids[valid].sort().values.long(), (slots >= 0).nonzero().flatten()
            )
            torch.testing.assert_close(bucket_ids[valid], slots[ids[valid].long()])


def test_shared_token_cache_misses_different_domains_and_resets():
    backend = TritonV2LoRABackend(4, torch.device("cuda"))
    runner, workspace = backend.runner, backend.runner.workspace
    slots = torch.arange(32, device="cuda", dtype=torch.int32) % 4
    _bind(runner, slots, graph_mode=True, prefill=True)
    route = runner.route(16)
    key = ("sorted", 16, 4)

    def unexpected_build():
        pytest.fail("compatible routes must be reused")

    def assert_miss(mapping, key=key):
        builds = []

        def build():
            builds.append(True)
            return route

        assert workspace.route(mapping, key, build) is route
        assert builds == [True]
        assert workspace.route(mapping, key, unexpected_build) is route

    assert workspace.route(slots[:], key, unexpected_build) is route
    assert_miss(slots.clone())
    assert_miss(slots[:8])
    assert_miss(slots[:16])
    assert_miss(slots[1:17])
    assert_miss(slots[::2])
    assert_miss(slots.view(torch.float32))
    assert_miss(slots, ("sorted", 32, 4))
    assert_miss(slots, ("sorted", 16, 8))
    with torch.cuda.stream(torch.cuda.Stream()):
        assert_miss(slots)
    workspace.begin_forward(graph_mode=True, is_prefill_graph=False)
    assert_miss(slots)
    workspace.begin_forward(graph_mode=False)
    assert_miss(slots)
    backend.reset_routing_cache()
    assert not workspace.routes


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
