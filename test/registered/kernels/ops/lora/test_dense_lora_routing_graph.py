"""Dense route cache lifetime across warmup, capture, and replay."""

import pytest
import torch

pytest.importorskip("triton")

from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend
from sglang.srt.lora.dense import runner as runner_module
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
