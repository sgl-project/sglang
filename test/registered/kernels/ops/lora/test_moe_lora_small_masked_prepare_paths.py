"""Single-launch masked prepare matches the standard three-launch BF16 path.
Assert the selected path and exact metadata/live rows at 1-4 tokens with
768-, 1536-, and 1032-wide inputs.
"""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the cutedsl masked providers need an SM90+ GPU",
)


def _provider(num_experts: int, hidden: int, intermediate: int, gen):
    from sglang.srt.lora.moe.base_gemm_provider.cutedsl_bf16 import (
        CuteDslBf16MaskedProvider,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

    w13 = (
        torch.randn((num_experts, 2 * intermediate, hidden), generator=gen) * 0.05
    ).to(torch.bfloat16)
    w2 = (torch.randn((num_experts, hidden, intermediate), generator=gen) * 0.05).to(
        torch.bfloat16
    )
    return CuteDslBf16MaskedProvider(
        MoeLoraBf16QuantInfo(
            w13_weight=w13.cuda(),
            w2_weight=w2.cuda(),
            num_local_experts=num_experts,
            intermediate_size=intermediate,
            hidden_size=hidden,
        )
    )


def _same_rows(ref, got, ids, hidden):
    rows_ref = ref.hidden_permuted.view(-1, hidden)
    rows_got = got.hidden_permuted.view(-1, hidden)
    for pair in range(ids.numel()):
        expert = int(ids.view(-1)[pair].item())
        r_ref, r_got = (
            int(ref.pair_to_row[pair].item()),
            int(got.pair_to_row[pair].item()),
        )
        if expert < 0:
            assert r_ref == -1 and r_got == -1
            continue
        assert r_got // ref.m_max == expert
        assert torch.equal(
            rows_ref[r_ref].view(torch.int8), rows_got[r_got].view(torch.int8)
        )


@pytest.mark.parametrize("hidden", [768, 1536, 1032])
@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_single_launch_prepare_matches_the_standard_path(monkeypatch, hidden, tokens):
    from sglang.srt.lora.moe.base_gemm_provider import cutedsl_bf16 as provider_module
    from sglang.srt.lora.moe.kernels import dispatch_masked_small as small

    num_experts, intermediate, top_k = 16, 256, 2
    gen = torch.Generator(device="cpu").manual_seed(7 + hidden + tokens)
    provider = _provider(num_experts, hidden, intermediate, gen)
    hidden_states = (
        (torch.randn((tokens, hidden), generator=gen) * 0.2).to(torch.bfloat16).cuda()
    )
    ids = torch.stack(
        [torch.randperm(num_experts, generator=gen)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    if tokens >= 2:
        ids[1, 0] = -1  # a pair without an expert
    ids = ids.cuda()

    calls = []
    fast_path = small.small_masked_prepare

    def counted(*args, **kwargs):
        calls.append(1)
        return fast_path(*args, **kwargs)

    monkeypatch.setattr(provider_module, "small_masked_prepare", counted)

    monkeypatch.setattr(small, "_SMALL_PREPARE_MAX_PAIRS", 0)
    ref = provider.prepare(hidden_states, ids, top_k)
    assert not calls, "the standard path must not enter the single launch"

    monkeypatch.setattr(small, "_SMALL_PREPARE_MAX_PAIRS", 64)
    got = provider.prepare(hidden_states, ids, top_k)
    torch.cuda.synchronize()
    assert len(calls) == 1, "the single-launch path did not run"

    assert torch.equal(ref.masked_m, got.masked_m)
    assert torch.equal(ref.gemm1_tiles, got.gemm1_tiles)
    assert torch.equal(ref.gemm2_tiles, got.gemm2_tiles)
    n1, n2 = int(ref.gemm1_tiles.item()), int(ref.gemm2_tiles.item())
    assert torch.equal(
        ref.gemm1_schedule[:n1].sort().values, got.gemm1_schedule[:n1].sort().values
    )
    assert torch.equal(
        ref.gemm2_schedule[:n2].sort().values, got.gemm2_schedule[:n2].sort().values
    )
    _same_rows(ref, got, ids, hidden)


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, result


def _same_state(ref, got, ids, hidden):
    assert torch.equal(ref.masked_m, got.masked_m)
    assert torch.equal(ref.gemm1_tiles, got.gemm1_tiles)
    assert torch.equal(ref.gemm2_tiles, got.gemm2_tiles)
    n1, n2 = int(ref.gemm1_tiles.item()), int(ref.gemm2_tiles.item())
    assert torch.equal(
        ref.gemm1_schedule[:n1].sort().values, got.gemm1_schedule[:n1].sort().values
    )
    assert torch.equal(
        ref.gemm2_schedule[:n2].sort().values, got.gemm2_schedule[:n2].sort().values
    )
    _same_rows(ref, got, ids, hidden)


@pytest.mark.parametrize("hidden", [768, 1536, 1032])
def test_single_launch_prepare_replays_under_a_cuda_graph(monkeypatch, hidden):
    """Replay changed values/routes, including all-dead pairs followed by a full route."""
    from sglang.srt.lora.moe.base_gemm_provider import cutedsl_bf16 as provider_module
    from sglang.srt.lora.moe.kernels import dispatch_masked_small as small
    from sglang.srt.lora.workspace import LoraWorkspace
    from sglang.srt.runtime_context import get_context

    num_experts, intermediate, top_k, tokens = 16, 256, 2, 4
    gen = torch.Generator(device="cpu").manual_seed(11 + hidden)
    provider = _provider(num_experts, hidden, intermediate, gen)

    def values():
        return (torch.randn((tokens, hidden), generator=gen) * 0.2).to(torch.bfloat16)

    def routes(dead_pair, offset=0):
        ids = torch.stack(
            [
                (torch.randperm(num_experts, generator=gen)[:top_k] + offset)
                % num_experts
                for _ in range(tokens)
            ]
        ).to(torch.int32)
        if dead_pair is not None:
            ids.view(-1)[dead_pair] = -1
        return ids

    hidden_states = values().cuda()
    ids = routes(1).cuda()

    calls = []
    fast_path = small.small_masked_prepare

    def counted(*args, **kwargs):
        calls.append(1)
        return fast_path(*args, **kwargs)

    monkeypatch.setattr(provider_module, "small_masked_prepare", counted)
    monkeypatch.setattr(small, "_SMALL_PREPARE_MAX_PAIRS", 64)
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True)
    with get_context().override_server_args():
        graph, state = _capture(
            lambda: provider.prepare(hidden_states, ids, top_k, workspace)
        )
    assert len(calls) == 3, (
        "two warm-ups and the capture must all take the single launch"
    )

    variants = [
        (values(), routes(1)),  # the captured route shape with new values
        (values(), routes(5)),  # the dead pair elsewhere
        (values(), routes(None, offset=7)),  # other experts, no dead pair
        (
            values(),
            torch.full((tokens, top_k), -1, dtype=torch.int32),
        ),  # every pair dead
        (values(), routes(0)),  # a full route again after the empty one
    ]
    for new_values, new_ids in variants:
        hidden_states.copy_(new_values)
        ids.copy_(new_ids)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setattr(small, "_SMALL_PREPARE_MAX_PAIRS", 0)
        ref = provider.prepare(hidden_states, ids, top_k)  # the standard path, eagerly
        monkeypatch.setattr(small, "_SMALL_PREPARE_MAX_PAIRS", 64)
        assert len(calls) == 3, "the reference must come from the standard path"
        _same_state(ref, state, ids, hidden)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
