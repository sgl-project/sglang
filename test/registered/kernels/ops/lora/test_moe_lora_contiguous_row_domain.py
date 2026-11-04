from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    contiguous_m_pad_ceiling,
)
from sglang.srt.lora.moe.plan import (
    ActivationFn,
    SelectedPlan,
    _build_plan,
    _load_plans,
    _tile_table,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.srt.lora.utils import Phase
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_GB300 = "sm100"
_H200 = "sm90"
_SWIGLU = ActivationFn.SILU


def _menu(architecture, layout, activation=_SWIGLU):
    """Build every declared plan row, bypassing phase and rank selection."""
    table = _load_plans(architecture)
    layout_name = "shared" if layout else "per_expert"
    return {
        row.name: SelectedPlan(
            name=row.name,
            base_gemm_rows=row.base_gemm_rows,
            plan=_build_plan(row.plan, activation=activation, is_shared_outer=layout),
            tiles=_tile_table(row.tiles, physical_rank=16),
        )
        for row in (*table.scenarios, *table.fallback)
        if row.layout in (None, layout_name)
    }


def _shipped_launch(sel, *, num_tokens=4096):
    return sel.tiles.config_for(num_tokens)


def _host_aligned_prefix(counts: list[int], alignment: int) -> list[int]:
    offsets = [0]
    for count in counts:
        aligned = -(-count // alignment) * alignment
        offsets.append(offsets[-1] + aligned)
    return offsets


class _DomainStub(ContiguousRowDomainProvider):
    """A row domain with a contract but no GEMM engine. Only ``prepare`` runs."""

    contract = MoeBaseProviderContract(
        key="contiguous_stub",
        quant_info_cls=MoeLoraBf16QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )


class TestAlignedSegmentMath:
    def test_ceiling_bounds_every_aligned_prefix_total(self) -> None:
        import random

        rng = random.Random(0xA11C)
        for _ in range(64):
            alignment = rng.choice((8, 64, 128))
            num_experts = rng.randint(1, 32)
            num_pairs = rng.randint(0, 4096)
            counts = [0] * num_experts
            for _ in range(num_pairs):
                counts[rng.randrange(num_experts)] += 1
            total = _host_aligned_prefix(counts, alignment)[-1]
            ceiling = contiguous_m_pad_ceiling(num_pairs, num_experts, alignment)
            assert total <= ceiling, (counts, alignment, total, ceiling)


class TestCuteDslContiguousScheduleGeometry:
    def test_capacity_bounds_every_actual_schedule_total(self) -> None:
        import random

        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            dual_stage_schedule_capacities_contiguous,
        )

        rng = random.Random(0xC0DE)
        for _ in range(64):
            num_experts = rng.randint(1, 64)
            num_pairs = rng.randint(0, 4096)
            token_width = rng.choice((8, 64, 128))
            counts = [0] * num_experts
            for _ in range(num_pairs):
                counts[rng.randrange(num_experts)] += 1
            m_pad_ceiling = contiguous_m_pad_ceiling(num_pairs, num_experts, 128)
            n_gemm1, n_gemm2 = 512, 256
            capacity1, capacity2 = dual_stage_schedule_capacities_contiguous(
                num_experts=num_experts,
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=max(num_pairs, 1),
                m_alignment=128,
                token_width=token_width,
                n_gemm1=n_gemm1,
                n_gemm2=n_gemm2,
                output_width=128,
            )
            # The builder writes cdiv(count, w) * out_clusters entries for
            # each expert. The capacity must bound that total for any split of
            # the counts.
            clusters = sum(-(-count // token_width) for count in counts)
            assert clusters * -(-n_gemm1 // 128) <= capacity1, (counts, token_width)
            assert clusters * -(-n_gemm2 // 128) <= capacity2, (counts, token_width)


_TOP_K = 2
_HIDDEN = 128
_INTERMEDIATE = 128
_PHYSICAL_RANK = 16
_SLOTS = 2
_ROUTED_SCALING = 0.75

triton_cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the contiguous S1 dispatch needs any CUDA device (plain Triton)",
)


def _cutedsl_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        return False
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
    except Exception:
        return False
    return True


cutedsl_cuda_only = pytest.mark.skipif(
    not _cutedsl_ready(),
    reason="the vendor oracle runs the CuTeDSL providers against Triton",
)


@pytest.fixture
def published_server_args():
    # The triton vendor's config loader reads the published exec namespace.
    with get_context().override_server_args():
        yield


@triton_cuda_only
@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts", "alignment"),
    ((64, 8, 6, 128), (13, 3, 4, 8)),
    ids=("lane-multiple-align128", "partial-tail-align8"),
)
def test_contiguous_dispatch_metadata_contract(
    num_tokens: int, top_k: int, num_experts: int, alignment: int
) -> None:
    """Dispatch writes aligned segments, compact rows, and expert labels.
    The unused label tail must contain -1.
    """
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(0xC017 + num_tokens)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=generator, dtype=torch.int32
    )
    topk_ids[topk_ids == 1] = 0  # expert 1 gets no pair, so its segment is empty
    topk_ids[torch.rand((num_tokens, top_k), generator=generator) < 0.15] = -1
    topk_ids[0] = -1  # a token with no routed pair
    topk_ids = topk_ids.to(device)
    hidden_states = (
        (torch.randn((num_tokens, _HIDDEN), generator=generator) * 0.2)
        .to(torch.bfloat16)
        .to(device)
    )

    quant_info = MoeLoraBf16QuantInfo(
        w13_weight=torch.zeros(
            (num_experts, 2 * _INTERMEDIATE, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        ),
        w2_weight=torch.zeros(
            (num_experts, _HIDDEN, _INTERMEDIATE), dtype=torch.bfloat16, device=device
        ),
        num_local_experts=num_experts,
        intermediate_size=_INTERMEDIATE,
        hidden_size=_HIDDEN,
    )
    provider = _DomainStub(quant_info, m_alignment=alignment)
    ws = provider.prepare(hidden_states, topk_ids, top_k, None)

    ids = topk_ids.view(-1).cpu()
    counts = [(ids == expert).sum().item() for expert in range(num_experts)]
    offsets = _host_aligned_prefix(counts, alignment)
    assert ws.seg_counts.cpu().tolist() == counts
    assert ws.seg_offsets.cpu().tolist() == offsets
    assert ws.m_pad_ceiling == contiguous_m_pad_ceiling(
        ids.numel(), num_experts, alignment
    )
    assert offsets[-1] <= ws.m_pad_ceiling

    mapping = ws.pair_to_row.cpu()
    compact = ws.hidden_compact.cpu()
    host_hidden = hidden_states.cpu()
    for expert in range(num_experts):
        pairs = (ids == expert).nonzero().flatten()
        rows = mapping[pairs]
        assert sorted((rows - offsets[expert]).tolist()) == list(range(len(pairs)))
        for pair, row in zip(pairs.tolist(), rows.tolist()):
            torch.testing.assert_close(
                compact[row], host_hidden[pair // top_k], atol=0.0, rtol=0.0
            )
    assert counts[1] == 0 and offsets[2] == offsets[1]  # the empty expert

    # Poison outputs to prove the launch overwrites every count and offset.
    from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
        dispatch_layout_contiguous,
    )

    dirty = {
        "seg_counts_out": torch.full(
            (num_experts,), 0x2BAD, dtype=torch.int32, device=device
        ),
        "seg_offsets_out": torch.full(
            (num_experts + 1,), -7, dtype=torch.int32, device=device
        ),
        "pair_to_row_out": torch.full(
            (ids.numel(),), 0x7FFFFFFF, dtype=torch.int32, device=device
        ),
    }
    dispatch_layout_contiguous(
        hidden_states, topk_ids, num_experts, top_k, alignment, **dirty
    )
    assert dirty["seg_counts_out"].cpu().tolist() == counts
    assert dirty["seg_offsets_out"].cpu().tolist() == offsets

    # Two providers with different alignments can share one workspace. Their
    # row counts differ, so each buffer name holds the alignment.
    from sglang.srt.lora.workspace import LoraWorkspace

    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=False)
    provider.prepare(hidden_states, topk_ids, top_k, workspace)
    names = {key[0] for key in workspace._eager_buffers}
    for suffix in ("seg_counts", "seg_offsets", "pair_to_row"):
        assert f"contiguous:a{alignment}:{suffix}" in names, (suffix, names)
    assert f"contiguous:a{alignment}:hidden_compact" in names


def _make_gpu_tensors(
    num_tokens: int, num_experts: int, device: torch.device, *, shared: bool = False
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0xC0A7 + num_experts + num_tokens)

    def rand_bf16(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    # In the shared-outer layout, each adapter has one gate/up-A factor and
    # one down-B factor. The gate/up-B and down-A factors stay per expert.
    gate_up_a_experts = 1 if shared else num_experts
    down_b_experts = 1 if shared else num_experts
    tensors = {
        "hidden_states": rand_bf16((num_tokens, _HIDDEN), 0.20),
        "w13_weight": rand_bf16((num_experts, 2 * _INTERMEDIATE, _HIDDEN), 0.08),
        "w2_weight": rand_bf16((num_experts, _HIDDEN, _INTERMEDIATE), 0.08),
        "gate_up_lora_a": rand_bf16(
            (_SLOTS, gate_up_a_experts, 2 * _PHYSICAL_RANK, _HIDDEN), 0.15
        ),
        "gate_up_lora_b": rand_bf16(
            (_SLOTS, num_experts, 2 * _INTERMEDIATE, _PHYSICAL_RANK), 0.15
        ),
        "down_lora_a": rand_bf16(
            (_SLOTS, num_experts, _PHYSICAL_RANK, _INTERMEDIATE), 0.15
        ),
        "down_lora_b": rand_bf16(
            (_SLOTS, down_b_experts, _HIDDEN, _PHYSICAL_RANK), 0.15
        ),
    }
    scores = torch.rand((num_tokens, num_experts), generator=generator)
    # Skew the routing so the expert counts are uneven. Some segments then
    # span many blocks, and some experts get no pair.
    scores[: num_tokens // 2, 0] += 4.0
    topk_ids = torch.topk(scores, _TOP_K, dim=1).indices.to(torch.int32)
    topk_ids[0] = -1  # a token with no routed pair
    topk_ids[5, 1] = -1  # one sentinel pair inside a token that keeps a routed pair
    tensors["topk_ids"] = topk_ids.contiguous()
    weights = torch.rand((num_tokens, _TOP_K), generator=generator) + 0.1
    tensors["topk_weights"] = (weights / weights.sum(dim=1, keepdim=True)).float()
    tensors["router_logits"] = torch.zeros(
        (num_tokens, num_experts), dtype=torch.float32
    )
    return {name: tensor.to(device) for name, tensor in tensors.items()}


def _token_lora_mapping(traffic: str, num_tokens: int) -> torch.Tensor:
    if traffic == "active":
        return torch.zeros(num_tokens, dtype=torch.int32)
    if traffic == "mixed":
        pattern = torch.tensor([0, -1, -1, 0, -1, -1], dtype=torch.int32)
        return pattern.repeat((num_tokens + pattern.numel() - 1) // pattern.numel())[
            :num_tokens
        ].contiguous()
    if traffic == "base_only":
        return torch.full((num_tokens,), -1, dtype=torch.int32)
    raise AssertionError(f"unknown traffic pattern {traffic}")


def _standalone_output_allocation(runner, *, num_tokens, dtype, device):
    """Allocate the output with the eager shape and no tensor-parallel group."""
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


def _bind_test_menu(runner, plan, launch_config):
    """Bind one plan to both phases to isolate execution from plan selection."""
    tiles = SimpleNamespace(config_for=lambda num_tokens: launch_config)
    selected = SelectedPlan(name="test", base_gemm_rows="test", plan=plan, tiles=tiles)
    runner.plans = {Phase.PREFILL: selected, Phase.DECODE: selected}


def _build_runner(choice, vendor: str, gpu, num_experts):
    from sglang.srt.lora.moe.runner import MoeLoraRunner

    quant_info = MoeLoraBf16QuantInfo(
        w13_weight=gpu["w13_weight"],
        w2_weight=gpu["w2_weight"],
        num_local_experts=num_experts,
        intermediate_size=_INTERMEDIATE,
        hidden_size=_HIDDEN,
    )
    provider = select_provider_cls(choice.base_gemm_rows, "bf16", vendor)(quant_info)
    runner = MoeLoraRunner(
        providers={"test": provider},
        top_k=_TOP_K,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFn.SILU,
    )
    runner.validate_plan(choice.plan, base_gemm_rows="test")
    _bind_test_menu(runner, choice.plan, _shipped_launch(choice))
    return runner


def _run_once(
    runner, gpu, token_lora_mapping, *, use_cuda_graph=False, is_prefill=True
):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.lora.moe.runner import MoeLoraBatch

    dispatch = StandardDispatchOutput(
        hidden_states=gpu["hidden_states"],
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=gpu["topk_weights"],
            topk_ids=gpu["topk_ids"],
            router_logits=gpu["router_logits"],
        ),
    )
    batch = MoeLoraBatch(
        gate_up_lora_a=gpu["gate_up_lora_a"],
        gate_up_lora_b=gpu["gate_up_lora_b"],
        down_lora_a=gpu["down_lora_a"],
        down_lora_b=gpu["down_lora_b"],
        token_lora_mapping=token_lora_mapping,
        use_cuda_graph=use_cuda_graph,
        is_prefill=is_prefill,
    )
    return runner.run(dispatch, batch)


# The two vendors tile the same FP32 accumulation in different ways, and on a
# decode row in different row domains, so they round the BF16 output at
# different points. Every stage after the GEMM is the same kernel.
_ORACLE_TOLERANCE = {"atol": 2e-2, "rtol": 0.05}

# One row for each eligible plan shape, each with its own factor layout. Both
# vendors run the same plan, so the test compares the base GEMMs alone.
_ELIGIBLE_PLAN_PARAMS = (
    pytest.param(_GB300, False, "fallback.decode", id="per-expert-materialized"),
    pytest.param(_GB300, True, "prefill.shared", id="shared-outer-b-activation"),
    pytest.param(
        _H200,
        True,
        "prefill.shared.rank_le16",
        id="shared-outer-token-gemm",
    ),
)


@cutedsl_cuda_only
@pytest.mark.parametrize(("architecture", "layout", "fragment"), _ELIGIBLE_PLAN_PARAMS)
def test_cutedsl_matches_triton_in_eager_and_cuda_graph(
    architecture,
    layout,
    fragment: str,
    published_server_args,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare providers in eager mode and after in-place graph input updates."""
    from sglang.srt.lora.moe.runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    num_tokens, num_experts = 64, 16
    choice = _menu(architecture, layout)[fragment]
    gpu = _make_gpu_tensors(num_tokens, num_experts, device, shared=layout)

    triton = _build_runner(choice, "triton", gpu, num_experts)
    cutedsl = _build_runner(choice, "cutedsl", gpu, num_experts)

    outputs = {}
    for traffic in ("active", "mixed", "base_only"):
        triton.workspace.routes.clear()
        cutedsl.workspace.routes.clear()
        token_lora_mapping = _token_lora_mapping(traffic, num_tokens).to(device)
        reference = _run_once(triton, gpu, token_lora_mapping).hidden_states
        actual = _run_once(cutedsl, gpu, token_lora_mapping).hidden_states
        outputs[traffic] = actual
        torch.testing.assert_close(
            actual,
            reference,
            **_ORACLE_TOLERANCE,
            msg=f"cutedsl vs triton: {choice.name}, {traffic} traffic",
        )
        # The token with no routed pair must be exactly zero for both vendors.
        assert actual[0].abs().max().item() == 0.0
        assert reference[0].abs().max().item() == 0.0
    # The vendor under test must apply the LoRA math.
    assert (outputs["active"] - outputs["base_only"]).abs().max().item() > 0.02

    token_lora_mapping = _token_lora_mapping("active", num_tokens).to(device)

    # The workspace keys scratch by stream: warm up on the capture stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm the JIT and keep the graph buffers before capture
            _run_once(cutedsl, gpu, token_lora_mapping, use_cuda_graph=True)
    torch.cuda.synchronize(device)

    # Cache only within a forward, so the shared-token rebuild is captured.
    cutedsl.workspace.routes.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _run_once(cutedsl, gpu, token_lora_mapping, use_cuda_graph=True)
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    triton.workspace.routes.clear()
    torch.testing.assert_close(
        output,
        _run_once(triton, gpu, token_lora_mapping).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="initial routing replay",
    )

    original_ids = gpu["topk_ids"].clone()
    mutated = torch.where(
        original_ids >= 0, (original_ids + 3) % num_experts, original_ids
    ).to(torch.int32)
    gpu["topk_ids"].copy_(mutated)
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    triton.workspace.routes.clear()
    torch.testing.assert_close(
        output,
        _run_once(triton, gpu, token_lora_mapping).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="mutated routing replay",
    )

    token_lora_mapping.copy_(
        torch.arange(num_tokens, dtype=torch.int32, device=device) % _SLOTS
    )
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    triton.workspace.routes.clear()
    torch.testing.assert_close(
        output,
        _run_once(triton, gpu, token_lora_mapping).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="reassigned active adapters replay",
    )

    token_lora_mapping.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    triton.workspace.routes.clear()
    torch.testing.assert_close(
        output,
        _run_once(triton, gpu, token_lora_mapping).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="base-only replay",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
