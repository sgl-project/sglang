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
from sglang.srt.lora.moe.execution_plan import (
    ActivationFn,
    DeviceArchitecture,
    Phase,
    SelectedPlan,
    build_plan,
    load_plans,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci


def _segments_of(mapping: torch.Tensor) -> torch.Tensor:
    """Request boundaries for a test batch: one request per token, so the
    segment route is exercised with many short requests and the helper stays
    free of host syncs (CUDA-graph capture forbids a device-to-host copy)."""
    return torch.arange(mapping.numel() + 1, dtype=torch.int32, device=mapping.device)


register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

_GB300 = DeviceArchitecture.GB300
_H200 = DeviceArchitecture.H200
_SWIGLU = ActivationFn.SILU


def _menu(architecture, layout, activation=_SWIGLU):
    """Build every plan row with the loader and builder that serving uses.

    ``resolve_plans`` checks the phase and the rank, then keeps one row for
    each phase. This helper skips those checks and returns every row.
    """
    table = load_plans(architecture)
    layout_name = "shared" if layout else "per_expert"
    return {
        row.name: SelectedPlan(
            key=f"{architecture.value}.{layout_name}.{row.name}",
            name=row.name,
            base_gemm_rows=row.base_gemm_rows,
            plan=build_plan(row.plan, activation=activation, is_shared_outer=layout),
        )
        for row in (*table.scenarios, *table.fallback)
        if row.layout in (None, layout_name)
    }


def _choice(architecture, layout, name: str):
    return _menu(architecture, layout)[name]


def _shipped_launch(architecture, sel, *, physical_rank=16, num_tokens=4096):
    return resolve_tiles(
        architecture_value=architecture.value,
        plan_key_name=sel.name,
        physical_rank=physical_rank,
    ).config_for(num_tokens)


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


def _stub_quant_info(num_experts: int = 4) -> MoeLoraBf16QuantInfo:
    return MoeLoraBf16QuantInfo(
        w13_weight=torch.zeros((num_experts, 2 * 8, 16), dtype=torch.bfloat16),
        w2_weight=torch.zeros((num_experts, 16, 8), dtype=torch.bfloat16),
        num_local_experts=num_experts,
        intermediate_size=8,
        hidden_size=16,
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

    def test_exact_ceiling_formula(self) -> None:
        # ceil((P + E*(a-1)) / a) * a
        assert contiguous_m_pad_ceiling(128, 16, 128) == 2176
        assert contiguous_m_pad_ceiling(24, 4, 128) == 640
        assert contiguous_m_pad_ceiling(0, 4, 128) == 512
        assert contiguous_m_pad_ceiling(16, 2, 1) == 16


class TestRowDomainConfig:
    def test_prefill_ships_contiguous_providers(self) -> None:
        # Every prefill row uses route-major rows, the fallbacks included:
        # the masked slab domain allocates local_experts x padded-chunk-token
        # x hidden scratch, which is unbounded exactly where the fallback
        # serves (out-of-domain geometries -- 24.7GiB at E=256 H=6144 with an
        # 8192-token chunk). Every vendor ships a contiguous provider, so the
        # fallback never needs the masked domain for prefill.
        gb300_pe = _menu(_GB300, False)
        assert gb300_pe["prefill.per_expert"].base_gemm_rows == "route_major"
        assert gb300_pe["fallback.prefill.per_expert"].base_gemm_rows == "route_major"
        gb300_sh = _menu(_GB300, True)
        assert gb300_sh["prefill.shared"].base_gemm_rows == "route_major"
        assert gb300_sh["fallback.prefill.shared"].base_gemm_rows == "route_major"
        h200_pe = _menu(_H200, False)
        assert h200_pe["prefill.per_expert"].base_gemm_rows == "route_major"
        assert h200_pe["fallback.prefill.per_expert"].base_gemm_rows == "route_major"
        # Every shared prefill row on H200 uses route-major rows. The
        # shared-finalize rows can do so for two reasons. Their reduce works on
        # pairs. Their tail reads the base rows through ``pair_to_row`` alone.
        h200_sh = _menu(_H200, True)
        assert h200_sh["prefill.shared.rank_le8"].base_gemm_rows == "route_major"
        assert h200_sh["prefill.shared.rank_le64"].base_gemm_rows == "route_major"
        assert h200_sh["prefill.shared"].base_gemm_rows == "route_major"
        assert h200_sh["fallback.prefill.shared"].base_gemm_rows == "route_major"

    def test_decode_choices_never_convert(self) -> None:
        # Every decode row uses expert-major rows. On GB300 the contiguous
        # decode kernel is slower than the masked kernel.
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                for activation in (_SWIGLU, ActivationFn.RELU2):
                    for name, choice in _menu(architecture, layout, activation).items():
                        if not name.startswith(("decode.", "fallback.decode")):
                            continue
                        assert choice.base_gemm_rows == "expert_major", name

    def test_h200_large_expert_prefill_selects_the_contiguous_row_domain(self) -> None:
        # On SM90 with many experts, the masked scratch buffer does not fit in
        # memory. The contiguous row domain is then the only prefill
        # choice. That row also uses the b_act middle and the down-B scatter.
        routed = resolve_plans(
            quant_family="bf16",
            architecture=_H200,
            is_shared_outer=False,
            physical_rank=16,
            activation=_SWIGLU,
            hidden_size=4096,
            num_local_experts=512,
        )[Phase.PREFILL]
        # rank 16 sits in the gated band, which keeps the campaign-tuned plan
        assert routed.name == "prefill.per_expert.rank_le16"
        assert routed.base_gemm_rows == "route_major"
        assert routed.plan.down_b_into_base is True

    def test_runner_accepts_the_contiguous_provider_keys(self) -> None:

        assert select_provider_cls("route_major", "bf16", "triton") is not None
        assert select_provider_cls("route_major", "bf16") is not None
        with pytest.raises(ValueError, match="row order"):
            select_provider_cls("cutedsl_bf16_contiguous", "bf16", "cutedsl")
        # An unlisted vendor resolves to the family default, logged.
        assert select_provider_cls(
            "route_major", "bf16", "nosuchvendor"
        ) is select_provider_cls("route_major", "bf16")


class TestCuteDslContiguousScheduleGeometry:
    def test_token_tile_must_divide_the_segment_alignment(self) -> None:
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            validate_tile_geometry_contiguous,
        )

        for width in (8, 64, 128):
            validate_tile_geometry_contiguous(width, 128)
        for width in (48, 96, 256):
            with pytest.raises(ValueError, match="divide the segment alignment"):
                validate_tile_geometry_contiguous(width, 128)
        with pytest.raises(ValueError, match="positive"):
            validate_tile_geometry_contiguous(0, 128)
        with pytest.raises(ValueError, match="positive"):
            validate_tile_geometry_contiguous(64, 0)

    def test_capacity_bounds_every_actual_schedule_total(self) -> None:
        import random

        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
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

    def test_capacity_scales_with_rows_not_expert_worst_case(self) -> None:
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            dual_stage_schedule_capacities_contiguous,
            dual_stage_schedule_capacities_masked,
        )

        num_experts, num_tokens, top_k, token_width = 256, 8192, 8, 64
        num_pairs = num_tokens * top_k
        m_pad_ceiling = contiguous_m_pad_ceiling(num_pairs, num_experts, 128)
        contiguous1, _ = dual_stage_schedule_capacities_contiguous(
            num_experts=num_experts,
            m_pad_ceiling=m_pad_ceiling,
            max_expert_rows=num_tokens,
            m_alignment=128,
            token_width=token_width,
            n_gemm1=4096,
            n_gemm2=2048,
            output_width=128,
        )
        masked1, _ = dual_stage_schedule_capacities_masked(
            num_experts=num_experts,
            m_max=(num_tokens // 256 + 1) * 256,
            token_width=token_width,
            n_gemm1=4096,
            n_gemm2=2048,
            output_width=128,
        )
        assert contiguous1 * 10 < masked1

    def test_rejects_unaligned_ceiling_and_bad_tiles(self) -> None:
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            dual_stage_schedule_capacities_contiguous,
        )

        kwargs = dict(
            num_experts=4,
            max_expert_rows=64,
            m_alignment=128,
            n_gemm1=256,
            n_gemm2=256,
            output_width=128,
        )
        with pytest.raises(ValueError, match="multiple"):
            dual_stage_schedule_capacities_contiguous(
                m_pad_ceiling=130, token_width=64, **kwargs
            )
        with pytest.raises(ValueError, match="divide the segment alignment"):
            dual_stage_schedule_capacities_contiguous(
                m_pad_ceiling=512, token_width=48, **kwargs
            )
        # The packed-schedule checks call the masked validator.
        with pytest.raises(ValueError, match="cluster"):
            dual_stage_schedule_capacities_contiguous(
                m_pad_ceiling=512,
                token_width=64,
                cluster_shape_mn=(2, 1),
                **kwargs,
            )


class TestContiguousDomainValidation:
    def test_rejects_invalid_alignment(self) -> None:
        with pytest.raises(ValueError, match="m_alignment"):
            _DomainStub(_stub_quant_info(), m_alignment=0)


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
    """Check what the dispatch stage writes.

    It must build aligned segments, compact rows, and one expert label for
    each row. The rows after the last segment must hold ``-1``.
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

    # No host code clears the layout outputs; the launch must write every
    # count and offset. The test fills the buffers with garbage first to
    # prove that.
    from sglang.srt.lora.moe.kernels.dispatch_contiguous import (
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
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

    workspace = MoeLoraWorkspace()
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
        "adapter_enabled": torch.tensor([1, 0], dtype=torch.int32),
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
    """Publish one plan as the runner's whole phase menu.

    ``run`` resolves from the runner's own ``plans``/``tiles``; a test binds
    its single plan to both phases so the batch's phase flag cannot matter.
    """
    selected = SelectedPlan(key="test", name="test", base_gemm_rows="test", plan=plan)
    tiles = SimpleNamespace(config_for=lambda num_tokens: launch_config)
    runner.plans = {Phase.PREFILL: selected, Phase.DECODE: selected}
    runner.tiles = {selected.key: tiles}


def _build_runner(architecture, choice, vendor: str, gpu, num_experts, layout):
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

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
    _bind_test_menu(runner, choice.plan, _shipped_launch(architecture, choice))
    return runner


def _run_once(
    runner, gpu, token_lora_mapping, layout, *, use_cuda_graph=False, is_prefill=True
):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraBatch

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
        seg_indptr=_segments_of(token_lora_mapping),
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
        "prefill.shared.rank_le64",
        id="shared-outer-token-gemm",
    ),
)


def _oracle_setup(architecture, layout, fragment, num_tokens, num_experts, device):
    choice = _choice(architecture, layout, fragment)
    gpu = _make_gpu_tensors(num_tokens, num_experts, device, shared=layout == True)
    return choice, gpu


@cutedsl_cuda_only
@pytest.mark.parametrize(("architecture", "layout", "fragment"), _ELIGIBLE_PLAN_PARAMS)
def test_cutedsl_matches_the_triton_vendor(
    architecture,
    layout,
    fragment: str,
    published_server_args,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one plan on the CuTeDSL vendor and on the Triton vendor; compare."""
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    num_tokens, num_experts = 64, 16
    choice, gpu = _oracle_setup(
        architecture, layout, fragment, num_tokens, num_experts, device
    )

    triton = _build_runner(architecture, choice, "triton", gpu, num_experts, layout)
    cutedsl = _build_runner(architecture, choice, "cutedsl", gpu, num_experts, layout)

    for traffic in ("active", "mixed", "base_only"):
        token_lora_mapping = _token_lora_mapping(traffic, num_tokens).to(device)
        reference = _run_once(triton, gpu, token_lora_mapping, layout).hidden_states
        actual = _run_once(cutedsl, gpu, token_lora_mapping, layout).hidden_states
        torch.testing.assert_close(
            actual,
            reference,
            **_ORACLE_TOLERANCE,
            msg=f"cutedsl vs triton: {choice.key}, {traffic} traffic",
        )
        # The token with no routed pair must be exactly zero for both vendors.
        assert actual[0].abs().max().item() == 0.0
        assert reference[0].abs().max().item() == 0.0
    # The vendor under test must apply the LoRA math.
    active = _run_once(
        cutedsl, gpu, _token_lora_mapping("active", num_tokens).to(device), layout
    ).hidden_states
    base_only = _run_once(
        cutedsl, gpu, _token_lora_mapping("base_only", num_tokens).to(device), layout
    ).hidden_states
    assert (active - base_only).abs().max().item() > 0.02


@cutedsl_cuda_only
@pytest.mark.parametrize(("architecture", "layout", "fragment"), _ELIGIBLE_PLAN_PARAMS)
def test_cutedsl_replays_in_a_real_cuda_graph(
    architecture,
    layout,
    fragment: str,
    published_server_args,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the graph once, then replay it after the routing changes.

    Every replay rebuilds the counts, the offsets, the labels, and the compact
    rows in place from device memory. A change to ``topk_ids`` must reach the
    output through the same pointers.
    """
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    num_tokens, num_experts = 64, 16
    choice, gpu = _oracle_setup(
        architecture, layout, fragment, num_tokens, num_experts, device
    )
    triton = _build_runner(architecture, choice, "triton", gpu, num_experts, layout)
    cutedsl = _build_runner(architecture, choice, "cutedsl", gpu, num_experts, layout)
    token_lora_mapping = _token_lora_mapping("active", num_tokens).to(device)

    for _ in range(2):  # warm the JIT and keep the graph buffers before capture
        _run_once(cutedsl, gpu, token_lora_mapping, layout, use_cuda_graph=True)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_once(
            cutedsl, gpu, token_lora_mapping, layout, use_cuda_graph=True
        )
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        output,
        _run_once(triton, gpu, token_lora_mapping, layout).hidden_states,
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
    torch.testing.assert_close(
        output,
        _run_once(triton, gpu, token_lora_mapping, layout).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="mutated routing replay",
    )

    token_lora_mapping.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        output,
        _run_once(
            triton, gpu, _token_lora_mapping("base_only", num_tokens).to(device), layout
        ).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="base-only replay",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
