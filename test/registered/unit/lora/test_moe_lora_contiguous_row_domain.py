"""Route-major (contiguous) row-domain coverage for the MoE LoRA engine.

CPU cases pin the aligned-segment math (per-expert alignment rounding, empty
experts, the host-static ``M_pad`` ceiling and its align-multiple property),
the shipped route-major prefill config (every serial prefill choice ships a
contiguous provider — ``cutedsl_contiguous`` on tuned-domain gb300 keys,
``deepgemm_contiguous`` on h200 and per-expert out-of-domain keys — while
every decode choice keeps the expert-major providers by measured config),
and the CuTeDSL contiguous schedule geometry: the segment-containment
divisor rule (token tile | alignment) and the O(rows) packed-schedule
capacity bound.

CUDA cases split by dependency:

* Metadata/dispatch cases need only Triton: they run the domain's S1 on
  random routing with sentinel pairs, empty experts, and a zero-routed token
  and assert the seg_offsets / grouped_layout / src2dst / gather contract
  against a host model — including over garbage-dirtied outputs, since the
  seg-layout launch itself writes the -1 ceiling tail (no host memset) —
  plus the alignment-tagged workspace buffer naming that keeps differently
  aligned instances distinct in one shared layer workspace, and the
  S1-fused CuTeDSL schedule pack against the standalone builder
  (entry-identical on random routings).
* The oracle needs DeepGEMM: the SAME random inputs through the masked
  DeepGEMM runner and the contiguous DeepGEMM runner must agree, for EVERY
  eligible prefill plan shape (per-expert serial materialized, the
  shared-outer serial b_activation plan, and the shared-outer serial
  shared-rank plan, each on its own factor layout).  Agreement is
  allclose rather than bitwise BY JUSTIFICATION: S1's per-pair values, the
  S3 stage (materialized activation join or fused B+activation middle), and
  the S5 finalize (post_reorder or the shared-rank reduce + from-scratch
  tail — the reduce is pure pair-domain and the tail reads base rows only
  through ``src2dst``) are the identical kernels in the identical
  arithmetic order in both domains (only the physical row behind each
  ``src2dst`` entry differs), but the S2/S4 DeepGEMM masked and contiguous
  kernel families may select different tile configurations, so their BF16
  outputs can differ by rounding.  The ``cutedsl_contiguous`` oracle leg
  (SM100 + CuTeDSL only) shares the same tolerance discipline for the same
  reason: both GEMM engines accumulate in FP32 and round to BF16 per tile,
  so an engine swap is the same rounding-class divergence as a
  tile-configuration swap.
* CUDA-graph cases (per plan/layout) capture the contiguous prefill pipeline
  and replay it with the ROUTING mutated in place between replays (the
  capture-stable rebuild of counts, seg_offsets, and grouped_layout — plus,
  on the shared plan, the joint route pair) and with the batch flipped to
  base-only through the token-slot sentinel.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    contiguous_m_pad_ceiling,
)
from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    DeviceArchitecture,
    Phase,
    iter_selected_plans,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

_GB300 = DeviceArchitecture.GB300
_H200 = DeviceArchitecture.H200
_SWIGLU = ActivationFamily.SWIGLU


def _menu(architecture, layout, activation=_SWIGLU):
    """The shipped rows for one layout, keyed by row name."""
    return {
        sel.name: sel
        for sel in iter_selected_plans(
            architecture=architecture,
            is_shared_outer=layout,
            activation=activation,
        )
    }


def _choice(architecture, layout, name: str):
    return _menu(architecture, layout)[name]


def _shipped_launch(architecture, sel, *, physical_rank=16, num_tokens=4096):
    """The shipped tile pick for one row, resolved the way serving does."""
    return resolve_tiles(
        architecture_value=architecture.value,
        plan_key_name=sel.name,
        physical_rank=physical_rank,
    ).config_for(num_tokens)


def _host_aligned_prefix(counts: list[int], alignment: int) -> list[int]:
    """The exclusive aligned prefix the S1 kernel mirrors."""
    offsets = [0]
    for count in counts:
        aligned = -(-count // alignment) * alignment
        offsets.append(offsets[-1] + aligned)
    return offsets


class _DomainStub(ContiguousRowDomainProvider):
    """The domain with a contract but no GEMM engine: S1/S3/S5 only."""

    contract = MoeBaseProviderContract(
        key="contiguous_stub",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
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
        # Any split of num_pairs over the experts — dense, skewed, or with
        # empty experts — must fit under the host-static ceiling.
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
        # Route-major is the shipped prefill posture: the gb300 tuned-domain
        # prefill scenarios ship the CuTeDSL contiguous backend, h200 (SM90
        # has no CuTeDSL contiguous port) and the per-expert out-of-domain
        # prefill fallback ship the shape-general DeepGEMM contiguous
        # backend.  The shared-outer fallback twin has no per-expert
        # is_shared_outer to re-target and stays on the masked DeepGEMM provider.
        gb300_pe = _menu(_GB300, False)
        assert gb300_pe["prefill.serial"].provider == "cutedsl_contiguous"
        assert gb300_pe["fallback.serial_prefill"].provider == "deepgemm_contiguous"
        gb300_sh = _menu(_GB300, True)
        assert gb300_sh["prefill.token_dedup"].provider == "cutedsl_contiguous"
        assert gb300_sh["fallback.serial_prefill"].provider == "deepgemm"
        h200_pe = _menu(_H200, False)
        assert h200_pe["prefill.serial"].provider == "deepgemm_contiguous"
        assert h200_pe["fallback.serial_prefill"].provider == "deepgemm_contiguous"
        # Every H200 shared prefill band converted: the small-rank and
        # materialized b_activation twins and the shared-rank band (its
        # reduce is pair-domain, its tail reads base rows only through
        # src2dst).
        h200_sh = _menu(_H200, True)
        assert h200_sh["prefill.materialized.small_rank"].provider == (
            "deepgemm_contiguous"
        )
        assert h200_sh["prefill.shared_rank"].provider == "deepgemm_contiguous"
        assert h200_sh["prefill.materialized"].provider == "deepgemm_contiguous"
        assert h200_sh["fallback.serial_prefill"].provider == "deepgemm"

    def test_decode_choices_never_convert(self) -> None:
        # DECODE is categorically expert-major by measured config (the
        # contiguous decode port lost to masked on GB300: CuTeDSL@align8 by
        # 5-10%, DeepGEMM@128 by 22-39%) — including the fully serial decode
        # fallback and the ReLU2 decode winner.
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                for activation in (_SWIGLU, ActivationFamily.RELU2):
                    for name, choice in _menu(architecture, layout, activation).items():
                        if not name.startswith("decode.") and name != "fallback.serial":
                            continue
                        assert choice.provider in ("cutedsl", "deepgemm"), name

    def test_h200_large_expert_prefill_selects_the_contiguous_serial(self) -> None:
        # The DeepGEMM contiguous backend is the only feasible prefill
        # posture at large-expert geometry on SM90 (the masked E x chunk
        # scratch does not fit).  The served plan carries the composed
        # b_act middle and down-B scatter, so it is no longer the bare
        # fully-serial shape.
        routed = resolve_plans(
            architecture=_H200,
            is_shared_outer=False,
            physical_rank=16,
            activation=_SWIGLU,
            hidden_size=4096,
            num_local_experts=512,
        )[Phase.PREFILL]
        assert routed.name == "prefill.serial"
        assert routed.provider == "deepgemm_contiguous"
        assert not routed.plan.is_fully_serial_materialized()
        assert routed.plan.down_b_scatter is True

    def test_runner_accepts_the_contiguous_provider_keys(self) -> None:
        from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

        assert MoeLoraRunner.select_provider_cls("deepgemm_contiguous") is not None
        try:
            assert MoeLoraRunner.select_provider_cls("cutedsl_contiguous") is not None
        except NotImplementedError:
            pass  # known name, device-gated (SM90 has no contiguous port)
        with pytest.raises(ValueError, match="row_major"):
            MoeLoraRunner.select_provider_cls("row_major")


class TestCuteDslContiguousScheduleGeometry:
    """Host math behind the CuTeDSL contiguous kernel's schedule and safety."""

    def test_token_tile_must_divide_the_segment_alignment(self) -> None:
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            validate_contiguous_tile_geometry,
        )

        for width in (8, 64, 128):  # every study-winner tile qualifies at 128
            validate_contiguous_tile_geometry(width, 128)
        for width in (48, 96, 256):
            with pytest.raises(ValueError, match="divide the segment alignment"):
                validate_contiguous_tile_geometry(width, 128)
        with pytest.raises(ValueError, match="positive"):
            validate_contiguous_tile_geometry(0, 128)
        with pytest.raises(ValueError, match="positive"):
            validate_contiguous_tile_geometry(64, 0)

    def test_capacity_bounds_every_actual_schedule_total(self) -> None:
        import random

        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            contiguous_dual_stage_schedule_capacities,
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
            capacity1, capacity2 = contiguous_dual_stage_schedule_capacities(
                num_experts=num_experts,
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=max(num_pairs, 1),
                m_alignment=128,
                token_width=token_width,
                n_gemm1=n_gemm1,
                n_gemm2=n_gemm2,
                output_width=128,
            )
            # The builder writes exactly cdiv(count, w) * out_clusters entries
            # per expert; the capacity must bound that for ANY count split.
            clusters = sum(-(-count // token_width) for count in counts)
            assert clusters * -(-n_gemm1 // 128) <= capacity1, (counts, token_width)
            assert clusters * -(-n_gemm2 // 128) <= capacity2, (counts, token_width)

    def test_capacity_scales_with_rows_not_expert_worst_case(self) -> None:
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            contiguous_dual_stage_schedule_capacities,
            dual_stage_schedule_capacities,
        )

        # Prefill-scale geometry: the contiguous capacity is O(rows), far
        # below the masked builder's experts x worst-case-per-expert bound —
        # the same scaling argument as the domain's activation buffers.
        num_experts, num_tokens, top_k, token_width = 256, 8192, 8, 64
        num_pairs = num_tokens * top_k
        m_pad_ceiling = contiguous_m_pad_ceiling(num_pairs, num_experts, 128)
        contiguous1, _ = contiguous_dual_stage_schedule_capacities(
            num_experts=num_experts,
            m_pad_ceiling=m_pad_ceiling,
            max_expert_rows=num_tokens,
            m_alignment=128,
            token_width=token_width,
            n_gemm1=4096,
            n_gemm2=2048,
            output_width=128,
        )
        masked1, _ = dual_stage_schedule_capacities(
            num_experts=num_experts,
            m_max=(num_tokens // 256 + 1) * 256,
            token_width=token_width,
            n_gemm1=4096,
            n_gemm2=2048,
            output_width=128,
        )
        assert contiguous1 * 10 < masked1

    def test_schedule_pack_constructor_matches_the_standalone_geometry(self) -> None:
        # The S1-fused build consumes a pack this constructor alone builds;
        # its capacities, shifts, and geometry gates must be the standalone
        # builder's own, not a parallel copy that could drift.
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_abi import (
            OUTPUT_CLUSTER_SHIFT,
            TOKEN_CLUSTER_SHIFT,
        )
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            contiguous_dual_stage_schedule_capacities,
            contiguous_dual_stage_schedule_pack,
        )

        kwargs = dict(
            num_experts=6,
            m_pad_ceiling=contiguous_m_pad_ceiling(64 * 8, 6, 128),
            max_expert_rows=64,
            m_alignment=128,
            token_width=64,
            n_gemm1=512,
            n_gemm2=256,
            output_width=128,
        )
        pack = contiguous_dual_stage_schedule_pack(device=torch.device("cpu"), **kwargs)
        capacity1, capacity2 = contiguous_dual_stage_schedule_capacities(**kwargs)
        assert pack.schedule1.shape == (capacity1,)
        assert pack.schedule2.shape == (capacity2,)
        assert pack.tiles1.shape == pack.tiles2.shape == (1,)
        assert pack.token_cluster_shift == TOKEN_CLUSTER_SHIFT
        assert pack.output_cluster_shift == OUTPUT_CLUSTER_SHIFT
        assert (pack.out_clusters1, pack.out_clusters2) == (4, 2)
        with pytest.raises(ValueError, match="divide the segment alignment"):
            contiguous_dual_stage_schedule_pack(
                device=torch.device("cpu"), **{**kwargs, "token_width": 48}
            )

    def test_rejects_unaligned_ceiling_and_bad_tiles(self) -> None:
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            contiguous_dual_stage_schedule_capacities,
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
            contiguous_dual_stage_schedule_capacities(
                m_pad_ceiling=130, token_width=64, **kwargs
            )
        with pytest.raises(ValueError, match="divide the segment alignment"):
            contiguous_dual_stage_schedule_capacities(
                m_pad_ceiling=512, token_width=48, **kwargs
            )
        # The packed-ABI checks still delegate to the masked validator.
        with pytest.raises(ValueError, match="cluster"):
            contiguous_dual_stage_schedule_capacities(
                m_pad_ceiling=512,
                token_width=64,
                cluster_shape_mn=(2, 1),
                **kwargs,
            )


class TestContiguousDomainValidation:
    def test_rejects_invalid_alignment(self) -> None:
        with pytest.raises(ValueError, match="m_alignment"):
            _DomainStub(_stub_quant_info(), m_alignment=0)


# ---- CUDA cases --------------------------------------------------------------

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


def _deep_gemm_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from sglang.srt.layers.deep_gemm_wrapper import ENABLE_JIT_DEEPGEMM
    except Exception:
        return False
    return bool(ENABLE_JIT_DEEPGEMM)


deepgemm_cuda_only = pytest.mark.skipif(
    not _deep_gemm_ready(),
    reason="the masked-vs-contiguous oracle runs both DeepGEMM providers",
)


def _cutedsl_contiguous_ready() -> bool:
    # The oracle's masked reference is the DeepGEMM runner, so the CuTeDSL
    # leg needs everything the DeepGEMM leg does PLUS SM100 and the CuTeDSL
    # stack (the SM90 kernel has no contiguous port).
    if not _deep_gemm_ready():
        return False
    if torch.cuda.get_device_capability() < (10, 0):
        return False
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
    except Exception:
        return False
    return True


# Both contiguous engines run the SAME oracle and CUDA-graph cases below.
_CONTIGUOUS_PROVIDER_PARAMS = (
    pytest.param("deepgemm_contiguous", marks=deepgemm_cuda_only),
    pytest.param(
        "cutedsl_contiguous",
        marks=pytest.mark.skipif(
            not _cutedsl_contiguous_ready(),
            reason=(
                "the CuTeDSL contiguous leg needs SM100+, the CuTeDSL stack, "
                "and the DeepGEMM masked reference"
            ),
        ),
    ),
)


@triton_cuda_only
@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts", "alignment"),
    ((64, 8, 6, 128), (13, 3, 4, 8)),
    ids=("lane-multiple-align128", "partial-tail-align8"),
)
def test_contiguous_dispatch_metadata_contract(
    num_tokens: int, top_k: int, num_experts: int, alignment: int
) -> None:
    """S1 must produce aligned segments, dense unique compact rows, expert-
    labeled aligned segments with a -1 ceiling tail, and a bitwise gather."""
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(0xC017 + num_tokens)
    # Collision-heavy routing over few experts, expert 1 left empty, one
    # token fully unrouted, plus scattered sentinel pairs.
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=generator, dtype=torch.int32
    )
    topk_ids[topk_ids == 1] = 0  # empty expert with a zero-length segment
    topk_ids[torch.rand((num_tokens, top_k), generator=generator) < 0.15] = -1
    topk_ids[0] = -1  # a token with zero routed pairs
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

    mapping = ws.src2dst.cpu()
    layout = ws.grouped_layout.cpu()
    compact = ws.hidden_compact.cpu()
    host_hidden = hidden_states.cpu()
    for expert in range(num_experts):
        pairs = (ids == expert).nonzero().flatten()
        rows = mapping[pairs]
        # Dense unique slots inside this expert's aligned segment.
        assert sorted((rows - offsets[expert]).tolist()) == list(range(len(pairs)))
        for pair, row in zip(pairs.tolist(), rows.tolist()):
            torch.testing.assert_close(
                compact[row], host_hidden[pair // top_k], atol=0.0, rtol=0.0
            )
        # The FULL aligned segment carries the expert id (routed rows and the
        # partial-block padding alike) so every m-block is group-uniform.
        aligned_len = offsets[expert + 1] - offsets[expert]
        assert (layout[offsets[expert] : offsets[expert + 1]] == expert).all(), (
            expert,
            aligned_len,
        )
    # Everything past the dynamic aligned total keeps the -1 skip label.
    assert (layout[offsets[-1] :] == -1).all()
    assert counts[1] == 0 and offsets[2] == offsets[1]  # the empty expert

    # No host-side -1 prefill exists anymore: the seg-layout launch itself
    # must overwrite a GARBAGE buffer end to end — expert ids over every
    # aligned segment and -1 over the ceiling tail (written by the launch's
    # extra tail program, not a memset).  Run the dispatch directly on
    # deliberately dirtied outputs and require the identical layout.
    from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
        contiguous_dispatch_fill,
    )

    dirty = {
        "seg_counts_out": torch.full(
            (num_experts,), 0x2BAD, dtype=torch.int32, device=device
        ),
        "seg_offsets_out": torch.full(
            (num_experts + 1,), -7, dtype=torch.int32, device=device
        ),
        "src2dst_out": torch.full(
            (ids.numel(),), 0x7FFFFFFF, dtype=torch.int32, device=device
        ),
        "grouped_layout_out": torch.full(
            (ws.m_pad_ceiling,), 0x1DEA, dtype=torch.int32, device=device
        ),
        "hidden_compact_out": torch.zeros(
            (ws.m_pad_ceiling, _HIDDEN), dtype=torch.bfloat16, device=device
        ),
    }
    contiguous_dispatch_fill(
        hidden_states, topk_ids, num_experts, top_k, alignment, **dirty
    )
    dirty_layout = dirty["grouped_layout_out"].cpu()
    assert (dirty_layout[offsets[-1] :] == -1).all()
    for expert in range(num_experts):
        assert (dirty_layout[offsets[expert] : offsets[expert + 1]] == expert).all()
    assert torch.equal(dirty_layout, layout)
    assert dirty["seg_counts_out"].cpu().tolist() == counts
    assert dirty["seg_offsets_out"].cpu().tolist() == offsets

    # Alignment-tagged workspace names: differently aligned instances may
    # share one layer workspace, and their row geometry (compact rows,
    # ceilings) is alignment-dependent, so the buffer names carry the
    # alignment.
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

    workspace = MoeLoraWorkspace()
    workspace.begin_forward(graph_mode=False)
    provider.prepare(hidden_states, topk_ids, top_k, workspace)
    names = {key[0] for key in workspace._eager_buffers}
    for suffix in ("seg_counts", "seg_offsets", "src2dst", "grouped_layout"):
        assert f"contig:a{alignment}:{suffix}" in names, (suffix, names)
    assert f"contig:a{alignment}:hidden_compact" in names


@triton_cuda_only
def test_fused_schedule_pack_matches_the_standalone_builder() -> None:
    """The dual-stage schedules the S1 seg-layout launch packs must be
    entry-identical to the standalone builder's on random routings.

    Both builds read the same device ``seg_counts``; the fused kernel claims
    verbatim packing arithmetic, so agreement is bitwise over the first
    ``tiles`` entries of each stage (the rest is uninspected capacity)."""
    from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
        contiguous_dispatch_fill,
    )
    from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
        build_dual_stage_schedules_contiguous,
        contiguous_dual_stage_schedule_pack,
    )

    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(0x5CED)
    alignment, output_width = 128, 128
    for num_tokens, top_k, num_experts, token_width, n_gemm1, n_gemm2 in (
        (64, 8, 6, 64, 512, 256),  # multi-block segments, uneven experts
        (13, 3, 4, 8, 384, 128),  # narrow decode-style tile, partial tail
        (96, 2, 16, 128, 256, 640),  # xwide tile, out_clusters2 > out_clusters1
    ):
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), generator=generator, dtype=torch.int32
        )
        topk_ids[torch.rand((num_tokens, top_k), generator=generator) < 0.2] = -1
        topk_ids[1, 0] = 0  # at least one routed pair stays deterministic
        topk_ids = topk_ids.to(device)
        hidden_states = (
            (torch.randn((num_tokens, _HIDDEN), generator=generator) * 0.1)
            .to(torch.bfloat16)
            .to(device)
        )
        num_pairs = topk_ids.numel()
        m_pad_ceiling = contiguous_m_pad_ceiling(num_pairs, num_experts, alignment)
        pack = contiguous_dual_stage_schedule_pack(
            num_experts=num_experts,
            m_pad_ceiling=m_pad_ceiling,
            max_expert_rows=num_tokens,
            m_alignment=alignment,
            token_width=token_width,
            n_gemm1=n_gemm1,
            n_gemm2=n_gemm2,
            output_width=output_width,
            device=device,
        )
        # Dirty the pack outputs so agreement cannot come from shared zeros.
        pack.schedule1.fill_(0x3BAD)
        pack.schedule2.fill_(0x3BAD)
        pack.tiles1.fill_(-1)
        pack.tiles2.fill_(-1)
        buffers = {
            "seg_counts_out": torch.empty(
                num_experts, dtype=torch.int32, device=device
            ),
            "seg_offsets_out": torch.empty(
                num_experts + 1, dtype=torch.int32, device=device
            ),
            "src2dst_out": torch.empty(num_pairs, dtype=torch.int32, device=device),
            "grouped_layout_out": torch.empty(
                m_pad_ceiling, dtype=torch.int32, device=device
            ),
            "hidden_compact_out": torch.zeros(
                (m_pad_ceiling, _HIDDEN), dtype=torch.bfloat16, device=device
            ),
        }
        contiguous_dispatch_fill(
            hidden_states,
            topk_ids,
            num_experts,
            top_k,
            alignment,
            **buffers,
            schedule_pack=pack,
        )
        reference1, ref_tiles1, reference2, ref_tiles2 = (
            build_dual_stage_schedules_contiguous(
                buffers["seg_counts_out"],
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=num_tokens,
                m_alignment=alignment,
                token_width=token_width,
                n_gemm1=n_gemm1,
                n_gemm2=n_gemm2,
                output_width=output_width,
            )
        )
        num_tiles1, num_tiles2 = ref_tiles1.item(), ref_tiles2.item()
        assert pack.tiles1.item() == num_tiles1
        assert pack.tiles2.item() == num_tiles2
        assert num_tiles1 > 0 and num_tiles2 > 0
        assert torch.equal(pack.schedule1[:num_tiles1], reference1[:num_tiles1])
        assert torch.equal(pack.schedule2[:num_tiles2], reference2[:num_tiles2])
        # Host cross-check: both totals equal the counts' cluster sum.
        counts = buffers["seg_counts_out"].cpu().tolist()
        clusters = sum(-(-count // token_width) for count in counts)
        assert num_tiles1 == clusters * -(-n_gemm1 // output_width)
        assert num_tiles2 == clusters * -(-n_gemm2 // output_width)


def _make_gpu_tensors(
    num_tokens: int, num_experts: int, device: torch.device, *, shared: bool = False
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0xC0A7 + num_experts + num_tokens)

    def rand_bf16(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    # The shared-outer resident layout: ONE gate/up-A and ONE down-B factor
    # per adapter (expert dim 1); gate/up-B and down-A stay per-expert.
    gate_a_experts = 1 if shared else num_experts
    down_b_experts = 1 if shared else num_experts
    tensors = {
        "hidden_states": rand_bf16((num_tokens, _HIDDEN), 0.20),
        "w13_weight": rand_bf16((num_experts, 2 * _INTERMEDIATE, _HIDDEN), 0.08),
        "w2_weight": rand_bf16((num_experts, _HIDDEN, _INTERMEDIATE), 0.08),
        "gate_up_lora_a": rand_bf16(
            (_SLOTS, gate_a_experts, 2 * _PHYSICAL_RANK, _HIDDEN), 0.15
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
    # Skew the routing so per-expert counts are uneven (multi-block segments
    # next to empty experts under the 128-row alignment).
    scores[: num_tokens // 2, 0] += 4.0
    topk_ids = torch.topk(scores, _TOP_K, dim=1).indices.to(torch.int32)
    topk_ids[0] = -1  # a token with zero routed pairs
    topk_ids[5, 1] = -1  # a sentinel pair inside a live token
    tensors["topk_ids"] = topk_ids.contiguous()
    weights = torch.rand((num_tokens, _TOP_K), generator=generator) + 0.1
    tensors["topk_weights"] = (weights / weights.sum(dim=1, keepdim=True)).float()
    tensors["router_logits"] = torch.zeros(
        (num_tokens, num_experts), dtype=torch.float32
    )
    return {name: tensor.to(device) for name, tensor in tensors.items()}


def _token_slots(traffic: str, num_tokens: int) -> torch.Tensor:
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
    """Match eager output geometry without requiring a serving TP group."""
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


def _build_runner(architecture, choice, provider_name: str, gpu, num_experts, layout):
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    quant_info = MoeLoraBf16QuantInfo(
        w13_weight=gpu["w13_weight"],
        w2_weight=gpu["w2_weight"],
        num_local_experts=num_experts,
        intermediate_size=_INTERMEDIATE,
        hidden_size=_HIDDEN,
    )
    provider = MoeLoraRunner.select_provider_cls(provider_name)(quant_info)
    runner = MoeLoraRunner(
        providers={"test": provider},
        top_k=_TOP_K,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFamily.SWIGLU,
    )
    runner._test_execution = dict(
        plan=choice.plan,
        launch_config=_shipped_launch(architecture, choice),
        provider_name="test",
    )
    runner.prepare_plan(choice.plan, provider_name="test", is_shared_outer=layout)
    return runner


def _run_once(
    runner, gpu, token_slots, layout, *, use_cuda_graph=False, is_prefill=True
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
        token_slots=token_slots,
        adapter_enabled=gpu["adapter_enabled"],
        use_cuda_graph=use_cuda_graph,
        is_prefill=is_prefill,
        has_active_lora=True,
    )
    return runner.run(
        dispatch, batch, output_dtype=torch.float32, **runner._test_execution
    )


# The one intended divergence: the masked and contiguous S2/S4 kernel
# families (and, on the CuTeDSL leg, the GEMM engine itself) may tile and
# round BF16 outputs differently over the same FP32 accumulation; everything
# downstream (materialized activation join or fused B+activation middle,
# LoRA path, finalize — post_reorder and the shared-rank reduce + tail
# alike, both reused verbatim across domains) is the identical kernel in
# the identical order, so an engine swap is the same rounding-class
# divergence as a tile-configuration swap and shares one tolerance.
_ORACLE_TOLERANCE = {"atol": 2e-2, "rtol": 0.05}

# Every eligible prefill plan shape, each on its own resident factor
# layout: the per-expert serial materialized reference (the shipped decode
# fallback carries exactly that plan with a complete tuned config — the
# shipped per-expert prefill choice composes the b_act middle and down-B
# scatter, which the dedicated b_act/scatter suites cover), the shared-outer
# serial token-dedup b_activation winner (fused middle + mapped down-A +
# joint route builder with its routing PDL chain), and the H200 shared-outer
# serial shared-rank twin (fused middle + mapped down-A + the
# SHARED_RANK_REDUCE finalize consuming shared down-B over the raw route).
# The masked reference runner executes the SAME plan on the SAME DeepGEMM
# masked provider either way — this oracle isolates the row-domain seam.
_ELIGIBLE_PLAN_PARAMS = (
    pytest.param(_GB300, False, "fallback.serial", id="per-expert-materialized"),
    pytest.param(_GB300, True, "prefill.token_dedup", id="shared-outer-b-activation"),
    pytest.param(
        _H200,
        True,
        "prefill.shared_rank",
        id="shared-outer-shared-rank",
    ),
)


def _oracle_setup(architecture, layout, fragment, num_tokens, num_experts, device):
    choice = _choice(architecture, layout, fragment)
    gpu = _make_gpu_tensors(num_tokens, num_experts, device, shared=layout == True)
    return choice, gpu


@pytest.mark.parametrize("contiguous_provider", _CONTIGUOUS_PROVIDER_PARAMS)
@pytest.mark.parametrize(("architecture", "layout", "fragment"), _ELIGIBLE_PLAN_PARAMS)
def test_contiguous_prefill_matches_masked(
    contiguous_provider: str,
    architecture,
    layout,
    fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracle: one eligible prefill plan, two row domains."""
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    num_tokens, num_experts = 64, 16
    choice, gpu = _oracle_setup(
        architecture, layout, fragment, num_tokens, num_experts, device
    )

    masked = _build_runner(architecture, choice, "deepgemm", gpu, num_experts, layout)
    contiguous = _build_runner(
        architecture, choice, contiguous_provider, gpu, num_experts, layout
    )

    for traffic in ("active", "mixed", "base_only"):
        token_slots = _token_slots(traffic, num_tokens).to(device)
        reference = _run_once(masked, gpu, token_slots, layout).hidden_states
        actual = _run_once(contiguous, gpu, token_slots, layout).hidden_states
        torch.testing.assert_close(
            actual,
            reference,
            **_ORACLE_TOLERANCE,
            msg=f"contiguous vs masked: {choice.key}, {traffic} traffic",
        )
        # The zero-routed token must be exact zero in both domains.
        assert actual[0].abs().max().item() == 0.0
        assert reference[0].abs().max().item() == 0.0
    # Guard against both domains silently bypassing the LoRA math.
    active = _run_once(
        contiguous, gpu, _token_slots("active", num_tokens).to(device), layout
    ).hidden_states
    base_only = _run_once(
        contiguous, gpu, _token_slots("base_only", num_tokens).to(device), layout
    ).hidden_states
    assert (active - base_only).abs().max().item() > 0.02


@pytest.mark.parametrize("contiguous_provider", _CONTIGUOUS_PROVIDER_PARAMS)
@pytest.mark.parametrize(("architecture", "layout", "fragment"), _ELIGIBLE_PLAN_PARAMS)
def test_contiguous_prefill_replays_in_a_real_cuda_graph(
    contiguous_provider: str,
    architecture,
    layout,
    fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture once; replay across ROUTING mutations and a base-only flip.

    The counts, aligned seg_offsets, grouped_layout, compact rows — plus, on
    the shared plans, the aligned/raw shared-outer route products and the
    token-dedup plan (and the shared-rank token_rank workspace tensor), and,
    on the CuTeDSL leg, the packed tile schedules — are rebuilt in place on
    every replay from device memory alone, so mutating ``topk_ids`` between
    replays must be observed through unchanged pointers.
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
    masked = _build_runner(architecture, choice, "deepgemm", gpu, num_experts, layout)
    contiguous = _build_runner(
        architecture, choice, contiguous_provider, gpu, num_experts, layout
    )
    token_slots = _token_slots("active", num_tokens).to(device)

    for _ in range(2):  # JIT + workspace graph-buffer retention before capture
        _run_once(contiguous, gpu, token_slots, layout, use_cuda_graph=True)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_once(contiguous, gpu, token_slots, layout, use_cuda_graph=True)
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        output,
        _run_once(masked, gpu, token_slots, layout).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="initial routing replay",
    )

    # Mutate the ROUTING in place: rotate every token's expert choices to a
    # different valid set (keeping the sentinel rows), then replay.
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
        _run_once(masked, gpu, token_slots, layout).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="mutated routing replay",
    )

    # Flip the batch to base-only through the token-slot sentinel.
    token_slots.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        output,
        _run_once(
            masked, gpu, _token_slots("base_only", num_tokens).to(device), layout
        ).hidden_states,
        **_ORACLE_TOLERANCE,
        msg="base-only replay",
    )
