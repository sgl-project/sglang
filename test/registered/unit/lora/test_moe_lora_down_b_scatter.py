"""Down-B "scatter-into-base" (``down_b_scatter``) coverage for the SGL
MoE-LoRA engine.

CPU cases pin the plan flag's invariants (built directly from the
execution-plan spec classes) and the shipped config structure: every serial
per-expert prefill choice — the GB300 and H200 in-domain prefill scenarios
plus the per-expert out-of-domain prefill fallback — ships its down tail
reordered to down-A -> base down GEMM -> down-B(scatter) -> materialized
finalize in no-pair-delta mode.  The standalone one-launch down-B stage is
retained — only its output addressing changes — and the ``[T, K, H]``
pair-major delta buffer is never allocated.  Decode choices, overlapped
prefill, and shared-outer down-B are never scattered.

CUDA cases split by dependency:

* Kernel equality needs only Triton: ``invoke_down_b_scatter`` into a copy of
  the base rows followed by no-pair-delta ``post_reorder_deepgemm`` against
  the exact shipped tail — ``one_launch_sliced_lora_b`` writing a
  materialized LoRA delta that ``post_reorder_deepgemm`` re-reads — on random
  routing with sentinel pairs, a zero-routed token, base-only tokens, and a
  zero-padded (mixed-rank, mlpb-style) slot, over BOTH row-domain
  ``src2dst`` shapes (masked expert-major rows and contiguous compact rows).
  Agreement is allclose rather than bitwise BY JUSTIFICATION: the scatter
  rounds the FP32 delta to BF16 JOINTLY with the base row before the FP32
  weighted sum, whereas the shipped tail rounds the delta to BF16 separately
  (pair-major) and keeps base and delta as two operands of that sum.  Rows
  no valid pair targets must be BITWISE untouched (base-only traffic makes
  the whole launch a no-op), and an independent FP32 oracle pins the
  mathematical ``scale * sum(weight * (base + bridge @ B))`` contract.
* The runner-level oracle needs DeepGEMM: the SAME random inputs through the
  serial materialized plan and its scatter reordering must agree, on the
  masked provider and on the contiguous provider (row-domain agnosticism at
  the seam), and the scatter runner's workspace must never allocate the
  ``down_b:delta`` buffer.
* A CUDA-graph case captures the reordered pipeline and replays it with the
  batch flipped to base-only and with routing mutated in place.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    BridgeLayout,
    DeviceArchitecture,
    FinalizeFamily,
    FinalizeSpec,
    LateOverlap,
    LoraAFamily,
    LoraASpec,
    LoraBFamily,
    LoraBSpec,
    MiddleFamily,
    MiddleSpec,
    MoeLoraExecutionPlan,
    Site,
    StageContract,
    iter_selected_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

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


def _shipped_launch(architecture, sel, *, physical_rank=16, num_tokens=4096):
    """The shipped tile pick for one row, resolved the way serving does."""
    return resolve_tiles(
        architecture_value=architecture.value,
        plan_key_name=sel.name,
        physical_rank=physical_rank,
    ).config_for(num_tokens)


def _build_plan(
    *,
    activation=_SWIGLU,
    is_shared_outer=False,
    finalize_family=FinalizeFamily.MATERIALIZED,
    late_overlap=LateOverlap.NONE,
) -> MoeLoraExecutionPlan:
    """The serial one-launch plan shape, built the way config.py's
    ``_build_plan`` materializes a scenario (spec classes directly)."""
    pe = False
    consumes_down_b = finalize_family is not FinalizeFamily.MATERIALIZED
    down_b_contract = StageContract(Site.DOWN, is_shared_outer, BridgeLayout.PAIR_MAJOR)
    return MoeLoraExecutionPlan(
        gate_up_a=LoraASpec(
            Site.GATE_UP,
            LoraAFamily.GROUPED,
            is_shared_outer,
            BridgeLayout.PAIR_MAJOR,
        ),
        gate_up_b=LoraBSpec(
            Site.GATE_UP,
            LoraBFamily.ONE_LAUNCH_SLICED,
            pe,
            BridgeLayout.PAIR_MAJOR,
        ),
        middle=MiddleSpec(MiddleFamily.MATERIALIZED, activation),
        down_a=LoraASpec(Site.DOWN, LoraAFamily.GROUPED, pe, BridgeLayout.PAIR_MAJOR),
        down_b=(
            None
            if consumes_down_b
            else LoraBSpec(
                Site.DOWN,
                LoraBFamily.ONE_LAUNCH_SLICED,
                is_shared_outer,
                BridgeLayout.PAIR_MAJOR,
            )
        ),
        finalize=FinalizeSpec(
            finalize_family, down_b_contract if consumes_down_b else None
        ),
        late_overlap=late_overlap,
    ).validate_ownership(is_shared_outer)


def _serial_plan():
    """The serial materialized per-expert plan (the scatter's base shape)."""
    return _build_plan()


# ---- CPU: plan flag invariants --------------------------------------------------


class TestDownBScatterPlan:
    def test_raw_serial_plan_admits_the_flag(self) -> None:
        plan = _serial_plan()
        assert plan.down_b_scatter is False
        assert plan.down_b_scatter_eligible()
        reordered = replace(plan, down_b_scatter=True)
        # The standalone one-launch stage is RETAINED — only its output
        # addressing changes — and the finalize stays materialized.
        assert reordered.down_b is not None
        assert reordered.down_b.family is LoraBFamily.ONE_LAUNCH_SLICED
        assert reordered.finalize.family is FinalizeFamily.MATERIALIZED

    def test_flagged_plan_leaves_the_shape_keyed_conversions(self) -> None:
        # The scatter couples down-B to the base down output; it must not
        # re-qualify for conversions keyed on the plain serial shape.
        plan = _serial_plan()
        assert plan.is_fully_serial_materialized()
        assert not replace(plan, down_b_scatter=True).is_fully_serial_materialized()

    def test_flag_requires_the_one_launch_family(self) -> None:
        indexed = replace(
            _serial_plan(),
            down_b=LoraBSpec(
                Site.DOWN,
                LoraBFamily.INDEXED_PAIRS,
                False,
                BridgeLayout.PAIR_MAJOR,
            ),
        )
        assert indexed.down_b is not None
        assert indexed.down_b.family is not LoraBFamily.ONE_LAUNCH_SLICED
        with pytest.raises(ValueError, match="down-B scatter"):
            replace(indexed, down_b_scatter=True)

    def test_flag_requires_a_standalone_down_b(self) -> None:
        # A finalize-consumed down-B (shared-rank reduce) has no standalone
        # down-B stage for the scatter to reorder.  Built directly: the H200
        # shared prefill.shared_rank scenario still ships this form, and the
        # flag must keep rejecting it.
        consumed = _build_plan(
            is_shared_outer=True,
            finalize_family=FinalizeFamily.SHARED_RANK_REDUCE,
        )
        assert consumed.down_b is None
        with pytest.raises(ValueError, match="down-B scatter"):
            replace(consumed, down_b_scatter=True)

    def test_flag_rejects_late_overlap_windows(self) -> None:
        overlapped = _build_plan(late_overlap=LateOverlap.DOWN_B)
        with pytest.raises(ValueError, match="down-B scatter"):
            replace(overlapped, down_b_scatter=True)


# ---- CPU: shipped config structure -----------------------------------------------


def _scatter_expected(name: str, layout) -> bool:
    """The scatter ships on per-expert prefill serial shapes only."""
    if layout != False:
        return False
    return name in ("prefill.serial", "fallback.serial_prefill")


class TestDownBScatterConfig:
    def test_config_never_touches_decode_shared_or_overlapped(self) -> None:
        cases = (
            ("gb300_pe", _GB300, False, _SWIGLU),
            ("h200_pe", _H200, False, _SWIGLU),
            ("h200_sh", _H200, True, _SWIGLU),
            ("gb300_sh", _GB300, True, _SWIGLU),
            # Rows are activation-agnostic: the ReLU2 build of the same menu
            # (including the fallback rows every menu carries) makes the
            # same per-row scatter decision.
            ("gb300_relu2", _GB300, False, ActivationFamily.RELU2),
        )
        for name, architecture, layout, activation in cases:
            for row_name, choice in _menu(architecture, layout, activation).items():
                assert choice.plan.down_b_scatter is _scatter_expected(
                    row_name, layout
                ), (name, row_name)


class TestProviderScatterSurface:
    def test_both_row_domains_implement_the_scatter_epilogue(self) -> None:
        from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
            ContiguousRowDomainProvider,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
            MaskedRowDomainProvider,
        )
        from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

        quant_info = MoeLoraBf16QuantInfo(
            w13_weight=torch.zeros((4, 2 * 8, 16), dtype=torch.bfloat16),
            w2_weight=torch.zeros((4, 16, 8), dtype=torch.bfloat16),
            num_local_experts=4,
            intermediate_size=8,
            hidden_size=16,
        )
        assert MaskedRowDomainProvider(quant_info).supports_down_b_scatter()
        assert ContiguousRowDomainProvider(
            quant_info, m_alignment=128
        ).supports_down_b_scatter()

    def test_the_base_seam_fails_closed(self) -> None:
        from sglang.srt.lora.moe.base_gemm_provider.base import (
            MoeBaseProvider,
            MoeBaseProviderContract,
        )

        provider = MoeBaseProvider()
        provider.contract = MoeBaseProviderContract(
            key="stub",
            gate_first=True,
            interleaved=False,
            gate_up_output_dtype=torch.bfloat16,
            lora_delta_dtype=torch.bfloat16,
            lora_activation_dtype=torch.bfloat16,
            supported_output_dtypes=(torch.bfloat16,),
        )
        assert provider.supports_down_b_scatter() is False
        with pytest.raises(NotImplementedError, match="scatter"):
            provider.run_down_b_scatter(
                None,
                down_out=torch.zeros(1),
                bridge=torch.zeros(1),
                b_down=torch.zeros(1),
                routing=None,
                config={},
            )


# ---- CUDA: kernel-level equality ----------------------------------------------


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the scatter kernel needs any CUDA device (plain Triton)",
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
    reason="the runner-level oracle runs the DeepGEMM providers",
)

_HIDDEN = 128
_INTERMEDIATE = 128
_RANK = 16
_SLOTS = 2
_TOP_K = 2
_ROUTED_SCALING = 0.75
_ROW_POISON = 2**30  # sentinel-pair src2dst entries must never be dereferenced


def _kernel_case(num_tokens: int, top_k: int, num_experts: int, seed: int):
    """Random routing with sentinel pairs, a zero-routed token, base-only
    tokens, and a zero-padded (mixed-rank) slot."""
    generator = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=generator, dtype=torch.int32
    )
    topk_ids[torch.rand((num_tokens, top_k), generator=generator) < 0.15] = -1
    topk_ids[0] = -1  # a token with zero routed pairs
    pattern = torch.tensor([0, 1, -1, 0, -1, 1], dtype=torch.int32)
    token_slots = pattern.repeat(-(-num_tokens // pattern.numel()))[:num_tokens]
    weights = torch.rand((num_tokens, top_k), generator=generator) + 0.1
    topk_weights = (weights / weights.sum(dim=1, keepdim=True)).float()

    pairs = num_tokens * top_k
    bridge = (torch.randn((pairs, _RANK), generator=generator) * 0.15).to(
        torch.bfloat16
    )
    b_down = (
        torch.randn((_SLOTS * num_experts, _HIDDEN, _RANK), generator=generator) * 0.15
    ).to(torch.bfloat16)
    # Slot 1 is a narrower (mlpb-style) adapter: its factors are zero-filled
    # past half the physical rank, exactly how mixed ranks reach the kernels.
    b_down.view(_SLOTS, num_experts, _HIDDEN, _RANK)[1, :, :, _RANK // 2 :] = 0
    bridge_view = bridge.view(num_tokens, top_k, _RANK)
    bridge_view[token_slots == 1, :, _RANK // 2 :] = 0
    return topk_ids, token_slots, topk_weights, bridge, b_down


def _src2dst_rows(topk_ids: torch.Tensor, num_experts: int, style: str, seed: int):
    """Host model of both row domains' pair-to-row mapping over random rows."""
    ids = topk_ids.view(-1)
    counts = [int((ids == expert).sum()) for expert in range(num_experts)]
    if style == "masked":
        m_max = -(-max(counts + [1]) // 8) * 8 + 8
        base = [expert * m_max for expert in range(num_experts)]
        total_rows = num_experts * m_max
    else:  # contiguous compact rows at an 8-row segment alignment
        alignment = 8
        base, offset = [], 0
        for count in counts:
            base.append(offset)
            offset += -(-count // alignment) * alignment
        total_rows = max(offset, alignment)
    src2dst = torch.full((ids.numel(),), _ROW_POISON, dtype=torch.int32)
    cursor = [0] * num_experts
    for pair, expert in enumerate(ids.tolist()):
        if expert >= 0:
            src2dst[pair] = base[expert] + cursor[expert]
            cursor[expert] += 1
    generator = torch.Generator().manual_seed(seed ^ 0xD05E)
    down_rows = (torch.randn((total_rows, _HIDDEN), generator=generator) * 0.2).to(
        torch.bfloat16
    )
    return src2dst, down_rows


def _fp32_finalize_oracle(
    down_rows, src2dst, bridge, b_down, topk_ids, token_slots, topk_weights, num_experts
):
    num_tokens, top_k = topk_ids.shape
    out = torch.zeros((num_tokens, _HIDDEN), dtype=torch.float32)
    for token in range(num_tokens):
        slot = int(token_slots[token])
        for k in range(top_k):
            expert = int(topk_ids[token, k])
            if expert < 0:
                continue
            pair = token * top_k + k
            row = down_rows[int(src2dst[pair])].float()
            if 0 <= slot < _SLOTS:
                veid = slot * num_experts + expert
                row = row + b_down[veid].float() @ bridge[pair].float()
            out[token] += float(topk_weights[token, k]) * row
    return out * _ROUTED_SCALING


# One config drives BOTH launches: preserving the shipped down-B tiling is
# the entire point of the scatter variant.
_DOWN_B_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 16,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}

# The one intended divergence versus the shipped tail: the scatter rounds
# the FP32 delta to BF16 jointly with the base row, the shipped tail rounds
# the delta to BF16 separately before the FP32 combine — the same rounding
# class as the shared-rank/b_activation contract notes.
_SCATTER_TOLERANCE = {"atol": 1e-2, "rtol": 0.05}


def _post_reorder(down_rows, output, src2dst, topk_ids, topk_weights, lora_delta=None):
    from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm

    num_tokens, top_k = topk_ids.shape
    post_reorder_deepgemm(
        down_rows,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        top_k,
        num_tokens,
        _HIDDEN,
        _ROUTED_SCALING,
        lora_delta=lora_delta,
    )


@cuda_only
@pytest.mark.parametrize("row_domain", ("masked", "contiguous"))
@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts"),
    ((64, 2, 4), (13, 3, 4)),
    ids=("even", "partial"),
)
def test_scatter_matches_the_standalone_downb_plus_post_reorder(
    row_domain: str, num_tokens: int, top_k: int, num_experts: int
) -> None:
    """Scatter-add + no-pair-delta post_reorder must reproduce the shipped
    tail (one-launch LoRA delta + delta-reading post_reorder), on both row
    domains' src2dst shapes, under one shared down-B tiling config."""
    from sglang.srt.lora.moe.base_gemm_provider.down_b_scatter import (
        invoke_down_b_scatter,
    )
    from sglang.srt.lora.moe.lora_b import one_launch_sliced_lora_b
    from sglang.srt.lora.moe.routing import (
        ROUTE_ALIGNED,
        build_virtual_expert_routing,
    )

    device = torch.device("cuda")
    seed = 0x5CA7 + num_tokens + num_experts
    topk_ids, token_slots, topk_weights, bridge, b_down = _kernel_case(
        num_tokens, top_k, num_experts, seed
    )
    src2dst, down_rows = _src2dst_rows(topk_ids, num_experts, row_domain, seed)
    gpu = {
        name: tensor.to(device)
        for name, tensor in {
            "topk_ids": topk_ids,
            "token_slots": token_slots,
            "topk_weights": topk_weights,
            "bridge": bridge,
            "b_down": b_down,
            "src2dst": src2dst,
            "down_rows": down_rows,
        }.items()
    }
    aligned = build_virtual_expert_routing(
        gpu["topk_ids"],
        gpu["token_slots"],
        lora_experts_per_adapter=num_experts,
        max_loras=_SLOTS,
        block_size=16,
        view=ROUTE_ALIGNED,
    )

    # Shipped tail: one-launch down-B writes the materialized LoRA delta,
    # post_reorder re-reads it next to the untouched base rows.
    lora_delta = torch.empty(
        (num_tokens * top_k, _HIDDEN), dtype=torch.bfloat16, device=device
    )
    one_launch_sliced_lora_b(
        gpu["bridge"],
        gpu["b_down"],
        lora_delta,
        aligned,
        destination_offsets=(0,),
        config=_DOWN_B_CONFIG,
    )
    reference = torch.empty((num_tokens, _HIDDEN), dtype=torch.float32, device=device)
    _post_reorder(
        gpu["down_rows"],
        reference,
        gpu["src2dst"],
        gpu["topk_ids"],
        gpu["topk_weights"],
        lora_delta=lora_delta.view(num_tokens, top_k, _HIDDEN),
    )

    # Scatter tail: the SAME tiling adds the delta into a copy of the base
    # rows; post_reorder then runs in no-pair-delta mode.
    scattered = gpu["down_rows"].clone()
    invoke_down_b_scatter(
        down_rows=scattered,
        src2dst=gpu["src2dst"],
        bridge=gpu["bridge"],
        b_down=gpu["b_down"],
        routing=aligned,
        config=_DOWN_B_CONFIG,
    )
    output = torch.empty_like(reference)
    _post_reorder(
        scattered, output, gpu["src2dst"], gpu["topk_ids"], gpu["topk_weights"]
    )
    torch.testing.assert_close(output, reference, **_SCATTER_TOLERANCE)

    # Rows no LoRA-active pair targets are BITWISE untouched: base-only and
    # sentinel pairs contribute no add and their (poisoned or unwritten)
    # src2dst entries are never dereferenced.
    lora_active = (topk_ids.view(-1) >= 0) & (token_slots.repeat_interleave(top_k) >= 0)
    touched = src2dst[lora_active].long()
    untouched = torch.ones(down_rows.shape[0], dtype=torch.bool)
    untouched[touched] = False
    assert torch.equal(
        scattered[untouched.to(device)], gpu["down_rows"][untouched.to(device)]
    )

    # And both tails must satisfy the independent FP32 contract.
    oracle = _fp32_finalize_oracle(
        down_rows,
        src2dst,
        bridge,
        b_down,
        topk_ids,
        token_slots,
        topk_weights,
        num_experts,
    )
    torch.testing.assert_close(output.cpu(), oracle, atol=1.8e-2, rtol=0.06)

    # Base-only traffic: the launch is a bitwise no-op on the base rows and
    # the outputs collapse to the plain base combine.
    base_slots = torch.full_like(gpu["token_slots"], -1)
    aligned_base = build_virtual_expert_routing(
        gpu["topk_ids"],
        base_slots,
        lora_experts_per_adapter=num_experts,
        max_loras=_SLOTS,
        block_size=16,
        view=ROUTE_ALIGNED,
    )
    scattered_base = gpu["down_rows"].clone()
    invoke_down_b_scatter(
        down_rows=scattered_base,
        src2dst=gpu["src2dst"],
        bridge=gpu["bridge"],
        b_down=gpu["b_down"],
        routing=aligned_base,
        config=_DOWN_B_CONFIG,
    )
    assert torch.equal(scattered_base, gpu["down_rows"])
    # LoRA must actually contribute on the active routing.
    assert (scattered - gpu["down_rows"]).abs().max().item() > 1e-3


@cuda_only
def test_scatter_rejects_a_mismatched_route_block() -> None:
    from sglang.srt.lora.moe.base_gemm_provider.down_b_scatter import (
        invoke_down_b_scatter,
    )
    from sglang.srt.lora.moe.routing import (
        ROUTE_ALIGNED,
        build_virtual_expert_routing,
    )

    device = torch.device("cuda")
    topk_ids, token_slots, _weights, bridge, b_down = _kernel_case(4, 2, 4, 7)
    src2dst, down_rows = _src2dst_rows(topk_ids, 4, "masked", 7)
    aligned = build_virtual_expert_routing(
        topk_ids.to(device),
        token_slots.to(device),
        lora_experts_per_adapter=4,
        max_loras=_SLOTS,
        block_size=32,
        view=ROUTE_ALIGNED,
    )
    with pytest.raises(ValueError, match="BLOCK_SIZE_M"):
        invoke_down_b_scatter(
            down_rows=down_rows.to(device),
            src2dst=src2dst.to(device),
            bridge=bridge.to(device),
            b_down=b_down.to(device),
            routing=aligned,
            config=_DOWN_B_CONFIG,  # declares BLOCK_SIZE_M=16, route uses 32
        )


# ---- CUDA: runner-level oracle and graph replay --------------------------------


def _make_gpu_tensors(num_tokens: int, num_experts: int, device: torch.device):
    generator = torch.Generator().manual_seed(0x5CA1 + num_tokens)

    def rand_bf16(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    tensors = {
        "hidden_states": rand_bf16((num_tokens, _HIDDEN), 0.20),
        "w13_weight": rand_bf16((num_experts, 2 * _INTERMEDIATE, _HIDDEN), 0.08),
        "w2_weight": rand_bf16((num_experts, _HIDDEN, _INTERMEDIATE), 0.08),
        "gate_up_lora_a": rand_bf16((_SLOTS, num_experts, 2 * _RANK, _HIDDEN), 0.15),
        "gate_up_lora_b": rand_bf16(
            (_SLOTS, num_experts, 2 * _INTERMEDIATE, _RANK), 0.15
        ),
        "down_lora_a": rand_bf16((_SLOTS, num_experts, _RANK, _INTERMEDIATE), 0.15),
        "down_lora_b": rand_bf16((_SLOTS, num_experts, _HIDDEN, _RANK), 0.15),
        "adapter_enabled": torch.tensor([1, 1], dtype=torch.int32),
    }
    # Slot 1 is a narrower adapter (mixed ranks in one batch): zero-fill its
    # rank padding across all four factors, exactly how mlpb residents look.
    half = _RANK // 2
    tensors["gate_up_lora_a"][1].view(num_experts, 2, _RANK, _HIDDEN)[
        :, :, half:, :
    ] = 0
    tensors["down_lora_a"][1, :, half:, :] = 0
    tensors["gate_up_lora_b"][1, :, :, half:] = 0
    tensors["down_lora_b"][1, :, :, half:] = 0

    scores = torch.rand((num_tokens, num_experts), generator=generator)
    scores[: num_tokens // 2, 0] += 4.0  # skewed segments next to sparse ones
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
        pattern = torch.tensor([0, 1], dtype=torch.int32)
    elif traffic == "mixed":
        pattern = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    elif traffic == "base_only":
        pattern = torch.tensor([-1], dtype=torch.int32)
    else:
        raise AssertionError(traffic)
    return pattern.repeat(-(-num_tokens // pattern.numel()))[:num_tokens].contiguous()


def _standalone_output_allocation(runner, *, num_tokens, dtype, device):
    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


def _scatter_pair():
    """The serial materialized plan and its scatter reordering, with ONE
    shipped launch config driving both.

    The shipped menu carries the two forms composed with the b_act middle
    (the dedicated b_act suite covers that composition); this oracle
    isolates the down-tail change alone, so the reference is the shipped
    decode fallback choice — exactly the serial materialized shape with a
    complete tuned config — and the reordering is the same plan with the
    flag flipped."""
    reference = _menu(_GB300, False)["fallback.serial"]
    assert reference.plan.down_b_scatter is False
    assert reference.plan.middle.family is MiddleFamily.MATERIALIZED
    assert reference.plan == _serial_plan()
    reordered_plan = replace(reference.plan, down_b_scatter=True)
    assert reordered_plan.middle.family is MiddleFamily.MATERIALIZED
    return reference.plan, reordered_plan, _shipped_launch(_GB300, reference)


def _build_runner(plan, launch_config, provider_name: str, gpu, num_experts: int):
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner
    from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

    provider = MoeLoraRunner.select_provider_cls(provider_name)(
        MoeLoraBf16QuantInfo(
            w13_weight=gpu["w13_weight"],
            w2_weight=gpu["w2_weight"],
            num_local_experts=num_experts,
            intermediate_size=_INTERMEDIATE,
            hidden_size=_HIDDEN,
        )
    )
    runner = MoeLoraRunner(
        providers={"test": provider},
        top_k=_TOP_K,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFamily.SWIGLU,
    )
    runner._test_execution = dict(
        plan=plan, launch_config=launch_config, provider_name="test"
    )
    runner.prepare_plan(plan, provider_name="test", is_shared_outer=False)
    return runner


def _run_once(runner, gpu, token_slots, *, use_cuda_graph=False):
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
        is_prefill=True,
        has_active_lora=True,
    )
    return runner.run(
        dispatch, batch, output_dtype=torch.float32, **runner._test_execution
    )


def _workspace_buffer_names(runner) -> set[str]:
    workspace = runner.workspace
    return {key[0] for key in workspace._eager_buffers} | {
        key[0] for key in workspace._graph_buffers
    }


@deepgemm_cuda_only
@pytest.mark.parametrize("provider_name", ("deepgemm", "deepgemm_contiguous"))
def test_runner_scatter_matches_the_materialized_reference(
    provider_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One serial prefill batch, shipped tail vs the scatter reordering — on
    the masked provider and on the contiguous provider (row-domain
    agnosticism at the seam), with the delta-buffer accounting checked."""
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    serial_plan, reordered_plan, launch_config = _scatter_pair()
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    reference_runner = _build_runner(
        serial_plan, launch_config, "deepgemm", gpu, num_experts
    )
    scatter_runner = _build_runner(
        reordered_plan, launch_config, provider_name, gpu, num_experts
    )

    for traffic in ("active", "mixed", "base_only"):
        token_slots = _token_slots(traffic, num_tokens).to(device)
        reference = _run_once(reference_runner, gpu, token_slots).hidden_states
        scatter = _run_once(scatter_runner, gpu, token_slots).hidden_states
        torch.testing.assert_close(
            scatter, reference, **_SCATTER_TOLERANCE, msg=f"{provider_name}: {traffic}"
        )

    # The disappearing allocation: the scatter path never materializes the
    # pair-major [T*K, H] down delta, while the shipped tail always does.
    assert "down_b:delta" in _workspace_buffer_names(reference_runner)
    assert "down_b:delta" not in _workspace_buffer_names(scatter_runner)


@deepgemm_cuda_only
def test_scatter_pipeline_replays_correctly_in_a_real_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the reordered pipeline and replay it against the eager
    materialized reference under in-place batch mutation."""
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    serial_plan, reordered_plan, launch_config = _scatter_pair()
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    reference_runner = _build_runner(
        serial_plan, launch_config, "deepgemm", gpu, num_experts
    )
    scatter_runner = _build_runner(
        reordered_plan, launch_config, "deepgemm", gpu, num_experts
    )
    token_slots = _token_slots("active", num_tokens).to(device)

    for _ in range(2):  # JIT + workspace graph-buffer retention before capture
        _run_once(scatter_runner, gpu, token_slots, use_cuda_graph=True)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_once(scatter_runner, gpu, token_slots, use_cuda_graph=True)
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    reference = _run_once(reference_runner, gpu, token_slots).hidden_states
    torch.testing.assert_close(output, reference, **_SCATTER_TOLERANCE)

    # Replay 2: the whole batch flips to base-only through the sentinel.
    token_slots.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    reference = _run_once(reference_runner, gpu, token_slots).hidden_states
    torch.testing.assert_close(output, reference, **_SCATTER_TOLERANCE)

    # Replay 3: new routing and activations arrive in place, adapters return.
    gpu["topk_ids"].copy_(gpu["topk_ids"].flip(dims=(1,)))
    gpu["hidden_states"].copy_((gpu["hidden_states"].float() * 1.5).to(torch.bfloat16))
    token_slots.copy_(_token_slots("mixed", num_tokens).to(device))
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    reference = _run_once(reference_runner, gpu, token_slots).hidden_states
    torch.testing.assert_close(output, reference, **_SCATTER_TOLERANCE)

    assert "down_b:delta" not in _workspace_buffer_names(scatter_runner)
