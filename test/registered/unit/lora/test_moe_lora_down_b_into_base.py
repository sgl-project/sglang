from __future__ import annotations

import pathlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.execution_plan import (
    ActFamily,
    ActivationFn,
    ActSpec,
    BridgeLayout,
    DeviceArchitecture,
    DownOverlap,
    FinalizeFamily,
    FinalizeSpec,
    LoraAFamily,
    LoraASpec,
    LoraBFamily,
    LoraBSpec,
    MoeLoraExecutionPlan,
    Phase,
    RouteRequirement,
    SelectedPlan,
    Site,
    build_plan,
    load_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.test.ci.ci_register import register_cuda_ci


def _segments_of(mapping: torch.Tensor) -> torch.Tensor:
    """Request boundaries for a test batch: one request per token, so the
    segment route is exercised with many short requests and the helper stays
    free of host syncs (CUDA-graph capture forbids a device-to-host copy)."""
    return torch.arange(mapping.numel() + 1, dtype=torch.int32, device=mapping.device)


register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

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


def _shipped_launch(architecture, sel, *, physical_rank=16, num_tokens=4096):
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
    down_overlap=DownOverlap.NONE,
) -> MoeLoraExecutionPlan:
    pe = False
    consumes_down_b = finalize_family is not FinalizeFamily.MATERIALIZED
    return MoeLoraExecutionPlan(
        gate_up_a=LoraASpec(
            Site.GATE_UP,
            LoraAFamily.GROUPED,
            is_shared_outer,
            BridgeLayout.PAIR_MAJOR,
        ),
        gate_up_b=LoraBSpec(
            Site.GATE_UP,
            LoraBFamily.GROUPED,
            pe,
            BridgeLayout.PAIR_MAJOR,
        ),
        act=ActSpec(ActFamily.MATERIALIZED, activation),
        down_a=LoraASpec(Site.DOWN, LoraAFamily.GROUPED, pe, BridgeLayout.PAIR_MAJOR),
        down_b=(
            None
            if consumes_down_b
            else LoraBSpec(
                Site.DOWN,
                LoraBFamily.GROUPED,
                is_shared_outer,
                BridgeLayout.PAIR_MAJOR,
            )
        ),
        finalize=FinalizeSpec(
            finalize_family, is_shared_outer if consumes_down_b else False
        ),
        down_overlap=down_overlap,
    )


def _serial_plan():
    return _build_plan()


class TestDownBIntoBasePlan:
    def test_raw_serial_plan_admits_the_flag(self) -> None:
        plan = _serial_plan()
        assert plan.down_b_into_base is False
        assert plan.down_b_into_base_eligible()
        reordered = replace(plan, down_b_into_base=True)
        # The grouped down-B stage stays. Only its output address changes.
        assert reordered.down_b is not None
        assert reordered.down_b.family is LoraBFamily.GROUPED
        assert reordered.finalize.family is FinalizeFamily.MATERIALIZED

    def test_flagged_plan_leaves_the_shape_keyed_conversions(self) -> None:
        # With the scatter, down-B writes into the base down rows. The plan is
        # then no longer the plain serial shape.
        plan = _serial_plan()
        assert plan.is_fully_serial_materialized()
        assert not replace(plan, down_b_into_base=True).is_fully_serial_materialized()

    def test_flag_leaves_the_b_family_to_provider_capability(self) -> None:
        # The provider says which down-B kernel does the scatter. The plan
        # does not fix the down-B family.
        indexed = replace(
            _serial_plan(),
            down_b=LoraBSpec(
                Site.DOWN,
                LoraBFamily.PER_PAIR,
                False,
                BridgeLayout.PAIR_MAJOR,
            ),
        )
        assert replace(indexed, down_b_into_base=True).down_b_into_base is True

    def test_flag_moves_the_down_b_route_to_the_aligned_view(self) -> None:
        # A raw down-B family asks for the raw route. Under into-base that
        # family kernel does not run, so the plan must ask for the aligned
        # route that the into-base kernel reads. Before this rule the plan
        # built only the raw route and the first forward failed.
        indexed = replace(
            _serial_plan(),
            gate_up_a=LoraASpec(Site.GATE_UP, LoraAFamily.PER_PAIR),
            gate_up_b=LoraBSpec(Site.GATE_UP, LoraBFamily.PER_PAIR),
            down_a=LoraASpec(Site.DOWN, LoraAFamily.PER_PAIR),
            down_b=LoraBSpec(
                Site.DOWN,
                LoraBFamily.PER_PAIR,
                False,
                BridgeLayout.PAIR_MAJOR,
            ),
        )
        assert RouteRequirement.ALIGNED_PER_EXPERT not in indexed.route_requirements()
        scattered = replace(indexed, down_b_into_base=True)
        assert RouteRequirement.ALIGNED_PER_EXPERT in scattered.route_requirements()

    def test_flag_requires_a_standalone_down_b(self) -> None:
        # The shared-rank reduce consumes down-B inside finalize. There is
        # then no separate down-B stage to move. The H200 shared row uses
        # this form.
        consumed = _build_plan(
            is_shared_outer=True,
            finalize_family=FinalizeFamily.SHARED_RANK_REDUCE,
        )
        assert consumed.down_b is None
        with pytest.raises(ValueError, match="down-B into-base"):
            replace(consumed, down_b_into_base=True)

    def test_flag_rejects_the_windows_that_race_the_base_gemm(self) -> None:
        # down-B must run after the base down GEMM writes its rows. DOWN_B and
        # DOWN_A_B put down-B on the side stream next to the base GEMM. down-B
        # then reads rows before the base GEMM writes them.
        for window in (DownOverlap.DOWN_B, DownOverlap.DOWN_A_B):
            overlapped = _build_plan(down_overlap=window)
            with pytest.raises(ValueError, match="down-B into-base"):
                replace(overlapped, down_b_into_base=True)

    def test_flag_admits_the_down_a_window(self) -> None:
        # DOWN_A runs down-A next to the base GEMM. It joins before down-B
        # starts. The base rows are complete at that point, so the scatter is
        # safe. No shipped row uses this window. The measurement showed no
        # gain from it.
        forked = _build_plan(down_overlap=DownOverlap.DOWN_A)
        assert replace(forked, down_b_into_base=True).down_b_into_base is True


def _into_base_expected(name: str, layout) -> bool:
    """Return whether one plan row must add down-B into the base output.

    The prefill rows use it. A row skips it when finalize consumes down-B.
    No separate down-B stage then remains to move.
    """
    if layout != False:
        # gb300's rank_le32 shared row copies its parent and keeps the scatter;
        # h200's rank_le16/le64 shared rows finalize through shared_rank_reduce
        # and are excluded.
        return name in (
            "prefill.shared",
            "prefill.shared.rank_le8",
            "prefill.shared.rank_le32",
            "fallback.prefill.shared",
        )
    return name in (
        "prefill.per_expert",
        "prefill.per_expert.rank_le16",
        "prefill.per_expert.rank_le32",
        "fallback.prefill.per_expert",
    )


class TestDownBIntoBaseConfig:
    def test_config_never_touches_decode_shared_or_overlapped(self) -> None:
        cases = (
            ("gb300_pe", _GB300, False, _SWIGLU),
            ("h200_pe", _H200, False, _SWIGLU),
            ("h200_sh", _H200, True, _SWIGLU),
            ("gb300_sh", _GB300, True, _SWIGLU),
            # A row does not depend on the activation. The ReLU2 build makes
            # the same scatter choice for each row.
            ("gb300_relu2", _GB300, False, ActivationFn.RELU2),
        )
        for name, architecture, layout, activation in cases:
            for row_name, choice in _menu(architecture, layout, activation).items():
                assert choice.plan.down_b_into_base is _into_base_expected(
                    row_name, layout
                ), (name, row_name)


class TestProviderIntoBaseSurface:
    def test_both_row_domains_implement_the_into_base_epilogue(self) -> None:
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
        # The into-base epilogue is a LoRA kernel, so the runner calls it.
        # No provider wraps it: the only provider state it needs is pair_to_row,
        # which the runner already holds from the down-A mapping.
        from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider

        for cls in (
            MaskedRowDomainProvider,
            ContiguousRowDomainProvider,
            MoeBaseProvider,
        ):
            assert not hasattr(cls, "run_down_b_into_base")
        runner_src = (
            pathlib.Path(__file__).resolve().parents[4]
            / "python/sglang/srt/lora/moe/moe_lora_runner.py"
        ).read_text()
        assert "invoke_down_b_into_base(" in runner_src
        # The mapping is unconditional, so no other plan field can decide
        # whether the runner holds pair_to_row.
        assert "= provider.mapped_down_lora_a_input(" in runner_src
        for cls in (MaskedRowDomainProvider, ContiguousRowDomainProvider):
            assert MaskedRowDomainProvider(quant_info).mapped_down_lora_a_input


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the scatter kernel needs any CUDA device (plain Triton)",
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
    reason="the runner-level oracle runs the CuTeDSL providers",
)

_HIDDEN = 128
_INTERMEDIATE = 128
_RANK = 16
_SLOTS = 2
_TOP_K = 2
_ROUTED_SCALING = 0.75
_ROW_POISON = 2**30  # the kernel must never read a sentinel pair's pair_to_row entry


def _kernel_case(num_tokens: int, top_k: int, num_experts: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), generator=generator, dtype=torch.int32
    )
    topk_ids[torch.rand((num_tokens, top_k), generator=generator) < 0.15] = -1
    topk_ids[0] = -1  # a token with no routed pair
    pattern = torch.tensor([0, 1, -1, 0, -1, 1], dtype=torch.int32)
    token_lora_mapping = pattern.repeat(-(-num_tokens // pattern.numel()))[:num_tokens]
    weights = torch.rand((num_tokens, top_k), generator=generator) + 0.1
    topk_weights = (weights / weights.sum(dim=1, keepdim=True)).float()

    pairs = num_tokens * top_k
    bridge = (torch.randn((pairs, _RANK), generator=generator) * 0.15).to(
        torch.bfloat16
    )
    b_down = (
        torch.randn((_SLOTS * num_experts, _HIDDEN, _RANK), generator=generator) * 0.15
    ).to(torch.bfloat16)
    # Slot 1 is a narrow adapter. Its rank padding is zero in the bridge and
    # in down-B. A batch with mixed ranks reaches the kernels in this form.
    b_down.view(_SLOTS, num_experts, _HIDDEN, _RANK)[1, :, :, _RANK // 2 :] = 0
    bridge_view = bridge.view(num_tokens, top_k, _RANK)
    bridge_view[token_lora_mapping == 1, :, _RANK // 2 :] = 0
    return topk_ids, token_lora_mapping, topk_weights, bridge, b_down


def _pair_to_row_rows(topk_ids: torch.Tensor, num_experts: int, style: str, seed: int):
    """Build the pair-to-row map of both row domains on the host."""
    ids = topk_ids.view(-1)
    counts = [int((ids == expert).sum()) for expert in range(num_experts)]
    if style == "masked":
        m_max = -(-max(counts + [1]) // 8) * 8 + 8
        base = [expert * m_max for expert in range(num_experts)]
        total_rows = num_experts * m_max
    else:  # contiguous rows, with each segment aligned to 8 rows
        alignment = 8
        base, offset = [], 0
        for count in counts:
            base.append(offset)
            offset += -(-count // alignment) * alignment
        total_rows = max(offset, alignment)
    pair_to_row = torch.full((ids.numel(),), _ROW_POISON, dtype=torch.int32)
    cursor = [0] * num_experts
    for pair, expert in enumerate(ids.tolist()):
        if expert >= 0:
            pair_to_row[pair] = base[expert] + cursor[expert]
            cursor[expert] += 1
    generator = torch.Generator().manual_seed(seed ^ 0xD05E)
    down_rows = (torch.randn((total_rows, _HIDDEN), generator=generator) * 0.2).to(
        torch.bfloat16
    )
    return pair_to_row, down_rows


def _fp32_finalize_oracle(
    down_rows,
    pair_to_row,
    bridge,
    b_down,
    topk_ids,
    token_lora_mapping,
    topk_weights,
    num_experts,
):
    num_tokens, top_k = topk_ids.shape
    out = torch.zeros((num_tokens, _HIDDEN), dtype=torch.float32)
    for token in range(num_tokens):
        slot = int(token_lora_mapping[token])
        for k in range(top_k):
            expert = int(topk_ids[token, k])
            if expert < 0:
                continue
            pair = token * top_k + k
            row = down_rows[int(pair_to_row[pair])].float()
            if 0 <= slot < _SLOTS:
                veid = slot * num_experts + expert
                row = row + b_down[veid].float() @ bridge[pair].float()
            out[token] += float(topk_weights[token, k]) * row
    return out * _ROUTED_SCALING


# Both launches use one config. The scatter must keep the same down-B tiling
# as the shipped path.
_DOWN_B_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 16,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}

# The scatter adds the FP32 delta to the base row, then rounds once. The
# shipped path rounds the delta to BF16 first, then adds. The two paths round
# at different points, so compare them with a tolerance.
_INTO_BASE_TOLERANCE = {"atol": 1e-2, "rtol": 0.05}


def _post_reorder(
    down_rows, output, pair_to_row, topk_ids, topk_weights, lora_delta=None
):
    from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm

    num_tokens, top_k = topk_ids.shape
    post_reorder_deepgemm(
        down_rows,
        output,
        pair_to_row,
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
def test_into_base_matches_the_standalone_downb_plus_post_reorder(
    row_domain: str, num_tokens: int, top_k: int, num_experts: int
) -> None:
    from sglang.srt.lora.moe.kernels.lora_b import (
        grouped_lora_b,
        invoke_down_b_into_base,
    )
    from sglang.srt.lora.moe.route_view import RouteViewKind
    from sglang.srt.lora.moe.routing import (
        build_virtual_expert_routing,
    )

    device = torch.device("cuda")
    seed = 0x5CA7 + num_tokens + num_experts
    topk_ids, token_lora_mapping, topk_weights, bridge, b_down = _kernel_case(
        num_tokens, top_k, num_experts, seed
    )
    pair_to_row, down_rows = _pair_to_row_rows(topk_ids, num_experts, row_domain, seed)
    gpu = {
        name: tensor.to(device)
        for name, tensor in {
            "topk_ids": topk_ids,
            "token_lora_mapping": token_lora_mapping,
            "topk_weights": topk_weights,
            "bridge": bridge,
            "b_down": b_down,
            "pair_to_row": pair_to_row,
            "down_rows": down_rows,
        }.items()
    }
    aligned = build_virtual_expert_routing(
        gpu["topk_ids"],
        gpu["token_lora_mapping"],
        num_local_experts=num_experts,
        max_loras=_SLOTS,
        block_size=16,
        view=RouteViewKind.ALIGNED,
    )

    # In the shipped path, down-B writes the LoRA delta to its own buffer.
    # post_reorder then reads that buffer and the unchanged base rows.
    lora_delta = torch.empty(
        (num_tokens * top_k, _HIDDEN), dtype=torch.bfloat16, device=device
    )
    grouped_lora_b(
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
        gpu["pair_to_row"],
        gpu["topk_ids"],
        gpu["topk_weights"],
        lora_delta=lora_delta.view(num_tokens, top_k, _HIDDEN),
    )

    # In the scatter path, the same tiling adds the delta into a copy of the
    # base rows. post_reorder then runs with no delta buffer.
    scattered = gpu["down_rows"].clone()
    invoke_down_b_into_base(
        down_rows=scattered,
        pair_to_row=gpu["pair_to_row"],
        bridge=gpu["bridge"],
        b_down=gpu["b_down"],
        routing=aligned,
        config=_DOWN_B_CONFIG,
    )
    output = torch.empty_like(reference)
    _post_reorder(
        scattered, output, gpu["pair_to_row"], gpu["topk_ids"], gpu["topk_weights"]
    )
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    # A row stays bitwise equal if no active LoRA pair targets it. A base-only
    # pair and a sentinel pair add nothing. The kernel never reads their
    # pair_to_row entries.
    lora_active = (topk_ids.view(-1) >= 0) & (
        token_lora_mapping.repeat_interleave(top_k) >= 0
    )
    touched = pair_to_row[lora_active].long()
    untouched = torch.ones(down_rows.shape[0], dtype=torch.bool)
    untouched[touched] = False
    assert torch.equal(
        scattered[untouched.to(device)], gpu["down_rows"][untouched.to(device)]
    )

    oracle = _fp32_finalize_oracle(
        down_rows,
        pair_to_row,
        bridge,
        b_down,
        topk_ids,
        token_lora_mapping,
        topk_weights,
        num_experts,
    )
    torch.testing.assert_close(output.cpu(), oracle, atol=1.8e-2, rtol=0.06)

    # With base-only traffic, the kernel must not change any base row.
    base_slots = torch.full_like(gpu["token_lora_mapping"], -1)
    aligned_base = build_virtual_expert_routing(
        gpu["topk_ids"],
        base_slots,
        num_local_experts=num_experts,
        max_loras=_SLOTS,
        block_size=16,
        view=RouteViewKind.ALIGNED,
    )
    scattered_base = gpu["down_rows"].clone()
    invoke_down_b_into_base(
        down_rows=scattered_base,
        pair_to_row=gpu["pair_to_row"],
        bridge=gpu["bridge"],
        b_down=gpu["b_down"],
        routing=aligned_base,
        config=_DOWN_B_CONFIG,
    )
    assert torch.equal(scattered_base, gpu["down_rows"])
    # The active routing must change the rows.
    assert (scattered - gpu["down_rows"]).abs().max().item() > 1e-3


def test_config_rejects_a_mismatched_route_block() -> None:
    # The grouped B kernels take their row tile from the aligned route, so a
    # config row tuned for another BLOCK_SIZE_M must fail at table load, not
    # silently run under the wrong tile.
    from sglang.srt.lora.moe.launch_config import MoeLoraLaunchConfig

    with pytest.raises(ValueError, match="BLOCK_SIZE_M"):
        MoeLoraLaunchConfig(
            routing_block_size=32,
            down_b=dict(_DOWN_B_CONFIG),  # the config says 16, the route uses 32
        )


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
    # Slot 1 is a narrow adapter. Its rank padding is zero in all four
    # factors. A batch with mixed ranks reaches the kernels in this form.
    half = _RANK // 2
    tensors["gate_up_lora_a"][1].view(num_experts, 2, _RANK, _HIDDEN)[
        :, :, half:, :
    ] = 0
    tensors["down_lora_a"][1, :, half:, :] = 0
    tensors["gate_up_lora_b"][1, :, :, half:] = 0
    tensors["down_lora_b"][1, :, :, half:] = 0

    scores = torch.rand((num_tokens, num_experts), generator=generator)
    scores[: num_tokens // 2, 0] += 4.0  # one expert gets many more pairs
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


def _into_base_pair():
    """Return the reference plan, the scatter plan, and one launch config.

    Every shipped row with the scatter also uses the b_act middle. The b_act
    tests cover that middle. These two plans differ in the down tail alone.
    The decode fallback row only lends its tuned launch config: its shipped
    plan is the swept winner (indexed gate_up_a plus both overlap windows),
    not the serial shape this comparison needs, so the plans are built by
    hand.
    """
    tiles_row = _menu(_GB300, False)["fallback.decode"]
    assert tiles_row.plan.act.family is ActFamily.MATERIALIZED
    reference_plan = _serial_plan()
    reordered_plan = replace(reference_plan, down_b_into_base=True)
    assert reordered_plan.act.family is ActFamily.MATERIALIZED
    return reference_plan, reordered_plan, _shipped_launch(_GB300, tiles_row)


def _bind_test_menu(runner, plan, launch_config):
    """Publish one plan as the runner's whole phase menu.

    ``run`` resolves from the runner's own ``plans``/``tiles``; a test binds
    its single plan to both phases so the batch's phase flag cannot matter.
    """
    selected = SelectedPlan(key="test", name="test", base_gemm_rows="test", plan=plan)
    tiles = SimpleNamespace(config_for=lambda num_tokens: launch_config)
    runner.plans = {Phase.PREFILL: selected, Phase.DECODE: selected}
    runner.tiles = {Phase.PREFILL: tiles, Phase.DECODE: tiles}


def _build_runner(plan, launch_config, base_gemm_rows: str, gpu, num_experts: int):
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner
    from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

    provider = MoeLoraRunner.select_provider_cls(base_gemm_rows, "cutedsl")(
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
        activation=ActivationFn.SILU,
    )
    runner.validate_plan(plan, base_gemm_rows="test")
    _bind_test_menu(runner, plan, launch_config)
    return runner


def _run_once(runner, gpu, token_lora_mapping, *, use_cuda_graph=False):
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
        is_prefill=True,
    )
    return runner.run(dispatch, batch)


def _workspace_buffer_names(runner) -> set[str]:
    workspace = runner.workspace
    return {key[0] for key in workspace._eager_buffers} | {
        key[0] for key in workspace._graph_buffers
    }


@cutedsl_cuda_only
@pytest.mark.parametrize("base_gemm_rows", ("expert_major", "route_major"))
def test_runner_into_base_matches_the_materialized_reference(
    base_gemm_rows: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare the shipped tail with the scatter on both providers.

    The result must not depend on the row domain.
    """
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    serial_plan, reordered_plan, launch_config = _into_base_pair()
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    reference_runner = _build_runner(
        serial_plan, launch_config, "expert_major", gpu, num_experts
    )
    into_base_runner = _build_runner(
        reordered_plan, launch_config, base_gemm_rows, gpu, num_experts
    )

    for traffic in ("active", "mixed", "base_only"):
        token_lora_mapping = _token_lora_mapping(traffic, num_tokens).to(device)
        reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
        into_base = _run_once(into_base_runner, gpu, token_lora_mapping).hidden_states
        torch.testing.assert_close(
            into_base,
            reference,
            **_INTO_BASE_TOLERANCE,
            msg=f"{base_gemm_rows}: {traffic}",
        )

    # The scatter path never allocates the pair-major delta buffer for down-B.
    # The shipped path always allocates it.
    assert "down_b:delta" in _workspace_buffer_names(reference_runner)
    assert "down_b:delta" not in _workspace_buffer_names(into_base_runner)


@cutedsl_cuda_only
def test_into_base_pipeline_replays_correctly_in_a_real_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    serial_plan, reordered_plan, launch_config = _into_base_pair()
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    reference_runner = _build_runner(
        serial_plan, launch_config, "expert_major", gpu, num_experts
    )
    into_base_runner = _build_runner(
        reordered_plan, launch_config, "expert_major", gpu, num_experts
    )
    token_lora_mapping = _token_lora_mapping("active", num_tokens).to(device)

    for _ in range(2):  # warm the JIT and keep the graph buffers before capture
        _run_once(into_base_runner, gpu, token_lora_mapping, use_cuda_graph=True)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_once(
            into_base_runner, gpu, token_lora_mapping, use_cuda_graph=True
        )
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    # Replay 2: every token in the batch becomes base-only.
    token_lora_mapping.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    # Replay 3: new routing and new activations arrive in the same buffers.
    gpu["topk_ids"].copy_(gpu["topk_ids"].flip(dims=(1,)))
    gpu["hidden_states"].copy_((gpu["hidden_states"].float() * 1.5).to(torch.bfloat16))
    token_lora_mapping.copy_(_token_lora_mapping("mixed", num_tokens).to(device))
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
    torch.testing.assert_close(output, reference, **_INTO_BASE_TOLERANCE)

    assert "down_b:delta" not in _workspace_buffer_names(into_base_runner)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
