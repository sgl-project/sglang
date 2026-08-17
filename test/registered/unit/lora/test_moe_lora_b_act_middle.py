"""B_ACTIVATION-middle coverage for the MoE LoRA engine.

The shipped config files serve the B_ACTIVATION middle on every serial
per-expert SWIGLU prefill choice — the GB300 and H200 in-domain prefill
scenarios plus the per-expert out-of-domain prefill fallback: the standalone
one-launch gate/up LoRA-B stage disappears and the route-block-tiled
activation join applies the same per-expert pair-major delta inline, with
the down tail reordered by the composed ``down_b_scatter``.  CPU cases pin
that config structure through the public resolver only (no menu internals).

CUDA cases are runner-level oracles on the DeepGEMM providers: the SAME
random inputs through the serial materialized reference plan versus the
b_act-swapped plan (and versus b_act + scatter, the full shipped
composition) must agree.  The reference plan/config pair is the shipped
decode fallback choice — exactly the serial materialized shape with a
complete tuned config — and the swapped plans are built directly from the
execution-plan spec classes, keeping the SAME launch config so the plan
change is isolated.  Agreement is allclose rather than bitwise BY
JUSTIFICATION: the b_act middle rounds each pair's gate/up delta into the
activation join's FP32 arithmetic instead of materializing a BF16 delta that
the standalone join re-reads — the same rounding class as the shared-rank
and scatter contract notes.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    BridgeLayout,
    DeviceArchitecture,
    FinalizeFamily,
    FinalizeSpec,
    LoraAFamily,
    LoraASpec,
    LoraBFamily,
    LoraBSpec,
    MiddleFamily,
    MiddleSpec,
    MoeLoraExecutionPlan,
    Phase,
    Site,
    StageContract,
    iter_selected_plans,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")

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
    middle_family=MiddleFamily.MATERIALIZED,
    down_b_scatter=False,
) -> MoeLoraExecutionPlan:
    """The serial one-launch plan shape, built the way
    ``execution_plan.build_plan`` materializes a table row (spec classes
    directly)."""
    pe = False
    consumes_gate_up_b = middle_family is MiddleFamily.B_ACTIVATION
    gate_up_b_contract = StageContract(Site.GATE_UP, pe, BridgeLayout.PAIR_MAJOR)
    return MoeLoraExecutionPlan(
        gate_up_a=LoraASpec(
            Site.GATE_UP,
            LoraAFamily.GROUPED,
            is_shared_outer,
            BridgeLayout.PAIR_MAJOR,
        ),
        gate_up_b=(
            None
            if consumes_gate_up_b
            else LoraBSpec(
                Site.GATE_UP,
                LoraBFamily.ONE_LAUNCH_SLICED,
                pe,
                BridgeLayout.PAIR_MAJOR,
            )
        ),
        middle=MiddleSpec(
            middle_family,
            activation,
            gate_up_b_contract if consumes_gate_up_b else None,
        ),
        down_a=LoraASpec(Site.DOWN, LoraAFamily.GROUPED, pe, BridgeLayout.PAIR_MAJOR),
        down_b=LoraBSpec(
            Site.DOWN,
            LoraBFamily.ONE_LAUNCH_SLICED,
            is_shared_outer,
            BridgeLayout.PAIR_MAJOR,
        ),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED),
        down_b_scatter=down_b_scatter,
    ).validate_ownership(is_shared_outer)


# ---- CPU: shipped config structure ----------------------------------------------


class TestBActMiddleConfig:
    def test_h200_serial_prefill_ships_it_too(self) -> None:
        # The H200 serial prefill shape is the same eligible form, on the
        # SM90-capable DeepGEMM contiguous backend.
        serial = _menu(_H200, False)["prefill.serial"]
        assert serial.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert serial.plan.gate_up_b is None
        assert serial.plan.down_b_scatter is True
        assert serial.provider == "deepgemm_contiguous"

    def test_decode_choices_keep_the_materialized_middle(self) -> None:
        # The swap is a prefill-only config: every decode-phase choice on
        # both architectures and layouts keeps the materialized middle with
        # a standalone gate/up-B owner.
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                for activation in (_SWIGLU, ActivationFamily.RELU2):
                    for name, choice in _menu(architecture, layout, activation).items():
                        if not name.startswith("decode.") and name != "fallback.serial":
                            continue
                        assert (
                            choice.plan.middle.family is MiddleFamily.MATERIALIZED
                        ), name
                        assert choice.plan.gate_up_b is not None, name

    def test_out_of_domain_prefill_twins_get_the_swap(self) -> None:
        def _prefill(is_shared_outer, activation=_SWIGLU):
            return resolve_plans(
                architecture=_GB300,
                is_shared_outer=is_shared_outer,
                physical_rank=16,
                activation=activation,
                hidden_size=8192,  # outside the gb300 tuned domain
                num_local_experts=256,
            )[Phase.PREFILL]

        per_expert = _prefill(False)
        shared = _prefill(True)
        relu2 = _prefill(False, ActivationFamily.RELU2)
        for sel in (per_expert, shared, relu2):
            assert sel.name == "fallback.serial_prefill"
        # The per-expert twin composes the swap with the scatter; the shared
        # twin gets the swap only (its down-B is shared, never scattered);
        # rows are activation-agnostic, so the ReLU2 twin ships the same
        # fused middle with its own activation injected.
        assert per_expert.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert per_expert.plan.gate_up_b is None
        assert per_expert.plan.down_b_scatter is True
        assert per_expert.provider == "deepgemm_contiguous"
        assert shared.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert shared.plan.gate_up_b is None
        assert shared.plan.down_b_scatter is False
        assert shared.provider == "deepgemm"
        assert relu2.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert relu2.plan.middle.activation is ActivationFamily.RELU2
        assert relu2.plan.down_b_scatter is True


# ---- CUDA: runner-level oracle ---------------------------------------------------


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

# The b_act middle folds each pair's gate/up delta into the activation
# join's FP32 arithmetic instead of materializing a BF16 delta the join
# re-reads — the same rounding class as the shared-rank and scatter notes.
_B_ACT_TOLERANCE = {"atol": 1e-2, "rtol": 0.05}


def _make_gpu_tensors(num_tokens: int, num_experts: int, device: torch.device):
    generator = torch.Generator().manual_seed(0xB0AC + num_tokens)

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


def _reference_choice():
    """The shipped choice carrying the serial materialized reference shape.

    The decode fallback is exactly that plan with a complete tuned launch
    config (one-launch B tilings, b_activation section, aligned-16 routes),
    so it doubles as the config donor for the swapped variants."""
    choice = _menu(_GB300, False)["fallback.serial"]
    assert choice.plan.middle.family is MiddleFamily.MATERIALIZED
    assert choice.plan.gate_up_b is not None
    assert choice.plan.down_b_scatter is False
    assert choice.plan == _build_plan()
    return choice


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


def _run_once(runner, gpu, token_slots):
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
        use_cuda_graph=False,
        is_prefill=True,
        has_active_lora=True,
    )
    return runner.run(
        dispatch, batch, output_dtype=torch.float32, **runner._test_execution
    )


@deepgemm_cuda_only
@pytest.mark.parametrize("provider_name", ("deepgemm", "deepgemm_contiguous"))
@pytest.mark.parametrize("scatter", (False, True))
def test_runner_b_act_matches_the_materialized_reference(
    provider_name: str, scatter: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One serial prefill batch, serial materialized middle vs the b_act swap
    (and vs b_act + scatter, the full shipped composition) — on the masked
    provider and on the contiguous provider (row-domain agnosticism at the
    seam), under ONE shared launch config so only the plan changes."""
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    reference_choice = _reference_choice()
    swapped_plan = _build_plan(
        middle_family=MiddleFamily.B_ACTIVATION, down_b_scatter=scatter
    )
    assert swapped_plan.middle.family is MiddleFamily.B_ACTIVATION
    assert swapped_plan.gate_up_b is None
    assert swapped_plan.down_b_scatter is scatter
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    shared_launch = _shipped_launch(_GB300, reference_choice)
    reference_runner = _build_runner(
        reference_choice.plan,
        shared_launch,
        "deepgemm",
        gpu,
        num_experts,
    )
    b_act_runner = _build_runner(
        swapped_plan,
        shared_launch,
        provider_name,
        gpu,
        num_experts,
    )

    for traffic in ("active", "mixed", "base_only"):
        token_slots = _token_slots(traffic, num_tokens).to(device)
        reference = _run_once(reference_runner, gpu, token_slots).hidden_states
        b_act = _run_once(b_act_runner, gpu, token_slots).hidden_states
        torch.testing.assert_close(
            b_act,
            reference,
            **_B_ACT_TOLERANCE,
            msg=f"{provider_name}: scatter={scatter} {traffic}",
        )
