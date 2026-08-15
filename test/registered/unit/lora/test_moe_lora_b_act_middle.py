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

from sglang.srt.lora.moe.config import (
    DeviceArchitecture,
    choices_for,
)
from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    FactorContract,
    FactorLayout,
    FactorSite,
    FinalizeFamily,
    FinalizeSpec,
    LoraAFamily,
    LoraASpec,
    LoraBFamily,
    LoraBSpec,
    MiddleFamily,
    MiddleSpec,
    MoeLoraExecutionPlan,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")

_GB300 = DeviceArchitecture.GB300
_H200 = DeviceArchitecture.H200
_SWIGLU = ActivationFamily.SWIGLU


def _choices(architecture, layout, activation=_SWIGLU, hidden_size=2048):
    return {
        choice.key: choice
        for choice in choices_for(
            architecture,
            layout,
            activation,
            hidden_size=hidden_size,
            num_local_experts=256,
        )
    }


def _pick(menu, fragment: str):
    matches = [choice for key, choice in menu.items() if fragment in key]
    assert len(matches) == 1, (fragment, sorted(menu))
    return matches[0]


def _choice(architecture, layout, fragment: str, activation=_SWIGLU):
    return _pick(_choices(architecture, layout, activation), fragment)


def _build_plan(
    *,
    activation=_SWIGLU,
    is_shared_outer=False,
    middle_family=MiddleFamily.MATERIALIZED,
    down_b_scatter=False,
) -> MoeLoraExecutionPlan:
    """The serial one-launch plan shape, built the way config.py's
    ``_build_plan`` materializes a scenario (spec classes directly)."""
    pe = False
    consumes_gate_b = middle_family is MiddleFamily.B_ACTIVATION
    gate_b_contract = FactorContract(FactorSite.GATE_UP, pe, FactorLayout.PAIR_MAJOR)
    return MoeLoraExecutionPlan(
        gate_a=LoraASpec(
            FactorSite.GATE_UP,
            LoraAFamily.GROUPED,
            is_shared_outer,
            FactorLayout.PAIR_MAJOR,
        ),
        gate_b=(
            None
            if consumes_gate_b
            else LoraBSpec(
                FactorSite.GATE_UP,
                LoraBFamily.ONE_LAUNCH_SLICED,
                pe,
                FactorLayout.PAIR_MAJOR,
            )
        ),
        middle=MiddleSpec(
            middle_family, activation, gate_b_contract if consumes_gate_b else None
        ),
        down_a=LoraASpec(
            FactorSite.DOWN, LoraAFamily.GROUPED, pe, FactorLayout.PAIR_MAJOR
        ),
        down_b=LoraBSpec(
            FactorSite.DOWN,
            LoraBFamily.ONE_LAUNCH_SLICED,
            is_shared_outer,
            FactorLayout.PAIR_MAJOR,
        ),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED),
        down_b_scatter=down_b_scatter,
    ).validate_ownership(is_shared_outer)


# ---- CPU: shipped config structure ----------------------------------------------


class TestBActMiddleConfig:
    def test_h200_serial_prefill_ships_it_too(self) -> None:
        # The H200 serial prefill shape is the same eligible form, on the
        # SM90-capable DeepGEMM contiguous backend.
        serial = _choice(_H200, False, ".prefill.serial.")
        assert serial.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert serial.plan.gate_b is None
        assert serial.plan.down_b_scatter is True
        assert serial.provider == "deepgemm_contiguous"

    def test_decode_choices_keep_the_materialized_middle(self) -> None:
        # The swap is a prefill-only config: every decode-phase choice on
        # both architectures and layouts keeps the materialized middle with
        # a standalone gate/up-B owner.
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                for activation in (_SWIGLU, ActivationFamily.RELU2):
                    for key, choice in _choices(
                        architecture, layout, activation
                    ).items():
                        if ".decode." not in key and ".fallback.serial." not in key:
                            continue
                        assert (
                            choice.plan.middle.family is MiddleFamily.MATERIALIZED
                        ), key
                        assert choice.plan.gate_b is not None, key

    def test_out_of_domain_prefill_twins_get_the_swap(self) -> None:
        per_expert = _pick(
            _choices(_GB300, False, hidden_size=8192),
            "fallback.serial_prefill",
        )
        shared = _pick(
            _choices(_GB300, True, hidden_size=8192),
            "fallback.serial_prefill",
        )
        relu2 = _pick(
            _choices(
                _GB300,
                False,
                ActivationFamily.RELU2,
                hidden_size=8192,
            ),
            "fallback.serial_prefill",
        )
        # The per-expert twin composes the swap with the scatter; the shared
        # twin gets the swap only (its down-B is shared, never scattered);
        # the ReLU2 twin keeps the materialized middle (the swap is
        # SWIGLU-only) but still scatters.
        assert per_expert.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert per_expert.plan.gate_b is None
        assert per_expert.plan.down_b_scatter is True
        assert per_expert.provider == "deepgemm_contiguous"
        assert shared.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert shared.plan.gate_b is None
        assert shared.plan.down_b_scatter is False
        assert shared.provider == "deepgemm"
        assert relu2.plan.middle.family is MiddleFamily.MATERIALIZED
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
    return torch.empty(
        (num_tokens, runner.provider.hidden_size), dtype=dtype, device=device
    )


def _reference_choice():
    """The shipped choice carrying the serial materialized reference shape.

    The decode fallback is exactly that plan with a complete tuned launch
    config (one-launch B tilings, b_activation section, aligned-16 routes),
    so it doubles as the config donor for the swapped variants."""
    choice = _pick(_choices(_GB300, False), ".fallback.serial.")
    assert choice.plan.middle.family is MiddleFamily.MATERIALIZED
    assert choice.plan.gate_b is not None
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
        provider=provider,
        top_k=_TOP_K,
        routed_scaling_factor=_ROUTED_SCALING,
        activation=ActivationFamily.SWIGLU,
        execution_plan=plan,
        launch_config=launch_config,
    )
    runner.validate_factors(
        gate_up_lora_a=gpu["gate_up_lora_a"],
        gate_up_lora_b=gpu["gate_up_lora_b"],
        down_lora_a=gpu["down_lora_a"],
        down_lora_b=gpu["down_lora_b"],
        is_shared_outer=False,
    )
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
        physical_rank=_RANK,
        is_shared_outer=False,
        use_cuda_graph=False,
        is_prefill=True,
        has_active_lora=True,
    )
    return runner.run(dispatch, batch, output_dtype=torch.float32)


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
    assert swapped_plan.gate_b is None
    assert swapped_plan.down_b_scatter is scatter
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    reference_runner = _build_runner(
        reference_choice.plan,
        reference_choice.launch_config,
        "deepgemm",
        gpu,
        num_experts,
    )
    b_act_runner = _build_runner(
        swapped_plan,
        reference_choice.launch_config,
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
