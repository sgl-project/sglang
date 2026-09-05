from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.execution_plan import (
    ActFamily,
    ActivationFn,
    ActSpec,
    BridgeLayout,
    DeviceArchitecture,
    FinalizeFamily,
    FinalizeSpec,
    LoraAFamily,
    LoraASpec,
    LoraBFamily,
    LoraBSpec,
    MoeLoraExecutionPlan,
    Phase,
    SelectedPlan,
    Site,
    build_plan,
    load_plans,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.test.ci.ci_register import register_cuda_ci


def _segments_of(mapping: torch.Tensor) -> torch.Tensor:
    """Request boundaries for a test batch: one request per token, so the
    segment route is exercised with many short requests and the helper stays
    free of host syncs (CUDA-graph capture forbids a device-to-host copy)."""
    return torch.arange(mapping.numel() + 1, dtype=torch.int32, device=mapping.device)


register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-large")

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
    act_family=ActFamily.MATERIALIZED,
    down_b_into_base=False,
) -> MoeLoraExecutionPlan:
    pe = False
    consumes_gate_up_b = act_family is ActFamily.B_ACTIVATION
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
                LoraBFamily.GROUPED,
                pe,
                BridgeLayout.PAIR_MAJOR,
            )
        ),
        act=ActSpec(act_family, activation),
        down_a=LoraASpec(Site.DOWN, LoraAFamily.GROUPED, pe, BridgeLayout.PAIR_MAJOR),
        down_b=LoraBSpec(
            Site.DOWN,
            LoraBFamily.GROUPED,
            is_shared_outer,
            BridgeLayout.PAIR_MAJOR,
        ),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED),
        down_b_into_base=down_b_into_base,
    )


class TestBActConfig:
    def test_h200_serial_prefill_ships_it_too(self) -> None:
        serial = _menu(_H200, False)["prefill.per_expert"]
        assert serial.plan.act.family is ActFamily.B_ACTIVATION
        assert serial.plan.gate_up_b is None
        assert serial.plan.down_b_into_base is True
        assert serial.base_gemm_rows == "route_major"

    def test_decode_choices_keep_the_materialized_act(self) -> None:
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                for activation in (_SWIGLU, ActivationFn.RELU2):
                    for name, choice in _menu(architecture, layout, activation).items():
                        if not name.startswith(("decode.", "fallback.decode")):
                            continue
                        if ".fp8" in name:
                            # The expert-banded fp8 row absorbs gate/up-B into
                            # the activation.
                            assert (
                                choice.plan.act.family is ActFamily.B_ACTIVATION
                            ), name
                            assert choice.plan.gate_up_b is None, name
                            continue
                        assert choice.plan.act.family is ActFamily.MATERIALIZED, name
                        assert choice.plan.gate_up_b is not None, name

    def test_out_of_domain_prefill_rows_get_the_swap(self) -> None:
        def _prefill(is_shared_outer, activation=_SWIGLU):
            return resolve_plans(
                quant_family="bf16",
                architecture=_GB300,
                is_shared_outer=is_shared_outer,
                physical_rank=16,
                activation=activation,
                hidden_size=12288,  # outside the gb300 tuned domain
                num_local_experts=256,
            )[Phase.PREFILL]

        per_expert = _prefill(False)
        shared = _prefill(True)
        relu2 = _prefill(False, ActivationFn.RELU2)
        for sel in (per_expert, relu2):
            assert sel.name == "fallback.prefill.per_expert"
        assert shared.name == "fallback.prefill.shared"
        # The per-expert row adds down-B into the base output; the shared row
        # consumes down-B in its shared_token_delta finalize, like every shared
        # prefill row. A row does not depend on the activation, so the ReLU2
        # row matches.
        assert per_expert.plan.act.family is ActFamily.B_ACTIVATION
        assert per_expert.plan.gate_up_b is None
        assert per_expert.plan.down_b_into_base is True
        assert per_expert.base_gemm_rows == "route_major"
        assert shared.plan.act.family is ActFamily.B_ACTIVATION
        assert shared.plan.gate_up_b is None
        assert shared.plan.down_b is None
        assert shared.plan.finalize.family is FinalizeFamily.SHARED_TOKEN_DELTA
        # Route-major keeps the fallback's scratch bounded by routed pairs;
        # the masked slab domain grows with experts x padded chunk tokens.
        assert shared.base_gemm_rows == "route_major"
        assert relu2.plan.act.family is ActFamily.B_ACTIVATION
        assert relu2.plan.act.activation is ActivationFn.RELU2
        assert relu2.plan.down_b_into_base is True


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

# The b_act middle adds each pair's gate/up delta inside the FP32 activation
# math. The shipped path writes that delta to BF16 first, then reads it back.
# The two paths round at different points, so compare them with a tolerance.
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


def _reference_choice():
    """Return the serial materialized reference with the decode fallback's tiles.

    The decode fallback carries a complete tuned launch config, so the swapped
    plans can share one config. Its plan is the swept winner (indexed gate_up_a
    plus both overlap windows), not the serial shape this reference needs, so
    the plan is built by hand; only the row's tiles are borrowed.
    """
    choice = _menu(_GB300, False)["fallback.decode"]
    assert choice.plan.act.family is ActFamily.MATERIALIZED
    assert choice.plan.down_b_into_base is False
    return SelectedPlan(
        key=choice.key,
        name=choice.name,
        base_gemm_rows=choice.base_gemm_rows,
        plan=_build_plan(),
    )


def _bind_test_menu(runner, plan, launch_config):
    """Publish one plan as the runner's whole phase menu.

    ``run`` resolves from the runner's own ``plans``/``tiles``; a test binds
    its single plan to both phases so the batch's phase flag cannot matter.
    """
    selected = SelectedPlan(key="test", name="test", base_gemm_rows="test", plan=plan)
    tiles = SimpleNamespace(config_for=lambda num_tokens: launch_config)
    runner.plans = {Phase.PREFILL: selected, Phase.DECODE: selected}
    runner.tiles = {selected.key: tiles}


def _build_runner(plan, launch_config, base_gemm_rows: str, gpu, num_experts: int):
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner
    from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

    provider = select_provider_cls(base_gemm_rows, "bf16")(
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


def _run_once(runner, gpu, token_lora_mapping):
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
        use_cuda_graph=False,
        is_prefill=True,
    )
    return runner.run(dispatch, batch)


@cutedsl_cuda_only
@pytest.mark.parametrize("base_gemm_rows", ("expert_major", "route_major"))
@pytest.mark.parametrize("into_base", (False, True))
def test_runner_b_act_matches_the_materialized_reference(
    base_gemm_rows: str, into_base: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare the serial materialized middle with the b_act middle.

    Both runners use the same launch config, so only the plan changes.
    """
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    reference_choice = _reference_choice()
    swapped_plan = _build_plan(
        act_family=ActFamily.B_ACTIVATION, down_b_into_base=into_base
    )
    assert swapped_plan.act.family is ActFamily.B_ACTIVATION
    assert swapped_plan.gate_up_b is None
    assert swapped_plan.down_b_into_base is into_base
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    shared_launch = _shipped_launch(_GB300, reference_choice)
    reference_runner = _build_runner(
        reference_choice.plan,
        shared_launch,
        "expert_major",
        gpu,
        num_experts,
    )
    b_act_runner = _build_runner(
        swapped_plan,
        shared_launch,
        base_gemm_rows,
        gpu,
        num_experts,
    )

    for traffic in ("active", "mixed", "base_only"):
        token_lora_mapping = _token_lora_mapping(traffic, num_tokens).to(device)
        reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
        b_act = _run_once(b_act_runner, gpu, token_lora_mapping).hidden_states
        torch.testing.assert_close(
            b_act,
            reference,
            **_B_ACT_TOLERANCE,
            msg=f"{base_gemm_rows}: into_base={into_base} {traffic}",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
