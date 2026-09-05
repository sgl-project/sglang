"""Numerics vs a deliberate FP32 reimplementation, never a benchmark kernel;
shared-outer layouts are covered separately at the serving boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.execution_plan import (
    ActivationFn,
    Phase,
    SelectedPlan,
    architecture_for_capability,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.srt.lora.moe.moe_lora_runner import (
    MoeLoraBatch,
    MoeLoraRunner,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.test.ci.ci_register import register_cuda_ci


def _segments_of(mapping: torch.Tensor) -> torch.Tensor:
    """Request boundaries for a test batch: one request per token, so the
    segment route is exercised with many short requests and the helper stays
    free of host syncs (CUDA-graph capture forbids a device-to-host copy)."""
    return torch.arange(mapping.numel() + 1, dtype=torch.int32, device=mapping.device)


register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


def _bind_test_menu(runner, plan, launch_config):
    """Publish one plan as the runner's whole phase menu.

    ``run`` resolves from the runner's own ``plans``/``tiles``; a test binds
    its single plan to both phases so the batch's phase flag cannot matter.
    """
    selected = SelectedPlan(key="test", name="test", base_gemm_rows="test", plan=plan)
    tiles = SimpleNamespace(config_for=lambda num_tokens: launch_config)
    runner.plans = {Phase.PREFILL: selected, Phase.DECODE: selected}
    runner.tiles = {Phase.PREFILL: tiles, Phase.DECODE: tiles}


def _cutedsl_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        return False
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _cutedsl_ready(), reason="the fp32 oracle runs the CuTeDSL providers"
)

_EXPERTS = 2
_TOP_K = 2
_HIDDEN = 128
_INTERMEDIATE = 128
_PHYSICAL_RANK = 16
_SLOTS = 2
_ROUTED_SCALING = 0.75


def _rand_bf16(
    shape: tuple[int, ...], *, generator: torch.Generator, scale: float
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)


def _resolve_execution(architecture, mode: Phase, num_tokens: int):
    selected = resolve_plans(
        architecture=architecture,
        is_shared_outer=False,
        physical_rank=_PHYSICAL_RANK,
        activation=ActivationFn.SILU,
        hidden_size=_HIDDEN,
        num_local_experts=_EXPERTS,
    )[mode]
    launch_config = resolve_tiles(
        architecture_value=architecture.value,
        plan_key_name=selected.name,
        physical_rank=_PHYSICAL_RANK,
    ).config_for(num_tokens)
    return selected, launch_config


def _make_cpu_tensors(num_tokens: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0x5A17 + num_tokens)
    tensors = {
        "hidden_states": _rand_bf16(
            (num_tokens, _HIDDEN), generator=generator, scale=0.20
        ),
        "w13_weight": _rand_bf16(
            (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN),
            generator=generator,
            scale=0.08,
        ),
        "w2_weight": _rand_bf16(
            (_EXPERTS, _HIDDEN, _INTERMEDIATE),
            generator=generator,
            scale=0.08,
        ),
        "gate_up_lora_a": _rand_bf16(
            (_SLOTS, _EXPERTS, 2 * _PHYSICAL_RANK, _HIDDEN),
            generator=generator,
            scale=0.15,
        ),
        "gate_up_lora_b": _rand_bf16(
            (_SLOTS, _EXPERTS, 2 * _INTERMEDIATE, _PHYSICAL_RANK),
            generator=generator,
            scale=0.15,
        ),
        "down_lora_a": _rand_bf16(
            (_SLOTS, _EXPERTS, _PHYSICAL_RANK, _INTERMEDIATE),
            generator=generator,
            scale=0.15,
        ),
        "down_lora_b": _rand_bf16(
            (_SLOTS, _EXPERTS, _HIDDEN, _PHYSICAL_RANK),
            generator=generator,
            scale=0.15,
        ),
    }

    alternating_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    tensors["topk_ids"] = alternating_ids.repeat((num_tokens + 1) // 2, 1)[
        :num_tokens
    ].contiguous()
    tensors["topk_weights"] = torch.tensor([[0.65, 0.35]], dtype=torch.float32).repeat(
        num_tokens, 1
    )
    tensors["router_logits"] = torch.zeros((num_tokens, _EXPERTS), dtype=torch.float32)
    tensors["adapter_enabled"] = torch.tensor([1, 0], dtype=torch.int32)
    return tensors


def _token_lora_mapping(traffic: str, num_tokens: int) -> torch.Tensor:
    if traffic == "active":
        return torch.zeros(num_tokens, dtype=torch.int32)
    if traffic == "mixed":
        # Batch preparation normalizes every inactive resident assignment
        # to -1 before any MoE layer consumes the mapping.
        pattern = torch.tensor([0, -1, -1, 0, -1, -1], dtype=torch.int32)
        return pattern.repeat((num_tokens + pattern.numel() - 1) // pattern.numel())[
            :num_tokens
        ].contiguous()
    if traffic == "base_only":
        # Config choice and graph shape stay fixed while every token follows
        # the base sentinel; one topology serves every batch.
        return torch.full((num_tokens,), -1, dtype=torch.int32)
    raise AssertionError(f"unknown traffic pattern {traffic}")


def _fp32_reference(
    tensors: dict[str, torch.Tensor], token_lora_mapping: torch.Tensor
) -> torch.Tensor:
    """Independent token/expert reference with no production route helpers."""

    hidden_states = tensors["hidden_states"].float()
    w13_weight = tensors["w13_weight"].float()
    w2_weight = tensors["w2_weight"].float()
    gate_up_lora_a = tensors["gate_up_lora_a"].float()
    gate_up_lora_b = tensors["gate_up_lora_b"].float()
    down_lora_a = tensors["down_lora_a"].float()
    down_lora_b = tensors["down_lora_b"].float()
    topk_ids = tensors["topk_ids"]
    topk_weights = tensors["topk_weights"].float()
    adapter_enabled = tensors["adapter_enabled"]

    output = torch.zeros((hidden_states.shape[0], _HIDDEN), dtype=torch.float32)
    for token_idx, hidden in enumerate(hidden_states):
        slot = int(token_lora_mapping[token_idx])
        slot_is_active = slot >= 0 and bool(adapter_enabled[slot])
        for topk_idx in range(_TOP_K):
            expert = int(topk_ids[token_idx, topk_idx])
            gate_up = torch.mv(w13_weight[expert], hidden)
            gate = gate_up[:_INTERMEDIATE]
            up = gate_up[_INTERMEDIATE:]

            if slot_is_active:
                gate_up_a = gate_up_lora_a[slot, expert]
                gate_up_b = gate_up_lora_b[slot, expert]
                gate_rank = torch.mv(gate_up_a[:_PHYSICAL_RANK], hidden)
                up_rank = torch.mv(gate_up_a[_PHYSICAL_RANK:], hidden)
                gate = gate + torch.mv(gate_up_b[:_INTERMEDIATE], gate_rank)
                up = up + torch.mv(gate_up_b[_INTERMEDIATE:], up_rank)

            activation = F.silu(gate) * up
            pair_output = torch.mv(w2_weight[expert], activation)
            if slot_is_active:
                down_rank = torch.mv(down_lora_a[slot, expert], activation)
                pair_output = pair_output + torch.mv(
                    down_lora_b[slot, expert], down_rank
                )

            output[token_idx].add_(
                pair_output, alpha=float(topk_weights[token_idx, topk_idx])
            )

    return output * _ROUTED_SCALING


def _standalone_output_allocation(
    runner: MoeLoraRunner,
    *,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Match eager output geometry without requiring a serving TP group."""

    return torch.empty((num_tokens, runner.hidden_size), dtype=dtype, device=device)


@pytest.mark.parametrize(
    ("mode", "num_tokens"),
    ((Phase.DECODE, 8), (Phase.PREFILL, 64)),
    ids=("decode", "lab-prefill"),
)
def test_config_chosen_per_expert_swiglu_matches_fp32_reference(
    monkeypatch: pytest.MonkeyPatch, mode: Phase, num_tokens: int
) -> None:
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (9, 10):  # the table mapper itself never raises
        pytest.skip(f"MoE LoRA does not support SM{capability[0]}{capability[1]}")

    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )

    cpu = _make_cpu_tensors(num_tokens)
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}

    choice, launch_config = _resolve_execution(
        architecture_for_capability(*capability), mode, num_tokens
    )
    assert choice.base_gemm_rows is not None
    assert choice.plan is not None

    provider_cls = MoeLoraRunner.select_provider_cls(choice.base_gemm_rows, "cutedsl")
    provider = provider_cls(
        MoeLoraBf16QuantInfo(
            w13_weight=gpu["w13_weight"],
            w2_weight=gpu["w2_weight"],
            num_local_experts=_EXPERTS,
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
    runner.validate_plan(choice.plan, base_gemm_rows="test")
    _bind_test_menu(runner, choice.plan, launch_config)

    dispatch = StandardDispatchOutput(
        hidden_states=gpu["hidden_states"],
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=gpu["topk_weights"],
            topk_ids=gpu["topk_ids"],
            router_logits=gpu["router_logits"],
        ),
    )
    references: dict[str, torch.Tensor] = {}
    for traffic in ("active", "mixed", "base_only"):
        cpu_slots = _token_lora_mapping(traffic, num_tokens)
        references[traffic] = _fp32_reference(cpu, cpu_slots)
        batch = MoeLoraBatch(
            gate_up_lora_a=gpu["gate_up_lora_a"],
            gate_up_lora_b=gpu["gate_up_lora_b"],
            down_lora_a=gpu["down_lora_a"],
            down_lora_b=gpu["down_lora_b"],
            token_lora_mapping=cpu_slots.to(device),
            seg_indptr=_segments_of(cpu_slots.to(device)),
            use_cuda_graph=False,
            is_prefill=mode is Phase.PREFILL,
        )
        actual = runner.run(dispatch, batch)
        torch.testing.assert_close(
            actual.hidden_states.detach().float().cpu(),
            references[traffic],
            atol=0.018,
            rtol=0.06,
            msg=f"{choice.key}: {traffic} traffic",
        )

    # Make the test sensitive to accidentally bypassing all LoRA math despite
    # the unavoidable BF16-vs-FP32 comparison tolerance above.
    assert (references["active"] - references["base_only"]).abs().max().item() > 0.02


@pytest.mark.parametrize(
    ("mode", "num_tokens"),
    ((Phase.DECODE, 8), (Phase.PREFILL, 64)),
    ids=("decode-graph", "prefill-graph"),
)
def test_selected_pipeline_replays_correctly_in_a_real_cuda_graph(
    monkeypatch: pytest.MonkeyPatch, mode: Phase, num_tokens: int
) -> None:
    """The graph contract: routes and launches rebuild from graph-stable
    metadata, so in-place batch mutations show through unchanged pointers."""
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (9, 10):  # the table mapper itself never raises
        pytest.skip(f"MoE LoRA does not support SM{capability[0]}{capability[1]}")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )

    cpu = _make_cpu_tensors(num_tokens)
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}
    token_lora_mapping = _token_lora_mapping("active", num_tokens).to(device)

    choice, launch_config = _resolve_execution(
        architecture_for_capability(*capability), mode, num_tokens
    )
    provider = MoeLoraRunner.select_provider_cls(choice.base_gemm_rows, "cutedsl")(
        MoeLoraBf16QuantInfo(
            w13_weight=gpu["w13_weight"],
            w2_weight=gpu["w2_weight"],
            num_local_experts=_EXPERTS,
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
    runner.validate_plan(choice.plan, base_gemm_rows="test")
    _bind_test_menu(runner, choice.plan, launch_config)
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
        use_cuda_graph=True,
        is_prefill=mode is Phase.PREFILL,
    )

    for _ in range(2):  # JIT + workspace graph-buffer retention before capture
        runner.run(dispatch, batch)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.run(dispatch, batch)
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        output.detach().float().cpu(),
        _fp32_reference(cpu, _token_lora_mapping("active", num_tokens)),
        atol=0.018,
        rtol=0.06,
        msg=f"{choice.key}: active replay",
    )

    token_lora_mapping.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    torch.testing.assert_close(
        output.detach().float().cpu(),
        _fp32_reference(cpu, _token_lora_mapping("base_only", num_tokens)),
        atol=0.018,
        rtol=0.06,
        msg=f"{choice.key}: base-only replay",
    )

    cpu["topk_ids"] = cpu["topk_ids"].flip(dims=(1,)).contiguous()
    cpu["hidden_states"] = (cpu["hidden_states"].float() * 1.5).to(torch.bfloat16)
    gpu["topk_ids"].copy_(cpu["topk_ids"])
    gpu["hidden_states"].copy_(cpu["hidden_states"])
    token_lora_mapping.copy_(_token_lora_mapping("mixed", num_tokens).to(device))
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    torch.testing.assert_close(
        output.detach().float().cpu(),
        _fp32_reference(cpu, _token_lora_mapping("mixed", num_tokens)),
        atol=0.018,
        rtol=0.06,
        msg=f"{choice.key}: mutated-routing replay",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
