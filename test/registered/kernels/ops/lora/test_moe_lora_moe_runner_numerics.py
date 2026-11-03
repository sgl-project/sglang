"""Per-expert MoE LoRA against independent FP32 and staged-BF16 references,
over BF16 and [128, 128]-block FP8 base weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import (
    ActFamily,
    ActivationFn,
    FinalizeFamily,
    MoePlan,
    SelectedPlan,
    resolve_plans,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo, MoeLoraFp8QuantInfo
from sglang.srt.lora.moe.runner import (
    MoeLoraBatch,
    MoeLoraRunner,
)
from sglang.srt.lora.utils import Phase, architecture_for_capability
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# SM100 runs its own plan rows (per-row B and split-K in decode) and base GEMMs.
# The test runs on one GPU: 4-gpu-b200 is the kernel lane's only SM100 pool, so
# its four GPUs are allocation granularity, not multi-GPU coverage.
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _bind_test_menu(runner, plan, launch_config):
    """Bind one plan to both phases to isolate execution from plan selection."""
    tiles = SimpleNamespace(config_for=lambda num_tokens: launch_config)
    selected = SelectedPlan(name="test", base_gemm_rows="test", plan=plan, tiles=tiles)
    runner.plans = {Phase.PREFILL: selected, Phase.DECODE: selected}


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
    not _cutedsl_ready(), reason="the numerical oracle runs the CuTeDSL providers"
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


def _resolve_execution(
    architecture,
    mode: Phase,
    num_tokens: int,
    *,
    hidden_size=_HIDDEN,
    physical_rank=_PHYSICAL_RANK,
    experts=_EXPERTS,
    quant_family="bf16",
):
    selected = resolve_plans(
        quant_family=quant_family,
        architecture=architecture,
        is_shared_outer=False,
        physical_rank=physical_rank,
        activation=ActivationFn.SILU,
        hidden_size=hidden_size,
        num_local_experts=experts,
    )[mode]
    launch_config = selected.tiles.config_for(num_tokens)
    if architecture == "sm100" and mode is Phase.DECODE:
        # Small-rank decode exercises serial split-K; rank 256 must stay unsplit.
        expected_split = 1
        if physical_rank <= 16:
            expected_split = 8 if num_tokens <= 4 else 4 if num_tokens <= 16 else 1
        assert launch_config.gate_up_a.get("SPLIT_K", 1) == expected_split
        assert launch_config.down_a.get("SPLIT_K", 1) == 1
    return selected, launch_config


def _make_cpu_tensors(
    num_tokens: int,
    *,
    hidden_size=_HIDDEN,
    intermediate_size=_INTERMEDIATE,
    physical_rank=_PHYSICAL_RANK,
    experts=_EXPERTS,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0x5A17 + num_tokens)
    # Keep each projection's variance comparable as fan-in and rank grow.
    hidden_scale = (_HIDDEN / hidden_size) ** 0.5
    intermediate_scale = (_INTERMEDIATE / intermediate_size) ** 0.5
    rank_scale = (_PHYSICAL_RANK / physical_rank) ** 0.5
    tensors = {
        "hidden_states": _rand_bf16(
            (num_tokens, hidden_size), generator=generator, scale=0.20
        ),
        "w13_weight": _rand_bf16(
            (experts, 2 * intermediate_size, hidden_size),
            generator=generator,
            scale=0.08 * hidden_scale,
        ),
        "w2_weight": _rand_bf16(
            (experts, hidden_size, intermediate_size),
            generator=generator,
            scale=0.08 * intermediate_scale,
        ),
        "gate_up_lora_a": _rand_bf16(
            (_SLOTS, experts, 2 * physical_rank, hidden_size),
            generator=generator,
            scale=0.15 * hidden_scale,
        ),
        "gate_up_lora_b": _rand_bf16(
            (_SLOTS, experts, 2 * intermediate_size, physical_rank),
            generator=generator,
            scale=0.15 * rank_scale,
        ),
        "down_lora_a": _rand_bf16(
            (_SLOTS, experts, physical_rank, intermediate_size),
            generator=generator,
            scale=0.15 * intermediate_scale,
        ),
        "down_lora_b": _rand_bf16(
            (_SLOTS, experts, hidden_size, physical_rank),
            generator=generator,
            scale=0.15 * rank_scale,
        ),
    }

    alternating_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    tensors["topk_ids"] = alternating_ids.repeat((num_tokens + 1) // 2, 1)[
        :num_tokens
    ].contiguous()
    tensors["topk_weights"] = torch.tensor([[0.65, 0.35]], dtype=torch.float32).repeat(
        num_tokens, 1
    )
    tensors["router_logits"] = torch.zeros((num_tokens, experts), dtype=torch.float32)
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


def _fp8_blocks(weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """[128, 128]-block FP8 checkpoint: (codes, FP32 scales, dequantized FP32)."""
    experts, rows, cols = weight.shape
    blocks = weight.float().view(experts, rows // 128, 128, cols // 128, 128)
    scales = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-6) / 448
    expanded = scales.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    codes = (weight.float() / expanded).to(torch.float8_e4m3fn)
    return codes, scales, codes.float() * expanded


def _fp8_groups(values: torch.Tensor) -> torch.Tensor:
    """Round-trip a base-GEMM input through FP8 with one FP32 scale per 128."""
    groups = values.reshape(-1, 128)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / 448
    codes = (groups / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (codes.float() * scales).view_as(values)


def _reference(
    tensors: dict[str, torch.Tensor],
    token_lora_mapping: torch.Tensor,
    *,
    plan: MoePlan | None = None,
    fp8_base: bool = False,
) -> torch.Tensor:
    """FP32 matvecs, optionally rounded at the selected plan's BF16 buffers.

    No production routing or kernels, nor a backend-dependent BF16 matmul.
    The default retains the full-FP32 oracle for the original small fixtures.
    An FP8 base also quantizes both base-GEMM inputs (the BF16 hidden row and
    the BF16 activation); LoRA reads them unquantized.
    """

    if plan is not None:
        assert plan.finalize.family is FinalizeFamily.MATERIALIZED

    def boundary(value):
        return value.to(torch.bfloat16).float() if plan is not None else value

    def base_input(value):
        return _fp8_groups(value) if fp8_base else value

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
    hidden_size = hidden_states.shape[-1]
    intermediate_size = w2_weight.shape[-1]
    physical_rank = down_lora_a.shape[-2]

    output = torch.zeros((hidden_states.shape[0], hidden_size), dtype=torch.float32)
    for token_idx, hidden in enumerate(hidden_states):
        slot = int(token_lora_mapping[token_idx])
        slot_is_active = slot >= 0 and bool(adapter_enabled[slot])
        for topk_idx in range(_TOP_K):
            expert = int(topk_ids[token_idx, topk_idx])
            gate_up = boundary(torch.mv(w13_weight[expert], base_input(hidden)))
            gate = gate_up[:intermediate_size]
            up = gate_up[intermediate_size:]

            if slot_is_active:
                gate_up_a = gate_up_lora_a[slot, expert]
                gate_up_b = gate_up_lora_b[slot, expert]
                gate_rank = boundary(torch.mv(gate_up_a[:physical_rank], hidden))
                up_rank = boundary(torch.mv(gate_up_a[physical_rank:], hidden))
                gate_delta = torch.mv(gate_up_b[:intermediate_size], gate_rank)
                up_delta = torch.mv(gate_up_b[intermediate_size:], up_rank)
                # Fused B+activation consumes FP32 deltas without a BF16 store.
                if plan is None or plan.act.family is ActFamily.MATERIALIZED:
                    gate_delta, up_delta = boundary(gate_delta), boundary(up_delta)
                gate = gate + gate_delta
                up = up + up_delta

            activation = boundary(F.silu(gate) * up)
            pair_output = boundary(torch.mv(w2_weight[expert], base_input(activation)))
            if slot_is_active:
                down_rank = boundary(torch.mv(down_lora_a[slot, expert], activation))
                down_delta = torch.mv(down_lora_b[slot, expert], down_rank)
                if plan is not None and plan.down_b_into_base:
                    pair_output = boundary(pair_output + down_delta)
                else:
                    pair_output = pair_output + boundary(down_delta)

            output[token_idx].add_(
                pair_output, alpha=float(topk_weights[token_idx, topk_idx])
            )

    return boundary(output * _ROUTED_SCALING)


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
    ("mode", "num_tokens", "hidden_size", "intermediate_size", "physical_rank"),
    (
        (Phase.DECODE, 4, _HIDDEN, _INTERMEDIATE, _PHYSICAL_RANK),
        (Phase.DECODE, 8, _HIDDEN, _INTERMEDIATE, _PHYSICAL_RANK),
        (Phase.PREFILL, 64, _HIDDEN, _INTERMEDIATE, _PHYSICAL_RANK),
        (Phase.DECODE, 4, 4096, 1024, 256),
        (Phase.PREFILL, 8, 4096, 1024, 256),
    ),
    ids=("decode-small", "decode", "lab-prefill", "wide-decode", "wide-prefill"),
)
def test_selected_pipeline_matches_reference_in_eager_and_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
    mode: Phase,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    physical_rank: int,
) -> None:
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (9, 10):  # the table mapper itself never raises
        pytest.skip(f"MoE LoRA does not support SM{capability[0]}{capability[1]}")

    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )

    cpu = _make_cpu_tensors(
        num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        physical_rank=physical_rank,
    )
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}

    choice, launch_config = _resolve_execution(
        architecture_for_capability(capability[0]),
        mode,
        num_tokens,
        hidden_size=hidden_size,
        physical_rank=physical_rank,
    )
    assert choice.base_gemm_rows is not None
    assert choice.plan is not None
    # Preserve the small FP32 cases; wide/rank-256 cases model BF16 stage stores.
    reference_plan = choice.plan if physical_rank == 256 else None

    provider_cls = select_provider_cls(choice.base_gemm_rows, "bf16")
    provider = provider_cls(
        MoeLoraBf16QuantInfo(
            w13_weight=gpu["w13_weight"],
            w2_weight=gpu["w2_weight"],
            num_local_experts=_EXPERTS,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
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
        references[traffic] = _reference(cpu, cpu_slots, plan=reference_plan)
        batch = MoeLoraBatch(
            gate_up_lora_a=gpu["gate_up_lora_a"],
            gate_up_lora_b=gpu["gate_up_lora_b"],
            down_lora_a=gpu["down_lora_a"],
            down_lora_b=gpu["down_lora_b"],
            token_lora_mapping=cpu_slots.to(device),
            use_cuda_graph=False,
            is_prefill=mode is Phase.PREFILL,
        )
        actual = runner.run(dispatch, batch)
        torch.testing.assert_close(
            actual.hidden_states.detach().float().cpu(),
            references[traffic],
            atol=0.018,
            rtol=0.06,
            msg=f"{choice.name}: {traffic} traffic",
        )

    # Make the test sensitive to accidentally bypassing all LoRA math despite
    # the unavoidable BF16-vs-FP32 comparison tolerance above.
    assert (references["active"] - references["base_only"]).abs().max().item() > 0.02

    token_lora_mapping = batch.token_lora_mapping
    token_lora_mapping.copy_(_token_lora_mapping("active", num_tokens).to(device))
    batch.use_cuda_graph = True

    # The workspace keys scratch by stream: warm up on the capture stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # JIT + workspace graph-buffer retention before capture
            runner.run(dispatch, batch)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = runner.run(dispatch, batch)
    output = captured.hidden_states
    output_ptr = output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        output.detach().float().cpu(),
        _reference(cpu, _token_lora_mapping("active", num_tokens), plan=reference_plan),
        atol=0.018,
        rtol=0.06,
        msg=f"{choice.name}: active replay",
    )

    token_lora_mapping.fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert output.data_ptr() == output_ptr
    torch.testing.assert_close(
        output.detach().float().cpu(),
        _reference(
            cpu, _token_lora_mapping("base_only", num_tokens), plan=reference_plan
        ),
        atol=0.018,
        rtol=0.06,
        msg=f"{choice.name}: base-only replay",
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
        _reference(cpu, _token_lora_mapping("mixed", num_tokens), plan=reference_plan),
        atol=0.018,
        rtol=0.06,
        msg=f"{choice.name}: mutated-routing replay",
    )


_FP8_EXPERTS = 4
# GB300 matched the staged reference bit for bit. A CPU sensitivity check
# that adds 2^-14 |W||x| noise to both base GEMMs used at most a quarter of
# this gate; it is not a model of Hopper rounding or a bound.
_FP8_ATOL = 6e-3
_FP8_RTOL = 3e-2


def _check_fp8(actual, cpu, mapping, plan, label):
    reference = _reference(cpu, mapping, plan=plan, fp8_base=True)
    # Dropping LoRA or swapping the two adapters must miss the gate on every
    # adapter row by more than the absolute tolerance.
    rows = mapping >= 0
    for wrong in (torch.full_like(mapping, -1), torch.where(rows, 1 - mapping, -1)):
        wrong_output = _reference(cpu, wrong, plan=plan, fp8_base=True)
        excess = (reference - wrong_output).abs() - (
            _FP8_ATOL + _FP8_RTOL * reference.abs()
        )
        assert excess[rows].amax(dim=1).min().item() > _FP8_ATOL, label
    torch.testing.assert_close(
        actual.detach().float().cpu(),
        reference,
        atol=_FP8_ATOL,
        rtol=_FP8_RTOL,
        msg=label,
    )


def test_fp8_base_pipeline_matches_reference_under_graph_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block-FP8 base weights with both adapters and base rows in one batch:
    eager, replayed, and replayed again after inputs, routes and adapters change."""
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in (9, 10):
        pytest.skip(f"MoE LoRA does not support SM{capability[0]}{capability[1]}")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )

    num_tokens = 6
    cpu = _make_cpu_tensors(num_tokens, experts=_FP8_EXPERTS)
    cpu["adapter_enabled"] = torch.tensor([1, 1], dtype=torch.int32)
    # The provider reads the checkpoint bytes; the reference their dequantization.
    w13_codes, w13_scales, cpu["w13_weight"] = _fp8_blocks(cpu["w13_weight"])
    w2_codes, w2_scales, cpu["w2_weight"] = _fp8_blocks(cpu["w2_weight"])
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}

    choice, launch_config = _resolve_execution(
        architecture_for_capability(capability[0]),
        Phase.DECODE,
        num_tokens,
        experts=_FP8_EXPERTS,
        quant_family="fp8",
    )
    provider = select_provider_cls(choice.base_gemm_rows, "fp8")(
        MoeLoraFp8QuantInfo(
            w13_weight=w13_codes.to(device),
            w13_scale=w13_scales.to(device),
            w2_weight=w2_codes.to(device),
            w2_scale=w2_scales.to(device),
            block_shape=(128, 128),
            num_local_experts=_FP8_EXPERTS,
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
    mapping = torch.tensor([0, -1, 1, 1, -1, 0], dtype=torch.int32)
    batch = MoeLoraBatch(
        gate_up_lora_a=gpu["gate_up_lora_a"],
        gate_up_lora_b=gpu["gate_up_lora_b"],
        down_lora_a=gpu["down_lora_a"],
        down_lora_b=gpu["down_lora_b"],
        token_lora_mapping=mapping.to(device),
    )
    actual = runner.run(dispatch, batch).hidden_states
    _check_fp8(actual, cpu, mapping, choice.plan, f"{choice.name}: eager")

    batch.use_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # JIT + workspace graph-buffer retention before capture
            runner.run(dispatch, batch)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = runner.run(dispatch, batch).hidden_states
    graph.replay()
    torch.cuda.synchronize(device)
    _check_fp8(output, cpu, mapping, choice.plan, f"{choice.name}: replay")

    # New rows and weights, routes onto the experts the capture left empty,
    # and every row on a different adapter or the base.
    cpu["hidden_states"] = _rand_bf16(
        (num_tokens, _HIDDEN), generator=torch.Generator().manual_seed(7), scale=0.3
    )
    cpu["topk_ids"] = torch.tensor(
        [[3, 2], [2, 3], [3, 1], [1, 3], [2, 0], [3, 2]], dtype=torch.int32
    )
    cpu["topk_weights"] = torch.tensor([[0.8, 0.2]]).repeat(num_tokens, 1)
    for name in ("hidden_states", "topk_ids", "topk_weights"):
        gpu[name].copy_(cpu[name])
    mapping = torch.tensor([1, 0, -1, 0, 1, -1], dtype=torch.int32)
    batch.token_lora_mapping.copy_(mapping)
    graph.replay()
    torch.cuda.synchronize(device)
    _check_fp8(output, cpu, mapping, choice.plan, f"{choice.name}: changed replay")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
