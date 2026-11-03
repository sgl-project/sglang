"""Shared plans and runner fixtures for the MoE LoRA down-B fold tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.plan import (
    ActFamily,
    ActivationFn,
    ActSpec,
    AFamily,
    ASpec,
    BFamily,
    BridgeLayout,
    BSpec,
    DownOverlap,
    FinalizeFamily,
    FinalizeSpec,
    MoePlan,
    SelectedPlan,
    Site,
    _build_plan,
    _load_plans,
    _tile_table,
)
from sglang.srt.lora.utils import Phase

_GB300 = "sm100"

_SWIGLU = ActivationFn.SILU


def _fallback_decode_tiles():
    """The SM100 table's per-expert fallback.decode tiles, bound at rank 16."""
    table = _load_plans(_GB300)
    row = next(row for row in table.fallback if row.name == "fallback.decode")
    assert row.layout in (None, "per_expert")
    # The borrowed tiles were tuned for a materialized plan, as _make_plan builds.
    plan = _build_plan(row.plan, activation=_SWIGLU, is_shared_outer=False)
    assert plan.act.family is ActFamily.MATERIALIZED
    return _tile_table(row.tiles, physical_rank=16)


def _make_plan(
    *,
    act_family=ActFamily.MATERIALIZED,
    down_overlap=DownOverlap.NONE,
    down_b_into_base=False,
) -> MoePlan:
    """A per-expert plan with grouped A and B stages and a materialized finalize."""
    return MoePlan(
        gate_up_a=ASpec(Site.GATE_UP, AFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR),
        gate_up_b=(
            None
            if act_family is ActFamily.B_ACTIVATION
            else BSpec(Site.GATE_UP, BFamily.GROUPED)
        ),
        act=ActSpec(act_family, _SWIGLU),
        down_a=ASpec(Site.DOWN, AFamily.GROUPED),
        down_b=BSpec(Site.DOWN, BFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED, False),
        down_overlap=down_overlap,
        down_b_into_base=down_b_into_base,
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

# The scatter adds the FP32 delta to the base row, then rounds once. The
# shipped path rounds the delta to BF16 first, then adds. The two paths round
# at different points, so compare them with a tolerance.
_INTO_BASE_TOLERANCE = {"atol": 1e-2, "rtol": 0.05}


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
    """Build plans differing only in the down-B fold, with shared fallback tiles."""
    reference_plan = _make_plan()
    reordered_plan = replace(reference_plan, down_b_into_base=True)
    return reference_plan, reordered_plan, _fallback_decode_tiles().config_for(4096)


def _build_runner(plan, launch_config, base_gemm_rows: str, gpu, num_experts: int):
    from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
    from sglang.srt.lora.moe.runner import MoeLoraRunner

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
    # Bind the same plan to both phases to isolate execution from selection.
    tiles = SimpleNamespace(config_for=lambda num_tokens: launch_config)
    selected = SelectedPlan(name="test", base_gemm_rows="test", plan=plan, tiles=tiles)
    runner.plans = {Phase.PREFILL: selected, Phase.DECODE: selected}
    return runner


def _run_once(runner, gpu, token_lora_mapping, *, use_cuda_graph=False):
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
        is_prefill=True,
    )
    return runner.run(dispatch, batch)


def _workspace_buffer_names(runner) -> set[str]:
    workspace = runner.workspace
    return {key[0] for key in workspace._eager_buffers} | {
        key[0] for key in workspace._graph_storage
    }
