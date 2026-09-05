"""Shared-outer decode against an independent fp32 reference.

The runner-numerics suite covers per-expert layouts only, so shared-outer
decode had no absolute check - equivalence against another engine path can
never catch a shared misconception. This test encodes the layout's factor
contract explicitly: the OUTER factors (gate_up-A on hidden, down-B on
hidden) are one per adapter slot, and the INNER factors (gate_up-B and
down-A on the intermediate dim) stay per expert, exactly as the pool shapes
them. A harness that collapses the inner factors to one per slot reads the
engine's correct (slot, expert) weight fetches as misattribution - that
mistake produced a false bug report once, which is why the shapes above are
spelled out here.

Cases: distinct adapters per slot (the attribution-sensitive one), identical
adapters in every slot (the attribution-blind case every e2e bench runs),
a replay of the pipeline inside a real CUDA graph, and the non-gated relu2
layout (one up projection, S = 1) that Inkling serves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.lora.moe.execution_plan import (
    ActivationFn,
    Phase,
    SelectedPlan,
    architecture_for_capability,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import resolve_tiles
from sglang.srt.lora.moe.moe_lora_runner import MoeLoraBatch, MoeLoraRunner
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci


def _segments_of(mapping: torch.Tensor) -> torch.Tensor:
    """Request boundaries for a test batch: one request per token, so the
    segment route is exercised with many short requests and the helper stays
    free of host syncs (CUDA-graph capture forbids a device-to-host copy)."""
    return torch.arange(mapping.numel() + 1, dtype=torch.int32, device=mapping.device)


register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="production MoE LoRA requires CUDA"
)


@pytest.fixture(autouse=True)
def _published_server_args():
    # The triton vendor's config loader reads the published exec namespace.
    with get_context().override_server_args():
        yield


_EXPERTS = 2
_TOP_K = 2
_HIDDEN = 128
_INTERMEDIATE = 128
_PHYSICAL_RANK = 16
_SLOTS = 2
_ROUTED_SCALING = 0.75
_NUM_TOKENS = 8


def _rand_bf16(shape, *, generator, scale):
    return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)


def _make_cpu_tensors(
    identical_slots: bool = False, gated: bool = True
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0xF01D)
    slices = 2 if gated else 1
    tensors = {
        "hidden_states": _rand_bf16(
            (_NUM_TOKENS, _HIDDEN), generator=generator, scale=0.20
        ),
        "w13_weight": _rand_bf16(
            (_EXPERTS, slices * _INTERMEDIATE, _HIDDEN),
            generator=generator,
            scale=0.08,
        ),
        "w2_weight": _rand_bf16(
            (_EXPERTS, _HIDDEN, _INTERMEDIATE), generator=generator, scale=0.08
        ),
        # Outer factors: one per slot. Inner factors: per (slot, expert).
        "gate_up_lora_a": _rand_bf16(
            (_SLOTS, 1, slices * _PHYSICAL_RANK, _HIDDEN),
            generator=generator,
            scale=0.15,
        ),
        "gate_up_lora_b": _rand_bf16(
            (_SLOTS, _EXPERTS, slices * _INTERMEDIATE, _PHYSICAL_RANK),
            generator=generator,
            scale=0.15,
        ),
        "down_lora_a": _rand_bf16(
            (_SLOTS, _EXPERTS, _PHYSICAL_RANK, _INTERMEDIATE),
            generator=generator,
            scale=0.15,
        ),
        "down_lora_b": _rand_bf16(
            (_SLOTS, 1, _HIDDEN, _PHYSICAL_RANK), generator=generator, scale=0.15
        ),
    }
    alternating = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    tensors["topk_ids"] = alternating.repeat((_NUM_TOKENS + 1) // 2, 1)[
        :_NUM_TOKENS
    ].contiguous()
    tensors["topk_weights"] = torch.tensor([[0.65, 0.35]], dtype=torch.float32).repeat(
        _NUM_TOKENS, 1
    )
    tensors["router_logits"] = torch.zeros((_NUM_TOKENS, _EXPERTS), dtype=torch.float32)
    if identical_slots:
        for name in ("gate_up_lora_a", "gate_up_lora_b", "down_lora_a", "down_lora_b"):
            tensors[name] = (
                tensors[name][:1]
                .repeat(_SLOTS, *([1] * (tensors[name].ndim - 1)))
                .contiguous()
            )
    tensors["adapter_enabled"] = torch.tensor([1, 1], dtype=torch.int32)
    # Mixed traffic: two slots and base-only tokens in one batch.
    tensors["token_lora_mapping"] = torch.tensor(
        [0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.int32
    )
    return tensors


def _fp32_reference(
    tensors: dict[str, torch.Tensor], gated: bool = True
) -> torch.Tensor:
    hidden_states = tensors["hidden_states"].float()
    w13 = tensors["w13_weight"].float()
    w2 = tensors["w2_weight"].float()
    gua = tensors["gate_up_lora_a"].float()
    gub = tensors["gate_up_lora_b"].float()
    da = tensors["down_lora_a"].float()
    db = tensors["down_lora_b"].float()
    topk_ids = tensors["topk_ids"]
    topk_weights = tensors["topk_weights"].float()
    mapping = tensors["token_lora_mapping"]

    output = torch.zeros((_NUM_TOKENS, _HIDDEN), dtype=torch.float32)
    for token_idx, hidden in enumerate(hidden_states):
        slot = int(mapping[token_idx])
        for topk_idx in range(_TOP_K):
            expert = int(topk_ids[token_idx, topk_idx])
            gate_up = torch.mv(w13[expert], hidden)
            if slot >= 0:
                rank = torch.mv(gua[slot, 0], hidden)
                for s in range(2 if gated else 1):
                    cols = slice(s * _INTERMEDIATE, (s + 1) * _INTERMEDIATE)
                    ranks = slice(s * _PHYSICAL_RANK, (s + 1) * _PHYSICAL_RANK)
                    gate_up[cols] += torch.mv(gub[slot, expert, cols], rank[ranks])
            if gated:
                activation = F.silu(gate_up[:_INTERMEDIATE]) * gate_up[_INTERMEDIATE:]
            else:
                activation = F.relu(gate_up).square()
            pair_output = torch.mv(w2[expert], activation)
            if slot >= 0:
                down_rank = torch.mv(da[slot, expert], activation)
                pair_output = pair_output + torch.mv(db[slot, 0], down_rank)
            output[token_idx].add_(
                pair_output, alpha=float(topk_weights[token_idx, topk_idx])
            )
    return output * _ROUTED_SCALING


def _standalone_output_allocation(runner, *, num_tokens, dtype, device):
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


def _build_runner(
    gpu: dict[str, torch.Tensor],
    vendor: str = "cutedsl",
    activation: ActivationFn = ActivationFn.SILU,
    gated: bool = True,
):
    device = torch.device("cuda")
    architecture = architecture_for_capability(
        *torch.cuda.get_device_capability(device)
    )
    choice = resolve_plans(
        quant_family="bf16",
        architecture=architecture,
        is_shared_outer=True,
        physical_rank=_PHYSICAL_RANK,
        activation=activation,
        hidden_size=_HIDDEN,
        num_local_experts=_EXPERTS,
    )[Phase.DECODE]
    launch_config = resolve_tiles(
        architecture_value=architecture.value,
        plan_key_name=choice.name,
        physical_rank=_PHYSICAL_RANK,
    ).config_for(_NUM_TOKENS)
    provider_cls = select_provider_cls(choice.base_gemm_rows, "bf16", vendor)
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
        activation=activation,
        is_gated=gated,
    )
    runner.validate_plan(choice.plan, base_gemm_rows="test")
    _bind_test_menu(runner, choice.plan, launch_config)
    return runner


def _run_once(runner, gpu, use_cuda_graph=False):
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
        token_lora_mapping=gpu["token_lora_mapping"],
        seg_indptr=_segments_of(gpu["token_lora_mapping"]),
        use_cuda_graph=use_cuda_graph,
        is_prefill=False,
    )
    return runner.run(dispatch, batch)


def _skip_unless_supported(device):
    # The runner admits SM90 and SM100 only; the table mapper itself never raises.
    major, minor = torch.cuda.get_device_capability(device)
    if major not in (9, 10):
        pytest.skip(f"MoE LoRA does not support SM{major}{minor}")


@pytest.mark.parametrize("vendor", ("cutedsl", "triton"))
@pytest.mark.parametrize(
    "identical_slots", (False, True), ids=("distinct", "identical")
)
def test_shared_outer_decode_matches_fp32_reference(
    monkeypatch: pytest.MonkeyPatch, identical_slots: bool, vendor: str
) -> None:
    device = torch.device("cuda")
    _skip_unless_supported(device)
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    cpu = _make_cpu_tensors(identical_slots=identical_slots)
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}
    reference = _fp32_reference(cpu)
    runner = _build_runner(gpu, vendor)
    actual = _run_once(runner, gpu).hidden_states.detach().float().cpu()
    torch.testing.assert_close(actual, reference, atol=0.018, rtol=0.06)
    # LoRA math must be present, not bypassed within tolerance.
    base_only = dict(cpu)
    base_only["token_lora_mapping"] = torch.full((_NUM_TOKENS,), -1, dtype=torch.int32)
    assert (reference - _fp32_reference(base_only)).abs().max().item() > 0.02


@pytest.mark.parametrize("vendor", ("cutedsl", "triton"))
def test_shared_outer_decode_replays_in_a_cuda_graph(
    monkeypatch: pytest.MonkeyPatch, vendor: str
) -> None:
    device = torch.device("cuda")
    _skip_unless_supported(device)
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    cpu = _make_cpu_tensors()
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}
    reference = _fp32_reference(cpu)
    runner = _build_runner(gpu, vendor)
    _run_once(runner, gpu, use_cuda_graph=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_once(runner, gpu, use_cuda_graph=True)
    graph.replay()
    torch.cuda.synchronize()
    out = captured.hidden_states.detach().float().cpu()
    torch.testing.assert_close(out, reference, atol=0.018, rtol=0.06)


@pytest.mark.parametrize("vendor", ("cutedsl", "triton"))
def test_shared_outer_relu2_non_gated_decode_matches_fp32_reference(
    monkeypatch: pytest.MonkeyPatch, vendor: str
) -> None:
    """Inkling's layout: one up projection (S = 1) and relu2, real adapters."""
    device = torch.device("cuda")
    _skip_unless_supported(device)
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    cpu = _make_cpu_tensors(gated=False)
    gpu = {name: tensor.to(device) for name, tensor in cpu.items()}
    reference = _fp32_reference(cpu, gated=False)
    runner = _build_runner(gpu, vendor, activation=ActivationFn.RELU2, gated=False)
    actual = _run_once(runner, gpu).hidden_states.detach().float().cpu()
    torch.testing.assert_close(actual, reference, atol=0.018, rtol=0.06)
    base_only = dict(cpu)
    base_only["token_lora_mapping"] = torch.full((_NUM_TOKENS,), -1, dtype=torch.int32)
    assert (
        reference - _fp32_reference(base_only, gated=False)
    ).abs().max().item() > 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
