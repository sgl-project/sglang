"""``serial_materialized_control`` — the Step-1 reference execution path.

The smallest complete local BF16 MoE-LoRA pipeline (plan §14 Step 1, §7.11):
production routing + production LoRA kernels + a naive expert-loop base GEMM,
every boundary materialized, strictly serial, deterministic.  It exists so
every future candidate can be compared at the complete-local-MoE boundary
against a readable path; it is not a performance candidate.

No new kernels: LoRA math uses the production ``sgl_lora`` primitives
(`grouped_lora_a`, `stock_grouped_lora_b` for gate/up AND the materialized
down-B delta); the base GEMMs are plain BF16 ``torch.matmul`` per expert
(FP32 accumulation) and the combine is plain fixed-order FP32 torch.

Coefficient semantics: the combine consumes FP32 coefficients
(``route_coeff_precision="fp32"``, the A2 ruling of plan section 48 — the form
every production backend computes) at exactly one
place — ``y = routed_scaling * sum_k fp32(w[t,k]) * (base_pair + delta_pair)``
computed in FP32 with a literal left-to-right slot loop — so the frozen
equation holds exactly, including non-unit routed scaling.  Declared rounding
order (A1): ``s x BF16(w)`` — weight rounded first, scaling applied at the
combine; the ``BF16(s x w)`` pre-fold is a different lowering (Step-2 axis).
"""

from __future__ import annotations

import msgspec
import torch

from benchmark.kernels.lora_moe.cases import CaseTensors, MoeLoraBenchCase
from sglang.srt.lora.sgl_lora.bf16 import (
    grouped_lora_a,
    stock_grouped_lora_b,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    RouteView,
    build_virtual_expert_routing,
)

# Single source of truth: the same provisional tiles serving uses, so the lab
# control and the production runner cannot drift apart.
LORA_A_CONFIG = PROVISIONAL_LAUNCH_CONFIG.lora_a
LORA_B_CONFIG = PROVISIONAL_LAUNCH_CONFIG.lora_b


_OUTPUT_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _device_tensors(tensors: CaseTensors, device: torch.device) -> CaseTensors:
    moved = {}
    for name in tensors.__struct_fields__:
        value = getattr(tensors, name)
        moved[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return CaseTensors(**moved)


def _expert_routing(case: MoeLoraBenchCase, tensors: CaseTensors) -> RouteView:
    """Per-expert factor route over the declared expert-ID domain."""
    return build_virtual_expert_routing(
        tensors.topk_ids,
        tensors.token_lora_mapping,
        lora_experts_per_adapter=case.num_experts_local,
        max_loras=case.slot_capacity,
        block_size=case.routing_block_size,
        lora_expert_map=tensors.lora_expert_map,
    )


def _shared_routing(case: MoeLoraBenchCase, tensors: CaseTensors) -> RouteView:
    """Adapter-only factor route (shared-outer, the section 60.5 form).

    Global-domain ids are LOCALIZED FIRST — production's convention, where
    the dispatcher hands the runner local ids — and every owned expert then
    maps to the adapter's single LoRA expert (id 0) via the constexpr form; no map tensor
    exists on the shared route anymore.
    """
    topk_ids = tensors.topk_ids
    table = tensors.lora_expert_map
    if table is not None:
        in_map = (topk_ids >= 0) & (topk_ids < table.numel())
        localized = table[topk_ids.clamp(min=0, max=table.numel() - 1).long()]
        topk_ids = torch.where(
            in_map, localized.to(topk_ids.dtype), torch.full_like(topk_ids, -1)
        )
    return build_virtual_expert_routing(
        topk_ids,
        tensors.token_lora_mapping,
        lora_experts_per_adapter=1,
        max_loras=case.slot_capacity,
        block_size=case.routing_block_size,
        shared_outer_local_expert_count=case.num_experts_local,
    )


def _pair_expert_ids(case: MoeLoraBenchCase, tensors: CaseTensors) -> torch.Tensor:
    """Local expert per pair (``-1`` non-owned), independent of adapters."""
    ids = tensors.topk_ids.to(torch.int64).reshape(-1)
    lora_expert_map = tensors.lora_expert_map
    if lora_expert_map is None:
        return ids
    table = lora_expert_map.to(torch.int64)
    in_map = (ids >= 0) & (ids < table.numel())
    return torch.where(
        in_map,
        table[ids.clamp(min=0, max=table.numel() - 1)],
        torch.full_like(ids, -1),
    )


def _base_gemm(
    rows_by_expert: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """Naive per-expert BF16 GEMM with zeros at non-owned pairs."""
    output = torch.zeros(
        (inputs.shape[0], out_features), dtype=torch.bfloat16, device=inputs.device
    )
    for expert in range(weights.shape[0]):
        rows = torch.nonzero(rows_by_expert == expert).reshape(-1)
        if rows.numel():
            output[rows] = inputs[rows] @ weights[expert].T
    return output


def _activate(case: MoeLoraBenchCase, gate_up: torch.Tensor) -> torch.Tensor:
    i_local = case.intermediate_size_local
    if case.expert_form == "gated_two_slice":
        gate = gate_up[:, :i_local].to(torch.float32)
        up = gate_up[:, i_local:].to(torch.float32)
        return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    activated = torch.relu(gate_up.to(torch.float32))
    return (activated * activated).to(torch.bfloat16)


class SerialControlResult(msgspec.Struct, kw_only=True):
    """Complete-local-MoE output plus the materialized boundaries.

    Intermediates are pair-major; rows at sentinel pairs are UNDEFINED in the
    LoRA-A outputs (poisonable) and exact zero in the delta buffer.
    """

    output: torch.Tensor  # [T, H] case output dtype
    gate_up_lora_a: torch.Tensor  # [P, slices*R_phys] BF16
    gate_up_delta: torch.Tensor  # [P, slices*I_local] BF16
    down_lora_a: torch.Tensor  # [P, R_phys] BF16
    down_delta: torch.Tensor  # [P, H] BF16 (zero at sentinel pairs)


def run_serial_materialized_control(
    case: MoeLoraBenchCase,
    tensors: CaseTensors,
    *,
    device: torch.device,
    poison_workspaces: bool = False,
) -> SerialControlResult:
    """Run the complete local MoE and return output plus boundaries.

    Output dtype follows ``case.output_dtype`` (bfloat16 or float32).  With
    ``poison_workspaces`` the pair-major LoRA workspaces start NaN-filled, so
    any read of an undefined sentinel row poisons the final output and is
    caught by ``require_finite``.
    """
    if device.type != "cuda":
        raise ValueError(
            "the serial control exercises the production Triton kernels and "
            "requires a CUDA device; use reference_local_moe on CPU"
        )
    unsupported = {
        "base_provider": (case.base_provider, "reference_loop"),
        "execution_mode": (case.execution_mode, "eager"),
        "overlap_strategy": (
            case.overlap_strategy,
            "serial_materialized_control",
        ),
        "provider_gate_up_layout": (
            case.provider_gate_up_layout,
            "gate_then_up",
        ),
        "route_coeff_precision": (case.route_coeff_precision, "fp32"),
        "cache_state": (case.cache_state, "hot"),
        "intermediate_size_physical": (
            case.intermediate_size_physical,
            case.intermediate_size_local,
        ),
        # K1 axes: the control materializes BF16 bridges; staged-rounding
        # arms get their execution with P6 (plan §63.1) and must not
        # silently ride this path.
        "bridge_gate_a_out": (case.bridge_gate_a_out, "bf16"),
        "bridge_gate_up_delta": (case.bridge_gate_up_delta, "bf16"),
        "bridge_activation_lora_input": (
            case.bridge_activation_lora_input,
            "bf16",
        ),
        "bridge_down_a_out": (case.bridge_down_a_out, "bf16"),
        "bridge_down_delta": (case.bridge_down_delta, "bf16"),
    }
    for field_name, (declared, supported) in unsupported.items():
        if declared != supported:
            raise ValueError(
                f"serial_materialized_control executes {field_name}="
                f"{supported!r}; the case declares {declared!r} — declare "
                "what actually runs or use the matching runner"
            )
    if case.routing_block_size != LORA_B_CONFIG["BLOCK_SIZE_M"]:
        raise ValueError(
            "stock_grouped_lora_b requires BLOCK_SIZE_M == routing block size"
        )
    data = _device_tensors(tensors, device)
    slices = 2 if case.expert_form == "gated_two_slice" else 1
    i_local = case.intermediate_size_local
    r_phys = case.physical_rank
    num_pairs = case.num_tokens * case.top_k
    shared_gate = case.shared_factor_signature in ("shared_gate_up_a", "shared_both")
    shared_down = case.shared_factor_signature in ("shared_down_b", "shared_both")

    expert_route = _expert_routing(case, data)
    gate_a_route = _shared_routing(case, data) if shared_gate else expert_route
    down_b_route = _shared_routing(case, data) if shared_down else expert_route
    pair_expert = _pair_expert_ids(case, data)

    def workspace(*shape: int) -> torch.Tensor:
        buffer = torch.empty(shape, dtype=torch.bfloat16, device=device)
        if poison_workspaces:
            buffer.fill_(float("nan"))
        return buffer

    # Gate/up LoRA A: token-major hidden -> pair-major [P, slices*R_phys].
    gate_a_out = workspace(num_pairs, slices * r_phys)
    grouped_lora_a(
        data.hidden_states,
        data.lora_a_gate_up.flatten(0, 1),
        gate_a_out,
        gate_a_route,
        config=LORA_A_CONFIG,
    )

    # Base W13 plus materialized gate/up delta, then activation.
    gate_up_base = _base_gemm(
        pair_expert,
        data.hidden_states[torch.arange(num_pairs, device=device) // case.top_k],
        data.w13,
        slices * i_local,
    )
    gate_up_delta = workspace(*gate_up_base.shape)
    stock_grouped_lora_b(
        gate_a_out,
        data.lora_b_gate_up.flatten(0, 1),
        gate_up_delta,
        expert_route,
        destination_offsets=(0, i_local) if slices == 2 else (0,),
        config=LORA_B_CONFIG,
    )
    activated = _activate(case, gate_up_base + gate_up_delta)

    # Base W2 and materialized down LoRA A (pair-major input).
    down_base = _base_gemm(pair_expert, activated, data.w2, case.moe_hidden_size)
    down_a_out = workspace(num_pairs, r_phys)
    grouped_lora_a(
        activated,
        data.lora_a_down.flatten(0, 1),
        down_a_out,
        expert_route,
        config=LORA_A_CONFIG,
        pair_input=True,
    )

    # Materialized down-B delta in canonical pair order (zero at sentinels).
    down_delta = workspace(num_pairs, case.moe_hidden_size)
    stock_grouped_lora_b(
        down_a_out,
        data.lora_b_down.flatten(0, 1),
        down_delta,
        down_b_route,
        destination_offsets=(0,),
        config=LORA_B_CONFIG,
    )

    # Combine: the frozen equation, exactly —
    #   y = routed_scaling * sum_k fp32(w) * (base_pair + delta_pair)
    # in FP32 with fixed slot order; coefficient and scaling applied once.
    coeff = data.topk_weights.to(torch.float32)
    pair_sum = (
        (down_base.to(torch.float32) + down_delta.to(torch.float32))
        * coeff.reshape(-1, 1)
    ).view(case.num_tokens, case.top_k, case.moe_hidden_size)
    output_dtype = _OUTPUT_DTYPES[case.output_dtype]
    accumulator = torch.zeros(
        case.num_tokens,
        case.moe_hidden_size,
        dtype=torch.float32,
        device=device,
    )
    for slot in range(case.top_k):
        accumulator = accumulator + pair_sum[:, slot]
    output = (accumulator * case.routed_scaling_factor).to(output_dtype)
    return SerialControlResult(
        output=output,
        gate_up_lora_a=gate_a_out,
        gate_up_delta=gate_up_delta,
        down_lora_a=down_a_out,
        down_delta=down_delta,
    )


def run_base_only_torch(
    case: MoeLoraBenchCase,
    tensors: CaseTensors,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Pure-base pipeline with the identical torch ops and no LoRA launches.

    Used as the bitwise zero-LoRA parity oracle: the serial control on a
    base-only batch must equal this exactly, proving the always-on LoRA
    launches contribute exact zeros.
    """
    data = _device_tensors(tensors, device)
    slices = 2 if case.expert_form == "gated_two_slice" else 1
    pair_expert = _pair_expert_ids(case, data)
    num_pairs = case.num_tokens * case.top_k
    gate_up_base = _base_gemm(
        pair_expert,
        data.hidden_states[torch.arange(num_pairs, device=device) // case.top_k],
        data.w13,
        slices * case.intermediate_size_local,
    )
    activated = _activate(case, gate_up_base)
    down_base = _base_gemm(pair_expert, activated, data.w2, case.moe_hidden_size)
    coeff = data.topk_weights.to(torch.float32)
    weighted = (down_base.to(torch.float32) * coeff.reshape(-1, 1)).view(
        case.num_tokens, case.top_k, case.moe_hidden_size
    )
    accumulator = torch.zeros(
        case.num_tokens,
        case.moe_hidden_size,
        dtype=torch.float32,
        device=device,
    )
    for slot in range(case.top_k):
        accumulator = accumulator + weighted[:, slot]
    return (accumulator * case.routed_scaling_factor).to(
        _OUTPUT_DTYPES[case.output_dtype]
    )
