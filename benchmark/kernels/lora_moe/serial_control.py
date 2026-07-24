"""``serial_materialized_control`` — the Step-1 reference execution path.

The smallest complete local BF16 MoE-LoRA pipeline (plan §14 Step 1, §7.11):
production routing + production LoRA kernels + a naive expert-loop base GEMM,
every boundary materialized, strictly serial, deterministic.  It exists so
every future candidate can be compared at the complete-local-MoE boundary
against a readable path; it is not a performance candidate.

No new kernels: LoRA math uses the production ``sgl_lora`` primitives
(`grouped_lora_a`, `stock_grouped_lora_b`, `token_owned_lora_b_add`); the base
GEMMs are plain BF16 ``torch.matmul`` per expert (FP32 accumulation).

Coefficient semantics: the delta path goes through ``token_owned_lora_b_add``,
which BF16-rounds route coefficients (FlashInfer parity).  Cases run through
this control therefore declare ``route_coeff_precision="bf16_rounded"``; the
control applies the same rounding to the base combine so the coefficient is
consumed at one declared precision, and folds ``routed_scaling_factor`` into
the delta coefficients so scaling is applied exactly once.
"""

from __future__ import annotations

import msgspec
import torch

from benchmark.kernels.lora_moe.cases import CaseTensors, MoeLoraBenchCase
from sglang.srt.lora.sgl_lora.bf16 import (
    grouped_lora_a,
    stock_grouped_lora_b,
    token_owned_lora_b_add,
)
from sglang.srt.lora.sgl_lora.routing import (
    VirtualExpertRouting,
    build_virtual_expert_routing,
)

# The production provisional tiles from lora/layers.py; correctness-only here.
LORA_A_CONFIG = {
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}
LORA_B_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


_OUTPUT_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _device_tensors(tensors: CaseTensors, device: torch.device) -> CaseTensors:
    moved = {}
    for name in tensors.__struct_fields__:
        value = getattr(tensors, name)
        moved[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return CaseTensors(**moved)


def _expert_routing(
    case: MoeLoraBenchCase, tensors: CaseTensors
) -> VirtualExpertRouting:
    """Per-expert factor route over the declared expert-ID domain."""
    return build_virtual_expert_routing(
        tensors.topk_ids,
        tensors.token_lora_mapping,
        factor_expert_count=case.num_experts_local,
        max_loras=case.slot_capacity,
        block_size=case.routing_block_size,
        routed_expert_to_factor_id=tensors.routed_expert_to_factor_id,
    )


def _shared_routing(
    case: MoeLoraBenchCase, tensors: CaseTensors
) -> VirtualExpertRouting:
    """Adapter-only factor route (shared-outer control form).

    EP ownership is preserved: every locally owned expert maps to shared
    factor 0; non-owned IDs stay ``-1``.
    """
    if tensors.routed_expert_to_factor_id is None:
        domain = case.num_experts_local
        owned = torch.arange(domain, device=tensors.topk_ids.device)
        shared_map = torch.zeros(domain, dtype=torch.int32, device=owned.device)
    else:
        base_map = tensors.routed_expert_to_factor_id
        shared_map = torch.where(
            base_map >= 0, torch.zeros_like(base_map), base_map
        )
    return build_virtual_expert_routing(
        tensors.topk_ids,
        tensors.token_lora_mapping,
        factor_expert_count=1,
        max_loras=case.slot_capacity,
        block_size=case.routing_block_size,
        routed_expert_to_factor_id=shared_map,
    )


def _pair_expert_ids(
    case: MoeLoraBenchCase, tensors: CaseTensors
) -> torch.Tensor:
    """Local expert per pair (``-1`` non-owned), independent of adapters."""
    ids = tensors.topk_ids.to(torch.int64).reshape(-1)
    factor_map = tensors.routed_expert_to_factor_id
    if factor_map is None:
        return ids
    table = factor_map.to(torch.int64)
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
    if case.route_coeff_precision != "bf16_rounded":
        raise ValueError(
            "the serial control consumes coefficients at bf16_rounded "
            "precision (token_owned_lora_b_add contract); declare the case "
            "accordingly"
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

    # Combine: coefficient consumed once at bf16_rounded precision, routed
    # scaling folded exactly once (base directly, delta via coefficients).
    coeff = data.topk_weights.to(torch.bfloat16).to(torch.float32)
    weighted_base = down_base.to(torch.float32) * coeff.reshape(-1, 1)
    output_dtype = _OUTPUT_DTYPES[case.output_dtype]
    output = (
        weighted_base.view(case.num_tokens, case.top_k, case.moe_hidden_size).sum(
            dim=1
        )
        * case.routed_scaling_factor
    ).to(output_dtype)

    token_owned_lora_b_add(
        down_a_out,
        data.lora_b_down.flatten(0, 1),
        output,
        down_b_route,
        data.topk_weights * case.routed_scaling_factor,
        config=LORA_B_CONFIG,
    )
    return SerialControlResult(
        output=output,
        gate_up_lora_a=gate_a_out,
        gate_up_delta=gate_up_delta,
        down_lora_a=down_a_out,
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
    coeff = data.topk_weights.to(torch.bfloat16).to(torch.float32)
    weighted = down_base.to(torch.float32) * coeff.reshape(-1, 1)
    return (
        weighted.view(case.num_tokens, case.top_k, case.moe_hidden_size).sum(dim=1)
        * case.routed_scaling_factor
    ).to(_OUTPUT_DTYPES[case.output_dtype])


__all__ = [
    "LORA_A_CONFIG",
    "LORA_B_CONFIG",
    "SerialControlResult",
    "run_base_only_torch",
    "run_serial_materialized_control",
]
