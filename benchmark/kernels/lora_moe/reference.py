"""Independent FP32 reference for the BF16 MoE-LoRA semantic pipeline.

This module is the plan §14 Step-1 oracle.  It shares no code with the Triton
kernels or any provider: plain PyTorch, expert-loop, FP32 accumulation, fixed
top-k order.  Canonical semantics (gate-1 freeze):

``y[t] = routed_scaling * sum_k coeff[t,k] * (base_pair + down_delta_pair)``

with the gate/up LoRA delta injected into raw W13 output before activation,
down LoRA-A reading the exact activated value, router coefficient and routed
scaling each applied exactly once, and ``-1`` sentinels contributing nothing.
``route_coeff_precision`` declares whether coefficients are consumed in FP32
or BF16-rounded (provider parity axis; adjudication item A2).
"""

from __future__ import annotations

import msgspec
import torch

from benchmark.kernels.lora_moe.cases import CaseTensors, MoeLoraBenchCase


class PairStageReference(msgspec.Struct, kw_only=True):
    """Pair-major FP32 references at every §4 boundary.

    All tensors use canonical pair order (``pair = token * K + k``) with exact
    zeros at invalid pairs; ``valid_pairs`` marks the semantic domain.
    """

    valid_pairs: torch.Tensor  # [P] bool
    pair_expert: torch.Tensor  # [P] int64 local factor expert, -1 invalid
    pair_adapter: torch.Tensor  # [P] int64 adapter slot, -1 base/invalid
    gate_up_lora_a: torch.Tensor  # [P, slices*R_phys]
    gate_up_base: torch.Tensor  # [P, slices*I_local]
    gate_up_delta: torch.Tensor  # [P, slices*I_local]
    activation: torch.Tensor  # [P, I_local]
    down_base: torch.Tensor  # [P, H]
    down_lora_a: torch.Tensor  # [P, R_phys]
    down_delta: torch.Tensor  # [P, H]


def _resolve_pair_domain(
    case: MoeLoraBenchCase, tensors: CaseTensors
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map declared-domain expert IDs to (valid, local expert, adapter)."""
    ids = tensors.topk_ids.to(torch.int64).reshape(-1)
    factor_map = tensors.routed_expert_to_factor_id
    if factor_map is None:
        expert = ids.clone()
    else:
        table = factor_map.to(torch.int64)
        in_map = (ids >= 0) & (ids < table.numel())
        expert = torch.where(
            in_map, table[ids.clamp(min=0, max=table.numel() - 1)], torch.tensor(-1)
        )
    adapter = (
        tensors.token_lora_mapping.to(torch.int64)
        .reshape(-1, 1)
        .expand(-1, case.top_k)
        .reshape(-1)
    )
    expert_valid = (expert >= 0) & (expert < case.num_experts_local)
    adapter_valid = (adapter >= 0) & (adapter < case.slot_capacity)
    return expert_valid, torch.where(expert_valid, expert, torch.tensor(-1)), torch.where(
        expert_valid & adapter_valid, adapter, torch.tensor(-1)
    )


def _activate(case: MoeLoraBenchCase, gate_up: torch.Tensor) -> torch.Tensor:
    """Apply the declared activation to [rows, slices*I] -> [rows, I]."""
    i_local = case.intermediate_size_local
    if case.expert_form == "gated_two_slice":
        gate = gate_up[:, :i_local]
        up = gate_up[:, i_local:]
        return torch.nn.functional.silu(gate) * up
    activated = torch.relu(gate_up)
    return activated * activated


def reference_pair_stages(
    case: MoeLoraBenchCase, tensors: CaseTensors
) -> PairStageReference:
    """Compute every pair-major boundary in FP32 with an expert loop."""
    slices = 2 if case.expert_form == "gated_two_slice" else 1
    h = case.moe_hidden_size
    i_local = case.intermediate_size_local
    r_phys = case.physical_rank
    num_pairs = case.num_tokens * case.top_k

    expert_valid, pair_expert, pair_adapter = _resolve_pair_domain(case, tensors)
    lora_valid = pair_adapter >= 0
    token_of_pair = torch.arange(num_pairs) // case.top_k

    hidden = tensors.hidden_states.to(torch.float32)
    w13 = tensors.w13.to(torch.float32)
    w2 = tensors.w2.to(torch.float32)
    a_gu = tensors.lora_a_gate_up.to(torch.float32)
    b_gu = tensors.lora_b_gate_up.to(torch.float32)
    a_dn = tensors.lora_a_down.to(torch.float32)
    b_dn = tensors.lora_b_down.to(torch.float32)
    shared_gate = a_gu.shape[1] == 1
    shared_down = b_dn.shape[1] == 1

    out = PairStageReference(
        valid_pairs=expert_valid,
        pair_expert=pair_expert,
        pair_adapter=pair_adapter,
        gate_up_lora_a=torch.zeros(num_pairs, slices * r_phys),
        gate_up_base=torch.zeros(num_pairs, slices * i_local),
        gate_up_delta=torch.zeros(num_pairs, slices * i_local),
        activation=torch.zeros(num_pairs, i_local),
        down_base=torch.zeros(num_pairs, h),
        down_lora_a=torch.zeros(num_pairs, r_phys),
        down_delta=torch.zeros(num_pairs, h),
    )

    for expert in range(case.num_experts_local):
        rows = torch.nonzero(expert_valid & (pair_expert == expert)).reshape(-1)
        if rows.numel() == 0:
            continue
        x = hidden[token_of_pair[rows]]  # [rows, H]
        gate_up_base = x @ w13[expert].T  # [rows, slices*I]

        gate_up_delta = torch.zeros_like(gate_up_base)
        lora_rows = rows[lora_valid[rows]]
        if lora_rows.numel():
            adapters = pair_adapter[lora_rows]
            x_lora = hidden[token_of_pair[lora_rows]]
            a_sel = a_gu[adapters, 0 if shared_gate else expert]  # [n, S*R, H]
            b_sel = b_gu[adapters, expert]  # [n, S*I, R]
            a_out = torch.einsum("nrh,nh->nr", a_sel, x_lora)  # [n, S*R]
            delta = torch.zeros(lora_rows.numel(), slices * i_local)
            for s in range(slices):
                delta[:, s * i_local : (s + 1) * i_local] = torch.einsum(
                    "nir,nr->ni",
                    b_sel[:, s * i_local : (s + 1) * i_local, :],
                    a_out[:, s * r_phys : (s + 1) * r_phys],
                )
            out.gate_up_lora_a[lora_rows] = a_out
            gate_up_delta[lora_valid[rows]] = delta

        activated = _activate(case, gate_up_base + gate_up_delta)
        down_base = activated @ w2[expert].T  # [rows, H]

        down_delta = torch.zeros(rows.numel(), h)
        if lora_rows.numel():
            adapters = pair_adapter[lora_rows]
            act_lora = activated[lora_valid[rows]]
            a_out_dn = torch.einsum(
                "nri,ni->nr", a_dn[adapters, expert], act_lora
            )  # [n, R]
            down_delta[lora_valid[rows]] = torch.einsum(
                "nhr,nr->nh",
                b_dn[adapters, 0 if shared_down else expert],
                a_out_dn,
            )
            out.down_lora_a[lora_rows] = a_out_dn

        out.gate_up_base[rows] = gate_up_base
        out.gate_up_delta[rows] = gate_up_delta
        out.activation[rows] = activated
        out.down_base[rows] = down_base
        out.down_delta[rows] = down_delta

    return out


def _coefficients(case: MoeLoraBenchCase, tensors: CaseTensors) -> torch.Tensor:
    coeff = tensors.topk_weights.to(torch.float32)
    if case.route_coeff_precision == "bf16_rounded":
        return coeff.to(torch.bfloat16).to(torch.float32)
    if case.route_coeff_precision != "fp32":
        raise ValueError(
            f"unknown route coefficient precision {case.route_coeff_precision!r}"
        )
    return coeff


def reference_local_moe(
    case: MoeLoraBenchCase,
    tensors: CaseTensors,
    *,
    include_lora: bool = True,
    stages: PairStageReference | None = None,
) -> torch.Tensor:
    """Token-domain FP32 output at the complete-local-MoE boundary ``[T, H]``.

    ``include_lora=False`` produces the matched base-only control ON THE SAME
    ROUTE, which every signal-gated comparison subtracts.  Note the base-only
    control still activates ``base`` alone (no delta), so it is the true
    provider base path, not the LoRA path with zero factors.
    """
    if include_lora:
        if stages is None:
            stages = reference_pair_stages(case, tensors)
        pair_out = stages.down_base + stages.down_delta
    else:
        pair_out = _base_only_stages(case, tensors).down_base
    coeff = _coefficients(case, tensors).reshape(-1, 1)
    weighted = pair_out * coeff
    tokens = weighted.reshape(case.num_tokens, case.top_k, case.moe_hidden_size)
    return tokens.sum(dim=1) * case.routed_scaling_factor


def _base_only_stages(
    case: MoeLoraBenchCase, tensors: CaseTensors
) -> PairStageReference:
    """Base-only pair stages: identical route, zero LoRA algebra."""
    base_tensors = CaseTensors(
        hidden_states=tensors.hidden_states,
        topk_ids=tensors.topk_ids,
        topk_weights=tensors.topk_weights,
        token_lora_mapping=torch.full_like(tensors.token_lora_mapping, -1),
        routed_expert_to_factor_id=tensors.routed_expert_to_factor_id,
        w13=tensors.w13,
        w2=tensors.w2,
        lora_a_gate_up=tensors.lora_a_gate_up,
        lora_b_gate_up=tensors.lora_b_gate_up,
        lora_a_down=tensors.lora_a_down,
        lora_b_down=tensors.lora_b_down,
    )
    return reference_pair_stages(case, base_tensors)


__all__ = [
    "PairStageReference",
    "reference_local_moe",
    "reference_pair_stages",
]
