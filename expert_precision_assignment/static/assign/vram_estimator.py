"""VRAM budget estimator for static expert-precision assignment.

Given an HF model config + SLO knobs, returns the number of BF16 expert
slots (``K``) we can afford for heter (dual-precision) experts. Every
term is surfaced so callers can print a breakdown.

See docs/superpowers/specs/2026-04-19-static-expert-assignment-design.md
for the formula and rationale.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict

BYTES_PER_BF16 = 2
BYTES_PER_INT32 = 4
BYTES_PER_FP16 = 2
INT4_PACK = 8  # 8 × int4 values packed per int32
GIB = 1 << 30


@dataclasses.dataclass(frozen=True)
class SLO:
    max_concurrency: int
    max_prompt_len: int
    max_output_len: int


@dataclasses.dataclass(frozen=True)
class BudgetKnobs:
    kv_reserve_frac: float = 0.5
    headroom_gb: float = 2.0
    headroom_frac: float = 0.05
    prefill_budget_tokens: int = 16384  # sglang --max-prefill-tokens default
    act_safety: float = 1.5
    group_size: int = 128  # GPTQ INT4 quantization group size


@dataclasses.dataclass
class Budget:
    # Fixed terms
    gpu_vram: int
    non_moe: int
    int4_weights: int
    headroom: int
    # Workload terms
    kv: int
    activations: int
    # Output
    bf16_budget: int
    bf16_expert_size: int
    k_heter_experts: int
    k_total_experts: int

    def as_breakdown(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.gpu_vram,
            "non_moe_bytes": self.non_moe,
            "int4_weights_bytes": self.int4_weights,
            "headroom_bytes": self.headroom,
            "kv_bytes": self.kv,
            "activations_bytes": self.activations,
            "bf16_budget_bytes": self.bf16_budget,
            "bf16_expert_size_bytes": self.bf16_expert_size,
            "k_heter_experts": self.k_heter_experts,
            "k_total_experts": self.k_total_experts,
        }


def int4_expert_bytes(hidden: int, intermediate: int, group_size: int) -> int:
    """INT4 GPTQ storage per MoE expert: qweight + scales + qzeros across
    gate_proj, up_proj, down_proj."""
    def one_proj(k: int, n: int) -> int:
        # qweight: [K/8, N] int32
        qweight = (k // INT4_PACK) * n * BYTES_PER_INT32
        # scales: [K/group_size, N] fp16
        scales = (k // group_size) * n * BYTES_PER_FP16
        # qzeros: [K/group_size, N/8] int32
        qzeros = (k // group_size) * (n // INT4_PACK) * BYTES_PER_INT32
        return qweight + scales + qzeros

    gate = one_proj(hidden, intermediate)
    up = one_proj(hidden, intermediate)
    down = one_proj(intermediate, hidden)
    return gate + up + down


def bf16_expert_bytes(hidden: int, intermediate: int) -> int:
    """BF16 storage per MoE expert: fused w13 [2I, H] + w2 [H, I]."""
    return BYTES_PER_BF16 * 3 * hidden * intermediate


def kv_per_token_bytes(config: Any) -> int:
    """K + V per token, summed over all attention layers, BF16."""
    num_layers = config.num_hidden_layers
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return 2 * kv_heads * head_dim * num_layers * BYTES_PER_BF16


def non_moe_param_count(config: Any) -> int:
    """Count non-MoE parameters: embeds + LM head + per-layer attention
    (Q/K/V/O) + router gate + norms. Assumes every layer is MoE (true for
    Qwen3-30B-A3B: decoder_sparse_step=1, mlp_only_layers=[])."""
    H = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or (H // n_heads)
    tied = getattr(config, "tie_word_embeddings", False)
    n_experts = config.num_experts

    embed = V * H
    lm_head = 0 if tied else V * H

    # Per-layer attention (Q/K/V/O projections, no bias for Qwen3).
    q_proj = H * (n_heads * head_dim)
    k_proj = H * (n_kv * head_dim)
    v_proj = H * (n_kv * head_dim)
    o_proj = (n_heads * head_dim) * H
    # Qwen3 adds per-head Q/K RMSNorm (scalar per head_dim).
    qk_norm = 2 * head_dim
    # 2 RMSNorms per layer (input + post-attn), each [H]
    layer_norms = 2 * H
    # Router gate: [num_experts, H]
    router = n_experts * H

    per_layer = q_proj + k_proj + v_proj + o_proj + qk_norm + layer_norms + router
    final_norm = H

    return embed + lm_head + per_layer * L + final_norm


def non_moe_bytes(config: Any) -> int:
    return non_moe_param_count(config) * BYTES_PER_BF16


def int4_weights_total_bytes(config: Any, group_size: int) -> int:
    """All experts' INT4 weights (always loaded regardless of heter assignment)."""
    return (
        config.num_experts
        * config.num_hidden_layers
        * int4_expert_bytes(
            config.hidden_size, config.moe_intermediate_size, group_size
        )
    )


def activation_bytes(config: Any, prefill_budget_tokens: int, safety: float) -> int:
    """Rough bound: peak prefill hidden-state tensor. Dominant term for
    moderate prefill budgets; far smaller than KV for serving workloads."""
    return int(
        prefill_budget_tokens * config.hidden_size * BYTES_PER_BF16 * safety
    )


def kv_bytes(config: Any, slo: SLO, kv_reserve_frac: float) -> int:
    peak_tokens = slo.max_concurrency * (slo.max_prompt_len + slo.max_output_len)
    return int(kv_reserve_frac * peak_tokens * kv_per_token_bytes(config))


def compute_budget(
    config: Any,
    gpu_vram_bytes: int,
    slo: SLO,
    knobs: BudgetKnobs,
) -> Budget:
    non_moe = non_moe_bytes(config)
    int4 = int4_weights_total_bytes(config, knobs.group_size)
    headroom = max(
        int(knobs.headroom_gb * GIB),
        int(knobs.headroom_frac * gpu_vram_bytes),
    )
    kv = kv_bytes(config, slo, knobs.kv_reserve_frac)
    act = activation_bytes(config, knobs.prefill_budget_tokens, knobs.act_safety)

    bf16_budget = gpu_vram_bytes - non_moe - int4 - headroom - kv - act
    bf16_sz = bf16_expert_bytes(config.hidden_size, config.moe_intermediate_size)
    total_experts = config.num_experts * config.num_hidden_layers
    k = max(0, min(total_experts, bf16_budget // bf16_sz))

    return Budget(
        gpu_vram=gpu_vram_bytes,
        non_moe=non_moe,
        int4_weights=int4,
        headroom=headroom,
        kv=kv,
        activations=act,
        bf16_budget=bf16_budget,
        bf16_expert_size=bf16_sz,
        k_heter_experts=k,
        k_total_experts=total_experts,
    )


def format_budget(b: Budget) -> str:
    def gb(x: int) -> str:
        return f"{x / GIB:7.2f} GB"

    return "\n".join([
        f"  GPU VRAM         : {gb(b.gpu_vram)}",
        f"  - non_moe        : {gb(b.non_moe)}",
        f"  - int4 weights   : {gb(b.int4_weights)}",
        f"  - headroom       : {gb(b.headroom)}",
        f"  - KV (reserved)  : {gb(b.kv)}",
        f"  - activations    : {gb(b.activations)}",
        f"  = BF16 budget    : {gb(b.bf16_budget)}",
        f"  bf16_expert_size : {b.bf16_expert_size / (1 << 20):.2f} MB",
        f"  K heter experts  : {b.k_heter_experts} / {b.k_total_experts}",
    ])
