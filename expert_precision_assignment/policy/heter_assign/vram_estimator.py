"""VRAM budget estimator for static expert-precision assignment.

Given an HF model config + SLO knobs, returns the number of BF16 expert
slots (``K``) we can afford for heter (dual-precision) experts. Every
term is surfaced so callers can print a breakdown.

The runtime reservation (activations + cuda graph buffers + constant
meta data) is computed by mirroring sglang's own auto-mfs logic in
``ServerArgs._handle_gpu_memory_settings`` (see sglang/srt/server_args.py
around lines 1310–1357). Keeping the formulas aligned means the K we
pick here matches the VRAM sglang actually carves out at runtime.

See docs/superpowers/specs/2026-04-19-static-expert-assignment-design.md
for the formula and rationale.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

BYTES_PER_BF16 = 2
BYTES_PER_INT32 = 4
BYTES_PER_FP16 = 2
INT4_PACK = 8  # 8 × int4 values packed per int32
GIB = 1 << 30
MIB = 1 << 20


@dataclasses.dataclass(frozen=True)
class SLO:
    """Service-level envelope for KV sizing.

    Two sizing modes are supported:

    * **Worst case** (default) — uses ``(max_prompt_len + max_output_len)``
      scaled by ``BudgetKnobs.kv_reserve_frac``. Safe but loose when real
      requests rarely hit the envelope simultaneously.
    * **Amortized / empirical** — when ``mean_total_len`` and
      ``std_total_len`` are set (from ``calib_kv.py``), KV is sized as
      ``mc × (μ + k·σ)``, where ``k = BudgetKnobs.kv_headroom_sigmas``.
      This is a reviewer-friendly tight bound: you can cite empirical
      per-task stats rather than a worst-case guess.
    """
    max_concurrency: int
    max_prompt_len: int
    max_output_len: int
    # Empirical per-request (input + output) token stats. Both must be
    # set to enable amortized mode; ``None`` falls back to worst case.
    mean_total_len: Optional[float] = None
    std_total_len: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class BudgetKnobs:
    """Knobs mirroring sglang's auto-mfs inputs, plus our extra headroom.

    Defaults are tuned for single-GPU Qwen3-30B-A3B on an 80 GB H100/A100
    (sglang's own defaults for gpu_mem ∈ [60 GB, 90 GB]).
    """
    # Worst-case KV: fraction of (mc × (max_in + max_out)) to reserve.
    kv_reserve_frac: float = 0.5
    # Amortized KV: standard deviations added above the empirical mean.
    kv_headroom_sigmas: float = 2.0
    headroom_gb: float = 2.0
    headroom_frac: float = 0.05
    group_size: int = 128  # GPTQ INT4 quantization group size
    # Mirrors of sglang ServerArgs fields that feed reserved_mem.
    chunked_prefill_size: int = 8192
    # If None, ``compute_budget`` falls back to ``slo.max_concurrency`` —
    # the runner passes ``--cuda-graph-max-bs $mc`` so sglang's actual
    # graph pool matches what we predicted.
    cuda_graph_max_bs: Optional[int] = None
    tp_size: int = 1
    pp_size: int = 1
    # Length of the default piecewise-cuda-graph capture list for
    # chunked_prefill_size=8192 (see ServerArgs._generate_piecewise_cuda_graph_tokens).
    num_piecewise_tokens: int = 58


@dataclasses.dataclass
class Budget:
    # Fixed terms
    gpu_vram: int
    non_moe: int
    int4_weights: int
    headroom: int
    # Workload terms
    kv: int
    sglang_reserved: int
    # Output
    bf16_budget: int
    bf16_expert_size: int
    k_heter_experts: int
    k_total_experts: int
    predicted_mfs: float

    def as_breakdown(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.gpu_vram,
            "non_moe_bytes": self.non_moe,
            "int4_weights_bytes": self.int4_weights,
            "headroom_bytes": self.headroom,
            "kv_bytes": self.kv,
            "sglang_reserved_bytes": self.sglang_reserved,
            "bf16_budget_bytes": self.bf16_budget,
            "bf16_expert_size_bytes": self.bf16_expert_size,
            "k_heter_experts": self.k_heter_experts,
            "k_total_experts": self.k_total_experts,
            "predicted_mfs": self.predicted_mfs,
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


def non_moe_bytes(config: Any) -> int:
    """Bytes for embeds + LM head + per-layer attention (Q/K/V/O) + router
    gate + norms, at BF16. Assumes every layer is MoE (true for
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
    q_proj = H * (n_heads * head_dim)
    k_proj = H * (n_kv * head_dim)
    v_proj = H * (n_kv * head_dim)
    o_proj = (n_heads * head_dim) * H
    qk_norm = 2 * head_dim
    layer_norms = 2 * H
    router = n_experts * H
    per_layer = q_proj + k_proj + v_proj + o_proj + qk_norm + layer_norms + router
    final_norm = H

    return (embed + lm_head + per_layer * L + final_norm) * BYTES_PER_BF16


def int4_weights_total_bytes(config: Any, group_size: int) -> int:
    """All experts' INT4 weights (always loaded regardless of heter assignment)."""
    return (
        config.num_experts
        * config.num_hidden_layers
        * int4_expert_bytes(
            config.hidden_size, config.moe_intermediate_size, group_size
        )
    )


def kv_bytes(config: Any, slo: SLO, knobs: "BudgetKnobs") -> int:
    """KV reserve bytes.

    Amortized mode (preferred) when SLO carries empirical per-request
    ``mean_total_len``/``std_total_len`` from ``calib_kv.py``:
        tokens = mc × (μ + k·σ)
    Worst case otherwise:
        tokens = mc × (max_in + max_out) × kv_reserve_frac
    """
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    per_token = (
        2 * kv_heads * head_dim * config.num_hidden_layers * BYTES_PER_BF16
    )
    if slo.mean_total_len is not None and slo.std_total_len is not None:
        per_req = slo.mean_total_len + knobs.kv_headroom_sigmas * slo.std_total_len
        peak_tokens = slo.max_concurrency * per_req
    else:
        peak_tokens = (
            slo.max_concurrency
            * (slo.max_prompt_len + slo.max_output_len)
            * knobs.kv_reserve_frac
        )
    return int(peak_tokens * per_token)


def sglang_reserved_mem_bytes(
    gpu_vram_bytes: int,
    chunked_prefill_size: int,
    cuda_graph_max_bs: int,
    tp_size: int,
    pp_size: int,
    num_piecewise_tokens: int,
) -> int:
    """Mirror of ServerArgs auto-mfs reserved_mem (MB) in server_args.py.

    Assumes: no DP attention, no MLA backend, no speculative decoding,
    piecewise cuda graphs enabled. These match our Qwen3-30B-A3B single-GPU
    setup; any deviation would need to be reflected here.
    """
    reserved_mb = 512.0
    reserved_mb += max(chunked_prefill_size, 2048) * 1.5
    reserved_mb += cuda_graph_max_bs * 2
    reserved_mb += tp_size * pp_size / 8 * 1024
    reserved_mb += num_piecewise_tokens * 8
    # sglang floors reserved at 10 GB on GPUs > 60 GB.
    if gpu_vram_bytes > 60 * GIB:
        reserved_mb = max(reserved_mb, 10 * 1024)
    return int(reserved_mb * MIB)


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
    kv = kv_bytes(config, slo, knobs)

    cg_max_bs = (
        knobs.cuda_graph_max_bs
        if knobs.cuda_graph_max_bs is not None
        else slo.max_concurrency
    )
    reserved = sglang_reserved_mem_bytes(
        gpu_vram_bytes,
        knobs.chunked_prefill_size,
        cg_max_bs,
        knobs.tp_size,
        knobs.pp_size,
        knobs.num_piecewise_tokens,
    )

    bf16_budget = gpu_vram_bytes - non_moe - int4 - headroom - kv - reserved
    bf16_sz = bf16_expert_bytes(config.hidden_size, config.moe_intermediate_size)
    total_experts = config.num_experts * config.num_hidden_layers
    k = max(0, min(total_experts, bf16_budget // bf16_sz))
    predicted_mfs = round((gpu_vram_bytes - reserved) / gpu_vram_bytes, 3)

    return Budget(
        gpu_vram=gpu_vram_bytes,
        non_moe=non_moe,
        int4_weights=int4,
        headroom=headroom,
        kv=kv,
        sglang_reserved=reserved,
        bf16_budget=bf16_budget,
        bf16_expert_size=bf16_sz,
        k_heter_experts=k,
        k_total_experts=total_experts,
        predicted_mfs=predicted_mfs,
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
        f"  - sglang reserved: {gb(b.sglang_reserved)}  (predicted mfs={b.predicted_mfs})",
        f"  = BF16 budget    : {gb(b.bf16_budget)}",
        f"  bf16_expert_size : {b.bf16_expert_size / (1 << 20):.2f} MB",
        f"  K heter experts  : {b.k_heter_experts} / {b.k_total_experts}",
    ])
