"""Resident-pool sizing for Paged Experts.

``K`` = the number of experts kept resident on the GPU; the rest are paged from host RAM. K and the KV
cache trade against the same VRAM (the K-slot pool counts as "weights" in sglang's memory model:
``rest = post_load - pre_load*(1 - mem_fraction_static)``), so we size K from sglang's own arithmetic
rather than a magic reserve:

    K = clamp(top_k, E, floor((free_vram*mem_fraction - nonexpert - kv_reserve) / (moe_layers * per_expert_bytes)))

These are **pure, stdlib** helpers. The runtime resolver (``method.py``) gathers the inputs from the live
``ServerArgs`` (the already-derived ``mem_fraction_static``) + ``ModelConfig`` (KV cell size, handling both
the MHA/GQA and MLA branches) and calls ``compute_num_resident_experts`` at ``create_weights`` — that path is
authoritative. K has no power-of-2 / divisibility requirement; it is just the slot count (``top_k <= K <= E``).
"""

from __future__ import annotations


def compute_num_resident_experts(
    *,
    free_vram_bytes: float,
    mem_fraction: float,
    nonexpert_bytes: float,
    kv_reserve_bytes: float,
    moe_layers: int,
    per_expert_layer_bytes: float,
    top_k: int,
    num_experts: int,
) -> int:
    """Largest K that fits, clamped to ``[top_k, num_experts]``.

    ``free_vram_bytes`` is the PRE-load free memory: sglang reserves ``free*(1-mem_fraction)`` for
    activations + cuda graphs and gives the rest to weights + KV, and the K-slot pool is "weights", so
    ``K_pool <= free*mem_fraction - nonexpert - kv_reserve``.

    The K-slot pool and the KV cache compete for the SAME budget, so ``kv_reserve_bytes`` is clamped to
    what's physically left after a minimum (``top_k``) pool. Without that clamp an over-estimated reserve
    (e.g. a high ``--max-running-requests`` x full context) drives the budget negative and floors K to
    ``top_k`` — starving K for a KV pool that can never be allocated (sglang sizes the real KV pool from
    the leftover afterwards, clamped to physical VRAM).
    """
    per_expert_pool = (
        moe_layers * per_expert_layer_bytes
    )  # VRAM for one resident expert (all layers)
    budget = (
        free_vram_bytes * mem_fraction - nonexpert_bytes
    )  # shared by the K-slot pool AND the KV pool
    kv_reserve_bytes = max(0.0, min(kv_reserve_bytes, budget - top_k * per_expert_pool))
    k = int((budget - kv_reserve_bytes) / per_expert_pool)
    return max(top_k, min(num_experts, k))


def compute_window_experts(
    *,
    pin_budget_bytes: float,
    moe_layers: int,
    per_expert_layer_bytes: float,
    num_experts: int,
) -> int:
    """Largest pinned-window size ``W`` (experts page-locked per layer) that fits ``pin_budget_bytes``.

    The window is a *memory* knob expressed as an expert count: pinning ``W`` experts costs
    ``W * moe_layers * per_expert_layer_bytes`` of page-locked host memory. This turns a pin budget into
    that count.

    Returns ``0`` when the whole store fits the budget — i.e. page-lock all ``E`` experts, no cold tier
    (the fast, no-window case; the plain pinned store). Otherwise clamps to ``[1, num_experts - 1]``:
    at least one hot expert, and strictly fewer than ``E`` (``W == E`` is just a full pin, i.e. ``0``).
    """
    per_expert_pool = (
        moe_layers * per_expert_layer_bytes
    )  # page-locked bytes for one expert across all layers
    if per_expert_pool <= 0:
        return 0
    w = int(max(0.0, pin_budget_bytes) / per_expert_pool)
    if w >= num_experts:
        return 0  # whole store fits the budget -> pin all, no window needed
    return max(1, min(num_experts - 1, w))


def kv_reserve_bytes_mha(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: float,
    max_running_requests: int,
    context_length: int,
) -> float:
    """KV reserve for the MHA/GQA case (the resolver carries the MLA branch separately). KV exists for
    ALL attention layers (dense + moe), hence ``num_layers``. ``2`` = K and V."""
    per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
    return max_running_requests * context_length * per_token
