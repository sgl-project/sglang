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
    activation_reserve_bytes: float | None = None,
    min_kv_pool_bytes: float = 0.0,
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

    ``activation_reserve_bytes`` (aggressive single-stream sizing): when given, it REPLACES the
    ``free*(1-mem_fraction)`` percentage reserve with a FIXED byte reserve for activations + cuda-graph
    pool + fragmentation slack. The percentage scales with card size, not with the bs=1 activation peak,
    so at ``max_running_requests==1`` it over-reserves ~a GB; a measured fixed floor reclaims that for K.
    Only spend it when the caller has gated on bs==1 and an outer boot-time back-off can catch an
    over-estimate as a benign restart.

    ``min_kv_pool_bytes`` floors the reserve so the leftover KV pool always holds at least one prefill
    chunk (+ window): the extend allocator grabs a whole chunk before the ring can free, so a pool below
    one chunk silently fails long prompts. Never floor the budget negative — a caller asking for more K
    than physically fits still bottoms out at ``top_k`` (and boots-loud downstream).
    """
    per_expert_pool = (
        moe_layers * per_expert_layer_bytes
    )  # VRAM for one resident expert (all layers)
    if activation_reserve_bytes is not None:
        # Fixed measured reserve (aggressive, bs=1): spend the percentage headroom down to a real floor.
        budget = free_vram_bytes - activation_reserve_bytes - nonexpert_bytes
    else:
        budget = (
            free_vram_bytes * mem_fraction - nonexpert_bytes
        )  # shared by the K-slot pool AND the KV pool
    # Floor the KV reserve so the real pool holds >= one prefill chunk (+ window); clamp to what's
    # physically left after a minimum (top_k) pool so an over-estimate floors K to top_k, not negative.
    kv_reserve_bytes = max(kv_reserve_bytes, min_kv_pool_bytes)
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


def compute_window_size(
    *,
    context_length: int,
    per_token_bytes: float,
    per_expert_pool_bytes: float,
    top_k: int,
    budget_bytes: float,
    window_cost_mult: float = 2.0,
    page_size: int = 1,
) -> int:
    """KV-streaming window ``W`` (tokens). ``0`` => full-KV (do NOT window).

    Fits/doesn't-fit policy — empirically justified (deep-position A/B) and machine-independent (no bandwidth
    constant, no fitted thresholds). Whenever the WHOLE context KV fits VRAM alongside a minimal (``top_k``)
    expert pool, full-KV strictly beats windowing at every decode position: resident KV is read from HBM
    (~free) while a window's tail is re-streamed from host over PCIe every step, and full-KV's only cost —
    expert page-in — is position-independent. So return ``0`` (let aggressive-K size the pool) whenever
    full-KV fits, regardless of context length.

    Window ONLY when full-KV is INFEASIBLE (context too large for VRAM even at ``top_k`` experts): give the
    leftover after a ``top_k`` expert floor to the device ring (cost ~``window_cost_mult * per_token`` per
    token: ring + pool retention), capped at the context. This is the capacity fallback — slow at depth,
    but the only way to serve a context that cannot be held resident. Page-aligned.
    """
    if per_token_bytes <= 0:
        return 0
    full_kv_cost = top_k * per_expert_pool_bytes + context_length * per_token_bytes
    if full_kv_cost <= budget_bytes:
        return 0  # full-KV fits -> no windowing (best throughput at every position)
    leftover = budget_bytes - top_k * per_expert_pool_bytes
    if leftover <= 0:
        return 0  # can't fit even top_k experts + a window; caller sizes down / boots loud
    w = int(leftover / (window_cost_mult * per_token_bytes))
    w = min(w, context_length)
    if w <= 0:
        return 0
    return max(page_size, (w // page_size) * page_size)


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
