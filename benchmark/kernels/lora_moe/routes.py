"""Route and adapter-assignment generators for MoE-LoRA benchmark cases.

Every generator is seeded and deterministic, returns canonical separate
``topk_ids`` / ``topk_weights`` tensors (plan §7.1), and never repeats an
expert within one token's top-k.  Route statistics are resolved host-side for
the case record; nothing here runs in a serving path.
"""

from __future__ import annotations

import msgspec
import torch

ROUTE_GENERATORS = (
    "balanced",
    "iid",
    "iid_with_sentinels",
    "hotset_80_20",
    "one_hot",
    "no_local",
)

WEIGHT_DISTRIBUTIONS = ("equal", "seeded_random")


class RouteStats(msgspec.Struct, frozen=True, kw_only=True):
    """Host-resolved statistics of one materialized route (plan §5 symbols).

    ``p_valid_routed``/``e_hit_routed`` cover every locally owned pair
    regardless of adapter (the base-work domain); the unsuffixed fields cover
    LoRA work pairs (owned route AND active adapter).
    """

    num_tokens: int
    top_k: int
    p_valid_routed: int
    e_hit_routed: int
    p_valid: int
    p_aligned: int
    e_hit: int
    group_count: int
    group_size_histogram: dict[int, int]


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def generate_topk_ids(
    *,
    route_generator: str,
    num_tokens: int,
    top_k: int,
    num_routable_experts: int,
    num_local_experts: int,
    owned_expert_start: int = 0,
    seed: int,
) -> torch.Tensor:
    """Generate ``[T, K]`` int32 expert IDs with distinct experts per token.

    ``no_local`` draws only experts outside the locally owned interval
    ``[owned_expert_start, owned_expert_start + num_local_experts)`` so the
    rank's LoRA expert map resolves every pair to the ``-1`` sentinel — correct
    for any EP rank, not only rank 0.  ``iid_with_sentinels`` additionally
    replaces a seeded quarter of the pairs with literal ``-1`` inputs, the
    standard-dispatcher form for non-owned routes in the local ID domain.
    """
    if top_k > num_routable_experts:
        raise ValueError("top_k exceeds the routable expert count")
    generator = _generator(seed)

    if route_generator == "balanced":
        # Round-robin so every expert receives an equal pair count where
        # divisible; consecutive K ids per token are distinct by construction.
        flat = (
            torch.arange(num_tokens * top_k, dtype=torch.int64) % num_routable_experts
        )
        ids = flat.view(num_tokens, top_k)
        if num_routable_experts >= top_k:
            return ids.to(torch.int32)
        raise ValueError("balanced route requires num_routable_experts >= top_k")

    if route_generator == "iid":
        scores = torch.rand((num_tokens, num_routable_experts), generator=generator)
        return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)

    if route_generator == "hotset_80_20":
        hot_count = max(1, num_routable_experts // 5)
        scores = torch.rand((num_tokens, num_routable_experts), generator=generator)
        scores[:, :hot_count] += 4.0 * torch.rand(
            (num_tokens, hot_count), generator=generator
        )
        return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)

    if route_generator == "one_hot":
        # Every token routes to the same K experts: maximal fragmentation of
        # zero and maximal group concentration.
        ids = torch.arange(top_k, dtype=torch.int32)
        return ids.expand(num_tokens, top_k).contiguous()

    if route_generator == "iid_with_sentinels":
        scores = torch.rand((num_tokens, num_routable_experts), generator=generator)
        ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
        sentinel = torch.rand((num_tokens, top_k), generator=generator) < 0.25
        return torch.where(sentinel, torch.full_like(ids, -1), ids)

    if route_generator == "no_local":
        owned_end = owned_expert_start + num_local_experts
        if owned_expert_start < 0 or owned_end > num_routable_experts:
            raise ValueError("owned interval must lie inside the routable domain")
        if num_routable_experts <= num_local_experts:
            raise ValueError(
                "no_local requires routable experts beyond the local domain"
            )
        scores = torch.rand((num_tokens, num_routable_experts), generator=generator)
        scores[:, owned_expert_start:owned_end] = float("-inf")
        if top_k > num_routable_experts - num_local_experts:
            raise ValueError("top_k exceeds the non-local expert count")
        return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)

    raise ValueError(f"unknown route generator {route_generator!r}")


def generate_topk_weights(
    *,
    weight_distribution: str,
    num_tokens: int,
    top_k: int,
    seed: int,
) -> torch.Tensor:
    """Generate normalized FP32 route coefficients ``[T, K]``."""
    if weight_distribution == "equal":
        return torch.full((num_tokens, top_k), 1.0 / top_k, dtype=torch.float32)
    if weight_distribution == "seeded_random":
        raw = torch.rand((num_tokens, top_k), generator=_generator(seed))
        return (raw / raw.sum(dim=1, keepdim=True)).to(torch.float32)
    raise ValueError(f"unknown weight distribution {weight_distribution!r}")


BASE_ROW_REPRESENTATIONS = ("sentinel", "disabled_slot")


def generate_token_lora_mapping(
    *,
    num_tokens: int,
    active_slot_ids: tuple[int, ...],
    include_base_rows: bool,
    seed: int,
    base_row_representation: str = "sentinel",
    base_slot_id: int = 0,
) -> torch.Tensor:
    """Assign tokens to adapter slots, round-robin.

    Base rows are interleaved deterministically so every batch phase sees
    mixed traffic rather than a base-only prefix.  Two representations exist
    and both must route identically:

    ``sentinel``      base rows carry ``-1`` directly.
    ``disabled_slot`` base rows carry a REAL resident slot whose factors are
                      zero-filled and whose ``adapter_enabled`` entry is 0 —
                      this is what serving actually produces, so the runner
                      canonicalizes it back to ``-1`` before routing.
    """
    if base_row_representation not in BASE_ROW_REPRESENTATIONS:
        raise ValueError(
            f"base_row_representation={base_row_representation!r} is not one "
            f"of {BASE_ROW_REPRESENTATIONS}"
        )
    base_marker = -1 if base_row_representation == "sentinel" else base_slot_id
    assignments = list(active_slot_ids)
    if include_base_rows or not assignments:
        assignments.append(base_marker)
    mapping = torch.tensor(
        [assignments[t % len(assignments)] for t in range(num_tokens)],
        dtype=torch.int64,
    )
    # Deterministic shuffle decorrelates slot from token position.
    permutation = torch.randperm(num_tokens, generator=_generator(seed))
    return mapping[permutation].contiguous()


def resolve_route_stats(
    *,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    lora_experts_per_adapter: int,
    max_loras: int,
    block_size: int,
    lora_expert_map: torch.Tensor | None = None,
) -> RouteStats:
    """Compute the plan §5 route symbols for the case record (host-side)."""
    num_tokens, top_k = topk_ids.shape
    ids = topk_ids.to(torch.int64)
    if lora_expert_map is None:
        lora_expert_ids = ids.clone()
    else:
        lora_expert_map = lora_expert_map.to(torch.int64)
        in_map = (ids >= 0) & (ids < lora_expert_map.numel())
        lora_expert_ids = torch.where(
            in_map,
            lora_expert_map[ids.clamp(min=0, max=lora_expert_map.numel() - 1)],
            -1,
        )
    adapters = token_lora_mapping.to(torch.int64)[:, None].expand_as(ids)
    valid = (
        (adapters >= 0)
        & (adapters < max_loras)
        & (lora_expert_ids >= 0)
        & (lora_expert_ids < lora_experts_per_adapter)
    )
    virtual_ids = torch.where(
        valid, adapters * lora_experts_per_adapter + lora_expert_ids, torch.tensor(-1)
    )

    routed_valid = (lora_expert_ids >= 0) & (lora_expert_ids < lora_experts_per_adapter)
    p_valid_routed = int(routed_valid.sum())
    e_hit_routed = (
        int(lora_expert_ids[routed_valid].unique().numel()) if p_valid_routed else 0
    )
    p_valid = int(valid.sum())
    hit_factors = lora_expert_ids[valid]
    e_hit = int(hit_factors.unique().numel()) if p_valid else 0
    groups = virtual_ids[valid]
    if p_valid:
        _, group_sizes = groups.unique(return_counts=True)
        group_count = int(group_sizes.numel())
        histogram: dict[int, int] = {}
        for size in group_sizes.tolist():
            histogram[size] = histogram.get(size, 0) + 1
        p_aligned = int(
            sum(-(-size // block_size) * block_size for size in group_sizes.tolist())
        )
    else:
        group_count = 0
        histogram = {}
        p_aligned = 0

    return RouteStats(
        num_tokens=num_tokens,
        top_k=top_k,
        p_valid_routed=p_valid_routed,
        e_hit_routed=e_hit_routed,
        p_valid=p_valid,
        p_aligned=p_aligned,
        e_hit=e_hit,
        group_count=group_count,
        group_size_histogram=histogram,
    )
