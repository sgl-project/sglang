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
    "hotset_80_20",
    "one_hot",
    "no_local",
)

WEIGHT_DISTRIBUTIONS = ("equal", "seeded_random")


class RouteStats(msgspec.Struct, frozen=True, kw_only=True):
    """Host-resolved statistics of one materialized route (plan §5 symbols)."""

    num_tokens: int
    top_k: int
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
    seed: int,
) -> torch.Tensor:
    """Generate ``[T, K]`` int32 expert IDs with distinct experts per token.

    ``no_local`` draws only experts outside ``[0, num_local_experts)`` so an
    EP-local factor map resolves every pair to the ``-1`` sentinel.
    """
    if top_k > num_routable_experts:
        raise ValueError("top_k exceeds the routable expert count")
    generator = _generator(seed)

    if route_generator == "balanced":
        # Round-robin so every expert receives an equal pair count where
        # divisible; consecutive K ids per token are distinct by construction.
        flat = (
            torch.arange(num_tokens * top_k, dtype=torch.int64)
            % num_routable_experts
        )
        ids = flat.view(num_tokens, top_k)
        if num_routable_experts >= top_k:
            return ids.to(torch.int32)
        raise ValueError("balanced route requires num_routable_experts >= top_k")

    if route_generator == "iid":
        scores = torch.rand(
            (num_tokens, num_routable_experts), generator=generator
        )
        return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)

    if route_generator == "hotset_80_20":
        hot_count = max(1, num_routable_experts // 5)
        scores = torch.rand(
            (num_tokens, num_routable_experts), generator=generator
        )
        scores[:, :hot_count] += 4.0 * torch.rand(
            (num_tokens, hot_count), generator=generator
        )
        return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)

    if route_generator == "one_hot":
        # Every token routes to the same K experts: maximal fragmentation of
        # zero and maximal group concentration.
        ids = torch.arange(top_k, dtype=torch.int32)
        return ids.expand(num_tokens, top_k).contiguous()

    if route_generator == "no_local":
        if num_routable_experts <= num_local_experts:
            raise ValueError(
                "no_local requires routable experts beyond the local domain"
            )
        nonlocal_count = num_routable_experts - num_local_experts
        if top_k > nonlocal_count:
            raise ValueError("top_k exceeds the non-local expert count")
        scores = torch.rand((num_tokens, nonlocal_count), generator=generator)
        return (
            torch.topk(scores, top_k, dim=1).indices + num_local_experts
        ).to(torch.int32)

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


def generate_token_lora_mapping(
    *,
    num_tokens: int,
    active_slot_ids: tuple[int, ...],
    include_base_rows: bool,
    seed: int,
) -> torch.Tensor:
    """Assign tokens to adapter slots (``-1`` = base-only), round-robin.

    Base rows, when requested, are interleaved deterministically so every
    batch phase sees mixed traffic rather than a base-only prefix.
    """
    assignments = list(active_slot_ids)
    if include_base_rows or not assignments:
        assignments.append(-1)
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
    factor_expert_count: int,
    max_loras: int,
    block_size: int,
    routed_expert_to_factor_id: torch.Tensor | None = None,
) -> RouteStats:
    """Compute the plan §5 route symbols for the case record (host-side)."""
    num_tokens, top_k = topk_ids.shape
    ids = topk_ids.to(torch.int64)
    if routed_expert_to_factor_id is None:
        factor_ids = ids.clone()
    else:
        factor_map = routed_expert_to_factor_id.to(torch.int64)
        in_map = (ids >= 0) & (ids < factor_map.numel())
        factor_ids = torch.where(
            in_map, factor_map[ids.clamp(min=0, max=factor_map.numel() - 1)], -1
        )
    adapters = token_lora_mapping.to(torch.int64)[:, None].expand_as(ids)
    valid = (
        (adapters >= 0)
        & (adapters < max_loras)
        & (factor_ids >= 0)
        & (factor_ids < factor_expert_count)
    )
    virtual_ids = torch.where(
        valid, adapters * factor_expert_count + factor_ids, torch.tensor(-1)
    )

    p_valid = int(valid.sum())
    hit_factors = factor_ids[valid]
    e_hit = int(hit_factors.unique().numel()) if p_valid else 0
    groups = virtual_ids[valid]
    if p_valid:
        _, group_sizes = groups.unique(return_counts=True)
        group_count = int(group_sizes.numel())
        histogram: dict[int, int] = {}
        for size in group_sizes.tolist():
            histogram[size] = histogram.get(size, 0) + 1
        p_aligned = int(
            sum(
                -(-size // block_size) * block_size
                for size in group_sizes.tolist()
            )
        )
    else:
        group_count = 0
        histogram = {}
        p_aligned = 0

    return RouteStats(
        num_tokens=num_tokens,
        top_k=top_k,
        p_valid=p_valid,
        p_aligned=p_aligned,
        e_hit=e_hit,
        group_count=group_count,
        group_size_histogram=histogram,
    )


__all__ = [
    "ROUTE_GENERATORS",
    "WEIGHT_DISTRIBUTIONS",
    "RouteStats",
    "generate_token_lora_mapping",
    "generate_topk_ids",
    "generate_topk_weights",
    "resolve_route_stats",
]
