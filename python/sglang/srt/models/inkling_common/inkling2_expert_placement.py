"""Static expert placement for Inkling-2 routed experts under expert parallelism.

By default rank r holds the contiguous block of logical experts
[r*E/ep, (r+1)*E/ep). Routed popularity is skewed and correlated with expert
index, so contiguous blocks concentrate hot experts on a few ranks, and the
per-layer combine waits for the most loaded one. A placement file reorders
which physical slot holds which logical expert, per layer, so routed load
evens out across ranks.

The placement is applied at load time and nowhere else: the gate's output rows
are permuted so it emits physical slot ids directly, and each rank loads the
experts for its own slots. Every downstream consumer already treats routed ids
as contiguous rank-owned ranges, which physical slot ids are, so no kernel or
forward-path code changes.

Build a placement from per-layer routing counts:

    python -m sglang.srt.models.inkling_common.inkling2_expert_placement \\
        --counts counts.json --ep-size 8 --out placement.json

where counts.json is {"<layer_id>": [tokens routed to each logical expert]}.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def expert_slots_for_layer(
    layer_id: int, num_experts: int, ep_size: int
) -> torch.Tensor | None:
    """Logical expert held by each physical slot for this layer, or None for
    contiguous placement (no placement file, or the file omits this layer)."""
    path = envs.SGLANG_OPT_INKLING2_EXPERT_PLACEMENT_FILE.get()
    # A single rank holds every expert, so placement cannot change its load.
    if path is None or ep_size == 1:
        return None
    slots = _load_placement(path, num_experts, ep_size).get(layer_id)
    if slots is None:
        _warn_unplaced(path, layer_id)
    return slots


@functools.cache
def _warn_unplaced(path: str, layer_id: int) -> None:
    logger.warning(
        f"Inkling2 expert placement {path} has no entry for layer {layer_id}; "
        "it keeps contiguous placement"
    )


@functools.cache
def _load_placement(
    path: str, num_experts: int, ep_size: int
) -> dict[int, torch.Tensor]:
    if num_experts % ep_size:
        raise ValueError(
            f"Expert placement needs {num_experts} experts divisible by EP size {ep_size}"
        )
    with open(path, "rb") as f:
        content = f.read()
    raw = json.loads(content)
    slots = raw.get("slots")
    if not isinstance(slots, dict):
        raise ValueError(f"{path}: expected an object with a 'slots' mapping")
    placement = {}
    for key, order in slots.items():
        # A non-permutation would silently route tokens to the wrong expert weights.
        if sorted(order) != list(range(num_experts)):
            raise ValueError(
                f"{path}: layer {key} is not a permutation of {num_experts} experts"
            )
        placement[int(key)] = torch.tensor(order, dtype=torch.long)
    # Every rank reads the file itself; matching hashes across ranks' logs confirm one placement.
    logger.info(
        f"Inkling2 expert placement: {len(placement)} layers "
        f"({min(placement, default=None)}..{max(placement, default=None)}) from {path}, "
        f"sha256 {hashlib.sha256(content).hexdigest()[:16]}"
    )
    return placement


def balance_slots(counts: list[int], ep_size: int) -> list[int]:
    """Physical slot order that evens per-rank routed load, holding E/ep experts
    per rank. Greedy heuristic: heaviest expert first onto the least loaded rank."""
    num_experts = len(counts)
    per_rank = num_experts // ep_size
    if per_rank * ep_size != num_experts:
        raise ValueError(f"{num_experts} experts do not divide across {ep_size} ranks")
    load = [0] * ep_size
    owned: list[list[int]] = [[] for _ in range(ep_size)]
    for expert in sorted(range(num_experts), key=lambda e: -counts[e]):
        rank = min(
            (r for r in range(ep_size) if len(owned[r]) < per_rank),
            key=lambda r: load[r],
        )
        owned[rank].append(expert)
        load[rank] += counts[expert]
    return [expert for rank_experts in owned for expert in sorted(rank_experts)]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--counts", required=True, help="per-layer routing counts JSON")
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--out", required=True, help="placement JSON to write")
    args = parser.parse_args()
    with open(args.counts) as f:
        counts = json.load(f)
    slots = {
        layer: balance_slots(layer_counts, args.ep_size)
        for layer, layer_counts in counts.items()
    }
    with open(args.out, "w") as f:
        json.dump({"slots": slots}, f)


if __name__ == "__main__":
    _main()
