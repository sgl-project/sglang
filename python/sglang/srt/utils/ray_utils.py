"""Lightweight Ray utilities for SGLang.

Only imports ray — no torch/cuda — safe for CPU-only head nodes.
"""

from typing import List

import ray
from ray.util.placement_group import PlacementGroup, placement_group


def create_placement_groups(
    tp_size: int, pp_size: int, nnodes: int
) -> List[PlacementGroup]:
    """Create per-node STRICT_PACK placement groups.

    Args:
        tp_size: Tensor parallelism size.
        pp_size: Pipeline parallelism size.
        nnodes: Number of nodes.

    Returns:
        List of placement groups, one per node.
    """
    world_size = tp_size * pp_size
    gpus_per_node = world_size // nnodes

    pgs = []
    for _ in range(nnodes):
        pg = placement_group(
            [{"GPU": 1}] * gpus_per_node, strategy="STRICT_PACK"
        )
        pgs.append(pg)
    ray.get([pg.ready() for pg in pgs])
    return pgs
