# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ray cluster discovery and placement group utilities for multi-node deployment."""

import dataclasses
from typing import TYPE_CHECKING, Dict, List, Tuple

import ray

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


@dataclasses.dataclass
class RayNodeInfo:
    """Information about a Ray node with GPU resources."""

    node_id: str
    gpu_count: int


@dataclasses.dataclass
class RayClusterTopology:
    """Discovered Ray cluster topology for multi-node deployment."""

    nodes: List[RayNodeInfo]
    total_gpus: int
    nnodes: int
    gpus_per_node: int  # Assumes homogeneous


def discover_gpu_nodes() -> List[RayNodeInfo]:
    """
    Discover Ray nodes with GPUs using available_resources_per_node().

    Returns list sorted by node_id for deterministic ordering.
    Does NOT use ray.nodes() (debugging only).
    """
    from ray._private.state import available_resources_per_node

    resources = available_resources_per_node()
    gpu_nodes = []
    for node_id, node_resources in resources.items():
        gpu_count = int(node_resources.get("GPU", 0))
        if gpu_count > 0:
            gpu_nodes.append(RayNodeInfo(node_id=node_id, gpu_count=gpu_count))

    # Sort by node_id for deterministic head node selection
    gpu_nodes.sort(key=lambda n: n.node_id)
    return gpu_nodes


def create_cluster_topology(
    gpu_nodes: List[RayNodeInfo],
    world_size: int,
) -> RayClusterTopology:
    """
    Create cluster topology based on world size requirements.

    Args:
        gpu_nodes: Discovered GPU nodes
        world_size: Total GPUs needed (tp_size * pp_size)

    Returns:
        RayClusterTopology with node assignment info

    Raises:
        RuntimeError: If insufficient GPUs or heterogeneous GPU counts
    """
    total_gpus = sum(n.gpu_count for n in gpu_nodes)
    if total_gpus < world_size:
        raise RuntimeError(
            f"Cluster has {total_gpus} GPUs but world_size={world_size} required"
        )

    # Require homogeneous GPU counts for simplicity
    gpu_counts = {n.gpu_count for n in gpu_nodes}
    if len(gpu_counts) > 1:
        raise RuntimeError(
            f"Multi-node Ray requires homogeneous GPU counts. Found: {gpu_counts}"
        )

    gpus_per_node = gpu_nodes[0].gpu_count
    nnodes_needed = (world_size + gpus_per_node - 1) // gpus_per_node

    return RayClusterTopology(
        nodes=gpu_nodes[:nnodes_needed],
        total_gpus=nnodes_needed * gpus_per_node,
        nnodes=nnodes_needed,
        gpus_per_node=gpus_per_node,
    )


def create_per_node_placement_groups(
    topology: RayClusterTopology,
) -> List["PlacementGroup"]:
    """
    Create one placement group per node based on world size.

    Each PG has gpus_per_node bundles with 1 GPU each.
    Uses STRICT_PACK to keep all bundles on same node.
    """
    from ray.util.placement_group import placement_group

    placement_groups = []
    for _ in range(topology.nnodes):
        bundles = [{"GPU": 1} for _ in range(topology.gpus_per_node)]
        pg = placement_group(bundles, strategy="STRICT_PACK")
        placement_groups.append(pg)

    # Wait for all PGs to be ready
    ray.get([pg.ready() for pg in placement_groups])
    return placement_groups


def compute_rank_to_node_assignment(
    tp_size: int,
    pp_size: int,
    topology: RayClusterTopology,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Compute mapping of (pp_rank, tp_rank) -> (node_idx, local_gpu_idx).

    Returns:
        Dict mapping (pp_rank, tp_rank) to (node_index, gpu_index_on_node)
    """
    assignment = {}
    world_size = tp_size * pp_size
    gpus_per_node = topology.gpus_per_node

    for global_idx in range(world_size):
        pp_rank = global_idx // tp_size
        tp_rank = global_idx % tp_size
        node_idx = global_idx // gpus_per_node
        local_gpu_idx = global_idx % gpus_per_node
        assignment[(pp_rank, tp_rank)] = (node_idx, local_gpu_idx)

    return assignment
