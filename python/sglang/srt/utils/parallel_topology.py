# SPDX-License-Identifier: Apache-2.0
"""Rank placement helpers for the standard homogeneous-node launcher."""

from typing import Tuple


def validate_standard_rank_layout(nnodes: int, pp_size: int, tp_size: int) -> None:
    """Validate that the standard launcher can partition TP x PP over nodes.

    The standard launcher supports two rectangular layouts: complete PP stages
    assigned evenly to nodes, or complete nodes assigned evenly to PP stages.
    """
    if nnodes <= 0 or pp_size <= 0 or tp_size <= 0:
        raise ValueError(
            "Parallel sizes must be positive: "
            f"tp_size={tp_size}, pp_size={pp_size}, nnodes={nnodes}."
        )

    topology = f"tp_size={tp_size}, pp_size={pp_size}, nnodes={nnodes}"

    if pp_size >= nnodes:
        if pp_size % nnodes != 0:
            raise ValueError(
                f"Unsupported standard rank layout ({topology}): pp_size must "
                "be divisible by nnodes when complete PP stages are assigned "
                "to each node."
            )
        return

    if nnodes % pp_size != 0:
        raise ValueError(
            f"Unsupported standard rank layout ({topology}): nnodes must be "
            "divisible by pp_size when each PP stage spans multiple nodes."
        )

    nnodes_per_pp_rank = nnodes // pp_size
    if tp_size % nnodes_per_pp_rank != 0:
        raise ValueError(
            f"Unsupported standard rank layout ({topology}): tp_size must be "
            f"divisible by nnodes / pp_size ({nnodes_per_pp_rank}) when each "
            "PP stage spans multiple nodes."
        )


def calculate_rank_ranges(
    nnodes: int, pp_size: int, tp_size: int, node_rank: int
) -> Tuple[range, range, int, int]:
    """Calculate the PP and TP rank ranges assigned to one standard node."""
    validate_standard_rank_layout(nnodes, pp_size, tp_size)

    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )

    tp_size_per_node = tp_size // nnodes_per_pp_rank
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_pp_rank),
        tp_size_per_node * (node_rank % nnodes_per_pp_rank + 1),
    )

    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node
