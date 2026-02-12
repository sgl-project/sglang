"""
Node Registry for Cross-Node Communication

This module provides functionality to map tp_ranks to their hosting nodes
and facilitate cross-node communication for remote instance transfer engine info.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a node in the distributed setup."""

    node_rank: int
    host: str
    tp_rank_range: range

    def __contains__(self, tp_rank: int) -> bool:
        """Check if this node hosts the given tp_rank."""
        return tp_rank in self.tp_rank_range


class NodeRegistry:
    """Registry for mapping tp_ranks to nodes in a distributed setup."""

    def __init__(self, server_args: ServerArgs):
        # NOTE: Assuming that in sglang, tp_size * pp_size = world_size.
        self.server_args = server_args
        self.nodes: Dict[int, NodeInfo] = {}
        if hasattr(self.server_args, "node_hosts") and self.server_args.node_hosts:
            self.noderank2host = json.loads(self.server_args.node_hosts)
        else:
            self.noderank2host = None
        self._populate_node_mapping()

    def _populate_node_mapping(self):
        """Build node registry using same logic as scheduler_launcher.py"""
        # Same calculation logic as in scheduler_launcher.py:90-95
        nnodes_per_tp_group = max(
            self.server_args.nnodes // self.server_args.pp_size, 1
        )
        tp_size_per_node = self.server_args.tp_size // nnodes_per_tp_group

        logger.info(
            f"Building node registry: nnodes={self.server_args.nnodes}, "
            f"tp_size={self.server_args.tp_size}, pp_size={self.server_args.pp_size}"
        )
        logger.info(
            f"Calculated: nnodes_per_tp_group={nnodes_per_tp_group}, "
            f"tp_size_per_node={tp_size_per_node}"
        )

        for node_rank in range(self.server_args.nnodes):
            tp_start = tp_size_per_node * (node_rank % nnodes_per_tp_group)
            tp_end = tp_size_per_node * (node_rank % nnodes_per_tp_group + 1)

            host = self._get_node_host(node_rank)

            node_info = NodeInfo(
                node_rank=node_rank, host=host, tp_rank_range=range(tp_start, tp_end)
            )

            self.nodes[node_rank] = node_info
            logger.info(
                f"Node {node_rank}: host={host}, tp_ranks={list(node_info.tp_rank_range)}"
            )

    def _get_node_host(self, node_rank: int) -> str:
        """
        Determine host for a given node_rank.
        """
        return (
            getattr(self.server_args, "host", "127.0.0.1")
            if self.noderank2host is None
            else self.noderank2host[str(node_rank)]
        )

    def get_node_for_rank(self, tp_rank: int) -> Optional[NodeInfo]:
        """Get the node info for a given tp_rank."""
        for node_info in self.nodes.values():
            if tp_rank in node_info:
                return node_info
        return None

    def validate_tp_rank(self, tp_rank: int) -> bool:
        """Validate if the tp_rank is within the valid range."""
        return 0 <= tp_rank < self.server_args.tp_size
