from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang.srt.runtime_context import get_parallel


@dataclass(frozen=True)
class EPLBStageDomain:
    """The rank and process-group domain that owns one PP stage's EPLB state."""

    pp_rank: int
    global_rank: int
    ep_rank: int
    ep_size: int
    global_ranks: tuple[int, ...]
    node_ids: tuple[int, ...]
    root_global_rank: int
    device_group: Any
    cpu_group: Any

    @classmethod
    def from_runtime(cls) -> EPLBStageDomain:
        parallel = get_parallel()
        group = parallel.moe_ep_group
        global_ranks = tuple(group.ranks)

        if group.world_size != parallel.moe_ep_size:
            raise RuntimeError(
                "EPLB EP group width does not match the published MoE EP size: "
                f"group={group.world_size}, published={parallel.moe_ep_size}"
            )
        if len(global_ranks) != group.world_size:
            raise RuntimeError(
                "EPLB EP group rank list does not match its width: "
                f"ranks={global_ranks}, width={group.world_size}"
            )
        if len(set(global_ranks)) != len(global_ranks):
            raise RuntimeError(
                f"EPLB EP group contains duplicate ranks: {global_ranks}"
            )
        if group.rank not in global_ranks:
            raise RuntimeError(
                "Current global rank is not a member of its EPLB EP group: "
                f"rank={group.rank}, ranks={global_ranks}"
            )

        launch_world_size = parallel.tp_size * parallel.pp_size
        if parallel.nnodes <= 0 or launch_world_size % parallel.nnodes != 0:
            raise RuntimeError(
                "EPLB cannot derive node ownership from a non-uniform launch: "
                f"world_size={launch_world_size}, nnodes={parallel.nnodes}"
            )
        ranks_per_node = launch_world_size // parallel.nnodes
        if any(rank < 0 or rank >= launch_world_size for rank in global_ranks):
            raise RuntimeError(
                "EPLB EP group ranks fall outside the launch world: "
                f"ranks={global_ranks}, world_size={launch_world_size}"
            )
        node_ids = tuple(rank // ranks_per_node for rank in global_ranks)
        # Hierarchical solvers reshape EP slots into equal, contiguous node
        # blocks. Passing only num_nodes for any other placement silently
        # describes a different topology, even if the updater knows each peer.
        node_order = tuple(dict.fromkeys(node_ids))
        node_counts = tuple(node_ids.count(node) for node in node_order)
        if len(set(node_counts)) != 1 or node_ids != tuple(
            node for node, count in zip(node_order, node_counts) for _ in range(count)
        ):
            raise RuntimeError(
                "EPLB solver requires uniform, contiguous node ownership: "
                f"ranks={global_ranks}, node_ids={node_ids}"
            )

        return cls(
            pp_rank=parallel.pp_rank,
            global_rank=group.rank,
            ep_rank=group.rank_in_group,
            ep_size=group.world_size,
            global_ranks=global_ranks,
            node_ids=node_ids,
            root_global_rank=group.first_rank,
            device_group=group.device_group,
            cpu_group=group.cpu_group,
        )

    @classmethod
    def from_world(cls) -> EPLBStageDomain:
        """Preserve the flat WORLD ownership used by elastic-EP scale paths."""

        parallel = get_parallel()
        group = parallel.world_group
        global_ranks = tuple(group.ranks)
        if len(global_ranks) != group.world_size:
            raise RuntimeError(
                "EPLB WORLD rank list does not match its width: "
                f"ranks={global_ranks}, width={group.world_size}"
            )

        launch_world_size = parallel.tp_size * parallel.pp_size
        ranks_per_node = max(launch_world_size // parallel.nnodes, 1)
        node_ids = tuple(rank // ranks_per_node for rank in global_ranks)
        return cls(
            pp_rank=parallel.pp_rank,
            global_rank=group.rank,
            ep_rank=group.rank_in_group,
            ep_size=group.world_size,
            global_ranks=global_ranks,
            node_ids=node_ids,
            root_global_rank=group.first_rank,
            device_group=group.device_group,
            cpu_group=group.cpu_group,
        )

    @property
    def is_root(self) -> bool:
        return self.global_rank == self.root_global_rank

    @property
    def num_nodes(self) -> int:
        return len(set(self.node_ids))

    def global_peer(self, ep_rank: int) -> int:
        if not 0 <= ep_rank < self.ep_size:
            raise ValueError(f"Invalid EP-local rank {ep_rank} for size {self.ep_size}")
        return self.global_ranks[ep_rank]
