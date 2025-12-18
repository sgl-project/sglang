"""
PyTorch Implementation of DisaggCommunicator.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import Work

from sglang.multimodal_gen.runtime.communication.base_communicator import (
    DisaggCommunicator,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class PyTorchDisaggCommunicator(DisaggCommunicator):
    def __init__(self):
        self.non_dit_group = None
        self.dit_group = None
        self.role = None  # "non_dit" or "dit"
        self.group_rank = -1
        self.world_rank = -1

        # We assume rank 0 of Non-DiT group communicates with rank 0 of DiT group.
        # These are GLOBAL ranks.
        self.non_dit_master_rank = -1
        self.dit_master_rank = -1

    def initialize_topology(self, server_args: Any) -> None:
        self.world_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # --- Logic from composed_pipeline_base.py (Refactored) ---
        # "do_disaggregation = server_args.num_gpus > 1 and (server_args.num_gpus - 1) % 2 == 0"
        # We should probably make this more explicit in server_args later.
        # For now, let's assume the same logic:
        # Rank 0 is Non-DiT
        # Ranks 1..N are DiT

        non_dit_ranks = list(
            range(world_size - server_args.num_non_dit_ranks, world_size)
        )
        dit_ranks = [r for r in range(world_size) if r not in non_dit_ranks]

        self.non_dit_master_rank = non_dit_ranks[0]
        self.dit_master_rank = dit_ranks[0]

        logger.info(
            f"Initializing Disagg Topology: Non-DiT Ranks={non_dit_ranks}, DiT Ranks={dit_ranks}"
        )

        # Create groups
        # Note: new_group requires all processes to call it in the same order
        self.non_dit_group = dist.new_group(ranks=non_dit_ranks)
        self.dit_group = dist.new_group(ranks=dit_ranks)

        if self.world_rank in non_dit_ranks:
            self.role = "non_dit"
            self.group_rank = non_dit_ranks.index(self.world_rank)
        elif self.world_rank in dit_ranks:
            self.role = "dit"
            self.group_rank = dit_ranks.index(self.world_rank)
        else:
            raise ValueError(f"Rank {self.world_rank} not assigned to any group!")

    def get_my_group(self) -> Optional[dist.ProcessGroup]:
        if self.role == "non_dit":
            return self.non_dit_group
        return self.dit_group

    def is_dit_rank(self) -> bool:
        return self.role == "dit"

    def is_non_dit_rank(self) -> bool:
        return self.role == "non_dit"

    def send_to_dit(
        self, tensor: torch.Tensor, metadata: Optional[Dict] = None
    ) -> None:
        """Called by Non-DiT Master (Rank 0) to send to DiT Master."""
        if self.world_rank != self.non_dit_master_rank:
            return  # Only master sends cross-group

        # P2P Send to DiT Master
        # Note: dist.send uses global rank
        dist.send(tensor, dst=self.dit_master_rank)

    def recv_from_non_dit(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """
        Called by DiT Ranks.
        DiT Master receives from Non-DiT, then broadcasts to other DiT ranks.
        """
        tensor = torch.empty(shape, dtype=dtype, device="cuda")  # Todo: proper device

        if self.world_rank == self.dit_master_rank:
            dist.recv(tensor, src=self.non_dit_master_rank)

        # Broadcast within DiT group so all workers have the input
        self.broadcast_in_group(tensor)
        return tensor

    def send_to_non_dit(self, tensor: torch.Tensor) -> None:
        """Called by DiT Master to send result back to Non-DiT."""
        if self.world_rank != self.dit_master_rank:
            return

        dist.send(tensor, dst=self.non_dit_master_rank)

    def recv_from_dit(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Called by Non-DiT Master to receive result."""
        tensor = torch.empty(shape, dtype=dtype, device="cuda")

        if self.world_rank == self.non_dit_master_rank:
            dist.recv(tensor, src=self.dit_master_rank)

        return tensor

    def broadcast_in_group(
        self, tensor: torch.Tensor, src_rank_in_group: int = 0
    ) -> None:
        """
        Wraps dist.broadcast using the current role's group.
        src_rank_in_group is relative to the group.
        """
        group = self.get_my_group()
        if group is None:
            return

        # Check group size - if only 1 member, no need to broadcast
        group_size = dist.get_world_size(group=group)
        if group_size == 1:
            return

        # CRITICAL: Ensure tensor is on CUDA for NCCL backend
        if dist.get_backend(group) == "nccl" and not tensor.is_cuda:
            raise RuntimeError(
                f"[Rank {dist.get_rank()}] Cannot broadcast CPU tensor with NCCL backend. "
                f"Tensor device: {tensor.device}, backend: {dist.get_backend(group)}"
            )

        # We need to translate group-relative src rank to global rank for dist.broadcast
        # Wait, dist.broadcast(group=group) usually expects the `src` to be the GLOBAL rank
        # of the broadcaster.

        if self.role == "dit":
            # Assuming linear mapping for now or finding from stored list
            # Simpler: just assume src is always 0 (master) of that group
            # We need to find the global rank of the group master
            global_src = self.dit_master_rank  # Logic for src=0
        else:
            global_src = self.non_dit_master_rank

        dist.broadcast(tensor, src=global_src, group=group)

    # --- Async Communication Implementation ---

    def isend_to_dit(
        self, tensor: torch.Tensor, metadata: Optional[Dict] = None
    ) -> Optional[Work]:
        """Non-blocking send from Non-DiT to DiT group."""
        if self.world_rank != self.non_dit_master_rank:
            return None  # Only master sends cross-group

        # P2P Non-blocking send to DiT Master
        work = dist.isend(tensor, dst=self.dit_master_rank)
        return work

    def irecv_from_non_dit(
        self, shape: torch.Size, dtype: torch.dtype
    ) -> tuple[torch.Tensor, Optional[Work]]:
        """
        Non-blocking receive from Non-DiT group at DiT group.
        DiT Master receives, then will broadcast to workers.
        """
        tensor = torch.empty(shape, dtype=dtype, device="cuda")

        work = None
        if self.world_rank == self.dit_master_rank:
            work = dist.irecv(tensor, src=self.non_dit_master_rank)

        # Return tensor and work handle
        # Note: broadcast to other DiT workers happens separately after wait
        return tensor, work

    def isend_to_non_dit(self, tensor: torch.Tensor) -> Optional[Work]:
        """Non-blocking send from DiT to Non-DiT group."""
        if self.world_rank != self.dit_master_rank:
            return None

        work = dist.isend(tensor, dst=self.non_dit_master_rank)
        return work

    def irecv_from_dit(
        self, shape: torch.Size, dtype: torch.dtype
    ) -> tuple[torch.Tensor, Optional[Work]]:
        """Non-blocking receive from DiT group at Non-DiT group."""
        tensor = torch.empty(shape, dtype=dtype, device="cuda")

        work = None
        if self.world_rank == self.non_dit_master_rank:
            work = dist.irecv(tensor, src=self.dit_master_rank)

        return tensor, work

    def wait_work(self, work: Optional[Work]) -> None:
        """Wait for a Work handle to complete."""
        if work is not None:
            work.wait()

    def wait_all_works(self, works: List[Optional[Work]]) -> None:
        """Wait for multiple Work handles to complete."""
        for work in works:
            if work is not None:
                work.wait()
