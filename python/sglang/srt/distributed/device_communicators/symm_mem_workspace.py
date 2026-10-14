"""A rendezvoused torch symmetric-memory buffer, shared by symm-mem communicators.

Callers own synchronization: a peer's region is safe to read only after the
writer and the reader have both passed a :meth:`SymmMemWorkspace.barrier`.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.kernels.ops.memory.ptr_table import make_ptr_table
from sglang.srt.distributed.parallel_state import in_the_same_node_as

logger = logging.getLogger(__name__)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _import_error: Optional[BaseException] = None
except ImportError as e:  # pragma: no cover
    torch_symm_mem = None
    _import_error = e


def all_ranks_agree(ok: bool, group: ProcessGroup) -> bool:
    """True only if ``ok`` holds on every rank of the CPU ``group``."""
    verdict = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
    return bool(verdict.item())


class SymmMemWorkspace:
    """One symmetric buffer, rendezvoused over a single-host CPU group.

    Build it with :meth:`create`, which returns ``None`` and logs why when the
    device, the group or the container cannot provide peer-mapped memory.
    """

    # Peer mapping is intra-node only.
    MAX_WORLD_SIZE = 8

    def __init__(
        self,
        *,
        buffer: torch.Tensor,
        handle,
        group: ProcessGroup,
        device: torch.device,
    ):
        self.buffer = buffer
        self.handle = handle
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        # [world_size] int64 base address of every rank's buffer, own rank included.
        self.peer_ptrs = make_ptr_table(handle.buffer_ptrs, device)

    @classmethod
    def create(
        cls,
        *,
        group: ProcessGroup,
        device: torch.device,
        nbytes: int,
        owner: str,
    ) -> Optional[SymmMemWorkspace]:
        """Allocate ``nbytes`` of symmetric memory on ``device`` and rendezvous.

        Collective over ``group``: every rank must call it with the same
        ``nbytes``. ``owner`` prefixes the warning logged on a fallback.
        """
        if torch_symm_mem is None:
            logger.warning(
                "%s: torch symmetric memory is unavailable (%s); disabled.",
                owner,
                _import_error,
            )
            return None
        backend = torch_symm_mem.get_backend(device)
        if backend != "XPU":
            logger.warning(
                "%s: symmetric-memory backend for %s is %s, expected 'XPU'; disabled.",
                owner,
                device,
                backend,
            )
            return None
        world_size = dist.get_world_size(group)
        if world_size > cls.MAX_WORLD_SIZE:
            logger.warning(
                "%s: world size %d exceeds the supported maximum %d; disabled.",
                owner,
                world_size,
                cls.MAX_WORLD_SIZE,
            )
            return None
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning(
                "%s: group spans multiple hosts, peer mapping is intra-node; disabled.",
                owner,
            )
            return None
        if nbytes <= 0:
            logger.warning("%s: requested %d bytes; disabled.", owner, nbytes)
            return None

        try:
            buffer = torch_symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
            handle = torch_symm_mem.rendezvous(buffer, group.group_name)
        except Exception as e:
            # Importing a peer's allocation goes through pidfd_getfd, which needs
            # ptrace permission; restrictive seccomp / yama fails here.
            logger.warning(
                "%s: symmetric-memory rendezvous failed (%s); disabled.", owner, e
            )
            return None
        return cls(buffer=buffer, handle=handle, group=group, device=device)

    def barrier(self) -> None:
        """Stream-ordered: work queued before it on any rank is visible after it."""
        self.handle.barrier()
