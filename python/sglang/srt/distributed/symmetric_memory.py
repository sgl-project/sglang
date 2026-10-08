"""Symmetric memory from the caching allocator.

``torch.distributed._symmetric_memory`` keeps one ``MemPool`` per device whose allocator
hands out symmetric segments. Allocating inside it makes a symmetric buffer an ordinary
``torch.empty`` that the caching allocator can reuse, rather than a dedicated mapping per
call site the way ``_SymmetricMemory.empty_strided_p2p`` gives.

The pool is not a normal one: it never lends its space to non-symmetric allocations and
never splits a segment, because a segment carries a signal pad. So every rank of a group
must allocate the same shapes in the same order.

Allocation is local; the peer and multicast addresses come from
``torch.distributed._symmetric_memory.rendezvous(tensor, group)``, which the consumer
that needs them calls itself -- a collective the first time a segment is seen, a cached
handle after.
"""

from __future__ import annotations

import torch
import torch.distributed._symmetric_memory as torch_symm_mem


def symmetric_context(device: torch.device):
    """Context manager for using the device's symmetric memory pool."""
    return torch.cuda.use_mem_pool(torch_symm_mem.get_mem_pool(device))


def symmetric_empty(*shape: int, dtype: torch.dtype, device: torch.device):
    """``torch.empty`` out of the device's symmetric pool."""
    with torch.cuda.use_mem_pool(torch_symm_mem.get_mem_pool(device)):
        return torch.empty(*shape, dtype=dtype, device=device)
