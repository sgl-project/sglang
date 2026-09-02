"""The communicator every K3 op resolves through.

A `Communicator` cannot cross a custom-op boundary, so the ops carry an `int`
world size and look the object up here. One registry serves all of them: the
model builds a single CustomAllReduceV2 per world size and every op wants that
same one, and per-module copies had already started reaching into each other's
private names.

The push kernels use its push plane and the pull kernels its barrier plane;
both carry their own multicast base, so nothing else is threaded through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator

_COMM_MAP: dict[int, Communicator] = {}


def register(comm: Communicator) -> None:
    """Bind `comm` to its world size for every K3 op in this process."""
    # world_size is the whole key, so at most one communicator per size can be
    # registered. That matches how these ops are called -- they take world_size
    # and nothing else, so a second group of the same size would silently
    # inherit the first one's peer pointers and semaphores, and the symptom
    # would be a hang or corruption rather than an error. Assert instead of
    # letting the overwrite happen; widening to per-group handles means changing
    # every custom-op signature, which is a separate change.
    prev = _COMM_MAP.get(comm.world_size)
    assert prev is None or prev is comm, (
        f"a different communicator is already registered for world_size="
        f"{comm.world_size}; these ops key only on world_size, so two groups of "
        f"the same size cannot coexist in one process"
    )
    _COMM_MAP[comm.world_size] = comm


def get(world_size: int) -> Communicator:
    comm = _COMM_MAP.get(world_size)
    assert comm is not None, (
        f"no K3 communicator registered for world_size={world_size}; "
        f"call register() during setup"
    )
    return comm
