"""The communicator behind the Kimi-K3 collectives (``gemm_ag``, ``moe_front_ag``):
the TP group's symmetric workspace, whose push plane they stage through."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.kernels.ops.communication.comm_registry import CommRegistry

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator

_COMMS = CommRegistry("kimi_k3")


def register_comm(provider: Callable[[], Communicator], *, world_size: int) -> None:
    """Register the provider of the communicator; it runs on first launch (see
    ``CommRegistry``)."""
    _COMMS.register(provider, world_size=world_size)


def get_comm(world_size: int) -> Communicator:
    return _COMMS.get(world_size)
