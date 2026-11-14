"""Communicator registry for the ops that are keyed by ``world_size`` alone.

Kept free of tvm_ffi so the op modules stay importable without CUDA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator


class CommRegistry:
    """The communicator behind ops that are keyed by ``world_size`` alone.

    A consumer registers a provider, a zero-argument callable returning its
    Communicator. The provider runs on the first launch, so registering never
    allocates; its result is cached. ``world_size`` is the whole key, so a process
    holds at most one provider per size: a second group of the same size would
    silently inherit the first one's planes (a hang or corruption, not an error).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._providers: Dict[int, Callable[[], Communicator]] = {}
        self._comms: Dict[int, Communicator] = {}

    def register(
        self, provider: Callable[[], Communicator], *, world_size: int
    ) -> None:
        prev = self._providers.get(world_size)
        assert prev is None or prev is provider, (
            f"{self._name}: a different communicator is already registered for "
            f"world_size={world_size}; these ops key only on world_size, so two "
            f"groups of the same size cannot coexist in one process"
        )
        self._providers[world_size] = provider

    def __contains__(self, world_size: int) -> bool:
        return world_size in self._providers

    def get(self, world_size: int) -> Communicator:
        comm = self._comms.get(world_size)
        if comm is None:
            provider = self._providers.get(world_size)
            assert provider is not None, (
                f"{self._name}: no communicator registered for "
                f"world_size={world_size}; call register_comm first"
            )
            comm = self._comms[world_size] = provider()
            assert comm.world_size == world_size, (comm.world_size, world_size)
        return comm

    def get_optional(self, world_size: int) -> Optional[Communicator]:
        return self.get(world_size) if world_size in self else None
