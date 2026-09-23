"""Point-to-point AFD transport, with FFN ranks first in the joint world."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from sglang.srt.afd.afd_config import AFDRole

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class AFDCommConfig:
    role: AFDRole
    num_attn_ranks: int
    num_ffn_ranks: int
    instance_rank: int
    store_host: str
    store_port: int
    device: str

    def __post_init__(self):
        if self.num_attn_ranks <= 0 or self.num_ffn_ranks <= 0:
            raise ValueError("AFD communicator rank counts must be positive")
        num_local = self.num_attn_ranks if self.role.is_attn() else self.num_ffn_ranks
        if not 0 <= self.instance_rank < num_local:
            raise ValueError(
                f"instance_rank {self.instance_rank} out of range for "
                f"{num_local} {self.role.value} ranks"
            )

    @property
    def world_size(self) -> int:
        return self.num_attn_ranks + self.num_ffn_ranks

    @property
    def global_rank(self) -> int:
        if self.role.is_ffn():
            return self.instance_rank
        return self.num_ffn_ranks + self.instance_rank

    def ffn_ranks_for_attn(self, attn_rank: int, *, scattered: bool) -> tuple[int, ...]:
        if scattered:
            return (attn_rank % self.num_ffn_ranks,)
        return tuple(range(attn_rank, self.num_ffn_ranks, self.num_attn_ranks))

    def attn_ranks_for_ffn(self, ffn_rank: int, *, scattered: bool) -> tuple[int, ...]:
        if scattered:
            return tuple(range(ffn_rank, self.num_attn_ranks, self.num_ffn_ranks))
        return (ffn_rank % self.num_attn_ranks,)

    def ffn_global_rank(self, ffn_rank: int) -> int:
        return ffn_rank

    def attn_global_rank(self, attn_rank: int) -> int:
        return self.num_ffn_ranks + attn_rank


class AFDCommBackend(ABC):
    """Point-to-point transport between attn and ffn ranks.

    All tensors must be contiguous and identically sized on both peers.
    Recv methods fill the given tensors in place.

    Every operation must be ordered on the caller's current device stream, or
    establish equivalent stream dependencies. Returning means the transfer is
    enqueued, not complete. Callers may reuse inputs after the ordered response
    and downstream consumers have been enqueued on the same stream.
    Independent-stream implementations require an explicit completion/fence.
    """

    def __init__(self, config: AFDCommConfig):
        self.config = config

    @abstractmethod
    def send_to(
        self,
        tensors: Sequence[torch.Tensor],
        peer_global_rank: int,
    ) -> None:
        """Enqueue tensors to an explicit rank in the joint communicator."""

    @abstractmethod
    def recv_from(
        self,
        tensors: Sequence[torch.Tensor],
        peer_global_rank: int,
    ) -> None:
        """Receive tensors from an explicit rank in the joint communicator."""

    @abstractmethod
    def startup_barrier(self) -> None:
        """Synchronize the joint attn+ffn group during startup."""

    @abstractmethod
    def close(self) -> None: ...


def create_afd_comm_backend(backend: str, config: AFDCommConfig) -> AFDCommBackend:
    if backend == "nccl":
        from sglang.srt.afd.comm.nccl import NCCLCommBackend

        return NCCLCommBackend(config)
    raise ValueError(f"Unknown AFD comm backend {backend!r}")
