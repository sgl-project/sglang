"""AFD data-plane backend abstraction.

The data plane moves per-layer MoE tensors (hidden states, topk ids/weights,
expert outputs) between attn and ffn ranks. Tensor sizes are never negotiated
here: both sides receive the exact wire capacity from the control plane and
provide matching views.

Joint logical/NCCL rank layout: ffn ranks come first, followed by each attn
instance in handshake order: [F][A0][A1].... This makes ffn rank 0 the global
rank 0, which hosts the rendezvous store — consistent with the ffn instance
being the passive server side whose address the attn instances already know
(--afd-ffn-addr).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from sglang.srt.afd.afd_config import AFDRole

if TYPE_CHECKING:
    import torch

    from sglang.srt.afd.afd_handshake import AFDHandshakeResult
    from sglang.srt.afd.comm.nvshmem import NVSHMEMRuntimeOwner


@dataclass(frozen=True)
class AFDCommConfig:
    role: AFDRole
    # Total attn ranks in this communicator. For staged/single-A backends this
    # is also the local attn instance size.
    num_attn_ranks: int
    num_ffn_ranks: int
    # Attn ranks visible to this backend view: one session for staged backends
    # and Direct A ranks, and the full Direct union on F ranks.
    num_local_attn_ranks: int
    # Rank within the local instance (attn: DP rank; ffn: local TP rank).
    instance_rank: int
    # Rendezvous store endpoint on the ffn side (host from --afd-ffn-addr on
    # the attn side; port negotiated in the handshake).
    store_host: str
    store_port: int
    device: str
    # Static multi-A placement within the communicator's contiguous attn rank
    # range.
    attn_instance_rank_offset: int = 0

    def __post_init__(self):
        if self.num_attn_ranks <= 0 or self.num_ffn_ranks <= 0:
            raise ValueError("AFD communicator rank counts must be positive")
        if self.num_local_attn_ranks <= 0:
            raise ValueError("AFD local attn rank count must be positive")
        if not (
            0
            <= self.attn_instance_rank_offset
            <= self.num_attn_ranks - self.num_local_attn_ranks
        ):
            raise ValueError(
                "AFD attn instance rank range is outside the communicator: "
                f"offset={self.attn_instance_rank_offset}, "
                f"local={self.num_local_attn_ranks}, total={self.num_attn_ranks}"
            )
        num_local = (
            self.num_local_attn_ranks if self.role.is_attn() else self.num_ffn_ranks
        )
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
        return self.num_ffn_ranks + self.attn_instance_rank_offset + self.instance_rank

    def ffn_ranks_for_attn(self, attn_rank: int, *, scattered: bool) -> tuple[int, ...]:
        union_attn_rank = self.attn_instance_rank_offset + attn_rank
        if scattered:
            return (union_attn_rank % self.num_ffn_ranks,)
        return tuple(range(union_attn_rank, self.num_ffn_ranks, self.num_attn_ranks))

    def attn_ranks_for_ffn(self, ffn_rank: int, *, scattered: bool) -> tuple[int, ...]:
        if scattered:
            return tuple(
                range(
                    ffn_rank,
                    self.num_local_attn_ranks,
                    self.num_ffn_ranks,
                )
            )
        return (ffn_rank % self.num_local_attn_ranks,)

    def ffn_global_rank(self, ffn_rank: int) -> int:
        return ffn_rank

    def attn_global_rank(self, attn_rank: int) -> int:
        return self.num_ffn_ranks + self.attn_instance_rank_offset + attn_rank


class AFDCommBackend(ABC):
    """Point-to-point transport between attn and ffn ranks.

    All tensors must be contiguous and identically sized on both peers.
    Decode transfers use bucket-capacity views; the prefill fallback still
    uses exact-size tensors. Recv methods fill the given tensors in place.

    Every operation must be ordered on the caller's current device stream, or
    establish equivalent stream dependencies. Returning means the transfer is
    enqueued, not complete. Fixed-slot callers may reuse a slot after its ordered
    response and downstream consumers have been enqueued on the same stream.
    Independent-stream implementations require an explicit completion/fence.
    """

    def __init__(self, config: AFDCommConfig):
        self.config = config

    @property
    def decode_transport(self) -> NVSHMEMRuntimeOwner | None:
        return None

    @property
    def nccl_comm_ptr(self) -> int | None:
        """Raw joint-world NCCL communicator for fused data planes."""
        return None

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


def create_afd_comm_backend(
    backend: str,
    config: AFDCommConfig,
    *,
    handshake: AFDHandshakeResult | None = None,
) -> AFDCommBackend:
    if backend == "nccl":
        from sglang.srt.afd.comm.nccl import NCCLCommBackend

        return NCCLCommBackend(config)
    if backend == "nvshmem":
        if handshake is None or handshake.nvshmem_uid is None:
            raise ValueError(
                "AFD NVSHMEM backend requires a completed handshake with a unique ID"
            )
        from sglang.srt.afd.comm.nvshmem import NVSHMEMCommBackend

        return NVSHMEMCommBackend(config=config, handshake=handshake)
    raise ValueError(f"Unknown AFD comm backend {backend!r}")
