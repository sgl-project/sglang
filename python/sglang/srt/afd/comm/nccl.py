"""NCCL data-plane backend.

Builds a joint attn∪ffn NCCL communicator with StatelessProcessGroup +
PyNcclCommunicator (the pattern proven by update_weights_from_distributed:
it forms a new group across processes whose default torch.distributed
groups are already initialized, without polluting global state).

All ops are asynchronous w.r.t. the CPU and ordered on the current CUDA stream.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from sglang.srt.afd.comm.base import AFDCommBackend, AFDCommConfig
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


class NCCLCommBackend(AFDCommBackend):
    def __init__(self, config: AFDCommConfig):
        super().__init__(config)
        num_local_ranks = (
            config.num_attn_ranks if config.role.is_attn() else config.num_ffn_ranks
        )
        logger.info(
            "AFD NCCL backend init: role=%s local_rank=%d/%d global_rank=%d/%d "
            "store=%s:%d device=%s",
            config.role.value,
            config.instance_rank,
            num_local_ranks,
            config.global_rank,
            config.world_size,
            config.store_host,
            config.store_port,
            config.device,
        )
        self._pg = StatelessProcessGroup.create(
            host=config.store_host,
            port=config.store_port,
            rank=config.global_rank,
            world_size=config.world_size,
        )
        self._comm = PyNcclCommunicator(self._pg, device=config.device)
        if not self._comm.available:
            raise RuntimeError(
                "AFD NCCL backend: NCCL library unavailable on this platform"
            )
        # PyNcclCommunicator constructs itself disabled (its ops silently
        # no-op); AFD always drives it explicitly.
        self._comm.disabled = False
        logger.info(
            "AFD NCCL backend ready: global_rank=%d/%d",
            config.global_rank,
            config.world_size,
        )

    def send_to(
        self,
        tensors: Sequence[torch.Tensor],
        peer_global_rank: int,
    ) -> None:
        self._transfer(tensors, send=True, peer=peer_global_rank)

    def recv_from(
        self,
        tensors: Sequence[torch.Tensor],
        peer_global_rank: int,
    ) -> None:
        self._transfer(tensors, send=False, peer=peer_global_rank)

    def startup_barrier(self) -> None:
        self._pg.barrier()

    def _transfer(
        self,
        tensors: Sequence[torch.Tensor],
        send: bool,
        peer: int,
    ) -> None:
        for t in tensors:
            if not t.is_contiguous():
                raise ValueError("AFD comm tensors must be contiguous")
        # Batch the per-message tensors into one NCCL group so the transport
        # sees a single transfer per logical message.
        self._comm.group_start()
        for t in tensors:
            if send:
                self._comm.send(t, peer)
            else:
                self._comm.recv(t, peer)
        self._comm.group_end()

    def close(self) -> None:
        # PyNcclCommunicator has no explicit destroy; dropping the refs is
        # enough for process teardown.
        self._comm.disabled = True
