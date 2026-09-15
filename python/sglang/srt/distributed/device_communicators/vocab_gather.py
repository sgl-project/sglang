"""All-gather of a vocab-parallel row block across a TP group.

Every implementation takes this rank's ``[rows, local_width]`` slice and returns
``[rows, world_size * local_width]`` with the ranks' slices side by side, the
layout ``GroupCoordinator.all_gather(dim=-1)`` produces. Callers pick one with
``make_vocab_gather`` at init and call it unconditionally afterwards; which
transport runs is the implementation's business, including any fallback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class VocabGather(ABC):
    """``[rows, local] -> [rows, world_size * local]``, ranks side by side."""

    @abstractmethod
    def __call__(self, local: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def gather_stacked(self, local: torch.Tensor) -> torch.Tensor:
        """Gather compact row blocks as [world_size * rows, local_width]."""
        ...


class LocalVocabGather(VocabGather):
    """A group of one: the slice is the whole row."""

    def __call__(self, local: torch.Tensor) -> torch.Tensor:
        return local

    def gather_stacked(self, local: torch.Tensor) -> torch.Tensor:
        return local


class NcclVocabGather(VocabGather):
    """The group coordinator's all_gather along the last dim (NCCL ring)."""

    def __init__(self, group) -> None:
        self.group = group

    def __call__(self, local: torch.Tensor) -> torch.Tensor:
        return self.group.all_gather(local, dim=-1)

    def gather_stacked(self, local: torch.Tensor) -> torch.Tensor:
        return self.group.all_gather(local, dim=0)


def _alloc_symm(
    group, shape: Tuple[int, int], dtype: torch.dtype
) -> Tuple[torch.Tensor, int]:
    """A symmetric-memory tensor on ``group`` and its multicast alias (0 when
    the group has none). Collective: every rank of the group must call it, in
    the same order, outside CUDA-graph capture."""
    from torch._C._distributed_c10d import _SymmetricMemory

    # a GroupCoordinator names the allocation by its cpu_group, as
    # CustomAllReduceV2 does; a torch process group names it itself
    pg = getattr(group, "cpu_group", group)
    buf = _SymmetricMemory.empty_strided_p2p(
        (shape[0] * shape[1],),
        [1],
        dtype,
        torch.device("cuda", torch.cuda.current_device()),
        pg.group_name,
    )
    mc_ptr = int(_SymmetricMemory.rendezvous(buf).multicast_ptr)
    return buf.view(shape), mc_ptr


class NVLinkVocabGather(VocabGather):
    """The NVLink collectives on CustomAllReduceV2's multicast plane.

    Both kernels gather along the row axis, so the ranks come back stacked and
    are transposed into place. A slice that fits one slot of the push plane
    takes the push kernel into a fresh tensor; a larger one that fits
    ``pull_out`` takes the pull kernel into that symmetric-memory output, which
    is reused every call, so the result is copied out of it; anything else goes
    to ``fallback``, the NCCL ring.

    ``pull_out`` (``[world_size * symm_rows, local_width]``) is allocated here:
    the allocation is collective and captured graphs keep its address, and with
    CUDA graphs on the capture warm-up reaches it at the largest batch anyway.
    """

    def __init__(
        self,
        *,
        ca_comm,
        group,
        local_width: int,
        dtype: torch.dtype,
        symm_rows: int,
        fallback: VocabGather,
    ) -> None:
        self.comm = ca_comm.obj
        self.world_size = int(group.world_size)
        self.slot_bytes = int(ca_comm.max_push_size)
        self.fallback = fallback
        self.pull_out: Optional[torch.Tensor] = None
        self.pull_mc_ptr = 0
        if symm_rows > 0 and self.comm.pull is not None:
            out, mc_ptr = _alloc_symm(
                group, (self.world_size * symm_rows, local_width), dtype
            )
            if mc_ptr != 0:
                self.pull_out, self.pull_mc_ptr = out, mc_ptr
                logger.info(
                    "NVLink vocab gather: pull output %s (%d MB)",
                    tuple(out.shape),
                    out.numel() * out.element_size() >> 20,
                )
            else:
                logger.warning("NVLink vocab gather: no multicast alias, pull path off")

    def __call__(self, local: torch.Tensor) -> torch.Tensor:
        rows = local.shape[0]
        if local.nbytes <= self.slot_bytes:
            return self._push(local)
        total_rows = self.world_size * rows
        if self.pull_out is not None and total_rows <= self.pull_out.shape[0]:
            return self._pull(local, self.pull_out[:total_rows])
        return self.fallback(local)

    def gather_stacked(self, local: torch.Tensor) -> torch.Tensor:
        # Compact argmax partials need rank-major output and no symmetric pull
        # buffer. Unaligned rows and payloads past the push slot use NCCL.
        if (
            local.is_contiguous()
            and local.shape[1] * local.element_size() % 16 == 0
            and local.nbytes <= self.slot_bytes
        ):
            return self._push_stacked(local)
        return self.fallback.gather_stacked(local)

    def _push(self, local: torch.Tensor) -> torch.Tensor:
        return self._unstack(self._push_stacked(local))

    def _push_stacked(self, local: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.communication import nvlink_comm

        rows, width = local.shape
        gathered = torch.empty(
            (self.world_size * rows, width), dtype=local.dtype, device=local.device
        )
        nvlink_comm.all_gather_push(self.comm, local, gathered)
        return gathered

    def _pull(self, local: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.communication import nvlink_comm

        nvlink_comm.all_gather_pull(self.comm, local, out, out_mc_ptr=self.pull_mc_ptr)
        full = self._unstack(out)
        # the transpose copies except at one row, where it would alias the
        # shared buffer that the next call overwrites
        return full.clone() if full.data_ptr() == out.data_ptr() else full

    def _unstack(self, gathered: torch.Tensor) -> torch.Tensor:
        """``[world_size * rows, width]`` stacked by rank -> ``[rows, world_size * width]``."""
        rows = gathered.shape[0] // self.world_size
        width = gathered.shape[1]
        if rows == 1:
            return gathered.view(1, self.world_size * width)
        return (
            gathered.view(self.world_size, rows, width)
            .transpose(0, 1)
            .reshape(rows, self.world_size * width)
        )


def _nvlink_ca_comm(group, *, local_width: int, dtype: torch.dtype):
    """The group's CustomAllReduceV2 when it can carry this gather on its
    multicast plane, else None."""
    ca_comm = getattr(group, "ca_comm", None)
    if ca_comm is None or getattr(ca_comm, "disabled", True):
        return None
    comm = getattr(ca_comm, "obj", None)
    if comm is None or not getattr(ca_comm, "has_multicast", False):
        return None
    if comm.push is None or comm.world_size != group.world_size:
        return None
    # the kernels move 16-byte vectors along the row
    if local_width % (128 // torch.finfo(dtype).bits) != 0:
        return None
    return ca_comm


def _default_symm_rows() -> int:
    """One row per request in the largest batch the server runs (the
    scheduler's max running requests, else the decode graph's max batch); 0
    when the server config is not published (offline use)."""
    try:
        from sglang.srt.runtime_context import get_exec, get_schedule

        return int(
            get_schedule().max_running_requests
            or get_exec().graph.cuda_graph_config.decode.max_bs
            or 0
        )
    except Exception:
        return 0


def make_vocab_gather(
    group,
    *,
    local_width: int,
    dtype: torch.dtype = torch.float32,
    prefer_nvlink: bool = True,
    symm_rows: Optional[int] = None,
) -> VocabGather:
    """The gather for ``group``: local for a group of one, NVLink when the
    group's custom all-reduce has a multicast plane (and ``prefer_nvlink``),
    the NCCL ring otherwise. ``symm_rows`` is the row capacity of the NVLink
    gather's symmetric-memory output for slices past the push slot; None sizes
    it for the server's largest batch, 0 leaves those slices to NCCL."""
    if group is None or group.world_size == 1:
        return LocalVocabGather()
    nccl = NcclVocabGather(group)
    if not prefer_nvlink:
        return nccl
    ca_comm = _nvlink_ca_comm(group, local_width=local_width, dtype=dtype)
    if ca_comm is None:
        return nccl
    return NVLinkVocabGather(
        ca_comm=ca_comm,
        group=group,
        local_width=local_width,
        dtype=dtype,
        symm_rows=_default_symm_rows() if symm_rows is None else symm_rows,
        fallback=nccl,
    )
