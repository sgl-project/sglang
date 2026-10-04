"""All-gather of a vocab-parallel row block across a TP group.

Every implementation takes this rank's ``[rows, local_width]`` slice and returns
``[rows, world_size * local_width]`` with the ranks' slices side by side, the
layout ``GroupCoordinator.all_gather(dim=-1)`` produces.
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


def _unstack(gathered: torch.Tensor, world_size: int) -> torch.Tensor:
    """Rank-major ``[world_size * rows, width]`` -> ``[rows, world_size * width]``."""
    rows = gathered.shape[0] // world_size
    width = gathered.shape[1]
    if rows == 1:
        return gathered.view(1, world_size * width)
    return (
        gathered.view(world_size, rows, width)
        .transpose(0, 1)
        .reshape(rows, world_size * width)
    )


# Collective: every rank of the group must call this, in the same order, outside
# CUDA-graph capture. The returned multicast alias is 0 when the group has none.
def _alloc_symm(
    group, shape: Tuple[int, int], dtype: torch.dtype
) -> Tuple[torch.Tensor, int]:
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

    A slice that fits one slot of the push plane takes the push kernel into a
    fresh tensor; a larger one that fits ``pull_out`` takes the pull kernel into
    that symmetric-memory output, which is reused every call; anything else goes
    to ``fallback``, the NCCL ring. Both kernels gather along the row axis, so
    the ranks come back stacked and are transposed into place. ``pull_out`` is
    allocated here: the allocation is collective and captured graphs keep its
    address.
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
        # Compact argmax partials need rank-major output, no symmetric pull buffer.
        if (
            local.is_contiguous()
            and local.shape[1] * local.element_size() % 16 == 0
            and local.nbytes <= self.slot_bytes
        ):
            return self._push_stacked(local)
        return self.fallback.gather_stacked(local)

    def _push(self, local: torch.Tensor) -> torch.Tensor:
        return _unstack(self._push_stacked(local), self.world_size)

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
        full = _unstack(out, self.world_size)
        # the transpose copies except at one row, where it would alias the
        # shared buffer that the next call overwrites
        return full.clone() if full.data_ptr() == out.data_ptr() else full


class PcieIpcVocabGather(VocabGather):
    """FlashInfer's PCIe-IPC all-gather, for PCIe-only groups (no NVLink, no
    multicast) where the NCCL ring's latency dominates small gathers.

    The kernel copies opaque 16-byte packs, so the result is bit-identical to
    NCCL's. Slices the workspace cannot take (another dtype, more than
    ``max_rows`` rows) go to ``fallback``.
    Construction is collective. The workspace serves one ordered stream and is
    rebound when the caller's stream changes (graph capture, then replay), as
    the PCIe-IPC all-reduce adapter does.
    """

    def __init__(
        self,
        *,
        group,
        local_width: int,
        dtype: torch.dtype,
        max_rows: int,
        fallback: VocabGather,
    ) -> None:
        from flashinfer.comm import (
            PcieIpcAllGatherLaunchConfig,
            PcieIpcAllGatherVariant,
            PcieIpcAllGatherWorkspace,
        )

        self.world_size = int(group.world_size)
        self.fallback = fallback
        self._stream: Optional[torch.cuda.Stream] = None
        self.workspace = PcieIpcAllGatherWorkspace(
            group=group.device_group,
            max_numel=max_rows * local_width,
            dtype=dtype,
        )
        # One fixed launch config: the same on every rank, nothing to tune at
        # startup, and it avoids FlashInfer's seed policy for untuned row counts,
        # which can be slower than NCCL.
        self.config = PcieIpcAllGatherLaunchConfig(
            24, 512, PcieIpcAllGatherVariant.RECURSIVE_DOUBLING
        )
        logger.info(
            "PCIe-IPC vocab gather: up to %d rows x %d (%s)",
            max_rows,
            local_width,
            dtype,
        )

    def __call__(self, local: torch.Tensor) -> torch.Tensor:
        gathered = self._gather(local)
        if gathered is None:
            return self.fallback(local)
        return _unstack(gathered, self.world_size)

    def gather_stacked(self, local: torch.Tensor) -> torch.Tensor:
        gathered = self._gather(local)
        if gathered is None:
            return self.fallback.gather_stacked(local)
        return gathered

    def _gather(self, local: torch.Tensor) -> Optional[torch.Tensor]:
        # The kernel needs a contiguous, 16-byte aligned input. Copying one that
        # is not keeps the choice of path a function of dtype and shape, which
        # every rank shares, rather than of a rank's own pointer.
        if not local.is_contiguous() or local.data_ptr() % 16 != 0:
            local = local.clone(memory_format=torch.contiguous_format)
        if not self.workspace.supports(local):
            return None
        stream = torch.cuda.current_stream()
        if self._stream is None or stream != self._stream:
            self.workspace.rebind_stream()
            self._stream = stream
        return self.workspace.all_gather(local, config=self.config)


def make_pcie_ipc_gather(
    group,
    *,
    local_width: int,
    dtype: torch.dtype,
    max_rows: int,
    fallback: VocabGather,
) -> Optional[PcieIpcVocabGather]:
    """A PCIe-IPC gather when ``group``'s PCIe-IPC all-reduce is enabled, else None.

    Only TP4 is taken, the configuration the launch config was measured on.
    The workspace is released with the all-reduce's, in ``GroupCoordinator.destroy``.
    Collective: every rank of the group must call this, in the same order,
    outside CUDA-graph capture.
    """
    # Settings shared by every rank; the per-rank state is in the agreement below.
    comm = getattr(group, "pcie_ipc_comm", None)
    if comm is None or group.world_size != 4 or max_rows <= 0:
        return None
    # FlashInfer allocates and maps the IPC buffer without telling the other
    # ranks when that fails on one of them, which leaves them waiting. So the
    # ranks first agree that each can build it, and then that each did.
    workspace_bytes = 2 * group.world_size * max_rows * local_width * dtype.itemsize
    ready = not comm.disabled and _can_build_pcie_ipc_gather(workspace_bytes)
    if not _all_ranks_agree(group, ready):
        return None
    gather = None
    try:
        gather = PcieIpcVocabGather(
            group=group,
            local_width=local_width,
            dtype=dtype,
            max_rows=max_rows,
            fallback=fallback,
        )
    except Exception as e:
        logger.warning("PCIe-IPC vocab gather unavailable (%s); using NCCL", e)
    if not _all_ranks_agree(group, gather is not None):
        # Releasing a workspace is collective, so one that only some ranks
        # built is left unused rather than destroyed. FlashInfer raises on
        # every rank alike after its buffer exchange, so this is a safeguard.
        return None
    comm.adopt(gather.workspace)
    return gather


def _can_build_pcie_ipc_gather(workspace_bytes: int) -> bool:
    """This rank's part of the agreement: the API exists and the buffer fits."""
    try:
        from flashinfer.comm import PcieIpcAllGatherWorkspace  # noqa: F401
    except ImportError as e:
        logger.warning("FlashInfer has no PCIe-IPC all-gather (%s); using NCCL", e)
        return False
    free_bytes, _ = torch.cuda.mem_get_info()
    # headroom for the signal words and allocation granularity
    if free_bytes < workspace_bytes + (64 << 20):
        logger.warning(
            "PCIe-IPC vocab gather needs %d MB, %d MB free; using NCCL",
            workspace_bytes >> 20,
            free_bytes >> 20,
        )
        return False
    return True


def _all_ranks_agree(group, ok: bool) -> bool:
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return bool(flag.item())


def _nvlink_ca_comm(group, *, local_width: int, dtype: torch.dtype):
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
    prefer_pcie_ipc: bool = False,
    symm_rows: Optional[int] = None,
) -> VocabGather:
    """The gather for ``group``: local for a group of one, NVLink when the
    group's custom all-reduce has a multicast plane (and ``prefer_nvlink``),
    PCIe-IPC when the group's PCIe-IPC all-reduce is enabled (and
    ``prefer_pcie_ipc``), the NCCL ring otherwise. ``symm_rows`` is the row capacity of the NVLink gather's
    symmetric-memory output for slices past the push slot, and of the PCIe-IPC
    workspace; None sizes it for the server's largest batch, 0 leaves those
    slices to NCCL."""
    if group is None or group.world_size == 1:
        return LocalVocabGather()
    nccl = NcclVocabGather(group)
    if symm_rows is None:
        symm_rows = _default_symm_rows()
    ca_comm = (
        _nvlink_ca_comm(group, local_width=local_width, dtype=dtype)
        if prefer_nvlink
        else None
    )
    if ca_comm is None:
        if not prefer_pcie_ipc:
            return nccl
        pcie_ipc = make_pcie_ipc_gather(
            group,
            local_width=local_width,
            dtype=dtype,
            max_rows=symm_rows,
            fallback=nccl,
        )
        return nccl if pcie_ipc is None else pcie_ipc
    return NVLinkVocabGather(
        ca_comm=ca_comm,
        group=group,
        local_width=local_width,
        dtype=dtype,
        symm_rows=symm_rows,
        fallback=nccl,
    )
