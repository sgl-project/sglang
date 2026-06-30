"""DWDPTransport: handle allocation, weight copy, exchange, and peer view creation.

Three-phase protocol:
  Phase 1: alloc fabric handle -> copy local weights -> free originals -> export -> allgather
  Phase 2: import peer handles -> create read-only tensor views
  Phase 3: barrier + cleanup

Ported from TensorRT-LLM ``_torch/modules/dwdp/transport.py``.
Uses ``torch.distributed`` (not MPI) for collective communication.
Supports both FABRIC (aarch64/GB200) and POSIX_FD (x86_64/B200/H100) paths.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.dwdp.specs import (
    DwdpExpertLayout,
    LayerWeightSpecs,
    MnnvlHandleSet,
    PeerRanges,
)
from sglang.srt.layers.moe.dwdp.vmm import (
    align_down,
    align_up,
    check_cu_result,
    create_fabric_handle,
    export_handle,
    free_va,
    get_allocation_granularity,
    import_handle,
    map_handle,
    peer_handle_type,
    release_handle,
    reserve_va,
    set_access,
    tensor_from_ptr,
    unmap_va,
)

logger = logging.getLogger(__name__)

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda  # type: ignore[no-redef]

from sglang.srt.layers.moe.dwdp.vmm import _to_devptr

# ---------------------------------------------------------------------------
# pidfd syscalls for POSIX_FD path (Linux 5.6+)
# ---------------------------------------------------------------------------
_SYS_pidfd_open = 434
_SYS_pidfd_getfd = 438


def _pidfd_open(pid: int) -> int:
    """Open a pidfd for the given process."""
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(_SYS_pidfd_open, pid, 0)
    if fd < 0:
        err = ctypes.get_errno()
        raise RuntimeError(
            f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}"
        )
    return fd


def _pidfd_getfd(pidfd: int, remote_fd: int) -> int:
    """Dup a file descriptor from a peer process into our fd table."""
    libc = ctypes.CDLL(None, use_errno=True)
    local_fd = libc.syscall(_SYS_pidfd_getfd, pidfd, remote_fd, 0)
    if local_fd < 0:
        err = ctypes.get_errno()
        msg = (
            f"pidfd_getfd(pidfd={pidfd}, fd={remote_fd}) failed with errno "
            f"{err}: {os.strerror(err)}."
        )
        if err == 1:  # EPERM
            msg += (
                " Permission denied. If running in a container, try adding "
                "--cap-add=SYS_PTRACE to your docker run command."
            )
        else:
            msg += " This may be due to kernel version (requires Linux 5.6+)."
        raise RuntimeError(msg)
    return local_fd


class DWDPTransport:
    """One-time setup: allocate fabric handles, copy weights, exchange with peers."""

    __slots__ = (
        "_handle_set",
        "_peer_views",
        "_peer_ranges",
        "_local_start",
        "_local_end",
        "_dwdp_rank",
        "_dwdp_size",
        "_device_id",
        "_imported_handles",
        "_peer_va_regions",
    )

    def __init__(self):
        self._handle_set: Optional[MnnvlHandleSet] = None
        self._peer_views: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._peer_ranges: PeerRanges = []
        self._local_start = 0
        self._local_end = 0
        self._dwdp_rank = 0
        self._dwdp_size = 0
        self._device_id = 0
        self._imported_handles: List[int] = []
        self._peer_va_regions: List[Tuple[int, int]] = []

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        local_params: Dict[Tuple[int, str], torch.Tensor],
        group: dist.ProcessGroup,
        layout: DwdpExpertLayout,
        device_id: int,
    ) -> DWDPTransport:
        transport = cls()
        transport._dwdp_rank = layout.dwdp_rank
        transport._dwdp_size = layout.dwdp_size
        transport._local_start = layout.local_expert_start
        transport._local_end = layout.local_expert_end
        transport._device_id = device_id
        transport._peer_ranges = layout.peer_ranges

        granularity = get_allocation_granularity(device_id)

        handles: Dict[Tuple[int, str], int] = {}
        sizes: Dict[Tuple[int, str], int] = {}
        all_exports: Dict[Tuple[int, str], list] = {}

        # Detect handle type
        ht = peer_handle_type()
        is_posix_fd = (
            ht
            == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )

        # POSIX_FD: track pidfds and local export fds
        peer_pidfds: Dict[int, int] = {}
        local_export_fds: List[int] = []

        sorted_keys = sorted(local_params.keys())

        try:
            # --------------------------------------------------------------
            # Phase 0 (POSIX_FD only): exchange PIDs and open pidfds
            # --------------------------------------------------------------
            if is_posix_fd:
                all_pids = [None] * layout.dwdp_size
                dist.all_gather_object(all_pids, os.getpid(), group=group.device_group)
                for peer_rank, peer_pid in enumerate(all_pids):
                    if peer_rank == layout.dwdp_rank:
                        continue
                    peer_pidfds[peer_rank] = _pidfd_open(peer_pid)
                logger.info(
                    f"[Transport] POSIX_FD: opened pidfds for {len(peer_pidfds)} peers"
                )

            # --------------------------------------------------------------
            # Phase 1: alloc handle, copy weights, export, allgather
            # --------------------------------------------------------------
            for layer_idx, name in sorted_keys:
                param = local_params[(layer_idx, name)]
                spec = layer_weight_specs[layer_idx][name]

                local_start_bytes = layout.local_expert_start * spec.expert_bytes
                local_end_bytes = layout.local_expert_end * spec.expert_bytes
                page_start = align_down(local_start_bytes, granularity)
                page_end = align_up(local_end_bytes, granularity)
                phys_size = page_end - page_start
                data_offset = local_start_bytes - page_start

                handle = create_fabric_handle(phys_size, device_id)

                temp_va = reserve_va(phys_size, granularity)
                try:
                    map_handle(temp_va, phys_size, handle)
                    set_access(temp_va, phys_size, device_id)

                    nbytes = param.numel() * param.element_size()
                    check_cu_result(
                        cuda.cuMemcpyDtoD(
                            _to_devptr(temp_va + data_offset),
                            _to_devptr(param.data_ptr()),
                            nbytes,
                        )
                    )
                    torch.cuda.synchronize()

                    unmap_va(temp_va, phys_size)
                finally:
                    free_va(temp_va, phys_size)

                param.untyped_storage().resize_(0)
                torch.cuda.empty_cache()

                exported = export_handle(handle)

                # POSIX_FD: track the exported fd (must stay open for peers)
                if is_posix_fd:
                    local_export_fds.append(int(exported))

                all_exported = [None] * layout.dwdp_size
                dist.all_gather_object(all_exported, exported, group=group.device_group)

                handles[(layer_idx, name)] = handle
                sizes[(layer_idx, name)] = phys_size
                all_exports[(layer_idx, name)] = all_exported

                logger.debug(
                    f"[Transport] Phase 1: layer={layer_idx}, name={name}, "
                    f"phys_size={phys_size}, data_offset={data_offset}"
                )

            transport._handle_set = MnnvlHandleSet(handles=handles, sizes=sizes)

            # --------------------------------------------------------------
            # Phase 2: import peer handles, create tensor views
            # --------------------------------------------------------------
            for layer_idx, name in sorted_keys:
                spec = layer_weight_specs[layer_idx][name]

                for peer_rank in range(layout.dwdp_size):
                    if peer_rank == layout.dwdp_rank:
                        continue

                    peer_exported = all_exports[(layer_idx, name)][peer_rank]

                    # Import handle (POSIX_FD needs pidfd_getfd first)
                    if is_posix_fd:
                        local_fd = _pidfd_getfd(
                            peer_pidfds[peer_rank], int(peer_exported)
                        )
                        try:
                            peer_handle = import_handle(local_fd)
                        finally:
                            try:
                                os.close(local_fd)
                            except OSError:
                                pass
                    else:
                        peer_handle = import_handle(peer_exported)

                    transport._imported_handles.append(peer_handle)

                    peer_start = layout.peer_ranges[peer_rank][0]
                    peer_end = layout.peer_ranges[peer_rank][1]
                    peer_start_bytes = peer_start * spec.expert_bytes
                    peer_end_bytes = peer_end * spec.expert_bytes
                    peer_page_start = align_down(peer_start_bytes, granularity)
                    peer_page_end = align_up(peer_end_bytes, granularity)
                    peer_phys_size = peer_page_end - peer_page_start
                    peer_data_offset = peer_start_bytes - peer_page_start

                    peer_va = reserve_va(peer_phys_size, granularity)
                    map_handle(peer_va, peer_phys_size, peer_handle)
                    set_access(peer_va, peer_phys_size, device_id)

                    transport._peer_va_regions.append((peer_va, peer_phys_size))

                    num_peer_experts = peer_end - peer_start
                    expert_shape = spec.full_shape[1:]
                    view_shape = (num_peer_experts,) + expert_shape
                    view_ptr = peer_va + peer_data_offset

                    peer_tensor = tensor_from_ptr(
                        ptr=view_ptr,
                        shape=view_shape,
                        dtype=spec.dtype,
                        device_id=device_id,
                    )
                    transport._peer_views[(peer_rank, layer_idx, name)] = peer_tensor

            # --------------------------------------------------------------
            # Phase 3: barrier + cleanup pidfds and export fds
            # --------------------------------------------------------------
            dist.barrier(group=group.device_group)

            for pidfd in peer_pidfds.values():
                try:
                    os.close(pidfd)
                except OSError:
                    pass
            peer_pidfds.clear()

            for fd in local_export_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            local_export_fds.clear()

            logger.info(
                f"[Transport] Complete: rank={layout.dwdp_rank}/{layout.dwdp_size}, "
                f"{len(handles)} handles, {len(transport._peer_views)} peer views"
            )

            return transport

        except Exception:
            # Cleanup on failure
            for pidfd in peer_pidfds.values():
                try:
                    os.close(pidfd)
                except OSError:
                    pass
            for fd in local_export_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            transport.release()
            raise

    @property
    def handle_set(self) -> MnnvlHandleSet:
        assert self._handle_set is not None
        return self._handle_set

    @property
    def peer_views(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        return self._peer_views

    @property
    def peer_ranges(self) -> PeerRanges:
        return self._peer_ranges

    def release(self) -> None:
        for va, size in self._peer_va_regions:
            try:
                unmap_va(va, size)
                free_va(va, size)
            except Exception:
                pass
        self._peer_va_regions.clear()

        for h in self._imported_handles:
            try:
                release_handle(h)
            except Exception:
                pass
        self._imported_handles.clear()

        self._peer_views.clear()
