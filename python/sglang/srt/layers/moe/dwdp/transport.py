# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Cross-rank expert weight handle exchange (FABRIC or POSIX fd) and peer view import."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
try:
    from cuda.bindings import driver as cuda
except ImportError:
    # For non-cuda platform.
    cuda = None

from sglang.srt.distributed.device_communicators.vmm_utils import (
    check_drv,
    exchange_posix_fds,
    export_shareable_handles,
    import_peer_handle,
)
from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    LayerWeightSpecs,
    MnnvlHandleSet,
)
from sglang.srt.layers.moe.dwdp.vmm import (
    align_down,
    align_up,
    create_fabric_handle,
    free_va,
    get_allocation_granularity,
    map_handle,
    release_handle,
    reserve_va,
    set_access,
    tensor_from_ptr,
    unmap_va,
)

logger = logging.getLogger(__name__)


def _close_fds(fds) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _copy_local_weights_to_handles(
    sorted_keys: List[Tuple[int, str]],
    local_params: Dict[Tuple[int, str], torch.Tensor],
    layer_weight_specs: LayerWeightSpecs,
    layout: DwdpExpertLayout,
    device_id: int,
) -> Tuple[Dict[Tuple[int, str], int], Dict[Tuple[int, str], int]]:
    granularity = get_allocation_granularity(device_id)
    handles: Dict[Tuple[int, str], int] = {}
    sizes: Dict[Tuple[int, str], int] = {}

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
        map_handle(temp_va, phys_size, handle)
        set_access(temp_va, phys_size, device_id)

        nbytes = param.numel() * param.element_size()
        check_drv(
            cuda.cuMemcpyDtoD(temp_va + data_offset, param.data_ptr(), nbytes),
            "cuMemcpyDtoD",
        )
        torch.cuda.synchronize()

        unmap_va(temp_va, phys_size)
        free_va(temp_va, phys_size)

        param.untyped_storage().resize_(0)

        handles[(layer_idx, name)] = handle
        sizes[(layer_idx, name)] = phys_size

        logger.debug(
            f"Phase 1: layer={layer_idx}, name={name}, "
            f"phys_size={phys_size}, data_offset={data_offset}"
        )

    torch.cuda.empty_cache()

    return handles, sizes


class DWDPTransport:
    def __init__(self):
        self._handle_set: Optional[MnnvlHandleSet] = None
        self._peer_views: Dict[Tuple[int, int, str], torch.Tensor] = {}
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
        sorted_keys = sorted(local_params.keys())

        handles, sizes = _copy_local_weights_to_handles(
            sorted_keys, local_params, layer_weight_specs, layout, device_id
        )
        transport._handle_set = MnnvlHandleSet(handles=handles, sizes=sizes)

        transport._import_peer_views(
            sorted_keys, layer_weight_specs, group, layout, device_id
        )

        dist.barrier(group=group.device_group)
        logger.debug(
            f"Transport complete: rank={layout.dwdp_rank}/{layout.dwdp_size}, "
            f"{len(handles)} handles, {len(transport._peer_views)} peer views"
        )
        return transport

    def _import_peer_views(
        self,
        sorted_keys: List[Tuple[int, str]],
        layer_weight_specs: LayerWeightSpecs,
        group: dist.ProcessGroup,
        layout: DwdpExpertLayout,
        device_id: int,
    ) -> None:
        cpu_group = group.cpu_group
        granularity = get_allocation_granularity(device_id)

        handle_list = [self._handle_set.get_handle(li, n) for li, n in sorted_keys]
        fabric_handles, local_posix_fds, use_fabric = export_shareable_handles(
            handle_list, cpu_group, layout.dwdp_rank
        )
        peer_fds: Dict[Tuple[int, int], int] = {}

        key_counts = [None] * layout.dwdp_size
        dist.all_gather_object(key_counts, len(sorted_keys), group=cpu_group)
        if any(count != len(sorted_keys) for count in key_counts):
            raise RuntimeError(
                f"Mismatched DWDP weight handle counts across ranks: {key_counts}"
            )

        if use_fabric:
            all_fabric = [None] * layout.dwdp_size
            dist.all_gather_object(all_fabric, fabric_handles, group=cpu_group)
        else:
            all_fabric = None
            peer_fds = exchange_posix_fds(
                cpu_group,
                layout.dwdp_rank,
                layout.dwdp_size,
                local_posix_fds,
                key_counts,
            )
        logger.info(
            "DWDP handle exchange via %s (%d handles)",
            "FABRIC" if use_fabric else "POSIX fd",
            len(sorted_keys),
        )

        for key_idx, (layer_idx, name) in enumerate(sorted_keys):
            spec = layer_weight_specs[layer_idx][name]

            for peer_rank in range(layout.dwdp_size):
                if peer_rank == layout.dwdp_rank:
                    continue

                fabric_handle = all_fabric[peer_rank][key_idx] if use_fabric else None
                fd = None if use_fabric else peer_fds[(peer_rank, key_idx)]
                peer_handle = import_peer_handle(
                    fabric_handle, fd, use_fabric=use_fabric, peer_rank=peer_rank
                )
                self._imported_handles.append(int(peer_handle))

                peer_start, peer_end = layout.peer_ranges[peer_rank]
                peer_start_bytes = peer_start * spec.expert_bytes
                peer_end_bytes = peer_end * spec.expert_bytes
                peer_page_start = align_down(peer_start_bytes, granularity)
                peer_page_end = align_up(peer_end_bytes, granularity)
                peer_phys_size = peer_page_end - peer_page_start
                peer_data_offset = peer_start_bytes - peer_page_start

                peer_va = reserve_va(peer_phys_size, granularity)
                map_handle(peer_va, peer_phys_size, int(peer_handle))
                set_access(peer_va, peer_phys_size, device_id)
                self._peer_va_regions.append((peer_va, peer_phys_size))

                num_peer_experts = peer_end - peer_start
                peer_tensor = tensor_from_ptr(
                    ptr=peer_va + peer_data_offset,
                    shape=(num_peer_experts,) + spec.full_shape[1:],
                    dtype=spec.dtype,
                    device_id=device_id,
                )
                self._peer_views[(peer_rank, layer_idx, name)] = peer_tensor

        _close_fds(local_posix_fds)
        _close_fds(peer_fds.values())

    @property
    def handle_set(self) -> MnnvlHandleSet:
        assert self._handle_set is not None
        return self._handle_set

    @property
    def peer_views(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        return self._peer_views

    def release(self) -> None:
        for va, size in self._peer_va_regions:
            unmap_va(va, size)
            free_va(va, size)
        self._peer_va_regions.clear()

        for h in self._imported_handles:
            release_handle(h)
        self._imported_handles.clear()

        self._peer_views.clear()
