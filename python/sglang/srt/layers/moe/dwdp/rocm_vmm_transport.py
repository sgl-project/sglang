"""HIP VMM transport for the ROCm DWDP composite-address backend."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.vmm_utils import exchange_posix_fds
from sglang.srt.layers.moe.dwdp import hip_vmm
from sglang.srt.layers.moe.dwdp.common import align_down, align_up
from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    LayerWeightSpecs,
    MnnvlHandleSet,
)

logger = logging.getLogger(__name__)


def _close_fds(fds) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


class RocmVmmTransport:
    def __init__(self):
        self._handle_set: Optional[MnnvlHandleSet] = None
        self._peer_views: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._peer_device_ids: Dict[int, int] = {}
        self._imported_handles: List[int] = []
        self._peer_mappings: List[hip_vmm.VmmMapping] = []
        self._released = False

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        local_params: Dict[Tuple[int, str], torch.Tensor],
        group,
        layout: DwdpExpertLayout,
        device_id: int,
    ) -> RocmVmmTransport:
        transport = cls()
        sorted_keys = sorted(local_params)
        granularity = hip_vmm.get_allocation_granularity(device_id)
        handles: Dict[Tuple[int, str], int] = {}
        sizes: Dict[Tuple[int, str], int] = {}

        try:
            for layer_idx, name in sorted_keys:
                param = local_params[(layer_idx, name)]
                spec = layer_weight_specs[layer_idx][name]
                local_start_bytes = layout.local_expert_start * spec.expert_bytes
                local_end_bytes = layout.local_expert_end * spec.expert_bytes
                page_start = align_down(local_start_bytes, granularity)
                page_end = align_up(local_end_bytes, granularity)
                physical_size = page_end - page_start
                data_offset = local_start_bytes - page_start

                handle = hip_vmm.create_shareable_handle(physical_size, device_id)
                handles[(layer_idx, name)] = handle
                sizes[(layer_idx, name)] = physical_size
                with hip_vmm.map_handles(
                    [handle],
                    [physical_size],
                    device_id,
                    alignment=granularity,
                ) as mapping:
                    mapped = hip_vmm.tensor_from_ptr(
                        mapping.address + data_offset,
                        tuple(param.shape),
                        param.dtype,
                        device_id,
                    )
                    mapped.copy_(param)
                    torch.cuda.synchronize(device_id)
                    del mapped

            transport._handle_set = MnnvlHandleSet(handles=handles, sizes=sizes)
            transport._import_peer_views(
                sorted_keys,
                layer_weight_specs,
                group,
                layout,
                device_id,
                granularity,
            )
            dist.barrier(group=group.device_group)
            return transport
        except Exception:
            transport._handle_set = MnnvlHandleSet(handles=handles, sizes=sizes)
            transport.release()
            raise

    def _import_peer_views(
        self,
        sorted_keys: List[Tuple[int, str]],
        layer_weight_specs: LayerWeightSpecs,
        group,
        layout: DwdpExpertLayout,
        device_id: int,
        granularity: int,
    ) -> None:
        local_fds = [
            hip_vmm.export_fd(self._handle_set.get_handle(layer_idx, name))
            for layer_idx, name in sorted_keys
        ]
        all_device_ids = [None] * layout.dwdp_size
        dist.all_gather_object(
            all_device_ids,
            device_id,
            group=group.cpu_group,
        )
        self._peer_device_ids = {
            rank: int(peer_device_id)
            for rank, peer_device_id in enumerate(all_device_ids)
        }
        key_counts = [None] * layout.dwdp_size
        dist.all_gather_object(key_counts, len(sorted_keys), group=group.cpu_group)
        if any(count != len(sorted_keys) for count in key_counts):
            _close_fds(local_fds)
            raise RuntimeError(f"Mismatched ROCm DWDP VMM handle counts: {key_counts}")

        peer_fds = exchange_posix_fds(
            group.cpu_group,
            layout.dwdp_rank,
            layout.dwdp_size,
            local_fds,
            key_counts,
        )
        try:
            for key_index, (layer_idx, name) in enumerate(sorted_keys):
                spec = layer_weight_specs[layer_idx][name]
                for peer_rank in range(layout.dwdp_size):
                    if peer_rank == layout.dwdp_rank:
                        continue
                    peer_handle = hip_vmm.import_fd(peer_fds[(peer_rank, key_index)])
                    self._imported_handles.append(peer_handle)

                    peer_start, peer_end = layout.peer_ranges[peer_rank]
                    peer_start_bytes = peer_start * spec.expert_bytes
                    peer_end_bytes = peer_end * spec.expert_bytes
                    peer_page_start = align_down(peer_start_bytes, granularity)
                    peer_page_end = align_up(peer_end_bytes, granularity)
                    physical_size = peer_page_end - peer_page_start
                    data_offset = peer_start_bytes - peer_page_start

                    mapping = hip_vmm.map_handles(
                        [peer_handle],
                        [physical_size],
                        device_id,
                        alignment=granularity,
                    )
                    self._peer_mappings.append(mapping)
                    self._peer_views[(peer_rank, layer_idx, name)] = (
                        hip_vmm.tensor_from_ptr(
                            mapping.address + data_offset,
                            (peer_end - peer_start,) + spec.full_shape[1:],
                            spec.dtype,
                            device_id,
                        )
                    )
        finally:
            _close_fds(local_fds)
            _close_fds(peer_fds.values())

    @property
    def handle_set(self) -> MnnvlHandleSet:
        if self._handle_set is None:
            raise RuntimeError("ROCm DWDP VMM transport is not initialized")
        return self._handle_set

    @property
    def peer_views(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        return self._peer_views

    @property
    def peer_device_ids(self) -> Dict[int, int]:
        return self._peer_device_ids

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._peer_views.clear()
        self._peer_device_ids.clear()
        for mapping in reversed(self._peer_mappings):
            mapping.close()
        self._peer_mappings.clear()
        for handle in reversed(self._imported_handles):
            hip_vmm.release_handle(handle)
        self._imported_handles.clear()
        if self._handle_set is not None:
            for handle in reversed(list(self._handle_set.handles.values())):
                hip_vmm.release_handle(handle)
            self._handle_set = None
