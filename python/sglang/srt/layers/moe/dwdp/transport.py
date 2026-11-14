# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Cross-rank expert weight handle exchange (FABRIC or POSIX fd) and peer view import."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    LayerWeightSpecs,
    MnnvlHandleSet,
)
from sglang.srt.utils.vmm_backend import VmmBackend, get_vmm_backend
from sglang.srt.utils.vmm_common import (
    Reservation,
    align_down,
    align_up,
    all_ranks_ok,
    close_fds_quietly,
)

logger = logging.getLogger(__name__)


def _copy_local_weights_to_handles(
    sorted_keys: List[Tuple[int, str]],
    local_params: Dict[Tuple[int, str], torch.Tensor],
    layer_weight_specs: LayerWeightSpecs,
    layout: DwdpExpertLayout,
    backend: VmmBackend,
) -> Tuple[Dict[Tuple[int, str], int], Dict[Tuple[int, str], int]]:
    granularity = backend.granularity
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

        reservation = backend.make_reservation(
            phys_size, exportable=True, alignment=granularity
        )
        handle = int(reservation.map(0, phys_size, retain_handle=True))

        backend.copy_tensor_to_pointer(reservation.base + data_offset, param)
        backend.synchronize()

        reservation.close(release_handles=False)

        param.untyped_storage().resize_(0)

        handles[(layer_idx, name)] = handle
        sizes[(layer_idx, name)] = phys_size

        logger.debug(
            f"Phase 1: layer={layer_idx}, name={name}, "
            f"phys_size={phys_size}, data_offset={data_offset}"
        )

    backend.empty_cache()

    return handles, sizes


class DWDPTransport:
    def __init__(self, backend: VmmBackend):
        self._backend = backend
        self._handle_set: Optional[MnnvlHandleSet] = None
        self._peer_views: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._imported_handles: List[int] = []
        self._peer_reservations: List[Reservation] = []

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        local_params: Dict[Tuple[int, str], torch.Tensor],
        group: GroupCoordinator,
        layout: DwdpExpertLayout,
        device_id: int,
    ) -> DWDPTransport:
        transport = cls(get_vmm_backend(device_id))
        sorted_keys = sorted(local_params.keys())

        handles, sizes = _copy_local_weights_to_handles(
            sorted_keys, local_params, layer_weight_specs, layout, transport._backend
        )
        transport._handle_set = MnnvlHandleSet(handles=handles, sizes=sizes)

        transport._import_peer_views(sorted_keys, layer_weight_specs, group, layout)

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
        group: GroupCoordinator,
        layout: DwdpExpertLayout,
    ) -> None:
        cpu_group = group.cpu_group
        granularity = self._backend.granularity

        handle_list = [
            self._handle_set.get_handle(layer_idx, name)
            for layer_idx, name in sorted_keys
        ]
        key_counts = [None] * layout.dwdp_size
        dist.all_gather_object(key_counts, len(sorted_keys), group=cpu_group)
        if any(count != len(sorted_keys) for count in key_counts):
            raise RuntimeError(
                f"Mismatched DWDP weight handle counts across ranks: {key_counts}"
            )

        self._check_peer_access(cpu_group, layout)

        use_fabric, peer_handles = self._backend.exchange_handles(
            handle_list, cpu_group, layout.dwdp_rank, layout.dwdp_size
        )
        try:
            logger.info(
                f"DWDP handle exchange via {'FABRIC' if use_fabric else 'POSIX fd'} "
                f"({len(sorted_keys)} handles)"
            )
            import_error: Exception | None = None
            try:
                for key_idx, (layer_idx, name) in enumerate(sorted_keys):
                    spec = layer_weight_specs[layer_idx][name]

                    for peer_rank in range(layout.dwdp_size):
                        if peer_rank == layout.dwdp_rank:
                            continue

                        peer_start, peer_end = layout.peer_ranges[peer_rank]
                        peer_start_bytes = peer_start * spec.expert_bytes
                        peer_end_bytes = peer_end * spec.expert_bytes
                        peer_page_start = align_down(peer_start_bytes, granularity)
                        peer_page_end = align_up(peer_end_bytes, granularity)
                        peer_phys_size = peer_page_end - peer_page_start
                        peer_data_offset = peer_start_bytes - peer_page_start

                        exported = peer_handles[(peer_rank, key_idx)]
                        # not all backends' handles carry the object's size
                        peer_handle = self._backend.import_handle(
                            exported if use_fabric else None,
                            None if use_fabric else exported,
                            use_fabric=use_fabric,
                            peer_rank=peer_rank,
                            size=peer_phys_size,
                        )
                        self._imported_handles.append(int(peer_handle))

                        peer_reservation = self._backend.make_reservation(
                            peer_phys_size,
                            exportable=True,
                            alignment=granularity,
                        )
                        peer_reservation.map_existing(
                            0, peer_phys_size, int(peer_handle)
                        )
                        self._peer_reservations.append(peer_reservation)

                        num_peer_experts = peer_end - peer_start
                        peer_tensor = self._backend.tensor_from_pointer(
                            peer_reservation.base + peer_data_offset,
                            peer_end_bytes - peer_start_bytes,
                            shape=(num_peer_experts,) + spec.full_shape[1:],
                            dtype=spec.dtype,
                        )
                        self._peer_views[(peer_rank, layer_idx, name)] = peer_tensor
            # any failure, not only driver errors, must reach the collective below
            except Exception as error:
                import_error = error
            # a rank failing alone would leave its peers waiting in the barrier
            if not all_ranks_ok(cpu_group, import_error is None):
                raise import_error or RuntimeError(
                    "DWDP peer import failed on another rank"
                )
        finally:
            if not use_fabric:
                close_fds_quietly(list(peer_handles.values()))

    def _check_peer_access(
        self, cpu_group: ProcessGroup, layout: DwdpExpertLayout
    ) -> None:
        # ordinals are process-local and visibility variables can be bypassed, so match peers by UUID
        local_uuids = self._backend.device_uuids()
        peer_uuids = [None] * layout.dwdp_size
        dist.all_gather_object(
            peer_uuids, local_uuids[self._backend.device_id], group=cpu_group
        )
        unreachable_uuids = [
            peer_uuid
            for peer_rank, peer_uuid in enumerate(peer_uuids)
            if peer_rank != layout.dwdp_rank
            and peer_uuid in local_uuids
            and not self._backend.can_access_peer(local_uuids.index(peer_uuid))
        ]
        # a rank that raised alone would leave its peers blocked in the exchange
        if not all_ranks_ok(cpu_group, not unreachable_uuids):
            raise RuntimeError(
                f"DWDP needs peer access between all ranks; devices unreachable "
                f"from device {peer_uuids[layout.dwdp_rank]}: {unreachable_uuids}"
            )

    @property
    def handle_set(self) -> MnnvlHandleSet:
        assert self._handle_set is not None
        return self._handle_set

    @property
    def peer_views(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        return self._peer_views

    def release(self) -> None:
        for reservation in self._peer_reservations:
            reservation.close()
        self._peer_reservations.clear()

        for imported_handle in self._imported_handles:
            self._backend.release_handle(imported_handle)
        self._imported_handles.clear()

        # local shards are mapped by the weight buffer, which is released first
        if self._handle_set is not None:
            for handle in self._handle_set.handles.values():
                self._backend.release_handle(handle)
            self._handle_set = None

        self._peer_views.clear()
