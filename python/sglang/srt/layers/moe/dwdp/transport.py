"""DWDPTransport: handle allocation, weight copy, exchange, and peer view creation.

Three-phase protocol:
  Phase 1: alloc fabric handle → copy local weights → free originals → export → allgather
  Phase 2: import peer handles → create read-only tensor views
  Phase 3: barrier + cleanup

Ported from TensorRT-LLM ``_torch/modules/dwdp/transport.py``.
Uses ``torch.distributed`` (not MPI) for collective communication.
"""

from __future__ import annotations

import logging
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
    align_up,
    create_fabric_handle,
    export_handle,
    free_va,
    get_allocation_granularity,
    import_handle,
    map_handle,
    release_handle,
    reserve_va,
    set_access,
    tensor_from_ptr,
    unmap_va,
)

logger = logging.getLogger(__name__)


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
        """Execute the three-phase transport protocol.

        Args:
            layer_weight_specs: Per-layer weight specs.
            local_params: Mapping (layer_idx, weight_name) -> local weight tensor.
            group: torch.distributed ProcessGroup for the DWDP group.
            layout: Expert layout with local_start/local_end.
            device_id: CUDA device ordinal.

        Returns:
            Initialized DWDPTransport with handles and peer views.
        """
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

        # Sorted iteration order (must match across ranks for allgather sync)
        sorted_keys = sorted(local_params.keys())

        # ----------------------------------------------------------------
        # Phase 1: For each (layer, weight), alloc handle, copy, export, allgather
        # ----------------------------------------------------------------
        for layer_idx, name in sorted_keys:
            param = local_params[(layer_idx, name)]
            spec = layer_weight_specs[layer_idx][name]

            # Compute physical size (page-aligned)
            local_start_bytes = layout.local_expert_start * spec.expert_bytes
            local_end_bytes = layout.local_expert_end * spec.expert_bytes
            from sglang.srt.layers.moe.dwdp.vmm import align_down

            page_start = align_down(local_start_bytes, granularity)
            page_end = align_up(local_end_bytes, granularity)
            phys_size = page_end - page_start
            data_offset = local_start_bytes - page_start

            # Create fabric handle
            handle = create_fabric_handle(phys_size, device_id)

            # Map to temp VA, copy weights in
            temp_va = reserve_va(phys_size, granularity)
            try:
                map_handle(temp_va, phys_size, handle)
                set_access(temp_va, phys_size, device_id)

                # Copy local param into the fabric handle via cuMemcpyDtoD
                from sglang.srt.layers.moe.dwdp.vmm import _to_devptr, check_cu_result

                try:
                    from cuda.bindings import driver as _cuda_drv
                except ImportError:
                    from cuda import cuda as _cuda_drv

                nbytes = param.numel() * param.element_size()
                check_cu_result(
                    _cuda_drv.cuMemcpyDtoD(
                        _to_devptr(temp_va + data_offset),
                        _to_devptr(param.data_ptr()),
                        nbytes,
                    )
                )
                torch.cuda.synchronize()

                # Unmap temp VA
                unmap_va(temp_va, phys_size)
            finally:
                free_va(temp_va, phys_size)

            # Free original param tensor to reclaim memory
            param.untyped_storage().resize_(0)
            torch.cuda.empty_cache()

            # Export handle
            exported = export_handle(handle)

            # Allgather exports across DWDP group
            all_exported = [None] * layout.dwdp_size
            dist.all_gather_object(all_exported, exported, group=group.device_group)

            handles[(layer_idx, name)] = handle
            sizes[(layer_idx, name)] = phys_size
            all_exports[(layer_idx, name)] = all_exported

            logger.debug(
                f"[Transport] Phase 1 done: layer={layer_idx}, name={name}, "
                f"phys_size={phys_size}, data_offset={data_offset}"
            )

        transport._handle_set = MnnvlHandleSet(handles=handles, sizes=sizes)

        # ----------------------------------------------------------------
        # Phase 2: Import peer handles, create tensor views
        # ----------------------------------------------------------------
        for layer_idx, name in sorted_keys:
            spec = layer_weight_specs[layer_idx][name]

            for peer_rank in range(layout.dwdp_size):
                if peer_rank == layout.dwdp_rank:
                    continue

                peer_exported = all_exports[(layer_idx, name)][peer_rank]

                # Import peer handle
                peer_handle = import_handle(peer_exported)
                transport._imported_handles.append(peer_handle)

                # Get peer's physical size
                peer_start = layout.peer_ranges[peer_rank][0]
                peer_end = layout.peer_ranges[peer_rank][1]
                peer_start_bytes = peer_start * spec.expert_bytes
                peer_end_bytes = peer_end * spec.expert_bytes
                from sglang.srt.layers.moe.dwdp.vmm import align_down

                peer_page_start = align_down(peer_start_bytes, granularity)
                peer_page_end = align_up(peer_end_bytes, granularity)
                peer_phys_size = peer_page_end - peer_page_start
                peer_data_offset = peer_start_bytes - peer_page_start

                # Map to VA and create tensor view
                peer_va = reserve_va(peer_phys_size, granularity)
                map_handle(peer_va, peer_phys_size, peer_handle)
                set_access(peer_va, peer_phys_size, device_id)

                transport._peer_va_regions.append((peer_va, peer_phys_size))

                # Create tensor view for the valid expert range
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

        # ----------------------------------------------------------------
        # Phase 3: Barrier
        # ----------------------------------------------------------------
        dist.barrier(group=group.device_group)

        logger.info(
            f"[Transport] Complete: rank={layout.dwdp_rank}/{layout.dwdp_size}, "
            f"{len(handles)} handles, {len(transport._peer_views)} peer views"
        )

        return transport

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
        """Release all imported handles and VA regions."""
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
