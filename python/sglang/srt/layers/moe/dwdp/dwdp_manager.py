"""DWDP Manager — expert layout, IPC handle exchange, and weight view assembly."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Weight parameter names tracked per MoE layer
WEIGHT_PARAM_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_sf",
    "w2_weight_sf",
    "w1_alpha",
    "w2_alpha",
)


# ---------------------------------------------------------------------------
# Expert layout
# ---------------------------------------------------------------------------


@dataclass
class DwdpExpertLayout:
    """Defines expert-to-rank mapping for DWDP.

    Supports overlapping allocation: when ``num_experts_per_worker`` exceeds
    ``num_routed_experts // dwdp_size``, expert ranges across ranks overlap,
    reducing NVLink prefetch volume at the cost of extra local memory.
    """

    num_routed_experts: int
    dwdp_size: int
    dwdp_rank: int
    num_experts_per_worker: int  # experts stored locally per rank

    def __post_init__(self):
        assert self.num_experts_per_worker >= self.num_routed_experts // self.dwdp_size, (
            f"num_experts_per_worker ({self.num_experts_per_worker}) must be >= "
            f"num_routed_experts // dwdp_size ({self.num_routed_experts // self.dwdp_size})"
        )
        assert self.num_experts_per_worker <= self.num_routed_experts

    @property
    def num_prefetch_experts(self) -> int:
        """Number of experts to prefetch from each peer."""
        return math.ceil(
            (self.num_routed_experts - self.num_experts_per_worker)
            / (self.dwdp_size - 1)
        )

    @property
    def local_expert_start(self) -> int:
        return min(
            self.num_prefetch_experts * self.dwdp_rank,
            self.num_routed_experts - self.num_experts_per_worker,
        )

    @property
    def local_expert_end(self) -> int:
        return self.local_expert_start + self.num_experts_per_worker

    def peer_expert_range(self, peer_rank: int) -> Tuple[int, int]:
        """Return (start, end) of the expert range owned by *peer_rank*."""
        peer_start = min(
            self.num_prefetch_experts * peer_rank,
            self.num_routed_experts - self.num_experts_per_worker,
        )
        return peer_start, peer_start + self.num_experts_per_worker

    def get_prefetch_src_offset(self, peer_rank: int) -> int:
        """Offset (in number of experts) into peer's local tensor for prefetch."""
        peer_start, peer_end = self.peer_expert_range(peer_rank)
        if self.dwdp_rank < peer_rank:
            # Need the tail of the peer's experts
            prefetch_start = peer_end - self.num_prefetch_experts
        else:
            # Need the head of the peer's experts
            prefetch_start = peer_start
        return prefetch_start - peer_start


# ---------------------------------------------------------------------------
# Weight view returned to the kernel
# ---------------------------------------------------------------------------


@dataclass
class NvFp4WeightView:
    """Multi-B weight tensor bundle for CuteDSL kernel dispatch.

    Each list has ``dwdp_size`` entries ordered by rank. The entry at
    ``dwdp_rank`` is the local model weight tensor; others are prefetch
    buffer views.
    """

    w13_weights: List[torch.Tensor]
    w2_weights: List[torch.Tensor]
    w13_weight_sfs: List[torch.Tensor]
    w2_weight_sfs: List[torch.Tensor]
    w1_alphas: List[torch.Tensor]
    w2_alphas: List[torch.Tensor]


# ---------------------------------------------------------------------------
# Per-layer IPC handle collector
# ---------------------------------------------------------------------------


class DwdpLayerHandleCollector:
    """Manages CUDA IPC handles for one MoE layer's weight tensors."""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.local_weights: Dict[str, torch.Tensor] = {}
        # peer_weights[peer_rank][param_name] -> tensor backed by IPC mapping
        self.peer_base_ptrs: Dict[Tuple[int, str], int] = {}
        self._ipc_mappings: List[int] = []  # base ptrs to close on cleanup

    def register(self, **kwargs: torch.Tensor) -> None:
        """Register local weight tensors for this layer."""
        for name in WEIGHT_PARAM_NAMES:
            if name in kwargs:
                self.local_weights[name] = kwargs[name]

    def get_ipc_handles(self) -> Dict[str, Tuple[bytes, int]]:
        """Return (handle_bytes, offset) for each local weight tensor."""
        from cuda import cudart
        from cuda import cuda as cuda_driver

        handles = {}
        for name, tensor in self.local_weights.items():
            data_ptr = tensor.data_ptr()
            err, handle = cudart.cudaIpcGetMemHandle(data_ptr)
            assert err == cudart.cudaError_t.cudaSuccess, f"cudaIpcGetMemHandle failed: {err}"
            err, alloc_base, alloc_size = cuda_driver.cuMemGetAddressRange(data_ptr)
            assert err == cuda_driver.CUresult.CUDA_SUCCESS, f"cuMemGetAddressRange failed: {err}"
            offset = data_ptr - int(alloc_base)
            handles[name] = (bytes(handle), offset)
        return handles

    def open_peer_handles(
        self,
        all_handles: List[Dict[str, Tuple[bytes, int]]],
        dwdp_rank: int,
    ) -> None:
        """Open peer IPC handles and compute NVLink-accessible pointers."""
        from cuda import cudart

        for peer_rank, peer_handles in enumerate(all_handles):
            if peer_rank == dwdp_rank:
                continue
            for name, (handle_bytes, offset) in peer_handles.items():
                # Reconstruct cudaIpcMemHandle_t from bytes
                handle = cudart.cudaIpcMemHandle_t()
                handle.reserved = list(handle_bytes)
                err, base_ptr = cudart.cudaIpcOpenMemHandle(
                    handle, cudart.cudaIpcMemLazyEnablePeerAccess
                )
                assert err == cudart.cudaError_t.cudaSuccess, (
                    f"cudaIpcOpenMemHandle failed for peer {peer_rank} param {name}: {err}"
                )
                self._ipc_mappings.append(int(base_ptr))
                self.peer_base_ptrs[(peer_rank, name)] = int(base_ptr) + offset

    def cleanup(self) -> None:
        """Close all IPC memory mappings."""
        from cuda import cudart

        for base_ptr in self._ipc_mappings:
            cudart.cudaIpcCloseMemHandle(base_ptr)
        self._ipc_mappings.clear()
        self.peer_base_ptrs.clear()


# ---------------------------------------------------------------------------
# DwdpManager — global singleton
# ---------------------------------------------------------------------------


class DwdpManager:
    """Orchestrates the DWDP lifecycle: weight registration, IPC exchange,
    prefetch buffer management, and weight view assembly."""

    def __init__(
        self,
        dwdp_size: int,
        dwdp_rank: int,
        num_routed_experts: int,
        num_moe_layers: int,
        first_k_dense_replace: int,
        total_num_layers: int,
        num_experts_per_worker: Optional[int] = None,
    ):
        self.dwdp_size = dwdp_size
        self.dwdp_rank = dwdp_rank
        self.num_moe_layers = num_moe_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.total_num_layers = total_num_layers

        if num_experts_per_worker is None:
            num_experts_per_worker = num_routed_experts // dwdp_size

        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed_experts,
            dwdp_size=dwdp_size,
            dwdp_rank=dwdp_rank,
            num_experts_per_worker=num_experts_per_worker,
        )

        # Per-layer handle collectors, keyed by absolute layer_id
        self.layer_handles: Dict[int, DwdpLayerHandleCollector] = {}

        # Prefetch buffer (created after IPC exchange)
        self._prefetch_buffer = None

        # Mapping from absolute layer_id to moe_layer_index (0-based)
        self._layer_id_to_moe_idx: Dict[int, int] = {}
        moe_idx = 0
        for layer_id in range(total_num_layers):
            if layer_id >= first_k_dense_replace:
                self._layer_id_to_moe_idx[layer_id] = moe_idx
                moe_idx += 1

        logger.info(
            f"DwdpManager initialized: dwdp_size={dwdp_size}, rank={dwdp_rank}, "
            f"num_routed_experts={num_routed_experts}, "
            f"num_experts_per_worker={self.layout.num_experts_per_worker}, "
            f"num_prefetch_experts={self.layout.num_prefetch_experts}, "
            f"local_expert_range=[{self.layout.local_expert_start}, {self.layout.local_expert_end}), "
            f"num_moe_layers={num_moe_layers}, first_k_dense={first_k_dense_replace}"
        )

    @property
    def expert_layout(self) -> DwdpExpertLayout:
        return self.layout

    # ----- Phase 2: Weight Registration -----

    def register_layer_weights(self, layer_id: int, **weight_tensors: torch.Tensor) -> None:
        """Called from process_weights_after_loading() for each MoE layer."""
        if layer_id not in self.layer_handles:
            self.layer_handles[layer_id] = DwdpLayerHandleCollector(layer_id)
        self.layer_handles[layer_id].register(**weight_tensors)

    # ----- Phase 3: IPC Exchange -----

    def exchange_ipc_handles(self) -> None:
        """AllGather IPC handles across DWDP group and open peer mappings."""
        from sglang.srt.distributed.parallel_state import get_dwdp_group

        group = get_dwdp_group()

        for layer_id, collector in self.layer_handles.items():
            local_handles = collector.get_ipc_handles()
            # AllGather handles across DWDP group
            all_handles = [None] * self.dwdp_size
            dist.all_gather_object(all_handles, local_handles, group=group.cpu_group)
            collector.open_peer_handles(all_handles, self.dwdp_rank)

        logger.info(
            f"DWDP IPC handles exchanged for {len(self.layer_handles)} MoE layers"
        )

    def init_prefetch_buffers(self) -> None:
        """Allocate double-buffered prefetch buffers."""
        from sglang.srt.layers.moe.dwdp.prefetch_buffer import DwdpPrefetchBuffer

        # Collect param shapes/dtypes from the first registered layer
        first_collector = next(iter(self.layer_handles.values()))
        param_shapes = {}
        param_dtypes = {}
        for name, tensor in first_collector.local_weights.items():
            # Shape per expert: tensor shape is [num_experts_per_worker, ...]
            param_shapes[name] = tensor.shape[1:]  # drop expert dim
            param_dtypes[name] = tensor.dtype

        self._prefetch_buffer = DwdpPrefetchBuffer(
            layout=self.layout,
            num_moe_layers=self.num_moe_layers,
            param_shapes=param_shapes,
            param_dtypes=param_dtypes,
            device=next(iter(first_collector.local_weights.values())).device,
        )
        logger.info("DWDP prefetch buffers allocated")

    def initialize_compute_events(self) -> None:
        """Pre-record initial compute events so the first prefetch can proceed."""
        assert self._prefetch_buffer is not None
        self._prefetch_buffer.initialize_compute_events()

    # ----- Phase 4: Forward Pass Operations -----

    def prefetch_first_layers(self) -> None:
        """Async prefetch weights for the first 2 MoE layers.

        Called at the start of forward_extend(). Dense layers 0..first_k_dense_replace-1
        provide the compute overlap window.
        """
        assert self._prefetch_buffer is not None
        # Prefetch moe_layer_idx 0 -> buf[0], moe_layer_idx 1 -> buf[1]
        for moe_idx in range(min(2, self.num_moe_layers)):
            layer_id = self.first_k_dense_replace + moe_idx
            if layer_id in self.layer_handles:
                self._prefetch_buffer.prefetch_layer(
                    moe_layer_idx=moe_idx,
                    layer_handles=self.layer_handles[layer_id],
                )

    def get_weight_view(self, layer_id: int) -> NvFp4WeightView:
        """Wait for prefetch to complete and return assembled weight view.

        Called from DeepseekV2MoE.forward_dwdp() before the MoE kernel.
        """
        moe_idx = self._layer_id_to_moe_idx[layer_id]
        collector = self.layer_handles[layer_id]

        # Wait for prefetch to complete
        self._prefetch_buffer.wait_for_prefetch(moe_idx)

        # Assemble weight view: local weights + prefetch buffer views
        buffer_views = self._prefetch_buffer.get_buffer_views(moe_idx)

        def _build_weight_list(param_name: str) -> List[torch.Tensor]:
            result = []
            for rank in range(self.dwdp_size):
                if rank == self.dwdp_rank:
                    result.append(collector.local_weights[param_name])
                else:
                    result.append(buffer_views[param_name][rank])
            return result

        return NvFp4WeightView(
            w13_weights=_build_weight_list("w13_weight"),
            w2_weights=_build_weight_list("w2_weight"),
            w13_weight_sfs=_build_weight_list("w13_weight_sf"),
            w2_weight_sfs=_build_weight_list("w2_weight_sf"),
            w1_alphas=_build_weight_list("w1_alpha"),
            w2_alphas=_build_weight_list("w2_alpha"),
        )

    def record_compute_and_prefetch_next(self, layer_id: int) -> None:
        """Record compute done event and trigger prefetch for layer_id + 2.

        Called from DeepseekV2MoE.forward_dwdp() after the MoE kernel.
        """
        moe_idx = self._layer_id_to_moe_idx[layer_id]

        # Record compute done on default stream
        self._prefetch_buffer.record_compute_done(moe_idx)

        # Trigger prefetch for moe_idx + 2 (same buffer slot)
        next_moe_idx = moe_idx + 2
        if next_moe_idx < self.num_moe_layers:
            next_layer_id = self.first_k_dense_replace + next_moe_idx
            if next_layer_id in self.layer_handles:
                self._prefetch_buffer.prefetch_layer(
                    moe_layer_idx=next_moe_idx,
                    layer_handles=self.layer_handles[next_layer_id],
                )

    # ----- Phase 5: Cleanup -----

    def cleanup(self) -> None:
        """Release all DWDP resources."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.cleanup()
            self._prefetch_buffer = None
        for collector in self.layer_handles.values():
            collector.cleanup()
        self.layer_handles.clear()
        logger.info("DWDP resources cleaned up")
