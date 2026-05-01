# Copyright 2024-2025 SGLang Team
# Licensed under the Apache License, Version 2.0

"""DWDP Manager: Expert layout, IPC handle exchange, weight views, lifecycle.

Uses PyTorch native IPC APIs (_share_cuda_ / _new_shared_cuda) for cross-process
GPU memory sharing — this is portable across CUDA and ROCm/HIP.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_global_dwdp_manager: Optional["DwdpManager"] = None


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


def set_global_dwdp_manager(manager: Optional["DwdpManager"]):
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def enable_dwdp() -> bool:
    if _global_dwdp_manager is not None:
        return True
    try:
        from sglang.srt.server_args import get_global_server_args
        args = get_global_server_args()
        return args is not None and getattr(args, "dwdp_size", 1) > 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DwdpExpertLayout
# ---------------------------------------------------------------------------
@dataclass
class DwdpExpertLayout:
    """Expert-to-rank mapping with optional overlapping allocation."""

    num_routed_experts: int  # e.g. 256 for DeepSeek V3
    dwdp_size: int  # number of GPUs in a DWDP group
    dwdp_rank: int  # 0..dwdp_size-1
    num_experts_per_worker: int  # experts stored locally per rank

    def __post_init__(self):
        assert self.num_experts_per_worker >= self.num_routed_experts // self.dwdp_size
        assert self.num_experts_per_worker <= self.num_routed_experts

        # Number of experts to prefetch from each peer
        self.num_prefetch_experts = math.ceil(
            (self.num_routed_experts - self.num_experts_per_worker)
            / (self.dwdp_size - 1)
        )

        # Local expert range [start, end)
        self.local_expert_start = min(
            self.num_prefetch_experts * self.dwdp_rank,
            self.num_routed_experts - self.num_experts_per_worker,
        )
        self.local_expert_end = (
            self.local_expert_start + self.num_experts_per_worker
        )

        # Precompute peer expert ranges
        self.peer_expert_ranges: Dict[int, Tuple[int, int]] = {}
        for r in range(self.dwdp_size):
            start = min(
                self.num_prefetch_experts * r,
                self.num_routed_experts - self.num_experts_per_worker,
            )
            end = start + self.num_experts_per_worker
            self.peer_expert_ranges[r] = (start, end)

    def get_prefetch_src_offset(self, peer_rank: int) -> int:
        """Get the expert offset within a peer's local weight tensor for prefetch."""
        peer_start, peer_end = self.peer_expert_ranges[peer_rank]
        if self.dwdp_rank < peer_rank:
            # Need tail of peer's experts
            prefetch_start = peer_end - self.num_prefetch_experts
        else:
            # Need head of peer's experts
            prefetch_start = peer_start
        return prefetch_start - peer_start  # offset in number of experts


# ---------------------------------------------------------------------------
# DwdpWeightView — weight bundle returned by get_weight_view()
# ---------------------------------------------------------------------------
@dataclass
class DwdpWeightView:
    """Dict-based multi-B weight tensor bundle for the MoE kernel.

    weights: {param_name: [rank0_tensor, rank1_tensor, ...]}
    Each list has dwdp_size entries in rank order. The entry at dwdp_rank
    is the local weight tensor (shape [num_experts_per_worker, ...]);
    others are prefetch buffer views (shape [num_prefetch_experts, ...]).
    """

    weights: Dict[str, List[torch.Tensor]]
    expert_size_per_partition: int  # num_experts_per_worker
    local_expert_start: int  # global expert ID offset


# ---------------------------------------------------------------------------
# DwdpManager
# ---------------------------------------------------------------------------
class DwdpManager:
    """Lifecycle manager for DWDP: init, prefetch, wait, get_weight_view."""

    def __init__(
        self,
        layout: DwdpExpertLayout,
        num_moe_layers: int,
        first_moe_layer_id: int,
        process_group: Any,  # torch ProcessGroup or GroupCoordinator
        num_fused_shared_experts: int = 0,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.first_moe_layer_id = first_moe_layer_id
        self.dwdp_size = layout.dwdp_size
        self.dwdp_rank = layout.dwdp_rank
        self._num_fused_shared = num_fused_shared_experts
        self._last_rank = self.dwdp_size - 1

        # Extract the raw ProcessGroup for dist.all_gather_object
        if hasattr(process_group, "cpu_group"):
            self._cpu_group = process_group.cpu_group
        elif hasattr(process_group, "device_group"):
            self._cpu_group = process_group.device_group
        else:
            self._cpu_group = process_group

        # Per-layer registered weights: {layer_id: {param_name: Tensor}}
        self._layer_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # Per-layer shared expert weights: {layer_id: {param_name: Tensor}}
        self._shared_expert_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # Pre-fused local tensor (routed+shared) for last rank: {layer_id: {param_name: Tensor}}
        self._local_fused: Dict[int, Dict[str, torch.Tensor]] = {}

        # Peer IPC opened tensors: {(peer_rank, layer_id, param_name): Tensor}
        self._peer_tensors: Dict[Tuple[int, int, str], torch.Tensor] = {}

        # Prefetch buffers (ping-pong double buffer)
        self._prefetch_buffers: Optional[List[Dict[str, List[Optional[torch.Tensor]]]]] = None

        # CUDA streams and events — one stream per peer for parallel D2D copies
        self._prefetch_streams: Dict[int, torch.cuda.Stream] = {}
        # Per-layer, per-peer events: events[buf_idx][layer_slot][peer_rank]
        self._prefetch_events: Optional[List[List[Dict[int, torch.cuda.Event]]]] = None
        self._compute_events: Optional[List[List[torch.cuda.Event]]] = None

        self._initialized = False

        logger.info(
            f"[DWDP] Created DwdpManager: rank={self.dwdp_rank}, "
            f"size={self.dwdp_size}, "
            f"experts_per_worker={layout.num_experts_per_worker}, "
            f"prefetch_experts={layout.num_prefetch_experts}, "
            f"local_range=[{layout.local_expert_start}, {layout.local_expert_end}), "
            f"num_moe_layers={num_moe_layers}, first_moe_layer={first_moe_layer_id}"
        )

    # ----- Weight Registration -----

    def register_layer_weights(
        self, layer_id: int, weights: Dict[str, torch.Tensor]
    ):
        """Register weight tensors for a MoE layer after model loading."""
        self._layer_weights[layer_id] = weights
        logger.debug(
            f"[DWDP] Registered weights for layer {layer_id}: "
            f"{list(weights.keys())}"
        )

    def register_shared_expert_weights(
        self, layer_id: int, weights: Dict[str, torch.Tensor]
    ):
        """Register shared expert weight slices for fused-shared-expert mode."""
        self._shared_expert_weights[layer_id] = weights

    # ----- IPC Handle Exchange -----

    def exchange_ipc_handles(self):
        """Exchange CUDA IPC handles for all registered weight tensors across ranks.

        Uses PyTorch native IPC: _share_cuda_() to get handles,
        _new_shared_cuda() to open peer handles. This is portable across
        CUDA and ROCm/HIP.
        """
        logger.info(f"[DWDP] Exchanging IPC handles for {len(self._layer_weights)} layers...")

        for layer_id in sorted(self._layer_weights.keys()):
            weights = self._layer_weights[layer_id]
            for param_name, tensor in weights.items():
                # Get local IPC handle via PyTorch native API
                storage = tensor.untyped_storage()
                local_handle = storage._share_cuda_()

                # All-gather handles across DWDP group
                all_handles = [None] * self.dwdp_size
                dist.all_gather_object(
                    all_handles, local_handle, group=self._cpu_group
                )

                # Open peer handles
                for peer_rank in range(self.dwdp_size):
                    if peer_rank == self.dwdp_rank:
                        continue

                    peer_handle = all_handles[peer_rank]
                    # Redirect to local device
                    device_idx = torch.cuda.current_device()
                    redirected_handle = (device_idx,) + tuple(peer_handle)[1:]

                    target_device = torch.device(f"cuda:{device_idx}")
                    with torch.cuda.device(target_device):
                        peer_storage = torch.UntypedStorage._new_shared_cuda(
                            *redirected_handle
                        )
                        # Reconstruct tensor with same shape/dtype/stride
                        peer_tensor = torch.empty(
                            0, dtype=tensor.dtype, device=target_device
                        ).set_(
                            peer_storage,
                            storage_offset=0,
                            size=tensor.shape,
                            stride=tensor.stride(),
                        )

                    self._peer_tensors[(peer_rank, layer_id, param_name)] = peer_tensor

        total_peers = len(self._peer_tensors)
        logger.info(
            f"[DWDP] IPC handle exchange complete: "
            f"{total_peers} peer tensor pointers opened"
        )

    # ----- Prefetch Buffer Allocation -----

    def init_prefetch_buffers(self):
        """Allocate double-buffered prefetch staging buffers."""
        if not self._layer_weights:
            logger.warning("[DWDP] No layer weights registered, skipping buffer init")
            return

        # Determine param shapes/dtypes from first registered layer
        first_layer_id = min(self._layer_weights.keys())
        first_weights = self._layer_weights[first_layer_id]
        if not first_weights:
            logger.warning("[DWDP] First layer has no weights, skipping buffer init")
            return

        device = next(iter(first_weights.values())).device
        num_prefetch = self.layout.num_prefetch_experts

        # Allocate ping-pong double buffers
        # buffers[buf_idx][param_name][peer_rank] = Tensor or None
        self._prefetch_buffers = []
        total_buffer_bytes = 0
        for buf_idx in range(2):
            buffer = {}
            for param_name, ref_tensor in first_weights.items():
                per_expert_shape = ref_tensor.shape[1:]  # strip expert dim
                tensor_list: List[Optional[torch.Tensor]] = [None] * self.dwdp_size
                for peer_rank in range(self.dwdp_size):
                    if peer_rank != self.dwdp_rank:
                        # Last rank buffer includes fused shared expert slot(s)
                        if peer_rank == self._last_rank and self._num_fused_shared > 0:
                            buf_shape = (num_prefetch + self._num_fused_shared, *per_expert_shape)
                        else:
                            buf_shape = (num_prefetch, *per_expert_shape)
                        t = torch.empty(buf_shape, dtype=ref_tensor.dtype, device=device)
                        tensor_list[peer_rank] = t
                        total_buffer_bytes += t.numel() * t.element_size()
                buffer[param_name] = tensor_list
            self._prefetch_buffers.append(buffer)

        # Build pre-fused local tensors for the last rank (routed + shared, one-time cat)
        if self._num_fused_shared > 0 and self.dwdp_rank == self._last_rank:
            for layer_id in sorted(self._layer_weights.keys()):
                fused_dict: Dict[str, torch.Tensor] = {}
                routed_weights = self._layer_weights[layer_id]
                shared_weights = self._shared_expert_weights.get(layer_id, {})
                for param_name, routed_tensor in routed_weights.items():
                    shared_tensor = shared_weights.get(param_name)
                    if shared_tensor is not None:
                        fused_dict[param_name] = torch.cat(
                            [routed_tensor, shared_tensor], dim=0
                        )
                    else:
                        fused_dict[param_name] = routed_tensor
                self._local_fused[layer_id] = fused_dict
            logger.info(
                f"[DWDP] Built {len(self._local_fused)} pre-fused local tensors "
                f"(routed + {self._num_fused_shared} shared expert)"
            )

        # Create per-peer prefetch streams (one per peer → parallel SDMA engines)
        self._prefetch_streams = {}
        for peer_rank in range(self.dwdp_size):
            if peer_rank != self.dwdp_rank:
                self._prefetch_streams[peer_rank] = torch.cuda.Stream(device=device)

        # Create per-layer, per-peer events for each buffer slot
        num_slots_per_buf = math.ceil(self.num_moe_layers / 2)
        peer_ranks = sorted(self._prefetch_streams.keys())
        self._prefetch_events = [
            [
                {pr: torch.cuda.Event(enable_timing=False) for pr in peer_ranks}
                for _ in range(num_slots_per_buf)
            ]
            for _ in range(2)
        ]
        self._compute_events = [
            [torch.cuda.Event(enable_timing=False) for _ in range(num_slots_per_buf)]
            for _ in range(2)
        ]

        self._initialized = True
        logger.info(
            f"[DWDP] Prefetch buffers allocated: "
            f"2 x {self.dwdp_size - 1} peers x {num_prefetch} experts, "
            f"total ~{total_buffer_bytes / 1024**3:.2f} GB, "
            f"{len(self._prefetch_streams)} per-peer streams (multi-stream D2D)"
        )

    def initialize_compute_events(self):
        """Pre-record first compute event for each buffer slot.

        Must be called before prefetch_first_layers() so that any subsequent
        wait_compute_event() has a valid event to wait on.
        """
        if not self._initialized:
            return
        current_stream = torch.cuda.current_stream()
        for buf_idx in range(2):
            self._compute_events[buf_idx][0].record(current_stream)

    # ----- Prefetch Operations -----

    def _moe_layer_idx(self, layer_id: int) -> int:
        """Convert global layer_id to MoE-layer index (0-based)."""
        return layer_id - self.first_moe_layer_id

    def _prefetch_layer(self, layer_id: int, wait_compute_layer_id: Optional[int] = None):
        """Async prefetch peer weights for a single MoE layer into double buffer.

        Uses per-peer dedicated streams so D2D copies from different peers
        execute in parallel on independent SDMA engines.
        """
        if not self._initialized:
            return

        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2
        layer_slot = moe_idx // 2

        weights = self._layer_weights.get(layer_id)
        if weights is None:
            return

        # Precompute wait event (shared by all per-peer streams)
        wait_event = None
        if wait_compute_layer_id is not None:
            wait_moe_idx = self._moe_layer_idx(wait_compute_layer_id)
            wait_buf_idx = wait_moe_idx % 2
            wait_slot = wait_moe_idx // 2
            wait_event = self._compute_events[wait_buf_idx][wait_slot]

        # Dispatch per-peer copies on per-peer streams (parallel SDMA)
        for peer_rank in range(self.dwdp_size):
            if peer_rank == self.dwdp_rank:
                continue

            stream = self._prefetch_streams[peer_rank]
            with torch.cuda.stream(stream):
                # Wait for compute to release this buffer slot (from 2 layers ago)
                if wait_event is not None:
                    stream.wait_event(wait_event)

                src_expert_offset = self.layout.get_prefetch_src_offset(peer_rank)
                num_prefetch = self.layout.num_prefetch_experts

                for param_name in weights:
                    peer_tensor = self._peer_tensors.get(
                        (peer_rank, layer_id, param_name)
                    )
                    if peer_tensor is None:
                        continue

                    dst_buf = self._prefetch_buffers[buf_idx][param_name][peer_rank]
                    src_slice = peer_tensor.narrow(0, src_expert_offset, num_prefetch)

                    # Last rank with fused shared expert: copy routed into [:num_prefetch],
                    # then copy local shared expert into the extra slot(s)
                    if peer_rank == self._last_rank and self._num_fused_shared > 0:
                        dst_buf.narrow(0, 0, num_prefetch).copy_(src_slice, non_blocking=True)
                        shared_src = self._shared_expert_weights.get(layer_id, {}).get(param_name)
                        if shared_src is not None:
                            dst_buf.narrow(0, num_prefetch, self._num_fused_shared).copy_(
                                shared_src, non_blocking=True
                            )
                    else:
                        dst_buf.copy_(src_slice, non_blocking=True)

                # Signal per-peer completion
                self._prefetch_events[buf_idx][layer_slot][peer_rank].record(stream)

    def prefetch_first_layers(self):
        """Trigger async prefetch for the first 2 MoE layers.

        Called at the start of forward_extend(), before dense layers execute.
        The dense layers provide the initial prefetch hidden window.
        """
        if not self._initialized:
            return

        sorted_layer_ids = sorted(self._layer_weights.keys())
        if len(sorted_layer_ids) == 0:
            return

        # Prefetch first MoE layer into buf[0]
        self._prefetch_layer(sorted_layer_ids[0], wait_compute_layer_id=None)

        # Prefetch second MoE layer into buf[1] if exists
        if len(sorted_layer_ids) > 1:
            self._prefetch_layer(sorted_layer_ids[1], wait_compute_layer_id=None)

    def wait_prefetch(self, layer_id: int):
        """Wait for prefetch completion for a given MoE layer.

        Called on the default stream before MoE compute.
        Waits on all per-peer prefetch events for this layer.
        """
        if not self._initialized:
            return

        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2
        layer_slot = moe_idx // 2

        current_stream = torch.cuda.current_stream()
        for event in self._prefetch_events[buf_idx][layer_slot].values():
            current_stream.wait_event(event)

    def record_compute_and_prefetch_next(self, layer_id: int):
        """Record compute completion and trigger next layer's prefetch.

        Called on the default stream after MoE compute.
        """
        if not self._initialized:
            return

        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2
        layer_slot = moe_idx // 2

        # Record compute done on default stream
        torch.cuda.current_stream().record_event(
            self._compute_events[buf_idx][layer_slot]
        )

        # Trigger prefetch for layer_id + 2 (same buffer slot)
        sorted_layer_ids = sorted(self._layer_weights.keys())
        current_pos = sorted_layer_ids.index(layer_id)
        next_pos = current_pos + 2
        if next_pos < len(sorted_layer_ids):
            next_layer_id = sorted_layer_ids[next_pos]
            self._prefetch_layer(
                next_layer_id,
                wait_compute_layer_id=layer_id,
            )

    # ----- Weight View Assembly -----

    def get_weight_view(self, layer_id: int) -> DwdpWeightView:
        """Assemble multi-B weight view from local weights + prefetch buffers.

        Returns a DwdpWeightView with dict-based weight lists that can be
        concatenated or passed to multi-B kernel APIs.
        """
        local_weights = self._layer_weights.get(layer_id, {})
        moe_idx = self._moe_layer_idx(layer_id)
        buf_idx = moe_idx % 2

        local_fused = self._local_fused.get(layer_id, {})

        result_weights: Dict[str, List[torch.Tensor]] = {}
        for param_name, local_tensor in local_weights.items():
            tensor_list = []
            for rank in range(self.dwdp_size):
                if rank == self.dwdp_rank:
                    # Use pre-fused tensor if this is the last rank with shared expert
                    fused = local_fused.get(param_name)
                    if fused is not None:
                        tensor_list.append(fused)
                    else:
                        tensor_list.append(local_tensor)
                else:
                    if self._prefetch_buffers is not None:
                        buf_tensor = self._prefetch_buffers[buf_idx][param_name][rank]
                        tensor_list.append(buf_tensor)
                    else:
                        # Fallback: use peer tensor directly (no prefetch buffer)
                        peer_tensor = self._peer_tensors.get(
                            (rank, layer_id, param_name)
                        )
                        if peer_tensor is not None:
                            tensor_list.append(peer_tensor)
            result_weights[param_name] = tensor_list

        return DwdpWeightView(
            weights=result_weights,
            expert_size_per_partition=self.layout.num_experts_per_worker,
            local_expert_start=self.layout.local_expert_start,
        )

    # ----- Cleanup -----

    def cleanup(self):
        """Release IPC resources and buffers."""
        self._peer_tensors.clear()
        self._prefetch_buffers = None
        self._prefetch_streams.clear()
        self._prefetch_events = None
        self._compute_events = None
        self._layer_weights.clear()
        self._shared_expert_weights.clear()
        self._local_fused.clear()
        self._initialized = False
        logger.info("[DWDP] Cleaned up DWDP resources")
