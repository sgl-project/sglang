"""Double-buffered async prefetch system for DWDP.

Uses a dedicated CUDA stream and ping-pong buffers to overlap
weight prefetch (D2D NVLink copy) with MoE compute.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import (
        DwdpExpertLayout,
        DwdpLayerHandleCollector,
    )

logger = logging.getLogger(__name__)


class DwdpPrefetchBuffer:
    """Double-buffered (ping-pong) async prefetch for DWDP expert weights.

    Buffer slot 0 is used by even-indexed MoE layers (0, 2, 4, ...),
    and slot 1 by odd-indexed (1, 3, 5, ...).
    """

    def __init__(
        self,
        layout: DwdpExpertLayout,
        num_moe_layers: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.param_shapes = param_shapes
        self.param_dtypes = param_dtypes
        self.device = device

        self.prefetch_stream = torch.cuda.Stream(device=device)

        num_prefetch_experts = layout.num_prefetch_experts
        dwdp_size = layout.dwdp_size
        dwdp_rank = layout.dwdp_rank

        # Allocate 2 buffer slots (ping-pong)
        # buffers[buf_idx][param_name] = list of tensors, one per rank
        # Entry at dwdp_rank is None (local weights used directly)
        self.buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []
        for buf_idx in range(2):
            buffer = {}
            for param_name, shape in param_shapes.items():
                tensor_list: List[Optional[torch.Tensor]] = [None] * dwdp_size
                for peer_rank in range(dwdp_size):
                    if peer_rank != dwdp_rank:
                        buffer_shape = (num_prefetch_experts,) + tuple(shape)
                        tensor_list[peer_rank] = torch.empty(
                            buffer_shape,
                            dtype=param_dtypes[param_name],
                            device=device,
                        )
                buffer[param_name] = tensor_list
            self.buffers.append(buffer)

        # Per-layer CUDA events
        # Number of layer slots per buffer = ceil(num_moe_layers / 2)
        num_slots_per_buffer = math.ceil(num_moe_layers / 2)
        self.prefetch_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]
        self.compute_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]

        logger.info(
            f"DwdpPrefetchBuffer allocated: "
            f"num_prefetch_experts={num_prefetch_experts}, "
            f"num_slots_per_buffer={num_slots_per_buffer}, "
            f"params={list(param_shapes.keys())}"
        )

    def initialize_compute_events(self) -> None:
        """Pre-record first compute events so prefetch_first_layers() can proceed.

        This records a compute event on the current (default) stream for
        layer_slot=0 of each buffer, ensuring subsequent ``wait_event()``
        calls in the prefetch path have valid events to synchronize on.
        """
        current_stream = torch.cuda.current_stream(self.device)
        for buf_idx in range(2):
            self.compute_events[buf_idx][0].record(current_stream)

    def prefetch_layer(
        self,
        moe_layer_idx: int,
        layer_handles: DwdpLayerHandleCollector,
    ) -> None:
        """Issue async D2D copies for one MoE layer's peer weights.

        Runs on ``prefetch_stream``. Waits for the previous compute on the
        same buffer slot before overwriting, then records a prefetch event.
        """
        from cuda import cudart

        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        layout = self.layout
        dwdp_rank = layout.dwdp_rank
        num_prefetch_experts = layout.num_prefetch_experts

        # Determine which compute event to wait on (previous user of this buffer slot)
        wait_compute_slot = layer_slot - 1 if moe_layer_idx >= 2 else None

        with torch.cuda.stream(self.prefetch_stream):
            # Wait for previous compute on this buffer slot to complete
            if wait_compute_slot is not None and wait_compute_slot >= 0:
                self.prefetch_stream.wait_event(
                    self.compute_events[buf_idx][wait_compute_slot]
                )

            # D2D copy from peer IPC tensors into local buffer
            for peer_rank in range(layout.dwdp_size):
                if peer_rank == dwdp_rank:
                    continue

                src_expert_offset = layout.get_prefetch_src_offset(peer_rank)

                for param_name in self.param_shapes:
                    shape = self.param_shapes[param_name]
                    dtype = self.param_dtypes[param_name]
                    expert_size = 1
                    for s in shape:
                        expert_size *= s
                    expert_size *= dtype.itemsize

                    src_ptr = (
                        layer_handles.peer_base_ptrs[(peer_rank, param_name)]
                        + src_expert_offset * expert_size
                    )
                    dst_tensor = self.buffers[buf_idx][param_name][peer_rank]
                    dst_ptr = dst_tensor.data_ptr()
                    data_size = num_prefetch_experts * expert_size

                    err = cudart.cudaMemcpyAsync(
                        dst_ptr,
                        src_ptr,
                        data_size,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        self.prefetch_stream.cuda_stream,
                    )
                    # cudaMemcpyAsync returns a tuple (err,)
                    if isinstance(err, tuple):
                        err = err[0]
                    assert err == cudart.cudaError_t.cudaSuccess, (
                        f"cudaMemcpyAsync failed: {err}"
                    )

            # Signal prefetch completion
            self.prefetch_events[buf_idx][layer_slot].record(self.prefetch_stream)

    def wait_for_prefetch(self, moe_layer_idx: int) -> None:
        """Default stream waits for prefetch of this layer to complete."""
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self.prefetch_events[buf_idx][layer_slot])

    def record_compute_done(self, moe_layer_idx: int) -> None:
        """Record compute completion on the default stream."""
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        current_stream = torch.cuda.current_stream(self.device)
        self.compute_events[buf_idx][layer_slot].record(current_stream)

    def get_buffer_views(
        self, moe_layer_idx: int
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Return buffer tensor views for the given MoE layer.

        Returns a dict mapping param_name -> list of tensors per rank.
        Entry at dwdp_rank is None (caller fills with local weight).
        """
        buf_idx = moe_layer_idx % 2
        return self.buffers[buf_idx]

    def cleanup(self) -> None:
        """Release prefetch buffers and events."""
        self.buffers.clear()
        self.prefetch_events.clear()
        self.compute_events.clear()
        self.prefetch_stream = None
