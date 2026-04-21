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
        param_full_shapes: Dict[str, torch.Size],
        param_full_strides: Dict[str, Tuple[int, ...]],
        param_dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.param_full_shapes = param_full_shapes
        self.param_full_strides = param_full_strides
        self.param_dtypes = param_dtypes
        self.device = device

        self.prefetch_stream = torch.cuda.Stream(device=device)

        num_prefetch_experts = layout.num_prefetch_experts
        num_experts_per_worker = layout.num_experts_per_worker
        dwdp_size = layout.dwdp_size
        dwdp_rank = layout.dwdp_rank

        # Per-param metadata. For each registered tensor we identify the
        # expert dimension as the logical dim whose size equals
        # ``num_experts_per_worker`` AND whose stride is maximal (i.e.
        # outermost in physical memory). This holds both for contiguous
        # tensors (expert dim 0) and for the MMA-layout SF strided views
        # (expert dim 5, but still outermost in physical memory).
        #
        # Having the expert dim be the outermost physical dim means:
        #   - per-expert bytes = stride[expert_dim] (elements) * itemsize
        #   - "first N experts" occupy a contiguous byte range of size
        #     N * per_expert_bytes at the start of physical storage, so a
        #     single cudaMemcpyAsync per expert range is sufficient.
        #   - The prefetch-view strides are identical to the original
        #     strides (none of the other strides depend on num_experts).
        self.per_expert_bytes: Dict[str, int] = {}
        self.prefetch_view_shapes: Dict[str, torch.Size] = {}
        self.prefetch_view_strides: Dict[str, Tuple[int, ...]] = {}
        self.prefetch_view_dtypes: Dict[str, torch.dtype] = {}
        for name, shape in param_full_shapes.items():
            strides = param_full_strides[name]
            itemsize = param_dtypes[name].itemsize
            self.prefetch_view_dtypes[name] = param_dtypes[name]

            candidates = [i for i, s in enumerate(shape) if s == num_experts_per_worker]
            assert candidates, (
                f"No dim with size num_experts_per_worker={num_experts_per_worker} "
                f"for param {name} with shape {tuple(shape)}"
            )
            expert_dim = max(candidates, key=lambda i: strides[i])
            assert strides[expert_dim] == max(strides), (
                f"Expert dim {expert_dim} for param {name} is not the outermost "
                f"physical dim (shape={tuple(shape)}, strides={strides}). "
                f"Per-expert prefetch slicing requires experts to be outermost."
            )

            self.per_expert_bytes[name] = strides[expert_dim] * itemsize
            view_shape = list(shape)
            view_shape[expert_dim] = num_prefetch_experts
            self.prefetch_view_shapes[name] = torch.Size(view_shape)
            self.prefetch_view_strides[name] = tuple(strides)

        # Allocate 2 buffer slots (ping-pong)
        # buffers[buf_idx][param_name] = list of tensors, one per rank
        # Entry at dwdp_rank is None (local weights used directly)
        self.buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []
        for buf_idx in range(2):
            buffer = {}
            for param_name in param_full_shapes:
                tensor_list: List[Optional[torch.Tensor]] = [None] * dwdp_size
                for peer_rank in range(dwdp_size):
                    if peer_rank != dwdp_rank:
                        buf_bytes = num_prefetch_experts * self.per_expert_bytes[param_name]
                        tensor_list[peer_rank] = torch.empty(
                            buf_bytes,
                            dtype=torch.uint8,
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
            f"per_expert_bytes={self.per_expert_bytes}, "
            f"params={list(param_full_shapes.keys())}"
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
        peer_handles: Dict[int, Dict[str, Tuple[bytes, int]]],
    ) -> None:
        """Issue async D2D copies for one MoE layer's peer weights.

        Opens IPC handles on the fly, copies data, then closes them
        to stay within the CUDA driver's per-process IPC mapping limit.
        """
        import os
        from cuda import cudart

        _dwdp_debug = bool(os.environ.get("SGL_DWDP_DEBUG"))
        if _dwdp_debug:
            import torch.distributed as _dist
            _rank = _dist.get_rank() if _dist.is_initialized() else -1
            print(
                f"[DWDP_DEBUG] prefetch_layer begin rank={_rank} "
                f"moe_layer_idx={moe_layer_idx} "
                f"current_device={torch.cuda.current_device()}",
                flush=True,
            )

        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        layout = self.layout
        dwdp_rank = layout.dwdp_rank
        num_prefetch_experts = layout.num_prefetch_experts

        wait_compute_slot = layer_slot - 1 if moe_layer_idx >= 2 else None

        with torch.cuda.stream(self.prefetch_stream):
            if wait_compute_slot is not None and wait_compute_slot >= 0:
                self.prefetch_stream.wait_event(
                    self.compute_events[buf_idx][wait_compute_slot]
                )

            opened_ptrs = []

            for peer_rank, handle_dict in peer_handles.items():
                src_expert_offset = layout.get_prefetch_src_offset(peer_rank)

                for param_name in self.per_expert_bytes:
                    if param_name not in handle_dict:
                        continue
                    handle_bytes, offset = handle_dict[param_name]

                    handle = cudart.cudaIpcMemHandle_t()
                    handle.reserved = list(handle_bytes)
                    err, base_ptr = cudart.cudaIpcOpenMemHandle(
                        handle, cudart.cudaIpcMemLazyEnablePeerAccess
                    )
                    assert err == cudart.cudaError_t.cudaSuccess, (
                        f"cudaIpcOpenMemHandle failed: peer={peer_rank} "
                        f"param={param_name}: {err}"
                    )
                    base_ptr_int = int(base_ptr)
                    opened_ptrs.append(base_ptr_int)

                    expert_bytes = self.per_expert_bytes[param_name]
                    src_ptr = base_ptr_int + offset + src_expert_offset * expert_bytes
                    dst_tensor = self.buffers[buf_idx][param_name][peer_rank]
                    dst_ptr = dst_tensor.data_ptr()
                    data_size = num_prefetch_experts * expert_bytes

                    err = cudart.cudaMemcpyAsync(
                        dst_ptr,
                        src_ptr,
                        data_size,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        self.prefetch_stream.cuda_stream,
                    )
                    if isinstance(err, tuple):
                        err = err[0]
                    assert err == cudart.cudaError_t.cudaSuccess, (
                        f"cudaMemcpyAsync failed: peer={peer_rank} "
                        f"param={param_name}: {err}"
                    )

            # Sync before closing handles
            sync_err = cudart.cudaStreamSynchronize(self.prefetch_stream.cuda_stream)
            if isinstance(sync_err, tuple):
                sync_err = sync_err[0]
            assert sync_err == cudart.cudaError_t.cudaSuccess, (
                f"cudaStreamSynchronize failed: {sync_err}"
            )

            for ptr in opened_ptrs:
                close_err = cudart.cudaIpcCloseMemHandle(ptr)
                if isinstance(close_err, tuple):
                    close_err = close_err[0]
                if _dwdp_debug and close_err != cudart.cudaError_t.cudaSuccess:
                    print(
                        f"[DWDP_DEBUG] cudaIpcCloseMemHandle failed: "
                        f"ptr={ptr} err={close_err}",
                        flush=True,
                    )

            self.prefetch_events[buf_idx][layer_slot].record(self.prefetch_stream)

            if _dwdp_debug:
                import torch.distributed as _dist
                _rank = _dist.get_rank() if _dist.is_initialized() else -1
                try:
                    torch.cuda.synchronize()
                except Exception as _e:
                    print(
                        f"[DWDP_DEBUG] post-prefetch sync failed rank={_rank} "
                        f"moe_layer_idx={moe_layer_idx}: {_e}",
                        flush=True,
                    )
                    raise

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
        Flat uint8 buffers are reinterpreted with the correct dtype and
        rebuilt with the same stride pattern as the original registered
        tensor (with ``num_prefetch_experts`` substituted for the expert
        dim). This preserves MMA-layout strided views over the physical
        storage laid out as ``(num_prefetch_experts, ...physical...)``.
        """
        buf_idx = moe_layer_idx % 2
        raw = self.buffers[buf_idx]
        views: Dict[str, List[Optional[torch.Tensor]]] = {}
        for param_name, tensor_list in raw.items():
            view_shape = self.prefetch_view_shapes[param_name]
            view_strides = self.prefetch_view_strides[param_name]
            view_dtype = self.prefetch_view_dtypes[param_name]
            view_list: List[Optional[torch.Tensor]] = []
            for t in tensor_list:
                if t is None:
                    view_list.append(None)
                else:
                    typed = t.view(view_dtype)
                    view_list.append(
                        torch.as_strided(
                            typed, view_shape, view_strides, storage_offset=0
                        )
                    )
            views[param_name] = view_list
        return views

    def cleanup(self) -> None:
        """Release prefetch buffers and events."""
        self.buffers.clear()
        self.prefetch_events.clear()
        self.compute_events.clear()
        self.prefetch_stream = None
