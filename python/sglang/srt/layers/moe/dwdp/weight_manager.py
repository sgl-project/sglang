# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Slot-rotated async prefetch of peer expert weights into the composite VA."""

from __future__ import annotations

import bisect
import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.dwdp.layout import PeerRanges, lookup_owner
from sglang.srt.layers.moe.dwdp.transport import DWDPTransport
from sglang.srt.layers.moe.dwdp.weight_buffer import PREFETCH_DEPTH, WeightBuffer
from sglang.srt.utils.common import create_device_stream, device_stream_context

logger = logging.getLogger(__name__)


class DWDPWeightManager:
    def __init__(
        self,
        weight_buffer: WeightBuffer,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
        peer_ranges: PeerRanges,
        moe_layer_indices: List[int],
        weight_names: List[str],
        dwdp_rank: int,
        dwdp_size: int,
        transport: Optional[DWDPTransport] = None,
    ) -> None:
        self._weight_buffer = weight_buffer
        self._peer_views = peer_views
        self._peer_ranges = peer_ranges
        self._moe_layer_indices = sorted(moe_layer_indices)
        self._moe_layer_set = set(self._moe_layer_indices)
        self._layer_positions = {
            layer_idx: position
            for position, layer_idx in enumerate(self._moe_layer_indices)
        }
        self._weight_names = list(weight_names)
        self._dwdp_rank = dwdp_rank
        self._dwdp_size = dwdp_size
        # transport handles underpin the VA mappings; must outlive this manager
        self._transport = transport

        self._device = weight_buffer.device
        self._device_module = torch.get_device_module(self._device)
        self._copy_stream = create_device_stream(self._device)

        event_type = self._device_module.Event
        num_slots = weight_buffer.num_slots
        assert num_slots >= PREFETCH_DEPTH, (num_slots, PREFETCH_DEPTH)
        self._prefetch_events = [event_type() for _ in range(num_slots)]
        self._consume_events = [event_type() for _ in range(num_slots)]

        # pre-record consume events so the first prefetch doesn't stall
        current = self._device_module.current_stream(self._device)
        for consume_event in self._consume_events:
            consume_event.record(current)

        logger.debug(
            f"WeightManager rank={dwdp_rank}/{dwdp_size}, "
            f"{len(moe_layer_indices)} MoE layers, weights={weight_names}"
        )

    @property
    def weight_buffer(self) -> WeightBuffer:
        return self._weight_buffer

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._moe_layer_set

    def next_moe_layer(self, layer_idx: int) -> Optional[int]:
        pos = bisect.bisect_right(self._moe_layer_indices, layer_idx)
        if pos < len(self._moe_layer_indices):
            return self._moe_layer_indices[pos]
        return None

    def first_moe_layer(self) -> int:
        return self._moe_layer_indices[0]

    def prefetch_layer(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)

        if self._weight_buffer.rebinds_pool_pages and not self._weight_buffer.is_bound(
            layer_idx
        ):
            # unmapping is a host call that does not wait for the copy writing or the compute reading these pages
            self._prefetch_events[buf_idx].synchronize()
            self._consume_events[buf_idx].synchronize()
            self._weight_buffer.bind_pool_pages(layer_idx)

        with device_stream_context(self._copy_stream):
            # WAR: wait for compute to finish reading this slot before overwriting
            self._copy_stream.wait_event(self._consume_events[buf_idx])

            self._prefetch_layer_per_slice(layer_idx)

            self._prefetch_events[buf_idx].record(self._copy_stream)

    def _prefetch_layer_per_slice(self, layer_idx: int) -> None:
        for name in self._weight_names:
            remote_slices = self._weight_buffer.get_remote_slices(layer_idx, name)
            for dst_slice, expert_start, expert_end in remote_slices:
                cursor = expert_start
                dst_offset = 0
                while cursor < expert_end:
                    peer_rank = lookup_owner(cursor, self._peer_ranges)
                    peer_start, peer_end = self._peer_ranges[peer_rank]
                    local_offset = cursor - peer_start
                    chunk_end = min(expert_end, peer_end)
                    num_experts = chunk_end - cursor
                    peer_view = self._peer_views[(peer_rank, layer_idx, name)]
                    dst_slice[dst_offset : dst_offset + num_experts].copy_(
                        peer_view[local_offset : local_offset + num_experts]
                    )
                    dst_offset += num_experts
                    cursor = chunk_end

    def wait_prefetch(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        compute_stream = self._device_module.current_stream(self._device)
        compute_stream.wait_event(self._prefetch_events[buf_idx])

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        compute_stream = self._device_module.current_stream(self._device)

        self._consume_events[buf_idx].record(compute_stream)

        ahead = self._layer_positions[layer_idx] + PREFETCH_DEPTH
        if ahead < len(self._moe_layer_indices):
            self.prefetch_layer(self._moe_layer_indices[ahead])

    def prefetch_first_layers(self) -> None:
        for layer_idx in self._moe_layer_indices[:PREFETCH_DEPTH]:
            self.prefetch_layer(layer_idx)

    def release(self) -> None:
        if self._weight_buffer is not None:
            # copies may still be writing the pages that are about to be unmapped
            self._copy_stream.synchronize()
            self._weight_buffer.synchronize()
            self._weight_buffer.release()
            self._weight_buffer = None
        if self._transport is not None:
            self._transport.release()
            self._transport = None
        self._peer_views.clear()
