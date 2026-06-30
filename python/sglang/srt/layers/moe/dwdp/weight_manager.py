"""DWDPWeightManager: runtime P2P prefetch scheduler.

Manages async D2D copies from peer MNNVL tensor views into the composite VA's
remote slices, using a double-buffered event protocol.

Ported from TensorRT-LLM ``_torch/modules/dwdp/weight_manager.py``.
"""

from __future__ import annotations

import bisect
import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.dwdp.specs import PeerRanges, lookup_owner
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer

logger = logging.getLogger(__name__)


class DWDPWeightManager:
    """Runtime P2P copy orchestrator for DWDP expert weight prefetch."""

    __slots__ = (
        "_weight_buffer",
        "_peer_views",
        "_peer_ranges",
        "_moe_layer_indices",
        "_moe_layer_set",
        "_weight_names",
        "_dwdp_rank",
        "_dwdp_size",
        "_copy_stream",
        "_prefetch_events",
        "_consume_events",
        "_transport",  # kept alive so handles aren't freed
    )

    def __init__(
        self,
        weight_buffer: WeightBuffer,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
        peer_ranges: PeerRanges,
        moe_layer_indices: List[int],
        weight_names: List[str],
        dwdp_rank: int,
        dwdp_size: int,
    ) -> None:
        self._weight_buffer = weight_buffer
        self._peer_views = peer_views
        self._peer_ranges = peer_ranges
        self._moe_layer_indices = sorted(moe_layer_indices)
        self._moe_layer_set = set(self._moe_layer_indices)
        self._weight_names = list(weight_names)
        self._dwdp_rank = dwdp_rank
        self._dwdp_size = dwdp_size
        self._transport = None  # set by dwdp_manager after creation

        device = torch.device("cuda", weight_buffer.device_id)
        self._copy_stream = torch.cuda.Stream(device=device)

        self._prefetch_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]
        self._consume_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]

        # Pre-record consume events so first prefetch doesn't stall
        current = torch.cuda.current_stream(device)
        for ev in self._consume_events:
            ev.record(current)

        logger.info(
            f"[WeightManager] rank={dwdp_rank}/{dwdp_size}, "
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
        """Enqueue async P2P copies for a layer's remote expert slices."""
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)

        with torch.cuda.stream(self._copy_stream):
            # WAR: wait for compute to finish reading this slot
            self._copy_stream.wait_event(self._consume_events[buf_idx])

            self._prefetch_layer_per_slice(layer_idx)

            # RAW: signal compute that copy is done
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
                    n = chunk_end - cursor

                    peer_key = (peer_rank, layer_idx, name)
                    src = self._peer_views[peer_key]
                    dst_slice[dst_offset : dst_offset + n].copy_(
                        src[local_offset : local_offset + n]
                    )
                    dst_offset += n
                    cursor = chunk_end

    # Wait + bind

    def wait_prefetch(self, layer_idx: int) -> None:
        """Compute stream waits for prefetch completion."""
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        device = torch.device("cuda", self._weight_buffer.device_id)
        compute_stream = torch.cuda.current_stream(device)
        compute_stream.wait_event(self._prefetch_events[buf_idx])

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        """Record compute done + trigger next layer's prefetch."""
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        other_buf = 1 - buf_idx
        device = torch.device("cuda", self._weight_buffer.device_id)
        compute_stream = torch.cuda.current_stream(device)

        # WAR signal for the other buffer slot
        self._consume_events[other_buf].record(compute_stream)

        # Trigger prefetch for the layer 2 ahead (same buffer slot)
        next_layer = self.next_moe_layer(layer_idx)
        if next_layer is not None:
            next_next = self.next_moe_layer(next_layer)
            if next_next is not None:
                self.prefetch_layer(next_next)

    def prefetch_first_layers(self) -> None:
        """Warm-up: prefetch first 2 MoE layers while dense layers compute."""
        if len(self._moe_layer_indices) >= 1:
            self.prefetch_layer(self._moe_layer_indices[0])
        if len(self._moe_layer_indices) >= 2:
            self.prefetch_layer(self._moe_layer_indices[1])

    def release(self) -> None:
        if self._weight_buffer is not None:
            self._weight_buffer.release()
            self._weight_buffer = None
        if self._transport is not None:
            self._transport.release()
            self._transport = None
        self._peer_views.clear()
