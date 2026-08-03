# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Double-buffered async prefetch of peer expert weights into the composite VA."""

from __future__ import annotations

import bisect
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.dwdp.layout import PeerRanges, lookup_owner
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer

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
        transport=None,
        copy_engine: Any = None,
        peer_device_ids: Optional[Dict[int, int]] = None,
    ) -> None:
        self._weight_buffer = weight_buffer
        self._peer_views = peer_views
        self._peer_ranges = peer_ranges
        self._moe_layer_indices = sorted(moe_layer_indices)
        self._moe_layer_set = set(self._moe_layer_indices)
        self._weight_names = list(weight_names)
        self._dwdp_rank = dwdp_rank
        self._dwdp_size = dwdp_size
        # transport handles underpin the VA mappings; must outlive this manager
        self._transport = transport
        self._copy_engine = copy_engine
        self._peer_device_ids = dict(peer_device_ids or {})

        device = torch.device("cuda", weight_buffer.device_id)
        self._copy_stream = torch.cuda.Stream(device=device)
        self._copy_tickets: List[List[int]] = [[], []]

        self._prefetch_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]
        self._consume_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]

        # pre-record consume events so the first prefetch doesn't stall
        current = torch.cuda.current_stream(device)
        for ev in self._consume_events:
            ev.record(current)

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

        if self._copy_engine is not None:
            # HSA queues cannot consume a HIP event. Host-wait before reusing
            # the slot; graph capture is disabled for DWDP.
            self._consume_events[buf_idx].synchronize()
            if self._copy_tickets[buf_idx]:
                raise RuntimeError(
                    f"DWDP copy slot {buf_idx} still has outstanding HSA tickets"
                )
            self._prefetch_layer_per_slice(layer_idx, buf_idx=buf_idx)
        else:
            with torch.cuda.stream(self._copy_stream):
                # WAR: wait for compute to finish reading this slot before overwriting
                self._copy_stream.wait_event(self._consume_events[buf_idx])

                self._prefetch_layer_per_slice(layer_idx)

                self._prefetch_events[buf_idx].record(self._copy_stream)

    def _prefetch_layer_per_slice(
        self, layer_idx: int, buf_idx: Optional[int] = None
    ) -> None:
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
                    destination = dst_slice[dst_offset : dst_offset + n]
                    source = src[local_offset : local_offset + n]
                    if self._copy_engine is None:
                        destination.copy_(source)
                    else:
                        assert buf_idx is not None
                        self._copy_tickets[buf_idx].append(
                            self._copy_engine.submit(
                                destination,
                                source,
                                destination_device=self._weight_buffer.device_id,
                                source_device=self._peer_device_ids.get(
                                    peer_rank,
                                    source.device.index,
                                ),
                            )
                        )
                    dst_offset += n
                    cursor = chunk_end

    def _disable_hsa_copy_engine(self, completed_slot: int) -> None:
        engine = self._copy_engine
        if engine is None:
            return
        self._copy_tickets[completed_slot].clear()
        for slot_idx, tickets in enumerate(self._copy_tickets):
            if slot_idx == completed_slot or not tickets:
                continue
            try:
                engine.wait_all(tickets)
            except Exception:
                logger.exception(
                    "Failed while draining HSA DWDP VMM slot %s during fallback",
                    slot_idx,
                )
            finally:
                tickets.clear()
        self._copy_engine = None

    def wait_prefetch(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        if self._copy_engine is not None:
            try:
                self._copy_engine.wait_all(self._copy_tickets[buf_idx])
            except Exception:
                logger.exception(
                    "HSA DWDP prefetch failed for layer %s; retrying with HIP copy_",
                    layer_idx,
                )
                self._disable_hsa_copy_engine(buf_idx)
                with torch.cuda.stream(self._copy_stream):
                    self._prefetch_layer_per_slice(layer_idx)
                    self._prefetch_events[buf_idx].record(self._copy_stream)
            else:
                return
            finally:
                self._copy_tickets[buf_idx].clear()
        device = torch.device("cuda", self._weight_buffer.device_id)
        compute_stream = torch.cuda.current_stream(device)
        compute_stream.wait_event(self._prefetch_events[buf_idx])

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        device = torch.device("cuda", self._weight_buffer.device_id)
        compute_stream = torch.cuda.current_stream(device)

        self._consume_events[buf_idx].record(compute_stream)

        # prefetch the layer 2 ahead — it reuses the same buffer slot
        next_layer = self.next_moe_layer(layer_idx)
        if next_layer is not None:
            next_next = self.next_moe_layer(next_layer)
            if next_next is not None:
                self.prefetch_layer(next_next)

    def prefetch_first_layers(self) -> None:
        if len(self._moe_layer_indices) >= 1:
            self.prefetch_layer(self._moe_layer_indices[0])
        if len(self._moe_layer_indices) >= 2:
            self.prefetch_layer(self._moe_layer_indices[1])

    def release(self) -> None:
        if self._copy_engine is not None:
            for tickets in self._copy_tickets:
                try:
                    self._copy_engine.wait_all(tickets)
                except Exception:
                    logger.exception("Failed while draining HSA tickets during release")
                finally:
                    tickets.clear()
        if self._weight_buffer is not None:
            self._weight_buffer.release()
            self._weight_buffer = None
        if self._transport is not None:
            self._transport.release()
            self._transport = None
        self._peer_views.clear()
