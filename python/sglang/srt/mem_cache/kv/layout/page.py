"""A host view of a complete typed entry's page envelope."""

from typing import Optional

import msgspec
import torch

from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions


class PageEnvelopeHostLayout(msgspec.Struct, frozen=True, kw_only=True):
    buffer: torch.Tensor
    page_size: int
    capacity: int
    # The shared allocation the pages live in, and its registrable bytes.
    allocation: Optional[torch.Tensor] = None
    registration_bytes: int = 0
    start_layer: int = 0
    end_layer: int = 0
    device_layer_num: int = 1

    @property
    def size(self):
        return self.capacity

    @property
    def host_capacity_tokens(self):
        return self.capacity

    @property
    def grow_direction(self):
        # This host allocation is independent of the device band's frontiers.
        return "down"

    def get_page_envelope_buffer(self):
        return self.buffer

    def get_contiguous_buf_infos(self):
        return [self.buffer.data_ptr()], [self.buffer.nbytes], [self.buffer.shape[1]]

    def build(self, options: HostPoolOptions, *, draft_buffer_infos=()):
        from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool

        if draft_buffer_infos:
            raise ValueError("declared entries already carry their draft parts")
        from sglang.srt.mem_cache.device_pool_info import PageEnvelopeBufferInfo

        return UnifiedPageEnvelopeHostPool(
            self,
            options.ratio,
            options.size,
            self.page_size,
            options.layout,
            allocator_type=options.allocator_type,
            pin_memory=options.pin_memory,
            pool_label=options.pool_label,
            _buffer_info=PageEnvelopeBufferInfo(
                allocation=self.allocation,
                registration_bytes=self.registration_bytes,
                pages=self.buffer,
                page_size=self.page_size,
                grow_direction=self.grow_direction,
            ),
        )


def page_envelope_host_layout(arena, name, capacity, page_size):
    band = arena.band(name)
    page_bytes = band.unit_bytes
    pages = arena.num_entries(name) // band.unit_entries
    return PageEnvelopeHostLayout(
        buffer=arena.raw[: pages * page_bytes].view(pages, page_bytes),
        allocation=arena.raw,
        registration_bytes=arena.accessible_bytes,
        page_size=page_size,
        capacity=capacity,
    )
