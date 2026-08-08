"""Request-level C4 sizing for DSV4 HiSparse PD decode."""

from __future__ import annotations

import msgspec

from sglang.srt.utils.common import ceil_div


class HiSparseC4PersistentBytes(msgspec.Struct, frozen=True):
    """Persistent GPU memory charged before sizing token pools."""

    c4_kv: int
    c4_data_ptrs: int
    allocator_free_pages: int
    logical_mapping_tail: int
    req_to_token: int
    coordinator_host_pool_ptrs: int
    coordinator: int

    @property
    def total(self) -> int:
        return (
            self.c4_kv
            + self.c4_data_ptrs
            + self.allocator_free_pages
            + self.logical_mapping_tail
            + self.req_to_token
            + self.coordinator_host_pool_ptrs
            + self.coordinator
        )


class HiSparseC4Layout(msgspec.Struct, frozen=True):
    """Final request-indexed C4 contract shared by pool consumers."""

    max_running_requests: int
    pd_extra_slots: int
    req_slot_count: int
    context_len: int
    compressed_context_len: int
    c4_page_size: int
    device_buffer_size: int
    padded_per_req: int
    top_k: int
    local_c4_layers: int
    c4_device_slot_capacity: int
    persistent_bytes: HiSparseC4PersistentBytes

    @property
    def resident_request_capacity(self) -> int:
        """C4 buffers cover every request-table row, including the reserve row."""
        return self.req_slot_count


class HiSparseC4Geometry(msgspec.Struct, frozen=True):
    """Static geometry resolved before max-running-request finalization."""

    context_len: int
    compressed_context_len: int
    c4_page_size: int
    device_buffer_size: int
    padded_per_req: int
    top_k: int
    local_c4_layers: int
    pd_extra_slots: int
    coordinator_instances: int

    @classmethod
    def create(
        cls,
        *,
        context_len: int,
        c4_page_size: int,
        device_buffer_size: int,
        top_k: int,
        local_c4_layers: int,
        pd_extra_slots: int,
        coordinator_instances: int,
    ) -> HiSparseC4Geometry:
        return cls(
            context_len=context_len,
            compressed_context_len=ceil_div(context_len, 4),
            c4_page_size=c4_page_size,
            device_buffer_size=device_buffer_size,
            padded_per_req=device_buffer_size + c4_page_size,
            top_k=top_k,
            local_c4_layers=local_c4_layers,
            pd_extra_slots=pd_extra_slots,
            coordinator_instances=coordinator_instances,
        )

    def finalize(self, max_running_requests: int) -> HiSparseC4Layout:
        assert max_running_requests > 0, "max_running_requests must be positive"
        req_slot_count = max_running_requests + self.pd_extra_slots + 1
        c4_device_slot_capacity = req_slot_count * self.padded_per_req
        alloc_pages = c4_device_slot_capacity // self.c4_page_size
        physical_pages = (
            c4_device_slot_capacity + self.c4_page_size + 1
        ) // self.c4_page_size
        kv_page_stride_bytes = ceil_div(self.c4_page_size * 584, 576) * 576
        persistent_bytes = HiSparseC4PersistentBytes(
            c4_kv=self.local_c4_layers * physical_pages * kv_page_stride_bytes,
            c4_data_ptrs=self.local_c4_layers * 8,
            allocator_free_pages=alloc_pages * 8,
            logical_mapping_tail=(self.c4_page_size + 1) * 8,
            req_to_token=req_slot_count * self.context_len * 4,
            coordinator_host_pool_ptrs=(
                self.coordinator_instances * self.local_c4_layers * 16
            ),
            coordinator=self.coordinator_instances
            * self._coordinator_bytes(req_slot_count),
        )
        return HiSparseC4Layout(
            max_running_requests=max_running_requests,
            pd_extra_slots=self.pd_extra_slots,
            req_slot_count=req_slot_count,
            context_len=self.context_len,
            compressed_context_len=self.compressed_context_len,
            c4_page_size=self.c4_page_size,
            device_buffer_size=self.device_buffer_size,
            padded_per_req=self.padded_per_req,
            top_k=self.top_k,
            local_c4_layers=self.local_c4_layers,
            c4_device_slot_capacity=c4_device_slot_capacity,
            persistent_bytes=persistent_bytes,
        )

    def _coordinator_bytes(self, req_slot_count: int) -> int:
        return (
            8 * req_slot_count * self.padded_per_req
            + 8 * req_slot_count * (self.compressed_context_len + self.c4_page_size)
            + 8 * self.local_c4_layers * req_slot_count * self.padded_per_req
            + 2 * self.local_c4_layers * req_slot_count * self.device_buffer_size
            + 6 * self.device_buffer_size
            + 8 * req_slot_count * self.top_k
            + 4
        )
