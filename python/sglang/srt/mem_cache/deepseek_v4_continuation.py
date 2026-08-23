from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.utils import get_device_module

DSV4_CONTINUATION_SWA_WINDOW = 128
DSV4_CONTINUATION_C4_READ_PAGE_SIZE = 4


def dsv4_continuation_payload_bytes(
    *,
    target_layer_num: int,
    draft_layer_num: int,
    c4_layer_num: int,
    attention_head_dim: int,
    indexer_head_dim: int,
    c4_state_element_size: int,
) -> int:
    ring_bytes = (
        (target_layer_num + draft_layer_num)
        * DSV4_CONTINUATION_SWA_WINDOW
        * attention_head_dim
        * torch.bfloat16.itemsize
    )
    c4_attention_bytes = (
        c4_layer_num
        * DSV4_CONTINUATION_C4_READ_PAGE_SIZE
        * 4
        * attention_head_dim
        * c4_state_element_size
    )
    c4_indexer_bytes = (
        c4_layer_num
        * DSV4_CONTINUATION_C4_READ_PAGE_SIZE
        * 4
        * indexer_head_dim
        * c4_state_element_size
    )
    return ring_bytes + c4_attention_bytes + c4_indexer_bytes


@dataclass(frozen=True)
class _TensorGroup:
    tensors: tuple[torch.Tensor, ...]
    rows_per_slot: int
    slot_view: torch.Tensor


class DeepSeekV4ContinuationPool:
    """Bounded device storage for reusable DSV4 endpoint state."""

    def __init__(
        self,
        *,
        target_pool,
        draft_pools: Sequence,
        req_to_token_pool,
        num_slots: int,
    ) -> None:
        if num_slots <= 0:
            raise ValueError(
                f"continuation pool requires positive slots, got {num_slots}"
            )
        if not getattr(target_pool, "_unified_kv", False):
            raise ValueError("DSV4 continuation storage requires unified target KV")
        if getattr(target_pool, "dcp_size", 1) <= 1:
            raise ValueError("DSV4 continuation storage requires dcp_size > 1")

        self.target_pool = target_pool
        self.draft_pools = tuple(draft_pools)
        self.req_to_token_pool = req_to_token_pool
        self.num_slots = num_slots
        self.size = num_slots
        self.device = torch.device(target_pool.device)
        self.device_module = get_device_module()
        self.layer_num = 1
        self.start_layer = 0
        self.end_layer = 1
        self.logical_page_size = int(target_pool.logical_page_size)
        if self.logical_page_size % 128 != 0:
            raise ValueError(
                "DSV4 continuation endpoints must be C128-aligned: "
                f"logical_page_size={self.logical_page_size}"
            )

        ring_pools = (target_pool, *self.draft_pools)
        target_geometry = (
            target_pool.unified_swa_window,
            target_pool.unified_swa_ring_size,
        )
        if target_pool.unified_swa_window != DSV4_CONTINUATION_SWA_WINDOW:
            raise ValueError(
                "DSV4 continuation storage requires a 128-token SWA window"
            )
        for pool in ring_pools:
            if not getattr(pool, "_unified_kv", False):
                raise ValueError("all DSV4 continuation pools must use unified KV")
            geometry = (pool.unified_swa_window, pool.unified_swa_ring_size)
            if geometry != target_geometry:
                raise ValueError(
                    "target and draft continuation ring geometry differs: "
                    f"target={target_geometry}, draft={geometry}"
                )

        c4_attention = tuple(
            pool.kv_score_buffer.kv_score
            for pool in target_pool.compress_state_pools
            if pool is not None and pool.ratio == 4
        )
        c4_indexer = tuple(
            pool.kv_score_buffer.kv_score
            for pool in target_pool.indexer_compress_state_pools
            if pool is not None and pool.ratio == 4
        )
        self.c4_ring_size = int(target_pool.get_ring_size(4))
        self.c4_read_page_size = DSV4_CONTINUATION_C4_READ_PAGE_SIZE
        self.c128_state_pools = tuple(
            pool
            for pools in (
                target_pool.compress_state_pools,
                target_pool.indexer_compress_state_pools,
            )
            for pool in pools
            if pool is not None and pool.ratio == 128
        )
        if not c4_attention or not c4_indexer:
            raise ValueError("DSV4 continuation storage requires C4 state pools")

        specs = [
            (
                tuple(pool.unified_kv_pool.kv_buffer),
                int(pool.unified_swa_window),
            )
            for pool in ring_pools
        ]
        specs.extend(
            [
                (c4_attention, self.c4_read_page_size),
                (c4_indexer, self.c4_read_page_size),
            ]
        )

        offsets = []
        payload_bytes = 0
        max_element_size = 1
        for tensors, rows_per_slot in specs:
            dtype, width = self._validate_tensor_group(tensors)
            element_size = torch.empty((), dtype=dtype).element_size()
            max_element_size = max(max_element_size, element_size)
            payload_bytes = self._align(payload_bytes, element_size)
            group_bytes = len(tensors) * rows_per_slot * width * element_size
            offsets.append((payload_bytes, group_bytes, dtype, width))
            payload_bytes += group_bytes
        self.payload_bytes = self._align(payload_bytes, max_element_size)

        adapter = target_pool.memory_saver_adapter
        custom_mem_pool = target_pool.custom_mem_pool
        with (
            adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(custom_mem_pool)
                if custom_mem_pool is not None
                else nullcontext()
            ),
        ):
            self.buffer = torch.empty(
                (num_slots + 1, self.payload_bytes),
                dtype=torch.uint8,
                device=self.device,
            )

        groups = []
        for (tensors, rows_per_slot), (
            offset,
            group_bytes,
            dtype,
            width,
        ) in zip(specs, offsets, strict=True):
            groups.append(
                _TensorGroup(
                    tensors=tensors,
                    rows_per_slot=rows_per_slot,
                    slot_view=self._make_slot_view(
                        offset=offset,
                        group_bytes=group_bytes,
                        dtype=dtype,
                        tensor_count=len(tensors),
                        rows_per_slot=rows_per_slot,
                        width=width,
                    ),
                )
            )

        ring_group_count = len(ring_pools)
        self.ring_groups = tuple(groups[:ring_group_count])
        self.c4_attention_group = groups[ring_group_count]
        self.c4_indexer_group = groups[ring_group_count + 1]
        self.allocator = MambaSlotAllocator(size=num_slots, device=self.device)
        self._ready_events: dict[int, object] = {}

    @staticmethod
    def _align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    @staticmethod
    def _validate_tensor_group(
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.dtype, int]:
        if not tensors:
            raise ValueError("continuation tensor group cannot be empty")
        first = tensors[0]
        if first.ndim != 2:
            raise ValueError(f"continuation buffers must be 2D, got {first.ndim}D")
        for tensor in tensors[1:]:
            if tensor.ndim != 2:
                raise ValueError(f"continuation buffers must be 2D, got {tensor.ndim}D")
            if tensor.dtype != first.dtype or tensor.shape[1] != first.shape[1]:
                raise ValueError("continuation tensor group has mixed layouts")
        return first.dtype, int(first.shape[1])

    def _make_slot_view(
        self,
        *,
        offset: int,
        group_bytes: int,
        dtype: torch.dtype,
        tensor_count: int,
        rows_per_slot: int,
        width: int,
    ) -> torch.Tensor:
        element_size = torch.empty((), dtype=dtype).element_size()
        raw = self.buffer[:, offset : offset + group_bytes]
        typed = raw.view(dtype)
        return typed.as_strided(
            size=(self.num_slots + 1, tensor_count, rows_per_slot, width),
            stride=(
                self.payload_bytes // element_size,
                rows_per_slot * width,
                width,
                1,
            ),
        )

    def available_size(self) -> int:
        return self.allocator.available_size()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.allocator.alloc(need_size)

    def free(self, slots: torch.Tensor) -> None:
        if slots.numel() == 0:
            return
        self.allocator.free(slots)

    def clear(self) -> None:
        for event in set(self._ready_events.values()):
            event.synchronize()
        self.allocator.clear()
        self._ready_events.clear()

    def get_kv_size_bytes(self) -> int:
        return self.buffer.nbytes

    def _validate_endpoint(self, endpoint: int) -> None:
        if endpoint <= 0 or endpoint % self.logical_page_size != 0:
            raise ValueError(
                "DSV4 continuation endpoint must be a positive logical-page "
                f"boundary: endpoint={endpoint}, page={self.logical_page_size}"
            )
        if endpoint < self.target_pool.unified_swa_window:
            raise ValueError(
                f"DSV4 continuation endpoint {endpoint} is shorter than its window"
            )

    def _c4_state_rows(self, req_pool_idx: int, endpoint: int) -> torch.Tensor:
        last_full_loc = self.req_to_token_pool.req_to_token[req_pool_idx, endpoint - 1]
        last_swa_loc = self.target_pool.translate_loc_from_full_to_swa(last_full_loc)
        group = torch.div(
            last_swa_loc,
            self.logical_page_size,
            rounding_mode="floor",
        )
        state_loc = group.to(torch.int64) * self.c4_ring_size + (
            last_swa_loc % self.c4_ring_size
        )
        read_page_start = (
            torch.div(
                state_loc,
                self.c4_read_page_size,
                rounding_mode="floor",
            )
            * self.c4_read_page_size
        )
        return read_page_start + torch.arange(
            self.c4_read_page_size,
            dtype=torch.int64,
            device=self.device,
        )

    def _c4_state_rows_batch(
        self, req_pool_indices: torch.Tensor, endpoints: torch.Tensor
    ) -> torch.Tensor:
        last_full_loc = self.req_to_token_pool.req_to_token[
            req_pool_indices, endpoints - 1
        ]
        last_swa_loc = self.target_pool.translate_loc_from_full_to_swa(last_full_loc)
        group = torch.div(
            last_swa_loc,
            self.logical_page_size,
            rounding_mode="floor",
        )
        state_loc = group.to(torch.int64) * self.c4_ring_size + (
            last_swa_loc % self.c4_ring_size
        )
        read_page_start = (
            torch.div(
                state_loc,
                self.c4_read_page_size,
                rounding_mode="floor",
            )
            * self.c4_read_page_size
        )
        return read_page_start[:, None] + torch.arange(
            self.c4_read_page_size,
            dtype=torch.int64,
            device=self.device,
        )

    def _clear_c128_state(self, req_pool_idx: int) -> None:
        self.target_pool.clear_c128_req_state(req_pool_idx)

    def _ring_rows(self, pool, req_pool_idx: int, endpoint: int) -> torch.Tensor:
        positions = torch.arange(
            endpoint - pool.unified_swa_window,
            endpoint,
            dtype=torch.int64,
            device=self.device,
        )
        return req_pool_idx * pool.unified_swa_ring_size + (
            positions % pool.unified_swa_ring_size
        )

    def _ring_rows_batch(
        self, pool, req_pool_indices: torch.Tensor, endpoints: torch.Tensor
    ) -> torch.Tensor:
        offsets = torch.arange(
            -pool.unified_swa_window,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        positions = endpoints[:, None] + offsets
        return req_pool_indices[:, None] * pool.unified_swa_ring_size + (
            positions % pool.unified_swa_ring_size
        )

    @staticmethod
    def _copy_into_group(
        group: _TensorGroup, slot: int, source_rows: torch.Tensor
    ) -> None:
        for layer, tensor in enumerate(group.tensors):
            group.slot_view[slot, layer].copy_(tensor.index_select(0, source_rows))

    @staticmethod
    def _copy_into_group_batch(
        group: _TensorGroup, slots: torch.Tensor, source_rows: torch.Tensor
    ) -> None:
        batch_size = int(source_rows.shape[0])
        flat_rows = source_rows.reshape(-1)
        for layer, tensor in enumerate(group.tensors):
            values = tensor.index_select(0, flat_rows).reshape(
                batch_size, group.rows_per_slot, tensor.shape[1]
            )
            group.slot_view[:, layer].index_copy_(0, slots, values)

    @staticmethod
    def _copy_from_group(
        group: _TensorGroup, slot: int, destination_rows: torch.Tensor
    ) -> None:
        for layer, tensor in enumerate(group.tensors):
            tensor.index_copy_(0, destination_rows, group.slot_view[slot, layer])

    @staticmethod
    def _copy_from_group_batch(
        group: _TensorGroup, slots: torch.Tensor, destination_rows: torch.Tensor
    ) -> None:
        flat_rows = destination_rows.reshape(-1)
        for layer, tensor in enumerate(group.tensors):
            values = (
                group.slot_view[:, layer]
                .index_select(0, slots)
                .reshape(-1, tensor.shape[1])
            )
            tensor.index_copy_(0, flat_rows, values)

    def capture(
        self,
        *,
        slot: int,
        req_pool_idx: int,
        endpoint: int,
    ) -> None:
        self._validate_endpoint(endpoint)
        if not 1 <= slot <= self.num_slots:
            raise ValueError(f"invalid DSV4 continuation slot {slot}")
        self.wait_ready(slot)

        for pool, group in zip(
            (self.target_pool, *self.draft_pools),
            self.ring_groups,
            strict=True,
        ):
            self._copy_into_group(
                group,
                slot,
                self._ring_rows(pool, req_pool_idx, endpoint),
            )

        state_rows = self._c4_state_rows(req_pool_idx, endpoint)
        self._copy_into_group(self.c4_attention_group, slot, state_rows)
        self._copy_into_group(self.c4_indexer_group, slot, state_rows)

        if self.device.type == "cuda":
            event = self.device_module.Event()
            event.record(self.device_module.current_stream(self.device))
            self._ready_events[slot] = event

    def capture_batch(
        self,
        *,
        slots: torch.Tensor,
        req_pool_indices: torch.Tensor,
        endpoints: Sequence[int],
    ) -> None:
        if slots.numel() == 0:
            return
        if slots.numel() != req_pool_indices.numel() or slots.numel() != len(endpoints):
            raise ValueError("DSV4 continuation capture batch lengths differ")
        for endpoint in endpoints:
            self._validate_endpoint(endpoint)

        slots = slots.to(device=self.device, dtype=torch.int64)
        req_pool_indices = req_pool_indices.to(device=self.device, dtype=torch.int64)
        endpoint_tensor = torch.tensor(endpoints, dtype=torch.int64, device=self.device)
        self.wait_ready_indices(slots)

        for pool, group in zip(
            (self.target_pool, *self.draft_pools),
            self.ring_groups,
            strict=True,
        ):
            self._copy_into_group_batch(
                group,
                slots,
                self._ring_rows_batch(pool, req_pool_indices, endpoint_tensor),
            )

        state_rows = self._c4_state_rows_batch(req_pool_indices, endpoint_tensor)
        self._copy_into_group_batch(self.c4_attention_group, slots, state_rows)
        self._copy_into_group_batch(self.c4_indexer_group, slots, state_rows)
        self.record_ready_indices(slots)

    def wait_ready(self, slot: int) -> None:
        event = self._ready_events.get(slot)
        if event is not None:
            self.device_module.current_stream(self.device).wait_event(event)

    def wait_ready_indices(self, slots: torch.Tensor) -> None:
        for slot in slots.detach().cpu().tolist():
            self.wait_ready(int(slot))

    def record_ready_indices(self, slots: torch.Tensor) -> None:
        if self.device.type != "cuda" or slots.numel() == 0:
            return
        event = self.device_module.Event()
        event.record(self.device_module.current_stream(self.device))
        for slot in slots.detach().cpu().tolist():
            self._ready_events[int(slot)] = event

    def restore(
        self,
        *,
        slot: int,
        req_pool_idx: int,
        endpoint: int,
    ) -> None:
        self._validate_endpoint(endpoint)
        if not 1 <= slot <= self.num_slots:
            raise ValueError(f"invalid DSV4 continuation slot {slot}")
        self.wait_ready(slot)

        for pool, group in zip(
            (self.target_pool, *self.draft_pools),
            self.ring_groups,
            strict=True,
        ):
            self._copy_from_group(
                group,
                slot,
                self._ring_rows(pool, req_pool_idx, endpoint),
            )

        state_rows = self._c4_state_rows(req_pool_idx, endpoint)
        self._copy_from_group(self.c4_attention_group, slot, state_rows)
        self._copy_from_group(self.c4_indexer_group, slot, state_rows)
        self._clear_c128_state(req_pool_idx)

    def restore_batch(
        self,
        *,
        slots: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: Sequence[int],
        endpoints: Sequence[int],
    ) -> None:
        if slots.numel() == 0:
            return
        if not (
            slots.numel()
            == req_pool_indices.numel()
            == len(req_pool_indices_cpu)
            == len(endpoints)
        ):
            raise ValueError("DSV4 continuation restore batch lengths differ")
        for endpoint in endpoints:
            self._validate_endpoint(endpoint)

        slots = slots.to(device=self.device, dtype=torch.int64)
        req_pool_indices = req_pool_indices.to(device=self.device, dtype=torch.int64)
        endpoint_tensor = torch.tensor(endpoints, dtype=torch.int64, device=self.device)
        self.wait_ready_indices(slots)

        for pool, group in zip(
            (self.target_pool, *self.draft_pools),
            self.ring_groups,
            strict=True,
        ):
            self._copy_from_group_batch(
                group,
                slots,
                self._ring_rows_batch(pool, req_pool_indices, endpoint_tensor),
            )

        state_rows = self._c4_state_rows_batch(req_pool_indices, endpoint_tensor)
        self._copy_from_group_batch(self.c4_attention_group, slots, state_rows)
        self._copy_from_group_batch(self.c4_indexer_group, slots, state_rows)
        for req_pool_idx in req_pool_indices_cpu:
            self._clear_c128_state(int(req_pool_idx))
        self.record_ready_indices(slots)
