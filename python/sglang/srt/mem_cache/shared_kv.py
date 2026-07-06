from __future__ import annotations

import math
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import torch

from sglang.srt.mem_cache.triton_ops.shared_kv import (
    shared_kv_local_slot_indices_triton,
    shared_kv_rank_major_slot_indices_triton,
)


def shared_kv_rank_major_slot_indices(
    slot_indices: torch.Tensor,
    *,
    cp_size: int,
    slots_per_page: int,
    pages_per_rank: int,
    padding_value: int = -1,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if cp_size <= 1:
        return (
            slot_indices.to(output_dtype)
            if output_dtype is not None
            else slot_indices
        )
    mapped = shared_kv_rank_major_slot_indices_triton(
        slot_indices,
        cp_size=cp_size,
        slots_per_page=slots_per_page,
        pages_per_rank=pages_per_rank,
        padding_value=padding_value,
        output_dtype=output_dtype,
    )
    if mapped is not None:
        return mapped
    valid = slot_indices != padding_value
    safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
    page_indices = torch.div(safe_indices, slots_per_page, rounding_mode="floor")
    page_offsets = safe_indices % slots_per_page
    owner_rank = page_indices % cp_size
    local_pages = torch.div(page_indices, cp_size, rounding_mode="floor")
    shared_pages = owner_rank * pages_per_rank + local_pages
    shared_slots = shared_pages * slots_per_page + page_offsets
    out = torch.where(valid, shared_slots, slot_indices)
    return out.to(output_dtype) if output_dtype is not None else out


def shared_kv_local_slot_indices(
    slot_indices: torch.Tensor,
    *,
    cp_size: int,
    slots_per_page: int,
    padding_value: int = -1,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if cp_size <= 1:
        return (
            slot_indices.to(output_dtype)
            if output_dtype is not None
            else slot_indices
        )
    mapped = shared_kv_local_slot_indices_triton(
        slot_indices,
        cp_size=cp_size,
        slots_per_page=slots_per_page,
        padding_value=padding_value,
        output_dtype=output_dtype,
    )
    if mapped is not None:
        return mapped
    valid = slot_indices != padding_value
    safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
    page_indices = torch.div(safe_indices, slots_per_page, rounding_mode="floor")
    page_offsets = safe_indices % slots_per_page
    local_pages = torch.div(page_indices, cp_size, rounding_mode="floor")
    local_slots = local_pages * slots_per_page + page_offsets
    out = torch.where(valid, local_slots, slot_indices)
    return out.to(output_dtype) if output_dtype is not None else out


def shared_kv_owned_slot_mask(
    slot_indices: torch.Tensor,
    *,
    owner_rank: int,
    cp_size: int,
    slots_per_page: int,
) -> torch.Tensor:
    valid = slot_indices >= 0
    if cp_size <= 1:
        return valid
    safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
    page_indices = torch.div(safe_indices, slots_per_page, rounding_mode="floor")
    return valid & ((page_indices % cp_size) == owner_rank)


@dataclass(frozen=True)
class SharedKVLayout:
    cp_size: int
    slots_per_page: int
    pages_per_rank: int
    padding_value: int = -1

    def translate_read_slots(
        self,
        slot_indices: torch.Tensor,
        *,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return shared_kv_rank_major_slot_indices(
            slot_indices,
            cp_size=self.cp_size,
            slots_per_page=self.slots_per_page,
            pages_per_rank=self.pages_per_rank,
            padding_value=self.padding_value,
            output_dtype=output_dtype,
        )

    def translate_write_slots(
        self,
        slot_indices: torch.Tensor,
        *,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return shared_kv_local_slot_indices(
            slot_indices,
            cp_size=self.cp_size,
            slots_per_page=self.slots_per_page,
            padding_value=self.padding_value,
            output_dtype=output_dtype,
        )

    def owned_slot_mask(
        self, slot_indices: torch.Tensor, *, owner_rank: int
    ) -> torch.Tensor:
        return shared_kv_owned_slot_mask(
            slot_indices,
            owner_rank=owner_rank,
            cp_size=self.cp_size,
            slots_per_page=self.slots_per_page,
        )

    def select_host_to_device_indices(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        *,
        owner_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.owned_slot_mask(device_indices, owner_rank=owner_rank)
        if not torch.any(mask):
            return host_indices[:0], device_indices[:0]
        return host_indices[mask], self.translate_write_slots(device_indices[mask]).to(
            device_indices.dtype
        )

    def select_device_to_host_indices(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        *,
        current_rank: int,
        io_rank: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current_rank != io_rank:
            return host_indices[:0], device_indices[:0]
        return host_indices, self.translate_read_slots(device_indices).to(
            device_indices.dtype
        )


class SharedHostTensorAllocator:
    def __init__(
        self,
        cpu_group,
        owner_rank: int,
        kind: str,
        log_label: str = "CP shared host memory",
    ):
        self.cpu_group = cpu_group
        self.owner_rank = owner_rank
        self.kind = kind
        self.log_label = log_label
        self.group_ranks = torch.distributed.get_process_group_ranks(cpu_group)
        self.cp_rank = torch.distributed.get_rank(group=cpu_group)
        self.is_owner = self.cp_rank == owner_rank
        self._shm = None
        self._tensor = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"SharedHostTensorAllocator only supports CPU allocations; got device={device!r}"
        if self._tensor is not None:
            return self._tensor

        numel = math.prod(dims)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        meta = [None]
        if self.is_owner:
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
            meta[0] = {"name": self._shm.name, "nbytes": nbytes}

        src_rank = self.group_ranks[self.owner_rank]
        torch.distributed.broadcast_object_list(meta, src=src_rank, group=self.cpu_group)
        if not self.is_owner:
            info = meta[0]
            self._shm = shared_memory.SharedMemory(name=info["name"], create=False)

        self._tensor = torch.frombuffer(self._shm.buf, dtype=dtype, count=numel).view(
            dims
        )
        return self._tensor

    def log_host_allocation(
        self,
        nbytes: int,
        logger,
        *,
        pool_name: str,
        token_capacity: int,
        page_num: int,
        page_size: int,
    ) -> None:
        action = "create" if self.is_owner else "attach"
        logger.info(
            "%s: %s %.2f GB host memory for %s "
            "(cp_rank=%s owner_rank=%s is_owner=%s token_capacity=%s "
            "page_num=%s page_size=%s kind=%s)",
            self.log_label,
            action,
            nbytes / 1e9,
            pool_name,
            self.cp_rank,
            self.owner_rank,
            self.is_owner,
            token_capacity,
            page_num,
            page_size,
            self.kind,
        )

    def destroy(self) -> None:
        shm = self._shm
        self._tensor = None
        self._shm = None
        if shm is None:
            return
        shm.close()
        if self.is_owner:
            shm.unlink()
