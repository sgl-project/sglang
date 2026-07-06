import torch
import triton
import triton.language as tl


@triton.jit
def shared_kv_rank_major_slot_indices_kernel(
    src,
    dst,
    n_elements: tl.constexpr,
    cp_size: tl.constexpr,
    slots_per_page: tl.constexpr,
    pages_per_rank: tl.constexpr,
    padding_value: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(src + offsets, mask=mask, other=padding_value)
    valid = values != padding_value
    safe_values = tl.where(valid, values, 0)
    pages = safe_values // slots_per_page
    page_offsets = safe_values - pages * slots_per_page
    owner_rank = pages % cp_size
    local_pages = pages // cp_size
    shared_pages = owner_rank * pages_per_rank + local_pages
    shared_slots = shared_pages * slots_per_page + page_offsets
    tl.store(dst + offsets, tl.where(valid, shared_slots, values), mask=mask)


@triton.jit
def shared_kv_local_slot_indices_kernel(
    src,
    dst,
    n_elements: tl.constexpr,
    cp_size: tl.constexpr,
    slots_per_page: tl.constexpr,
    padding_value: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(src + offsets, mask=mask, other=padding_value)
    valid = values != padding_value
    safe_values = tl.where(valid, values, 0)
    pages = safe_values // slots_per_page
    page_offsets = safe_values - pages * slots_per_page
    local_pages = pages // cp_size
    local_slots = local_pages * slots_per_page + page_offsets
    tl.store(dst + offsets, tl.where(valid, local_slots, values), mask=mask)


def _shared_kv_map_slots(
    kernel,
    slot_indices: torch.Tensor,
    *args,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if not slot_indices.is_cuda:
        return None
    out = torch.empty_like(slot_indices, dtype=output_dtype or slot_indices.dtype)
    n_elements = slot_indices.numel()
    block = 256
    kernel[(triton.cdiv(n_elements, block),)](
        slot_indices, out, n_elements, *args, BLOCK=block
    )
    return out


def shared_kv_rank_major_slot_indices_triton(
    slot_indices: torch.Tensor,
    *,
    cp_size: int,
    slots_per_page: int,
    pages_per_rank: int,
    padding_value: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    return _shared_kv_map_slots(
        shared_kv_rank_major_slot_indices_kernel,
        slot_indices,
        cp_size,
        slots_per_page,
        pages_per_rank,
        padding_value,
        output_dtype=output_dtype,
    )


def shared_kv_local_slot_indices_triton(
    slot_indices: torch.Tensor,
    *,
    cp_size: int,
    slots_per_page: int,
    padding_value: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    return _shared_kv_map_slots(
        shared_kv_local_slot_indices_kernel,
        slot_indices,
        cp_size,
        slots_per_page,
        padding_value,
        output_dtype=output_dtype,
    )
