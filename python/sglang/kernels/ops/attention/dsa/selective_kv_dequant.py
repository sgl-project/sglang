import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 256


@triton.jit
def _begin_dense_epoch_kernel(epoch_ptr, num_unique_ptr):
    current_epoch = tl.load(epoch_ptr)
    tl.store(epoch_ptr, current_epoch + 1)
    tl.store(num_unique_ptr, 0)


@triton.jit
def _mark_selected_slots_kernel(
    page_table_ptr,
    flat_topk_ptr,
    slot_epoch_ptr,
    epoch_ptr,
    num_occurrences,
    page_table_len,
    num_pool_rows,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < num_occurrences
    logical = tl.load(flat_topk_ptr + offsets, mask=in_bounds, other=-1).to(tl.int32)
    valid_logical = in_bounds & (logical >= 0) & (logical < page_table_len)
    physical = tl.load(
        page_table_ptr + logical,
        mask=valid_logical,
        other=0,
    ).to(tl.int32)
    valid_physical = valid_logical & (physical >= 0) & (physical < num_pool_rows)
    current_epoch = tl.load(epoch_ptr)
    # Duplicate occurrences may race here, but atomic exchange makes the
    # generation publication formally ordered; every writer publishes the
    # same int64 value.
    tl.atomic_xchg(
        slot_epoch_ptr + physical,
        current_epoch,
        mask=valid_physical,
    )


@triton.jit
def _local_scan_selected_slots_kernel(
    slot_epoch_ptr,
    slot_to_compact_ptr,
    block_offsets_ptr,
    epoch_ptr,
    num_pool_rows,
    BLOCK_SIZE: tl.constexpr,
):
    physical = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = physical < num_pool_rows
    current_epoch = tl.load(epoch_ptr)
    selected = in_bounds & (
        tl.load(slot_epoch_ptr + physical, mask=in_bounds, other=-1) == current_epoch
    )
    selected_i32 = selected.to(tl.int32)
    local_rank = tl.cumsum(selected_i32, axis=0) - 1
    tl.store(
        slot_to_compact_ptr + physical,
        local_rank,
        mask=selected,
    )
    block_count = tl.sum(selected_i32, axis=0)
    tl.store(block_offsets_ptr + tl.program_id(0), block_count)


@triton.jit
def _scan_block_counts_kernel(
    block_offsets_ptr,
    num_unique_ptr,
    num_blocks,
    SCAN_SIZE: tl.constexpr,
):
    block = tl.arange(0, SCAN_SIZE)
    in_bounds = block < num_blocks
    counts = tl.load(block_offsets_ptr + block, mask=in_bounds, other=0).to(tl.int32)
    exclusive_offsets = tl.cumsum(counts, axis=0) - counts
    tl.store(block_offsets_ptr + block, exclusive_offsets, mask=in_bounds)
    tl.store(num_unique_ptr, tl.sum(counts, axis=0))


@triton.jit
def _finalize_compact_rows_kernel(
    slot_epoch_ptr,
    slot_to_compact_ptr,
    selected_physical_slots_ptr,
    block_offsets_ptr,
    epoch_ptr,
    num_pool_rows,
    selection_capacity,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    physical = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = physical < num_pool_rows
    current_epoch = tl.load(epoch_ptr)
    selected = in_bounds & (
        tl.load(slot_epoch_ptr + physical, mask=in_bounds, other=-1) == current_epoch
    )
    local_rank = tl.load(
        slot_to_compact_ptr + physical,
        mask=selected,
        other=0,
    ).to(tl.int32)
    block_offset = tl.load(block_offsets_ptr + block_id).to(tl.int32)
    compact = local_rank + block_offset
    within_capacity = selected & (compact < selection_capacity)
    tl.store(slot_to_compact_ptr + physical, compact, mask=within_capacity)
    tl.store(
        selected_physical_slots_ptr + compact,
        physical,
        mask=within_capacity,
    )


@triton.jit
def _remap_topk_kernel(
    page_table_ptr,
    flat_topk_ptr,
    slot_to_compact_ptr,
    remapped_topk_ptr,
    num_occurrences,
    page_table_len,
    num_pool_rows,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < num_occurrences
    logical = tl.load(flat_topk_ptr + offsets, mask=in_bounds, other=-1).to(tl.int32)
    valid_logical = in_bounds & (logical >= 0) & (logical < page_table_len)
    physical = tl.load(
        page_table_ptr + logical,
        mask=valid_logical,
        other=0,
    ).to(tl.int32)
    valid_physical = valid_logical & (physical >= 0) & (physical < num_pool_rows)
    compact = tl.load(
        slot_to_compact_ptr + physical,
        mask=valid_physical,
        other=-1,
    ).to(tl.int32)
    remapped = tl.where(valid_physical, compact, -1)
    tl.store(remapped_topk_ptr + offsets, remapped, mask=in_bounds)


def _require_cuda_int_tensor(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def deduplicate_kv_slots_dense_epoch(
    page_table_1_flattened: torch.Tensor,
    flat_topk_indices: torch.Tensor,
    *,
    slot_epoch: torch.Tensor,
    slot_to_compact: torch.Tensor,
    selected_physical_slots: torch.Tensor,
    remapped_topk: torch.Tensor,
    block_offsets: torch.Tensor,
    epoch: torch.Tensor,
    num_unique: torch.Tensor,
    num_pool_rows: int,
    selection_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate physical DSA KV slots with persistent generation tables.

    The six launch phases are ordered on the current CUDA stream:

    1. advance the int64 generation and reset ``num_unique``;
    2. mark every valid physical slot;
    3. prefix-scan each 256-slot block;
    4. prefix-scan the block counts;
    5. publish deterministic physical-slot-sorted compact rows;
    6. remap every logical top-k occurrence to its compact row.

    No device value is converted to Python, so the wrapper introduces no host
    synchronization and can be replayed inside a CUDA Graph after buffers have
    been allocated.
    """
    _require_cuda_int_tensor(
        "page_table_1_flattened", page_table_1_flattened, torch.int32
    )
    _require_cuda_int_tensor("flat_topk_indices", flat_topk_indices, torch.int32)
    _require_cuda_int_tensor("slot_epoch", slot_epoch, torch.int64)
    _require_cuda_int_tensor("slot_to_compact", slot_to_compact, torch.int32)
    _require_cuda_int_tensor(
        "selected_physical_slots", selected_physical_slots, torch.int32
    )
    _require_cuda_int_tensor("remapped_topk", remapped_topk, torch.int32)
    _require_cuda_int_tensor("block_offsets", block_offsets, torch.int32)
    _require_cuda_int_tensor("epoch", epoch, torch.int64)
    _require_cuda_int_tensor("num_unique", num_unique, torch.int32)

    if page_table_1_flattened.ndim != 1:
        raise ValueError("page_table_1_flattened must be one-dimensional")
    if remapped_topk.shape != flat_topk_indices.shape:
        raise ValueError(
            "remapped_topk shape must match flat_topk_indices: "
            f"{tuple(remapped_topk.shape)} != {tuple(flat_topk_indices.shape)}"
        )
    if epoch.numel() != 1 or num_unique.numel() != 1:
        raise ValueError("epoch and num_unique must each contain one element")
    if num_pool_rows <= 0 or num_pool_rows > slot_epoch.numel():
        raise ValueError(
            f"num_pool_rows must be in [1, {slot_epoch.numel()}], got {num_pool_rows}"
        )
    if slot_to_compact.numel() < num_pool_rows:
        raise ValueError("slot_to_compact is smaller than num_pool_rows")

    required_capacity = min(page_table_1_flattened.numel(), flat_topk_indices.numel())
    if selection_capacity < required_capacity:
        raise ValueError(
            "selection_capacity must cover the conservative unique-row bound: "
            f"{selection_capacity} < {required_capacity}"
        )
    if selection_capacity > selected_physical_slots.numel():
        raise ValueError(
            "selected_physical_slots is smaller than selection_capacity: "
            f"{selected_physical_slots.numel()} < {selection_capacity}"
        )

    num_occurrences = flat_topk_indices.numel()
    num_scan_blocks = triton.cdiv(num_pool_rows, _BLOCK_SIZE)
    if block_offsets.numel() < num_scan_blocks:
        raise ValueError(
            "block_offsets is smaller than the dense scan grid: "
            f"{block_offsets.numel()} < {num_scan_blocks}"
        )
    scan_size = triton.next_power_of_2(num_scan_blocks)
    _begin_dense_epoch_kernel[(1,)](epoch, num_unique)
    _mark_selected_slots_kernel[(triton.cdiv(num_occurrences, _BLOCK_SIZE),)](
        page_table_1_flattened,
        flat_topk_indices,
        slot_epoch,
        epoch,
        num_occurrences,
        page_table_1_flattened.numel(),
        num_pool_rows,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    _local_scan_selected_slots_kernel[(num_scan_blocks,)](
        slot_epoch,
        slot_to_compact,
        block_offsets,
        epoch,
        num_pool_rows,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    _scan_block_counts_kernel[(1,)](
        block_offsets,
        num_unique,
        num_scan_blocks,
        SCAN_SIZE=scan_size,
    )
    _finalize_compact_rows_kernel[(num_scan_blocks,)](
        slot_epoch,
        slot_to_compact,
        selected_physical_slots,
        block_offsets,
        epoch,
        num_pool_rows,
        selection_capacity,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    _remap_topk_kernel[(triton.cdiv(num_occurrences, _BLOCK_SIZE),)](
        page_table_1_flattened,
        flat_topk_indices,
        slot_to_compact,
        remapped_topk,
        num_occurrences,
        page_table_1_flattened.numel(),
        num_pool_rows,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return (
        selected_physical_slots[:selection_capacity],
        remapped_topk,
        num_unique,
    )
