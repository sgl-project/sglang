import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_pages", "full_page_representatives"])
def get_and_clear_swa_pages_kernel(
    full_page_representatives,
    mapping,
    swa_pages,
    peers_mapped,
    page_mappings_valid,
    num_pages,
    index_stride: tl.constexpr,
    page_size: tl.constexpr,
    CHECK_PAGE_MAPPINGS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_OFFSETS: tl.constexpr,
):
    rep_offsets = tl.program_id(0) * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    rep_mask = rep_offsets < num_pages
    full_reps = tl.load(
        full_page_representatives + rep_offsets * index_stride,
        mask=rep_mask,
        other=0,
    ).to(tl.int64)

    # Resolve one SWA page per FULL-page representative:
    # swa_reps = mapping[full_reps]
    # swa_pages = swa_reps // page_size
    swa_reps = tl.load(mapping + full_reps, mask=rep_mask, other=0)
    tl.store(swa_pages + rep_offsets, swa_reps // page_size, mask=rep_mask)
    tl.store(peers_mapped + rep_offsets, swa_reps > 0, mask=rep_mask)

    page_offsets = tl.arange(0, BLOCK_OFFSETS)
    full_page_starts = full_reps // page_size * page_size
    mapping_offsets = full_page_starts[:, None] + page_offsets[None, :]
    mapping_mask = rep_mask[:, None] & (page_offsets[None, :] < page_size)
    if CHECK_PAGE_MAPPINGS:
        page_mapping = tl.load(
            mapping + mapping_offsets,
            mask=mapping_mask,
            other=0,
        )
        # Ignore zeros; each mapped slot must share its representative's SWA page.
        same_swa_page = page_mapping // page_size == swa_reps[:, None] // page_size
        page_mapping_valid = (swa_reps > 0) & (
            tl.sum(((page_mapping > 0) & ~same_swa_page).to(tl.int32), axis=1) == 0
        )
        tl.store(
            page_mappings_valid + rep_offsets,
            page_mapping_valid,
            mask=rep_mask,
        )

    # `mapping_offsets` includes `full_reps`. Finish every `mapping[full_reps]`
    # load before any warp clears `mapping[mapping_offsets]`.
    tl.debug_barrier()
    tl.store(
        mapping + mapping_offsets,
        0,
        mask=mapping_mask,
    )


def get_and_clear_swa_pages(
    full_page_representatives: torch.Tensor,
    mapping: torch.Tensor,
    page_size: int,
    check_page_mappings: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Resolve and clear mappings; input must represent distinct FULL pages."""
    if check_page_mappings:
        assert torch.all(
            (full_page_representatives >= 0)
            & (full_page_representatives < mapping.numel() // page_size * page_size)
        ), "FULL page representative out of bounds"
    num_pages = full_page_representatives.numel()
    swa_pages = torch.empty(num_pages, dtype=mapping.dtype, device=mapping.device)
    peers_mapped = torch.empty(num_pages, dtype=torch.bool, device=mapping.device)
    page_mappings_valid = (
        torch.empty(num_pages, dtype=torch.bool, device=mapping.device)
        if check_page_mappings
        else None
    )
    if num_pages:
        block_offsets = triton.next_power_of_2(page_size)
        block_pages = max(1, 256 // block_offsets)
        get_and_clear_swa_pages_kernel[(triton.cdiv(num_pages, block_pages),)](
            full_page_representatives,
            mapping,
            swa_pages,
            peers_mapped,
            page_mappings_valid if page_mappings_valid is not None else peers_mapped,
            num_pages,
            full_page_representatives.stride(0),
            page_size,
            check_page_mappings,
            block_pages,
            block_offsets,
        )
    return swa_pages, peers_mapped, page_mappings_valid


# free_page_ptr aliases self.free_pages, which the paged allocator re-slices
# after every allocation (self.free_pages = self.free_pages[num_new_pages:]).
# Slicing only advances data_ptr() by num_new_pages * 8 bytes, so the pointer
# flips between 16-byte-aligned and unaligned across calls. Triton specializes
# on pointer alignment by default and bakes it into the cache key, compiling two
# kernel variants (one with tt.divisibility=16 on free_page_ptr, one without)
# so the second prefill on a fresh DCP server hits the alternate alignment and
# pays an extra ~100ms JIT for that kernel variant. do_not_specialize skips
# that specialization so only one kernel is ever compiled; the perf cost is
# negligible (this kernel runs in ~10us and only loads ~4KB through this ptr).
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages using a dynamic blocked loop.
    # The loop bound is derived from num_part2 (runtime value), so Triton
    # generates a real loop instead of unrolling -- no constexpr dependency
    # on extend size and only one kernel compilation.
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


# Same free_page_ptr alignment rationale as alloc_extend_kernel above.
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)
