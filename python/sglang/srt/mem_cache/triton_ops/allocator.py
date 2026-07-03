import triton
import triton.language as tl


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


@triton.jit
def free_dual_pool_kernel(
    free_index_ptr,
    n_slots,
    self_epoch_ptr,
    self_cur_epoch,
    self_ring_ptr,
    self_cap,
    self_tail_ptr,
    mapping_ptr,
    swa_epoch_ptr,
    swa_cur_epoch,
    swa_ring_ptr,
    swa_cap,
    swa_tail_ptr,
    page_size: tl.constexpr,
    BLOCK: tl.constexpr,
    MARK_SELF: tl.constexpr,
    SCAN_SWA: tl.constexpr,
):
    """Single-launch page free. Elects one winner lane per touched page via a
    monotone epoch atomic_max (page-granular dedup, no assumptions about input
    structure) and appends freed pages straight to the device ring, so there
    is no bitmap sweep and no data-dependent shape anywhere.

    MARK_SELF: append elected pages to the self ring (off for standalone
    free_swa, which only elects to dedup the mapping scan).
    SCAN_SWA: winners additionally scan their page's full_to_swa mapping
    slots, append live swa pages to the swa ring (second epoch claim), and
    zero the mapping. Free is page-granular: a partial page in free_index
    still clears the whole page.
    Ring append order is nondeterministic (atomic), which only affects which
    physical page a later alloc picks, not any KV content.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    lane_m = offs < n_slots
    slots = tl.load(free_index_ptr + offs, mask=lane_m, other=0)
    fp = slots // page_size
    old = tl.atomic_max(self_epoch_ptr + fp, self_cur_epoch, mask=lane_m)
    winner = lane_m & (old < self_cur_epoch)
    if MARK_SELF:
        # broadcast the scalar tail pointer to a block so per-lane atomics work
        pos = tl.atomic_add(self_tail_ptr + fp * 0, 1, mask=winner)
        tl.store(self_ring_ptr + pos % self_cap, fp.to(tl.int64), mask=winner)
    if SCAN_SWA:
        base = fp * page_size
        for j in range(page_size):
            v = tl.load(mapping_ptr + base + j, mask=winner, other=0)
            sp = v // page_size
            live = winner & (v > 0)
            old2 = tl.atomic_max(swa_epoch_ptr + sp, swa_cur_epoch, mask=live)
            sw = live & (old2 < swa_cur_epoch)
            pos2 = tl.atomic_add(swa_tail_ptr + sp * 0, 1, mask=sw)
            tl.store(swa_ring_ptr + pos2 % swa_cap, sp.to(tl.int64), mask=sw)
            tl.store(mapping_ptr + base + j, 0, mask=winner)
