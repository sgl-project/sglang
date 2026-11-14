import triton
import triton.language as tl


@triton.jit
def _page_offset(length, page_size: tl.constexpr):
    if (page_size & (page_size - 1)) == 0:
        return length.to(tl.int32) & (page_size - 1)
    return (length % page_size).to(tl.int32)


@triton.jit(do_not_specialize=["free_page_ptr", "last_loc_ptr"])
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    page_size: tl.constexpr,
):
    """Allocate paged slots in the NPU small-page path.

    Preserve absolute length dtypes and use int32 for small growth and residues.
    The caller selects this kernel when fewer than 200 new pages are needed.
    """
    BLOCK_EXTEND: tl.constexpr = 2048
    BLOCK_BS: tl.constexpr = 256
    pid = tl.program_id(0)
    ix = tl.arange(0, BLOCK_BS)
    pre = tl.load(pre_lens_ptr + ix, mask=ix <= pid, other=0)
    seq = tl.load(seq_lens_ptr + ix, mask=ix <= pid, other=0)
    growth = (seq - pre).to(tl.int32)
    rem = _page_offset(pre, page_size)
    pages = (rem + growth + page_size - 1) // page_size - (rem > 0).to(tl.int32)
    for start in range(BLOCK_BS, pid + 1, BLOCK_BS):
        pos = start + ix
        pre_b = tl.load(pre_lens_ptr + pos, mask=pos <= pid, other=0)
        seq_b = tl.load(seq_lens_ptr + pos, mask=pos <= pid, other=0)
        g = (seq_b - pre_b).to(tl.int32)
        r = _page_offset(pre_b, page_size)
        growth += g
        pages += (r + g + page_size - 1) // page_size - (r > 0).to(tl.int32)
    pre_self = tl.load(pre_lens_ptr + pid)
    seq_self = tl.load(seq_lens_ptr + pid)
    size = (seq_self - pre_self).to(tl.int32)
    rem_self = _page_offset(pre_self, page_size)
    own_pages = (rem_self + size + page_size - 1) // page_size - (rem_self > 0).to(
        tl.int32
    )
    out_start = tl.sum(growth) - size
    page_start = tl.sum(pages) - own_pages
    if (page_size & (page_size - 1)) == 0:
        existing = tl.minimum(size, (-rem_self) & (page_size - 1))
    else:
        existing = tl.minimum(size, (page_size - rem_self) % page_size)
    last = tl.load(last_loc_ptr + pid).to(tl.int64)
    # Complete the existing partial page, then fill full pages and the tail.
    offset_page = tl.arange(0, page_size)
    tl.store(
        out_indices + out_start + offset_page,
        last + offset_page + 1,
        mask=offset_page < existing,
    )
    remaining = size - existing
    if remaining == 0:
        return
    full = (remaining // page_size) * page_size
    for block in range(tl.cdiv(full, BLOCK_EXTEND)):
        offset = block * BLOCK_EXTEND + tl.arange(0, BLOCK_EXTEND)
        mask = offset < full
        page = tl.load(
            free_page_ptr + page_start + offset // page_size, mask=mask, other=0
        ).to(tl.int64)
        tl.store(
            out_indices + out_start + existing + offset,
            page * page_size + offset % page_size,
            mask=mask,
        )
    tail = remaining - full
    if tail == 0:
        return
    page = tl.load(free_page_ptr + page_start + full // page_size).to(tl.int64)
    tl.store(
        out_indices + out_start + existing + full + offset_page,
        page * page_size + offset_page,
        mask=offset_page < tail,
    )
