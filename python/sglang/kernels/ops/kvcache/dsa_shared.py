import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op

_INDEXER_MATERIALIZE_DENSE_MAX_BATCH = 7
_INDEXER_MATERIALIZE_WORKERS_PER_REQUEST = 128


def _materialize_indexer_pages_grid(
    batch_size: int, pages_per_request: int
) -> tuple[int]:
    """Bound graph replay work by batch size, not maximum context width."""
    if min(batch_size, pages_per_request) <= 0:
        raise ValueError("Indexer materialize dimensions must be positive")
    if batch_size <= _INDEXER_MATERIALIZE_DENSE_MAX_BATCH:
        return (batch_size * pages_per_request,)
    return (batch_size * _INDEXER_MATERIALIZE_WORKERS_PER_REQUEST,)


def _validate_materialize_indexer_pages_dynamic(
    source_pages: torch.Tensor,
    target_pages: torch.Tensor,
    seq_len: torch.Tensor,
    *,
    page_size: int,
    source_page_capacity: int,
    target_page_capacity: int,
) -> None:
    # The page tables and sequence lengths are graph-stable buffers whose
    # values change on replay. Capturing these reductions would replay three
    # validation chains per layer, so keep dynamic bounds checks at the eager
    # boundary while retaining the geometry checks on every invocation.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    torch._assert_async(
        torch.all(seq_len <= source_pages.shape[1] * page_size),
        "Indexer request exceeds the supplied page table",
    )
    torch._assert_async(
        torch.all((source_pages < 0) | (source_pages < source_page_capacity)),
        "Indexer source page exceeds shared VMM capacity",
    )
    torch._assert_async(
        torch.all((target_pages < 0) | (target_pages < target_page_capacity)),
        "Indexer target page exceeds Pool Demand Cache capacity",
    )


@triton.jit
def materialize_indexer_pages_kernel(
    target_ptr,
    source_ptr,
    source_pages_ptr,
    target_pages_ptr,
    seq_len_ptr,
    tags_ptr,
    epoch_ptr,
    source_row_stride: tl.constexpr,
    target_row_stride: tl.constexpr,
    source_page_batch_stride: tl.constexpr,
    source_page_stride: tl.constexpr,
    target_page_batch_stride: tl.constexpr,
    target_page_stride: tl.constexpr,
    seq_len_stride: tl.constexpr,
    pages_per_request: tl.constexpr,
    page_size: tl.constexpr,
    page_bytes: tl.constexpr,
    USE_TAGS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    table_entry = tl.program_id(0)
    batch = table_entry // pages_per_request
    request_page = table_entry - batch * pages_per_request
    seq_len = tl.load(seq_len_ptr + batch * seq_len_stride).to(tl.int64)
    active_pages = (seq_len + page_size - 1) // page_size
    if request_page >= active_pages:
        return
    source_offset = batch * source_page_batch_stride + request_page * source_page_stride
    target_offset = batch * target_page_batch_stride + request_page * target_page_stride
    source_page = tl.load(source_pages_ptr + source_offset).to(tl.int64)
    target_page = tl.load(target_pages_ptr + target_offset).to(tl.int64)
    if source_page < 0 or target_page < 0:
        return
    expected_tag = source_page + 1
    if USE_TAGS:
        epoch = tl.load(epoch_ptr).to(tl.int64)
        expected_tag = (epoch << 32) | expected_tag
        observed_tag = tl.load(tags_ptr + target_page)
        if request_page != active_pages - 1 and observed_tag == expected_tag:
            return
    for byte_begin in tl.static_range(0, page_bytes, BLOCK):
        offsets = byte_begin + tl.arange(0, BLOCK)
        mask = offsets < page_bytes
        values = tl.load(
            source_ptr + source_page * source_row_stride + offsets,
            mask=mask,
        )
        tl.store(
            target_ptr + target_page * target_row_stride + offsets,
            values,
            mask=mask,
        )
    if USE_TAGS:
        tl.store(tags_ptr + target_page, expected_tag)


@triton.jit
def materialize_indexer_pages_worker_kernel(
    target_ptr,
    source_ptr,
    source_pages_ptr,
    target_pages_ptr,
    seq_len_ptr,
    tags_ptr,
    epoch_ptr,
    source_row_stride: tl.constexpr,
    target_row_stride: tl.constexpr,
    source_page_batch_stride: tl.constexpr,
    source_page_stride: tl.constexpr,
    target_page_batch_stride: tl.constexpr,
    target_page_stride: tl.constexpr,
    seq_len_stride: tl.constexpr,
    page_size: tl.constexpr,
    page_bytes: tl.constexpr,
    USE_TAGS: tl.constexpr,
    WORKERS_PER_REQUEST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    worker_entry = tl.program_id(0)
    batch = worker_entry // WORKERS_PER_REQUEST
    worker = worker_entry - batch * WORKERS_PER_REQUEST
    seq_len = tl.load(seq_len_ptr + batch * seq_len_stride).to(tl.int64)
    active_pages = (seq_len + page_size - 1) // page_size
    if USE_TAGS:
        epoch = tl.load(epoch_ptr).to(tl.int64)
    request_page = worker
    while request_page < active_pages:
        source_offset = (
            batch * source_page_batch_stride + request_page * source_page_stride
        )
        target_offset = (
            batch * target_page_batch_stride + request_page * target_page_stride
        )
        source_page = tl.load(source_pages_ptr + source_offset).to(tl.int64)
        target_page = tl.load(target_pages_ptr + target_offset).to(tl.int64)
        valid = (source_page >= 0) & (target_page >= 0)
        expected_tag = source_page + 1
        if USE_TAGS:
            expected_tag = (epoch << 32) | expected_tag
            observed_tag = tl.load(tags_ptr + target_page, mask=valid, other=0)
            needs_copy = valid & (
                (request_page == active_pages - 1) | (observed_tag != expected_tag)
            )
        else:
            needs_copy = valid
        for byte_begin in tl.static_range(0, page_bytes, BLOCK):
            offsets = byte_begin + tl.arange(0, BLOCK)
            mask = needs_copy & (offsets < page_bytes)
            values = tl.load(
                source_ptr + source_page * source_row_stride + offsets,
                mask=mask,
                other=0,
            )
            tl.store(
                target_ptr + target_page * target_row_stride + offsets,
                values,
                mask=mask,
            )
        # tl.debug_barrier lowers to a CTA-wide PTX bar.sync. One CTA handles
        # multiple pages, so keep all warps on the same page until its payload
        # is complete, then publish the tag before the next page.
        tl.debug_barrier()
        if USE_TAGS:
            tl.store(tags_ptr + target_page, expected_tag, mask=needs_copy)
        tl.debug_barrier()
        request_page += WORKERS_PER_REQUEST


@register_custom_op(mutates_args=["target", "tags"])
def materialize_indexer_pages_triton(
    target: torch.Tensor,
    source: torch.Tensor,
    source_pages: torch.Tensor,
    target_pages: torch.Tensor,
    seq_len: torch.Tensor,
    *,
    page_size: int,
    tags: torch.Tensor | None = None,
    epoch: torch.Tensor | None = None,
) -> None:
    if target.dim() != 2 or source.dim() != 2:
        raise ValueError("Indexer page buffers must be two-dimensional")
    if target.shape[1] != source.shape[1]:
        raise ValueError("Indexer source and target page widths must match")
    if (
        source_pages.dim() != 2
        or target_pages.shape != source_pages.shape
        or seq_len.dim() != 1
        or source_pages.shape[0] != seq_len.shape[0]
    ):
        raise ValueError("Indexer Pool Demand Cache metadata has invalid geometry")
    _validate_materialize_indexer_pages_dynamic(
        source_pages,
        target_pages,
        seq_len,
        page_size=page_size,
        source_page_capacity=source.shape[0],
        target_page_capacity=target.shape[0],
    )
    if (tags is None) != (epoch is None):
        raise ValueError("Indexer Pool Demand Cache tags and epoch must be paired")
    use_tags = tags is not None
    if use_tags and (
        tags.dtype != torch.int64
        or tags.dim() != 1
        or tags.shape[0] < target.shape[0]
        or epoch.dtype != torch.int32
        or epoch.numel() != 1
    ):
        raise ValueError("Indexer Pool Demand Cache metadata has invalid geometry")
    tag_storage = tags if tags is not None else target
    epoch_storage = epoch if epoch is not None else seq_len
    common_args = (
        target,
        source,
        source_pages,
        target_pages,
        seq_len,
        tag_storage,
        epoch_storage,
        source.stride(0),
        target.stride(0),
        source_pages.stride(0),
        source_pages.stride(1),
        target_pages.stride(0),
        target_pages.stride(1),
        seq_len.stride(0),
    )
    grid = _materialize_indexer_pages_grid(source_pages.shape[0], source_pages.shape[1])
    if source_pages.shape[0] <= _INDEXER_MATERIALIZE_DENSE_MAX_BATCH:
        materialize_indexer_pages_kernel[grid](
            *common_args,
            source_pages.shape[1],
            page_size,
            target.shape[1],
            USE_TAGS=use_tags,
            BLOCK=1024,
            num_warps=8,
        )
    else:
        materialize_indexer_pages_worker_kernel[grid](
            *common_args,
            page_size,
            target.shape[1],
            USE_TAGS=use_tags,
            WORKERS_PER_REQUEST=_INDEXER_MATERIALIZE_WORKERS_PER_REQUEST,
            BLOCK=1024,
            num_warps=4,
        )


@triton.jit
def set_mla_kv_buffer_owner_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    owner_rank: tl.constexpr,
    owner_size: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    page = loc // page_size
    page_offset = loc - page * page_size
    owned = (loc >= 0) & ((page % owner_size) == owner_rank)
    local_loc = (page // owner_size) * page_size + page_offset
    mask = (offs < total_dim) & owned
    dst_ptr = kv_buffer_ptr + local_loc * buffer_stride + offs

    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs,
        mask=mask & is_nope,
        other=0,
    )
    src_rope = tl.load(
        cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
        mask=mask & ~is_nope,
        other=0,
    )
    src = tl.where(is_nope, src_nope, src_rope)
    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_owner_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    owner_rank: int,
    owner_size: int,
    page_size: int,
) -> None:
    if loc.numel() == 0:
        return
    total_dim = cache_k_nope.shape[-1] + cache_k_rope.shape[-1]
    set_mla_kv_buffer_owner_kernel[(loc.numel(),)](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        cache_k_nope.shape[-1],
        cache_k_rope.shape[-1],
        owner_rank,
        owner_size,
        page_size,
        BLOCK=triton.next_power_of_2(total_dim),
    )


@triton.jit
def set_mla_kv_buffer_owner_and_current_kernel(
    kv_buffer_ptr,
    encoded_rows_ptr,
    physical_rows_ptr,
    counts_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    encoded_query_stride: tl.constexpr,
    encoded_current_stride: tl.constexpr,
    physical_query_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    rows_per_request: tl.constexpr,
    max_current_rows: tl.constexpr,
    owner_rank: tl.constexpr,
    owner_size: tl.constexpr,
    page_size: tl.constexpr,
    pages_per_rank: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query_row = tl.program_id(0)
    current_row = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim

    position = query_row % rows_per_request
    count = position + 1
    request_begin = query_row - position
    source_row = request_begin + current_row
    query_loc = tl.load(loc_ptr + query_row).to(tl.int64)
    loc = tl.load(loc_ptr + source_row).to(tl.int64)
    valid = (query_loc >= 0) & (current_row < count) & (loc >= 0)
    page = loc // page_size
    page_offset = loc - page * page_size
    owner = page % owner_size
    owner_page = page // owner_size
    physical_row = (owner * pages_per_rank + owner_page) * page_size + page_offset
    owned = valid & ((page % owner_size) == owner_rank)
    local_loc = owner_page * page_size + page_offset
    byte_mask = (offs < total_dim) & valid

    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + source_row * nope_stride + offs,
        mask=byte_mask & is_nope,
        other=0,
    )
    src_rope = tl.load(
        cache_k_rope_ptr + source_row * rope_stride + (offs - nope_dim),
        mask=byte_mask & ~is_nope,
        other=0,
    )
    src = tl.where(is_nope, src_nope, src_rope)
    tl.store(
        encoded_rows_ptr
        + query_row * encoded_query_stride
        + current_row * encoded_current_stride
        + offs,
        src,
        mask=byte_mask,
    )
    tl.store(
        kv_buffer_ptr + local_loc * buffer_stride + offs,
        src,
        mask=(offs < total_dim) & owned & (current_row == position),
    )
    tl.store(
        physical_rows_ptr + query_row * physical_query_stride + current_row,
        tl.where(valid, physical_row, -1),
    )
    tl.store(
        counts_ptr + query_row,
        tl.where(query_loc >= 0, count, 0),
        mask=current_row == 0,
    )


def set_mla_kv_buffer_owner_and_current_triton(
    kv_buffer: torch.Tensor,
    encoded_rows: torch.Tensor,
    physical_rows: torch.Tensor,
    counts: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    rows_per_request: int,
    owner_rank: int,
    owner_size: int,
    page_size: int,
    pages_per_rank: int,
) -> None:
    if loc.numel() == 0:
        return
    total_dim = cache_k_nope.shape[-1] + cache_k_rope.shape[-1]
    set_mla_kv_buffer_owner_and_current_kernel[(loc.numel(), encoded_rows.shape[1])](
        kv_buffer,
        encoded_rows,
        physical_rows,
        counts,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        encoded_rows.stride(0),
        encoded_rows.stride(1),
        physical_rows.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        cache_k_nope.shape[-1],
        cache_k_rope.shape[-1],
        rows_per_request,
        encoded_rows.shape[1],
        owner_rank,
        owner_size,
        page_size,
        pages_per_rank,
        BLOCK=triton.next_power_of_2(total_dim),
    )
