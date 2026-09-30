"""BF16 KPool Triton kernels for Ascend NPU.

The NPU cache stores pooled index keys directly as BF16. It does not apply a
Hadamard transform, FP8 quantization, or per-slot scale storage.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════
# 2. BF16 TRITON KERNEL
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _kpool_decode_update_and_maybe_write_cache_bf16_kernel(
    buf_bf16_ptr,           # [num_pages, SLOTS_PER_PAGE * HEAD_DIM] bfloat16
    tail_k_ptr,             # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] bfloat16
    tail_score_ptr,         # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] same as score dtype
    key_ptr,                # [batch, HEAD_DIM] bfloat16
    slot_score_ptr,         # [batch, HEAD_DIM] same as score dtype
    ape_ptr,                # [POOL_SIZE, HEAD_DIM] float32
    block_tables_ptr,       # [batch, BLOCK_TABLE_COLS] int32
    req_pool_indices_ptr,   # [>=batch] int
    positions_ptr,          # [>=batch] int
    seq_lens_ptr,           # [>=batch] int
    out_cache_loc_ptr,      # [>=batch] int
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    key_stride_0,
    slot_score_stride_0,
    ape_stride_0,
    block_tables_stride_0,
    block_tables_stride_1,
    REQ_POOL_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TABLE_COLS: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    # ── Load metadata ──
    req_raw = tl.load(req_pool_indices_ptr + row)
    req_valid = (req_raw >= 0) & (req_raw < REQ_POOL_SIZE)
    req = tl.minimum(tl.maximum(req_raw, 0), REQ_POOL_SIZE - 1)

    pos = tl.load(positions_ptr + row)
    safe_pos = tl.maximum(pos, 0)
    seq_len = tl.load(seq_lens_ptr + row)
    cache_loc = tl.load(out_cache_loc_ptr + row)
    pos_valid = req_valid & (cache_loc != 0) & (pos >= 0) & (pos < seq_len)

    slot = safe_pos % POOL_SIZE
    phys_slot = safe_pos % TAIL_SIZE

    # ── Load current key & score ──
    key = tl.load(
        key_ptr + row * key_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    score_current = tl.load(
        slot_score_ptr + row * slot_score_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    # ── (B) Pool compression (only when pool is full) ──
    if pos_valid & (slot == POOL_SIZE - 1):
        pool_logical_start = safe_pos - slot

        # Pass 1: find max score
        max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            max_score = tl.maximum(max_score, score)

        # Pass 2: softmax weighted average
        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            prob = tl.exp(score - max_score)
            denom += prob
            k_buf = tl.load(
                tail_k_ptr + req * tail_k_stride_0 + phys * tail_k_stride_1 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            k = tl.where(is_current, key, k_buf)
            acc += k * prob

        # BF16: no Hadamard, no FP8 quantize — just write directly
        compressed = (acc / denom).to(tl.bfloat16)

        # ── Compute write location in buf ──
        pool_id = safe_pos // POOL_SIZE
        pool_page_group = pool_id // SLOTS_PER_PAGE
        token_page_row = pool_page_group * POOL_SIZE
        token_page_row = tl.minimum(tl.maximum(token_page_row, 0), BLOCK_TABLE_COLS - 1)
        packed_page = tl.load(
            block_tables_ptr
            + row * block_tables_stride_0
            + token_page_row * block_tables_stride_1,
        )
        loc_page_index = packed_page.to(tl.int64)
        loc_token_offset_in_page = pool_id % SLOTS_PER_PAGE
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * HEAD_DIM
            + offs
        )

        tl.store(buf_bf16_ptr + out_k_offsets, compressed, mask=dim_mask)

    # ── (A) Tail buffer update (ALWAYS) ──
    tail_k_offset = req * tail_k_stride_0 + phys_slot * tail_k_stride_1 + offs
    tail_score_offset = (
        req * tail_score_stride_0 + phys_slot * tail_score_stride_1 + offs
    )
    update_mask = dim_mask & pos_valid
    tl.store(tail_k_ptr + tail_k_offset, key, mask=update_mask)
    tl.store(tail_score_ptr + tail_score_offset, score_current, mask=update_mask)


def kpool_decode_update_and_maybe_write_cache_bf16(
    buf: torch.Tensor,             # [num_pages, SLOTS_PER_PAGE * HEAD_DIM] bfloat16
    tail_k: torch.Tensor,          # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] bfloat16
    tail_score: torch.Tensor,      # [REQ_POOL_SIZE, TAIL_SIZE, HEAD_DIM] float32
    key: torch.Tensor,             # [batch, HEAD_DIM] bfloat16
    slot_score: torch.Tensor,      # [batch, HEAD_DIM] float32
    ape: torch.Tensor,             # [POOL_SIZE, HEAD_DIM] float32
    block_tables: torch.Tensor,    # [batch, BLOCK_TABLE_COLS] int32
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    pool_size: int,
    slots_per_page: int,
    head_dim: int = 128,
) -> None:
    """BF16 variant of kpool_decode_update_and_maybe_write_cache.

    Differences from FP8 original:
    - buf is bfloat16 (no FP8 quantization, no per-slot scale)
    - No Hadamard128 rotation
    - Softmax-weighted average written directly as BF16
    """
    batch = key.shape[0]
    if batch == 0:
        return

    assert buf.dtype == torch.bfloat16
    assert tail_k.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert tail_k.ndim == 3
    assert tail_score.shape == tail_k.shape
    assert tail_k.shape[2] == head_dim
    assert ape.shape == (pool_size, head_dim)

    tail_size = tail_k.shape[1]
    req_pool_size = tail_k.shape[0]
    buf_numel_per_page = buf.shape[1]
    block_table_cols = block_tables.shape[1]

    # Ensure contiguous
    buf = buf.contiguous()
    tail_k = tail_k.contiguous()
    tail_score = tail_score.contiguous()
    key = key.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    block_tables = block_tables.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    positions = positions.contiguous()
    seq_lens = seq_lens.contiguous()
    out_cache_loc = out_cache_loc.contiguous()

    _kpool_decode_update_and_maybe_write_cache_bf16_kernel[(batch,)](
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        key.stride(0),
        slot_score.stride(0),
        ape.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        REQ_POOL_SIZE=req_pool_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_size,
        HEAD_DIM=head_dim,
        BLOCK_TABLE_COLS=block_table_cols,
        SLOTS_PER_PAGE=slots_per_page,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


INDEX_HEAD_DIM = 128
# Scores may retain their upstream precision; pooling always accumulates in FP32.
KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _default_triton_num_programs(device: torch.device, n_pools: int) -> int:
    """Choose a Vector Core-sized Triton grid for ``n_pools`` work items.

    Args:
        device: NPU device that owns the input tensors.
        n_pools: Number of pooled keys to assemble.

    Returns:
        The number of Triton programs to launch. Each program processes one or
        more pools with a grid-stride loop when ``n_pools`` exceeds this value.
    """

    # Triton permits up to 65535 logical programs. Use that limit if the driver
    # does not expose the physical Vector Core count.
    fallback = min(n_pools, 65535)
    try:
        device_index = device.index
        if device_index is None and hasattr(torch, "npu"):
            device_index = torch.npu.current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(
            device_index
        )
        vector_core_count = int(properties["num_vectorcore"])
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return fallback
    if vector_core_count <= 0:
        return fallback
    return min(n_pools, vector_core_count)


@triton.jit
def _kpool_assemble_softmax_write_cache_npu_kernel(
    cache_ptr,  # Flattened BF16 cache: [CACHE_ROWS, HEAD_DIM].
    chunk_k_ptr,  # Current extend keys: [CHUNK_TOKENS, HEAD_DIM].
    chunk_score_ptr,  # Scores paired with chunk_k_ptr.
    tail_k_ptr,  # Per-request BF16 ring buffer: [REQS, TAIL_SIZE, HEAD_DIM].
    tail_score_ptr,  # Scores paired with tail_k_ptr.
    req_pool_idx_ptr,  # Request/ring-buffer row for each output pool.
    n_from_tail_ptr,  # Number of leading pool slots sourced from the tail.
    chunk_src_start_ptr,  # First chunk row used by each output pool.
    tail_logical_base_ptr,  # Logical start in the request tail ring.
    ape_ptr,  # Additive pooling position scores: [POOL_SIZE, HEAD_DIM].
    loc_ptr,  # Flattened cache row written by each output pool.
    write_mask_ptr,  # Optional per-pool write mask.
    n_pools,  # Number of entries in every plan tensor above.
    # Only leading strides are passed; HEAD_DIM is contiguous by contract.
    cache_stride_0,
    chunk_k_stride_0,
    chunk_score_stride_0,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    ape_stride_0,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Cap the launch grid at the Vector Core count. Programs consume any
    # remaining pools through this grid-stride loop.
    row = tl.program_id(0)
    program_count = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    head_mask = offs < HEAD_DIM

    while row < n_pools:
        active = True
        if HAS_WRITE_MASK:
            active = tl.load(write_mask_ptr + row)

        if active:
            # Load the plan entry that describes how this pool is assembled.
            n_tail = tl.load(n_from_tail_ptr + row)
            req = tl.load(req_pool_idx_ptr + row)
            chunk_src = tl.load(chunk_src_start_ptr + row)
            tail_base = tl.load(tail_logical_base_ptr + row)

            # Dimension-wise online softmax keeps the reduction numerically
            # stable without loading the complete pool into local memory.
            running_max = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)

            for slot in tl.static_range(0, POOL_SIZE):
                # A pool consists of a tail prefix followed by current chunk
                # rows. Tail positions wrap around the request ring buffer.
                if slot < n_tail:
                    tail_slot = (tail_base + slot) % TAIL_SIZE
                    k_offset = (
                        req * tail_k_stride_0 + tail_slot * tail_k_stride_1 + offs
                    )
                    score_offset = (
                        req * tail_score_stride_0
                        + tail_slot * tail_score_stride_1
                        + offs
                    )
                    key = tl.load(
                        tail_k_ptr + k_offset, mask=head_mask, other=0.0
                    ).to(tl.float32)
                    score = tl.load(
                        tail_score_ptr + score_offset,
                        mask=head_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    chunk_row = chunk_src + slot - n_tail
                    k_offset = chunk_row * chunk_k_stride_0 + offs
                    score_offset = chunk_row * chunk_score_stride_0 + offs
                    key = tl.load(
                        chunk_k_ptr + k_offset, mask=head_mask, other=0.0
                    ).to(tl.float32)
                    score = tl.load(
                        chunk_score_ptr + score_offset,
                        mask=head_mask,
                        other=0.0,
                    ).to(tl.float32)

                score += tl.load(
                    ape_ptr + slot * ape_stride_0 + offs,
                    mask=head_mask,
                    other=0.0,
                ).to(tl.float32)
                new_max = tl.maximum(running_max, score)
                rescale = tl.exp(running_max - new_max)
                prob = tl.exp(score - new_max)
                denom = denom * rescale + prob
                acc = acc * rescale + key * prob
                running_max = new_max

            # NPU index cache stores raw BF16 keys: no Hadamard, FP8, or scale.
            pooled = (acc / denom).to(tl.bfloat16)

            write_loc = tl.load(loc_ptr + row)
            out_offsets = write_loc * cache_stride_0 + offs
            tl.store(cache_ptr + out_offsets, pooled, mask=head_mask)

        row += program_count


def kpool_assemble_softmax_write_cache_npu(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
) -> None:
    """Assemble pooled BF16 keys and write them to the NPU index cache.

    Each plan row assembles one pool. Its first ``n_from_tail[row]`` slots are
    read from the request-local tail ring; the remaining slots are read from
    ``chunk_k`` starting at ``chunk_src_start[row]``. A dimension-wise softmax
    over ``score + ape`` produces one BF16 key, which is written in place to
    flattened cache row ``loc[row]``. Dynamic inputs produced by the current
    forward are materialized as contiguous tensors before the kernel launch.

    Args:
        index_k_cache: Contiguous BF16 index cache. Its storage must be
            flattenable to ``[cache_rows, 128]``.
        chunk_k: BF16 current-chunk keys with shape ``[N, 128]``.
        chunk_score: Scores for ``chunk_k`` with shape ``[N, 128]`` and dtype
            FP16, BF16, or FP32.
        tail_k: Contiguous BF16 request tail rings with shape ``[R, T, 128]``.
        tail_score: Scores for ``tail_k`` with shape ``[R, T, 128]`` and dtype
            FP16, BF16, or FP32.
        req_pool_idx: Int64 tensor of shape ``[n_pools]`` selecting the request
            row in ``tail_k`` for each pool.
        n_from_tail: Int32 or int64 tensor of shape ``[n_pools]`` containing
            the number of leading slots read from the tail ring.
        chunk_src_start: Int64 tensor of shape ``[n_pools]`` containing the
            first row in ``chunk_k`` for each pool's chunk suffix.
        tail_logical_base: Int32 or int64 tensor of shape ``[n_pools]`` giving
            the logical first tail slot; physical slots wrap by ``T``.
        ape: FP32 additive pooling position scores with shape
            ``[pool_size, 128]``.
        loc: Int64 tensor of shape ``[n_pools]`` containing flattened cache
            rows to update.
        write_mask: Optional bool tensor of shape ``[n_pools]``. False entries
            leave their corresponding cache rows unchanged.
        num_programs: Optional Triton grid size. By default it is capped at the
            NPU Vector Core count; each program handles additional pools with
            a grid-stride loop.

    Raises:
        AssertionError: If an input violates the kernel layout contract.
        ValueError: If ``num_programs`` is not positive.
    """

    # These assertions protect the dtype and layout assumptions used by the
    # kernel's pointer arithmetic. Plan value ranges are guaranteed upstream.
    assert index_k_cache.dtype == torch.bfloat16
    assert chunk_k.dtype == torch.bfloat16 and tail_k.dtype == torch.bfloat16
    assert chunk_score.dtype in KPOOL_SCORE_DTYPES
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert ape.dtype == torch.float32
    assert chunk_k.ndim == 2 and chunk_k.shape[1] == INDEX_HEAD_DIM
    assert chunk_score.shape == chunk_k.shape
    assert tail_k.ndim == 3 and tail_k.shape[2] == INDEX_HEAD_DIM
    assert tail_score.shape == tail_k.shape
    assert ape.ndim == 2 and ape.shape[1] == INDEX_HEAD_DIM
    assert index_k_cache.numel() % INDEX_HEAD_DIM == 0

    n_pools = req_pool_idx.numel()
    pool_size = int(ape.shape[0])
    tail_size = int(tail_k.shape[1])
    assert pool_size > 0 and tail_size > 0

    plan_tensors = (
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        loc,
    )
    assert all(t.ndim == 1 and t.numel() == n_pools for t in plan_tensors)
    assert req_pool_idx.dtype == torch.int64
    assert chunk_src_start.dtype == torch.int64 and loc.dtype == torch.int64
    assert n_from_tail.dtype in (torch.int32, torch.int64)
    assert tail_logical_base.dtype in (torch.int32, torch.int64)
    if write_mask is not None:
        assert write_mask.dtype == torch.bool and write_mask.shape == (n_pools,)

    # Match the GPU wrapper contract for dynamic, read-only forward inputs.
    # Calling contiguous() is a no-op when the input already has this layout.
    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is not None:
        write_mask = write_mask.contiguous()

    inputs = (
        index_k_cache,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
    )
    if write_mask is not None:
        inputs += (write_mask,)
    assert all(t.device == index_k_cache.device for t in inputs)
    assert all(t.is_contiguous() for t in inputs)

    if n_pools == 0:
        return

    if num_programs is None:
        num_programs = _default_triton_num_programs(chunk_k.device, n_pools)
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")
    num_programs = min(num_programs, n_pools)

    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        has_write_mask = True

    cache_2d = index_k_cache.view(-1, INDEX_HEAD_DIM)
    _kpool_assemble_softmax_write_cache_npu_kernel[(num_programs,)](
        cache_2d,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        n_pools,
        cache_2d.stride(0),
        chunk_k.stride(0),
        chunk_score.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        ape.stride(0),
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_size,
        HEAD_DIM=INDEX_HEAD_DIM,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


@triton.jit
def _kpool_write_tail_and_maybe_compress_npu_kernel(
    key_ptr,  # Draft keys: [BATCH_SIZE * N, HEAD_DIM].
    score_ptr,  # Scores paired with key_ptr.
    tail_k_ptr,  # Per-request BF16 ring buffer: [REQS, TAIL_SIZE, HEAD_DIM].
    tail_score_ptr,  # Scores paired with tail_k_ptr.
    ape_ptr,  # Additive pooling position scores: [POOL_SIZE, HEAD_DIM].
    req_pool_indices_ptr,  # Request/ring-buffer row for each batch item.
    write_start_ptr,  # Logical tail position of the first draft token.
    tail_logical_start_ptr,  # Logical start of the pool to compress.
    write_loc_ptr,  # Cache rows: [BATCH_SIZE, MAX_CLOSED_POOLS].
    out_cache_loc_ptr,  # Token cache locations; zero marks a padded batch.
    effective_n_ptr,  # Optional valid draft-token count per batch item.
    cache_ptr,  # Flattened BF16 index cache: [CACHE_ROWS, HEAD_DIM].
    batch_size,
    key_stride_0,
    score_stride_0,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    ape_stride_0,
    write_loc_stride_0,
    cache_stride_0,
    N: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_CLOSED_POOLS: tl.constexpr,
    HAS_EFFECTIVE_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    program_count = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    # Programs consume multiple batch rows when the logical batch is larger
    # than the physical Vector Core count.
    while batch < batch_size:
        cache_loc_0 = tl.load(out_cache_loc_ptr + batch * N)
        if cache_loc_0 != 0:
            req = tl.load(req_pool_indices_ptr + batch)
            write_start = tl.load(write_start_ptr + batch)

            # Tail writes always cover all N graph rows. effective_n only gates
            # compression, matching the CUDA target-verify kernel contract.
            for draft_offset in tl.static_range(0, N):
                draft_row = batch * N + draft_offset
                draft_key = tl.load(
                    key_ptr + draft_row * key_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                )
                draft_score = tl.load(
                    score_ptr + draft_row * score_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                )
                draft_tail_slot = (write_start + draft_offset) % TAIL_SIZE
                draft_tail_k_offset = (
                    req * tail_k_stride_0
                    + draft_tail_slot * tail_k_stride_1
                    + offs
                )
                draft_tail_score_offset = (
                    req * tail_score_stride_0
                    + draft_tail_slot * tail_score_stride_1
                    + offs
                )
                tl.store(
                    tail_k_ptr + draft_tail_k_offset,
                    draft_key,
                    mask=dim_mask,
                )
                tl.store(
                    tail_score_ptr + draft_tail_score_offset,
                    draft_score,
                    mask=dim_mask,
                )

            if HAS_EFFECTIVE_N:
                effective_n = tl.load(effective_n_ptr + batch).to(tl.int32)
            else:
                effective_n = N
            effective_n = tl.minimum(tl.maximum(effective_n, 0), N)

            base_pool = write_start // POOL_SIZE
            completed_pools = (
                (write_start + effective_n) // POOL_SIZE - base_pool
            )
            completed_pools = tl.minimum(
                tl.maximum(completed_pools, 0), MAX_CLOSED_POOLS
            )
            tail_logical_start = tl.load(tail_logical_start_ptr + batch)

            # write_loc[b, p] maps the p-th candidate pool to its cache row.
            # Only the runtime-completed prefix is reduced and written.
            for pool_offset in tl.static_range(0, MAX_CLOSED_POOLS):
                if pool_offset < completed_pools:
                    pool_logical_start = (
                        tail_logical_start + pool_offset * POOL_SIZE
                    )
                    running_max = tl.full(
                        (BLOCK_D,), -float("inf"), tl.float32
                    )
                    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
                    denom = tl.full((BLOCK_D,), 0.0, tl.float32)

                    for slot in tl.static_range(0, POOL_SIZE):
                        pool_tail_slot = (
                            pool_logical_start + slot
                        ) % TAIL_SIZE
                        pool_tail_k_offset = (
                            req * tail_k_stride_0
                            + pool_tail_slot * tail_k_stride_1
                            + offs
                        )
                        pool_tail_score_offset = (
                            req * tail_score_stride_0
                            + pool_tail_slot * tail_score_stride_1
                            + offs
                        )
                        pool_key = tl.load(
                            tail_k_ptr + pool_tail_k_offset,
                            mask=dim_mask,
                            other=0.0,
                        ).to(tl.float32)
                        pool_score = tl.load(
                            tail_score_ptr + pool_tail_score_offset,
                            mask=dim_mask,
                            other=0.0,
                        ).to(tl.float32)
                        pool_score += tl.load(
                            ape_ptr + slot * ape_stride_0 + offs,
                            mask=dim_mask,
                            other=0.0,
                        ).to(tl.float32)

                        new_max = tl.maximum(running_max, pool_score)
                        rescale = tl.exp(running_max - new_max)
                        prob = tl.exp(pool_score - new_max)
                        denom = denom * rescale + prob
                        acc = acc * rescale + pool_key * prob
                        running_max = new_max

                    pooled = (acc / denom).to(tl.bfloat16)
                    write_loc = tl.load(
                        write_loc_ptr
                        + batch * write_loc_stride_0
                        + pool_offset
                    )
                    cache_offsets = write_loc * cache_stride_0 + offs
                    tl.store(
                        cache_ptr + cache_offsets, pooled, mask=dim_mask
                    )

        batch += program_count


def kpool_write_tail_and_maybe_compress_npu(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
) -> None:
    """Write draft tokens to tail rings and compress newly completed pools.

    The input rows are grouped as ``[batch, num_draft_tokens, 128]``. For each
    valid batch item, all draft rows are written to its request-local tail ring.
    Every pool boundary crossed by ``write_start + effective_n`` is reduced
    with a dimension-wise softmax over ``tail_score + ape`` and written to the
    corresponding BF16 index-cache row in ``write_loc``.

    Args:
        index_k_cache: Contiguous BF16 cache flattenable to
            ``[cache_rows, 128]``. Updated in place.
        key: BF16 draft keys with shape ``[batch * N, 128]``.
        score: Draft scores with the same shape as ``key`` and dtype FP16,
            BF16, or FP32.
        tail_k: Contiguous BF16 tail rings with shape
            ``[R, pool_size + N, 128]``. Updated in place for valid batches.
        tail_score: Contiguous score tail rings matching ``tail_k``. Updated in
            place for valid batch items.
        ape: FP32 additive pooling position scores with shape
            ``[pool_size, 128]``.
        req_pool_indices: Request row in the tail buffers for each batch item.
        write_start: Logical tail position where each batch starts writing.
        tail_logical_start: Logical start of the candidate completed pool.
        write_loc: Candidate BF16 cache rows with shape
            ``[batch, ceil(num_draft_tokens / pool_size)]``.
        out_cache_loc: Token cache locations with at least ``batch * N``
            entries. A zero first entry skips the entire batch item.
        num_draft_tokens: Static number ``N`` of draft rows per batch item.
        effective_n_per_batch: Optional effective draft count used only to
            decide whether a pool was completed. Tail writes still cover ``N``.
        num_programs: Optional Triton grid size. Defaults to the NPU Vector Core
            count and uses a grid-stride loop for larger batches.

    Raises:
        AssertionError: If an input violates the kernel layout contract.
        ValueError: If ``num_programs`` is not positive.
    """

    assert num_draft_tokens > 0
    assert key.ndim == 2 and key.shape[1] == INDEX_HEAD_DIM
    assert score.shape == key.shape
    assert key.dtype == torch.bfloat16
    assert score.dtype in KPOOL_SCORE_DTYPES
    assert tail_k.ndim == 3 and tail_k.shape[2] == INDEX_HEAD_DIM
    assert tail_score.shape == tail_k.shape
    assert tail_k.dtype == torch.bfloat16
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert ape.ndim == 2 and ape.shape[1] == INDEX_HEAD_DIM
    assert ape.dtype == torch.float32
    assert index_k_cache.dtype == torch.bfloat16
    assert index_k_cache.numel() % INDEX_HEAD_DIM == 0

    total_rows = key.shape[0]
    if total_rows == 0:
        return
    assert total_rows % num_draft_tokens == 0
    batch_size = total_rows // num_draft_tokens
    pool_size = int(ape.shape[0])
    tail_size = int(tail_k.shape[1])
    assert pool_size > 0
    max_closed_pools = (num_draft_tokens + pool_size - 1) // pool_size
    assert tail_size == pool_size + num_draft_tokens

    batch_metadata = (
        req_pool_indices,
        write_start,
        tail_logical_start,
    )
    assert all(t.ndim == 1 and t.numel() >= batch_size for t in batch_metadata)
    assert write_loc.shape == (batch_size, max_closed_pools), write_loc.shape
    assert write_loc.stride(1) == 1, write_loc.stride()
    assert out_cache_loc.ndim == 1 and out_cache_loc.numel() >= total_rows
    assert req_pool_indices.dtype == torch.int64
    assert write_start.dtype in (torch.int32, torch.int64)
    assert tail_logical_start.dtype in (torch.int32, torch.int64)
    assert write_loc.dtype == torch.int64
    assert out_cache_loc.dtype in (torch.int32, torch.int64)
    if effective_n_per_batch is not None:
        assert effective_n_per_batch.ndim == 1
        assert effective_n_per_batch.numel() >= batch_size
        assert effective_n_per_batch.dtype in (torch.int32, torch.int64)

    # These forward and plan tensors are read-only and may be views.
    key = key.contiguous()
    score = score.contiguous()
    ape = ape.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    write_start = write_start.contiguous()
    tail_logical_start = tail_logical_start.contiguous()
    write_loc = write_loc.contiguous()
    out_cache_loc = out_cache_loc.contiguous()
    if effective_n_per_batch is not None:
        effective_n_per_batch = effective_n_per_batch.contiguous()

    inputs = (
        index_k_cache,
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
    )
    if effective_n_per_batch is not None:
        inputs += (effective_n_per_batch,)
    assert all(t.device == index_k_cache.device for t in inputs)
    assert all(t.is_contiguous() for t in inputs)

    if num_programs is None:
        num_programs = _default_triton_num_programs(key.device, batch_size)
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")
    num_programs = min(num_programs, batch_size)

    if effective_n_per_batch is None:
        effective_n_per_batch = torch.empty(
            (1,), dtype=torch.int32, device=key.device
        )
        has_effective_n = False
    else:
        has_effective_n = True

    cache_2d = index_k_cache.view(-1, INDEX_HEAD_DIM)
    _kpool_write_tail_and_maybe_compress_npu_kernel[(num_programs,)](
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        effective_n_per_batch,
        cache_2d,
        batch_size,
        key.stride(0),
        score.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        ape.stride(0),
        write_loc.stride(0),
        cache_2d.stride(0),
        N=num_draft_tokens,
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_size,
        HEAD_DIM=INDEX_HEAD_DIM,
        MAX_CLOSED_POOLS=max_closed_pools,
        HAS_EFFECTIVE_N=has_effective_n,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


@triton.jit
def _scatter_kpool_tail_updates_npu_kernel(
    # --- chunk 张量指针 ---
    chunk_k_ptr,
    chunk_score_ptr,
    # --- tail 张量指针 (原地更新) ---
    tail_k_ptr,
    tail_score_ptr,
    # --- 元数据指针 ---
    req_pool_idx_ptr,
    dst_logical_start_ptr,
    chunk_src_start_ptr,
    n_write_ptr,
    # --- stride 参数 ---
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    # --- shape 参数 ---
    n_rows,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """NPU 亲和 scatter 写入 kernel。

    Grid: (NUM_CORES,) — 固定核数, 每个核跨步处理一组 row, 内层按 n_write[row] 迭代 slot。
    每个工作项: 从 chunk 加载一行 K/Score (128×bf16), 写入 tail ring buffer 对应位置。
    """
    pid = tl.program_id(0)

    # ===== 每个核跨步处理一组 row =====
    for row in range(pid, n_rows, NUM_CORES):

        # ===== 加载该 row 的元数据 (仅一次) =====
        n_w = tl.load(n_write_ptr + row).to(tl.int32)
        req = tl.load(req_pool_idx_ptr + row).to(tl.int64)
        dst_start = tl.load(dst_logical_start_ptr + row).to(tl.int32)
        src_base = tl.load(chunk_src_start_ptr + row).to(tl.int64)

        # ===== 遍历该 row 需要写入的 slot =====
        for slot in range(n_w):

            # ===== 计算 chunk 源行 & tail 物理 slot (环形取模) =====
            src_row = src_base + slot.to(tl.int64)
            phys_slot = (dst_start + slot) % TAIL_SIZE

            # ===== 计算 chunk 中偏移 =====
            src_offset = src_row * chunk_stride_0.to(tl.int64)

            # ===== Block Pointer 加载 chunk_k[src_row, :] =====
            k_block = tl.make_block_ptr(
                base=chunk_k_ptr + src_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            k = tl.load(k_block)  # [HEAD_DIM] bf16 整行 DMA

            # ===== Block Pointer 加载 chunk_score[src_row, :] =====
            s_block = tl.make_block_ptr(
                base=chunk_score_ptr + src_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            s = tl.load(s_block)  # [HEAD_DIM] bf16 整行 DMA

            # ===== 计算 tail 目标偏移 =====
            dst_offset = (
                req * tail_stride_0.to(tl.int64)
                + phys_slot.to(tl.int64) * tail_stride_1.to(tl.int64)
            )

            # ===== Block Pointer 存储 tail_k[req, phys_slot, :] =====
            tail_k_block = tl.make_block_ptr(
                base=tail_k_ptr + dst_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            tl.store(tail_k_block, k)

            # ===== Block Pointer 存储 tail_score[req, phys_slot, :] =====
            tail_score_block = tl.make_block_ptr(
                base=tail_score_ptr + dst_offset,
                shape=(HEAD_DIM,),
                strides=(1,),
                offsets=(0,),
                block_shape=(HEAD_DIM,),
                order=(0,),
            )
            tl.store(tail_score_block, s)


def scatter_kpool_tail_updates_npu(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    """NPU 亲和版 scatter_kpool_tail_updates — drop-in 替换 GPU 原始实现。

    与 ``kpool_fp8_index.scatter_kpool_tail_updates`` 签名完全一致
    """
    pool_size = pool.index_kpool
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()

    num_programs = _default_triton_num_programs(chunk_k.device, n_rows)

    _scatter_kpool_tail_updates_npu_kernel[(num_programs,)](
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        n_rows,
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        NUM_CORES=num_programs,
    )


@triton.jit
def _update_kpool_write_plan_npu_kernel(
    write_start_ptr,
    req_pool_indices_ptr,
    real_page_table_ptr,
    req_out_ptr,
    write_start_out_ptr,
    tail_logical_start_out_ptr,
    write_loc_out_ptr,
    pool_seqlens_per_q_out_ptr,
    seqlens_per_q_out_ptr,
    real_page_table_stride_0,
    real_page_table_cols,
    write_loc_out_stride_0,
    bs,
    POOL_SIZE: tl.constexpr,
    N: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    MAX_CLOSED_POOLS: tl.constexpr,
    HAS_PER_Q: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """NPU 亲和 kpool write-plan kernel.

    Grid: (NUM_CORES,) — 固定核数, 每个核跨步处理一组 batch 元素。
    Each work item computes base_pool using integer division and remainder,
    gathers real_page_table for every candidate closed pool, writes req,
    write_start, tail_logical_start, write_loc, and optional per-query seqlens,
    and fuses more than seven small PyTorch operators from
    update_kpool_write_plan_npu.
    """
    pid = tl.program_id(0)

    for b in range(pid, bs, NUM_CORES):
        ws = tl.load(write_start_ptr + b).to(tl.int32)
        req = tl.load(req_pool_indices_ptr + b)
        base_pool = ws // POOL_SIZE

        if HAS_PER_Q:
            for k in tl.static_range(0, N):
                row = b * N + k
                seqlen_per_q = ws + k + 1
                tl.store(seqlens_per_q_out_ptr + row, seqlen_per_q)
                tl.store(pool_seqlens_per_q_out_ptr + row, seqlen_per_q // POOL_SIZE)

        tl.store(req_out_ptr + b, req)
        tl.store(write_start_out_ptr + b, ws)
        tl.store(tail_logical_start_out_ptr + b, (base_pool * POOL_SIZE).to(tl.int32))
        for pool_offset in tl.static_range(0, MAX_CLOSED_POOLS):
            pool_id = base_pool + pool_offset
            pool_page_group = pool_id // SLOTS_PER_PAGE
            token_page_row = pool_page_group * POOL_SIZE
            token_page_row = tl.minimum(
                tl.maximum(token_page_row, 0), real_page_table_cols - 1
            )
            packed_page = tl.load(
                real_page_table_ptr
                + b * real_page_table_stride_0
                + token_page_row
            ).to(tl.int64)
            write_loc = (
                packed_page * SLOTS_PER_PAGE
                + (pool_id % SLOTS_PER_PAGE)
            )
            tl.store(
                write_loc_out_ptr
                + b * write_loc_out_stride_0
                + pool_offset,
                write_loc.to(tl.int64),
            )


def update_kpool_write_plan_triton_npu(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out: torch.Tensor,
    write_start_out: torch.Tensor,
    tail_logical_start_out: torch.Tensor,
    write_loc_out: torch.Tensor,
    pool_seqlens_per_q_out: Optional[torch.Tensor],
    seqlens_per_q_out: Optional[torch.Tensor],
    *,
    pool_size: int,
    num_draft_tokens: int,
    slots_per_page: int,
) -> None:
    """NPU Triton variant of update_kpool_write_plan_cuda_graph.

    Fuses the 7+ small PyTorch ops in update_kpool_write_plan_npu (div / arange /
    gather / clamp / remainder / elementwise) into a single Triton kernel, one
    program per batch element. Grid is sized by _default_triton_num_programs to
    match the Ascend Vector Core count when bs exceeds it.
    """
    bs = write_start.shape[0]
    if bs == 0 or num_draft_tokens == 0:
        return
    assert pool_size > 0
    max_closed_pools = (num_draft_tokens + pool_size - 1) // pool_size
    assert write_loc_out.shape == (bs, max_closed_pools), write_loc_out.shape
    assert write_loc_out.stride(1) == 1, write_loc_out.stride()

    has_per_q_outputs = pool_seqlens_per_q_out is not None
    assert has_per_q_outputs == (
        seqlens_per_q_out is not None
    ), "pool_seqlens_per_q_out and seqlens_per_q_out must be both set or both None"
    per_q_dummy = (
        pool_seqlens_per_q_out
        if has_per_q_outputs
        else torch.empty(1, dtype=torch.int32, device=write_start.device)
    )

    num_programs = _default_triton_num_programs(write_start.device, bs)

    _update_kpool_write_plan_npu_kernel[(num_programs,)](
        write_start,
        req_pool_indices,
        real_page_table,
        req_out,
        write_start_out,
        tail_logical_start_out,
        write_loc_out,
        pool_seqlens_per_q_out if has_per_q_outputs else per_q_dummy,
        seqlens_per_q_out if has_per_q_outputs else per_q_dummy,
        real_page_table.stride(0),
        real_page_table.shape[1],
        write_loc_out.stride(0),
        bs,
        POOL_SIZE=pool_size,
        N=num_draft_tokens,
        SLOTS_PER_PAGE=slots_per_page,
        MAX_CLOSED_POOLS=max_closed_pools,
        HAS_PER_Q=has_per_q_outputs,
        NUM_CORES=num_programs,
    )
