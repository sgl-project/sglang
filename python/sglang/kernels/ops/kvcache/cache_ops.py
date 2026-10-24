import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_stride,
    value_block_stride,
    cache_token_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    value_cache_token_stride: tl.constexpr,
    value_cache_head_stride: tl.constexpr,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
    USE_SCALE: tl.constexpr,
    reserved_skip_index=-1,
    size_limit=0,
    HAS_RESERVED_SKIP: tl.constexpr = False,
    HAS_SIZE_LIMIT: tl.constexpr = False,
    slot_mapping_stride=1,
):
    """
    Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.

    Source layout:
        key/value: [num_tokens, num_heads, head_size]

    Target cache layout:
        cache: [num_blocks, block_size, num_heads, head_size]
        Token and head strides may describe a view of physical HND storage.

    Each Triton program instance handles:
        - one token (program_id(0))
        - one block of heads (program_id(1))

    Features:
        - optional SWA slot remapping
        - optional FP8 scale dequantization before cache write
        - optional reserved-slot skip and slot upper bound

    Args:
        key_ptr: Pointer to source key tensor.
        value_ptr: Pointer to source value tensor.
        key_cache_ptr: Pointer to destination key cache tensor.
        value_cache_ptr: Pointer to destination value cache tensor.
        slot_mapping_ptr: Maps token -> cache slot.
        swa_slot_mapping_ptr: Optional second-stage slot remap for SWA mode.
        k_scale_ptr: Optional key scaling factor pointer.
        v_scale_ptr: Optional value scaling factor pointer.
        block_stride: Stride between key cache blocks.
        value_block_stride: Stride between value cache blocks.
        cache_token_stride: Stride between key cache tokens.
        cache_head_stride: Stride between key cache heads.
        value_cache_token_stride: Stride between value cache tokens.
        value_cache_head_stride: Stride between value cache heads.
        key_stride: Stride between source key tokens.
        value_stride: Stride between source value tokens.
        num_heads: Number of attention heads.
        head_size: Hidden dimension per head.
        block_size: Number of slots per cache block.
        HEAD_BLOCK: Number of heads processed per program.
        BLOCK_D: Vectorized dimension size (power-of-2 padded).
        HAS_SWA: Enable SWA remapping.
        USE_SCALE: Enable scale division before storing.
        reserved_skip_index: Slot whose writes are skipped; read only when
            HAS_RESERVED_SKIP.
        size_limit: Writes to slots >= size_limit are skipped; read only when
            HAS_SIZE_LIMIT. Checked after the SWA remap, so it does not bound the
            swa_slot_mapping lookup.
        HAS_RESERVED_SKIP: Enable the reserved_skip_index skip.
        HAS_SIZE_LIMIT: Enable the size_limit bound.
        slot_mapping_stride: Element stride between token slot mappings.
    """

    # ----------------------------------
    # program ids
    # pid0 = token
    # pid1 = head block
    # ----------------------------------
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    # ----------------------------------
    # slot mapping
    # ----------------------------------
    slot_idx = tl.load(slot_mapping_ptr + token_idx * slot_mapping_stride)

    if HAS_SWA:
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)

    # Keep destination offsets in int64: a valid int32 slot can overflow when
    # multiplied by the cache row stride.
    slot_idx = slot_idx.to(tl.int64)
    skip = slot_idx < 0
    if HAS_RESERVED_SKIP:
        skip = skip | (slot_idx == reserved_skip_index)
    if HAS_SIZE_LIMIT:
        skip = skip | (slot_idx >= size_limit)
    if skip:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # ----------------------------------
    # head range
    # ----------------------------------
    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)

    head_mask = head_idx < num_heads

    dim_idx = tl.arange(0, BLOCK_D)

    # shape = [HEAD_BLOCK, BLOCK_D]
    offs = head_idx[:, None] * head_size + dim_idx[None, :]

    mask = head_mask[:, None] & (dim_idx[None, :] < head_size)

    # ----------------------------------
    # source load
    # ----------------------------------
    src_key = token_idx * key_stride + offs
    src_value = token_idx * value_stride + offs

    k = tl.load(key_ptr + src_key, mask=mask)
    v = tl.load(value_ptr + src_value, mask=mask)

    # ----------------------------------
    # optional scale
    # ----------------------------------
    if USE_SCALE:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

        k = k / k_scale
        v = v / v_scale

    # ----------------------------------
    # target layout
    # [block_idx, block_offset, head, dim]
    # ----------------------------------
    tgt_k = (
        block_idx * block_stride
        + block_offset * cache_token_stride
        + head_idx[:, None] * cache_head_stride
        + dim_idx[None, :]
    )
    tgt_v = (
        block_idx * value_block_stride
        + block_offset * value_cache_token_stride
        + head_idx[:, None] * value_cache_head_stride
        + dim_idx[None, :]
    )
    tl.store(key_cache_ptr + tgt_k, k, mask=mask)
    tl.store(value_cache_ptr + tgt_v, v, mask=mask)


def launch_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
    reserved_skip_index=-1,
    size_limit=None,
):
    """
    Launch wrapper for reshape_and_cache_flash Triton kernel.

    This wrapper prepares launch configuration and dispatches the Triton kernel
    that writes token-major K/V tensors into paged KV cache layout.

    Args:
        key: Source key tensor [num_tokens, num_heads, head_size]
        value: Source value tensor [num_tokens, num_heads, head_size]
        key_cache: Destination key cache [num_blocks, block_size, num_heads, head_size]
        value_cache: Destination value cache [num_blocks, block_size, num_heads, head_size]
        slot_mapping: Token-to-cache slot mapping
        swa_slot_mapping: Optional SWA remapping table
        k_scale: Optional key scaling factor
        v_scale: Optional value scaling factor
        reserved_skip_index: Slot whose writes are skipped, e.g. the reserved
            padding slot 0 that store_cache skips; a negative value (default -1)
            skips none.
        size_limit: If set, writes to slots >= size_limit are skipped; must not
            exceed the caches' slot count. Not supported with swa_slot_mapping,
            whose lookup it would not bound. None (default) checks no upper bound.
    """

    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    if size_limit is not None:
        assert swa_slot_mapping is None, (
            "size_limit does not bound the swa_slot_mapping lookup"
        )
        num_slots = min(
            key_cache.shape[0] * key_cache.shape[1],
            value_cache.shape[0] * value_cache.shape[1],
        )
        assert 0 <= size_limit <= num_slots, (
            f"size_limit={size_limit} is outside the caches' {num_slots} slots"
        )

    HEAD_BLOCK = 4

    BLOCK_D = triton.next_power_of_2(head_size)

    grid = (
        num_tokens,
        triton.cdiv(num_heads, HEAD_BLOCK),
    )

    reshape_and_cache_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        k_scale if k_scale is not None else key,
        v_scale if v_scale is not None else key,
        key_cache.stride(0),
        value_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(1),
        value_cache.stride(2),
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        key_cache.shape[1],
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        HAS_SWA=(swa_slot_mapping is not None),
        USE_SCALE=(k_scale is not None),
        reserved_skip_index=reserved_skip_index,
        size_limit=size_limit if size_limit is not None else 0,
        HAS_RESERVED_SKIP=(reserved_skip_index >= 0),
        HAS_SIZE_LIMIT=(size_limit is not None),
        slot_mapping_stride=slot_mapping.stride(0),
    )
