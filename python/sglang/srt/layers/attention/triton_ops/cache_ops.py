import torch
import triton
import triton.language as tl


@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, head_cnt)

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)

    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


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
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
    USE_SCALE: tl.constexpr,
):
    """
    Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.

    Source layout:
        key/value: [num_tokens, num_heads, head_size]

    Target cache layout:
        cache: [num_blocks, block_size, num_heads, head_size]

    Each Triton program instance handles:
        - one token (program_id(0))
        - one block of heads (program_id(1))

    Features:
        - optional SWA slot remapping
        - optional FP8 scale dequantization before cache write

    Args:
        key_ptr: Pointer to source key tensor.
        value_ptr: Pointer to source value tensor.
        key_cache_ptr: Pointer to destination key cache tensor.
        value_cache_ptr: Pointer to destination value cache tensor.
        slot_mapping_ptr: Maps token -> cache slot.
        swa_slot_mapping_ptr: Optional second-stage slot remap for SWA mode.
        k_scale_ptr: Optional key scaling factor pointer.
        v_scale_ptr: Optional value scaling factor pointer.
        block_stride: Stride between cache blocks.
        key_stride: Stride between source key tokens.
        value_stride: Stride between source value tokens.
        num_heads: Number of attention heads.
        head_size: Hidden dimension per head.
        block_size: Number of slots per cache block.
        HEAD_BLOCK: Number of heads processed per program.
        BLOCK_D: Vectorized dimension size (power-of-2 padded).
        HAS_SWA: Enable SWA remapping.
        USE_SCALE: Enable scale division before storing.
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
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if HAS_SWA:
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)

    if slot_idx < 0:
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
    tgt = block_idx * block_stride + block_offset * num_heads * head_size + offs

    tl.store(key_cache_ptr + tgt, k, mask=mask)
    tl.store(value_cache_ptr + tgt, v, mask=mask)


def launch_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
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
    """

    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

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
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        key_cache.shape[1],
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        HAS_SWA=(swa_slot_mapping is not None),
        USE_SCALE=(k_scale is not None),
    )
