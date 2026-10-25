import torch
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_shuffle_5d(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    key_stride_token,
    value_stride_token,
    num_heads,
    head_size,
    block_size,
    X: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
):
    """Scatter per-token (num_tokens, num_heads, head_size) K/V into the
    SHUFFLE 5D "vectorized" KV cache layout used by aiter CK
    `mha_batch_prefill_func` and aiter `pa_decode_gluon`.

    K cache shape: (num_blocks, num_heads, head_size // X, block_size, X)
    V cache shape: (num_blocks, num_heads, block_size // X, head_size, X)
    where X = 16 // element_size (=8 for bf16/fp16, =16 for fp8).
    block_size must be divisible by X, and head_size must be divisible by X.

    Each program handles one token and a HEAD_BLOCK-wide slice of heads.
    """
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if HAS_SWA:
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    slot_in_page = slot_idx % block_size
    page_outer = slot_in_page // X
    page_inner = slot_in_page % X

    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    head_mask = head_idx < num_heads
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_size
    d_outer = d // X
    d_inner = d % X

    src_off = token_idx * key_stride_token + head_idx[:, None] * head_size + d[None, :]
    src_mask = head_mask[:, None] & d_mask[None, :]
    k = tl.load(key_ptr + src_off, mask=src_mask)
    src_off_v = (
        token_idx * value_stride_token + head_idx[:, None] * head_size + d[None, :]
    )
    v = tl.load(value_ptr + src_off_v, mask=src_mask)

    layer_stride = num_heads * head_size * block_size
    head_stride = head_size * block_size

    k_tgt = (
        block_idx * layer_stride
        + head_idx[:, None] * head_stride
        + d_outer[None, :] * block_size * X
        + slot_in_page * X
        + d_inner[None, :]
    )
    tl.store(key_cache_ptr + k_tgt, k, mask=src_mask)

    v_tgt = (
        block_idx * layer_stride
        + head_idx[:, None] * head_stride
        + page_outer * head_size * X
        + d[None, :] * X
        + page_inner
    )
    tl.store(value_cache_ptr + v_tgt, v, mask=src_mask)


def launch_reshape_and_cache_shuffle_5d(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    swa_slot_mapping=None,
):
    """Launcher for reshape_and_cache_shuffle_5d.

    Args:
        key/value: (num_tokens, num_heads, head_size) source tensors
        key_cache: (num_blocks, num_heads, head_size//X, block_size, X)
        value_cache: (num_blocks, num_heads, block_size//X, head_size, X)
        slot_mapping: per-token destination slot in [0, num_blocks*block_size)
    """
    num_tokens, num_heads, head_size = key.shape
    assert value.shape == key.shape, "K/V must share token-major shape"
    assert key_cache.dim() == 5 and value_cache.dim() == 5
    num_blocks, kc_H, kc_D_over_X, block_size, X = key_cache.shape
    assert kc_H == num_heads and kc_D_over_X * X == head_size
    vb_blocks, vc_H, vc_page_over_X, vc_D, vc_X = value_cache.shape
    assert (
        vc_H == num_heads
        and vc_page_over_X * X == block_size
        and vc_D == head_size
        and vc_X == X
    )
    assert block_size % X == 0 and head_size % X == 0

    HEAD_BLOCK = min(4, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_size)
    grid = (num_tokens, triton.cdiv(num_heads, HEAD_BLOCK))

    reshape_and_cache_shuffle_5d[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping if swa_slot_mapping is not None else slot_mapping,
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        block_size,
        X=X,
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        HAS_SWA=(swa_slot_mapping is not None),
    )


@triton.jit
def gather_shuffle_5d_to_linear(
    key_cache_ptr,
    value_cache_ptr,
    key_out_ptr,  # (T, num_heads, head_size), store dtype
    value_out_ptr,  # (T, num_heads, head_size), store dtype
    slot_mapping_ptr,  # (T,) absolute pool slot id per token
    key_out_stride_token,
    value_out_stride_token,
    num_heads,
    head_size,
    block_size,
    X: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Inverse of :func:`reshape_and_cache_shuffle_5d`.

    Gather one token's K/V from the SHUFFLE 5D paged cache into the
    canonical (T, H, D) layout that aiter's ``mha_batch_prefill_func``
    expects in LINEAR mode. Source addressing is identical to the
    writer kernel so any bit-exact round-trip is guaranteed.
    """
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    block_idx = slot_idx // block_size
    slot_in_page = slot_idx % block_size
    page_outer = slot_in_page // X
    page_inner = slot_in_page % X

    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    head_mask = head_idx < num_heads
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_size
    d_outer = d // X
    d_inner = d % X

    layer_stride = num_heads * head_size * block_size
    head_stride = head_size * block_size

    src_mask = head_mask[:, None] & d_mask[None, :]
    k_src = (
        block_idx * layer_stride
        + head_idx[:, None] * head_stride
        + d_outer[None, :] * block_size * X
        + slot_in_page * X
        + d_inner[None, :]
    )
    k = tl.load(key_cache_ptr + k_src, mask=src_mask)
    v_src = (
        block_idx * layer_stride
        + head_idx[:, None] * head_stride
        + page_outer * head_size * X
        + d[None, :] * X
        + page_inner
    )
    v = tl.load(value_cache_ptr + v_src, mask=src_mask)

    dst_k = (
        token_idx * key_out_stride_token + head_idx[:, None] * head_size + d[None, :]
    )
    tl.store(key_out_ptr + dst_k, k, mask=src_mask)
    dst_v = (
        token_idx * value_out_stride_token + head_idx[:, None] * head_size + d[None, :]
    )
    tl.store(value_out_ptr + dst_v, v, mask=src_mask)


def launch_gather_shuffle_5d_to_linear(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Inverse of :func:`launch_reshape_and_cache_shuffle_5d`.

    Returns ``(key_out, value_out)`` each shaped
    ``(T, num_heads, head_size)`` in ``key_cache.dtype`` /
    ``value_cache.dtype``. The caller is responsible for passing the
    right per-tensor descales downstream when ``store_dtype`` is fp8.

    Args:
        key_cache:   (num_blocks, num_heads, head_size // X, block_size, X)
        value_cache: (num_blocks, num_heads, block_size // X, head_size, X)
        slot_mapping: (T,) per-token absolute slot id in
            ``[0, num_blocks * block_size)``
    """
    assert key_cache.dim() == 5 and value_cache.dim() == 5
    num_blocks, num_heads, kc_D_over_X, block_size, X = key_cache.shape
    vc_blocks, vc_H, vc_page_over_X, vc_D, vc_X = value_cache.shape
    assert vc_blocks == num_blocks and vc_H == num_heads
    assert vc_page_over_X * X == block_size and vc_X == X
    head_size = kc_D_over_X * X
    assert vc_D == head_size

    num_tokens = slot_mapping.numel()
    key_out = torch.empty(
        (num_tokens, num_heads, head_size),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    value_out = torch.empty(
        (num_tokens, num_heads, head_size),
        dtype=value_cache.dtype,
        device=value_cache.device,
    )

    HEAD_BLOCK = min(4, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_size)
    grid = (num_tokens, triton.cdiv(num_heads, HEAD_BLOCK))

    gather_shuffle_5d_to_linear[grid](
        key_cache,
        value_cache,
        key_out,
        value_out,
        slot_mapping,
        key_out.stride(0),
        value_out.stride(0),
        num_heads,
        head_size,
        block_size,
        X=X,
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
    )
    return key_out, value_out
