import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.pad import (
    pad_sequence_with_mask as pad_sequence_with_mask,
)
from sglang.kernels.ops.attention.pad import (
    pad_sequence_with_mask_kernel as pad_sequence_with_mask_kernel,
)
from sglang.kernels.ops.attention.pad import (
    seqlens_expand_kernel as seqlens_expand_kernel,
)
from sglang.kernels.ops.attention.pad import (
    seqlens_expand_triton as seqlens_expand_triton,
)
from sglang.kernels.ops.kvcache.cache_ops import (
    concat_and_cast_mha_k_kernel as concat_and_cast_mha_k_kernel,
)
from sglang.kernels.ops.kvcache.cache_ops import (
    concat_and_cast_mha_k_triton as concat_and_cast_mha_k_triton,
)
from sglang.kernels.ops.kvcache.cache_ops import (
    concat_and_cast_q_fp8_pad as concat_and_cast_q_fp8_pad,
)
from sglang.kernels.ops.kvcache.cache_ops import (
    concat_and_cast_q_fp8_pad_kernel as concat_and_cast_q_fp8_pad_kernel,
)
from sglang.kernels.ops.kvcache.cache_ops import (
    launch_reshape_and_cache_flash as launch_reshape_and_cache_flash,
)
from sglang.kernels.ops.kvcache.cache_ops import (
    q8kv8_topk_length_from_indices as q8kv8_topk_length_from_indices,
)
from sglang.kernels.ops.kvcache.cache_ops import (
    reshape_and_cache_flash as reshape_and_cache_flash,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashinfer_kv_indices_triton as create_flashinfer_kv_indices_triton,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashmla_kv_indices_triton as create_flashmla_kv_indices_triton,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla as get_num_kv_index_blocks_flashmla,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_page_per_block_flashmla as get_num_page_per_block_flashmla,
)
from sglang.kernels.ops.kvcache.rope_cache import (
    fused_qk_rope_reshape_and_cache as fused_qk_rope_reshape_and_cache,
)
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q


# When num_kv_heads=1, we have tensors with degenerate strides,
# For example, as below, where we have stride[-3] == stride[-2]:
# - shape: [num_pages, 1, 64, 128]
# - stride: [8192, 128, 128, 1]
# This will cause TMA desc validation fail in flashinfer (trtllm-mha backend).
#
# See: https://github.com/flashinfer-ai/flashinfer/issues/2232
def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adjust degenerate strides for a tensor, make it canonical.
    """
    sizes = tensor.size()
    strides = tensor.stride()
    ndim = tensor.dim()

    need_fix = any(
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)
    )

    if not need_fix:
        return tensor

    # canonicalize the stride
    # Example:
    # - shape: [num_pages, 1, 64, 128]
    # - stride: [8192, 128, 128, 1] (wrong!)
    # Gives new stride: [8192, 8192, 128 ,1] (correct!)
    new_strides = [0] * ndim
    new_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]

    return tensor.as_strided(sizes, new_strides)


def mla_quantize_and_rope_for_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flashinfer.rope

    """Quantize and apply RoPE for FP8 attention path.

        This function handles the FP8 quantization and RoPE application for MLA attention.
        It takes separate query/key nope and rope components, applies RoPE to the rope parts,
        quantizes all components to FP8, and merges the query components into a single tensor.

        Args:
            q_nope: Query no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            q_rope: Query RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            k_nope: Key no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            k_rope: Key RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            pos_ids: Position indices for each token
                - expected dtype: torch.int64 or torch.int32
            cos_sin_cache: Precomputed cosine/sine cache for RoPE
                - expected dtype: matches q_/k_ input dtype (torch.bfloat16)
            is_neox: Whether to use NeoX-style RoPE (interleaved) or GPT-style (half rotation)
            kv_lora_rank: Dimension of the no-position-encoding component
            qk_rope_head_dim: Dimension of the RoPE component

        Returns:
            tuple: (merged_q_out, k_nope_out, k_rope_out) quantized to FP8
                - merged_q_out: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim], dtype=torch.float8_e4m3fn
                - k_nope_out:   [seq_len, num_heads, kv_lora_rank], dtype=torch.float8_e4m3fn
                - k_rope_out:   [seq_len, num_heads, qk_rope_head_dim], dtype=torch.float8_e4m3fn
        """
    attn_dtype = torch.float8_e4m3fn
    q_len, num_heads = q_rope.shape[0], q_rope.shape[1]

    # Allocate output tensors with FP8 dtype
    # Query output will contain merged nope + rope components
    q_out = q_rope.new_empty(
        q_len,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=attn_dtype,
    )

    # Key outputs maintain original shapes but with FP8 dtype
    k_rope_out = k_rope.new_empty(k_rope.shape, dtype=attn_dtype)
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)

    # Apply RoPE and quantize all components in a single fused kernel call
    # This kernel handles:
    # 1. RoPE application to q_rope and k_rope using cos_sin_cache and positions
    # 2. Quantization of all components to FP8 format
    # 3. Output placement into pre-allocated tensors
    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=is_neox,
        quantize_dtype=attn_dtype,
        # Output tensor slicing: q_out contains [nope_part, rope_part]
        q_rope_out=q_out[..., kv_lora_rank:],  # RoPE part goes to end
        k_rope_out=k_rope_out,
        q_nope_out=q_out[..., :kv_lora_rank],  # Nope part goes to beginning
        k_nope_out=k_nope_out,
        # Quantization scales (set to 1.0 for no additional scaling)
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
        enable_pdl=is_arch_support_pdl(),
    )

    return q_out, k_nope_out, k_rope_out


def mla_quantize_without_rope_for_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize MLA components to FP8 without applying rotary embeddings."""
    attn_dtype = torch.float8_e4m3fn
    q = concat_mla_absorb_q_general(q_nope, q_rope).to(attn_dtype)
    return q, k_nope.to(attn_dtype), k_rope.to(attn_dtype)


def concat_mla_absorb_q_general(q_nope, q_rope):
    if _is_cuda and q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64:
        return concat_mla_absorb_q(q_nope, q_rope)
    else:
        return torch.cat([q_nope, q_rope], dim=-1)


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


def assert_buffer_fits(used: int, capacity: int, what: str, **context) -> None:
    """Safety guard: a preallocated cuda-graph buffer must hold the runtime write.

    The kv_indices / page_table scatter kernels bound writes only per-row, not
    against the destination buffer, so an undersized buffer silently overflows
    into the adjacent row. Fail fast on the host-known extent instead. All args
    are host ints, so this is always-on (no device sync, unlike async probes).
    """
    assert used <= capacity, f"{what}: used {used} > capacity {capacity}" + (
        f" ({', '.join(f'{k}={v}' for k, v in context.items())})" if context else ""
    )
