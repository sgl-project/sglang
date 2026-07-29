"""
Fused FP8 quantization + paged KV cache write kernel for TRTLLM MHA backend.

This kernel fuses the following operations:
1. FP8 quantization of K and V tensors (from BF16/FP16 to FP8)
2. Per-token or per-page scale computation
3. Writing quantized K/V to paged KV cache layout

Performance benefits:
- Eliminates intermediate FP8 tensors in memory
- Reduces kernel launch overhead
- Better memory bandwidth utilization
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _process_kv_tensor(
    token_id,
    head_block_id,
    page_id,
    page_offset,
    input_ptr,
    cache_ptr,
    inv_scale,
    use_provided_scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_head: tl.constexpr,
    input_stride_dim: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Process a block of heads for a single K or V tensor."""
    head_idx = head_block_id * BLOCK_HEAD
    num_heads_in_block = min(BLOCK_HEAD, num_kv_heads - head_idx)

    for dim_idx in range(0, head_dim, BLOCK_DIM):
        num_dims_in_block = min(BLOCK_DIM, head_dim - dim_idx)

        head_offsets = head_idx + tl.arange(0, BLOCK_HEAD)
        dim_offsets = dim_idx + tl.arange(0, BLOCK_DIM)

        head_mask = head_offsets < (head_idx + num_heads_in_block)
        dim_mask = dim_offsets < (dim_idx + num_dims_in_block)

        # Load from input using 3D strides
        input_offsets = (
            token_id * input_stride_token
            + head_offsets[:, None] * input_stride_head
            + dim_offsets[None, :] * input_stride_dim
        )
        mask = head_mask[:, None] & dim_mask[None, :]

        block = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

        # Quantize to FP8
        if use_provided_scale:
            block_fp8 = (block * inv_scale).to(tl.float8e4nv)
        else:
            block_fp8 = block.to(tl.float8e4nv)

        # Write to cache at [page_id, page_offset, head, dim]
        cache_offsets = (
            page_id * cache_stride_page
            + page_offset * cache_stride_offset
            + head_offsets[:, None] * cache_stride_head
            + dim_offsets[None, :] * cache_stride_dim
        )

        tl.store(cache_ptr + cache_offsets, block_fp8, mask=mask)


@triton.jit
def _fused_fp8_set_kv_buffer_kernel(
    # Input tensors (post-RoPE K and V in FP16/BF16)
    k_ptr,  # [num_tokens, num_kv_heads, head_dim]
    v_ptr,  # [num_tokens, num_kv_heads, head_dim]
    # Output KV cache buffers (FP8 paged layout)
    k_cache_ptr,  # [total_slots, num_kv_heads, head_dim]
    v_cache_ptr,  # [total_slots, num_kv_heads, head_dim]
    # Cache location indices
    cache_loc_ptr,  # [num_tokens] -> token to cache location mapping
    # Pointers to scalar inverse scales (computed on GPU in wrapper)
    inv_k_scale_ptr,  # pointer to 0-D tensor on GPU
    inv_v_scale_ptr,  # pointer to 0-D tensor on GPU
    use_provided_scale: tl.constexpr,  # whether to use provided scale
    # Tensor dimensions
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    # Strides for K input [num_tokens, num_kv_heads, head_dim]
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    # Strides for K cache [total_slots, num_kv_heads, head_dim] (logically paged)
    k_cache_stride_page: tl.constexpr,
    k_cache_stride_offset: tl.constexpr,
    k_cache_stride_head: tl.constexpr,
    k_cache_stride_dim: tl.constexpr,
    # Strides for V input [num_tokens, num_kv_heads, head_dim]
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    # Strides for V cache [total_slots, num_kv_heads, head_dim] (logically paged)
    v_cache_stride_page: tl.constexpr,
    v_cache_stride_offset: tl.constexpr,
    v_cache_stride_head: tl.constexpr,
    v_cache_stride_dim: tl.constexpr,
    # Block sizes
    BLOCK_HEAD: tl.constexpr,  # Number of heads per block
    BLOCK_DIM: tl.constexpr,  # Head dimension block size
):
    """
    Fused FP8 quantization + paged KV cache write kernel.

    Each program processes one token-head_block-kv combination, quantizing and writing
    to the appropriate page in the KV cache.

    Grid: (num_tokens, num_head_blocks, 2) where dim2: 0=K, 1=V
    """
    # Get program IDs
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    kv_idx = tl.program_id(2)  # 0 for K, 1 for V

    # Get cache location for this token
    cache_loc = tl.load(cache_loc_ptr + token_id)

    # Compute page_id and offset within page
    page_id = cache_loc // page_size
    page_offset = cache_loc % page_size

    # Select K or V based on kv_idx
    if kv_idx == 0:
        # Process K tensor
        if use_provided_scale:
            inv_scale = tl.load(inv_k_scale_ptr)
        else:
            inv_scale = 1.0
        _process_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            k_ptr,
            k_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            k_stride_token,
            k_stride_head,
            k_stride_dim,
            k_cache_stride_page,
            k_cache_stride_offset,
            k_cache_stride_head,
            k_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )
    else:
        # Process V tensor
        if use_provided_scale:
            inv_scale = tl.load(inv_v_scale_ptr)
        else:
            inv_scale = 1.0
        _process_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            v_ptr,
            v_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            v_stride_token,
            v_stride_head,
            v_stride_dim,
            v_cache_stride_page,
            v_cache_stride_offset,
            v_cache_stride_head,
            v_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )


def fused_fp8_set_kv_buffer(
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] or [num_tokens, num_kv_heads * head_dim]
    v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] or [num_tokens, num_kv_heads * head_dim]
    k_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim]
    cache_loc: torch.Tensor,  # [num_tokens], dtype=int32
    k_scale: Optional[
        float
    ] = None,  # Scalar scale (matching original set_kv_buffer signature)
    v_scale: Optional[float] = None,
    page_size: int = 16,
    use_triton: bool = True,  # Whether to use Triton kernel (set to False to force naive fallback)
) -> None:
    """
    Python wrapper for the fused FP8 quantization + paged KV cache write kernel.

    This function replicates the exact behavior of the original set_kv_buffer but with
    a fused kernel that combines FP8 quantization and cache write.

    Args:
        k: Key tensor after RoPE, can be 2D or 3D
        v: Value tensor, can be 2D or 3D
        k_cache: Paged K cache buffer in FP8
        v_cache: Paged V cache buffer in FP8
        cache_loc: Cache location for each token, shape [num_tokens]
        k_scale: Optional scalar scale for K (matching original set_kv_buffer)
        v_scale: Optional scalar scale for V (matching original set_kv_buffer)
        page_size: Number of tokens per page
        use_triton: Whether to use optimized Triton kernel
    """
    num_tokens = k.shape[0]

    # Step 1: Infer num_kv_heads and head_dim from cache shape
    if k_cache.ndim == 3:
        # 3D cache layout: [total_slots, num_kv_heads, head_dim]
        total_slots, num_kv_heads, head_dim = k_cache.shape
        assert (
            total_slots % page_size == 0
        ), f"total_slots ({total_slots}) must be divisible by page_size ({page_size})"
        num_pages = total_slots // page_size
    elif k_cache.ndim == 4:
        # 4D cache layout: [num_pages, page_size, num_kv_heads, head_dim]
        num_pages, ps, num_kv_heads, head_dim = k_cache.shape
        assert (
            ps == page_size
        ), f"page_size mismatch: cache has {ps}, expected {page_size}"
        total_slots = num_pages * page_size
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}, expected 3 or 4")

    # Step 2: Validate k, v shapes and normalize
    # Store original 3D shape for Triton path
    k_3d = None
    v_3d = None

    if k.ndim == 3:
        # Input is [num_tokens, num_kv_heads, head_dim]
        assert (
            k.shape[1] == num_kv_heads
        ), f"num_kv_heads mismatch: k.shape[1]={k.shape[1]} vs cache={num_kv_heads}"
        assert (
            k.shape[2] == head_dim
        ), f"head_dim mismatch: k.shape[2]={k.shape[2]} vs cache={head_dim}"
        assert v.shape[1] == num_kv_heads and v.shape[2] == head_dim, "v shape mismatch"

        # Keep 3D for Triton kernel
        k_3d = k
        v_3d = v
        # Create 2D view for naive fallback (will be used only if use_triton=False)
        k_2d = k.reshape(num_tokens, num_kv_heads * head_dim)
        v_2d = v.reshape(num_tokens, num_kv_heads * head_dim)
    elif k.ndim == 2:
        # Input is already [num_tokens, num_kv_heads * head_dim]
        assert (
            k.shape[1] == num_kv_heads * head_dim
        ), f"k.shape[1]={k.shape[1]} != {num_kv_heads * head_dim}"
        assert (
            v.shape[1] == num_kv_heads * head_dim
        ), f"v.shape[1]={v.shape[1]} != {num_kv_heads * head_dim}"

        # Create 3D view for Triton kernel
        k_3d = k.view(num_tokens, num_kv_heads, head_dim)
        v_3d = v.view(num_tokens, num_kv_heads, head_dim)
        # Keep 2D for naive
        k_2d = k
        v_2d = v
    else:
        raise ValueError(f"Unsupported k.ndim={k.ndim}, expected 2 or 3")

    # Step 3: Compute cache strides based on layout
    if k_cache.ndim == 3:
        # 3D cache: [total_slots, num_kv_heads, head_dim]
        stride_slot = k_cache.stride(0)
        stride_head = k_cache.stride(1)
        stride_dim = k_cache.stride(2)

        k_cache_stride_page = stride_slot * page_size
        k_cache_stride_offset = stride_slot
        k_cache_stride_head = stride_head
        k_cache_stride_dim = stride_dim

        v_stride_slot = v_cache.stride(0)
        v_stride_head = v_cache.stride(1)
        v_stride_dim = v_cache.stride(2)

        v_cache_stride_page = v_stride_slot * page_size
        v_cache_stride_offset = v_stride_slot
        v_cache_stride_head = v_stride_head
        v_cache_stride_dim = v_stride_dim
    else:
        # 4D cache: [num_pages, page_size, num_kv_heads, head_dim]
        k_cache_stride_page = k_cache.stride(0)
        k_cache_stride_offset = k_cache.stride(1)
        k_cache_stride_head = k_cache.stride(2)
        k_cache_stride_dim = k_cache.stride(3)

        v_cache_stride_page = v_cache.stride(0)
        v_cache_stride_offset = v_cache.stride(1)
        v_cache_stride_head = v_cache.stride(2)
        v_cache_stride_dim = v_cache.stride(3)

    # Decide whether to use provided scale
    use_provided_scale = k_scale is not None and v_scale is not None

    if use_triton and num_tokens > 0:
        # Use optimized Triton kernel
        # Compute input strides for 3D k, v: [num_tokens, num_kv_heads, head_dim]
        k_stride_token = k_3d.stride(0)
        k_stride_head = k_3d.stride(1)
        k_stride_dim = k_3d.stride(2)

        v_stride_token = v_3d.stride(0)
        v_stride_head = v_3d.stride(1)
        v_stride_dim = v_3d.stride(2)

        # Block sizes for tiling (tunable)
        BLOCK_HEAD = min(num_kv_heads, 8)  # Process up to 8 heads at once
        BLOCK_DIM = min(head_dim, 128)  # Process up to 128 dims at once

        # Compute number of head blocks
        num_head_blocks = (num_kv_heads + BLOCK_HEAD - 1) // BLOCK_HEAD

        # Grid: (num_tokens, num_head_blocks, 2)
        # - dim 0: tokens
        # - dim 1: head blocks
        # - dim 2: K/V (0=K, 1=V)
        grid = (num_tokens, num_head_blocks, 2)

        device = k_3d.device

        def _to_tensor_scale(scale):
            """Convert scale to 0-D CUDA tensor (accepts Python float or Tensor)."""
            if isinstance(scale, torch.Tensor):
                return scale.to(device=device, dtype=torch.float32)
            else:
                # Python float / np scalar
                return torch.tensor(float(scale), device=device, dtype=torch.float32)

        # Compute inverse scales on GPU to avoid GPU→CPU sync in CUDA graph capture.
        # Previously we used float(k_scale) which triggers synchronization and fails
        # during CUDA graph capture with cudaErrorStreamCaptureUnsupported.
        if use_provided_scale:
            k_scale_tensor = _to_tensor_scale(k_scale)
            v_scale_tensor = _to_tensor_scale(v_scale)

            # Pure GPU scalar operation, safe for CUDA graph
            inv_k_scale = (1.0 / k_scale_tensor).to(device=device, dtype=torch.float32)
            inv_v_scale = (1.0 / v_scale_tensor).to(device=device, dtype=torch.float32)

            inv_k_scale_ptr = inv_k_scale
            inv_v_scale_ptr = inv_v_scale
        else:
            # When use_provided_scale=False, kernel uses constant 1.0 for inv_scale.
            # Triton will optimize away the tl.load() calls via constant folding.
            # We pass dummy pointers (k_3d) which won't be accessed in the kernel.
            # This avoids creating new GPU tensors during CUDA graph capture.
            inv_k_scale_ptr = k_3d
            inv_v_scale_ptr = k_3d

        # Launch Triton kernel
        _fused_fp8_set_kv_buffer_kernel[grid](
            k_3d,
            v_3d,
            k_cache,
            v_cache,
            cache_loc,
            inv_k_scale_ptr,
            inv_v_scale_ptr,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            page_size,
            k_stride_token,
            k_stride_head,
            k_stride_dim,
            k_cache_stride_page,
            k_cache_stride_offset,
            k_cache_stride_head,
            k_cache_stride_dim,
            v_stride_token,
            v_stride_head,
            v_stride_dim,
            v_cache_stride_page,
            v_cache_stride_offset,
            v_cache_stride_head,
            v_cache_stride_dim,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_DIM=BLOCK_DIM,
        )
    else:
        # Fallback to naive implementation
        _naive_fp8_set_kv_buffer(
            k_2d, v_2d, k_cache, v_cache, cache_loc, k_scale, v_scale, page_size
        )


def _naive_fp8_set_kv_buffer(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    k_scale: Optional[float],
    v_scale: Optional[float],
    page_size: int,
) -> None:
    """
    Naive fallback implementation that mimics the original set_kv_buffer logic.

    This directly replicates the behavior of MHATokenToKVPool.set_kv_buffer:
    1. Apply scale (if k.dtype != cache.dtype and scale is provided)
    2. Convert to FP8
    3. Write to cache at cache_loc

    Args:
        k: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D
        v: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D
        k_cache: [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim]
        v_cache: Same shape as k_cache
        cache_loc: [num_tokens]
        k_scale: Optional scale for K
        v_scale: Optional scale for V
        page_size: Tokens per page
    """
    num_tokens = k.shape[0]

    # Infer dimensions from cache
    if k_cache.ndim == 3:
        num_kv_heads = k_cache.shape[1]
        head_dim = k_cache.shape[2]
    elif k_cache.ndim == 4:
        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}")

    # Determine target dtype and storage dtype
    # See: python/sglang/srt/mem_cache/memory_pool.py:445-449
    store_dtype = k_cache.dtype
    if store_dtype == torch.uint8:
        # Cache is stored as uint8 for FP8 (due to index_put limitation)
        dtype = torch.float8_e4m3fn  # Logical dtype
    else:
        dtype = store_dtype  # Cache dtype is the logical dtype

    # Replicate the original set_kv_buffer behavior
    # See: python/sglang/srt/mem_cache/memory_pool.py:777-799
    if k.dtype != dtype:
        # Need quantization - clone first to avoid modifying input
        k = k.clone()
        v = v.clone()

        if k_scale is not None:
            k.div_(k_scale)  # In-place division
        if v_scale is not None:
            v.div_(v_scale)  # In-place division

        k = k.to(dtype)
        v = v.to(dtype)

    # View FP8 as uint8 if needed (for index_put compatibility)
    if store_dtype == torch.uint8 and dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        k = k.view(torch.uint8)
        v = v.view(torch.uint8)

    # Reshape from [T, H*D] to [T, H, D]
    k = k.view(num_tokens, num_kv_heads, head_dim)
    v = v.view(num_tokens, num_kv_heads, head_dim)

    # Write to cache using advanced indexing (same as original)
    if k_cache.ndim == 3:
        # 3D cache: [total_slots, H, D]
        k_cache[cache_loc] = k
        v_cache[cache_loc] = v
    else:
        # 4D cache: [num_pages, page_size, H, D]
        # Decompose loc into page_id and page_offset (vectorized)
        page_ids = cache_loc // page_size
        page_offsets = cache_loc % page_size
        k_cache[page_ids, page_offsets] = k
        v_cache[page_ids, page_offsets] = v
