"""Store already quantized DSA index keys in a packed NPU cache.

Each page contains [page_size * 128 K bytes][page_size FP32 scales].
This module requires Triton Ascend; it does not perform quantization.
"""

import torch

from sglang.kernels.ops.kvcache.dsa_index_k_cache import (
    _set_k_and_s_triton_kernel,
)


def set_index_k_scale_buffer_npu(
    cache: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
) -> None:
    """Write K and scale in place with one Triton kernel launch.

    Args:
        cache: Contiguous uint8 [num_pages, page_size * 132] buffer.
        loc: Contiguous int32/int64 [T] physical token slots.
        index_k: Contiguous float8_e4m3fn [T, 128], already quantized by NPU.
        index_k_scale: Contiguous float32 [T] or [T, 1].
        page_size: Tokens per page; 16..1024, divisible by 16, matching
            the packed Ascend 950 quant_lightning_indexer reader.

    All tensors must be on the same NPU. Locations must be in
    [0, num_pages * page_size), with unique nonzero locations. Padding tokens
    may write the reserved slot 0; its contents are unspecified and must not
    be used as valid cached data. These value constraints are not checked here
    to avoid a device synchronization. Inputs must not alias the output cache.

    K is loaded and stored as FP8, matching the GPU writer's pointer types.
    Both cache views alias the same allocation. No packing temporary or
    quantization is performed. Unwritten slots are left unchanged.
    """
    if not isinstance(page_size, int) or not (
        16 <= page_size <= 1024 and page_size % 16 == 0
    ):
        raise ValueError("page_size must be a multiple of 16 in [16, 1024]")
    if cache.device.type != "npu":
        raise ValueError("cache must be on an NPU")
    tensors = (cache, loc, index_k, index_k_scale)
    if any(t.device != cache.device for t in tensors):
        raise ValueError("all tensors must be on the same NPU")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    if cache.dtype != torch.uint8 or cache.ndim != 2:
        raise ValueError("cache must be a rank-2 uint8 tensor")
    if cache.shape[0] == 0 or cache.shape[1] != page_size * 132:
        raise ValueError("cache shape must be [num_pages > 0, page_size * 132]")
    if cache.storage_offset() % 4:
        raise ValueError("cache storage offset must be FP32 aligned")
    if loc.ndim != 1 or loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("loc must be a rank-1 int32/int64 tensor")
    num_tokens = loc.numel()
    if index_k.dtype != torch.float8_e4m3fn or index_k.shape != (num_tokens, 128):
        raise ValueError("index_k must be float8_e4m3fn [T, 128]")
    if index_k_scale.dtype != torch.float32 or index_k_scale.shape not in (
        (num_tokens,),
        (num_tokens, 1),
    ):
        raise ValueError("index_k_scale must be float32 [T] or [T, 1]")
    if num_tokens == 0:
        return

    # Call the existing kernel directly, bypassing the CUDA page-size wrapper.
    loc = loc.to(torch.int64)
    _set_k_and_s_triton_kernel[(num_tokens,)](
        cache.view(torch.float8_e4m3fn),
        cache.view(torch.float32),
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=cache.shape[1],
        NUM_K_ELEMS_PER_TOKEN=index_k.shape[1],
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_k.shape[1],
        PRESHUFFLE_TILE=0,
    )
