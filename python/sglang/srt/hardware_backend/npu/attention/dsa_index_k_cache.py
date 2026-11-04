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
    """Write FP8 index K and FP32 scales into a packed NPU cache.

    Each page stores [page_size * 128 K bytes][page_size FP32 scales].
    Nonzero loc values must be unique and in bounds; slot 0 is padding.
    These value constraints are not checked here.
    """
    loc = loc.to(torch.int64)
    _, buf_numel_per_page = cache.shape
    (num_tokens,) = loc.shape
    num_key_tokens, index_head_dim = index_k.shape
    assert index_k_scale.ndim in (1, 2), "index_k_scale must be 1D or 2D"
    num_scale_tokens = index_k_scale.shape[0]
    scale_dim = index_k_scale.shape[1] if index_k_scale.ndim == 2 else 1

    assert buf_numel_per_page == page_size * (128 + 4)
    assert num_tokens == num_key_tokens == num_scale_tokens
    assert index_head_dim == 128
    assert scale_dim == 1
    assert 16 <= page_size <= 1024 and page_size % 16 == 0, (
        "page_size must be a multiple of 16 in [16, 1024]"
    )
    assert cache.dtype == torch.uint8
    assert loc.dtype == torch.int64
    assert index_k.dtype == torch.float8_e4m3fn
    assert index_k_scale.dtype == torch.float32
    assert cache.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()
    if num_tokens == 0:
        return

    # Call the existing kernel directly, bypassing the CUDA page-size wrapper.
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
