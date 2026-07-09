from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsv4.dequant_k_cache import DIM_NOPE, DIM_ROPE


def gather_bf16_k_cache_paged(
    k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather the DeepSeek v4 BF16 paged KV cache for a list of token IDs.

    Args:
        k_cache: (num_pages, bytes_per_page) uint8 containing 512 bf16 values
            per token: 448 nope + 64 rope.
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_NOPE + DIM_ROPE) bf16 destination.

    Returns:
        (num_tokens, 1, DIM_NOPE + DIM_ROPE) bfloat16.
    """
    assert k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    num_tokens = page_table_1_flattened.shape[0]
    k_cache_bf16 = k_cache.view(torch.bfloat16)
    bf16_per_page = k_cache_bf16.shape[-1]

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16

    _gather_bf16_k_cache_paged_kernel[(num_tokens,)](
        out,
        k_cache_bf16,
        page_table_1_flattened,
        out.stride(0),
        BF16_PER_PAGE=bf16_per_page,
        PAGE_SIZE=page_size,
        DIM=DIM_NOPE + DIM_ROPE,
        BLOCK_DIM=triton.next_power_of_2(DIM_NOPE + DIM_ROPE),
    )
    return out


@triton.jit
def _gather_bf16_k_cache_paged_kernel(
    output_ptr,
    k_cache_ptr,
    page_table_ptr,
    output_stride_0,
    BF16_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE

    offsets = tl.arange(0, BLOCK_DIM)
    mask = offsets < DIM
    in_offsets = page_idx * BF16_PER_PAGE + in_page * DIM + offsets
    out_offsets = token_id * output_stride_0 + offsets
    data = tl.load(k_cache_ptr + in_offsets, mask=mask)
    tl.store(output_ptr + out_offsets, data, mask=mask)
