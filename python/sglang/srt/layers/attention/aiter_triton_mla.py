"""gfx1250 MLA decode through aiter's Triton ``mla_decode_fwd``.

``aiter.ops.triton.attention.mla.mla_decode_fwd`` reads the MLA KV pool through
block tables with a bf16 query; aiter picks the kernel behind it. Needs aiter
with ROCm/aiter#5935. There is no other aiter kernel for a bf16 query on
gfx1250, so no env turns it off; ``--decode-attention-backend triton`` leaves
aiter.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.utils import is_gfx1250_supported


def supports_aiter_triton_mla_decode(
    *,
    input_dtype: torch.dtype,
    dcp_world_size: int,
    qk_rope_head_dim: int,
) -> bool:
    """Whether single-token MLA decode runs ``aiter_triton_mla_decode``."""
    return (
        is_gfx1250_supported()
        and input_dtype == torch.bfloat16
        # BailingMoeV3 with use_mla_nope sets 0, a shape the kernel cannot take.
        and qk_rope_head_dim > 0
        and dcp_world_size == 1
    )


def triton_mla_block_size(page_size: int) -> int:
    """KV tokens per block the kernel reads; each page holds page_size // block.

    aiter's gfx1250 decode kernel, Gluon at the pinned aiter, needs a power-of-2
    block size (its shared layouts assert one), and 256 overflows the 320 KB LDS
    at 128 heads with bf16 KV.
    Block sizes 1-128 are verified at 8, 16 and 128 heads with bf16 and fp8 KV.
    """
    return min(page_size & -page_size, 128)


def triton_mla_block_tables(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    max_kv_len: int,
    page_size: int,
) -> torch.Tensor:
    """[bs, cdiv(max_kv_len, block)] kernel block ids of each request.

    The paged allocator keeps a page's slots contiguous from a multiple of
    page_size, so each block-aligned slot divided by the block size is a block id.
    """
    block = triton_mla_block_size(page_size)
    return req_to_token[req_pool_indices, :max_kv_len:block] // block


def aiter_triton_mla_decode(
    *,
    q: torch.Tensor,
    o: torch.Tensor,
    kv_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sm_scale: float,
    kv_descale: Optional[torch.Tensor],
) -> None:
    """Decode one query token per sequence into ``o`` [bs, H, kv_lora_rank].

    q is [bs, H, kv_lora_rank + qk_rope_head_dim] and kv_buffer the layer's MLA
    key buffer [num_slots, 1, kv_lora_rank + qk_rope_head_dim]. The kernel reads
    only the first cdiv(seqused_k, block) entries of each block_tables row, as
    built by ``triton_mla_block_tables`` for the same page_size.
    """
    from aiter.ops.triton.attention.mla import mla_decode_fwd

    block = triton_mla_block_size(page_size)
    mla_decode_fwd(
        q=q,
        kv_buffer=kv_buffer.view(-1, block, kv_buffer.shape[-2], kv_buffer.shape[-1]),
        out=o,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        # 0: the kernel sizes its KV split from a fixed guess, which keeps
        # CUDA graph replay valid for any sequence length.
        max_seqlen_kv=0,
        block_tables=block_tables,
        softmax_scale=sm_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        causal=True,
        q_descale=None,
        kv_descale=kv_descale,
    )
