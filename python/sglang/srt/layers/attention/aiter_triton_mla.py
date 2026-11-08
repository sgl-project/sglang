"""gfx1250 MLA decode through aiter's Triton ``mla_decode_fwd``.

``aiter.ops.triton.attention.mla.mla_decode_fwd`` reads the MLA KV pool through
block tables with a bf16 query; aiter picks the kernel behind it, Gluon kernels
on gfx1250. It is ``SGLANG_AITER_MLA_DECODE_BACKEND=triton``, the default
(``auto``) on gfx1250 with bf16 activations and no DCP, where ``asm`` and
``gluon`` are ignored. That value is not
sglang's ``--decode-attention-backend triton``, which leaves aiter for sglang's
own Triton decode kernel. aiter without ROCm/aiter#5935 computes wrong outputs
for batches it runs with one KV segment per sequence. aiter's ASM
``mla_decode_fwd`` also has a bf16-query FlyDSL kernel on gfx1250, but only in
persistent mode at page size 1 with ``AITER_MLA_DECODE_PS1_FLYDSL`` set; this
path does not read that switch.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.utils import is_gfx1250_supported, print_info_once


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


def resolve_aiter_mla_decode_backend(
    value: str,
    *,
    use_mla: bool,
    input_dtype: torch.dtype,
    dcp_world_size: int,
    qk_rope_head_dim: int,
) -> str:
    """``SGLANG_AITER_MLA_DECODE_BACKEND`` as "asm", "gluon" or "triton".

    "auto" is "triton" where ``supports_aiter_triton_mla_decode`` holds, else
    "asm". There, "asm" and "gluon" are ignored and give "triton": neither
    kernel takes a bf16 query in this backend's KV layout on gfx1250. A value
    no kernel can serve raises here rather than at the first decode or CUDA
    graph capture.
    """
    backend = value.lower()
    if backend not in ("auto", "asm", "gluon", "triton"):
        raise ValueError(
            "SGLANG_AITER_MLA_DECODE_BACKEND must be 'auto', 'asm', 'gluon' or "
            f"'triton', got {value!r}"
        )
    gfx1250 = is_gfx1250_supported()
    if backend == "triton" and not gfx1250:
        raise ValueError(
            "SGLANG_AITER_MLA_DECODE_BACKEND=triton selects aiter's Triton "
            "mla_decode_fwd, which the aiter backend runs only on gfx1250"
        )
    if not (use_mla and gfx1250):
        return "asm" if backend == "auto" else backend
    triton = supports_aiter_triton_mla_decode(
        input_dtype=input_dtype,
        dcp_world_size=dcp_world_size,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    if triton:
        if backend in ("asm", "gluon"):
            print_info_once(
                f"aiter gfx1250: SGLANG_AITER_MLA_DECODE_BACKEND={backend} is "
                "ignored for MLA decode with bf16 activations and no DCP; it "
                "runs aiter's Triton mla_decode_fwd"
            )
        return "triton"
    if backend == "auto":
        return "asm"
    if backend == "triton":
        raise ValueError(
            "SGLANG_AITER_MLA_DECODE_BACKEND=triton needs bf16 activations, no "
            f"DCP and qk_rope_head_dim > 0; got {input_dtype}, DCP size "
            f"{dcp_world_size}, qk_rope_head_dim {qk_rope_head_dim}"
        )
    if backend == "gluon":
        raise ValueError(
            "SGLANG_AITER_MLA_DECODE_BACKEND=gluon selects aiter's mla_gluon, "
            "which asserts gfx950; use 'auto' or 'asm' on gfx1250"
        )
    return backend


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
