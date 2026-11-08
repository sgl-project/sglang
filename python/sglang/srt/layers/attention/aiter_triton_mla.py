"""gfx1250 MLA decode through aiter's Triton ``mla_decode_fwd``.

The wrapper calls ``aiter.ops.triton.attention.mla.mla_decode_fwd`` with a bf16
query and reads the MLA KV pool through block tables; aiter picks the kernel
behind that entry (its Gluon MLA decode kernels on gfx1250). It runs for
``SGLANG_AITER_MLA_DECODE_BACKEND=triton``, which ``auto`` picks on gfx1250 for
MLA with bf16 activations, no DCP and ``qk_rope_head_dim > 0``. That value is
not sglang's ``--decode-attention-backend triton``, which leaves aiter for
sglang's own Triton decode kernel. Explicit values are honored as-is on every
arch: ``asm`` uses aiter's ASM ``mla_decode_fwd``, and ``mla_gluon`` only where
``prefer_mla_gluon_decode`` holds and the ASM persistent kernel cannot run the
batch; ``gluon`` uses aiter's ``mla_gluon`` for DCP decode and where
``prefer_mla_gluon_decode`` holds, and the ASM decode otherwise. aiter without
ROCm/aiter#5935 computes wrong outputs for batches it runs with one KV segment
per sequence. This path does not read ``AITER_MLA_DECODE_PS1_FLYDSL``.
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


def resolve_aiter_mla_decode_backend(
    value: str,
    *,
    use_mla: bool,
    input_dtype: torch.dtype,
    dcp_world_size: int,
    qk_rope_head_dim: int,
) -> str:
    """``SGLANG_AITER_MLA_DECODE_BACKEND`` as "asm", "gluon" or "triton".

    "auto" is "triton" for MLA where ``supports_aiter_triton_mla_decode``
    holds, else "asm". Explicit values are honored as-is: "asm" and "gluon"
    are returned unchanged on every arch. Unknown values raise, and so does
    "triton" where this backend does not call aiter's Triton
    ``mla_decode_fwd``: off gfx1250, and on gfx1250 without bf16 activations,
    with DCP or with ``qk_rope_head_dim`` 0.
    """
    backend = value.lower()
    if backend not in ("auto", "asm", "gluon", "triton"):
        raise ValueError(
            "SGLANG_AITER_MLA_DECODE_BACKEND must be 'auto', 'asm', 'gluon' or "
            f"'triton', got {value!r}"
        )
    if backend in ("asm", "gluon"):
        return backend
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
    if backend == "auto":
        return "triton" if triton else "asm"
    if not triton:
        raise ValueError(
            "SGLANG_AITER_MLA_DECODE_BACKEND=triton needs bf16 activations, no "
            f"DCP and qk_rope_head_dim > 0; got {input_dtype}, DCP size "
            f"{dcp_world_size}, qk_rope_head_dim {qk_rope_head_dim}"
        )
    return backend


def triton_mla_block_size(page_size: int) -> int:
    """KV tokens per kernel block: the largest power of 2 dividing page_size, capped at 128.

    aiter's Triton ``mla_decode_fwd`` on gfx1250 takes a power-of-2 block size,
    and 256 overflows the 320 KB LDS at 128 heads with bf16 KV. Block sizes
    1-128 are verified at 8, 16 and 128 heads with bf16 and fp8 KV.
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
