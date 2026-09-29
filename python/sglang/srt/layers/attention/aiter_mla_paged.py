"""aiter's paged MLA attention (``aiter.ops.triton.attention.mla``).

These kernels are not arch-specific -- they pick a Gluon implementation on
gfx1250 and a plain Triton one elsewhere -- but they differ from the ASM
``aiter.mla`` entry points SGLang normally uses: they read the KV cache as a
paged ``[num_blocks, page_size, num_kv_heads, qk_head_dim]`` tensor, a plain
view of SGLang's pool, and take a 2-D page table. Nothing about Q or the cache
has to be repacked, and the cache may be bf16 or fp8 e4m3 with a scalar
descale.

Only gfx1250 routes here. gfx942 and gfx950 have tuned ASM MLA paths already;
gfx1250 has none that work, its decode kernels wanting seg-packed fp8 KV and
returning non-finite output at 64 and 128 query heads, its MHA extend wedging
the HSA queue partway through a chunked prefill.

Decode additionally requires an aiter carrying the ``NUM_SEGMENTS_PER_SEQ == 1``
fix in ``_mla_decode_fwd_kernel_non_pipelined``: without it the kernel writes
its softmax max/expsum through pointers the host aliased onto the output
buffer, corrupting the result whenever the batch is large enough to drive the
segment count to 1 (roughly batch > CU_count/2). ``_probe_ok`` checks for it at
runtime because the corruption is otherwise silent. Prefill has no segments and
is unaffected.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx1250_supported, is_hip

logger = logging.getLogger(__name__)

# The Gluon kernels are built for a 64-token page (128 is FP4-only).
_SUPPORTED_PAGE_SIZE = 64
_SUPPORTED_KV_DTYPES = (torch.bfloat16, fp8_dtype)


@dataclass
class PagedMlaMetadata:
    """Per-batch inputs the paged MLA kernels need beyond the KV pool.

    ``page_table`` is ``[bs, max_pages]`` of physical page ids and may be wider
    than the batch needs; the kernels bound their reads with ``seq_lens`` and
    only ever see ``page_table.stride(0)``. ``seq_lens`` is the full per-request
    KV length, while ``cu_seqlens_q`` spans only the tokens being attended --
    the two coincide for decode and differ for extend.
    """

    page_table: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor


def _routed_here() -> bool:
    if not (is_hip() and is_gfx1250_supported()):
        return False
    if not envs.SGLANG_AITER_MLA_PAGED.get():
        logger.info("aiter paged MLA is off; set SGLANG_AITER_MLA_PAGED=1 to enable.")
        return False
    return True


@functools.lru_cache(maxsize=1)
def _decode_fn():
    if not _routed_here():
        return None
    try:
        from aiter.ops.triton.attention.mla import mla_decode_fwd
    except ImportError as exc:
        logger.info("aiter paged MLA decode import error: %s", exc)
        return None
    if not _probe_ok(mla_decode_fwd):
        logger.warning(
            "aiter paged MLA decode is disabled: this aiter miscomputes decode "
            "at large batch (missing the NUM_SEGMENTS_PER_SEQ == 1 guard around "
            "the segm_max / segm_expsum stores)."
        )
        return None
    return mla_decode_fwd


@functools.lru_cache(maxsize=1)
def _prefill_fn():
    if not _routed_here():
        return None
    try:
        from aiter.ops.triton.attention.mla import mla_prefill_fwd
    except ImportError as exc:
        logger.info("aiter paged MLA prefill import error: %s", exc)
        return None
    return mla_prefill_fwd


def _probe_ok(mla_decode_fwd) -> bool:
    """One batch large enough to force NUM_SEGMENTS_PER_SEQ == 1, checked
    against a torch reference. Costs one compile, once per process."""
    try:
        dev = torch.device("cuda")
        batch, heads, lora, rope = 256, 16, 512, 64
        ctx, head_dim = _SUPPORTED_PAGE_SIZE, lora + rope
        kv = torch.randn(batch, ctx, 1, head_dim, dtype=torch.bfloat16, device=dev)
        q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device=dev)
        out = torch.empty(batch, heads, lora, dtype=torch.bfloat16, device=dev)
        scale = head_dim**-0.5
        mla_decode_fwd(
            q=q,
            kv_buffer=kv,
            out=out,
            cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=dev),
            seqused_k=torch.full((batch,), ctx, dtype=torch.int32, device=dev),
            max_seqlen_kv=ctx,
            block_tables=torch.arange(batch, dtype=torch.int32, device=dev)[:, None],
            softmax_scale=scale,
            kv_lora_rank=lora,
            qk_rope_head_dim=rope,
            causal=True,
            q_descale=None,
            kv_descale=None,
        )
        k = kv.view(batch, ctx, head_dim).float()
        attn = torch.softmax(torch.einsum("bqd,bkd->bqk", q.float(), k) * scale, -1)
        ref = torch.einsum("bqk,bkd->bqd", attn, k[..., :lora])
        return torch.allclose(out.float(), ref, atol=0.05, rtol=0.05)
    except Exception as exc:  # noqa: BLE001 - a probe failure just disables the path
        logger.info("aiter paged MLA decode probe failed: %s", exc)
        return False


def log_paged_mla_capability(log: logging.Logger | None = None) -> None:
    """Report the decision where it applies; silent on other archs."""
    if not (is_hip() and is_gfx1250_supported()):
        return
    (log or logger).info(
        "aiter paged MLA: decode %s, prefill %s",
        "enabled" if _decode_fn() is not None else "disabled",
        "enabled" if _prefill_fn() is not None else "disabled",
    )


def _shape_supported(page_size: int, kv_cache_dtype: torch.dtype) -> bool:
    return page_size == _SUPPORTED_PAGE_SIZE and kv_cache_dtype in _SUPPORTED_KV_DTYPES


def prefer_paged_mla_decode(*, page_size: int, kv_cache_dtype: torch.dtype) -> bool:
    return _shape_supported(page_size, kv_cache_dtype) and _decode_fn() is not None


def prefer_paged_mla_prefill(*, page_size: int, kv_cache_dtype: torch.dtype) -> bool:
    return _shape_supported(page_size, kv_cache_dtype) and _prefill_fn() is not None


def _run(
    fn,
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    out: torch.Tensor,
    meta: PagedMlaMetadata,
    max_seqlen_kv: int,
    page_size: int,
    qk_head_dim: int,
    v_head_dim: int,
    sm_scale: float,
    kv_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    fn(
        q=q,
        kv_buffer=k_buffer.view(-1, page_size, 1, qk_head_dim),
        out=out,
        cu_seqlens_q=meta.cu_seqlens_q,
        seqused_k=meta.seq_lens,
        max_seqlen_kv=max_seqlen_kv,
        block_tables=meta.page_table,
        softmax_scale=sm_scale,
        kv_lora_rank=v_head_dim,
        qk_rope_head_dim=qk_head_dim - v_head_dim,
        causal=True,
        q_descale=None,
        kv_descale=kv_descale,
    )
    return out


def paged_mla_decode(**kwargs) -> torch.Tensor:
    """Decode fused Q ``[num_tokens, H, qk_head_dim]`` into ``out``."""
    return _run(_decode_fn(), **kwargs)


def paged_mla_prefill(**kwargs) -> torch.Tensor:
    """Absorbed MLA extend: ragged Q attending the paged prefix plus itself.

    The kernel derives each query's causal bound from ``seq_lens`` minus its
    own query length, so a no-prefix chunk works the same as a cached one.
    """
    return _run(_prefill_fn(), **kwargs)
