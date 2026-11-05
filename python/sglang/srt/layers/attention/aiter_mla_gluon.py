"""Gluon MLA decode wrapper for low head-count MLA (e.g. Kimi K3 TP8: 12 heads/GPU).

Uses aiter ``mla_gluon`` when import succeeds and Triton Gluon exposes ``cga_layout``
(needs Triton >= 3.7). Falls back to the caller (zero-pad + ``mla_decode_fwd``) when
Gluon is unavailable.

Requires aiter ``main`` with ROCm/aiter #4480 (batch>1 ``bh16bn128``) and #4555
(decode CUDA graph KV splits). SGLang probes import + Triton API only; aiter version
is not pinned at build time.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _gluon_fn():
    """aiter mla gluon entry point return None if disabled"""
    if not envs.SGLANG_AITER_MLA_GLUON.get():
        logger.info("aiter mla gluon is disabled manually.")
        return None
    try:
        import triton
        import triton.experimental.gluon.language as gl
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    except ImportError as exc:
        logger.info("aiter mla gluon import error message: %s", exc)
        return None
    # mla_gluon builds its shared layouts with cga_layout, added in Triton 3.7;
    # older Triton fails at compile time with an opaque error instead.
    if "cga_layout" not in inspect.signature(gl.PaddedSharedLayout).parameters:
        logger.info(
            "aiter mla gluon is disabled due to triton %s has no Gluon cga_layout (need >= 3.7)",
            getattr(triton, "__version__", "unknown"),
        )
        return None
    return mla_gluon


def log_mla_gluon_capability(log: logging.Logger | None = None) -> None:
    """Report whether Gluon MLA decode is valid; the reason is logged by _gluon_fn."""
    ready = _gluon_fn() is not None
    (log or logger).info(
        "aiter mla gluon is %s",
        "enabled" if ready else "disabled; falling back to zero-pad mla_decode_fwd",
    )


def _is_gfx950() -> bool:
    # mla_gluon asserts this exact arch; is_gfx95_supported() also admits other gfx95x.
    return (
        torch.version.hip is not None
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"
    )


def mla_gluon_dcp_unsupported_reason(
    *,
    gathered_num_head: int,
    q_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> str | None:
    """Why aiter's Gluon MLA cannot run a DCP decode or verify here, or None if it can.

    Mirrors mla_gluon's asserts for its bh16 regimes. Its bh64 regime (64 or 128
    heads) also needs batch % 64 and a long min_kv_seq_len, which DCP cannot promise.
    """
    mla_gluon = _gluon_fn()
    if mla_gluon is None:
        return "Gluon MLA is unavailable (see the 'aiter mla gluon' log)"
    if "return_lse" not in inspect.signature(mla_gluon).parameters:
        return "this aiter's mla_gluon cannot return the lse the DCP merge needs"
    if not _is_gfx950():
        return "mla_gluon runs only on gfx950"
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        return (
            "mla_gluon needs kv_lora_rank 512 and qk_rope_head_dim 64, "
            f"got {kv_lora_rank} and {qk_rope_head_dim}"
        )
    if not 1 <= gathered_num_head <= 96 or gathered_num_head == 64:
        return (
            "mla_gluon runs at most 96 gathered heads other than 64, "
            f"got {gathered_num_head}"
        )
    if q_dtype != torch.bfloat16:
        return f"mla_gluon needs a bf16 query, got {q_dtype}"
    if kv_cache_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return f"mla_gluon needs a bf16 or fp8 e4m3fn KV cache, got {kv_cache_dtype}"
    return None


def prefer_mla_gluon_decode(
    *, head_pad_mode: str, num_head: int, kv_cache_dtype: torch.dtype
) -> bool:
    return (
        head_pad_mode == "zero"
        and num_head == 12
        and kv_cache_dtype == fp8_dtype
        and _gluon_fn() is not None
    )


def mla_gluon_decode(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer: RadixAttention,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float,
    min_kv_seq_len: int,
    kv_scale: float = 1.0,
    qlen: int = 1,
    use_2d_view: bool = False,
    return_lse: bool = False,
):
    """Run Gluon MLA decode for fused Q [num_tokens, H, 576] and MLA KV pool.
    Returns [num_tokens, H, v_head_dim] (or ``(out, lse)`` when ``return_lse``),
    or None when Gluon is unavailable.
    """
    mla_gluon = _gluon_fn()
    if mla_gluon is None:
        return None

    num_head = layer.tp_q_head_num
    kv_lora_rank = layer.v_head_dim
    qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank
    batch_size = q.shape[0] // qlen

    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    if qlen > 1:
        # Splitting the leading dim is a stride change, so these stay views of
        # the non-contiguous torch.split outputs.
        q_nope = q_nope.view(batch_size, qlen, num_head, kv_lora_rank)
        q_pe = q_pe.view(batch_size, qlen, num_head, qk_rope_head_dim)
        o = q.new_empty((batch_size, qlen, num_head, kv_lora_rank))
    else:
        o = q.new_empty((batch_size, num_head, kv_lora_rank))

    extra_kwargs = {}
    if return_lse:
        extra_kwargs["return_lse"] = True

    result = mla_gluon(
        q_nope,
        q_pe,
        k_buffer.view(-1, layer.qk_head_dim),
        o,
        kv_indices,
        kv_indptr,
        sm_scale,
        k_pe=None,
        kv_pe_offset=kv_lora_rank,
        use_2d_view=use_2d_view,
        kv_scale=kv_scale,
        min_kv_seq_len=min_kv_seq_len,
        **extra_kwargs,
    )
    # Hand back the caller's flat [num_tokens, H, v] layout either way.
    out = o.flatten(0, 1) if qlen > 1 else o
    if not return_lse:
        return out
    # mla_gluon writes the output into `o` and returns it alongside the lse.
    _, lse = result
    return out, lse
