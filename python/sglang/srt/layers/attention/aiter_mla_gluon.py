"""Gluon MLA decode wrapper for low head-count MLA (e.g. Kimi K3 TP8: 12 heads/GPU).

Uses aiter ``mla_gluon`` when import/JIT succeeds. Falls back to the caller when
Gluon is unavailable or fails to compile (known on some Triton 3.6.0 builds).
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

_MLA_GLUON_ENABLED = get_bool_env_var("SGLANG_AITER_MLA_GLUON", "True")
_mla_gluon_fn = None
_mla_gluon_import_failed = False


def _in_cuda_graph_capture() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def mla_gluon_available() -> bool:
    if not _MLA_GLUON_ENABLED:
        return False
    global _mla_gluon_fn, _mla_gluon_import_failed
    if _mla_gluon_import_failed:
        return False
    if _mla_gluon_fn is not None:
        return True
    try:
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon as fn

        _mla_gluon_fn = fn
        return True
    except ImportError:
        _mla_gluon_import_failed = True
        logger.warning("mla_gluon import failed; Gluon MLA decode disabled.")
        return False


def mla_gluon_decode(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer: RadixAttention,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    kv_scale: float = 1.0,
    min_kv_seq_len: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Run Gluon MLA decode for fused Q [B, H, 576] and MLA KV pool.

    Returns output [B, H, v_head_dim] on success, or None to fall back.

    ``min_kv_seq_len`` must be supplied by the caller during CUDA graph capture
    (no GPU->CPU sync from ``seq_lens``). For eager decode, omit it to derive
    from ``seq_lens`` when safe.
    """
    if not mla_gluon_available():
        return None

    batch_size = q.shape[0]

    kv_lora_rank = layer.v_head_dim
    qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank
    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_rope_head_dim], dim=-1)

    o = q.new_empty((batch_size, layer.tp_q_head_num, kv_lora_rank))

    kv_c = k_buffer.view(-1, layer.qk_head_dim)
    if min_kv_seq_len is None:
        if _in_cuda_graph_capture():
            logger.warning(
                "mla_gluon_decode: min_kv_seq_len missing during CUDA graph capture"
            )
            min_kv_seq_len = 1
        elif seq_lens.numel():
            min_kv_seq_len = int(seq_lens.max().item())
        else:
            min_kv_seq_len = 1

    try:
        _mla_gluon_fn(
            q_nope,
            q_pe,
            kv_c,
            o,
            kv_indices,
            kv_indptr,
            sm_scale,
            k_pe=None,
            kv_pe_offset=kv_lora_rank,
            use_2d_view=False,
            kv_scale=kv_scale,
            min_kv_seq_len=min_kv_seq_len,
        )
        return o
    except Exception as exc:
        logger.warning(
            "mla_gluon decode failed (num_head=%s, kv_dtype=%s, batch=%s): %s",
            layer.tp_q_head_num,
            k_buffer.dtype,
            batch_size,
            exc,
        )
        return None


def prefer_mla_gluon_decode(*, num_head: int, kv_cache_dtype: torch.dtype) -> bool:
    """Route h12 decode through Gluon when FP8 KV (persist ASM lacks fp8 qh16)."""
    if not mla_gluon_available():
        return False
    if num_head == 12 and kv_cache_dtype == fp8_dtype:
        return True
    return get_bool_env_var("SGLANG_AITER_MLA_GLUON_FORCE", "False")
