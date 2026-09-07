"""Gluon MLA decode wrapper for low head-count MLA (e.g. Kimi K3 TP8: 12 heads/GPU).

Uses aiter ``mla_gluon`` when import succeeds and Triton Gluon exposes ``cga_layout``
(needs Triton >= 3.7). Falls back to the caller (zero-pad + ``mla_decode_fwd``) when
Gluon is unavailable or fails at runtime.

Requires aiter ``main`` with ROCm/aiter #4480 (batch>1 ``bh16bn128``) and #4555
(decode CUDA graph KV splits). SGLang probes import + Triton API only; aiter version
is not pinned at build time.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

_mla_gluon_fn = None
_mla_gluon_import_failed = False
_capability_cache: Optional[MlaGluonCapability] = None


@dataclass(frozen=True)
class MlaGluonCapability:
    """Runtime probe of aiter/Triton Gluon prerequisites for h12 + FP8 decode."""

    enabled_by_env: bool
    import_ok: bool
    triton_version: str
    triton_cga_layout_ok: bool
    ready: bool
    summary: str

    def missing_for_ready(self) -> list[str]:
        missing = []
        if not self.enabled_by_env:
            missing.append("SGLANG_AITER_MLA_GLUON=0")
        if not self.import_ok:
            missing.append("aiter.ops.triton.gluon.mla_gluon import")
        if not self.triton_cga_layout_ok:
            missing.append(
                f"Triton Gluon cga_layout (have {self.triton_version or 'unknown'}, need >= 3.7)"
            )
        return missing


def _triton_version() -> str:
    try:
        import triton

        return getattr(triton, "__version__", "unknown")
    except Exception:
        return "missing"


def _triton_cga_layout_ok() -> bool:
    try:
        import triton.experimental.gluon.language as gl

        return "cga_layout" in inspect.signature(gl.PaddedSharedLayout).parameters
    except Exception:
        return False


def _gluon_runtime_ok() -> bool:
    return mla_gluon_available() and _triton_cga_layout_ok()


def _mla_gluon_enabled() -> bool:
    return envs.SGLANG_AITER_MLA_GLUON.get()


def probe_mla_gluon_capability(*, force_refresh: bool = False) -> MlaGluonCapability:
    global _capability_cache
    if _capability_cache is not None and not force_refresh:
        return _capability_cache

    enabled = _mla_gluon_enabled()
    triton_ver = _triton_version()
    import_ok = mla_gluon_available() if enabled else False
    cga_ok = _triton_cga_layout_ok()
    ready = enabled and import_ok and cga_ok

    if ready:
        summary = f"Gluon MLA h12+fp8 ready (Triton={triton_ver})"
    else:
        cap = MlaGluonCapability(
            enabled_by_env=enabled,
            import_ok=import_ok,
            triton_version=triton_ver,
            triton_cga_layout_ok=cga_ok,
            ready=False,
            summary="",
        )
        missing = cap.missing_for_ready()
        summary = (
            "Gluon MLA h12+fp8 disabled; fallback to zero-pad mla_decode_fwd "
            f"({'; '.join(missing)})"
        )

    _capability_cache = MlaGluonCapability(
        enabled_by_env=enabled,
        import_ok=import_ok,
        triton_version=triton_ver,
        triton_cga_layout_ok=cga_ok,
        ready=ready,
        summary=summary,
    )
    return _capability_cache


def log_mla_gluon_capability(log: logging.Logger | None = None) -> MlaGluonCapability:
    cap = probe_mla_gluon_capability()
    (log or logger).info(cap.summary)
    if not cap.ready:
        for item in cap.missing_for_ready():
            (log or logger).info("  missing: %s", item)
    return cap


def _in_cuda_graph_capture() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def mla_gluon_available() -> bool:
    if not _mla_gluon_enabled():
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
            "mla_gluon decode failed (num_head=%s, kv_dtype=%s, batch=%s): %s; "
            "falling back to zero-pad mla_decode_fwd",
            layer.tp_q_head_num,
            k_buffer.dtype,
            batch_size,
            exc,
        )
        return None


def prefer_mla_gluon_decode(
    *, head_pad_mode: str, num_head: int, kv_cache_dtype: torch.dtype
) -> bool:
    """Route Kimi-style h12 zero-pad MLA decode through Gluon when FP8 KV holds.

    ``head_pad_mode == "zero"`` selects the legacy ``mla_decode_fwd`` padding
    topology (N heads padded to 16). Gluon is only validated for ``num_head == 12``
    today; other zero-pad head counts must stay on zero-pad + ``mla_decode_fwd``.
    """
    if not _mla_gluon_enabled():
        return False
    if head_pad_mode == "zero" and num_head == 12 and kv_cache_dtype == fp8_dtype:
        return _gluon_runtime_ok()
    return False


def reset_mla_gluon_state_for_test() -> None:
    """Test helper: clear import/probe caches."""
    global _mla_gluon_fn, _mla_gluon_import_failed, _capability_cache
    _mla_gluon_fn = None
    _mla_gluon_import_failed = False
    _capability_cache = None
