"""Cake SM120/SM121 DeepSeek-V4.1 mixed-cache sparse-MLA decode via FlashInfer.

FlashInfer entry: ``flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed``
(lazily re-exported from ``flashinfer.mla`` as
``cake_sparse_mla_sm120_dsv41_mixed_decode`` plus the ``*_num_chunks`` /
``*_scratch_bytes`` / ``*_supported_heads`` / ``*_format_info`` helpers), JIT
module ``flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed``. Added after the
``46340689a5ab`` baseline by FlashInfer PR flashinfer-ai/flashinfer#5983
(commit ``2c1c05250``, "feat(cake_dsv4_sparse_mla): SM120/SM121 DeepSeek-V4.1
mixed-cache sparse-MLA decode"); contract read at main ``e4f94f948``.

Contract (``kv_cache_format="fp8_dsv41_fp4_ca"`` on the public API,
``SparseMLASm120Wrapper(backend="cake", kv_cache_format="fp8",
kv_scale_format="ue8m0_g32", extra_kv_fp4=True)`` on the wrapper):

* ``q [T, H, 512]`` BF16 contiguous, ``H`` in the exported head table
  (``cake_sparse_mla_sm120_dsv41_mixed_supported_heads()``; 8 and every
  multiple of 16 up to 128 in the v1 family); ``output [T, H, 512]`` BF16;
  ``out_lse [T, H]`` fp32 base-2 times ``lse_scale``.
* main (SWA) cache: opaque uint8 pages of 528 bytes per token (512 E4M3
  values + a per-page footer of UE8M0 scales over 32-wide groups), written by
  ``dsv41_fp8_quantize_pack_sparse_mla_cache`` /
  ``dsv41_fp8_quantize_append_sparse_mla_cache`` (same FlashInfer PR; both
  forwarded here, JIT module ``flashinfer.jit.mla``); optional extra
  (compressed) cache: 288 bytes per token (512 E2M1 values packed in 256 bytes
  + a per-page footer of E4M3 scales over 16-wide groups), written by the
  pre-baseline ``dsv41_fp4_quantize_pack_sparse_mla_cache`` (not a Cake entry,
  not forwarded). Accepted views for both: 2-D ``[pages, page_bytes]``, 3-D
  ``[pages, page_size, bpt]``, HND ``[pages, 1, page_size, bpt]`` or NHD
  ``[pages, page_size, 1, bpt]``; rows packed inside a page, page stride a
  16-byte multiple (padded pools allowed), independent runtime page sizes.
* ``indices`` / ``extra_indices`` ``[T, topk]`` (or ``[T, 1, topk]``) int32,
  ``-1`` masks a slot; ``topk_length`` / ``extra_topk_length`` ``[T]`` int32;
  ``attn_sink [H]`` fp32 (sigmoid gate of the output, logaddexp into LSE).
* ``compute_precision``: ``"bf16"`` (default; both caches dequantized exactly
  to BF16 on chip, fp32 accumulation / softmax) or ``"fp8"`` (only when the
  generated family exports it); ``"nvfp4"`` is never valid for this format.
* Split-K: the planner (fitted on RTX PRO 6000 / RTX 5090 / GB10) may split a
  row across ``num_splits`` CTAs; caller-owned ``mid_out [T, H, S, 512]`` BF16
  / ``mid_lse [T, H, S]`` fp32 with ``S >= num_splits`` are then required
  (size them with ``*_num_chunks`` to cover every plan). The direct entry is
  allocation-free and CUDA-graph safe; the wrapper owns grow-only scratch and
  must be warmed on every shape before capture.

Not supported here (keep the existing SGLang path): SM90 / SM100 / SM103
devices, prefill (decode-only route), NVFP4 (384 B/token) caches (use
``attention_mla.sparse_mla_sm120_dsv4_nvfp4_decode``), head counts outside
the exported table, ``d_v != 512``, ``compute_precision="nvfp4"``, main caches
whose row width is not 528 bytes or extra caches not 288 bytes, and trees
without the generated family (``kernels_available()`` is ``False`` and the
module load raises ``FileNotFoundError``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from sglang.kernels.cake_kernels.attention_common import (
    SM120,
    SM121,
    archs_in,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed"
FI_JIT_MODULE = "flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed"
# The DSv4.1 FP8 cache writers live in the hand-written SM120 sparse-MLA
# module (same FlashInfer PR, different JIT family).
FI_API_MODULE = "flashinfer.mla._sparse_mla_sm120._api"
FI_API_JIT_MODULE = "flashinfer.jit.mla"
ARCHS = (SM120, SM121)
HEAD_DIM = 512
MAIN_BYTES_PER_TOKEN = 528
EXTRA_BYTES_PER_TOKEN = 288
COMPUTE_PRECISIONS = ("default", "bf16", "fp8")
KV_LAYOUTS = ("HND", "NHD")


def _cache_view_ok(cache, bytes_per_token: int) -> bool:
    """Shape-level mirror of FlashInfer's ``cache_page_geometry`` acceptance."""
    import torch

    if cache.dtype != torch.uint8 or cache.ndim not in (2, 3, 4):
        return False
    if cache.ndim == 2:
        page_bytes = int(cache.shape[1])
        return page_bytes >= bytes_per_token and page_bytes % bytes_per_token == 0
    if int(cache.shape[-1]) != bytes_per_token:
        return False
    if cache.ndim == 4 and cache.shape[1] != 1 and cache.shape[2] != 1:
        return False
    return int(cache.shape[0]) >= 1


def kernels_available() -> bool:
    """``True`` when the generated DSv4.1 mixed-cache family is in the FlashInfer tree.

    Imports ``flashinfer.jit`` (no CUDA initialisation, no build); ``False``
    when FlashInfer or the module is missing.
    """
    if not flashinfer_module_available(FI_MODULE, FI_JIT_MODULE):
        return False
    try:
        from flashinfer.jit.cake_sparse_mla_sm120_dsv41_mixed import (
            cake_sparse_mla_sm120_dsv41_mixed_available,
        )

        return bool(cake_sparse_mla_sm120_dsv41_mixed_available())
    except Exception:
        return False


def supports_sparse_mla_sm120_dsv41_mixed_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    compute_precision: str = "bf16",
) -> bool:
    """Admission check for the direct mixed-cache decode entry; never raises.

    Cheap checks first (device, module presence, dtypes, cache row widths,
    index shapes); then the exported head table and the generated family's
    presence are read from FlashInfer (host-only, no build).
    """
    try:
        import torch

        tensors = [q, kv_cache, indices]
        if extra_kv_cache is not None:
            tensors.append(extra_kv_cache)
        if extra_indices is not None:
            tensors.append(extra_indices)
        if not (
            archs_in(ARCHS, *tensors)
            and flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
            and (extra_kv_cache is None) == (extra_indices is None)
            and compute_precision in COMPUTE_PRECISIONS
            and q.dtype == torch.bfloat16
            and q.ndim == 3
            and q.is_contiguous()
            and int(q.shape[2]) == HEAD_DIM
            and _cache_view_ok(kv_cache, MAIN_BYTES_PER_TOKEN)
            and indices.dtype == torch.int32
            and indices.ndim in (2, 3)
            and int(indices.shape[0]) == int(q.shape[0])
            and int(indices.shape[-1]) >= 1
        ):
            return False
        if extra_kv_cache is not None and not (
            _cache_view_ok(extra_kv_cache, EXTRA_BYTES_PER_TOKEN)
            and extra_indices.dtype == torch.int32
            and extra_indices.ndim in (2, 3)
            and int(extra_indices.shape[0]) == int(q.shape[0])
            and int(extra_indices.shape[-1]) >= 1
        ):
            return False
        from flashinfer.mla._sparse_mla_sm120.cake_dsv41_mixed import (
            kernel_geometry,
            normalize_compute_precision,
        )

        geometry = kernel_geometry()
        return (
            not geometry.provisional
            and int(q.shape[1]) in geometry.head_counts
            and normalize_compute_precision(compute_precision) in geometry.precisions
        )
    except Exception:
        return False


def sparse_mla_sm120_dsv41_mixed_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    lse_scale: float = 1.0,
    num_splits: Optional[int] = None,
    max_splits: int = 16,
    head_tiles: Optional[int] = None,
    compute_precision: str = "bf16",
    enable_pdl: Optional[bool] = None,
) -> Dict[str, int]:
    """Forward to ``flashinfer.mla.cake_sparse_mla_sm120_dsv41_mixed_decode``.

    Writes ``output`` / ``out_lse`` in place and returns the resolved plan
    ``{"head_tiles", "num_splits", "chunks_per_block", "precision"}``.
    ``mid_out`` / ``mid_lse`` are required when the plan splits.
    """
    from flashinfer.mla import cake_sparse_mla_sm120_dsv41_mixed_decode

    return cake_sparse_mla_sm120_dsv41_mixed_decode(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
        lse_scale=lse_scale,
        num_splits=num_splits,
        max_splits=max_splits,
        head_tiles=head_tiles,
        compute_precision=compute_precision,
        enable_pdl=enable_pdl,
    )


def create_sparse_mla_sm120_dsv41_mixed_wrapper(
    max_num_tokens: Optional[int] = None,
    max_num_heads: Optional[int] = None,
    *,
    compute_precision: str = "default",
    device=None,
):
    """``SparseMLASm120Wrapper(backend="cake")`` on the DSv4.1 mixed cache.

    Fixes ``kv_cache_format="fp8"``, ``kv_scale_format="ue8m0_g32"`` and
    ``extra_kv_fp4=True`` (the only combination FlashInfer's Cake backend
    accepts for this format; ``d_v`` stays at FlashInfer's 512 default).
    ``compute_precision`` ``"default"`` resolves to the BF16 route.
    ``max_num_tokens`` / ``max_num_heads`` must be given together; the
    wrapper's grow-only scratch means every shape must be warmed before
    CUDA-graph capture. ``wrapper.run(...)`` keeps FlashInfer's contract
    (decode-only; ``prefill_impl`` must be ``None`` or ``"auto"``).
    """
    from flashinfer.mla import SparseMLASm120Wrapper

    return SparseMLASm120Wrapper(
        max_num_tokens=max_num_tokens,
        max_num_heads=max_num_heads,
        kv_scale_format="ue8m0_g32",
        kv_cache_format="fp8",
        extra_kv_fp4=True,
        compute_precision=compute_precision,
        device=device,
        backend="cake",
    )


# --------------------------------------------------------------------------
# DSv4.1 FP8 main-cache writers (flashinfer.mla._sparse_mla_sm120._api)
# --------------------------------------------------------------------------


def supports_dsv41_fp8_quantize_pack(
    latent_kv: torch.Tensor, *, kv_layout: str = "HND"
) -> bool:
    """Admission check for the full-page DSv4.1 FP8 pack; never raises."""
    try:
        import torch

        if not (
            cuda_tensor_on(latent_kv, ARCHS)
            and flashinfer_module_available(FI_API_MODULE, FI_API_JIT_MODULE)
            and kv_layout in KV_LAYOUTS
            and latent_kv.dtype in (torch.bfloat16, torch.float16)
            and latent_kv.is_contiguous()
            and int(latent_kv.shape[-1]) == HEAD_DIM
        ):
            return False
        if latent_kv.ndim == 3:
            return True
        return latent_kv.ndim == 4 and (
            latent_kv.shape[1] == 1 or latent_kv.shape[2] == 1
        )
    except Exception:
        return False


def dsv41_fp8_quantize_pack_sparse_mla_cache(
    latent_kv: torch.Tensor, *, kv_layout: str = "HND"
) -> torch.Tensor:
    """Forward to ``flashinfer.mla.dsv41_fp8_quantize_pack_sparse_mla_cache``.

    ``latent_kv`` BF16/FP16 ``[pages, page_size, 512]`` (optional singleton
    latent-head axis) -> opaque uint8 cache ``[pages, 1, page_size, 528]``
    (HND) or ``[pages, page_size, 1, 528]`` (NHD); groups of 32 values share
    one UE8M0 scale ``2**ceil(log2(max(amax / 448, 1e-4)))``.
    """
    from flashinfer.mla import dsv41_fp8_quantize_pack_sparse_mla_cache

    return dsv41_fp8_quantize_pack_sparse_mla_cache(latent_kv, kv_layout=kv_layout)


def supports_dsv41_fp8_quantize_append(
    latent_kv: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor
) -> bool:
    """Admission check for the slot-addressed DSv4.1 FP8 append; never raises."""
    try:
        import torch

        return (
            archs_in(ARCHS, latent_kv, slot_mapping, cache)
            and flashinfer_module_available(FI_API_MODULE, FI_API_JIT_MODULE)
            and latent_kv.dtype in (torch.bfloat16, torch.float16)
            and latent_kv.is_contiguous()
            and int(latent_kv.shape[-1]) == HEAD_DIM
            and slot_mapping.dtype in (torch.int32, torch.int64)
            and slot_mapping.ndim == 1
            and slot_mapping.is_contiguous()
            and _cache_view_ok(cache, MAIN_BYTES_PER_TOKEN)
        )
    except Exception:
        return False


def dsv41_fp8_quantize_append_sparse_mla_cache(
    latent_kv: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor
) -> None:
    """Forward to ``flashinfer.mla.dsv41_fp8_quantize_append_sparse_mla_cache``.

    One 512-wide BF16/FP16 row per ``slot_mapping`` entry
    (``page_id * page_size + entry_id``; negative / out-of-range slots are
    padding) is quantized into the 528-byte uint8 ``cache`` in place.
    """
    from flashinfer.mla import dsv41_fp8_quantize_append_sparse_mla_cache

    dsv41_fp8_quantize_append_sparse_mla_cache(latent_kv, slot_mapping, cache)


# --------------------------------------------------------------------------
# Host-only helpers (unregistered; no kernel launch)
# --------------------------------------------------------------------------


def sparse_mla_sm120_dsv41_mixed_num_chunks(topk: int, extra_topk: int = 0) -> int:
    """Host-only forward of ``cake_sparse_mla_sm120_dsv41_mixed_num_chunks``."""
    from flashinfer.mla import cake_sparse_mla_sm120_dsv41_mixed_num_chunks

    return cake_sparse_mla_sm120_dsv41_mixed_num_chunks(topk, extra_topk)


def sparse_mla_sm120_dsv41_mixed_scratch_bytes(
    num_tokens: int, num_heads: int, topk: int, extra_topk: int = 0
) -> int:
    """Host-only forward of ``cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes``."""
    from flashinfer.mla import cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes

    return cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes(
        num_tokens, num_heads, topk, extra_topk
    )


def sparse_mla_sm120_dsv41_mixed_supported_heads() -> Tuple[int, ...]:
    """Host-only forward of ``cake_sparse_mla_sm120_dsv41_mixed_supported_heads``."""
    from flashinfer.mla import cake_sparse_mla_sm120_dsv41_mixed_supported_heads

    return tuple(cake_sparse_mla_sm120_dsv41_mixed_supported_heads())


def sparse_mla_sm120_dsv41_mixed_compute_precisions() -> Tuple[str, ...]:
    """Routes exported by the generated family (``"bf16"`` always, ``"fp8"`` optional)."""
    from flashinfer.mla import cake_sparse_mla_sm120_dsv41_mixed_format_info

    return tuple(cake_sparse_mla_sm120_dsv41_mixed_format_info()["compute_precisions"])
