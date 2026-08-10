"""Fused MXFP4 decode attention for DeepSeek V4 — PyTorch custom op.

Uses a standalone CUDA kernel (no sgl-kernel / TVM-FFI deps) registered
via TORCH_LIBRARY so the kernel launch is fully CUDA-graph capturable.

Returns (output, log_sum_exp) for merging with extra (C4/C128) attention
via online softmax composition.
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# JIT compile + cache
# ---------------------------------------------------------------------------

_KERNEL_SRC_PATH = (
    pathlib.Path(__file__).parent.parent
    / "csrc"
    / "deepseek_v4"
    / "mxfp4_decode_standalone.cu"
)
_CACHE_DIR = pathlib.Path.home() / ".cache" / "sglang_jit" / "mxfp4_decode"


def _source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(_KERNEL_SRC_PATH.read_bytes())
    return digest.hexdigest()[:16]


def _ensure_compiled() -> None:
    """Compile the standalone CUDA op if not already cached."""
    src_hash = _source_hash()
    build_dir = _CACHE_DIR / src_hash
    build_dir.mkdir(parents=True, exist_ok=True)

    marker = build_dir / ".build_done"
    so_candidates = list(build_dir.glob("*.so"))
    if marker.exists() and so_candidates:
        torch.classes.load_library(str(so_candidates[0]))
        return

    # Copy kernel source into build dir and create a minimal cpp stub.
    kernel_dst = build_dir / "mxfp4_decode.cu"
    shutil.copy2(_KERNEL_SRC_PATH, kernel_dst)
    cpp_stub = build_dir / "stub.cpp"
    cpp_stub.write_text("// stub — all code is in the .cu file\n")

    import torch.utils.cpp_extension as ext

    ext.load(
        name="sglang_mxfp4_decode",
        sources=[str(cpp_stub), str(kernel_dst)],
        build_directory=str(build_dir),
        verbose=False,
        is_python_module=False,
    )
    marker.touch()

    so_files = list(build_dir.glob("*.so"))
    if so_files:
        torch.classes.load_library(str(so_files[0]))


# Eagerly compile on import (one-time cost, cached thereafter).
_ensure_compiled()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mxfp4_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_indices: torch.Tensor,
    sm_scale: float,
    swa_width: int = 128,
    num_valid: int = 0,
    attn_sink: Optional[torch.Tensor] = None,
    swa_lengths: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_lengths: Optional[torch.Tensor] = None,
    extra_topk_width: int = 0,
    extra_page_size: int = 0,
) -> torch.Tensor:
    """Fused MXFP4 decode attention — CUDA-graph safe.

    One call attends over the SWA window and (optionally) the extra C4/C128
    cache with per-query valid lengths, fusing the online softmax.

    Args:
        q:            [N, 512] BF16 — flattened Q (batch * num_heads).
        k_cache:      [*, 368] uint8 — row-major MXFP4 SWA K-cache (one row
                      per flat SWA slot, tightly packed).
        page_indices: [N, W] int32 — flat SWA slot ids per query (W = swa_width,
                      padded with -1); rows are addressed as slot * 368 so
                      windows crossing storage-page boundaries work.
        sm_scale:     float — 1/sqrt(head_dim).
        swa_width:    int — padded width W of page_indices (128 for DSV4 SWA).
        num_valid:    int — fallback valid tokens when swa_lengths is None
                      (0 = scan all swa_width).
        attn_sink:    [N] float32 or None — per-head sink values.
        swa_lengths:  [N] int32 or None — per-query valid SWA tokens
                      (clamped to swa_width); None falls back to num_valid.
        extra_k_cache: [*, 368] uint8 or None — MXFP4 C4/C128 cache.
        extra_indices: [N, W] int32 or None — flattened token ids per query.
        extra_topk_lengths: [N] int32 or None — per-query valid count.
        extra_topk_width: int — padded width W of extra_indices.
        extra_page_size: int — tokens per page in the extra cache.

    Returns:
        o: [N, 512] BF16 attention output.
    """
    if q.ndim != 2 or q.shape[1] != 512:
        raise ValueError(f"q must be [N, 512] BF16, got {q.shape}")
    if k_cache.ndim != 2 or k_cache.shape[1] != 368:
        raise ValueError(f"k_cache must be [*, 368] uint8, got {k_cache.shape}")
    if page_indices.ndim != 2 or page_indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"page_indices must be [{q.shape[0]}, W], got {page_indices.shape}"
        )
    if page_indices.shape[1] != swa_width:
        raise ValueError(
            f"page_indices width {page_indices.shape[1]} != swa_width {swa_width}"
        )
    n = q.shape[0]
    if swa_lengths is not None:
        if swa_lengths.shape != (n,):
            raise ValueError(f"swa_lengths must be [{n}], got {swa_lengths.shape}")
        swa_lengths = swa_lengths.contiguous().to(torch.int32)
    have_extra = extra_k_cache is not None
    if have_extra:
        if extra_indices is None or extra_topk_lengths is None:
            raise ValueError("extra_k_cache requires extra_indices and extra_topk_lengths")
        if extra_k_cache.ndim != 2 or extra_k_cache.shape[1] != 368:
            raise ValueError(
                f"extra_k_cache must be [*, 368] uint8, got {extra_k_cache.shape}"
            )
        if extra_indices.ndim != 2 or extra_indices.shape[0] != n:
            raise ValueError(f"extra_indices must be [{n}, W], got {extra_indices.shape}")
        if extra_topk_lengths.shape != (n,):
            raise ValueError(f"extra_topk_lengths must be [{n}], got {extra_topk_lengths.shape}")
        if extra_topk_width <= 0:
            raise ValueError("extra_topk_width must be positive")
        extra_k_cache = extra_k_cache.contiguous()
        extra_indices = extra_indices.contiguous().to(torch.int32)
        extra_topk_lengths = extra_topk_lengths.contiguous().to(torch.int32)

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    page_indices = page_indices.contiguous().to(torch.int32)
    if attn_sink is not None:
        attn_sink = attn_sink.contiguous().to(torch.float32)

    o = torch.empty_like(q)
    lse = torch.empty(q.shape[0], dtype=torch.float32, device=q.device)

    o, _lse = torch.ops.sglang_mxfp4.decode(
        q,
        k_cache,
        page_indices,
        swa_lengths,
        extra_k_cache,
        extra_indices,
        extra_topk_lengths,
        attn_sink,
        o,
        lse,
        sm_scale,
        swa_width,
        num_valid,
        extra_topk_width,
        extra_page_size,
    )
    return o


__all__ = ["mxfp4_decode_attention"]
