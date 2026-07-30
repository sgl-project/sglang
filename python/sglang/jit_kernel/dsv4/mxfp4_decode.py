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
    page_size: int = 128,
    num_valid: int = 0,
    attn_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MXFP4 decode attention — CUDA-graph safe.

    Args:
        q:            [N, 512] BF16 — flattened Q (batch * num_heads).
        k_cache:      [*, 368] uint8 — row-major MXFP4 K-cache.
        page_indices: [N] int32 — kernel-page indices per query.
        sm_scale:     float — 1/sqrt(head_dim).
        page_size:    int — tokens per kernel page (128 for DSV4 SWA).
        num_valid:    int — actual valid tokens (0 = scan all page_size).
        attn_sink:    [N] float32 or None — per-head sink values.

    Returns:
        o: [N, 512] BF16 attention output.
    """
    if q.ndim != 2 or q.shape[1] != 512:
        raise ValueError(f"q must be [N, 512] BF16, got {q.shape}")
    if k_cache.ndim != 2 or k_cache.shape[1] != 368:
        raise ValueError(f"k_cache must be [*, 368] uint8, got {k_cache.shape}")
    if page_indices.ndim != 1 or page_indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"page_indices must be [{q.shape[0]}], got {page_indices.shape}"
        )

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
        attn_sink,
        o,
        lse,
        sm_scale,
        page_size,
        num_valid,
    )
    return o


__all__ = ["mxfp4_decode_attention"]
