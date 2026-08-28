# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Tile and kernel selection for the SM100/SM103 residue fold GEMM."""

from __future__ import annotations

import os

import torch

# Escape hatch for harnesses that call the fallback on purpose.
_SILENCE_DEFAULT = os.environ.get("SGLANG_RESIDUE_FOLD_SILENCE_DEFAULT", "0") == "1"

from .host import mext_fold_gemm_sm103

# ---------------------------------------------------------------------------
# Arch-aware, token-count-dependent selection. Each arch lists ordered
# (m_max, kernel_arch, mma_tiler, cluster); the first entry with m_tok <= m_max
# wins; the last entry always matches (fallback in select_arch_tactic).
_SENTINEL_M = 1 << 30
ARCH_TACTIC_BY_M: dict[str, list[tuple[int, str, tuple[int, int], tuple[int, int]]]] = {
    "sm100": [
        (64, "sm100", (128, 64), (1, 1)),  # decode: N=64 1SM floor
        (_SENTINEL_M, "sm100", (256, 128), (2, 1)),  # m>64: 2SM (safe blanket)
    ],
    "sm103": [
        # SM100's narrow tile covers decode sizes; larger inputs use SM103.
        (64, "sm100", (128, 64), (1, 1)),  # decode: sm100 kernel, N=64 1SM
        (256, "sm103", (256, 128), (2, 1)),  # transition band (mixed winners)
        (_SENTINEL_M, "sm103", (256, 256), (2, 1)),  # m>256: K=96 amortization wins
    ],
}


def kernel_arch_for_capability(major: int, minor: int) -> str | None:
    """Map a CUDA compute capability to the fold kernel arch, or None if the
    fold path is unsupported (caller falls back to unfused)."""
    if (major, minor) == (10, 0):
        return "sm100"
    if (major, minor) == (10, 3):
        return "sm103"
    return None


def select_arch_tactic(
    arch: str, m_tok: int
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """(kernel_arch, mma_tiler, cluster) for this arch + token count.

    `arch` must come from kernel_arch_for_capability (KeyError otherwise). The
    last list entry is an unconditional fallback (matches any m_tok, incl. the
    _SENTINEL_M edge), so this never raises for a valid arch.
    """
    entries = ARCH_TACTIC_BY_M[arch]
    for m_max, kernel_arch, tiler, cluster in entries[:-1]:
        if m_tok <= m_max:
            return kernel_arch, tiler, cluster
    _m_max, kernel_arch, tiler, cluster = entries[-1]  # unconditional fallback
    return kernel_arch, tiler, cluster


def run_fold_default(
    major: int,
    minor: int,
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the fold GEMM with the validated default tactic for this GPU/m_tok."""
    arch = kernel_arch_for_capability(major, minor)
    if arch is None:
        raise RuntimeError(f"no fold kernel for capability ({major},{minor})")
    m_tok = act_fp4_rowpair.shape[0] // 2
    kernel_arch, tiler, cluster = select_arch_tactic(arch, m_tok)
    if not _SILENCE_DEFAULT:
        from ..fold import warn_default_path

        warn_default_path(arch, m_tok, f"tile={tiler}")
    return mext_fold_gemm_sm103(
        weight_fp4,
        act_fp4_rowpair,
        weight_sf,
        act_sf,
        alpha,
        out_dtype,
        mma_tiler_mn=tiler,
        cluster_shape_mn=cluster,
        store_mode="tma",
        kernel_arch=kernel_arch,
    )


def warmup_specs(arch: str, strided=(False, True)) -> list[dict]:
    """Enumerate every static row-pair kernel variant reachable on ``arch``."""
    specs = []
    for _m_max, kernel_arch, tiler, cluster in ARCH_TACTIC_BY_M[arch]:
        for strided_weight in strided:
            specs.append(
                dict(
                    mma_tiler_mn=tiler,
                    cluster_shape_mn=cluster,
                    mode="fold_tma",
                    kernel_arch=kernel_arch,
                    ab_stage_override=None,
                    strided_weight=strided_weight,
                )
            )
    return specs


def warmup_fold_default(
    major: int, minor: int, *, strided=(False, True), out_dtype=None
) -> int:
    """Pre-compile the fold kernels this GPU can reach. Returns the count.

    Call at engine init outside CUDA graph capture.

    Best-effort per spec: one illegal combination must not cost the rest of
    the coverage. Failures are printed, never swallowed -- a warmup that
    silently covers nothing is exactly the state this replaced.
    """
    from .host import _compile_fold_gemm

    arch = kernel_arch_for_capability(major, minor)
    if arch is None:
        return 0
    dtype = out_dtype if out_dtype is not None else torch.bfloat16

    compiled = failed = 0
    for spec in warmup_specs(arch, strided=strided):
        try:
            _compile_fold_gemm(
                spec["mma_tiler_mn"],
                spec["cluster_shape_mn"],
                dtype,
                mode=spec["mode"],
                kernel_arch=spec["kernel_arch"],
                ab_stage_override=spec["ab_stage_override"],
                strided_weight=spec["strided_weight"],
            )
            compiled += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(
                f"[residue] fold warmup: {spec['mode']} "
                f"{spec['mma_tiler_mn']} strided={spec['strided_weight']} "
                f"failed: {type(e).__name__}: {str(e)[:70]}",
                flush=True,
            )
    if failed:
        print(
            f"[residue] fold warmup: {compiled} compiled, {failed} FAILED "
            f"-- those shapes will JIT on first use",
            flush=True,
        )
    return compiled


# Candidate order is part of the persisted FlashInfer tuner cache contract.
FAMILY_TACTICS: dict[tuple[str, str], list[tuple]] = {
    ("row_pair", "std"): [
        # Narrow N tiles reduce padding at decode sizes. Explicit AB depths
        # keep the smallest tiles within the shared-memory limit.
        ("tma", (128, 8), (1, 1), 5),
        ("tma", (128, 16), (1, 1), 6),
        ("tma", (128, 16), (1, 1), 5),
        ("tma", (128, 24), (1, 1)),
        ("tma", (128, 32), (1, 1), 5),
        ("tma", (128, 32), (1, 1)),
        ("tma", (128, 64), (1, 1)),
        ("tma", (128, 128), (1, 1)),
        ("tma", (256, 64), (2, 1)),
        ("tma", (256, 128), (2, 1)),
        ("tma", (256, 256), (2, 1)),
    ],
}


def _norm_tactic(t: tuple) -> tuple[str, tuple[int, int], tuple[int, int], int | None]:
    """Accept 3- or 4-element entries; the 4th is ab_stage (None = auto)."""
    if len(t) == 3:
        return t[0], t[1], t[2], None
    return t[0], t[1], t[2], t[3]


def _rank_tactics(family: tuple[str, str], m_tok: int, kernel_arch: str):
    """Filter legal row-pair candidates and rank by token-axis waste."""
    del kernel_arch
    if family != ("row_pair", "std"):
        raise KeyError(family)
    prob = 2 * m_tok
    scored = []
    for raw in FAMILY_TACTICS[family]:
        mode, tiler, cluster, ab = _norm_tactic(raw)
        tile = tiler[1]
        if tile < 64 and prob > tile:
            continue
        n_tiles = (prob + tile - 1) // tile
        waste = n_tiles * tile - prob
        scored.append(
            ((waste, n_tiles, 0 if ab is None else 1), (mode, tiler, cluster, ab))
        )
    scored.sort(key=lambda kv: kv[0])
    return [t for _, t in scored]
