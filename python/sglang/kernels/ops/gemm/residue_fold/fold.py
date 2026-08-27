# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Dispatch and warmup for the SM100/SM103 residue fold GEMM."""

from __future__ import annotations

import contextlib
import sys
import warnings

import torch

_DEFAULT_PATH_WARNED = False


# ── JIT-compile observability ──────────────────────────────────────────────
# Read `sys.modules` rather than importing so observability never loads CUTLASS.
_SM10X_HOST = "sglang.kernels.ops.gemm.residue_fold.cute_fold.host"


def compile_stats() -> dict[str, dict | None]:
    """Fold kernels JIT-compiled by this process so far, per arch.

    Side-effect free: never imports a module that is not already loaded.
    `None` for an arch means its host module has not been imported at all,
    which is the normal state for the arch this GPU is not.
    """
    out: dict[str, dict | None] = {"sm10x": None}

    mod = sys.modules.get(_SM10X_HOST)
    if mod is not None:
        cache = getattr(mod, "_KERNEL_CACHE", None)
        if cache is not None:
            out["sm10x"] = {"compiled": len(cache), "keys": list(cache)}

    return out


def warmup(major: int, minor: int, shapes=None) -> int:
    """Pre-compile whatever this GPU's fold path can need. Returns the count.

    The compile key has no shape or token count, so ``shapes`` is ignored.
    Call this at engine init, before CUDA graph capture.
    """
    from .cute_fold import tactics as _t

    return _t.warmup_fold_default(major, minor)


@contextlib.contextmanager
def observe_compiles(where: str, *, warn: bool = True):
    """Report any fold kernel JIT-compiled inside this block.

    Wrap the regions that must already be warm -- cudagraph capture above all,
    where a JIT is both slow and, for the sm10x path, a sync inside capture.
    A clean block prints nothing; the whole point is that the silent case
    stays silent and the broken case does not.
    """
    before = compile_stats()
    try:
        yield
    finally:
        after = compile_stats()
        for arch in ("sm10x",):
            b, a = before.get(arch), after.get(arch)
            if not a:
                continue
            n = a["compiled"] - (b["compiled"] if b else 0)
            if n <= 0:
                continue
            detail = ""
            if a["keys"] is not None:
                fresh = a["keys"][len(b["keys"]) if b else 0 :]
                detail = "".join(f"\n[residue]     {k}" for k in fresh[:8])
                if len(fresh) > 8:
                    detail += f"\n[residue]     ... and {len(fresh) - 8} more"
            if warn:
                print(
                    f"[residue] WARMUP UNDER-COVERAGE: {n} {arch} fold "
                    f"kernel(s) JIT-compiled during {where}. Each costs ~1.3 s "
                    f"and blocks the caller. The pre-compile missed:{detail}",
                    flush=True,
                )


def warn_default_path(arch: str, m_tok: int, detail: str = "") -> None:
    """Announce, once per process, that the untuned sm10x table is serving.

    This warning means a tuned path exists but the static fallback is serving.

    Once, not per call: this sits in the decode path and the message is a
    build-configuration fact, not a per-request event.
    """
    global _DEFAULT_PATH_WARNED
    if _DEFAULT_PATH_WARNED:
        return
    _DEFAULT_PATH_WARNED = True
    warnings.warn(
        f"[residue] MExt fold is running on the STATIC tactic table "
        f"(arch={arch}, m_tok={m_tok}{', ' + detail if detail else ''}). "
        "On sm10x that table is a correctness fallback, not the tuned path: "
        "it takes (128,64) for every m_tok<=64, which at m_tok=8 wastes three "
        "quarters of the token axis and measured ~19% slower than the "
        "exact-fit tile. Enable and precompile the FlashInfer fold tuner at "
        "engine init, or set SGLANG_RESIDUE_FOLD_SILENCE_DEFAULT=1 if the "
        "fallback is intentional here.",
        RuntimeWarning,
        stacklevel=3,
    )


def run_fold(
    major: int,
    minor: int,
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the SM100/SM103 row-pair fold GEMM via the static fallback."""
    from .cute_fold import tactics as _t

    return _t.run_fold_default(
        major, minor, weight_fp4, act_fp4_rowpair, weight_sf, act_sf, alpha, out_dtype
    )
