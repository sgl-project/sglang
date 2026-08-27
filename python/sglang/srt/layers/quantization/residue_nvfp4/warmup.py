"""Fold warmup: pre-compile the residue fold kernels before CUDA graph capture.

The CuTeDSL fold JIT-compiles. A JIT inside cudagraph capture is both slow
(~1.3 s per kernel, multiplied by capture sizes) and, on some paths, a sync in
a place that cannot take one. The two halves of the defense:

  1. Every fold-eligible layer registers its shape at the end of its
     process_weights_after_loading -- PWAL always runs, always before capture,
     and the layer knows its own shape there. Registering from a runner hook
     was tried in the reference implementation and failed silently twice.
  2. capture setup calls maybe_warmup_residue_fold() before the first capture
     and wraps the capture in observe_residue_fold_compiles(), which turns
     warmup under-coverage from "the server is mysteriously slow" into a
     printed report naming the missed kernels.

The SM100/SM103 kernels compile per tactic rather than per shape, so one
warmup covers every registered layer.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# (n_w, k_base, k_ext_in_fp4_elements, is_mext_r1)
_FOLD_SHAPES: set[tuple[int, int, int, bool]] = set()
_WARMED_UP = False


def register_fold_shape(n_w: int, k_base: int, k_ext: int, is_mext_r1: bool) -> None:
    """Record a fold-eligible layer's shape for the pre-capture warmup.

    k_ext is the STORED weight width in FP4 elements: == k_base for a
    contiguous mext_r1 weight, > k_base for an ext-K prefix view (which
    selects a different compiled kernel via the strided weight read).
    """
    _FOLD_SHAPES.add((int(n_w), int(k_base), int(k_ext), bool(is_mext_r1)))


def registered_fold_shapes() -> frozenset[tuple[int, int, int, bool]]:
    return frozenset(_FOLD_SHAPES)


def maybe_warmup_residue_fold() -> int:
    """Compile every fold kernel the registered layers can reach.

    Returns the number of kernels compiled; 0 when no residue layer is
    loaded (the common case: the whole function is a set-emptiness check).
    Call at engine init, OUTSIDE cudagraph capture. Idempotent.

    Also pre-compiles the FlashInfer tuner candidates: a persisted autotune
    cache entry restored on a later boot must find its winning kernel
    already compiled, or the first serving call would JIT inside cudagraph
    capture. Best-effort per spec -- one failure must not cost the rest.
    """
    global _WARMED_UP
    if not _FOLD_SHAPES or _WARMED_UP:
        return 0
    _WARMED_UP = True

    from sglang.kernels.ops.gemm.residue_fold import warmup

    major, minor = torch.cuda.get_device_capability()
    shapes = sorted(_FOLD_SHAPES)
    count = warmup(int(major), int(minor), shapes=shapes)
    count += _precompile_tuner_candidates(int(major), shapes)
    logger.info(
        "Residue fold warmup: %d kernel(s) compiled for %d registered "
        "shape(s) on sm%d%d.",
        count,
        len(shapes),
        major,
        minor,
    )
    return count


def _precompile_tuner_candidates(major: int, shapes) -> int:
    try:
        from sglang.kernels.ops.gemm.residue_fold import tuners
    except ImportError as e:
        logger.info("Residue fold tuner precompile skipped: %s", e)
        return 0

    compiled = 0
    has_mext_r1 = any(is_mext for (_, _, _, is_mext) in shapes)
    try:
        if major == 10:
            if tuners.sm10x_tuner_enabled():
                # Symbolic shapes: one candidate set serves every layer.
                if tuners.precompile_row_pair():
                    compiled += 1
            if has_mext_r1:
                compiled += tuners.precompile_kloop_sm100()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Residue fold tuner precompile failed (%s: %s); cache hits may "
            "degrade to the fallback path.",
            type(e).__name__,
            e,
        )
    return compiled


def observe_residue_fold_compiles(where: str):
    """Context manager reporting any fold JIT that happens inside the block.

    No-ops (cheaply) when no residue layer is loaded. Wrap cudagraph capture
    with it: a clean block prints nothing; a missed warmup names the keys.
    """
    if not _FOLD_SHAPES:
        import contextlib

        return contextlib.nullcontext()

    from sglang.kernels.ops.gemm.residue_fold import observe_compiles

    return observe_compiles(where)
