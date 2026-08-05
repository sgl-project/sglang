# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# TeaCache -- timestep-embedding-similarity step cache: skip recomputing the
# denoiser when the accumulated, polynomial-rescaled L1 distance of the
# timestep-conditioned "modulated input" stays below a threshold, and reuse the
# cached output instead.
#
# The GENERIC core (distance accumulation + rescale + skip decision) lives
# here; the MODEL-SPECIFIC seam is the "modulated input" signal each step. The
# model stashes that signal into ctx.scratch[("teacache_signal", cache_key)]
# before the step compute; the polynomial ``coefficients`` are a
# model-calibrated rescale (identity by default).
#
# Phase ON_STEP, writes STEP_OUTPUT (exclusive -> conflicts with StepCache,
# which is correct: you don't run two whole-step caches at once). enabled OFF
# / before start_step / no signal -> always compute == byte-identical baseline.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_technique
from sglang.multimodal_gen.runtime.efficiency.technique import (
    Phase,
    Seam,
    Technique,
    TechniqueContext,
)


@register_technique("teacache")
class TeaCache(Technique):
    """Reuse the denoiser output while the rescaled cumulative timestep-embedding
    distance stays under ``threshold``.

    Parameters
    ----------
    threshold : accumulate rescaled rel-L1 distance; reuse while < threshold.
    start_step : always compute before this step (warmup / seed).
    coefficients : polynomial (highest-degree first) rescaling rel-L1 -> the
        TeaCache "indicator"; model-calibrated. Default [1, 0] (identity).
    max_continuous_hits : cap consecutive reuses (1 = never skip two in a row).
    periodic_recompute : force a recompute every N steps (0 = off).
    """

    name = "teacache"
    phase = Phase.ON_STEP
    reads = frozenset({Seam.STEP_OUTPUT})
    writes = frozenset({Seam.STEP_OUTPUT})

    def __init__(
        self,
        threshold: float = 0.04,
        start_step: int = 6,
        coefficients=None,
        max_continuous_hits: int = 1,
        periodic_recompute: int = 0,
        enabled: object = True,
    ):
        super().__init__(enabled=enabled)
        self.threshold = float(threshold)
        self.start_step = int(start_step)
        self.coefficients = list(coefficients) if coefficients else [1.0, 0.0]
        self.max_continuous_hits = int(max_continuous_hits)
        self.periodic_recompute = int(periodic_recompute)

    def _rescale(self, x: float) -> float:
        # Horner evaluation of the (highest-degree-first) polynomial.
        y = 0.0
        for c in self.coefficients:
            y = y * x + c
        return y

    def on_step(self, ctx: TechniqueContext, run_step):
        key = ("teacache", ctx.cache_key)
        st = ctx.scratch.get(key) or {
            "prev": None,
            "acc": 0.0,
            "hits": 0,
            "out": None,
            "since": 0,
        }
        modulated = ctx.scratch.get(("teacache_signal", ctx.cache_key))

        force = (
            not self.is_active(ctx)
            or modulated is None
            or ctx.step < self.start_step
            or st["prev"] is None
            or (self.periodic_recompute and st["since"] >= self.periodic_recompute)
        )
        reuse = False
        if not force:
            num = float((modulated - st["prev"]).abs().mean())
            den = max(float(st["prev"].abs().mean()), 1e-8)
            st["acc"] += self._rescale(num / den)
            reuse = st["acc"] < self.threshold and st["hits"] < self.max_continuous_hits

        if modulated is not None:
            st["prev"] = modulated.detach() if hasattr(modulated, "detach") else modulated

        if reuse and st["out"] is not None:
            st["hits"] += 1
            st["since"] += 1
            ctx.scratch[key] = st
            return st["out"]

        out = run_step()  # full compute
        st["out"] = out.detach() if hasattr(out, "detach") else out
        st["acc"] = 0.0
        st["hits"] = 0
        st["since"] = 0
        ctx.scratch[key] = st
        return out
