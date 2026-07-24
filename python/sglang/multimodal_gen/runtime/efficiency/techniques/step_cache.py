# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# StepCache -- whole-step denoiser-output cache: on scheduled steps, skip
# recomputing the denoiser output and reuse / delta-extrapolate the previous
# step's output.
#
# Phase ON_STEP: wraps the per-step compute. writes STEP_OUTPUT (exclusive --
# only one thing may own the step output). enabled schedule selects the SKIP
# steps; step 0 always computes to seed the buffer. OFF (no skip step active) ==
# byte-identical baseline.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_technique
from sglang.multimodal_gen.runtime.efficiency.schedule import as_schedule
from sglang.multimodal_gen.runtime.efficiency.technique import (
    Phase,
    Seam,
    Technique,
    TechniqueContext,
)


@register_technique("step_cache")
class StepCache(Technique):
    """Skip-and-reuse the whole denoiser-output on scheduled steps.

    Parameters
    ----------
    skip : Schedule[bool] | bool -- True on steps whose compute is skipped and
        replaced by the cached (optionally delta-extrapolated) previous output.
    delta_scale : float -- 0.0 reuses the last output verbatim; >0 linearly
        extrapolates using the last computed delta (output_t - output_{t-1}).
    """

    name = "step_cache"
    phase = Phase.ON_STEP
    reads = frozenset({Seam.STEP_OUTPUT})
    writes = frozenset({Seam.STEP_OUTPUT})

    def __init__(self, skip="", delta_scale: float = 0.0, enabled: object = True):
        # the cache is "on" exactly on the skip steps
        super().__init__(enabled=as_schedule(skip) if skip != "" else enabled)
        self.delta_scale = float(delta_scale)

    def on_step(self, ctx: TechniqueContext, run_step):
        key = ("step_cache", ctx.cache_key)
        prev = ctx.scratch.get(key)  # (last_output, last_delta)
        if not self.is_active(ctx) or prev is None:
            out = run_step()  # full compute (and always on the seed step)
            last_out = prev[0] if prev is not None else None
            delta = (out - last_out) if last_out is not None else None
            ctx.scratch[key] = (out.detach() if hasattr(out, "detach") else out, delta)
            return out
        # SKIP this step: reuse / delta-extrapolate the cached output
        last_out, last_delta = prev
        if self.delta_scale and last_delta is not None:
            return last_out + self.delta_scale * last_delta
        return last_out
