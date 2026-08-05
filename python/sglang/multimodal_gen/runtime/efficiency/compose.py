# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# compose() -- the type/effect checker and the executable Plan.
#
# Accepts a mixed list of runtime Techniques and build/load ModelTransforms +
# a ModelSpec, and:
#   1. CAPABILITY CHECK (type check): every item's required_capabilities must be
#      provided by the spec, else refused.
#   2. CONFLICT CHECK (effect system):
#        a. EXCLUSIVE-seam rule (global, across techniques+transforms): an
#           exclusive seam (one attention backend / one FFN precision / one
#           token-set owner / one step-output owner) may have at most ONE active
#           writer. >1 => conflict.
#        b. Same-phase ordering rule (runtime techniques only): within a runtime
#           Phase, a write-read on a SHARED seam between two techniques is
#           order-ambiguous => conflict.
#      Shared seams with multiple writers across phases compose in order (no
#      conflict).
#   3. ORDERING: transforms by TransformPhase, techniques by Phase.
# The Plan applies transforms at build/load and runs technique hooks per step.

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from sglang.multimodal_gen.runtime.efficiency.spec import ModelSpec
from sglang.multimodal_gen.runtime.efficiency.technique import (
    EXCLUSIVE_SEAMS,
    Phase,
    Technique,
    TechniqueContext,
)
from sglang.multimodal_gen.runtime.efficiency.transform import (
    ModelTransform,
    TransformContext,
)


class CompositionError(ValueError):
    """Raised when items cannot be composed on a model (capability or conflict)."""


def _always_on(item) -> bool:
    """A transform is always active; a technique is active per its schedule.

    For conflict analysis a technique with a schedule that is truthy somewhere
    counts as a potential writer."""
    return not isinstance(item, Technique) or bool(item.enabled.truthy_steps())


def _tech_steps_overlap(a: Technique, b: Technique, horizon: int) -> bool:
    return bool(a.enabled.truthy_steps(horizon) & b.enabled.truthy_steps(horizon))


def check_conflicts(items: Sequence, horizon: int = 64) -> list[str]:
    """Return structural-conflict messages (empty == provably clean)."""
    problems: list[str] = []
    active = [it for it in items if _always_on(it)]

    # (a) exclusive-seam rule: at most one active writer per exclusive seam.
    for seam in EXCLUSIVE_SEAMS:
        writers = [it for it in active if seam in getattr(it, "writes", frozenset())]
        clash = []
        for a, b in combinations(writers, 2):
            if isinstance(a, Technique) and isinstance(b, Technique):
                if not _tech_steps_overlap(a, b, horizon):
                    continue
            clash.append((a, b))
        if clash:
            names = sorted({it.name for pair in clash for it in pair})
            problems.append(
                f"exclusive seam {seam.value!r} has multiple active writers: {names}"
            )

    # (b) same-phase write-read ordering rule (runtime techniques only).
    techs = [it for it in active if isinstance(it, Technique)]
    for a, b in combinations(techs, 2):
        if a.phase != b.phase or not _tech_steps_overlap(a, b, horizon):
            continue
        wr = (a.writes & b.reads) | (b.writes & a.reads)
        wr = wr - EXCLUSIVE_SEAMS  # exclusive already handled in (a)
        if wr:
            problems.append(
                f"write-read conflict on {[s.value for s in wr]} between "
                f"{a.name!r} and {b.name!r} in the same phase {a.phase.name}"
            )
    return problems


@dataclass
class Plan:
    """An ordered, conflict-checked set of transforms + techniques for a model.

    apply_transforms(transformer) installs the build/load transforms once;
    before_blocks/after_blocks/wrap_attention run the active runtime techniques.
    """

    transforms: list[ModelTransform]
    techniques: list[Technique]
    spec: ModelSpec

    # ---- build/load: applied once ----
    def apply_transforms(self, transformer, stage: str = "", env=None):
        ctx = (
            TransformContext(stage=stage, spec=self.spec)
            if env is None
            else TransformContext(stage=stage, spec=self.spec, env=env)
        )
        for t in self.transforms:  # already phase-ordered
            transformer = t.apply(transformer, ctx)
        return transformer

    # ---- runtime: per step ----
    def _active(self, ctx: TechniqueContext, phase: Phase) -> list[Technique]:
        return [t for t in self.techniques if t.phase == phase and t.is_active(ctx)]

    def before_blocks(self, ctx: TechniqueContext, hidden):
        carries: list[tuple[Technique, object]] = []
        for t in self._active(ctx, Phase.PRE_BLOCKS):
            hidden, carry = t.before_blocks(ctx, hidden)
            carries.append((t, carry))
        return hidden, carries

    def after_blocks(self, ctx: TechniqueContext, hidden, carries):
        for t, carry in reversed(carries):
            hidden = t.after_blocks(ctx, hidden, carry)
        return hidden

    def wrap_attention(self, ctx: TechniqueContext, attn_fn):
        for t in self._active(ctx, Phase.WRAP_ATTENTION):
            attn_fn = t.wrap_attention(ctx, attn_fn)
        return attn_fn

    def on_step(self, ctx: TechniqueContext, run_step):
        active = self._active(ctx, Phase.ON_STEP)
        if not active:
            return run_step()
        # nest the step-level wrappers (e.g. step cache) around run_step
        fn = run_step
        for t in reversed(active):
            fn = (lambda t_, f_: (lambda: t_.on_step(ctx, f_)))(t, fn)
        return fn()

    def __repr__(self) -> str:
        tr = ", ".join(t.name for t in self.transforms)
        tq = ", ".join(t.name for t in self.techniques)
        return f"Plan(spec={self.spec.name!r}, transforms=[{tr}], techniques=[{tq}])"


def compose(items: Sequence, spec: ModelSpec, horizon: int = 64) -> Plan:
    """Type-check + conflict-check + order a mixed list of Techniques and
    ModelTransforms for ``spec``."""
    techniques = [it for it in items if isinstance(it, Technique)]
    transforms = [it for it in items if isinstance(it, ModelTransform)]
    unknown = [it for it in items if it not in techniques and it not in transforms]
    if unknown:
        raise CompositionError(f"not a Technique or ModelTransform: {unknown}")

    # 1. capability (type) check
    for it in items:
        missing = spec.missing(it.required_capabilities)
        if missing:
            raise CompositionError(
                f"{it.name!r} requires capabilities {[c.value for c in missing]} "
                f"not provided by model {spec.name!r} "
                f"(provided: {[c.value for c in spec.capabilities]})"
            )
    # 2. structural conflict (effect) check
    problems = check_conflicts(items, horizon)
    if problems:
        raise CompositionError("conflicts:\n  - " + "\n  - ".join(problems))
    # 3. deterministic ordering
    transforms.sort(key=lambda t: int(t.phase))
    techniques.sort(key=lambda t: int(t.phase))
    return Plan(transforms, techniques, spec)
