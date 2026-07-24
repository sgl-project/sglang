# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# Technique -- a composable, model-agnostic inference-acceleration primitive.
#
# A Technique is the framework's "first-class verb": it declares (a) WHEN it is
# active as a Schedule[bool], (b) WHICH execution phase it runs in, and (c) its
# effect set -- the seams it `reads` and `writes`. The phase + effect set are
# what let compose() statically order techniques and reject structural
# conflicts (write-write / write-read on the same seam in the same phase) the
# way a small effect/type system would -- see compose.py.
#
# Each Technique only implements the lifecycle hooks it needs (all default to
# no-ops). A model exposes the structural seams a technique needs via a
# ModelSpec (see spec.py); a technique requires a set of Capabilities and
# compose() refuses to install it on a model that does not provide them.

from __future__ import annotations

import enum
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Hashable

from sglang.multimodal_gen.runtime.efficiency.schedule import Schedule, as_schedule


class Phase(enum.IntEnum):
    """Execution phases, ordered. Lower value runs earlier in a forward step.

    compose() sorts techniques by phase, so a deterministic order exists across
    phases; conflicts are only possible *within* a phase.
    """

    WRAP_ATTENTION = 10  # swap/wrap the attention op (e.g. sparse attention)
    PRE_BLOCKS = 20  # before the transformer-block loop (e.g. token-prune gather)
    IN_BLOCKS = 30  # per-block (e.g. block-level cache decisions)
    POST_BLOCKS = 40  # after the block loop (e.g. token-prune scatter)
    ON_STEP = 50  # whole-step decision (e.g. denoiser-output cache replay)


class Seam(enum.Enum):
    """Named mutation points. A technique/transform's effect set is drawn from
    these. Some seams are EXCLUSIVE (at most one active writer across the whole
    plan -- e.g. only one attention backend, one FFN precision); the rest are
    SHARED (multiple writers compose in phase order)."""

    ATTENTION = "attention"  # attention output values (shared)
    ATTENTION_BACKEND = "attention_backend"  # which attention kernel (EXCLUSIVE)
    TOKEN_SET = "token_set"  # the set/count of tokens in the blocks (EXCLUSIVE)
    HIDDEN_STATES = "hidden_states"  # block hidden-state values (shared)
    KERNEL_FUSION = "kernel_fusion"  # op-fusion of attn/adaLN/FFN kernels (shared)
    RESIDUAL_CACHE = "residual_cache"  # cached block/step residuals (shared)
    STEP_OUTPUT = "step_output"  # denoiser output for the whole step (EXCLUSIVE)
    FFN_PRECISION = "ffn_precision"  # numeric precision of FFN compute (EXCLUSIVE)


# Seams of which at most ONE active writer may exist across the whole plan
# (regardless of phase): two attention backends / two FFN precisions / two
# pruners are genuine conflicts. Shared seams compose in order.
EXCLUSIVE_SEAMS = frozenset(
    {Seam.ATTENTION_BACKEND, Seam.TOKEN_SET, Seam.STEP_OUTPUT, Seam.FFN_PRECISION}
)


class Capability(enum.Enum):
    """Structural seams a model can provide; techniques require a subset."""

    BLOCKS = "blocks"  # an iterable transformer-block list
    PRUNABLE_TOKENS = "prunable_tokens"  # a separable prunable token segment
    RESIDUAL_TUPLE = "residual_tuple"  # block forward returns a residual-compatible tuple
    SWAPPABLE_ATTENTION = "swappable_attention"  # attention goes through the backend layer


@dataclass
class TechniqueContext:
    """Per-step context handed to every hook.

    ``scratch`` is a persistent dict the technique owns across steps (e.g. the
    token-prune 'previous hidden' compensation buffer), keyed however the
    technique likes (commonly by ``cache_key``).
    """

    step: int
    stage: str = ""
    spec: Any = None  # ModelSpec (avoid import cycle)
    cache_key: Hashable = None
    scratch: dict = field(default_factory=dict)


class Technique(ABC):
    """Base class for an acceleration technique.

    Subclasses set the class attributes (name/phase/reads/writes/
    required_capabilities) and override the hook(s) they use. ``enabled`` is a
    Schedule[bool]; OFF must be byte-identical to baseline (the default hooks
    are no-ops, so an inactive technique provably changes nothing).
    """

    name: str = "technique"
    phase: Phase = Phase.PRE_BLOCKS
    reads: frozenset[Seam] = frozenset()
    writes: frozenset[Seam] = frozenset()
    required_capabilities: frozenset[Capability] = frozenset()

    def __init__(self, enabled: "Schedule | bool" = True):
        self.enabled: Schedule = as_schedule(enabled)

    def is_active(self, ctx: TechniqueContext) -> bool:
        return bool(self.enabled.at(ctx.step, ctx.stage))

    # ---- lifecycle hooks (no-op defaults; override what you need) ----

    def before_blocks(self, ctx: TechniqueContext, hidden):
        """Return ``(hidden, carry)``. ``carry`` is opaque state passed back to
        after_blocks (e.g. the kept-token indices)."""
        return hidden, None

    def after_blocks(self, ctx: TechniqueContext, hidden, carry):
        """Return the (possibly restored) hidden states."""
        return hidden

    def wrap_attention(self, ctx: TechniqueContext, attn_fn):
        """Return a (possibly wrapped) attention callable."""
        return attn_fn

    def on_step(self, ctx: TechniqueContext, run_step):
        """Wrap the whole-step compute. ``run_step()`` produces the step output."""
        return run_step()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, phase={self.phase.name})"
