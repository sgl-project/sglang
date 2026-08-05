# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# ModelTransform -- a build/load-time model transformation (the second plugin
# category, complementing the runtime Technique).
#
# Unlike a Technique (which inserts hooks into the per-step data flow), a
# ModelTransform is applied ONCE -- at weight load (LOAD) or module construction
# (BUILD) -- and then the model just runs. It is the right home for things that
# install a kernel / quantize weights rather than decide something each step:
#   * attention-backend selection (PISA)   -- BUILD
#   * NVFP4 FFN weight quantization         -- LOAD
#   * KWL operator-fusion configuration     -- BUILD
#
# A transform may still carry Schedule params (e.g. PISA's per-step sparsity):
# the installed kernel reads the schedule at runtime. The transform's job is
# only to *install* it.
#
# These transforms delegate to the EXISTING mechanisms (they set the env/config
# the current load/build code already reads) -- the framework is the unified
# declaration + compose/validate layer, it does not reimplement NVFP4/KWL/PISA.

from __future__ import annotations

import enum
import os
from abc import ABC
from dataclasses import dataclass, field

from sglang.multimodal_gen.runtime.efficiency.technique import Capability, Seam


class TransformPhase(enum.IntEnum):
    """When a transform is applied. Lower runs earlier."""

    LOAD = 10  # at weight load (e.g. quantize FFN to NVFP4)
    BUILD = 20  # at module construction (e.g. install attention backend, fusions)


@dataclass
class TransformContext:
    """Context for apply(). ``env`` is the dict a transform mutates (defaults to
    os.environ); tests pass a throwaway dict to inspect what would be set."""

    stage: str = ""
    spec: object = None
    env: dict = field(default_factory=lambda: os.environ)


class ModelTransform(ABC):
    """Base class for a build/load-time model transformation."""

    name: str = "transform"
    phase: TransformPhase = TransformPhase.BUILD
    writes: frozenset[Seam] = frozenset()
    required_capabilities: frozenset[Capability] = frozenset()

    def applies_to(self, spec) -> bool:
        """Whether this transform actually affects ``spec``'s model.

        Generic transforms (e.g. attention-backend selection) return True for
        any model. A transform that triggers a MODEL-SPECIFIC mechanism (e.g.
        env flags only some models' build/load code reads) overrides this so
        compose() can WARN/skip instead of silently no-op'ing on an unsupported
        model. Default True."""
        return True

    def apply(self, transformer, ctx: TransformContext):
        """Install the transform. Default sets env via set_env(); override to
        wrap/replace modules directly. Returns the (possibly wrapped) module."""
        self.set_env(ctx)
        return transformer

    def set_env(self, ctx: TransformContext) -> None:
        """Set the env/config the existing mechanism reads. Override per transform."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, phase={self.phase.name})"
