# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# Model-agnostic inference-acceleration framework.
#
# A small typed/effect-checked composition layer for efficiency techniques:
#   * Schedule[T]  -- time-varying params (per step/stage), the time sub-DSL.
#   * Technique    -- a composable primitive with a phase + effect set (reads/
#                     writes) + Schedule[bool] enable; OFF == byte-identical.
#   * ModelSpec    -- a model's declaration of structural seams (capabilities).
#   * compose()    -- type-checks capabilities, rejects structural conflicts
#                     (effect system), orders by phase -> an executable Plan.
#   * registry     -- register_technique / register_model_spec / register_transform.
#
# Adapt a new model by writing ONE ModelSpec; reuse every technique. See
# techniques/ for the concrete techniques and models/ for the per-model specs.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.compose import (
    CompositionError,
    Plan,
    check_conflicts,
    compose,
)
from sglang.multimodal_gen.runtime.efficiency.registry import (
    build_technique,
    build_transform,
    get_model_spec,
    is_supported,
    register_model_spec,
    register_technique,
    register_transform,
    registered_models,
    registered_techniques,
    registered_transforms,
)
from sglang.multimodal_gen.runtime.efficiency.schedule import (
    Schedule,
    as_schedule,
    at_steps,
    before,
    by_stage,
    const,
    parse_steps,
    predicate,
)
from sglang.multimodal_gen.runtime.efficiency.spec import ModelSpec
from sglang.multimodal_gen.runtime.efficiency.technique import (
    Capability,
    Phase,
    Seam,
    Technique,
    TechniqueContext,
)
from sglang.multimodal_gen.runtime.efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)

# Register built-in techniques, transforms, and model specs via import
# side-effects. Order matters: technique/transform decorators run when their
# modules are imported, then the model specs are registered last.
from sglang.multimodal_gen.runtime.efficiency import techniques  # noqa: E402,F401
from sglang.multimodal_gen.runtime.efficiency import transforms  # noqa: E402,F401
from sglang.multimodal_gen.runtime.efficiency import models  # noqa: E402,F401

__all__ = [
    # schedule
    "Schedule",
    "as_schedule",
    "at_steps",
    "before",
    "by_stage",
    "const",
    "predicate",
    "parse_steps",
    # technique
    "Technique",
    "TechniqueContext",
    "Phase",
    "Seam",
    "Capability",
    # transform
    "ModelTransform",
    "TransformContext",
    "TransformPhase",
    # spec
    "ModelSpec",
    # compose
    "compose",
    "Plan",
    "CompositionError",
    "check_conflicts",
    # registry
    "register_technique",
    "register_transform",
    "register_model_spec",
    "build_technique",
    "build_transform",
    "get_model_spec",
    "is_supported",
    "registered_techniques",
    "registered_transforms",
    "registered_models",
]
