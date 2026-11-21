# SPDX-License-Identifier: Apache-2.0
"""The monokernel contract as the T-Head PPU plugin imports it.

The contract itself (registration, API version, requirements, capabilities
and the build/forward shapes) is shared across monokernel backends and lives
in ``plugin_kernels.amd.spec``, keyed per backend in one registry; this
module gives PPU-side code its documented import path and names its own
backend and flag. ``sglang.kernels.plugin_kernels`` is the layout's
per-hardware surface: a plugin imports here, never another vendor's module.
"""

from sglang.kernels.plugin_kernels.amd.spec import (
    MONOKERNEL_API_VERSION,
    BuildContext,
    BuiltMonokernels,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Plan,
    Requirements,
    ShapeKey,
    StepMetadata,
    find_spec,
    register,
    spec_registered,
)

# the backend the sgl-t-head-mega plugin registers its specs under
BACKEND = "t_head"
FLAG = "--enable-t-head-monokernel"

__all__ = [
    "MONOKERNEL_API_VERSION",
    "BACKEND",
    "FLAG",
    "BuildContext",
    "BuiltMonokernels",
    "DeviceLimits",
    "Feature",
    "MonoSpec",
    "Numerics",
    "NumericsKind",
    "Plan",
    "Requirements",
    "ShapeKey",
    "StepMetadata",
    "find_spec",
    "register",
    "spec_registered",
]
