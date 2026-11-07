# SPDX-License-Identifier: Apache-2.0
"""The Intel XPU monokernels use the shared monokernel contract
(``sglang.kernels.plugin_kernels.amd.spec``); a plugin registers its specs under
this backend name, and --enable-xpu-monokernel selects them."""

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

# the backend sgl-kernel-xpu registers its specs under
BACKEND = "xpu"
FLAG = "--enable-xpu-monokernel"

# the contract, re-exported for XPU plugins
__all__ = [
    "BACKEND",
    "FLAG",
    "MONOKERNEL_API_VERSION",
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
