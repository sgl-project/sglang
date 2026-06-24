# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared native acceleration config types.

Phase 1 removed the vendored native CUDA extension tree
(``native/omnidreams_singleview/``) and its JIT loader. Only the config-type
surface remains: ``NativeAccelerationMode`` (valid modes plus ``auto``/``required``
back-compat aliases) and ``NativeAccelerationConfig``. FP8 acceleration now runs
through PyTorch-native primitives (``weight_only_fp8`` today; a ``fp8_compute``
mode via ``torch._scaled_mm`` + ``sageattn3`` is planned for Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger

NativeAccelerationMode = Literal[
    "disabled", "weight_only_fp8", "fp8_compute", "auto", "required"
]
"""Native DiT acceleration policy.

- ``disabled``: run the eager bf16 DiT (default).
- ``weight_only_fp8``: dequantize pre-quantized FP8 weights to bf16 and run the
  eager PyTorch DiT (Ideogram 4 style weight-only FP8).
- ``fp8_compute``: FP8-compute GEMM via ``torch._scaled_mm`` + sage3 attention
  (Phase 2; GPU-gated).
- ``auto`` / ``required``: inert back-compat aliases, mapped to ``disabled`` /
  ``weight_only_fp8`` (see :func:`normalize_native_acceleration_mode`).
"""

_VALID_NATIVE_MODES: tuple[str, ...] = ("disabled", "weight_only_fp8", "fp8_compute")
_NATIVE_MODE_ALIASES: dict[str, str] = {
    "auto": "disabled",
    "required": "weight_only_fp8",
}


def normalize_native_acceleration_mode(mode: str) -> str:
    """Map ``auto``/``required`` back-compat aliases to real modes and validate.

    The native FP8 DiT path (``optimized_dit_forward``) was removed in Phase 1,
    so ``auto`` no longer has a native path to opt into (→ ``disabled``) and
    ``required`` is satisfied by the weight-only FP8 dequant path
    (→ ``weight_only_fp8``). A warning is logged on alias use.
    """
    if mode in _NATIVE_MODE_ALIASES:
        mapped = _NATIVE_MODE_ALIASES[mode]
        logger.warning(
            "native_acceleration mode {!r} is a back-compat alias; mapping to "
            "{!r} (native FP8 DiT removed in Phase 1).",
            mode,
            mapped,
        )
        return mapped
    if mode not in _VALID_NATIVE_MODES:
        raise ValueError(
            f"native_acceleration mode must be one of {_VALID_NATIVE_MODES} "
            f"(or 'auto'/'required' back-compat alias), got {mode!r}"
        )
    return mode


@dataclass(kw_only=True)
class NativeAccelerationConfig:
    """Common native acceleration knobs for pipeline components.

    Retained as a config-type container; the extension-loading helpers that used
    to live here were removed with the native CUDA tree.
    """

    mode: NativeAccelerationMode = "disabled"
    """Native execution policy (see :data:`NativeAccelerationMode`)."""

    build_root: str | None = None
    """Unused after native-tree removal; kept for config-compat."""

    max_jobs: int | str | None = None
    """Unused after native-tree removal; kept for config-compat."""

    verbose_build: bool = False
    """Unused after native-tree removal; kept for config-compat."""

    def __post_init__(self) -> None:
        self.mode = normalize_native_acceleration_mode(self.mode)
