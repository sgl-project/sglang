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

"""Shared native acceleration policy helpers.

The DiT and VAE native ports should share CUDA-extension selection semantics:

- ``disabled`` never loads native code.
- ``auto`` uses native code only when it is available and compatible.
- ``required`` raises instead of continuing without native code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Literal, Protocol, get_args

from loguru import logger

NativeAccelerationMode = Literal["auto", "disabled", "required"]
NATIVE_EXTENSION_SYNC_COMMAND = (
    "uv run --package flashdreams-omnidreams python "
    "integrations/omnidreams/omnidreams_singleview/tools/sync_thirdparty.py sync"
)


class NativeExtensionLoader(Protocol):
    def __call__(
        self,
        *,
        build_root: str | None = None,
        max_jobs: int | str | None = None,
        verbose: bool = False,
    ) -> ModuleType | None: ...


NativeAvailabilityCheck = Callable[[ModuleType], bool | tuple[bool, str]]


class NativeAccelerationUnavailable(RuntimeError):
    """Raised when native execution is required but unavailable."""


@dataclass(kw_only=True)
class NativeAccelerationConfig:
    """Common native acceleration knobs for pipeline components."""

    mode: NativeAccelerationMode = "auto"
    """Native execution policy: ``auto``, ``disabled``, or ``required``."""

    build_root: str | None = None
    """Optional native build/cache root forwarded to the extension loader."""

    max_jobs: int | str | None = None
    """Optional PyTorch/Ninja job cap forwarded to the extension loader."""

    verbose_build: bool = False
    """Forward verbose build output from the extension loader."""

    def __post_init__(self) -> None:
        if self.mode not in get_args(NativeAccelerationMode):
            raise ValueError(
                f"mode must be one of {get_args(NativeAccelerationMode)}, "
                f"got {self.mode!r}"
            )


@dataclass(frozen=True)
class NativeBackendSelection:
    """Resolved native backend choice for one pipeline component."""

    component: str
    mode: NativeAccelerationMode
    enabled: bool
    reason: str
    extension: ModuleType | None = field(default=None, repr=False, compare=False)
    error: Exception | None = field(default=None, repr=False, compare=False)

    def require_extension(self) -> ModuleType:
        """Return the loaded extension or raise with this selection's reason."""

        if self.enabled and self.extension is not None:
            return self.extension
        raise NativeAccelerationUnavailable(self.reason)


def require_extension_symbols(
    *symbols: str,
) -> Callable[[ModuleType], tuple[bool, str]]:
    """Return an availability check for extension symbols needed by a component."""

    def check(extension: ModuleType) -> tuple[bool, str]:
        missing = tuple(symbol for symbol in symbols if not hasattr(extension, symbol))
        if missing:
            return False, "missing native symbol(s): " + ", ".join(missing)
        return True, "required native symbols are available"

    return check


def _native_extension_unavailable_reason(
    component: str,
    error: BaseException | None = None,
) -> str:
    base = f"native extension unavailable for {component}"
    if error is not None:
        base = f"{base}: {error}"
    return (
        f"{base}. To sync third-party native sources, run:\n"
        f"  {NATIVE_EXTENSION_SYNC_COMMAND}"
    )


def select_native_extension(
    config: NativeAccelerationConfig,
    *,
    component: str,
    extension_loader: NativeExtensionLoader,
    extension_error: Callable[[], Exception | None],
    availability_check: NativeAvailabilityCheck | None = None,
) -> NativeBackendSelection:
    """Resolve native use for one component."""

    if config.mode == "disabled":
        return NativeBackendSelection(
            component=component,
            mode=config.mode,
            enabled=False,
            reason=f"native acceleration disabled for {component}",
        )

    try:
        extension = extension_loader(
            build_root=config.build_root,
            max_jobs=config.max_jobs,
            verbose=config.verbose_build,
        )
    except Exception as exc:
        return _unavailable_or_raise(config, component, str(exc), error=exc)

    if extension is None:
        error = extension_error()
        reason = _native_extension_unavailable_reason(component, error)
        logger.warning("[native] {}", reason)
        return _unavailable_or_raise(config, component, reason, error=error)

    check = availability_check or _default_availability_check
    try:
        ok, reason = _normalize_availability_result(check(extension))
    except Exception as exc:
        return _unavailable_or_raise(config, component, str(exc), error=exc)
    if not ok:
        return _unavailable_or_raise(config, component, reason)

    return NativeBackendSelection(
        component=component,
        mode=config.mode,
        enabled=True,
        reason=reason,
        extension=extension,
    )


def _unavailable_or_raise(
    config: NativeAccelerationConfig,
    component: str,
    reason: str,
    *,
    error: Exception | None = None,
) -> NativeBackendSelection:
    if config.mode == "required":
        raise NativeAccelerationUnavailable(reason) from error
    return NativeBackendSelection(
        component=component,
        mode=config.mode,
        enabled=False,
        reason=reason,
        error=error,
    )


def _default_availability_check(extension: ModuleType) -> bool | tuple[bool, str]:
    is_available = getattr(extension, "is_available", None)
    if is_available is None:
        return True, "native extension loaded"
    if not callable(is_available):
        return False, "native extension is_available is not callable"
    if is_available():
        return True, "native extension is_available returned true"
    return False, "native extension is_available returned false"


def _normalize_availability_result(
    result: bool | tuple[bool, str],
) -> tuple[bool, str]:
    if isinstance(result, tuple):
        ok, reason = result
        return ok, reason
    if result:
        return True, "native extension is available"
    return False, "native extension is not available"
