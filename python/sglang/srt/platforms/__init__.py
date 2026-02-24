# SPDX-License-Identifier: Apache-2.0
"""
SRT Platform Re-exports.

This module re-exports the unified platform abstraction from sglang.platforms
for backward compatibility with SRT code that imports from sglang.srt.platforms.

Usage:
    # Old import (still works):
    from sglang.srt.platforms import current_platform

    # New preferred import:
    from sglang.platforms import current_platform
"""

# Re-export everything from the top-level platforms module
from sglang.platforms import (
    AttentionBackendEnum,
    CpuArchEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
    UnspecifiedPlatform,
    _init_trace,
    current_platform,
)

__all__ = [
    "Platform",
    "PlatformEnum",
    "DeviceCapability",
    "CpuArchEnum",
    "UnspecifiedPlatform",
    "AttentionBackendEnum",
    "current_platform",
    "_init_trace",
]
