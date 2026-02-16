# SPDX-License-Identifier: Apache-2.0
"""
ROCm Platform Re-exports (Backward Compatibility).

This module re-exports the unified ROCm platform from sglang.platforms.rocm
for backward compatibility. Existing imports from this module will continue
to work, but new code should import from sglang.platforms.rocm directly.

Usage:
    # Old import (still works):
    from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform

    # New preferred import:
    from sglang.platforms.rocm import RocmPlatform
"""

# Re-export from unified platform module
from sglang.platforms.rocm import RocmPlatform

__all__ = ["RocmPlatform"]
