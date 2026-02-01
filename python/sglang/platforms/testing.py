# SPDX-License-Identifier: Apache-2.0
"""
Testing utilities for platform abstraction.

Provides MockPlatform and helpers for unit tests that don't require hardware.
"""
from __future__ import annotations

from typing import Any, Callable

from sglang.platforms.interface import Platform, PlatformEnum


class MockPlatform(Platform):
    """
    Mock platform for unit testing.

    Usage:
        platform = MockPlatform()
        platform.register_op("silu_and_mul", lambda x, out: out.copy_(x))

        # Use in tests
        result = platform.silu_and_mul(input_tensor, output_tensor)
    """

    _enum = PlatformEnum.CPU  # Default to CPU-like behavior
    device_name = "mock"
    device_type = "cpu"

    def __init__(self, platform_enum: PlatformEnum = PlatformEnum.CPU):
        self._enum = platform_enum
        self._mock_ops: dict[str, Callable] = {}

    def register_op(self, name: str, impl: Callable) -> None:
        """Register a mock op implementation."""
        self._mock_ops[name] = impl

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._mock_ops:
            return self._mock_ops[name]
        # Return a no-op callable for any unregistered op
        return lambda *args, **kwargs: None


def set_current_platform(platform: Platform) -> None:
    """
    Set the current platform singleton for testing.

    WARNING: Only use in tests! This bypasses normal platform detection.

    Usage:
        from sglang.platforms.testing import set_current_platform, MockPlatform

        def test_my_function():
            mock = MockPlatform()
            mock.register_op("silu_and_mul", my_mock_impl)
            set_current_platform(mock)

            # Test code that uses current_platform
            ...
    """
    import sglang.platforms as platforms_module

    platforms_module._current_platform = platform


def reset_current_platform() -> None:
    """Reset current_platform to trigger re-detection on next access."""
    import sglang.platforms as platforms_module

    platforms_module._current_platform = None


__all__ = [
    "MockPlatform",
    "set_current_platform",
    "reset_current_platform",
]
