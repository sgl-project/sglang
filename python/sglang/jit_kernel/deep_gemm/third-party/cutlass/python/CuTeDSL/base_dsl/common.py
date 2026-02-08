# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

import os
from typing import Any, Dict, Iterable, Optional, Union

"""
This module provides a Exception classes DSL class for any Dialect.
"""


# Add color codes at the top of the file after imports
class Colors:
    """ANSI color codes for error messages"""

    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# =============================================================================
# DSL Exceptions
# =============================================================================


class DSLBaseError(Exception):
    """
    Base exception for DSL-related errors.
    Provides optional contextual metadata to aid in debugging.
    """

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        snippet: Optional[str] = None,
        filename: Optional[str] = None,
        error_code: Optional[Union[str, int]] = None,
        context: Optional[Union[Dict[str, Any], str]] = None,
        suggestion: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.message = message
        self.line = line
        self.filename = filename
        self.snippet = snippet
        self.error_code = error_code
        self.context = context
        self.suggestion = suggestion
        self.cause = cause

        super().__init__(self._format_message())

    def _format_message(self):
        """
        Formats the complete error message with available metadata.
        Override this in subclasses if you want to change formatting logic.
        """
        parts = [f"{self.__class__.__name__}: {self.message}"]

        if self.error_code is not None:
            parts.append(f"{Colors.BOLD}Error Code:{Colors.RESET} {self.error_code}\n")

        if self.line is not None:
            parts.append(f"  Line: {self.line}")

        if self.filename is not None:
            parts.append(f"  File: {self.filename}")

        if self.snippet:
            # Optionally truncate long snippets for readability
            parts.append(f"  Snippet: \n {self.snippet}")

        if self.cause:
            parts.append(f"  Caused exception: {self.cause}")

        if self.context:
            if isinstance(self.context, dict):
                parts.append(f"{Colors.BLUE}🔍 Additional Context:{Colors.RESET}\n")
                for key, value in self.context.items():
                    parts.append(f"    {key}: {value}")
            else:
                parts.append(
                    f"{Colors.BLUE}🔍 Additional Context:{Colors.RESET} {self.context}"
                )

        if self.suggestion:
            parts.append(f"{Colors.GREEN}💡 Suggestions:{Colors.RESET}")
            if isinstance(self.suggestion, (list, tuple)):
                for suggestion in self.suggestion:
                    parts.append(f" {Colors.GREEN}{suggestion}{Colors.RESET}")
            else:
                parts.append(f" {self.suggestion}")

        return "\n".join(parts)


class DSLRuntimeError(DSLBaseError):
    """
    Raised when an error occurs during JIT-time code generation in the DSL.
    """

    # Inherits all logic from DSLBaseError; override methods if you need
    # specialized behavior or formatting for runtime errors.
    pass


def _get_friendly_cuda_error_message(error_code, error_name):
    # Avoid circular dependency
    from .runtime.cuda import get_device_info

    """Get a user-friendly error message for common CUDA errors."""
    # Strip the byte string markers if present
    if isinstance(error_name, bytes):
        error_name = error_name.decode("utf-8")
    elif (
        isinstance(error_name, str)
        and error_name.startswith("b'")
        and error_name.endswith("'")
    ):
        error_name = error_name[2:-1]

    # Add target architecture info
    target_arch = os.getenv("CUTE_DSL_ARCH", "unknown")

    error_messages = {
        "CUDA_ERROR_INVALID_SOURCE": (
            f"{Colors.RED}❌ Failed to load CUDA kernel - likely architecture mismatch.{Colors.RESET}\n\n"
        ),
        "CUDA_ERROR_NO_BINARY_FOR_GPU": (
            f"{Colors.RED}❌ CUDA kernel not compatible with your GPU.{Colors.RESET}\n\n"
        ),
        "CUDA_ERROR_OUT_OF_MEMORY": (
            f"{Colors.RED}💾 CUDA out of memory error.{Colors.RESET}\n\n"
        ),
        "CUDA_ERROR_INVALID_DEVICE": (
            f"{Colors.RED}❌ Invalid CUDA device.{Colors.RESET}\n\n"
        ),
        "CUDA_ERROR_NOT_INITIALIZED": (
            f"{Colors.RED}❌ CUDA context not initialized.{Colors.RESET}\n\n"
        ),
        "CUDA_ERROR_INVALID_VALUE": (
            f"{Colors.RED}⚠️ Invalid parameter passed to CUDA operation.{Colors.RESET}\n\n"
            f"{Colors.YELLOW}This is likely a bug - please report it with:{Colors.RESET}"
        ),
    }

    error_suggestions = {
        "CUDA_ERROR_INVALID_SOURCE": (
            f"1. Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            f"2. Clear the compilation cache and regenerate the kernel",
            f"3. Check CUDA toolkit installation",
        ),
        "CUDA_ERROR_NO_BINARY_FOR_GPU": (
            f"Set env CUTE_DSL_ARCH to match your GPU architecture",
        ),
        "CUDA_ERROR_OUT_OF_MEMORY": (
            f"1. Reduce batch size",
            f"2. Reduce model size",
            f"3. Free unused GPU memory",
        ),
        "CUDA_ERROR_INVALID_DEVICE": (
            f"1. Check if CUDA device is properly initialized",
            f"2. Verify GPU is detected: nvidia-smi",
            f"3. Check CUDA_VISIBLE_DEVICES environment variable",
        ),
        "CUDA_ERROR_NOT_INITIALIZED": (
            f"1. Check CUDA driver installation",
            f"2. call `cuda.cuInit(0)` before any other CUDA operation",
            f"3. Run nvidia-smi to confirm GPU status",
        ),
        "CUDA_ERROR_INVALID_VALUE": (
            f"1. Your GPU model",
            f"2. SM ARCH setting",
            f"3. Steps to reproduce",
        ),
    }

    message = error_messages.get(
        error_name, f"{Colors.RED}Unknown CUDA error{Colors.RESET}"
    )

    # Add debug information
    debug_info = f"\n- {Colors.BOLD}Error name: {error_name}\n"
    debug_info += f"- CUDA_TOOLKIT_PATH: {os.getenv('CUDA_TOOLKIT_PATH', 'not set')}\n"
    debug_info += (
        f"- Target SM ARCH: {os.getenv('CUTE_DSL_ARCH', 'not set')}{Colors.RESET}\n"
    )

    try:
        # Get GPU information using CUDA Python API
        debug_info += f"\n{Colors.BLUE}📊 GPU Information:{Colors.RESET}\n"
        gpu_info = get_device_info()
        debug_info += gpu_info.pretty_str()

        if target_arch and gpu_info.compatible_archs:
            debug_info += f"\n{Colors.BOLD}Compatibility Check:{Colors.RESET}\n"

            if target_arch not in gpu_info.compatible_archs:
                debug_info += (
                    f"{Colors.RED}❌ Error: Target SM ARCH {target_arch} is not compatible\n"
                    f"💡 Please use one of SM ARCHs: "
                    f"{Colors.GREEN}{', '.join(gpu_info.compatible_archs or [])}{Colors.RESET}\n"
                )
            elif target_arch != gpu_info.sm_arch:
                debug_info += (
                    f"{Colors.YELLOW}⚠️  Warning: Using compatible but non-optimal architecture\n"
                    f"• Current: {target_arch}\n"
                    f"• Recommended: {Colors.GREEN}{gpu_info.sm_arch}{Colors.RESET} (native)\n"
                )
            else:
                debug_info += f"{Colors.GREEN}✓ Using optimal architecture: {gpu_info.sm_arch}{Colors.RESET}\n"

    except Exception as e:
        debug_info += (
            f"\n{Colors.YELLOW}ℹ️  Could not retrieve GPU info: {str(e)}{Colors.RESET}"
        )

    return message, debug_info, error_suggestions.get(error_name, "")


class DSLCudaRuntimeError(DSLBaseError):
    """
    Raised when an error occurs during CUDA runtime code generation in the DSL.
    """

    # Inherits all logic from DSLRuntimeError; override methods if you need
    # specialized behavior or formatting for runtime errors.
    def __init__(self, error_code, error_name) -> None:
        self._error_code = error_code
        self._error_name = error_name
        message, debug_info, suggestion = _get_friendly_cuda_error_message(
            error_code, error_name
        )

        super().__init__(
            message, error_code=error_code, context=debug_info, suggestion=suggestion
        )


class DSLAstPreprocessorError(DSLBaseError):
    """
    Raised when an error occurs during AST preprocessing or visiting in the DSL.
    """

    # Same approach: You could override _format_message if you want
    # to emphasize AST node details or anything specific to preprocessing.
    pass


class DSLNotImplemented(DSLBaseError):
    """
    Raised when a feature of the DSL is not implemented yet.
    """

    # Useful for stubs in your DSL that you plan to implement in the future.
    pass
