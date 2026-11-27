# SPDX-License-Identifier: Apache-2.0
"""Intel XPU entry point for the shared monokernel registry."""

from sglang.kernels.plugin_kernels.amd.spec import register

BACKEND = "xpu"
FLAG = "--enable-xpu-monokernel"

__all__ = ["BACKEND", "FLAG", "register"]
