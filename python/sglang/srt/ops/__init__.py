# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
SGLang Operations Registry - Platform-Agnostic Kernel Dispatch
===============================================================

This module is the entry point for **RFC Feature #3: Platform-specific Op / Kernel
Resolution** (see: https://github.com/sgl-project/sglang/issues/15299).

PROBLEM (Before):
-----------------
Previously, platform-specific kernel imports were scattered throughout the codebase:

    # OLD PATTERN - scattered imports and if-else checks everywhere
    _is_cuda = is_cuda()
    _is_npu = is_npu()
    _is_hip = is_hip()
    _is_xpu = is_xpu()

    if _is_cuda or _is_xpu:
        from sgl_kernel import silu_and_mul
    elif _is_hip:
        from sgl_kernel import silu_and_mul, gelu_quick
    # NPU uses a different API entirely...
    if is_npu():
        import torch_npu

This approach has several problems:
1. No IDE autocompletion or type hints for the kernel functions
2. Typos in function names are not detected until runtime
3. Adding a new platform requires modifying many files
4. Hard to track which ops are available on which platform

SOLUTION (After):
-----------------
Now, all operations are imported from this single module:

    # NEW PATTERN - simple, type-safe import
    from sglang.srt.ops import silu_and_mul, gelu_and_mul

    # Full IDE support! Autocomplete, docstrings, type checking
    silu_and_mul(x, out)  # Works on CUDA, ROCm, MUSA, XPU, NPU, CPU...

Benefits:
1. ✅ Full IDE autocompletion and type hints
2. ✅ Typos caught at import time (Python's normal import error)
3. ✅ Adding a new platform = add one file (platforms/new_platform.py)
4. ✅ Central registry of all ops and their availability

ARCHITECTURE:
-------------
    sglang/srt/ops/          # This module - user-facing API
        __init__.py          # Exports all ops (this file)
        activation.py        # Activation ops (silu_and_mul, gelu_and_mul, etc.)
        base.py              # OpProxy base class for advanced use cases

    sglang/srt/platforms/    # Platform implementations
        __init__.py          # Platform detection and current_platform singleton
        interface.py         # Base Platform class
        cuda.py              # NVIDIA CUDA - registers CUDA kernels
        rocm.py              # AMD ROCm/HIP - registers HIP kernels
        musa.py              # Moore Threads MUSA - registers MUSA kernels
        xpu.py               # Intel XPU - registers XPU kernels
        npu.py               # Huawei Ascend NPU - registers NPU kernels
        cpu.py               # CPU fallback
        hpu.py               # Intel Habana HPU

HOW TO ADD A NEW OPERATION:
---------------------------
1. Define the function in the appropriate submodule (e.g., activation.py):
   - Add complete docstring with Args, Returns, and Example
   - Add full type hints
   - Define a native PyTorch fallback

2. Export it in this __init__.py

3. Register in each platform's _init_ops() method (platforms/cuda.py, etc.):
   - Import the platform-specific kernel
   - Add to the ops dictionary: {"op_name": kernel_function}

See activation.py for a complete example of this pattern.
"""

# =============================================================================
# Activation Operations
# =============================================================================
# These fused activation+multiply kernels are used in feed-forward networks:
# - SwiGLU (LLaMA, Mistral, DeepSeek): silu_and_mul
# - GEGLU (various models): gelu_and_mul, gelu_tanh_and_mul

from sglang.srt.ops.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

__all__ = [
    # Activation ops
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
]
