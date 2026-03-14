# Copyright 2026 SGLang Team
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
"""Python wrapper for the expert-offload CUDA UVM extension (uvm_ops.cu).

The extension is compiled JIT via torch.utils.cpp_extension.load() on first
use, so no separate build step is required.

Public API
----------
uvm_copy_from_tensor(src)
    Allocate managed memory with the same shape/dtype as the GPU tensor `src`,
    copy its data into it, and return the new managed CUDA tensor.

uvm_advise(tensor, advice, device_id)
    Call cudaMemAdvise on `tensor`.  Use the ADVISE_* constants below.
    Pass CUDA_CPU_DEVICE (-1) as device_id to address the CPU.

uvm_prefetch_async(tensor, device_id, stream=None)
    Call cudaMemPrefetchAsync on `tensor`.  Migrates pages to `device_id`
    (or to CPU when device_id == CUDA_CPU_DEVICE).  Defaults to the current
    CUDA stream if `stream` is None.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# cudaMemoryAdvise enum values (from cuda_runtime_api.h)
# ---------------------------------------------------------------------------
ADVISE_SET_PREFERRED_LOCATION = 3  # cudaMemAdviseSetPreferredLocation
ADVISE_SET_ACCESSED_BY = 5  # cudaMemAdviseSetAccessedBy

# cudaCpuDeviceId -- pass as device_id to target the CPU
CUDA_CPU_DEVICE: int = -1

# ---------------------------------------------------------------------------
# JIT extension loading
# ---------------------------------------------------------------------------
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load as cpp_load

        src_path = os.path.join(os.path.dirname(__file__), "uvm_ops.cu")
        _ext = cpp_load(
            name="sglang_expert_uvm",
            sources=[src_path],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            verbose=False,
        )
    return _ext


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def uvm_copy_from_tensor(src: torch.Tensor) -> torch.Tensor:
    """Allocate managed memory and copy *src* (a GPU tensor) into it.

    Returns a new CUDA tensor backed by `cudaMallocManaged`.  The allocation
    is freed automatically when the tensor's reference count reaches zero.
    """
    return _get_ext().uvm_copy_from_tensor(src)


def uvm_advise(tensor: torch.Tensor, advice: int, device_id: int) -> None:
    """Apply ``cudaMemAdvise`` to *tensor* (must be contiguous).

    Parameters
    ----------
    tensor    : a contiguous (possibly sliced) managed-memory tensor.
    advice    : one of the ADVISE_* constants defined in this module.
    device_id : GPU ordinal, or CUDA_CPU_DEVICE (-1) to target the CPU.
    """
    _get_ext().uvm_advise(tensor, advice, device_id)


def uvm_prefetch_async(
    tensor: torch.Tensor,
    device_id: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Asynchronously prefetch *tensor* pages to *device_id*.

    Parameters
    ----------
    tensor    : a contiguous managed-memory tensor.
    device_id : GPU ordinal to migrate pages to, or CUDA_CPU_DEVICE to
                migrate back to CPU DRAM.
    stream    : CUDA stream to enqueue the prefetch on.  Defaults to the
                current stream.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    _get_ext().uvm_prefetch_async(tensor, device_id, stream.cuda_stream)
