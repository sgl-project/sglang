# Copyright 2025 SGLang Team
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
Standalone CUDA VMM (Virtual Memory Management) allocator for MNNVL-compatible
KV cache allocation.

Uses cuMemCreate with CU_MEM_HANDLE_TYPE_FABRIC so the allocated pages are
addressable over the NVLink fabric (MNNVL), enabling zero-copy KV cache
transfers between nodes via UCX or other MNNVL-aware transports.

Activated by setting: SGLANG_NIXL_VMM_MEM_POOL=1
"""

import logging
import os
import tempfile
from typing import Optional, Tuple

import torch
from torch.cuda.memory import CUDAPluggableAllocator

logger = logging.getLogger(__name__)

# C++ source for the VMM allocator compiled at first use via
# torch.utils.cpp_extension.load_inline.
_VMM_ALLOCATOR_SOURCE = r"""
#include <cuda.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

extern "C" {

#define VMM_MAX_DEVICES 128

static size_t g_granularity[VMM_MAX_DEVICES];
static int    g_initialized[VMM_MAX_DEVICES];

/*
 * Query and cache the allocation granularity for a device.
 * Tries CU_MEM_HANDLE_TYPE_FABRIC first (required for MNNVL / NVLink fabric),
 * falls back to CU_MEM_HANDLE_TYPE_NONE if FABRIC is not supported.
 */
static size_t vmm_get_granularity(int device) {
    if (device < 0 || device >= VMM_MAX_DEVICES) {
        return 2UL * 1024 * 1024;  /* safe fallback for out-of-range device */
    }
    if (!g_initialized[device]) {
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.type                   = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type          = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id            = device;
        prop.requestedHandleTypes   = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t gran = 0;
        CUresult r = cuMemGetAllocationGranularity(
            &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        if (r != CUDA_SUCCESS || gran == 0) {
            /* FABRIC not supported -- fall back to no handle type */
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
            r = cuMemGetAllocationGranularity(
                &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        }

        g_granularity[device] = (r == CUDA_SUCCESS && gran > 0)
                                 ? gran
                                 : (2UL * 1024 * 1024);  /* 2 MB fallback */
        g_initialized[device] = 1;
    }
    return g_granularity[device];
}

static size_t vmm_align_up(size_t size, size_t gran) {
    return ((size + gran - 1) / gran) * gran;
}

/*
 * Allocate size bytes of VMM-backed GPU memory on device.
 * Physical pages are created with CU_MEM_HANDLE_TYPE_FABRIC when the
 * platform supports it, making them accessible via the NVLink fabric.
 */
void* nixl_vmm_alloc(size_t size, int device, void* stream) {
    CUresult r;
    size_t gran       = vmm_get_granularity(device);
    size_t alloc_size = vmm_align_up(size, gran);

    /* 1. Reserve virtual address space */
    CUdeviceptr ptr = 0;
    r = cuMemAddressReserve(&ptr, alloc_size, gran, 0, 0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[nixl_vmm] cuMemAddressReserve failed: %d (size=%zu)\n",
                r, alloc_size);
        return NULL;
    }

    /* 2. Create physical allocation -- prefer FABRIC handle for MNNVL */
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    CUmemGenericAllocationHandle handle;
    r = cuMemCreate(&handle, alloc_size, &prop, 0);
    if (r != CUDA_SUCCESS) {
        /* FABRIC not available -- fall back */
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
        r = cuMemCreate(&handle, alloc_size, &prop, 0);
        if (r != CUDA_SUCCESS) {
            fprintf(stderr, "[nixl_vmm] cuMemCreate failed: %d\n", r);
            cuMemAddressFree(ptr, alloc_size);
            return NULL;
        }
    }

    /* 3. Map physical pages into the reserved VA range */
    r = cuMemMap(ptr, alloc_size, 0, handle, 0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[nixl_vmm] cuMemMap failed: %d\n", r);
        cuMemRelease(handle);        /* no mapping holds pages; release frees them */
        cuMemAddressFree(ptr, alloc_size);
        return NULL;
    }
    /* Mapping is live; drop handle ref-count (mapping keeps pages alive) */
    cuMemRelease(handle);

    /* 4. Grant read-write access to the owning device */
    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = device;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = cuMemSetAccess(ptr, alloc_size, &access, 1);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[nixl_vmm] cuMemSetAccess failed: %d\n", r);
        cuMemUnmap(ptr, alloc_size);
        cuMemAddressFree(ptr, alloc_size);
        return NULL;
    }

    return (void*)ptr;
}

/*
 * Free a VMM allocation.  size must be the same value passed to nixl_vmm_alloc
 * (torch.cuda.MemPool guarantees this).
 */
void nixl_vmm_free(void* ptr, size_t size, int device, void* stream) {
    if (!ptr) return;
    CUdeviceptr cu_ptr = (CUdeviceptr)ptr;
    size_t gran        = vmm_get_granularity(device);
    size_t alloc_size  = vmm_align_up(size, gran);
    cuMemUnmap(cu_ptr, alloc_size);
    cuMemAddressFree(cu_ptr, alloc_size);
}

} /* extern "C" */
"""

_vmm_allocator = None
_vmm_mem_pool: Optional[torch.cuda.MemPool] = None


def init_nixl_vmm_mem_pool(device: str) -> Tuple[bool, Optional[torch.cuda.MemPool]]:
    """
    Compile (once) and return a torch.cuda.MemPool backed by a CUDA VMM
    (cuMemCreate) allocator.

    The pool allocates pages with CU_MEM_HANDLE_TYPE_FABRIC when supported,
    making them addressable over the NVLink fabric for MNNVL transfers.

    Args:
        device: CUDA device string, e.g. "cuda:0"  (used for logging only;
                the C allocator receives the device index from PyTorch).

    Returns:
        (True, mem_pool) on success, (False, None) on failure.
    """
    global _vmm_allocator, _vmm_mem_pool

    if _vmm_mem_pool is not None:
        return True, _vmm_mem_pool

    try:
        import torch.utils.cpp_extension

        out_dir = os.path.join(tempfile.gettempdir(), "nixl_vmm_allocator")
        os.makedirs(out_dir, exist_ok=True)

        # Remove stale build artifacts left by a previously crashed compilation.
        # Wiping the directory ensures a clean rebuild rather than reusing a
        # potentially corrupted main.cpp or lock file.
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        lib_name = "nixl_vmm_allocator"
        torch.utils.cpp_extension.load_inline(
            name=lib_name,
            cpp_sources=_VMM_ALLOCATOR_SOURCE,
            with_cuda=True,          # adds CUDA include paths automatically
            extra_ldflags=["-lcuda"],  # link against CUDA driver library
            verbose=False,
            is_python_module=False,
            build_directory=out_dir,
        )

        so_path = os.path.join(out_dir, f"{lib_name}.so")
        _vmm_allocator = CUDAPluggableAllocator(
            so_path, "nixl_vmm_alloc", "nixl_vmm_free"
        ).allocator()
        _vmm_mem_pool = torch.cuda.MemPool(_vmm_allocator)

        logger.info(
            "Initialized NIXL VMM memory pool on %s "
            "(cuMemCreate + CU_MEM_HANDLE_TYPE_FABRIC, MNNVL-compatible)",
            device,
        )
        return True, _vmm_mem_pool

    except Exception as e:
        logger.warning(
            "Failed to initialize NIXL VMM memory pool on %s: %s. "
            "Falling back to cudaMalloc.",
            device,
            e,
        )
        return False, None
