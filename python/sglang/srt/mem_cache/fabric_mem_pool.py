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
"""Native CUDA VMM CU_MEM_HANDLE_TYPE_FABRIC custom memory pool for MNNVL PD."""

from __future__ import annotations

import ctypes
import logging
import os
import tempfile
import threading
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg, get_exec
from sglang.srt.utils.cuda_vmm_utils import is_gpu_fabric_ready

logger = logging.getLogger(__name__)


def _fabric_vmm_allocator_source(sfx: str) -> str:
    """Return C++ source for a size-unbounded CUDA VMM FABRIC pluggable allocator.

    Compiled with ``with_cuda=False`` and resolves Driver API symbols from
    ``libcuda.so.1`` via ``dlopen``/``dlsym`` at runtime so neither ``nvcc`` nor
    CUDA toolkit headers are required on the host.
    """
    return f"""
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <unordered_map>

namespace {{

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = uint64_t;
using CUmemGenericAllocationHandle = uint64_t;

constexpr CUresult kCudaSuccess = 0;
constexpr int kCuMemAllocationTypePinned = 0x1;
constexpr int kCuMemLocationTypeDevice = 0x1;
constexpr int kCuMemHandleTypeFabric = 0x8;
constexpr unsigned int kCuMemAllocGranularityRecommended = 0x1;
constexpr int kCuMemAccessFlagsProtReadWrite = 0x3;

struct CUmemLocation_v1 {{
  int type;
  int id;
}};
static_assert(sizeof(CUmemLocation_v1) == 8, "CUmemLocation ABI mismatch");

struct CUmemAllocationProp_v1 {{
  int type;
  int requestedHandleTypes;
  CUmemLocation_v1 location;
  void* win32HandleMetaData;
  struct {{
    unsigned char compressionType;
    unsigned char gpuDirectRDMACapable;
    unsigned short usage;
    unsigned char reserved[4];
  }} allocFlags;
}};
static_assert(
    sizeof(CUmemAllocationProp_v1) == 32, "CUmemAllocationProp ABI mismatch");

struct CUmemAccessDesc_v1 {{
  CUmemLocation_v1 location;
  int flags;
}};
static_assert(sizeof(CUmemAccessDesc_v1) == 12, "CUmemAccessDesc ABI mismatch");

using FnCuDeviceGet = CUresult (*)(CUdevice*, int);
using FnCuMemCreate =
    CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp_v1*, unsigned long long);
using FnCuMemRelease = CUresult (*)(CUmemGenericAllocationHandle);
using FnCuMemAddressReserve =
    CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
using FnCuMemAddressFree = CUresult (*)(CUdeviceptr, size_t);
using FnCuMemMap =
    CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
using FnCuMemUnmap = CUresult (*)(CUdeviceptr, size_t);
using FnCuMemSetAccess =
    CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc_v1*, size_t);
using FnCuMemGetAllocationGranularity =
    CUresult (*)(size_t*, const CUmemAllocationProp_v1*, unsigned int);

struct DriverApi {{
  void* lib = nullptr;
  FnCuDeviceGet cuDeviceGet = nullptr;
  FnCuMemCreate cuMemCreate = nullptr;
  FnCuMemRelease cuMemRelease = nullptr;
  FnCuMemAddressReserve cuMemAddressReserve = nullptr;
  FnCuMemAddressFree cuMemAddressFree = nullptr;
  FnCuMemMap cuMemMap = nullptr;
  FnCuMemUnmap cuMemUnmap = nullptr;
  FnCuMemSetAccess cuMemSetAccess = nullptr;
  FnCuMemGetAllocationGranularity cuMemGetAllocationGranularity = nullptr;
  bool ready = false;
}};

struct AllocEntry {{
  CUmemGenericAllocationHandle handle;
  size_t alloc_size;
}};

static std::mutex g_mu;
static DriverApi g_drv;
static bool g_drv_attempted = false;
static std::unordered_map<CUdeviceptr, AllocEntry> g_allocs;
static std::unordered_map<int, int> g_rdma_capable_by_dev;
static size_t g_mapped_bytes = 0;
static int g_last_error = 0;

// Unit-test mock mode for CPU-only CI runners without libcuda.so.1.
static bool g_mock_mode = false;
static size_t g_mock_granularity = 2ULL * 1024ULL * 1024ULL;
static CUdeviceptr g_mock_next_addr = 0x100000000ULL;
static uint64_t g_mock_next_handle = 1;

static size_t align_up(size_t value, size_t alignment) {{
  return ((value + alignment - 1) / alignment) * alignment;
}}

static bool ensure_driver_loaded_locked() {{
  if (g_drv.ready) return true;
  if (g_drv_attempted) return false;
  g_drv_attempted = true;

  void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!lib) {{
    lib = dlopen("libcuda.so.1", RTLD_LAZY);
  }}
  if (!lib) {{
    g_last_error = -1;
    return false;
  }}

  g_drv.lib = lib;
  g_drv.cuDeviceGet = reinterpret_cast<FnCuDeviceGet>(dlsym(lib, "cuDeviceGet"));
  g_drv.cuMemCreate = reinterpret_cast<FnCuMemCreate>(dlsym(lib, "cuMemCreate"));
  g_drv.cuMemRelease =
      reinterpret_cast<FnCuMemRelease>(dlsym(lib, "cuMemRelease"));
  g_drv.cuMemAddressReserve = reinterpret_cast<FnCuMemAddressReserve>(
      dlsym(lib, "cuMemAddressReserve"));
  g_drv.cuMemAddressFree =
      reinterpret_cast<FnCuMemAddressFree>(dlsym(lib, "cuMemAddressFree"));
  g_drv.cuMemMap = reinterpret_cast<FnCuMemMap>(dlsym(lib, "cuMemMap"));
  g_drv.cuMemUnmap = reinterpret_cast<FnCuMemUnmap>(dlsym(lib, "cuMemUnmap"));
  g_drv.cuMemSetAccess =
      reinterpret_cast<FnCuMemSetAccess>(dlsym(lib, "cuMemSetAccess"));
  g_drv.cuMemGetAllocationGranularity =
      reinterpret_cast<FnCuMemGetAllocationGranularity>(
          dlsym(lib, "cuMemGetAllocationGranularity"));

  g_drv.ready = g_drv.cuDeviceGet && g_drv.cuMemCreate && g_drv.cuMemRelease &&
                g_drv.cuMemAddressReserve && g_drv.cuMemAddressFree &&
                g_drv.cuMemMap && g_drv.cuMemUnmap && g_drv.cuMemSetAccess &&
                g_drv.cuMemGetAllocationGranularity;
  if (!g_drv.ready) {{
    g_last_error = -2;
  }}
  return g_drv.ready;
}}

static CUmemAllocationProp_v1 make_prop(CUdevice cu_dev, unsigned char rdma) {{
  CUmemAllocationProp_v1 prop;
  std::memset(&prop, 0, sizeof(prop));
  prop.type = kCuMemAllocationTypePinned;
  prop.requestedHandleTypes = kCuMemHandleTypeFabric;
  prop.location.type = kCuMemLocationTypeDevice;
  prop.location.id = static_cast<int>(cu_dev);
  prop.allocFlags.gpuDirectRDMACapable = rdma;
  return prop;
}}

}}  // namespace

extern "C" {{

void* fabric_vmm_malloc_{sfx}(size_t size, int device, void* /*stream*/) {{
  std::lock_guard<std::mutex> lk(g_mu);
  size_t req_size = size > 0 ? size : 1;

  if (g_mock_mode) {{
    size_t gran = g_mock_granularity > 0 ? g_mock_granularity : 4096;
    size_t alloc_size = align_up(req_size, gran);
    CUdeviceptr ptr = align_up(g_mock_next_addr, gran);
    g_mock_next_addr = ptr + alloc_size;
    CUmemGenericAllocationHandle handle = g_mock_next_handle++;
    g_allocs.emplace(ptr, AllocEntry{{handle, alloc_size}});
    g_mapped_bytes += alloc_size;
    g_last_error = 0;
    return reinterpret_cast<void*>(ptr);
  }}

  if (!ensure_driver_loaded_locked()) {{
    return nullptr;
  }}

  CUdevice cu_dev = device;
  CUresult err = g_drv.cuDeviceGet(&cu_dev, device);
  if (err != kCudaSuccess) {{
    g_last_error = err;
    return nullptr;
  }}

  int rdma_pref = 1;
  auto it = g_rdma_capable_by_dev.find(device);
  if (it != g_rdma_capable_by_dev.end()) {{
    rdma_pref = it->second;
  }}

  CUmemGenericAllocationHandle handle = 0;
  size_t granularity = 0;
  size_t alloc_size = 0;
  bool created = false;

  for (int attempt = 0; attempt < 2; ++attempt) {{
    unsigned char rdma = static_cast<unsigned char>(rdma_pref);
    CUmemAllocationProp_v1 prop = make_prop(cu_dev, rdma);
    granularity = 0;
    err = g_drv.cuMemGetAllocationGranularity(
        &granularity, &prop, kCuMemAllocGranularityRecommended);
    if (err == kCudaSuccess && granularity > 0) {{
      alloc_size = align_up(req_size, granularity);
      err = g_drv.cuMemCreate(&handle, alloc_size, &prop, 0);
      if (err == kCudaSuccess) {{
        g_rdma_capable_by_dev[device] = rdma_pref;
        created = true;
        break;
      }}
    }}
    g_last_error = err;
    if (rdma_pref == 0) {{
      break;
    }}
    rdma_pref = 0;
  }}

  if (!created) {{
    return nullptr;
  }}

  CUdeviceptr ptr = 0;
  err = g_drv.cuMemAddressReserve(&ptr, alloc_size, granularity, 0, 0);
  if (err != kCudaSuccess) {{
    g_last_error = err;
    g_drv.cuMemRelease(handle);
    return nullptr;
  }}

  err = g_drv.cuMemMap(ptr, alloc_size, 0, handle, 0);
  if (err != kCudaSuccess) {{
    g_last_error = err;
    g_drv.cuMemAddressFree(ptr, alloc_size);
    g_drv.cuMemRelease(handle);
    return nullptr;
  }}

  CUmemAccessDesc_v1 access_desc;
  std::memset(&access_desc, 0, sizeof(access_desc));
  access_desc.location.type = kCuMemLocationTypeDevice;
  access_desc.location.id = static_cast<int>(cu_dev);
  access_desc.flags = kCuMemAccessFlagsProtReadWrite;

  err = g_drv.cuMemSetAccess(ptr, alloc_size, &access_desc, 1);
  if (err != kCudaSuccess) {{
    g_last_error = err;
    g_drv.cuMemUnmap(ptr, alloc_size);
    g_drv.cuMemAddressFree(ptr, alloc_size);
    g_drv.cuMemRelease(handle);
    return nullptr;
  }}

  g_allocs.emplace(ptr, AllocEntry{{handle, alloc_size}});
  g_mapped_bytes += alloc_size;
  g_last_error = 0;
  return reinterpret_cast<void*>(ptr);
}}

void fabric_vmm_free_{sfx}(
    void* ptr, size_t /*size*/, int /*device*/, void* /*stream*/) {{
  if (!ptr) return;
  std::lock_guard<std::mutex> lk(g_mu);
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
  auto it = g_allocs.find(dptr);
  if (it == g_allocs.end()) {{
    return;
  }}
  AllocEntry entry = it->second;
  g_allocs.erase(it);
  if (g_mapped_bytes >= entry.alloc_size) {{
    g_mapped_bytes -= entry.alloc_size;
  }} else {{
    g_mapped_bytes = 0;
  }}

  if (g_mock_mode) {{
    return;
  }}

  if (g_drv.ready) {{
    g_drv.cuMemUnmap(dptr, entry.alloc_size);
    g_drv.cuMemRelease(entry.handle);
    g_drv.cuMemAddressFree(dptr, entry.alloc_size);
  }}
}}

void fabric_vmm_set_mock_mode_{sfx}(int enabled, size_t granularity) {{
  std::lock_guard<std::mutex> lk(g_mu);
  g_mock_mode = (enabled != 0);
  if (granularity > 0) {{
    g_mock_granularity = granularity;
  }}
}}

size_t fabric_vmm_active_count_{sfx}() {{
  std::lock_guard<std::mutex> lk(g_mu);
  return g_allocs.size();
}}

size_t fabric_vmm_mapped_bytes_{sfx}() {{
  std::lock_guard<std::mutex> lk(g_mu);
  return g_mapped_bytes;
}}

int fabric_vmm_last_error_{sfx}() {{
  std::lock_guard<std::mutex> lk(g_mu);
  return g_last_error;
}}

}}  // extern "C"
"""


class CudaVmmFabricAllocatorStub:
    """JIT-built CUDA VMM ``CU_MEM_HANDLE_TYPE_FABRIC`` pluggable allocator.

    Allocates physical pages via ``cuMemCreate`` with
    ``CU_MEM_HANDLE_TYPE_FABRIC`` (without any minimum size threshold) so every
    KV cache buffer — including target ``KVCache`` pools, speculative EAGLE
    draft ``KVCache`` pools, ``MambaPool`` state, and disaggregation staging
    buffers — is fabric-exportable across MNNVL cliques without requiring
    ``mooncake-transfer-engine``.
    """

    _instance_count = 0

    def __init__(self, *, build_pluggable_allocator: bool = True) -> None:
        self.sfx = f"{os.getpid()}_{CudaVmmFabricAllocatorStub._instance_count}"
        CudaVmmFabricAllocatorStub._instance_count += 1
        self._lib = self._build()
        self.malloc_symbol = f"fabric_vmm_malloc_{self.sfx}"
        self.free_symbol = f"fabric_vmm_free_{self.sfx}"
        self.allocator = None
        if build_pluggable_allocator:
            from torch.cuda.memory import CUDAPluggableAllocator

            self.allocator = CUDAPluggableAllocator(
                self._so_path,
                self.malloc_symbol,
                self.free_symbol,
            ).allocator()

    def _build(self) -> ctypes.CDLL:
        import shutil
        import subprocess

        import torch.utils.cpp_extension

        out_dir = os.path.join(tempfile.gettempdir(), "sgl_fabric_vmm_pool", self.sfx)
        os.makedirs(out_dir, exist_ok=True)
        libname = f"sgl_fabric_vmm_alloc_{self.sfx}"
        self._so_path = os.path.join(out_dir, f"{libname}.so")
        cpp_src = _fabric_vmm_allocator_source(self.sfx)

        if shutil.which("ninja") is not None:
            torch.utils.cpp_extension.load_inline(
                name=libname,
                cpp_sources=cpp_src,
                extra_ldflags=["-ldl"],
                with_cuda=False,
                is_python_module=False,
                verbose=False,
                build_directory=out_dir,
                no_implicit_headers=True,
            )
        else:
            cxx = (
                os.environ.get("CXX")
                or shutil.which("c++")
                or shutil.which("g++")
                or shutil.which("clang++")
            )
            if cxx is None:
                raise RuntimeError(
                    "Neither ninja nor a C++ compiler (c++/g++/clang++) was "
                    "found on PATH to build CudaVmmFabricAllocatorStub."
                )
            src_path = os.path.join(out_dir, f"{libname}.cpp")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(cpp_src)
            subprocess.run(
                [
                    cxx,
                    "-std=c++17",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    src_path,
                    "-o",
                    self._so_path,
                    "-ldl",
                ],
                check=True,
            )

        lib = ctypes.CDLL(self._so_path)

        self._fn_malloc = lib[f"fabric_vmm_malloc_{self.sfx}"]
        self._fn_malloc.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self._fn_malloc.restype = ctypes.c_void_p

        self._fn_free = lib[f"fabric_vmm_free_{self.sfx}"]
        self._fn_free.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._fn_free.restype = None

        self._fn_set_mock = lib[f"fabric_vmm_set_mock_mode_{self.sfx}"]
        self._fn_set_mock.argtypes = [ctypes.c_int, ctypes.c_size_t]
        self._fn_set_mock.restype = None

        self._fn_active_count = lib[f"fabric_vmm_active_count_{self.sfx}"]
        self._fn_active_count.argtypes = []
        self._fn_active_count.restype = ctypes.c_size_t

        self._fn_mapped_bytes = lib[f"fabric_vmm_mapped_bytes_{self.sfx}"]
        self._fn_mapped_bytes.argtypes = []
        self._fn_mapped_bytes.restype = ctypes.c_size_t

        self._fn_last_error = lib[f"fabric_vmm_last_error_{self.sfx}"]
        self._fn_last_error.argtypes = []
        self._fn_last_error.restype = ctypes.c_int
        return lib

    def set_mock_mode(self, enabled: bool, granularity: int = 2 * 1024 * 1024) -> None:
        self._fn_set_mock(int(enabled), ctypes.c_size_t(granularity))

    @property
    def active_allocations(self) -> int:
        return int(self._fn_active_count())

    @property
    def mapped_bytes(self) -> int:
        return int(self._fn_mapped_bytes())

    @property
    def last_error(self) -> int:
        return int(self._fn_last_error())


_POOL_LOCK = threading.Lock()
_ALLOCATOR_STUB: Optional[CudaVmmFabricAllocatorStub] = None
_FABRIC_MEM_POOLS: dict[int, Any] = {}


def _resolve_cuda_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device!r}")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _get_allocator_stub() -> CudaVmmFabricAllocatorStub:
    global _ALLOCATOR_STUB
    if _ALLOCATOR_STUB is None:
        _ALLOCATOR_STUB = CudaVmmFabricAllocatorStub()
    return _ALLOCATOR_STUB


def get_cuda_vmm_fabric_mem_pool(device: torch.device | str) -> Any:
    """Return a cached per-device ``torch.cuda.MemPool`` backed by CUDA VMM
    ``CU_MEM_HANDLE_TYPE_FABRIC`` physical allocations.

    Reusing one ``MemPool`` per CUDA device allows multiple KV pools on the same
    rank (for example, target ``MLATokenToKVPool`` plus speculative EAGLE draft
    ``MLATokenToKVPool``) to allocate fabric-exportable segments safely without
    rebuilding the pluggable allocator.
    """
    cuda_dev = _resolve_cuda_device(device)
    device_id = int(cuda_dev.index)
    with _POOL_LOCK:
        pool = _FABRIC_MEM_POOLS.get(device_id)
        if pool is None:
            stub = _get_allocator_stub()
            # Bind MemPool to cuda_dev explicitly so non-main threads (where the
            # current device defaults to 0) do not trigger CUDACachingAllocator
            # use_count assertions inside torch.cuda.use_mem_pool().
            with torch.cuda.device(cuda_dev):
                pool = torch.cuda.MemPool(stub.allocator)
            _FABRIC_MEM_POOLS[device_id] = pool
        return pool


def init_cuda_vmm_fabric_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Initialize (or reuse) the native CUDA VMM FABRIC custom memory pool."""
    try:
        pool = get_cuda_vmm_fabric_mem_pool(device)
        logger.debug("Initialized native CUDA VMM FABRIC mem pool on %s", device)
        return True, pool, "FABRIC"
    except Exception as error:
        logger.error(
            "Failed to initialize native CUDA VMM FABRIC mem pool on %s: %s. "
            "Falling back to default memory pool.",
            device,
            error,
        )
        return False, None, None


def nixl_fabric_custom_mem_pool_planned(
    cfg: Any, device: Optional[torch.device | str] = None
) -> bool:
    """Return True when NIXL disaggregated serving will auto-enable the native
    CUDA VMM FABRIC custom memory pool during server arg / engine setup."""
    if getattr(cfg, "device", "cuda") != "cuda":
        return False
    if getattr(cfg, "enable_memory_saver", False):
        return False
    if getattr(cfg, "disaggregation_mode", "null") not in ("prefill", "decode"):
        return False
    if getattr(cfg, "disaggregation_transfer_backend", None) != "nixl":
        return False
    if not envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.get():
        return False
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        return False
    target_device = (
        _resolve_cuda_device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    return is_gpu_fabric_ready(target_device)


def should_use_nixl_fabric_mem_pool(device: str) -> bool:
    """Return True when the published runtime context selects the native CUDA
    VMM FABRIC custom memory pool for ``device``."""
    if not envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.get():
        return False
    try:
        disagg = get_disagg()
        exec_cfg = get_exec()
    except ValueError:
        return False
    if disagg.disaggregation_mode not in ("prefill", "decode"):
        return False
    if disagg.disaggregation_transfer_backend != "nixl":
        return False
    if exec_cfg.features.enable_memory_saver:
        return False
    try:
        cuda_dev = _resolve_cuda_device(device)
    except ValueError:
        return False
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        return False
    return is_gpu_fabric_ready(cuda_dev)
