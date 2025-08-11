#pragma once

#include <torch/all.h>

#include <mutex>

#include "cuda_utils.h"

#define CUDA_DEFINED_SAFE_CALL(symbol_name_str, ...)                                                                  \
  [](auto... args) -> CUresult {                                                                                      \
    using pfn_t = CUresult(CUDAAPI*)(__VA_ARGS__);                                                                    \
    static pfn_t pfn = nullptr;                                                                                       \
                                                                                                                      \
    static std::once_flag pfn_probed_flag;                                                                            \
    std::call_once(                                                                                                   \
        pfn_probed_flag, []() { cuGetProcAddress(symbol_name_str, reinterpret_cast<void**>(&pfn), 0, 0, nullptr); }); \
                                                                                                                      \
    if (!pfn) {                                                                                                       \
      return CUDA_ERROR_NOT_SUPPORTED;                                                                                \
    }                                                                                                                 \
    return pfn(args...);                                                                                              \
  }

auto SAFE_cuDeviceGetDevResource =
    CUDA_DEFINED_SAFE_CALL("cuDeviceGetDevResource", CUdevice, CUdevResource*, CUdevResourceType);

auto SAFE_cuGreenCtxStreamCreate =
    CUDA_DEFINED_SAFE_CALL("cuGreenCtxStreamCreate", CUstream*, CUgreenCtx, unsigned int, int);

auto SAFE_cuGreenCtxCreate =
    CUDA_DEFINED_SAFE_CALL("cuGreenCtxCreate", CUgreenCtx*, CUdevResourceDesc, CUdevice, unsigned int);

auto SAFE_cuGreenCtxDestroy = CUDA_DEFINED_SAFE_CALL("cuGreenCtxDestroy", CUgreenCtx);

auto SAFE_cuGreenCtxGetDevResource =
    CUDA_DEFINED_SAFE_CALL("cuGreenCtxGetDevResource", CUgreenCtx, CUdevResource*, CUdevResourceType);

auto SAFE_cuDevSmResourceSplitByCount = CUDA_DEFINED_SAFE_CALL(
    "cuDevSmResourceSplitByCount",
    CUdevResource*,
    unsigned int*,
    const CUdevResource*,
    CUdevResource*,
    unsigned int,
    unsigned int);

auto SAFE_cuDevResourceGenerateDesc =
    CUDA_DEFINED_SAFE_CALL("cuDevResourceGenerateDesc", CUdevResourceDesc*, CUdevResource*, unsigned int);

auto SAFE_cuCtxFromGreenCtx = CUDA_DEFINED_SAFE_CALL("cuCtxFromGreenCtx", CUcontext*, CUgreenCtx);

auto SAFE_cuCtxPushCurrent = CUDA_DEFINED_SAFE_CALL("cuCtxPushCurrent", CUcontext);

auto SAFE_cuCtxPopCurrent = CUDA_DEFINED_SAFE_CALL("cuCtxPopCurrent", CUcontext*);

#define CHECK_CUDA_VERSION_GREEN_CTX_SUPPORT()                                           \
  do {                                                                                   \
    if (CUDA_VERSION < 12040) {                                                          \
      TORCH_CHECK(false, "Green Contexts feature requires CUDA Toolkit 12.4 or newer."); \
    }                                                                                    \
  } while (0)
