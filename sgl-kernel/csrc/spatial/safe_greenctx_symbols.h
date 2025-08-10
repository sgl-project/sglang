#pragma once

#include <torch/all.h>

#include <mutex>

#include "cuda_utils.h"

#define CUDA_SAFE_SYMBOL(symbol_name, fallback_return)                                                             \
  [](auto... args) -> decltype(symbol_name(args...)) {                                                             \
    static decltype(symbol_name)* pfn = nullptr;                                                                   \
    static std::once_flag pfn_probed_flag;                                                                         \
    std::call_once(                                                                                                \
        pfn_probed_flag, []() { cuGetProcAddress(#symbol_name, reinterpret_cast<void**>(&pfn), 0, 0, nullptr); }); \
    if (!pfn) {                                                                                                    \
      return fallback_return;                                                                                      \
    }                                                                                                              \
    return pfn(args...);                                                                                           \
  }

#define SAFE_cuDeviceGetDevResource(device, resource, type) \
  CUDA_SAFE_SYMBOL(cuDeviceGetDevResource, CUDA_ERROR_NOT_SUPPORTED)(device, resource, type)

#define SAFE_cuGreenCtxStreamCreate(stream, gctx, flags, priority) \
  CUDA_SAFE_SYMBOL(cuGreenCtxStreamCreate, CUDA_ERROR_NOT_SUPPORTED)(stream, gctx, flags, priority)

#define SAFE_cuGreenCtxCreate(gctx, desc, device, flags) \
  CUDA_SAFE_SYMBOL(cuGreenCtxCreate, CUDA_ERROR_NOT_SUPPORTED)(gctx, desc, device, flags)

#define SAFE_cuGreenCtxDestroy(gctx) CUDA_SAFE_SYMBOL(cuGreenCtxDestroy, CUDA_ERROR_NOT_SUPPORTED)(gctx)

#define SAFE_cuGreenCtxGetDevResource(gctx, resource, type) \
  CUDA_SAFE_SYMBOL(cuGreenCtxGetDevResource, CUDA_ERROR_NOT_SUPPORTED)(gctx, resource, type)

#define SAFE_cuDevSmResourceSplitByCount(resource1, nbGroups, input, resource2, flags, count) \
  CUDA_SAFE_SYMBOL(cuDevSmResourceSplitByCount, CUDA_ERROR_NOT_SUPPORTED)                     \
  (resource1, nbGroups, input, resource2, flags, count)

#define SAFE_cuDevResourceGenerateDesc(desc, resource, count) \
  CUDA_SAFE_SYMBOL(cuDevResourceGenerateDesc, CUDA_ERROR_NOT_SUPPORTED)(desc, resource, count)

#define SAFE_cuCtxFromGreenCtx(ctx, gctx) CUDA_SAFE_SYMBOL(cuCtxFromGreenCtx, CUDA_ERROR_NOT_SUPPORTED)(ctx, gctx)

#define SAFE_cuCtxPushCurrent(ctx) CUDA_SAFE_SYMBOL(cuCtxPushCurrent, CUDA_ERROR_NOT_SUPPORTED)(ctx)

#define SAFE_cuCtxPopCurrent(ctx) CUDA_SAFE_SYMBOL(cuCtxPopCurrent, CUDA_ERROR_NOT_SUPPORTED)(ctx)

#define CHECK_CUDA_VERSION_GREEN_CTX_SUPPORT()                                           \
  do {                                                                                   \
    if (CUDA_VERSION < 12040) {                                                          \
      TORCH_CHECK(false, "Green Contexts feature requires CUDA Toolkit 12.4 or newer."); \
    }                                                                                    \
  } while (0)
