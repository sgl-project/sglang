#include <torch/all.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cuda_utils.h"
#include "greenctx_stream.h"

#if CUDA_VERSION >= 12040

static std::vector<int64_t> create_greenctx_stream_fallback(CUgreenCtx gctx[2]) {
  CUstream streamA, streamB;
  CUcontext ctx;

  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[0]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamA, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[1]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamB, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  return {(int64_t)streamA, (int64_t)streamB};
}

typedef CUresult(CUDAAPI* PFN_cuGreenCtxStreamCreate)(CUstream*, CUgreenCtx, unsigned int, int);

static std::vector<int64_t> create_greenctx_stream_direct_dynamic(CUgreenCtx gctx[2]) {
  static PFN_cuGreenCtxStreamCreate pfn = nullptr;
  static std::once_flag pfn_probed_flag;

  // detect compatibility in runtime
  std::call_once(pfn_probed_flag, []() {
    cuGetProcAddress("cuGreenCtxStreamCreate", reinterpret_cast<void**>(&pfn), 0, 0, nullptr);
  });

  if (!pfn) {  // fallback if not compatible
    return create_greenctx_stream_fallback(gctx);
  }

  CUstream streamA, streamB;
  CUDA_DRV(pfn(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(pfn(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

  return {(int64_t)streamA, (int64_t)streamB};
}

inline void destroy_green_context(int64_t h) {
  if (h) CUDA_DRV(cuGreenCtxDestroy(reinterpret_cast<CUgreenCtx>(h)));
}

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  TORCH_CHECK(CUDA_VERSION >= 12040, "Green Contexts feature requires CUDA Toolkit 12.4 or newer.");

  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];
  if (smA <= 0 || smB <= 0) {
    TORCH_CHECK(false, "SM counts must be positive");
  }

  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));

  const unsigned minCount = smA + smB;
  const unsigned minCountA = smA;
  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  unsigned nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  const int smCountA = resources[0].sm.smCount;
  const int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> streams = create_greenctx_stream_direct_dynamic(gctx);

  CUDA_DRV(cuGreenCtxDestroy(gctx[2]));

  std::vector<int64_t> vec = {
      streams[0],  // streamA
      streams[1],  // streamB
      (int64_t)smCountA,
      (int64_t)smCountB};

  return vec;
}

#else

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  TORCH_CHECK(
      false,
      "Green Contexts feature requires CUDA Toolkit 12.4 or newer. Current CUDA version: " +
          std::to_string(CUDA_VERSION));

  // This is a stub function that should never be reached
  // Return empty vector to satisfy return type requirement
  return {};
}

#endif
