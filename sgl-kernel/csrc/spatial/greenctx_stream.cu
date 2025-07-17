#include <torch/all.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cuda_utils.h"
#include "greenctx_stream.h"

std::vector<int64_t> create_greenctx_stream_fallback(CUgreenCtx gctx[2]) {
  CUstream streamA, streamB;
  CUcontext ctx;

  // Stream A
  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[0]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamA, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  // Stream B
  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[1]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamB, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  return {(int64_t)streamA, (int64_t)streamB};
}

#if CUDA_VERSION >= 12050
std::vector<int64_t> create_greenctx_stream_direct(CUgreenCtx gctx[2]) {
  CUstream streamA;
  CUstream streamB;

  CUDA_DRV(cuGreenCtxStreamCreate(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(cuGreenCtxStreamCreate(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

  std::vector<int64_t> vec = {(int64_t)streamA, (int64_t)streamB};
  return vec;
}
#endif

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  TORCH_CHECK(CUDA_VERSION >= 12040, "Green Contexts feature requires CUDA Toolkit 12.4 or newer.");

  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];
  unsigned int nbGroups = 1;

  if (smA <= 0 || smB <= 0) {
    TORCH_CHECK(false, "SM counts must be positive");
  }

  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));
  unsigned int minCount = (unsigned int)(smA + smB);
  unsigned int minCountA = (unsigned int)(smA);
  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  int smCountA = resources[0].sm.smCount;
  int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> stream_handles;

#if CUDA_VERSION >= 12050
  stream_handles = create_greenctx_stream_direct(gctx);
#else
  stream_handles = create_greenctx_stream_fallback(gctx);
#endif

  CUDA_DRV(cuGreenCtxDestroy(gctx[2]));

  std::vector<int64_t> vec = {
      stream_handles[0],  // streamA
      stream_handles[1],  // streamB
      (int64_t)smCountA,
      (int64_t)smCountB};

  return vec;
}
