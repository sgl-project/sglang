// Documentation: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
#include <torch/all.h>

#include <cstdlib>

#include "cuda_utils.h"
#include "greenctx_stream.h"
#include "safe_greenctx_symbols.h"

static std::vector<int64_t> create_greenctx_stream_fallback(CUgreenCtx gctx[2]) {
  CUstream streamA, streamB;
  CUcontext ctx;

  CUDA_DRV(SAFE_cuCtxFromGreenCtx(&ctx, gctx[0]));
  CUDA_DRV(SAFE_cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamA, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(SAFE_cuCtxPopCurrent(nullptr));

  CUDA_DRV(SAFE_cuCtxFromGreenCtx(&ctx, gctx[1]));
  CUDA_DRV(SAFE_cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamB, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(SAFE_cuCtxPopCurrent(nullptr));

  return {(int64_t)streamA, (int64_t)streamB};
}

inline void destroy_green_context(CUgreenCtx gctx) {
  if (!gctx) return;
  CUDA_DRV(SAFE_cuGreenCtxDestroy(gctx));
}

static std::vector<int64_t> create_greenctx_stream_direct_dynamic(CUgreenCtx gctx[2]) {
  CUstream streamA, streamB;

  CUresult resultA = SAFE_cuGreenCtxStreamCreate(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0);
  CUresult resultB = SAFE_cuGreenCtxStreamCreate(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0);

  // if any call fails, fallback to the fallback method
  if (resultA != CUDA_SUCCESS || resultB != CUDA_SUCCESS) {
    return create_greenctx_stream_fallback(gctx);
  }

  return {(int64_t)streamA, (int64_t)streamB};
}

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  CHECK_CUDA_VERSION_GREEN_CTX_SUPPORT();

  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];
  if (smA <= 0 || smB <= 0) {
    TORCH_CHECK(false, "SM counts must be positive");
  }

  CUDA_DRV(SAFE_cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));

  const unsigned minCount = static_cast<unsigned>(smA + smB);
  const unsigned minCountA = static_cast<unsigned>(smA);
  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  unsigned nbGroups = 1;
  CUDA_DRV(SAFE_cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));
  CUDA_DRV(SAFE_cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(SAFE_cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(SAFE_cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  nbGroups = 1;
  CUDA_DRV(SAFE_cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));
  CUDA_DRV(SAFE_cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(SAFE_cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(SAFE_cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(SAFE_cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  const int smCountA = resources[0].sm.smCount;
  const int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> streams = create_greenctx_stream_direct_dynamic(gctx);

  destroy_green_context(gctx[2]);

  std::vector<int64_t> vec = {
      streams[0],  // streamA
      streams[1],  // streamB
      (int64_t)smCountA,
      (int64_t)smCountB};

  return vec;
}
