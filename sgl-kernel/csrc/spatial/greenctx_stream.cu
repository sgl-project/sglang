#include <torch/all.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cuda_utils.h"
#include "greenctx_stream.h"

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];
  CUstream streamA;
  CUstream streamB;

  unsigned int nbGroups = 1;

  if (smA <= 0 || smB <= 0) {
    TORCH_CHECK(false, "SM counts must be positive");
  }

  // Initialize device
  CUDA_RT(cudaInitDevice(device, 0, 0));

  // Query input SMs
  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));
  // We want 3/4 the device for our green context
  unsigned int minCount = (unsigned int)(smA + smB);
  unsigned int minCountA = (unsigned int)(smA);

  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  // Split resources
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));

  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));

  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  CUDA_DRV(cuGreenCtxStreamCreate(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(cuGreenCtxStreamCreate(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

  int smCountA = resources[0].sm.smCount;
  int smCountB = resources[1].sm.smCount;

  CUDA_DRV(cuGreenCtxDestroy(gctx[2]));

  std::vector<int64_t> vec = {(int64_t)streamA, (int64_t)streamB, smCountA, smCountB};
  return vec;
}
