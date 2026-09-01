// Documentation: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
#include <torch/all.h>

#include <cstdlib>

#include "cuda_utils.h"
#include "greenctx_stream.h"

static int CUDA_DRIVER_VERSION;

using PFN_cuGreenCtxStreamCreate = CUresult(CUDAAPI*)(CUstream*, CUgreenCtx, unsigned int, int);

auto probe_cuGreenCtxStreamCreate() -> PFN_cuGreenCtxStreamCreate {
  static PFN_cuGreenCtxStreamCreate pfn = nullptr;
  CUDA_DRV(cuGetProcAddress("cuGreenCtxStreamCreate", reinterpret_cast<void**>(&pfn), CUDA_DRIVER_VERSION, 0, nullptr));
  return pfn;
}

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

inline void destroy_green_context(CUgreenCtx gctx) {
  if (!gctx) return;
  CUDA_DRV(cuGreenCtxDestroy(gctx));
}

static std::vector<int64_t> create_greenctx_stream_direct_dynamic(CUgreenCtx gctx[2]) {
  // This symbol is introduced in CUDA 12.5
  const static auto pfn = probe_cuGreenCtxStreamCreate();
  if (!pfn) {
    TORCH_WARN("cuGreenCtxStreamCreate(cuda>=12.5) is not available, using fallback");
    return create_greenctx_stream_fallback(gctx);
  }

  CUstream streamA, streamB;
  CUDA_DRV(pfn(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(pfn(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

  return {(int64_t)streamA, (int64_t)streamB};
}

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  CUDA_DRV(cuDriverGetVersion(&CUDA_DRIVER_VERSION));

  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];

  TORCH_CHECK(smA > 0 && smB > 0, "SM counts must be positive");

  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));

  const unsigned minCount = static_cast<unsigned>(smA + smB);
  const unsigned minCountA = static_cast<unsigned>(smA);
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

  destroy_green_context(gctx[2]);

  std::vector<int64_t> vec = {
      streams[0],  // streamA
      streams[1],  // streamB
      (int64_t)smCountA,
      (int64_t)smCountB};

  return vec;
}
