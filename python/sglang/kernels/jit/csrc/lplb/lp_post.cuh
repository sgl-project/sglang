// LP post kernel: build the final log2phy_prob tensor from the IPM output
// `x` and the prep's `t1`.
//
// Python equivalent (~5 torch ops; this kernel is one launch):
//
//     x_ratios   = clamp(x[:NUM_RED_PHY], min=0)
//     phy_prob   = zeros(NUM_SINGLE + NUM_RED_PHY + 1)         # +1 = sink slot
//     phy_prob[phy_replicated] = x_ratios
//     phy_prob[phy_single]      = t1
//     log2phy_prob = take(phy_prob, log2phy)                    # (-1 wraps to sink)
//
// `log2phy` may contain -1 for unused replicas (DP-attention padding); we
// emulate torch.take's wrap-around by adding `phy_prob_size` to negative
// indices, which lands at the always-zero sink slot.
//
// Single-block launch. `phy_prob` lives in shared memory.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

template <
    int NUM_LOGICAL,
    int MAX_COPIES,
    int NUM_SINGLE,
    int NUM_RED_PHY,
    int BLOCK_DIM>
__global__ void lp_post_kernel(
    float* __restrict__ log2phy_prob,            // (NUM_LOGICAL, MAX_COPIES) — written
    const float* __restrict__ x,                 // (NV,) — IPM output
    const float* __restrict__ t1,                // (NUM_SINGLE,) — from prep
    const int64_t* __restrict__ phy_single,      // (NUM_SINGLE,)
    const int64_t* __restrict__ phy_replicated,  // (NUM_RED_PHY,)
    const int64_t* __restrict__ log2phy) {       // (NUM_LOGICAL, MAX_COPIES)
  constexpr int PHY_PROB_SIZE = NUM_SINGLE + NUM_RED_PHY + 1;

  extern __shared__ unsigned char raw_smem[];
  float* phy_prob = reinterpret_cast<float*>(raw_smem);

  const int tid = threadIdx.x;

  // Stage 1: zero-init phy_prob (covers the sink slot at index PHY_PROB_SIZE-1).
  for (int i = tid; i < PHY_PROB_SIZE; i += BLOCK_DIM) {
    phy_prob[i] = 0.f;
  }
  __syncthreads();

  // Stage 2: scatter x_ratios = clamp(x[:NUM_RED_PHY], min=0) at phy_replicated.
  for (int i = tid; i < NUM_RED_PHY; i += BLOCK_DIM) {
    int64_t idx = phy_replicated[i];
    phy_prob[idx] = fmaxf(x[i], 0.f);
  }
  // Stage 3: scatter t1 at phy_single.
  for (int i = tid; i < NUM_SINGLE; i += BLOCK_DIM) {
    int64_t idx = phy_single[i];
    phy_prob[idx] = t1[i];
  }
  __syncthreads();

  // Stage 4: gather log2phy_prob[i,j] = phy_prob[log2phy[i,j]].
  // -1 entries wrap to the sink slot (PHY_PROB_SIZE - 1), which is 0.
  const int total = NUM_LOGICAL * MAX_COPIES;
  for (int idx = tid; idx < total; idx += BLOCK_DIM) {
    int64_t k = log2phy[idx];
    if (k < 0) k += PHY_PROB_SIZE;
    log2phy_prob[idx] = phy_prob[k];
  }
}

template <int NUM_LOGICAL, int MAX_COPIES, int NUM_SINGLE, int NUM_RED_PHY, int BLOCK_DIM>
void lp_post(
    tvm::ffi::TensorView log2phy_prob,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView t1,
    tvm::ffi::TensorView phy_single,
    tvm::ffi::TensorView phy_replicated,
    tvm::ffi::TensorView log2phy) {
  using namespace host;

  SymbolicDevice device_;
  TensorMatcher({NUM_LOGICAL, MAX_COPIES}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(log2phy_prob);
  TensorMatcher({NUM_SINGLE}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(t1);
  TensorMatcher({NUM_SINGLE}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(phy_single);
  TensorMatcher({NUM_RED_PHY}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(phy_replicated);
  TensorMatcher({NUM_LOGICAL, MAX_COPIES}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(log2phy);
  // x has shape (NV,) which we don't constrain at this layer.

  constexpr int PHY_PROB_SIZE = NUM_SINGLE + NUM_RED_PHY + 1;
  const size_t smem_bytes = PHY_PROB_SIZE * sizeof(float);

  using KernelT = void (*)(float*, const float*, const float*, const int64_t*, const int64_t*, const int64_t*);
  KernelT kernel = lp_post_kernel<NUM_LOGICAL, MAX_COPIES, NUM_SINGLE, NUM_RED_PHY, BLOCK_DIM>;

  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  }

  const DLDevice device = device_.unwrap();
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device, smem_bytes)(
      kernel,
      static_cast<float*>(log2phy_prob.data_ptr()),
      static_cast<const float*>(x.data_ptr()),
      static_cast<const float*>(t1.data_ptr()),
      static_cast<const int64_t*>(phy_single.data_ptr()),
      static_cast<const int64_t*>(phy_replicated.data_ptr()),
      static_cast<const int64_t*>(log2phy.data_ptr()));
}

}  // namespace
