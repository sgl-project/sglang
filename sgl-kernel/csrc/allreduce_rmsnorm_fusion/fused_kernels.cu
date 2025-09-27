#include "type_convert.cuh"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
#include <cub/cub.cuh>
#else
#include <hipcub/hipcub.hpp>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && CUDART_VERSION >= 12010
#define NVCC_SUPPORTS_MULTICAST 1
#endif

#include <ATen/ATen.h>
#if !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif

#include "multimem_utils.cuh"
#include <cassert>


//copy from https://github.com/microsoft/tokenweave/blob/main/csrc/tokenweave_fused_kernels.cu


/* 
* ********************************************************* *
* FUSED RS + RESIDUAL ADD + RMS NORM + AG CTA-BASED KERNEL  *
* Function specialization in the case of BF16/FP16 tensors. *
* ********************************************************* *
*/
namespace vllm
{
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  auto *__restrict__ weight_v = reinterpret_cast<const vec_t *>(weight);
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    float variance[1] = {0.0f};
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    __shared__ float s_variance;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;
    auto residual_o = residual_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      vec_t temp = *(reinterpret_cast<vec_t *>(&mtemp));
      temp += residual_o[idx];
      variance[0] += temp.sum_squares(); // FP32 accumulation
      residual_o[idx] = temp;
      // multimem_st<16>(residual_mcptr + offset_scalar + idx * width, 
      //                *(reinterpret_cast<Vec<16> *>(&temp)));
    }

    blockReduceSum<float, 1>(variance);
    if (threadIdx.x == 0)
    {
      s_variance = rsqrtf(variance[0] / hidden_size + epsilon);
    }
    __syncthreads();

    // Second pass: normalize and apply weight
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t shared_weight = weight_v[idx];
      vec_t temp = residual_o[idx];
      temp *= s_variance;
      temp *= shared_weight;
      multimem_st<16>(mcptr + offset_scalar + idx * width, *(reinterpret_cast<Vec<16> *>(&temp)));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/* 
* ********************************************************* *
* FUSED RS + RESIDUAL ADD + RMS NORM + AG CTA-BASED KERNEL  *
* GENERIC NOT SUPPORTED                                     *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
  assert(false && "TokenWeave currently only supports bf16/fp16 with width 8.");
}
} //namespace vllm

/* 
* ******************************************************************* *
* Fused ReduceScatter plus Fused(Residual, RMSNorm) plus AllGather    *
* ******************************************************************* *
*/
#define LAUNCH_FUSED_RS_LN_AG_CTA(width)                                                                                                   \
  VLLM_BF16_DISPATCH_FLOATING_TYPES(                                                                                                       \
      input.scalar_type(), "fused_rs_ln_ag_cta_kernel", [&] { vllm::fused_rs_ln_ag_cta_kernel<scalar_t, width>                             \
                                                                  <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),                 \
                                                                                               reinterpret_cast<scalar_t *>(mcptr),        \
                                                                                               residual.data_ptr<scalar_t>(),              \
                                                                                               weight.data_ptr<scalar_t>(),                \
                                                                                               reinterpret_cast<uint32_t **>(signal_pads), \
                                                                                               static_cast<size_t>(rank),                  \
                                                                                               static_cast<size_t>(world_size),            \
                                                                                               epsilon, num_tokens, hidden_size); });
void fused_rs_ln_ag_cta(torch::Tensor &input,    // [..., hidden_size]
                        torch::Tensor &residual, // [..., hidden_size]
                        torch::Tensor &weight,   // [hidden_size]
                        int64_t mcptr,           // [..., hidden_size] multimem_ptr
                        int64_t signal_pads,     // [..., hidden_size] signal pads
                        int64_t rank,
                        int64_t world_size,
                        int64_t MAX_CTAS,
                        double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(MAX_CTAS);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_FUSED_RS_LN_AG_CTA(8);
  }
  else
  {
    TORCH_CHECK(false, "Input, residual, and weight tensors must be 16-byte aligned and hidden_size must be divisible by 8 for optimized kernel.");
  }
}