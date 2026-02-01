#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <cooperative_groups/reduce.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace {

struct QKNormParams {
  void* __restrict__ q;
  void* __restrict__ k;  // k is offset by (-num_qo_heads * head_dim) elements
  int64_t q_stride;
  int64_t k_stride;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  float eps;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  uint32_t num_tokens;
};

constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kThreadsPerBlock = kWarpsPerBlock * device::kWarpThreads;

template <typename packed_t>
SGL_DEVICE packed_t rms(packed_t& val, packed_t& weight, float rsqrt_rms) {
  float2 valf = device::cast<fp32x2_t, packed_t>(val);
  float2 weightf = device::cast<fp32x2_t, packed_t>(weight);
  return device::cast<packed_t, fp32x2_t>(
      make_float2(valf.x * weightf.x * rsqrt_rms, valf.y * weightf.y * rsqrt_rms));
}

template <typename T, int VEC_SIZE_IN_BYTE, int64_t kHeadDim, bool kUsePDL>
__global__ void fused_qknorm(const QKNormParams __grid_constant__ params) {
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;
  constexpr int elements_in_vec = VEC_SIZE_IN_BYTE / sizeof(T);
  constexpr int vec_head_dim = kHeadDim / elements_in_vec;
  
  static_assert(sizeof(T) == 2, "Only support FP16/BF16");
  static_assert(kHeadDim % elements_in_vec == 0, "head_dim must be divisible by elements_in_vec");
  
  __shared__ float shared_memory[32];
  
  using vec_t = typename device::VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename device::VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;
  
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  
  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel
  
  // Each block processes one head of one token
  for (uint32_t idx = blockIdx.x; idx < num_q_and_k_heads * num_tokens; idx += gridDim.x) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    
    T* input_ptr = load_q ? 
        reinterpret_cast<T*>(pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))) :
        reinterpret_cast<T*>(pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim)));
    const T* weight_ptr = reinterpret_cast<const T*>(load_q ? q_weight : k_weight);
    
    vec_t v_input;
    vec_t v_weight;
    
    float2 acc_square = make_float2(0.0f, 0.0f);  // Sum of squares for each thread
    
    // Load and compute sum of squares
    if (threadIdx.x < vec_head_dim) {
      vec_t* p_input = reinterpret_cast<vec_t*>(input_ptr);
      const vec_t* p_weight = reinterpret_cast<const vec_t*>(weight_ptr);
      
      v_input = p_input[threadIdx.x];
      v_weight = p_weight[threadIdx.x];
      
      #pragma unroll
      for (int i = 0; i < inner_loop; i++) {
        float2 val = device::cast<fp32x2_t, packed_t>(v_input[i]);
        acc_square.x += val.x * val.x;
        acc_square.y += val.y * val.y;
      }
    }
    
    // Step 0: Warp reduce
    auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    float warp_sum = cooperative_groups::reduce(cg_warp, acc_square.x + acc_square.y, cooperative_groups::plus<float>());
    
    float* buffer = shared_memory;
    if (threadIdx.x % 32 == 0) {
      buffer[threadIdx.x / 32] = warp_sum;  // Write warp_sum to buffer
    }
    
    // Step 1: CTA reduce
    __syncthreads();
    if (threadIdx.x < 32) {
      float cta_sum = cooperative_groups::reduce(
          cg_warp, 
          (threadIdx.x < blockDim.x / 32) ? buffer[threadIdx.x] : 0.0f, 
          cooperative_groups::plus<float>());
      buffer[threadIdx.x] = device::math::rsqrt(eps + cta_sum / kHeadDim);
    }
    __syncthreads();
    
    if (threadIdx.x < vec_head_dim) {
      float rsqrt_rms = buffer[threadIdx.x / 32];
      vec_t v_out;
      #pragma unroll
      for (int i = 0; i < inner_loop; i++) {
        v_out[i] = rms(v_input[i], v_weight[i], rsqrt_rms);
      }
      vec_t* p_out = reinterpret_cast<vec_t*>(input_ptr);
      p_out[threadIdx.x] = v_out;
    }
  }
  
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHeadDim, bool kUsePDL, typename DType>
struct QKNormKernel {
  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(!host::norm::should_use_cta<DType, kHeadDim>(), "Head dim too large for QKNorm");

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto Sq = SymbolicSize{"q_stride"};
    auto Sk = SymbolicSize{"k_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D})  // q input
        .with_strides({Sq, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Sk, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight)
        .verify(k_weight);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const auto params = QKNormParams{
        .q = q.data_ptr(),
        .k = pointer::offset(k.data_ptr(), -2 * static_cast<int64_t>(num_qo_heads) * kHeadDim),
        .q_stride = static_cast<int64_t>(Sq.unwrap()),
        .k_stride = static_cast<int64_t>(Sk.unwrap()),
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .eps = eps,
        .q_weight = q_weight.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .num_tokens = num_tokens,
    };

    auto cc_major = host::runtime::get_cc_major(device.unwrap().device_id);
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    int max_vec_size_byte = cc_major >= 10 ? 32 : 16;
    int elements_in_vec = max_vec_size_byte / sizeof(DType);
    
    host::RuntimeCheck(
        kHeadDim % elements_in_vec == 0,
        "head_dim",
        kHeadDim,
        " can not align to elements_in_vec ",
        elements_in_vec);
    
    int vec_head_dim = kHeadDim / elements_in_vec;
    uint threads = (vec_head_dim + 31) / 32 * 32;
    
    host::RuntimeCheck(
        threads <= kThreadsPerBlock,
        "Required threads ",
        threads,
        " exceeds max threads per block ",
        kThreadsPerBlock);
    
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    
    auto kernel = max_vec_size_byte == 32 ? 
        fused_qknorm<DType, 32, kHeadDim, kUsePDL> : 
        fused_qknorm<DType, 16, kHeadDim, kUsePDL>;
    uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, threads);
    uint32_t num_blocks = std::min(kNumSM * max_occupancy, num_works);
    
    LaunchKernel(num_blocks, threads, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
