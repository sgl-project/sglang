#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups/reduce.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <type_traits>

namespace {

template <typename T, int VEC_SIZE_IN_BYTE>
struct VecTypeTrait;

template <>
struct VecTypeTrait<bf16_t, 16> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<fp16_t, 16> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<bf16_t, 32> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <>
struct VecTypeTrait<fp16_t, 32> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <typename packed_t>
SGL_DEVICE packed_t rms(packed_t& val, packed_t& weight, float rsqrt_square_sum) {
  float2 valf = device::cast<fp32x2_t, packed_t>(val);
  float2 weightf = device::cast<fp32x2_t, packed_t>(weight);
  return device::cast<packed_t, fp32x2_t>(
      make_float2(valf.x * weightf.x * rsqrt_square_sum, valf.y * weightf.y * rsqrt_square_sum));
}

template <typename T, int VEC_SIZE_IN_BYTE>
__global__ void qknorm_across_heads_reg_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const T* __restrict__ q_weight,
    const T* __restrict__ k_weight,
    int vec_hidden_size,
    float eps) {
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;

  __shared__ float shared_memory[64];  // Used for CTA reduce, store both Q and K rsqrt

  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  vec_t v_q;         // Save q
  vec_t v_k;         // Save k
  vec_t v_q_weight;  // Save q_weight
  vec_t v_k_weight;  // Save k_weight
  vec_t v_q_out;     // Save q output
  vec_t v_k_out;     // Save k output

  auto token_id = blockIdx.x;
  float2 acc_square_q = make_float2(0.0f, 0.0f);  // Sum of squares for q
  float2 acc_square_k = make_float2(0.0f, 0.0f);  // Sum of squares for k

  if (threadIdx.x < vec_hidden_size) {
    // Compute address for q and k
    vec_t* p_q = reinterpret_cast<vec_t*>(q) + token_id * vec_hidden_size;
    vec_t* p_k = reinterpret_cast<vec_t*>(k) + token_id * vec_hidden_size;
    const vec_t* p_q_weight = reinterpret_cast<const vec_t*>(q_weight);
    const vec_t* p_k_weight = reinterpret_cast<const vec_t*>(k_weight);

    // Load data
    v_q = p_q[threadIdx.x];
    v_k = p_k[threadIdx.x];
    v_q_weight = p_q_weight[threadIdx.x];
    v_k_weight = p_k_weight[threadIdx.x];

    // Compute sum of squares for q
    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v_q[i]);
      acc_square_q.x += val.x * val.x;
      acc_square_q.y += val.y * val.y;
    }

    // Compute sum of squares for k
    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v_k[i]);
      acc_square_k.x += val.x * val.x;
      acc_square_k.y += val.y * val.y;
    }
  }

  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  float* buffer_q = shared_memory;       // [0, 31] for Q
  float* buffer_k = shared_memory + 32;  // [32, 63] for K

  // ========== Reduction phase: Compute rsqrt for both Q and K ==========

  // Step 0: Warp Reduce for Q
  float warp_sum_q =
      cooperative_groups::reduce(cg_warp, acc_square_q.x + acc_square_q.y, cooperative_groups::plus<float>());
  if (threadIdx.x % 32 == 0) {
    buffer_q[threadIdx.x / 32] = warp_sum_q;
  }

  // Step 0: Warp Reduce for K
  float warp_sum_k =
      cooperative_groups::reduce(cg_warp, acc_square_k.x + acc_square_k.y, cooperative_groups::plus<float>());
  if (threadIdx.x % 32 == 0) {
    buffer_k[threadIdx.x / 32] = warp_sum_k;
  }

  // Step 1: CTA Reduce for both Q and K
  __syncthreads();
  if (threadIdx.x < 32) {
    // CTA Reduce for Q
    float cta_sum_q = cooperative_groups::reduce(
        cg_warp, (threadIdx.x < blockDim.x / 32) ? buffer_q[threadIdx.x] : 0.0f, cooperative_groups::plus<float>());
    buffer_q[threadIdx.x] =
        rsqrtf(eps + cta_sum_q * (1.0f / static_cast<float>(vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T)))));

    // CTA Reduce for K
    float cta_sum_k = cooperative_groups::reduce(
        cg_warp, (threadIdx.x < blockDim.x / 32) ? buffer_k[threadIdx.x] : 0.0f, cooperative_groups::plus<float>());
    buffer_k[threadIdx.x] =
        rsqrtf(eps + cta_sum_k * (1.0f / static_cast<float>(vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T)))));
  }
  __syncthreads();

  // ========== Apply normalization phase: Compute and write back Q and K ==========

  if (threadIdx.x < vec_hidden_size) {
    // Apply RMSNorm for Q
    float rsqrt_q = buffer_q[threadIdx.x / 32];
    for (int i = 0; i < inner_loop; i++) {
      v_q_out[i] = rms(v_q[i], v_q_weight[i], rsqrt_q);
    }
    vec_t* p_q_out = reinterpret_cast<vec_t*>(q) + token_id * vec_hidden_size;
    p_q_out[threadIdx.x] = v_q_out;

    // Apply RMSNorm for K
    float rsqrt_k = buffer_k[threadIdx.x / 32];
    for (int i = 0; i < inner_loop; i++) {
      v_k_out[i] = rms(v_k[i], v_k_weight[i], rsqrt_k);
    }
    vec_t* p_k_out = reinterpret_cast<vec_t*>(k) + token_id * vec_hidden_size;
    p_k_out[threadIdx.x] = v_k_out;
  }
}

template <typename DType>
struct QKNormAcrossHeadsKernel {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // q
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, D})  // k
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})  // q_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight);
    TensorMatcher({D})  // k_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_weight);

    auto cc_major = host::runtime::get_cc_major(device.unwrap().device_id);
    int hidden_size = static_cast<int>(D.unwrap());
    if ((cc_major <= 9 && hidden_size <= 8192) || (cc_major >= 10 && hidden_size <= 12288)) {
      int max_vec_size_byte = cc_major >= 10 ? 32 : 16;
      int elements_in_vec = max_vec_size_byte / sizeof(DType);
      int vec_hidden_size = hidden_size / elements_in_vec;
      uint threads = (vec_hidden_size + 31) / 32 * 32;

      // Runtime check
      host::RuntimeCheck(
          hidden_size % elements_in_vec == 0,
          "hidden_size",
          hidden_size,
          " can not align to elements_in_vec ",
          elements_in_vec);

      // Launch single kernel for both q and k
      auto kernel = max_vec_size_byte == 32 ? qknorm_across_heads_reg_kernel<DType, 32>
                                            : qknorm_across_heads_reg_kernel<DType, 16>;

      LaunchKernel(static_cast<uint>(N.unwrap()), threads, device.unwrap())
          .enable_pdl(false)(
              kernel,
              reinterpret_cast<DType*>(q.data_ptr()),
              reinterpret_cast<DType*>(k.data_ptr()),
              reinterpret_cast<DType*>(q_weight.data_ptr()),
              reinterpret_cast<DType*>(k_weight.data_ptr()),
              vec_hidden_size,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
