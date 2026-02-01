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
__global__ void fused_add_rmsnorm_reg_kernel(
    T* __restrict__ input, T* __restrict__ residual, const T* __restrict__ weight, int vec_hidden_size, float eps) {
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;

  __shared__ float shared_memory[32];  // Used for CTA reduce

  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  vec_t v;         // Save input
  vec_t v_res;     // Save residual
  vec_t v_weight;  // Save weight
  vec_t v_out;     // Save output

  auto token_id = blockIdx.x;
  float2 acc_square = make_float2(0.0f, 0.0f);  // Sum of squares for each thread

  if (threadIdx.x < vec_hidden_size) {
    // Compute address
    vec_t* p = reinterpret_cast<vec_t*>(input) + token_id * vec_hidden_size;
    vec_t* p_res = reinterpret_cast<vec_t*>(residual) + token_id * vec_hidden_size;
    const vec_t* p_weight = reinterpret_cast<const vec_t*>(weight);

    // Load data
    v = p[threadIdx.x];
    v_res = p_res[threadIdx.x];
    v_weight = p_weight[threadIdx.x];

    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v[i]);
      float2 res = device::cast<fp32x2_t, packed_t>(v_res[i]);
      float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
      acc_square.x += inp_res.x * inp_res.x;
      acc_square.y += inp_res.y * inp_res.y;
      v[i] = device::cast<packed_t, fp32x2_t>(inp_res);
    }

    // Store inp+res to residual
    p_res[threadIdx.x] = v;
  }

  // CTA Reduce
  // Step 0: Warp Reduce
  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  float warp_sum = cooperative_groups::reduce(cg_warp, acc_square.x + acc_square.y, cooperative_groups::plus<float>());

  float* buffer = shared_memory;
  if (threadIdx.x % 32 == 0) {
    buffer[threadIdx.x / 32] = warp_sum;  // Write warp_sum to buffer
  }

  // Step 1: CTA Reduce
  __syncthreads();
  if (threadIdx.x < 32) {
    float cta_sum = cooperative_groups::reduce(
        cg_warp, (threadIdx.x < blockDim.x / 32) ? buffer[threadIdx.x] : 0.0f, cooperative_groups::plus<float>());
    buffer[threadIdx.x] =
        rsqrtf(eps + cta_sum * (1.0f / static_cast<float>(vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T)))));
  }
  __syncthreads();

  // Compute RMSNorm
  if (threadIdx.x < vec_hidden_size) {
    float rsqrt_square_sum = buffer[threadIdx.x / 32];  // Read rsqrt from Shared Memory(Broadcast)
    for (int i = 0; i < inner_loop; i++) {
      v_out[i] = rms(v[i], v_weight[i], rsqrt_square_sum);
    }
    vec_t* p_out = reinterpret_cast<vec_t*>(input) + token_id * vec_hidden_size;
    p_out[threadIdx.x] = v_out;
  }
}

template <typename DType>
struct FusedAddRMSNormKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({N, D})  // residual
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual);

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

      // Launch kernel
      auto kernel =
          max_vec_size_byte == 32 ? fused_add_rmsnorm_reg_kernel<DType, 32> : fused_add_rmsnorm_reg_kernel<DType, 16>;
      LaunchKernel(static_cast<uint>(N.unwrap()), threads, device.unwrap())
          .enable_pdl(false)(
              kernel,
              reinterpret_cast<DType*>(input.data_ptr()),
              reinterpret_cast<DType*>(residual.data_ptr()),
              reinterpret_cast<DType*>(weight.data_ptr()),
              vec_hidden_size,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
