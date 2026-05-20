#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
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

template <typename T, int VEC_SIZE_IN_BYTE>
__global__ void fused_add_rmsnorm_per_tensor_quant_reg_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    fp8_e4m3_t* __restrict__ output,
    const float* __restrict__ scale,
    int vec_hidden_size,
    float eps) {
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;
  // Number of fp8 elements per thread: each inner_loop iteration yields 2 scalars
  constexpr int fp8_per_thread = inner_loop * 2;

  __shared__ float shared_memory[32];  // Used for CTA reduce

  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  vec_t v;         // Save input
  vec_t v_res;     // Save residual
  vec_t v_weight;  // Save weight

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

  // Compute RMSNorm + FP8 quantization
  if (threadIdx.x < vec_hidden_size) {
    float rsqrt_square_sum = buffer[threadIdx.x / 32];  // Read rsqrt from Shared Memory(Broadcast)
    const float inv_scale = 1.0f / (*scale);

    device::AlignedVector<fp8_e4m3_t, fp8_per_thread> fp8_out;
#pragma unroll
    for (int i = 0; i < inner_loop; i++) {
      float2 valf = device::cast<fp32x2_t, packed_t>(v[i]);
      float2 weightf = device::cast<fp32x2_t, packed_t>(v_weight[i]);
      float val0 = valf.x * weightf.x * rsqrt_square_sum * inv_scale;
      float val1 = valf.y * weightf.y * rsqrt_square_sum * inv_scale;
      fp8_out[i * 2 + 0] = static_cast<fp8_e4m3_t>(
          device::math::max(-device::math::FP8_E4M3_MAX, device::math::min(val0, device::math::FP8_E4M3_MAX)));
      fp8_out[i * 2 + 1] = static_cast<fp8_e4m3_t>(
          device::math::max(-device::math::FP8_E4M3_MAX, device::math::min(val1, device::math::FP8_E4M3_MAX)));
    }

    // Vectorized fp8 store
    // Each thread handles fp8_per_thread fp8 elements at the corresponding position
    auto* output_row = output + token_id * vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T));
    fp8_out.store(reinterpret_cast<fp8_e4m3_t*>(output_row), threadIdx.x);
  }
}

template <typename DType>
struct FusedAddRMSNormPerTensorQuantKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView scale,
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
    TensorMatcher({N, D})  // output (fp8)
        .with_strides({D, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({1})  // scale
        .with_dtype<float>()
        .with_device(device)
        .verify(scale);

    int hidden_size = static_cast<int>(D.unwrap());
    if (hidden_size <= (device::kMaxVecBytes == 32 ? 12288 : 8192)) {
      int elements_in_vec = device::kMaxVecBytes / sizeof(DType);
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
      auto kernel = fused_add_rmsnorm_per_tensor_quant_reg_kernel<DType, device::kMaxVecBytes>;
      LaunchKernel(static_cast<uint>(N.unwrap()), threads, device.unwrap())
          .enable_pdl(false)(
              kernel,
              reinterpret_cast<DType*>(input.data_ptr()),
              reinterpret_cast<DType*>(residual.data_ptr()),
              reinterpret_cast<const DType*>(weight.data_ptr()),
              reinterpret_cast<fp8_e4m3_t*>(output.data_ptr()),
              reinterpret_cast<const float*>(scale.data_ptr()),
              vec_hidden_size,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
