#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

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
SGL_DEVICE packed_t rms(const packed_t& val, const packed_t& weight, float rsqrt_square_sum) {
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

  __shared__ float shared_memory[32];

  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  vec_t v_data;
  vec_t v_weight;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int warp_count = (blockDim.x + 31) >> 5;
  const float inv_hidden_size = 1.0f / static_cast<float>(vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T)));
  const bool is_q = blockIdx.y == 0;

  const auto token_id = blockIdx.x;
  float2 acc_square = make_float2(0.0f, 0.0f);
  vec_t* data = reinterpret_cast<vec_t*>(is_q ? q : k) + token_id * vec_hidden_size;
  const vec_t* weight = reinterpret_cast<const vec_t*>(is_q ? q_weight : k_weight);

  if (threadIdx.x < vec_hidden_size) {
    v_data = data[threadIdx.x];
    v_weight = weight[threadIdx.x];
    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v_data[i]);
      acc_square.x += val.x * val.x;
      acc_square.y += val.y * val.y;
    }
  }

  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  float* buffer = shared_memory;
  float warp_sum = cooperative_groups::reduce(cg_warp, acc_square.x + acc_square.y, cooperative_groups::plus<float>());
  if (lane_id == 0) {
    buffer[warp_id] = warp_sum;
  }

  __syncthreads();
  if (threadIdx.x < 32) {
    float cta_sum = cooperative_groups::reduce(
        cg_warp, (threadIdx.x < warp_count) ? buffer[threadIdx.x] : 0.0f, cooperative_groups::plus<float>());
    if (threadIdx.x == 0) {
      buffer[0] = rsqrtf(eps + cta_sum * inv_hidden_size);
    }
  }
  __syncthreads();

  if (threadIdx.x < vec_hidden_size) {
    float rsqrt_val = buffer[0];
    for (int i = 0; i < inner_loop; i++) {
      v_data[i] = rms(v_data[i], v_weight[i], rsqrt_val);
    }
    data[threadIdx.x] = v_data;
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

      auto kernel = qknorm_across_heads_reg_kernel<DType, device::kMaxVecBytes>;

      LaunchKernel(dim3(static_cast<uint>(N.unwrap()), 2), threads, device.unwrap())
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
