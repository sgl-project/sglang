#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <cooperative_groups/reduce.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <type_traits>

namespace {

union U16B_bf162 {
  int4 load_store_unit;
  __nv_bfloat162 compute_unit[4];
};

union U16B_f162 {
  int4 load_store_unit;
  __half2 compute_unit[4];
};

union U32B_bf162 {
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a load_store_unit;
#else
  longlong4 load_store_unit;
#endif
  __nv_bfloat162 compute_unit[8];
};

union U32B_f162 {
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a load_store_unit;
#else
  longlong4 load_store_unit;
#endif
  __half2 compute_unit[8];
};

template <typename T, int VEC_SIZE_IN_BYTE>
struct UVTypeTrait;

template <>
struct UVTypeTrait<__nv_bfloat16, 16> {
  using U = U16B_bf162;
  using V = int4;
};

template <>
struct UVTypeTrait<__half, 16> {
  using U = U16B_f162;
  using V = int4;
};

template <>
struct UVTypeTrait<__nv_bfloat16, 32> {
  using U = U32B_bf162;
#if __CUDACC_VER_MAJOR__ >= 13
  using V = longlong4_32a;
#else
  using V = longlong4;
#endif
};

template <>
struct UVTypeTrait<__half, 32> {
  using U = U32B_f162;
#if __CUDACC_VER_MAJOR__ >= 13
  using V = longlong4_32a;
#else
  using V = longlong4;
#endif
};

template <typename T>
SGL_DEVICE T rms(T& val, T& weight, float rsqrt_square_sum);

template <>
SGL_DEVICE __nv_bfloat162 rms<__nv_bfloat162>(__nv_bfloat162& val, __nv_bfloat162& weight, float rsqrt_square_sum) {
  float2 valf = __bfloat1622float2(val);
  float2 weightf = __bfloat1622float2(weight);
  return __float22bfloat162_rn(
      make_float2(valf.x * weightf.x * rsqrt_square_sum, valf.y * weightf.y * rsqrt_square_sum));
}

template <>
SGL_DEVICE __half2 rms<__half2>(__half2& val, __half2& weight, float rsqrt_square_sum) {
  float2 valf = __half22float2(val);
  float2 weightf = __half22float2(weight);
  return __float22half2_rn(make_float2(valf.x * weightf.x * rsqrt_square_sum, valf.y * weightf.y * rsqrt_square_sum));
}

template <typename T, int VEC_SIZE_IN_BYTE>
__global__ void fused_add_rmsnorm_reg_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    uint tokens,
    int vec_hidden_size,
    float eps) {
  using U = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::U;
  using V = typename UVTypeTrait<T, VEC_SIZE_IN_BYTE>::V;
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;

  __shared__ float shared_memory[32];  // Used for CTA reduce

  U u;         // Save input
  U u_res;     // Save residual
  U u_weight;  // Save weight
  U u_out;     // Save output

  auto token_id = blockIdx.x;
  float2 acc_square = make_float2(0.0f, 0.0f);  // Sum of squares for each thread

  if (threadIdx.x < vec_hidden_size) {
    // Compute address
    V* p = reinterpret_cast<V*>(input) + token_id * vec_hidden_size;
    V* p_res = reinterpret_cast<V*>(residual) + token_id * vec_hidden_size;
    const V* p_weight = reinterpret_cast<const V*>(weight);

    // Load data
    u.load_store_unit = p[threadIdx.x];
    u_res.load_store_unit = p_res[threadIdx.x];
    if constexpr (std::is_same_v<V, int4>) {
      u_weight.load_store_unit = __ldg(&p_weight[threadIdx.x]);
    } else {
      u_weight.load_store_unit = p_weight[threadIdx.x];  // The longlong4_32a has no overloaded __ldg
    }

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      for (int i = 0; i < inner_loop; i++) {
        float2 val = __bfloat1622float2(u.compute_unit[i]);
        float2 res = __bfloat1622float2(u_res.compute_unit[i]);
        float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
        acc_square.x += inp_res.x * inp_res.x;
        acc_square.y += inp_res.y * inp_res.y;
        u.compute_unit[i] = __float22bfloat162_rn(inp_res);
      }
    } else if constexpr (std::is_same_v<T, __half>) {
      for (int i = 0; i < inner_loop; i++) {
        float2 val = __half22float2(u.compute_unit[i]);
        float2 res = __half22float2(u_res.compute_unit[i]);
        float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
        acc_square.x += inp_res.x * inp_res.x;
        acc_square.y += inp_res.y * inp_res.y;
        u.compute_unit[i] = __float22half2_rn(inp_res);
      }
    }

    // Store inp+res to residual
    p_res[threadIdx.x] = u.load_store_unit;
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
      u_out.compute_unit[i] = rms(u.compute_unit[i], u_weight.compute_unit[i], rsqrt_square_sum);
    }
    V* p_out = reinterpret_cast<V*>(input) + token_id * vec_hidden_size;
    p_out[threadIdx.x] = u_out.load_store_unit;
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
              static_cast<uint>(D.unwrap()),
              vec_hidden_size,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
