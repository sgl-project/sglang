#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <torch/extension.h>

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

namespace cute {
template <typename T>
__device__ inline T max_func(T a, T b) {
  return ::max(a, b);
}
template <>
__device__ inline nv_half max_func(nv_half a, nv_half b) {
  return __hmax(a, b);
}
template <>
__device__ inline nv_bfloat16 max_func(nv_bfloat16 a, nv_bfloat16 b) {
  return __hmax(a, b);
}
template <typename T>
__device__ inline T min_func(T a, T b) {
  return ::min(a, b);
}
template <>
__device__ inline nv_half min_func(nv_half a, nv_half b) {
  return __hmin(a, b);
}
template <>
__device__ inline nv_bfloat16 min_func(nv_bfloat16 a, nv_bfloat16 b) {
  return __hmin(a, b);
}
template <typename T>
__device__ inline T abs_func(T val) {
  return ::abs(val);
}
template <>
__device__ inline nv_half abs_func(nv_half val) {
  return __habs(val);
}
template <>
__device__ inline nv_bfloat16 abs_func(nv_bfloat16 val) {
  return __habs(val);
}

template <typename T, int N>
__device__ inline T warp_reduce_max(T val) {
  // perform warp_reduce_max with N groups, N should be power of 2
  // e.g. N=2, with 2 groups: thread0~15 and thread16~31
  static_assert(16 % N == 0);
  CUTE_UNROLL
  for (int offset = 16 / N; offset > 0; offset >>= 1) {
    T shfl_val = __shfl_xor_sync(0xffffffff, val, offset);
    val = max_func(val, shfl_val);
  }
  return val;
}

template <typename T>
__device__ inline T convert_scale(float scale) {
  // store ue8m0 scale as float if output_scale dtype is torch.float
  return scale;
}

template <>
__device__ inline uint8_t convert_scale(float scale) {
  return (uint8_t)(((int)log2f(scale)) + 127);
}

template <
    typename TI,
    typename TO,
    typename TS,
    typename ProbShape,
    typename InStride,
    typename OutStride,
    typename ScaleLayout,
    typename WarpTiler>
__global__ void per_token_group_quant_with_prologue_kernel(
    TO* __restrict__ out_ptr,
    TS* __restrict__ scale_ptr,
    const TI* __restrict__ in_ptr,
    ProbShape shape,
    InStride in_stride,
    OutStride out_stride,
    ScaleLayout scale_layout,
    WarpTiler warp_tiler,
    float eps,
    float min_val,
    float max_val,
    bool scale_ue8m0) {
  int global_warp_idx = threadIdx.x / 32 + blockIdx.y * blockDim.x / 32;
  int lane_idx = threadIdx.x % 32;
  if (global_warp_idx * get<0>(warp_tiler) >= get<1>(shape)) {
    return;
  }
  constexpr int VEC_SIZE = get<0>(warp_tiler) / 32;
  auto copy_in = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint_byte_t<VEC_SIZE * sizeof(TI)>>, TI>{},
      Layout<Shape<_32>>{},
      Layout<Shape<Int<VEC_SIZE>>>{});
  auto copy_out = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint_byte_t<VEC_SIZE * sizeof(TO)>>, TO>{},
      Layout<Shape<_32>>{},
      Layout<Shape<Int<VEC_SIZE>>>{});

  auto in = make_tensor(make_gmem_ptr(in_ptr), shape, in_stride)(blockIdx.x, _);
  auto warp_in = local_tile(in, warp_tiler, global_warp_idx);
  auto thd_in = copy_in.get_slice(lane_idx).partition_S(warp_in);
  auto reg_in = make_tensor_like(thd_in);
  copy(copy_in, thd_in, reg_in);

  // compute scale
  TI abs_max(eps);
  CUTE_UNROLL
  for (int i = 0; i < size(reg_in); i += 2) {
    abs_max = max_func(abs_max, abs_func(reg_in(i)));
  }
  abs_max = warp_reduce_max<TI, 1>(abs_max);
  float y_s = float(abs_max) / float(max_val);
  if (scale_ue8m0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  // store scale
  auto scale = make_tensor(scale_ptr, scale_layout);
  if (lane_idx == 0) {
    scale(blockIdx.x, global_warp_idx) = convert_scale<TS>(y_s);
  }

  // quant and store output
  auto out = make_tensor(make_gmem_ptr(out_ptr), shape, out_stride)(blockIdx.x, _);
  auto warp_out = local_tile(out, warp_tiler, global_warp_idx);
  auto thd_out = copy_out.get_slice(lane_idx).partition_D(warp_out);
  auto reg_out = make_tensor_like(thd_out);

  CUTE_UNROLL
  for (int i = 0; i < size(reg_out); ++i) {
    reg_out(i) = TO(min_func(max_func(float(reg_in(i)) / y_s, min_val), max_val));
  }
  copy(copy_out, reg_out, thd_out);
}

template <typename TI, typename TO, typename TS, int GROUP_SIZE>
void per_token_group_quant_with_prologue(
    void* output_q,
    void* output_scale,
    void* input,
    int token_count,
    int hidden_dim,
    int in_hidden_stride,
    int out_hidden_stride,
    int scale_stride0,
    int scale_stride1,
    float eps,
    float min_val,
    float max_val,
    bool scale_ue8m0,
    cudaStream_t stream) {
  static_assert(GROUP_SIZE % 32 == 0);
  constexpr int VEC_SIZE = GROUP_SIZE / 32;
  auto shape = make_shape(token_count, hidden_dim);
  auto in_stride = make_stride(in_hidden_stride, _1{});
  auto out_stride = make_stride(out_hidden_stride, _1{});

  int num_threads = 128;
  dim3 grid(token_count, (hidden_dim + VEC_SIZE * num_threads - 1) / (VEC_SIZE * num_threads));
  auto warp_tiler = make_shape(Int<VEC_SIZE * 32>{});
  if (scale_ue8m0 && std::is_same<TS, uint8_t>::value) {
    auto scale_layout = make_layout(
        make_shape(token_count, make_shape(_4{}, ceil_div(hidden_dim / GROUP_SIZE, 4))),
        make_stride(_4{}, make_stride(scale_stride0, 4 * scale_stride1)));
    per_token_group_quant_with_prologue_kernel<<<grid, num_threads, 0, stream>>>(
        (TO*)output_q,
        (TS*)output_scale,
        (TI*)input,
        shape,
        in_stride,
        out_stride,
        scale_layout,
        warp_tiler,
        eps,
        min_val,
        max_val,
        scale_ue8m0);
  } else {
    auto scale_layout =
        make_layout(make_shape(token_count, hidden_dim / GROUP_SIZE), make_stride(scale_stride0, scale_stride1));
    per_token_group_quant_with_prologue_kernel<<<grid, num_threads, 0, stream>>>(
        (TO*)output_q,
        (TS*)output_scale,
        (TI*)input,
        shape,
        in_stride,
        out_stride,
        scale_layout,
        warp_tiler,
        eps,
        min_val,
        max_val,
        scale_ue8m0);
  }
}

}  // namespace cute

namespace {
template <typename T>
struct ScaleTypeConvert {
  using type = T;
};

template <>
struct ScaleTypeConvert<int> {
  using type = uint8_t;
};

template <typename T>
struct TorchTypeToNvType {
  using type = T;
};

template <>
struct TorchTypeToNvType<c10::BFloat16> {
  using type = __nv_bfloat16;
};

template <>
struct TorchTypeToNvType<c10::Half> {
  using type = nv_half;
};

template <>
struct TorchTypeToNvType<c10::Float8_e4m3fn> {
  using type = __nv_fp8_e4m3;
};

union ComposedKey {
  struct {
    at::ScalarType type_in;
    at::ScalarType type_out;
    at::ScalarType type_scale;
    at::ScalarType _unused;
    int group_size;
  } keys;
  int64_t composed;
};

class PerTokenGroupQuantDispatcher {
  using FuncPtr = decltype(&cute::per_token_group_quant_with_prologue<float, int8_t, float, 128>);

 public:
  PerTokenGroupQuantDispatcher() {
    register_func<
        std::size(supported_in_types) - 1,
        std::size(supported_out_types) - 1,
        std::size(supported_scale_types) - 1,
        std::size(supported_group_size) - 1>(dispatch_map);
  }

  FuncPtr
  get_kernel_launcher(at::ScalarType type_in, at::ScalarType type_out, at::ScalarType type_scale, int group_size) {
    static std::unordered_map<int64_t, FuncPtr> dispatch_map;
    if (dispatch_map.empty()) {
      register_func<
          std::size(supported_in_types) - 1,
          std::size(supported_out_types) - 1,
          std::size(supported_scale_types) - 1,
          std::size(supported_group_size) - 1>(dispatch_map);
    }
    ComposedKey composed_key;
    composed_key.keys.type_in = type_in;
    composed_key.keys.type_out = type_out;
    composed_key.keys.type_scale = type_scale;
    composed_key.keys.group_size = group_size;

    return dispatch_map.at(composed_key.composed);
  }

 private:
  static constexpr at::ScalarType supported_in_types[] = {
      at::ScalarType::Float,
#ifdef FLASHINFER_ENABLE_F16
      at::ScalarType::Half,
#endif
#ifdef FLASHINFER_ENABLE_BF16
      at::ScalarType::BFloat16,
#endif
  };
  static constexpr at::ScalarType supported_out_types[] = {
#ifdef FLASHINFER_ENABLE_FP8_E4M3
      // for FP8
      at::ScalarType::Float8_e4m3fn,
#endif
      // for Int8
      at::ScalarType::Char,
  };
  static constexpr at::ScalarType supported_scale_types[] = {
      at::ScalarType::Float,
      // for UE8M0 packed
      at::ScalarType::Int,
  };
  static constexpr int supported_group_size[] = {32, 64, 128};
  std::unordered_map<int64_t, FuncPtr> dispatch_map;

  template <at::ScalarType type_in, at::ScalarType type_out, at::ScalarType type_scale, int group_size>
  FuncPtr get_func_by_params() {
    using TI_TORCH = typename c10::impl::ScalarTypeToCPPType<type_in>::type;
    using TI = typename TorchTypeToNvType<TI_TORCH>::type;
    using TO_TORCH = typename c10::impl::ScalarTypeToCPPType<type_out>::type;
    using TO = typename TorchTypeToNvType<TO_TORCH>::type;
    using TS_PACKED = typename c10::impl::ScalarTypeToCPPType<type_scale>::type;
    using TS = typename ScaleTypeConvert<TS_PACKED>::type;
    return &cute::per_token_group_quant_with_prologue<TI, TO, TS, group_size>;
  }

  template <int TYPE_IN_IDX, int TYPE_OUT_IDX, int TYPE_SCALE_IDX, int GROUP_SIZE_IDX>
  void register_func(std::unordered_map<int64_t, FuncPtr>& dispatch_map) {
    constexpr auto type_in = supported_in_types[TYPE_IN_IDX];
    constexpr auto type_out = supported_out_types[TYPE_OUT_IDX];
    constexpr auto type_scale = supported_scale_types[TYPE_SCALE_IDX];
    constexpr auto group_size = supported_group_size[GROUP_SIZE_IDX];
    ComposedKey composed_key;
    composed_key.keys.type_in = type_in;
    composed_key.keys.type_out = type_out;
    composed_key.keys.type_scale = type_scale;
    composed_key.keys.group_size = group_size;
    dispatch_map[composed_key.composed] = get_func_by_params<type_in, type_out, type_scale, group_size>();
    if constexpr (TYPE_IN_IDX > 0) {
      register_func<TYPE_IN_IDX - 1, TYPE_OUT_IDX, TYPE_SCALE_IDX, GROUP_SIZE_IDX>(dispatch_map);
    } else if constexpr (TYPE_OUT_IDX > 0) {
      register_func<std::size(supported_in_types) - 1, TYPE_OUT_IDX - 1, TYPE_SCALE_IDX, GROUP_SIZE_IDX>(dispatch_map);
    } else if constexpr (TYPE_SCALE_IDX > 0) {
      register_func<
          std::size(supported_in_types) - 1,
          std::size(supported_out_types) - 1,
          TYPE_SCALE_IDX - 1,
          GROUP_SIZE_IDX>(dispatch_map);
    } else if constexpr (GROUP_SIZE_IDX > 0) {
      register_func<
          std::size(supported_in_types) - 1,
          std::size(supported_out_types) - 1,
          std::size(supported_scale_types) - 1,
          GROUP_SIZE_IDX - 1>(dispatch_map);
    }
  }
};

PerTokenGroupQuantDispatcher dispatcher;

}  // namespace

void check_contiguous_except_last(torch::Tensor x) {
  CHECK(x.stride(x.dim() - 1) == 1);
  for (int i = 0; i < x.dim() - 3; ++i) {
    CHECK(x.stride(i) / x.stride(i + 1) == x.size(i + 1));
  }
}

void sgl_per_token_group_quant_8bit(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_scale,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0) {
  // dispatch different impl by input.dtype, output_q.dtype, output_scale.dtype and group_size
  check_contiguous_except_last(input);
  check_contiguous_except_last(output_q);

  int in_hidden_dim = input.size(input.dim() - 1);
  int out_hidden_dim = output_q.size(output_q.dim() - 1);
  CHECK(in_hidden_dim == out_hidden_dim);
  CHECK(in_hidden_dim % group_size == 0);

  int in_token_count = input.numel() / in_hidden_dim;
  int out_token_count = output_q.numel() / out_hidden_dim;
  CHECK(in_token_count == out_token_count);

  int in_hidden_stride = input.stride(input.dim() - 2);
  int out_hidden_stride = output_q.stride(output_q.dim() - 2);

  CHECK(output_scale.dim() == 2);
  int scale_hidden_dim = output_scale.size(1);

  int output_scale_stride0 = output_scale.stride(0);
  int output_scale_stride1 = output_scale.stride(1);

  auto kernel_launcher = dispatcher.get_kernel_launcher(
      input.scalar_type(), output_q.scalar_type(), output_scale.scalar_type(), group_size);
  kernel_launcher(
      output_q.data_ptr(),
      output_scale.data_ptr(),
      input.data_ptr(),
      in_token_count,
      in_hidden_dim,
      in_hidden_stride,
      out_hidden_stride,
      output_scale_stride0,
      output_scale_stride1,
      eps,
      min_8bit,
      max_8bit,
      scale_ue8m0,
      at::cuda::getCurrentCUDAStream());
}
