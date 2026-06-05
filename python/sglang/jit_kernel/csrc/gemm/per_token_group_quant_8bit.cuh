#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreadsPerGroup = 16;

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <bool kScaleUE8M0>
using scale_packed_t_t = std::conditional_t<kScaleUE8M0, uint32_t, float>;

template <bool kScaleUE8M0>
using scale_element_t_t = std::conditional_t<kScaleUE8M0, uint8_t, float>;

template <typename T, typename DST_DTYPE, bool kIsColumnMajor, bool kScaleUE8M0>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    scale_packed_t_t<kScaleUE8M0>* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const int num_groups_per_row = 0,
    const int scale_stride = 0) {
  using namespace device;
  namespace math = device::math;

  (void)num_groups;

  const int local_group_id = static_cast<int>(threadIdx.x / kThreadsPerGroup);
  const int lane_id = threadIdx.x % kThreadsPerGroup;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  using scale_packed_t = scale_packed_t_t<kScaleUE8M0>;
  using scale_element_t = scale_element_t_t<kScaleUE8M0>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output = nullptr;

  if constexpr (kIsColumnMajor) {
    constexpr int kElemsPerPack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int row_idx = global_group_id / num_groups_per_row;
    const int col_idx_unpacked = global_group_id % num_groups_per_row;
    const int col_idx = col_idx_unpacked / kElemsPerPack;
    const int pack_idx = col_idx_unpacked % kElemsPerPack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * kElemsPerPack + row_idx * kElemsPerPack + pack_idx);
  } else {
    static_assert(!kScaleUE8M0);
    scale_output = output_s + global_group_id;
  }

  constexpr uint32_t kVecSize = 16 / sizeof(T);
  using vec_t = AlignedVector<T, kVecSize>;
  const auto gmem_in = tile::Memory<vec_t>::thread();

  const int32_t num_vec_elems = group_size / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
    const vec_t input_vec = gmem_in.load(group_input, i);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const float val = static_cast<float>(input_vec[j]);
      local_absmax = math::max(local_absmax, math::abs(val));
    }
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);

  float y_s = local_absmax / max_8bit;
  if constexpr (kScaleUE8M0) {
    y_s = exp2f(ceilf(log2f(math::max(y_s, 1e-10f))));
  }

  scale_element_t y_s_quant;
  if constexpr (kScaleUE8M0) {
    y_s_quant = static_cast<uint8_t>(((int)log2f(y_s)) + 127);
  } else {
    y_s_quant = y_s;
  }

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += kThreadsPerGroup) {
    const vec_t input_vec = gmem_in.load(group_input, i);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const float val = static_cast<float>(input_vec[j]);
      const float q_val = math::min(math::max(val / y_s, min_8bit), max_8bit);
      group_output[i * kVecSize + j] = DST_DTYPE(q_val);
    }
  }
}

inline int compute_groups_per_block(int64_t num_groups) {
  if (num_groups % 16 == 0) return 16;
  if (num_groups % 8 == 0) return 8;
  if (num_groups % 4 == 0) return 4;
  if (num_groups % 2 == 0) return 2;
  return 1;
}

template <typename DType, typename OutType>
void per_token_group_quant_8bit(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output_q,
    tvm::ffi::TensorView output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"hidden_dim"};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, K}).with_dtype<DType>().with_device(device).verify(input);
  TensorMatcher({M, K}).with_dtype<OutType>().with_device(device).verify(output_q);

  const auto num_tokens = M.unwrap();
  const auto hidden_dim = K.unwrap();

  const int64_t num_groups_per_row = hidden_dim / group_size;
  const int64_t num_groups = num_tokens * num_groups_per_row;

  const int groups_per_block = compute_groups_per_block(num_groups);
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * kThreadsPerGroup;
  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_stride = output_s.stride(1);

  const float feps = static_cast<float>(eps);
  const float fmin8 = static_cast<float>(min_8bit);
  const float fmax8 = static_cast<float>(max_8bit);

  if (is_column_major) {
    if (scale_ue8m0) {
      LaunchKernel(num_blocks, num_threads, input.device())(
          per_token_group_quant_8bit_kernel<DType, OutType, true, true>,
          static_cast<const DType*>(input.data_ptr()),
          static_cast<OutType*>(output_q.data_ptr()),
          static_cast<uint32_t*>(output_s.data_ptr()),
          static_cast<int>(group_size),
          static_cast<int>(num_groups),
          static_cast<int>(groups_per_block),
          feps,
          fmin8,
          fmax8,
          static_cast<int>(num_groups_per_row),
          scale_stride);
    } else {
      LaunchKernel(num_blocks, num_threads, input.device())(
          per_token_group_quant_8bit_kernel<DType, OutType, true, false>,
          static_cast<const DType*>(input.data_ptr()),
          static_cast<OutType*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          static_cast<int>(group_size),
          static_cast<int>(num_groups),
          static_cast<int>(groups_per_block),
          feps,
          fmin8,
          fmax8,
          static_cast<int>(num_groups_per_row),
          scale_stride);
    }
  } else {
    LaunchKernel(num_blocks, num_threads, input.device())(
        per_token_group_quant_8bit_kernel<DType, OutType, false, false>,
        static_cast<const DType*>(input.data_ptr()),
        static_cast<OutType*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<int>(group_size),
        static_cast<int>(num_groups),
        static_cast<int>(groups_per_block),
        feps,
        fmin8,
        fmax8,
        0,
        0);
  }
}
}  // namespace
