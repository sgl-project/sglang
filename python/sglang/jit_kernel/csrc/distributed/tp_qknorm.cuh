// Adapted from https://github.com/NVIDIA/TensorRT-LLM/pull/12163
// We reuse the custom all reduce push buffer in SGLang
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/common.cuh>
#include <sgl_kernel/distributed/custom_all_reduce.cuh>

#include <cstdint>
#include <cstring>

namespace {

using device::distributed::PushController;
using host::distributed::CustomAllReduceBase, host::distributed::CustomAllReduceRef;

struct ParallelQKNormParams {
  void* __restrict__ buffer[device::distributed::kMaxNumGPU];
  void* q_ptr;
  void* k_ptr;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  float eps;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t epoch_bytes;
  uint32_t num_clean_up_count = 0;
};

template <typename T>
SGL_DEVICE void ld_global_volatile_8B(T& x, const void* addr, int64_t offset) {
  static_assert(alignof(T) == 8 && sizeof(T) == 8);
  addr = device::pointer::offset<T>(addr, offset);
  uint2 val;
  asm volatile("ld.volatile.global.v2.b32 {%0, %1}, [%2];" : "=r"(val.x), "=r"(val.y) : "l"(addr));
  x = *reinterpret_cast<const T*>(&val);
}

template <typename T>
SGL_DEVICE void st_global_volatile_8B(const T& x, void* addr, int64_t offset) {
  static_assert(alignof(T) == 8 && sizeof(T) == 8);
  const uint2 val = *reinterpret_cast<const uint2*>(&x);
  addr = device::pointer::offset<T>(addr, offset);
  asm volatile("st.volatile.global.v2.b32 [%2], {%0, %1};" ::"r"(val.x), "r"(val.y), "l"(addr));
}

[[maybe_unused]]
SGL_DEVICE float sync_float(float x) {
  return __shfl_sync(0xffffffffu, x, 0);
}

[[maybe_unused]]
constexpr auto next_pow_of_2(uint32_t x) {
  uint32_t y = 1;
  while (y < x)
    y *= 2;
  return y;
}

template <typename DType_, uint32_t kNumGPU_, int64_t kQDim_, int64_t kKDim_, bool kUsePDL_>
struct KernelTrait {
  // rename the arguments to avoid confusion with the template parameters
  using DType = DType_;
  static constexpr uint32_t kNumGPU = kNumGPU_;
  static constexpr int64_t kQDim = kQDim_;
  static constexpr int64_t kKDim = kKDim_;
  static constexpr bool kUsePDL = kUsePDL_;

  static constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  static constexpr int64_t kLocalQDim = kQDim / kNumGPU;
  static constexpr int64_t kLocalKDim = kKDim / kNumGPU;
  static constexpr uint32_t kNumQThreads = kLocalQDim / (kVecSize * 2);
  static constexpr uint32_t kNumKThreads = kLocalKDim / (kVecSize * 2);
  static constexpr uint32_t kNumQWarps = kNumQThreads / device::kWarpThreads;
  static constexpr uint32_t kNumKWarps = host::div_ceil(kNumKThreads, device::kWarpThreads);
  static constexpr uint32_t kBlockSize = (kNumQWarps + kNumKWarps) * device::kWarpThreads;
  static constexpr uint32_t kOccupancy = 2048 / kBlockSize;

  using DType2 = packed_t<DType>;
  using Storage = device::AlignedVector<DType2, kVecSize>;

  static_assert(std::has_single_bit(kNumGPU), "must be pow of 2");
  static_assert(kQDim % kNumGPU == 0);
  static_assert(kKDim % kNumGPU == 0);
  static_assert(kLocalQDim % (kVecSize * 2) == 0);
  static_assert(kLocalKDim % (kVecSize * 2) == 0);
  static_assert(kNumQThreads % device::kWarpThreads == 0);
  static_assert(kBlockSize <= 1024);
  static_assert(sizeof(Storage) == 16 && alignof(Storage) == 16);
  static_assert(kOccupancy * kBlockSize <= 2048);
};

template <typename Trait>
__global__ __launch_bounds__(Trait::kBlockSize, Trait::kOccupancy) void parallel_qknorm_across_head(
    const ParallelQKNormParams __grid_constant__ params, const PushController __grid_constant__ ctrl) {
  using namespace device;

  // each cta will handle exactly 1 token
  using Storage = typename Trait::Storage;
  using DType2 = typename Trait::DType2;
  const auto &[
      buffer, q_ptr, k_ptr, q_weight, k_weight, q_stride_bytes, k_stride_bytes, //
      eps, rank, num_tokens, epoch_bytes, num_clean_up_count
  ] = params;

  using Package = AlignedVector<float, 2>;
  constexpr uint32_t kNumGPU = Trait::kNumGPU;
  constexpr uint32_t kNumQReduce = next_pow_of_2(Trait::kNumQWarps);
  constexpr uint32_t kNumKReduce = next_pow_of_2(Trait::kNumKWarps);
  __shared__ float smem_qk[Trait::kNumQWarps + Trait::kNumKWarps];
  __shared__ float scale_q;
  __shared__ float scale_k;
  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  /// NOTE: this can hint compiler to optimize `is_valid` out when not needed
  constexpr uint32_t kActiveThreads = Trait::kNumQThreads + Trait::kNumKThreads;
  const auto is_valid = Trait::kBlockSize == kActiveThreads || tx < kActiveThreads;
  const auto smem_q = smem_qk + 0;
  const auto smem_k = smem_qk + Trait::kNumQWarps;
  const auto load_q = tx < Trait::kNumQThreads;
  const auto offset = load_q ? tx : tx - Trait::kNumQThreads;
  const auto input_ptr = load_q ? q_ptr : k_ptr;
  const auto weight_ptr = load_q ? q_weight : k_weight;
  const auto input_stride_bytes = load_q ? q_stride_bytes : k_stride_bytes;
  PDLWaitPrimary<Trait::kUsePDL>();
  PDLTriggerSecondary<Trait::kUsePDL>();
  if (bx >= num_tokens) {
    [[unlikely]];
    // In this case, we use the last few blocks to clean up other controllers
    const auto start = (bx - num_tokens) * blockDim.x + threadIdx.x;
    const auto stride = (gridDim.x - num_tokens) * blockDim.x;
    for (uint32_t i = start; i < num_clean_up_count; i += stride)
      ctrl.exit_unsafe(num_tokens + i);
    return;
  }
  const auto epoch_offset = ctrl.epoch() * epoch_bytes;  // only for comm

  __builtin_assume(bx < num_tokens);  // since we have `bx >= num_tokens`
  Storage next_input;
  void* input_i_ptr = pointer::offset(input_ptr, bx * input_stride_bytes);
  if (is_valid) next_input.load(input_i_ptr, offset);

  for (uint32_t i = bx; i < num_tokens; i += gridDim.x) {
    // Stage 1. local reduce (warp-level)
    Storage local_input;
    {
      float local_sum = 0.0;
      if (is_valid) {
        local_input = next_input;
#pragma unroll
        for (uint32_t j = 0; j < Trait::kVecSize; ++j) {
          const auto [x, y] = cast<fp32x2_t>(local_input[j]);
          local_sum += x * x + y * y;
        }
      }
      smem_qk[threadIdx.x / kWarpThreads] = warp::reduce_sum(local_sum);
    }

    // Stage 2. block reduce + push to peer ranks + poll from local rank
    __syncthreads();

    Storage local_weight;
    const auto input_next_ptr = pointer::offset(input_i_ptr, gridDim.x * input_stride_bytes);
    /**
     * NOTE: Prefetch to hide the latency.
     * This brings around 20% of performance gain in large batches
     * The P2P communication is mainly latency bound, so during this waiting period,
     * We can let some data loading transparently in the background.
     */
    if (is_valid) {
      local_weight.load(weight_ptr, offset);
      if (i + gridDim.x < num_tokens) next_input.load(input_next_ptr, offset);
    }

    if (tx < kWarpThreads) {
      const auto local_sum_q = tx < Trait::kNumQWarps ? smem_q[tx] : 0.0f;
      const auto local_sum_k = tx < Trait::kNumKWarps ? smem_k[tx] : 0.0f;
      const auto sum_q = sync_float(warp::reduce_sum<kNumQReduce>(local_sum_q));
      const auto sum_k = sync_float(warp::reduce_sum<kNumKReduce>(local_sum_k));
      if (tx < kNumGPU) {  // push a float2 pack to the peer
        Package sum_q_k;
        /// NOTE: eps should be scaled down by kNumGPU from host side
        /// we add here to ensure that the sum is never zero
        sum_q_k[0] = sum_q + eps;
        sum_q_k[1] = sum_k + eps;
        const auto push_ptr = pointer::offset(buffer[tx], epoch_offset);
        st_global_volatile_8B(sum_q_k, push_ptr, i * kNumGPU + rank);
        const auto poll_ptr = pointer::offset(buffer[rank], epoch_offset);
        while (true) {
          ld_global_volatile_8B(sum_q_k, poll_ptr, i * kNumGPU + tx);
          if (sum_q_k[0] != 0.0f && sum_q_k[1] != 0.0f) break;
        }
        constexpr uint32_t kActiveMask = (1 << kNumGPU) - 1;
        const auto global_sum_q = warp::reduce_sum<kNumGPU>(sum_q_k[0], kActiveMask);
        const auto global_sum_k = warp::reduce_sum<kNumGPU>(sum_q_k[1], kActiveMask);
        scale_q = math::rsqrt(global_sum_q / static_cast<float>(Trait::kQDim));
        scale_k = math::rsqrt(global_sum_k / static_cast<float>(Trait::kKDim));
        Package zeros;
        zeros.fill(0.0f);
        zeros.store(poll_ptr, i * kNumGPU + tx);
      }
    }

    __syncthreads();
    const auto scale = load_q ? scale_q : scale_k;
    if (is_valid) {
#pragma unroll
      for (uint32_t j = 0; j < Trait::kVecSize; ++j) {
        const auto fp32_input = cast<fp32x2_t>(local_input[j]);
        const auto fp32_weight = cast<fp32x2_t>(local_weight[j]);
        const auto scaled_x = fp32_input.x * scale * fp32_weight.x;
        const auto scaled_y = fp32_input.y * scale * fp32_weight.y;
        local_input[j] = cast<DType2>(fp32x2_t{scaled_x, scaled_y});
      }
      local_input.store(input_i_ptr, offset);
    }
    input_i_ptr = input_next_ptr;
  }
  ctrl.exit();
}

template <typename DType, uint32_t kNumGPU, int64_t kQDim, int64_t kKDim, bool kUsePDL>
struct FusedParallelQKNormAcrossHead : public CustomAllReduceBase {
  using Trait = KernelTrait<DType, kNumGPU, kQDim, kKDim, kUsePDL>;
  static constexpr auto kernel = parallel_qknorm_across_head<Trait>;
  static_assert(kNumGPU <= device::distributed::kMaxNumGPU, "kNumGPU exceeds the maximum supported GPUs");

  void _run(
      const tvm::ffi::Tensor q,
      const tvm::ffi::Tensor k,
      const tvm::ffi::Tensor q_weight,
      const tvm::ffi::Tensor k_weight,
      const float eps  // passed in unscaled
  ) {
    using namespace host;
    constexpr auto Q = Trait::kLocalQDim;
    constexpr auto K = Trait::kLocalKDim;
    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, Q})  // q
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q);
    TensorMatcher({N, K})  // k
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(k);
    TensorMatcher({Q})  // q_weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_weight);
    TensorMatcher({K})  // k_weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(k_weight);
    const auto device = device_.unwrap();
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    // use at most `world_size` blocks to clean up,
    // this is based on the observation that occupancy is usually linear
    // with respect to the world size
    const bool need_clean = num_tokens < m_max_num_cta_push;
    const auto num_clean = need_clean ? (m_max_num_cta_push - num_tokens) : 0;
    const auto num_blocks = need_clean ? num_tokens + div_ceil(num_clean, Trait::kBlockSize)  //
                                       : m_max_num_cta_push;                                  //
    const auto num_threads = Trait::kBlockSize;
    RuntimeCheck(num_blocks <= m_max_num_cta_push, "internal error");
    ParallelQKNormParams params;
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      params.buffer[i] = get_push_buffer(m_peer_storage[i]);
    }
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.q_weight = q_weight.data_ptr();
    params.k_weight = k_weight.data_ptr();
    params.q_stride_bytes = q.stride(0) * sizeof(DType);
    params.k_stride_bytes = k.stride(0) * sizeof(DType);
    params.eps = eps / kNumGPU;  // scale down eps by number of GPUs
    params.rank = m_rank;
    params.num_tokens = num_tokens;
    params.epoch_bytes = m_push_buffer_bytes;
    params.num_clean_up_count = num_clean;

    const auto needed_buffer_bytes = static_cast<int64_t>(num_tokens) * 2 * sizeof(float);
    RuntimeCheck(m_num_gpu == kNumGPU, "Number of GPUs mismatch");
    RuntimeCheck(m_push_ctrl.has_value(), "Controller is not initialized");
    RuntimeCheck(std::bit_cast<intptr_t>(params.q_ptr) % 16 == 0, "q pointer is not properly aligned");
    RuntimeCheck(std::bit_cast<intptr_t>(params.k_ptr) % 16 == 0, "k pointer is not properly aligned");
    RuntimeCheck(std::bit_cast<intptr_t>(params.q_weight) % 16 == 0, "q_weight pointer is not properly aligned");
    RuntimeCheck(std::bit_cast<intptr_t>(params.k_weight) % 16 == 0, "k_weight pointer is not properly aligned");
    RuntimeCheck(needed_buffer_bytes <= m_push_buffer_bytes, "Push buffer is too small");

    LaunchKernel(num_blocks, num_threads, device)  //
        .enable_pdl(kUsePDL)(kernel, params, *m_push_ctrl);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }

  static void
  run(CustomAllReduceRef obj,
      const tvm::ffi::Tensor q,
      const tvm::ffi::Tensor k,
      const tvm::ffi::Tensor q_weight,
      const tvm::ffi::Tensor k_weight,
      const float eps) {
    using Self = FusedParallelQKNormAcrossHead;
    return static_cast<Self*>(obj.get())->_run(q, k, q_weight, k_weight, eps);
  }
};

}  // namespace
