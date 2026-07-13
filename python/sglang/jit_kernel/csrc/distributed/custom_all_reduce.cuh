// Custom all-reduce kernels over the decoupled Communicator storage plane.
//
// Three algorithms are provided behind one entry point:
//   - 1shot_push: lamport-style push of local data to every peer's push
//     workspace, then a local polling reduce (best at small sizes).
//   - 1shot_pull: every rank reduces all peers' data (from the symmetric pull
//     workspaces, a CUDA-graph pointer table, or a multicast address).
//   - 2shot_pull: reduce-scatter fused with all-gather; each rank reduces its
//     shard in place so every workspace ends up holding the full result.
//
// Unlike the previous implementation, the kernels carry no storage or IPC
// logic: all pointers arrive via `CommunicatorObj` (owned by Python) and the
// per-call `AllReduceParams`.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/extra/stl.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>

namespace {

using device::distributed::Counter, device::distributed::Semaphore;
using host::distributed::CommunicatorRef;

inline constexpr uint32_t kMaxWorldSize = device::distributed::kMaxWorldSize;

enum class PullMode {
  Graph,
  Eager,
  Multicast,  // also eager
};

template <typename T>
struct fp_trait {};

template <>
struct fp_trait<bf16_t> {
  using type = uint16_t;
  [[maybe_unused]]
  static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]]
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<fp16_t> {
  using type = uint16_t;
  [[maybe_unused]]
  static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]]
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  [[maybe_unused]]
  static constexpr uint32_t pos_zero = 0x00000000u;
  [[maybe_unused]]
  static constexpr uint32_t neg_zero = 0x80000000u;
};

template <typename DType>
SGL_DEVICE void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) *ptr = Trait::neg_zero;
}

template <typename DType>
SGL_DEVICE bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
SGL_DEVICE DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}

template <typename T2, size_t N, size_t M>
SGL_DEVICE auto reduce(device::AlignedVector<T2, N> (&vec)[M]) -> device::AlignedVector<T2, N> {
  fp32x2_t acc_vec[N];
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      const auto [x, y] = device::cast<fp32x2_t>(vec[i][j]);
      auto& [acc_x, acc_y] = acc_vec[j];
      acc_x = i == 0 ? x : acc_x + x;
      acc_y = i == 0 ? y : acc_y + y;
    }
  }
  device::AlignedVector<T2, N> out_vec;
#pragma unroll
  for (size_t j = 0; j < N; ++j) {
    out_vec[j] = device::cast<T2>(acc_vec[j]);
  }
  return out_vec;
}

template <typename V>
SGL_DEVICE void ld_global_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint4 val;
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_global_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :  //
               : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

template <typename V>
SGL_DEVICE void ld_relaxed_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint4 val;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_relaxed_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :  //
               : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

template <typename V>
SGL_DEVICE void ld_multimem_16B(V& x, const void* mc_addr, int64_t vec_offset) {
#if SGL_ARCH_HOPPER_OR_GREATER
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  mc_addr = static_cast<const uint8_t*>(mc_addr) + vec_offset * 16;
  float4 val;
  if constexpr (std::is_same_v<V, device::AlignedVector<fp32x2_t, 2>>) {
    asm volatile("multimem.ld_reduce.weak.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(mc_addr));
  } else if constexpr (std::is_same_v<V, device::AlignedVector<fp16x2_t, 4>>) {
    asm volatile("multimem.ld_reduce.weak.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(mc_addr));
  } else {
    static_assert(std::is_same_v<V, device::AlignedVector<bf16x2_t, 4>>);  // 4x bf16x2
    asm volatile("multimem.ld_reduce.weak.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(mc_addr));
  }
  x = *reinterpret_cast<const V*>(&val);
#else
  assert(false && "multimem load is only supported on Hopper or later architecture");
#endif
}

template <typename V>
SGL_DEVICE void st_multimem_16B(const V& x, void* mc_addr, int64_t vec_offset) {
#if SGL_ARCH_HOPPER_OR_GREATER
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto val = *reinterpret_cast<const float4*>(&x);
  mc_addr = static_cast<uint8_t*>(mc_addr) + vec_offset * 16;
  asm volatile("multimem.st.weak.v4.f32 [%4], {%0, %1, %2, %3};"
               :
               : "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w), "l"(mc_addr));
#else
  assert(false && "multimem store is only supported on Hopper or later architecture");
#endif
}

struct AllReduceParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_elements;
  uint32_t rank;
  void* const* __restrict__ graph_params;
  uint8_t* pull_workspaces[kMaxWorldSize];    // must be symmetric memory
  uint8_t* push_workspaces[kMaxWorldSize];    // must be symmetric memory
  Semaphore* pull_semaphores[kMaxWorldSize];  // must be symmetric memory
  Counter* push_counter;
  uint8_t* pull_mc_workspace;  // must be a multicast address
  int64_t push_buffer_stride;  // per-buffer bytes; each rank holds 2 * kMaxWorldSize buffers
};

template <typename T, uint32_t kWorldSize, bool kUsePDL>
struct AllReducePushImpl {
 private:
  using T2 = packed_t<T>;
  /// NOTE: force 16B load/store to reduce register pressure
  static constexpr uint32_t kVecSize = 16 / sizeof(T2);
  static constexpr uint32_t kElemsPerVec = 16 / sizeof(T);
  using vec_t = device::AlignedVector<T2, kVecSize>;
  static_assert(kWorldSize <= kMaxWorldSize);

  static SGL_DEVICE bool sync_enter_push(const AllReduceParams& params) {
    device::PDLWaitPrimary<kUsePDL>();
    return (params.push_counter[blockIdx.x].get() % 2) != 0;
  }

  static SGL_DEVICE void sync_exit_push(const AllReduceParams& params) {
    device::PDLTriggerSecondary<kUsePDL>();
    __syncthreads();
    if (threadIdx.x == 0) {
      params.push_counter[blockIdx.x].inc(1);  // NOTE: u32 overflow is safe under mod 2
    }
  }

  static SGL_DEVICE void push_impl(uint32_t num_vecs, void* (&data)[kWorldSize], const void* src) {
    const auto num_threads = blockDim.x * gridDim.x;
    const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
      vec_t vec;
      ld_global_16B(vec, src, vid);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        clear_pos_zero(vec[j].x);
        clear_pos_zero(vec[j].y);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        st_relaxed_16B(vec, data[i], vid);
      }
    }
  }

  static SGL_DEVICE void poll_impl(uint32_t num_vecs, void* (&data)[kWorldSize], void* out) {
    // need polling to ensure data is ready
    const auto num_threads = blockDim.x * gridDim.x;
    const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // pos_zero-filled vec we write back after consuming each slot, so the
    // double-buffered phase comes back around with the "slot empty" marker
    // re-established.
    vec_t pos_zero_vec;
    {
      const auto z = get_pos_zero<T>();
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        pos_zero_vec[j].x = z;
        pos_zero_vec[j].y = z;
      }
    }
#pragma unroll
    for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
      vec_t vec[kWorldSize];
      do {
        bool has_zero = false;
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ld_relaxed_16B(vec[i], data[i], vid);
        }
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
#pragma unroll
          for (uint32_t j = 0; j < kVecSize; ++j) {
            has_zero |= is_pos_zero(vec[i][j].x);
            has_zero |= is_pos_zero(vec[i][j].y);
          }
        }
        if (!has_zero) break;
      } while (true);
      const auto out_vec = reduce(vec);
      st_global_16B(out_vec, out, vid);
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        st_global_16B(pos_zero_vec, data[i], vid);
      }
    }
  }

 public:
  static SGL_DEVICE void forward_1shot(const AllReduceParams& params) {
    // push local data to peer ranks, then reduce locally
    const auto phase = sync_enter_push(params);
    const auto r = params.rank;
    const auto num_vecs = device::div_ceil(params.num_elements, kElemsPerVec);
    const auto stride_bytes = params.push_buffer_stride;
    const auto phase_stride_bytes = phase * stride_bytes * kWorldSize;

    // push to peer
    void* push_buf[kWorldSize];
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      push_buf[i] = params.push_workspaces[i] + r * stride_bytes + phase_stride_bytes;
    }
    push_impl(num_vecs, push_buf, params.input);

    // poll from local
    void* poll_buf[kWorldSize];
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      poll_buf[i] = params.push_workspaces[r] + i * stride_bytes + phase_stride_bytes;
    }
    poll_impl(num_vecs, poll_buf, params.output);

    sync_exit_push(params);
  }
};

template <typename T, uint32_t kWorldSize, PullMode kMode, bool kUsePDL>
struct AllReducePullImpl {
 private:
  using T2 = packed_t<T>;
  static constexpr uint32_t kVecSize = 16 / sizeof(T2);
  static constexpr uint32_t kElemsPerVec = 16 / sizeof(T);
  using vec_t = device::AlignedVector<T2, kVecSize>;
  static_assert(kWorldSize <= kMaxWorldSize);

  template <bool kFence>
  static SGL_DEVICE uint32_t sync_enter_pull(const AllReduceParams& params) {
    uint32_t current_counter_val = 0;
    if (const auto tx = threadIdx.x; tx < kWorldSize) {
      device::PDLWaitPrimary<kUsePDL>();
      const auto bx = blockIdx.x;
      const auto semaphore = &params.pull_semaphores[tx][bx];
      const auto counter = semaphore->counter_ptr();
      const auto current = tx == params.rank ? counter->inc(2 * kWorldSize) : 0;
      current_counter_val = current;
      if constexpr (kFence) {
        semaphore->put_release();
      } else {
        semaphore->put_relaxed();
      }
      if (tx == params.rank) {
        if constexpr (kFence) {
          while (semaphore->get_acquire() - current < kWorldSize)
            ;
        } else {
          while (semaphore->get_relaxed() - current < kWorldSize)
            ;
        }
      }
    }
    __syncthreads();
    return current_counter_val + kWorldSize;
  }

  template <bool kFence>
  static SGL_DEVICE void sync_exit_pull(const AllReduceParams& params, uint32_t current) {
    device::PDLTriggerSecondary<kUsePDL>();
    __syncthreads();
    if (const auto tx = threadIdx.x; tx < kWorldSize) {
      const auto bx = blockIdx.x;
      const auto semaphore = &params.pull_semaphores[tx][bx];
      if constexpr (kFence) {
        semaphore->put_release();
      } else {
        semaphore->put_relaxed();
      }
      if (tx == params.rank) {
        if constexpr (kFence) {
          while (semaphore->get_acquire() - current < kWorldSize)
            ;
        } else {
          while (semaphore->get_relaxed() - current < kWorldSize)
            ;
        }
      }
    }
  }

  template <bool kIs2shot>
  static SGL_DEVICE void reduce_impl(
      uint32_t num_vecs,  //
      [[maybe_unused]] void* (&data)[kWorldSize],
      [[maybe_unused]] void* out,
      [[maybe_unused]] void* mc_addr) {
    const auto num_threads = blockDim.x * gridDim.x;
    const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
      if constexpr (kMode == PullMode::Multicast) {
        vec_t out_vec;
        ld_multimem_16B(out_vec, mc_addr, vid);
        if constexpr (kIs2shot) {
          // inplace write to workspace for 2-shot all reduce
          st_multimem_16B(out_vec, mc_addr, vid);
        } else {
          // write to output for 1-shot all reduce
          out_vec.store(out, vid);
        }
      } else {
        vec_t vec[kWorldSize];
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          vec[i].load(data[i], vid);
        }
        const auto out_vec = reduce(vec);
        if constexpr (kIs2shot) {
          // inplace write to buffer for 2-shot all reduce
#pragma unroll
          for (uint32_t i = 0; i < kWorldSize; ++i) {
            out_vec.store(data[i], vid);
          }
        } else {
          // write to output for 1-shot all reduce
          out_vec.store(out, vid);
        }
      }
    }
  }

 public:
  static SGL_DEVICE void forward_1shot(const AllReduceParams& params) {
    const auto total_num_vecs = device::div_ceil(params.num_elements, kElemsPerVec);
    void* data[kWorldSize];
    if constexpr (kMode == PullMode::Graph) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        data[i] = params.graph_params[i];
      }
    } else if constexpr (kMode == PullMode::Eager) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        data[i] = params.pull_workspaces[i];
      }
    }
    const auto counter = sync_enter_pull<false>(params);
    reduce_impl<false>(total_num_vecs, data, params.output, params.pull_mc_workspace);
    sync_exit_pull<false>(params, counter);
  }

  static SGL_DEVICE void forward_2shot(const AllReduceParams& params) {
    const auto total_num_vecs = device::div_ceil(params.num_elements, kElemsPerVec);
    const auto avg_vecs = total_num_vecs / kWorldSize;
    const auto rem_vecs = total_num_vecs % kWorldSize;
    // usually, hidden size is a multiple of 1024, so 1024 / 8 = 128 is typically 128-bytes aligned
    const auto local_vec_bias = avg_vecs * params.rank + min(params.rank, rem_vecs);
    const auto local_num_vecs = avg_vecs + (params.rank < rem_vecs ? 1 : 0);
    void* data[kWorldSize];
    if constexpr (kMode == PullMode::Graph) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        data[i] = reinterpret_cast<vec_t*>(params.graph_params[i]) + local_vec_bias;
      }
    } else if constexpr (kMode == PullMode::Eager) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        data[i] = reinterpret_cast<vec_t*>(params.pull_workspaces[i]) + local_vec_bias;
      }
    }
    const auto counter = sync_enter_pull<false>(params);
    const auto mc_addr = reinterpret_cast<vec_t*>(params.pull_mc_workspace) + local_vec_bias;
    reduce_impl<true>(local_num_vecs, data, params.output, mc_addr);
    sync_exit_pull<true>(params, counter);
  }
};

template <typename Impl, int kShot>
__global__ __launch_bounds__(1024, 1)  //
    void all_reduce_kernel(const __grid_constant__ AllReduceParams params) {
  static_assert(kShot == 1 || kShot == 2, "invalid shot");
  if constexpr (kShot == 1) {
    return Impl::forward_1shot(params);
  } else {
    return Impl::forward_2shot(params);
  }
}

template <uint32_t N>
__global__ void memcpy_kernel(void* __restrict__ dst, const void* __restrict__ src, uint32_t num_vecs) {
  static_assert(N % 4 == 0, "at least 4-bytes aligned for uint32_t load/store");
  using vec_t = device::AlignedVector<uint32_t, N / 4>;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  if (tid < num_vecs) {
    vec_t vec;
    vec.load(src, tid);
    vec.store(dst, tid);
  }
}

// Pick the smallest block size whose grid still fits in one wave; the kernels
// are grid-stride so any choice is correct, this only tunes occupancy.
[[maybe_unused]]
uint32_t choose_block_size(uint32_t num_threads) {
  static const uint32_t kNumSM = [] {
    int device = 0, sm_count = 0;
    host::RuntimeDeviceCheck(cudaGetDevice(&device));
    host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    return static_cast<uint32_t>(sm_count);
  }();
  for (const uint32_t block_size : {128u, 256u, 512u}) {
    if (host::div_ceil(num_threads, block_size) <= kNumSM) return block_size;
  }
  return 1024u;
}

template <typename T, uint32_t kWorldSize, bool kUsePDL>
struct AllReduceKernel {
 private:
  using Tensor = tvm::ffi::Tensor;
  using TensorView = tvm::ffi::TensorView;
  template <int kShot, PullMode kPullMode>
  static constexpr auto kernel_pull = all_reduce_kernel<AllReducePullImpl<T, kWorldSize, kPullMode, kUsePDL>, kShot>;
  template <int kShot>
  static constexpr auto kernel_push = all_reduce_kernel<AllReducePushImpl<T, kWorldSize, kUsePDL>, kShot>;

 public:
  static Tensor run(CommunicatorRef ref, Tensor in_, std::string algo, std::variant<TensorView, bool> pull_arg) {
    using namespace host;
    const auto& data = *ref.get();
    RuntimeCheck(algo == "1shot_pull" || algo == "2shot_pull" || algo == "1shot_push", "Invalid algo: ", algo);
    RuntimeCheck(data.world_size == kWorldSize, "Mismatch world size");
    RuntimeCheck(in_.IsContiguous(), "Input tensor must be contiguous");
    RuntimeCheck(is_type<T>(in_.dtype()), "Input dtype mismatch");
    RuntimeCheck(in_.device().device_type == kDLCUDA, "Only CUDA device is supported");
    RuntimeCheck(std::bit_cast<intptr_t>(in_.data_ptr()) % 16 == 0, "Input pointer is not properly aligned");
    const auto num_elems_int64 = in_.numel();
    const auto num_elems = static_cast<uint32_t>(num_elems_int64);
    RuntimeCheck(static_cast<int64_t>(num_elems) == num_elems_int64, "Number of items exceeds 4G limit");
    const bool use_graph = std::holds_alternative<TensorView>(pull_arg);
    const auto graph_ptr = use_graph ? std::get<TensorView>(pull_arg).data_ptr() : nullptr;
    const bool inplace = use_graph && algo == "2shot_pull";
    Tensor out = inplace ? in_ : ffi::empty_like(in_);
    AllReduceParams params{
        .input = in_.data_ptr(),
        .output = out.data_ptr(),
        .num_elements = num_elems,
        .rank = data.rank,
        .graph_params = static_cast<void* const*>(graph_ptr),
        .pull_workspaces = {},
        .push_workspaces = {},
        .pull_semaphores = {},
        .push_counter = data.push_counter,
        .pull_mc_workspace = data.pull_mc_workspace,
        .push_buffer_stride = data.push_bytes,
    };
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      params.pull_workspaces[i] = data.pull_workspaces[i];
      params.push_workspaces[i] = data.push_workspaces[i];
      params.pull_semaphores[i] = data.pull_semaphores[i];
    }
    const int64_t nbytes = num_elems_int64 * sizeof(T);
    RuntimeCheck(nbytes % 16 == 0, "Input bytes must be a multiple of 16, got: ", nbytes);
    const uint32_t num_vecs = num_elems / (16 / sizeof(T));
    const auto stream = LaunchKernel::resolve_device(in_.device());

    if (algo == "1shot_push") {
      RuntimeCheck(!use_graph, "Push mode doesn't have graph mode optimization");
      RuntimeCheck(nbytes <= data.push_bytes, "Input size ", nbytes, " exceeds push workspace size ", data.push_bytes);
      // the grid is bound to the counter array and must stay constant
      const uint32_t num_blocks = data.num_push_blocks;
      LaunchKernel(num_blocks, choose_block_size(num_vecs), stream)  //
          .enable_pdl(kUsePDL)(kernel_push<1>, params);
      return out;
    }

    using enum PullMode;
    RuntimeCheck(nbytes <= data.pull_bytes, "Input size ", nbytes, " exceeds pull workspace size ", data.pull_bytes);
    const auto pull_mode = use_graph ? Graph : std::get<bool>(pull_arg) ? Multicast : Eager;
    RuntimeCheck(pull_mode != Multicast || data.pull_mc_workspace != nullptr, "Multicast requires an mc workspace");

    const uint32_t num_blocks = data.num_pull_blocks;
    const auto cuda_memcpy = [&](void* dst, const void* src) {
      if constexpr (SGL_ARCH_HOPPER_OR_GREATER) {  // PDL memcpy is faster
        // based on micro benchmark, only enable when batch size is small + aligned
        constexpr int64_t threshold_MB = SGL_ARCH_BLACKWELL_OR_GREATER ? 1024 : 8;
        if (nbytes % device::kMaxVecBytes == 0 && nbytes <= threshold_MB * 1024 * 1024) {
          const auto copy_kernel = memcpy_kernel<device::kMaxVecBytes>;
          const uint32_t num_copy_vecs = nbytes / device::kMaxVecBytes;
          const uint32_t num_copy_threads = 128u;
          const uint32_t num_copy_blocks = div_ceil(num_copy_vecs, num_copy_threads);
          LaunchKernel(num_copy_blocks, num_copy_threads, stream)
              .enable_pdl(kUsePDL)(copy_kernel, dst, src, num_copy_vecs);
          return;
        }
      }
      // safe fallback to cudaMemcpyAsync for large size or older architecture
      RuntimeDeviceCheck(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    };

    const auto local_workspace = data.pull_workspaces[data.rank];
    if (algo == "1shot_pull") {
      // first copy to workspace
      if (!use_graph) cuda_memcpy(local_workspace, in_.data_ptr());
      const auto kernel = (pull_mode == Graph) ? kernel_pull<1, Graph>
                          : pull_mode == Eager ? kernel_pull<1, Eager>
                                               : kernel_pull<1, Multicast>;
      // then launch kernel to reduce and write to output
      LaunchKernel(num_blocks, choose_block_size(num_vecs), stream)  //
          .enable_pdl(kUsePDL)(kernel, params);
    } else /* 2shot_pull */ {
      const uint32_t avg_vecs = div_ceil(num_vecs, kWorldSize);
      // first copy to workspace
      if (!use_graph) cuda_memcpy(local_workspace, in_.data_ptr());
      // then launch kernel to reduce in workspace
      const auto kernel = (pull_mode == Graph) ? kernel_pull<2, Graph>
                          : pull_mode == Eager ? kernel_pull<2, Eager>
                                               : kernel_pull<2, Multicast>;
      if (pull_mode == Multicast) {
        const auto max_blocks = data.num_multicast_blocks;
        constexpr uint32_t kMulticastNumThreads = 512u;
        // NOTE: too much traffic will degrade performance in multicast
        LaunchKernel(std::min(num_blocks, max_blocks), kMulticastNumThreads, stream)
            .enable_pdl(kUsePDL)(kernel, params);
      } else {
        LaunchKernel(num_blocks, choose_block_size(avg_vecs), stream)  //
            .enable_pdl(kUsePDL)(kernel, params);
      }
      // finally copy from workspace to output
      if (!use_graph) cuda_memcpy(out.data_ptr(), local_workspace);
    }
    return out;
  }
};

template <typename T, uint32_t kWorldSize, bool kUsePDL>
tvm::ffi::Tensor custom_all_reduce(
    CommunicatorRef comm, tvm::ffi::Tensor input, std::string algo, std::variant<tvm::ffi::TensorView, bool> pull_arg) {
  return AllReduceKernel<T, kWorldSize, kUsePDL>::run(comm, input, algo, pull_arg);
}

}  // namespace
