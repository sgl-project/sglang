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
// logic: all pointers arrive via the communication planes (owned by Python)
// and the per-call params. The push and pull families take disjoint params,
// so neither carries the other's pointer table into its grid constants.
//
// The pull family reduces over the pull plane's workspaces, which exist
// because this kernel's callers hand it plain tensors: the host stages the
// input in and copies the result back out. Callers that already allocate
// from symmetric memory (the K3 fused paths) reduce in place instead and
// borrow the plane only to barrier on.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/extra/stl.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>

namespace sglang {

using device::distributed::Counter;
using device::distributed::PullWorkSpace, device::distributed::PushWorkSpace;
using host::distributed::CommunicatorRef;

inline constexpr uint32_t kMaxWorldSize = device::distributed::kMaxWorldSize;

enum class PullMode {
  Graph,
  Eager,
  Multicast,  // also eager
};

template <uint32_t kWorldSize>
struct AllReducePushParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;
  uint32_t rank;
  Counter* counter;  // rank-local
  PushWorkSpace<kWorldSize> ws;
};

template <uint32_t kWorldSize>
struct AllReducePullParams {
  void* __restrict__ output;
  uint32_t num_vecs;
  uint32_t rank;
  void* const* __restrict__ graph_params;
  PullWorkSpace<kWorldSize> ws;
};

/// `vec_offset` is a *vector index*, and it is folded into the base pointers
/// here rather than added to `vid` at every access. Both halves of that matter:
/// a typed 32-bit bias becomes one widening multiply-add off a constant-bank
/// base, which ptxas keeps on the uniform datapath, so the whole peer table
/// lives in uniform registers. Biasing `vid` instead makes all `kWorldSize`
/// addresses thread-varying (~2 ordinary registers each), and a 64-bit byte
/// bias does the same because the uniform datapath has no 64-bit add. Either
/// way 2shot pays ~12 registers per thread and ~10% throughput.
template <typename V, uint32_t kWorldSize, bool kUseGraph>
struct LoadStoreImpl {
 public:
  static constexpr uint32_t size() {
    return kWorldSize;
  }
  SGL_DEVICE LoadStoreImpl(const AllReducePullParams<kWorldSize>& params, uint32_t vec_offset = 0) {
    if constexpr (kUseGraph) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        m_data[i] = reinterpret_cast<V*>(params.graph_params[i]) + vec_offset;
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        m_data[i] = reinterpret_cast<V*>(params.ws.workspaces[i]) + vec_offset;
      }
    }
  }

  SGL_DEVICE void load_reduce(V& vec, uint32_t vid) const {
    V vecs[kWorldSize];
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      vecs[i].load(m_data[i], vid);
    }
    vec = device::reduce_vec(vecs);
  }

  SGL_DEVICE void store_multi(const V& val, uint32_t vid) const {
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      val.store(m_data[i], vid);
    }
  }

 private:
  V* m_data[kWorldSize];
};

template <typename V, uint32_t kWorldSize, bool kUseGraph>
struct MultiCastImpl {
 public:
  static_assert(kUseGraph == false);
  static constexpr uint32_t size() {
    return kWorldSize;
  }

  SGL_DEVICE MultiCastImpl(const AllReducePullParams<kWorldSize>& params, uint32_t vec_offset = 0)
      : m_multicast_ptr(reinterpret_cast<V*>(params.ws.mc_workspace) + vec_offset) {}

  SGL_DEVICE void load_reduce(V& vec, uint32_t vid) const {
    device::ptx::ld_multimem_16B(vec, m_multicast_ptr, vid);
  }

  SGL_DEVICE void store_multi(const V& val, uint32_t vid) const {
    return device::ptx::st_multimem_16B(val, m_multicast_ptr, vid);
  }

 private:
  V* m_multicast_ptr;
};

#define ALL_REDUCE_KERNEL __global__ __launch_bounds__(1024, 1)

template <typename Impl, typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void all_reduce_1shot_push_kernel(const __grid_constant__ AllReducePushParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto stride_bytes = params.ws.slot_bytes;
  PDLWaitPrimary<kUsePDL>();
  const auto phase = params.counter[blockIdx.x].get() % 2;
  const auto phase_stride_bytes = phase * stride_bytes * kWorldSize;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  // push to peer
  void* push_ptr[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_ptr[i] = params.ws.workspaces[i] + r * stride_bytes + phase_stride_bytes;
  }

  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    vec.load(params.input, vid);
    Lamport::clear_pos_zero(vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(vec, push_ptr[i], vid);
    }
  }

  // poll from local
  void* poll_ptr[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptr[i] = params.ws.workspaces[r] + i * stride_bytes + phase_stride_bytes;
  }
  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B(vec[i], poll_ptr[i], vid);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= Lamport::has_pos_zero(vec[i].data());
      }
      if (!has_zero) break;
    } while (true);
    const auto out_vec = reduce_vec(vec);
    ptx::st_global_16B(out_vec, params.output, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B(pos_zero_vec, poll_ptr[i], vid);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (threadIdx.x == 0) params.counter[blockIdx.x].set(phase ^ 1);
}

template <typename Impl, typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void all_reduce_1shot_pull_kernel(const __grid_constant__ AllReducePullParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto num_vecs = params.num_vecs;
  const auto impl = Impl{params};

  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::Barrier<kWorldSize>(params.ws.semaphores.data(), params.rank, /*num_arrives=*/2);
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    impl.load_reduce(vec, vid);
    vec.store(params.output, vid);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_relaxed(/*n=*/1);
}

template <typename Impl, typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void all_reduce_2shot_pull_kernel(const __grid_constant__ AllReducePullParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto num_total_vecs = params.num_vecs;
  const auto avg_vecs = num_total_vecs / kWorldSize;
  const auto rem_vecs = num_total_vecs % kWorldSize;
  const auto num_vecs = avg_vecs + (params.rank < rem_vecs ? 1 : 0);
  const auto vec_offset = params.rank * avg_vecs + min(params.rank, rem_vecs);
  const auto impl = Impl{params, vec_offset};
  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::Barrier<kWorldSize>(params.ws.semaphores.data(), params.rank, /*num_arrives=*/2);
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    impl.load_reduce(vec, vid);
    impl.store_multi(vec, vid);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);
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

inline auto choose_block_size(uint32_t num_threads) -> uint32_t {
  static const uint32_t kNumSM = [] {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    return host::runtime::get_sm_count(device);
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
  using PushParams = AllReducePushParams<kWorldSize>;
  using PullParams = AllReducePullParams<kWorldSize>;
  using vec_t = device::AlignedVector<packed_t<T>, 16 / (sizeof(T) * 2)>;

 public:
  static Tensor run(CommunicatorRef comm_ref, Tensor in_, std::string algo, std::variant<TensorView, bool> pull_arg) {
    using namespace host;
    const auto& comm = *comm_ref.get();
    RuntimeCheck(algo == "1shot_pull" || algo == "2shot_pull" || algo == "1shot_push", "Invalid algo: ", algo);
    RuntimeCheck(comm.get_world_size() == kWorldSize, "Mismatch world size");
    RuntimeCheck(in_.IsContiguous(), "Input tensor must be contiguous");
    RuntimeCheck(is_type<T>(in_.dtype()), "Input dtype mismatch");
    RuntimeCheck(in_.device().device_type == kDLCUDA, "Only CUDA device is supported");
    RuntimeCheck(std::bit_cast<intptr_t>(in_.data_ptr()) % 16 == 0, "Input pointer is not properly aligned");
    const auto num_elems_int64 = in_.numel();
    const auto num_elems = static_cast<uint32_t>(num_elems_int64);
    RuntimeCheck(static_cast<int64_t>(num_elems) == num_elems_int64, "Number of items exceeds 4G limit");
    const int64_t nbytes = num_elems_int64 * sizeof(T);
    RuntimeCheck(nbytes % 16 == 0, "Input bytes must be a multiple of 16, got: ", nbytes);
    const uint32_t num_vecs = num_elems / (16 / sizeof(T));
    const bool use_graph = std::holds_alternative<TensorView>(pull_arg);
    const auto stream = LaunchKernel::resolve_device(in_.device());

    if (algo == "1shot_push") {
      const auto& push = comm.get_push_obj();
      RuntimeCheck(!use_graph, "Push mode doesn't have graph mode optimization");
      Tensor out = ffi::empty_like(in_);
      const PushParams params{
          .input = in_.data_ptr(),
          .output = out.data_ptr(),
          .num_vecs = num_vecs,
          .rank = push.rank,
          .counter = push.counter,
          .ws = push.get_workspace<kWorldSize>(nbytes),
      };
      using Impl = LoadStoreImpl<vec_t, kWorldSize, /*kUseGraph=*/false>;
      const auto kernel = all_reduce_1shot_push_kernel<Impl, T, kWorldSize, kUsePDL>;
      // the grid is bound to the counter array and must stay constant
      LaunchKernel(push.num_blocks, choose_block_size(num_vecs), stream)  //
          .enable_pdl(kUsePDL)(kernel, params);
      return out;
    }

    using enum PullMode;
    const auto& pull = comm.get_pull_obj();
    const auto pull_mode = use_graph ? Graph : std::get<bool>(pull_arg) ? Multicast : Eager;
    // Graph mode reduces over the caller's own registered buffers and only
    // barriers on this plane; the eager modes stage the input through it.
    const auto ws = pull.get_workspace<kWorldSize>(pull_mode == Graph ? 0 : nbytes);
    RuntimeCheck(pull_mode != Multicast || ws.mc_workspace != nullptr, "Multicast requires an mc workspace");

    const bool inplace = use_graph && algo == "2shot_pull";
    Tensor out = inplace ? in_ : ffi::empty_like(in_);
    const PullParams params{
        .output = out.data_ptr(),
        .num_vecs = num_vecs,
        .rank = pull.rank,
        .graph_params = use_graph ? static_cast<void* const*>(std::get<TensorView>(pull_arg).data_ptr()) : nullptr,
        .ws = ws,
    };

    const uint32_t num_blocks = comm.get_pull_blocks();
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
    const auto local_workspace = ws.workspaces[pull.rank];

    using LS = LoadStoreImpl<vec_t, kWorldSize, /*kUseGraph=*/false>;
    using LS_GRAPH = LoadStoreImpl<vec_t, kWorldSize, /*kUseGraph=*/true>;
    using MC = MultiCastImpl<vec_t, kWorldSize, /*kUseGraph=*/false>;
    if (algo == "1shot_pull") {
      // first copy to the workspace
      if (!use_graph) cuda_memcpy(local_workspace, in_.data_ptr());
      const auto kernel = (pull_mode == Graph) ? all_reduce_1shot_pull_kernel<LS_GRAPH, T, kWorldSize, kUsePDL>
                          : pull_mode == Eager ? all_reduce_1shot_pull_kernel<LS, T, kWorldSize, kUsePDL>
                                               : all_reduce_1shot_pull_kernel<MC, T, kWorldSize, kUsePDL>;
      // then launch kernel to reduce and write to output
      LaunchKernel(num_blocks, choose_block_size(num_vecs), stream)  //
          .enable_pdl(kUsePDL)(kernel, params);
    } else /* 2shot_pull */ {
      const uint32_t avg_vecs = div_ceil(num_vecs, kWorldSize);
      // first copy to the workspace
      if (!use_graph) cuda_memcpy(local_workspace, in_.data_ptr());
      // then launch kernel to reduce in the workspace
      const auto kernel = (pull_mode == Graph) ? all_reduce_2shot_pull_kernel<LS_GRAPH, T, kWorldSize, kUsePDL>
                          : pull_mode == Eager ? all_reduce_2shot_pull_kernel<LS, T, kWorldSize, kUsePDL>
                                               : all_reduce_2shot_pull_kernel<MC, T, kWorldSize, kUsePDL>;
      if (pull_mode == Multicast) {
        constexpr uint32_t kMulticastNumThreads = 512u;
        // NOTE: too much traffic will degrade performance in multicast
        LaunchKernel(comm.get_pull_multicast_blocks(), kMulticastNumThreads, stream)
            .enable_pdl(kUsePDL)(kernel, params);
      } else {
        LaunchKernel(num_blocks, choose_block_size(avg_vecs), stream)  //
            .enable_pdl(kUsePDL)(kernel, params);
      }
      // finally copy from the workspace to output
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

}  // namespace sglang
