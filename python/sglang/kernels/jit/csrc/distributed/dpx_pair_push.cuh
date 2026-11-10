// The pair stages of CustomAllReduceV2's hier_2shot_push, for ROCm gfx1250 (MI450)
// DPX: each SoC's two partitions reduce-scatter between themselves, every
// partition all-reduces its half with its peers on the other SoCs (a regular
// 2shot_push), and the pair all-gathers the result. This file holds the pair
// reduce-scatter / all-gather; it runs over a 2-rank push plane of its own.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/object.h>

#include <cstdint>

#ifdef USE_ROCM
#define cudaGetDevice hipGetDevice
#endif

namespace sglang {

namespace dpx_pair {

using device::distributed::PushWorkSpace;

struct Params {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;  // 16-byte vectors of the full tensor
  uint32_t rank;
  PushWorkSpace<2> ws;
};

/// One half of a 2-rank 2shot_push. Chunk c of the full tensor belongs to rank c.
///   reduce-scatter: `input` is the full tensor, `output` this rank's reduced chunk.
///   all-gather:     `input` is this rank's chunk, `output` the full tensor.
/// Same Lamport markers and epochs as 2shot_push, with a one-chunk slot: each
/// launch both pushes to and polls (and clears) the peer, so the two stages
/// alternate epochs in step on both ranks and may share the slot.
template <typename T, bool kAllGather, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void pair_push_kernel(const __grid_constant__ Params params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto peer = r ^ 1;
  const auto chunk_vecs = params.num_vecs / 2;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<2>{params.ws};
  const auto push_ptr = epoch.slot_ptr(/*dst=*/peer, /*src=*/r);
  const auto poll_ptr = epoch.slot_ptr(/*dst=*/r, /*src=*/peer);

  for (auto j = global_tid; j < chunk_vecs; j += num_threads) {
    vec_t vec;
    vec.load(params.input, kAllGather ? j : peer * chunk_vecs + j);
    Lamport::clear_pos_zero(vec.data());
    ptx::st_relaxed_16B(vec, push_ptr, j);
    // remap before the local store too, so both ranks hold identical bits
    if constexpr (kAllGather) vec.store(params.output, r * chunk_vecs + j);
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  for (auto j = global_tid; j < chunk_vecs; j += num_threads) {
    // vec[0] arrives from the peer, vec[1] is local: a two-term sum is the same
    // in either order
    vec_t vec[2];
    do {
      ptx::ld_relaxed_16B(vec[0], poll_ptr, j);
    } while (Lamport::has_pos_zero(vec[0].data()));
    if constexpr (kAllGather) {
      vec[0].store(params.output, peer * chunk_vecs + j);
    } else {
      vec[1].load(params.input, r * chunk_vecs + j);
      reduce_vec(vec).store(params.output, j);
    }
    ptx::st_global_16B(pos_zero_vec, poll_ptr, j);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();  // every thread has read the phase before this block commits
  epoch.commit();
}

// The smallest block that keeps the grid within one block per SM.
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

}  // namespace dpx_pair

template <typename T, bool kUsePDL>
struct DpxPairPushKernel {
  using Tensor = tvm::ffi::Tensor;

  /// Reduce-scatter (`all_gather` false: the full tensor in, this rank's reduced
  /// half out) or all-gather (this rank's half in, the full tensor out) over a
  /// 2-rank communicator's push plane.
  static Tensor run(const host::distributed::CommunicatorRef comm_ref, const Tensor in, const bool all_gather) {
    using namespace host;
    const auto& comm = *comm_ref.get();
    CHECK_HOST(comm.get_world_size() == 2) << "a pair stage needs a 2-rank communicator: " << comm.get_world_size();
    CHECK_HOST(in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_elems = in.numel();
    const auto nbytes = static_cast<int64_t>(num_elems * sizeof(T));
    CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0 && nbytes % 16 == 0);
    const auto in_vecs = static_cast<uint32_t>(nbytes / 16);
    CHECK_HOST(int64_t{in_vecs} * 16 == nbytes) << nbytes;
    // both stages see the full tensor's vector count
    const auto full_vecs = all_gather ? 2 * in_vecs : in_vecs;
    CHECK_HOST(full_vecs % 2 == 0) << "a pair reduce-scatter needs an even vector count: " << full_vecs;

    const auto& push = comm.get_push_obj();
    const auto out_elems = all_gather ? 2 * num_elems : num_elems / 2;
    Tensor out = ffi::empty({out_elems}, in.dtype(), in.device());
    const auto params = dpx_pair::Params{
        .input = in.data_ptr(),
        .output = out.data_ptr(),
        .num_vecs = full_vecs,
        .rank = push.rank,
        .ws = push.get_workspace<2>(int64_t{full_vecs / 2} * 16),
    };
    const auto kernel =
        all_gather ? dpx_pair::pair_push_kernel<T, true, kUsePDL> : dpx_pair::pair_push_kernel<T, false, kUsePDL>;
    const auto stream = LaunchKernel::resolve_device(in.device());
    LaunchKernel(push.default_blocks, dpx_pair::choose_block_size(full_vecs / 2), stream)  //
        .enable_pdl(kUsePDL)(kernel, params);
    return out;
  }
};

}  // namespace sglang
