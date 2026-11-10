// fp32 all-gather of a [M, N / world] column block into [M, N] for the K3
// TP-sharded MoE front, on all_reduce_1shot_push's push-workspace phase / Lamport
// protocol.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace k3_front_ag {

using device::distributed::Counter;
using Lamport = device::distributed::LamportTrait<float, 4>;

constexpr uint32_t kVec = 16 / sizeof(float);  // 4 fp32 (16B) per thread

struct Params {
  uint8_t* ws_local;      // LOCAL VA of the push workspace base (poll + reset)
  uint8_t* ws_mc;         // multicast VA of the push workspace base (push)
  Counter* counter;       // per-block phase counters
  uint32_t num_counters;  // PushPlane::num_blocks
  uint32_t half_bytes;    // world_size * slot_bytes
  uint32_t rank;
  const float* local;     // [M, N / world], rows local_stride floats apart
  uint32_t local_stride;  // a multiple of kVec
  float* out;             // [M, N]
  uint32_t num_rows;
};

// Slot layout inside a phase half: dense [world][M][N / world] fp32.
template <uint32_t kNLocal, uint32_t kWorld, uint32_t kBlock, bool kUsePDL>
__global__ __launch_bounds__(kBlock) void front_all_gather_kernel(const __grid_constant__ Params params) {
  using namespace device;
  constexpr uint32_t N = kNLocal * kWorld;
  static_assert(kNLocal % kVec == 0, "a vector must never cross a rank block");
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const uint32_t tid = bx * kBlock + tx;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = params.counter[bx].get() & 1;
  const uint32_t half_off = phase * params.half_bytes;

  // the last block flips its own counter and every one past the grid
  if (const auto num_blocks = gridDim.x; bx == num_blocks - 1) {
    [[unlikely]];
    PDLTriggerSecondary<kUsePDL>();
    __syncthreads();
    for (uint32_t i = num_blocks - 1 + tx; i < params.num_counters; i += kBlock) {
      params.counter[i].set(phase ^ 1);
    }
    return void();
  }

  // Bound the grid to the existing AR counters; larger batches use a stride.
  const uint32_t stride = (gridDim.x - 1) * kBlock * kVec;
  const uint32_t local_numel = params.num_rows * kNLocal;
  for (uint32_t local_elem = tid * kVec; local_elem < local_numel; local_elem += stride) {
    const auto local_row = local_elem / kNLocal;
    const auto local_col = local_elem % kNLocal;
    uint4 raw = *reinterpret_cast<const uint4*>(params.local + local_row * params.local_stride + local_col);
    // +0.0 is the "not arrived" marker; -0.0 is numerically identical
    Lamport::clear_pos_zero(&raw);
    const uint32_t byte_off = half_off + (params.rank * local_numel + local_elem) * sizeof(float);
    ptx::st_multimem_16B(raw, params.ws_mc + byte_off, 0);
  }
  PDLTriggerSecondary<kUsePDL>();

  // Every producer finishes its local pushes before polling output vectors.
  const auto base = reinterpret_cast<float*>(params.ws_local + half_off);
  for (uint32_t elem = tid * kVec; elem < params.num_rows * N; elem += stride) {
    const auto row = elem / N;
    const auto col = elem % N;
    const auto src = base + ((col / kNLocal) * params.num_rows + row) * kNLocal + col % kNLocal;
    uint4 raw;
    do {
      ptx::ld_relaxed_16B(raw, src, 0);
    } while (Lamport::has_pos_zero(&raw));
    *reinterpret_cast<uint4*>(params.out + elem) = raw;
    *reinterpret_cast<uint4*>(src) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  if (tx == 0) params.counter[bx].set(phase ^ 1);
}

}  // namespace k3_front_ag

using host::distributed::CommunicatorRef;

// kBlock: threads per block, picked host side (moe_front_ag.py).
template <uint32_t kNLocal, uint32_t kWorld, uint32_t kBlock, bool kUsePDL>
struct K3FrontAllGather {
  using TensorView = tvm::ffi::TensorView;
  static constexpr uint32_t N = kNLocal * kWorld;
  static constexpr auto kernel = k3_front_ag::front_all_gather_kernel<kNLocal, kWorld, kBlock, kUsePDL>;

  static void run(CommunicatorRef ref, TensorView local, TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto S = SymbolicSize{"local_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({M, kNLocal})
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(local);
    TensorMatcher({M, N}).with_dtype<float>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1);
    CHECK_HOST(push.world_size == kWorld) << "the kernel is compiled for TP" << kWorld;
    CHECK_HOST(int64_t(num_tokens) * kNLocal * 4 <= push.slot_bytes)
        << "this rank's block exceeds the push slot size " << push.slot_bytes;
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";
    CHECK_HOST(push.num_blocks >= 2) << "gather needs a work block and a counter-update block";

    const auto params = k3_front_ag::Params{
        .ws_local = push.workspaces[push.rank],
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
        .local = static_cast<const float*>(local.data_ptr()),
        .local_stride = static_cast<uint32_t>(num_tokens > 1 ? S.unwrap() : kNLocal),
        .out = static_cast<float*>(out.data_ptr()),
        .num_rows = num_tokens,
    };
    const auto num_vecs = num_tokens * N / k3_front_ag::kVec;
    const auto num_work_blocks = std::min(host::div_ceil(num_vecs, kBlock), push.num_blocks - 1);
    host::LaunchKernel(num_work_blocks + 1, kBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
