// bf16 all-gather of a [M, N / world] column slice into [M, N] (+ b) (+ c),
// through the CustomAllReduceV2 push workspace (same phase / Lamport protocol
// as all_reduce_1shot_push). Each rank multicast-stores its slice into every
// peer's workspace, then spins its full rows together. Any TP world size.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <algorithm>
#include <cstdint>
#include <optional>

namespace sglang {

namespace muse_all_gather {

using device::distributed::Counter;

constexpr uint32_t kBlock = 128;
constexpr uint32_t kVec = 16 / sizeof(bf16_t);  // 8 bf16 (16B) per thread

template <uint32_t kWorld>
struct Params {
  uint8_t* ws_local;      // LOCAL VA of the push workspace base (poll + reset)
  uint8_t* ws_mc;         // multicast VA of the push workspace base (push)
  Counter* counter;       // per-block phase counters
  uint32_t num_counters;  // PushPlane::num_blocks
  uint32_t half_bytes;    // world_size * slot_bytes
  uint32_t rank;
  const bf16_t* local;  // [M, N / world]
  const bf16_t* b;      // [M, N] or null
  const bf16_t* c;      // [M, N] or null
  bf16_t* out;          // [M, N]
  uint32_t num_rows;
};

// Slot layout inside a phase half: dense [world][M][N / world] bf16.
template <uint32_t N, uint32_t kWorld, bool kHasB, bool kHasC, bool kUsePDL>
__global__ __launch_bounds__(kBlock) void all_gather_kernel(const __grid_constant__ Params<kWorld> params) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, kVec / 2>;
  constexpr uint32_t kNLocal = N / kWorld;
  static_assert(N % kVec == 0 && kNLocal % kVec == 0, "a vector must never cross a rank block");
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
  for (uint32_t local_elem = tid * kVec; local_elem < params.num_rows * kNLocal; local_elem += stride) {
    uint4 raw = *reinterpret_cast<const uint4*>(params.local + local_elem);
    // +0.0 is the "not arrived" marker; -0.0 is numerically identical
    uint32_t* w = reinterpret_cast<uint32_t*>(&raw);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      if ((w[j] & 0xffffu) == 0) w[j] |= 0x8000u;
      if ((w[j] >> 16) == 0) w[j] |= 0x80000000u;
    }
    const uint32_t byte_off = half_off + (params.rank * params.num_rows * kNLocal + local_elem) * sizeof(bf16_t);
    ptx::st_multimem_16B(raw, params.ws_mc + byte_off, 0);
  }
  PDLTriggerSecondary<kUsePDL>();

  // Every producer finishes its local pushes before polling output vectors.
  for (uint32_t elem = tid * kVec; elem < params.num_rows * N; elem += stride) {
    const auto row = elem / N;
    const auto col = elem % N;
    const auto base = reinterpret_cast<bf16_t*>(params.ws_local + half_off);
    const auto src = base + ((col / kNLocal) * params.num_rows + row) * kNLocal + col % kNLocal;
    vec_t b_vec, c_vec;
    if constexpr (kHasB) b_vec.load(params.b + elem);
    if constexpr (kHasC) c_vec.load(params.c + elem);
    uint4 raw;
    do {
      ptx::ld_relaxed_16B(raw, src, 0);
    } while ((raw.x & 0xffffu) == 0 || (raw.x >> 16) == 0 || (raw.y & 0xffffu) == 0 || (raw.y >> 16) == 0 ||
             (raw.z & 0xffffu) == 0 || (raw.z >> 16) == 0 || (raw.w & 0xffffu) == 0 || (raw.w >> 16) == 0);
    const auto& gathered = *reinterpret_cast<const vec_t*>(&raw);
    vec_t out_vec;
#pragma unroll
    for (uint32_t j = 0; j < kVec / 2; ++j) {
      using Trait = DTypeTrait<bf16x2_t>;
      out_vec[j] = gathered[j];
      if constexpr (kHasB) out_vec[j] = Trait::add(out_vec[j], b_vec[j]);
      if constexpr (kHasC) out_vec[j] = Trait::add(out_vec[j], c_vec[j]);
    }
    out_vec.store(params.out + elem);
    AlignedVector<uint32_t, 4> zero;
    zero.fill(0);
    zero.store(src);
  }
  __syncthreads();
  if (tx == 0) params.counter[bx].set(phase ^ 1);
}

}  // namespace muse_all_gather

using host::distributed::CommunicatorRef;

template <uint32_t N, uint32_t kWorld, bool kUsePDL>
struct MuseSparkAllGather {
  using TensorView = tvm::ffi::TensorView;
  static constexpr uint32_t kNLocal = N / kWorld;

  template <bool kHasB, bool kHasC>
  static constexpr auto kernel = muse_all_gather::all_gather_kernel<N, kWorld, kHasB, kHasC, kUsePDL>;

  static void
  run(CommunicatorRef ref, TensorView local, std::optional<TensorView> b, std::optional<TensorView> c, TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kNLocal}).with_dtype<bf16_t>().with_device(device).verify(local);
    if (b.has_value()) TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(b.value());
    if (c.has_value()) TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(c.value());
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1);
    CHECK_HOST(push.world_size == kWorld) << "the kernel is compiled for TP" << kWorld;
    CHECK_HOST(int64_t(num_tokens) * N * 2 <= push.slot_bytes)
        << "gathered rows exceed the push slot size " << push.slot_bytes;
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";

    muse_all_gather::Params<kWorld> params{
        .ws_local = push.workspaces[push.rank],
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
        .local = static_cast<const bf16_t*>(local.data_ptr()),
        .b = b.has_value() ? static_cast<const bf16_t*>(b.value().data_ptr()) : nullptr,
        .c = c.has_value() ? static_cast<const bf16_t*>(c.value().data_ptr()) : nullptr,
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .num_rows = num_tokens,
    };
    const auto num_vecs = num_tokens * N / muse_all_gather::kVec;
    CHECK_HOST(push.num_blocks >= 2) << "gather needs a work block and a counter-update block";
    const auto num_work_blocks = std::min(host::div_ceil(num_vecs, muse_all_gather::kBlock), push.num_blocks - 1);
    const auto fn =
        b.has_value() ? (c.has_value() ? kernel<1, 1> : kernel<1, 0>) : (c.has_value() ? kernel<0, 1> : kernel<0, 0>);
    host::LaunchKernel(num_work_blocks + 1, muse_all_gather::kBlock, device.unwrap()).enable_pdl(kUsePDL)(fn, params);
  }
};

}  // namespace sglang
