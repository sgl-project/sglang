#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace sglang {

namespace gemm_ag {

using device::distributed::Counter;
// 16B vector = 4 packed bf16 pairs; +0.0 in a pair word is the "not arrived" marker
using Lamport = device::distributed::LamportTrait<bf16_t, 16 / sizeof(bf16_t), /*kAtom=*/4>;

constexpr uint32_t kVecSize = 32 / sizeof(bf16_t);  // 16 bf16 per 32B vector
constexpr uint32_t kSpinBlock = 128;                // min consumer threads per block
constexpr uint32_t kMaxSpinBlock = 1024;            // max consumer threads per block
constexpr uint32_t kSpinVec = 16 / sizeof(bf16_t);  // 8 bf16 (16B) per consumer thread

// Producer: per-rank column-slice GEMV, multicast store with Lamport markers.

struct ProducerParams {
  uint8_t* ws_mc;       // multicast VA of the push workspace base
  Counter* counter;     // per-block phase counters (READ only here)
  uint32_t half_bytes;  // bytes per phase half (world_size * slot_bytes)
  uint32_t rank;
};

template <uint32_t K, uint32_t N, uint32_t kWorld, uint32_t M, uint32_t N_SPLIT, bool kUsePDL>
__global__ __launch_bounds__(K / kVecSize) void gemm_ag_gemv_kernel(
    const __grid_constant__ ProducerParams params,
    const bf16_t* __restrict__ x,         // [M, K]
    const bf16_t* __restrict__ weight) {  // [N, K] FULL replicated weight
  using namespace device;
  using vec_t = AlignedVector<bf16_t, kVecSize>;
  constexpr uint32_t kNLocal = N / kWorld;  // columns computed by this rank
  constexpr uint32_t kGemvBlock = K / kVecSize;
  constexpr uint32_t kNumWarps = kGemvBlock / kWarpThreads;
  static_assert(K % kVecSize == 0, "K must be a multiple of the 32B vector width");
  static_assert(kGemvBlock % kWarpThreads == 0, "K / vec_size must fill whole warps");
  static_assert(kGemvBlock <= 1024, "K / vec_size exceeds the maximum block size");
  static_assert(N % kWorld == 0, "N must split evenly over the TP world");
  static_assert(kNLocal % N_SPLIT == 0, "the local column slice must split into whole tiles");
  static_assert(M * N_SPLIT <= kGemvBlock, "output tile must fit one thread each for the final reduce");
  static_assert(N_SPLIT % 2 == 0, "epilogue stores adjacent column pairs");

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  // this rank's rows of the replicated [N, K] weight, sliced HERE (the
  // Python side always hands the full weight)
  const bf16_t* weight_tile = weight + (params.rank * kNLocal + bx * N_SPLIT) * K;

  // weight prefetch before the PDL wait (input-independent addresses)
  vec_t weight_vec[N_SPLIT];
#pragma unroll
  for (uint32_t n = 0; n < N_SPLIT; ++n) {
    weight_vec[n].load(weight_tile + n * K, tx);
  }

  PDLWaitPrimary<kUsePDL>();
  // Every push-workspace consumer flips the WHOLE counter array each round
  // (each has a tail loop up to num_counters), so all counters hold the same
  // phase at this point and counter[0] is equivalent to counter[bx]. Reading a
  // single counter is what frees the producer grid from the counter array size.
  const uint32_t phase = params.counter[0].get() & 1;

  vec_t input_vec[M];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    input_vec[m].load(x + m * K, tx);
  }

  __shared__ alignas(16) float s_acc[kNumWarps][M * N_SPLIT];
  const uint32_t warp_id = tx / kWarpThreads;

#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t n = 0; n < N_SPLIT; ++n) {
      float acc = 0.0f;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        acc = device::math::fma_f32_bf16(input_vec[m][i], weight_vec[n][i], acc);
      }
      s_acc[warp_id][m * N_SPLIT + n] = warp::reduce_sum(acc);
    }
  }
  __syncthreads();
  constexpr uint32_t kNumPairs = M * N_SPLIT / 2;
  if (tx < kNumPairs) {
    auto packed = load_as<float2>(s_acc[0], tx);
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i) {
      const auto [lo, hi] = load_as<float2>(s_acc[i], tx);
      packed.x += lo;
      packed.y += hi;
    }
    const auto pair = cast<bf16x2_t>(packed);
    auto bits = *reinterpret_cast<const uint32_t*>(&pair);
    if (bits == 0) bits = 0x8000u;  // -0.0 in the first element: never all-zero
    const uint32_t m = (2 * tx) / N_SPLIT;
    const uint32_t n = (2 * tx) % N_SPLIT;  // even column within the tile
    // bf16 index in the phase half's dense [world][M][N / world] prefix;
    // one multicast store lands this rank's pair on EVERY peer
    const uint32_t elem = (params.rank * M + m) * kNLocal + bx * N_SPLIT + n;
    const auto base = reinterpret_cast<bf16_t*>(params.ws_mc + phase * params.half_bytes);
    const auto dst = reinterpret_cast<uint32_t*>(base + elem);
    ptx::multimem_store_relaxed(dst, bits);
  }
  PDLTriggerSecondary<kUsePDL>();
}

// Consumer: Lamport spin + add3, one 16B vector (8 bf16) per thread per step.

struct ConsumerParams {
  uint8_t* ws_local;      // LOCAL VA of the push workspace base (poll + reset)
  Counter* counter;       // per-block phase counters (read + flip)
  uint32_t num_counters;  // full counter array size (PushPlane::num_blocks)
  uint32_t half_bytes;    // bytes per phase half (world_size * slot_bytes)
  const bf16_t* b;        // [M, N]
  const bf16_t* c;        // may be null
  bf16_t* out;            // [M, N]
  uint32_t num_rows;      // M
};

// out[elem : elem + 8] = gathered + b (+ c); waits for the vector's 4 packed
// pairs to land, then restores the +0.0 marker for the next round.
template <uint32_t N, uint32_t kWorld, bool kHasC>
SGL_DEVICE void spin_add3_vec(const ConsumerParams& params, uint8_t* half, uint32_t elem) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, kSpinVec / 2>;  // 8 bf16 as 4 pairs
  constexpr uint32_t kNLocal = N / kWorld;
  const auto row = elem / N;
  const auto col = elem % N;
  // out[row, col] lives at half[col / kNLocal][row][col % kNLocal] of the
  // dense [world][M][N / world] prefix of the current phase half
  const auto base = reinterpret_cast<bf16_t*>(half);
  const auto src = base + ((col / kNLocal) * params.num_rows + row) * kNLocal + col % kNLocal;
  vec_t b_vec, c_vec;
  b_vec.load(params.b + elem);
  if constexpr (kHasC) c_vec.load(params.c + elem);
  // spin until all 4 packed pairs of the vector have landed
  uint4 raw;
  do {
    ptx::ld_relaxed_16B(raw, src, 0);
  } while (Lamport::has_pos_zero(&raw));
  const auto& gathered = *reinterpret_cast<const vec_t*>(&raw);
  vec_t out_vec;
#pragma unroll
  for (uint32_t j = 0; j < kSpinVec / 2; ++j) {
    using Trait = DTypeTrait<bf16x2_t>;
    out_vec[j] = Trait::add(gathered[j], b_vec[j]);
    if constexpr (kHasC) out_vec[j] = Trait::add(out_vec[j], c_vec[j]);
  }
  out_vec.store(params.out + elem);
  AlignedVector<uint32_t, 4> zero;
  zero.fill(0);
  zero.store(src);
}

// The grid is bound to the counter array (one work block per counter slot plus
// the cleanup block), so larger batches widen the block (host side) and then
// stride over the output.
template <uint32_t N, uint32_t kWorld, bool kHasC, bool kUsePDL>
__global__ __launch_bounds__(kMaxSpinBlock) void spin_add3_kernel(const __grid_constant__ ConsumerParams params) {
  using namespace device;
  static_assert(N % kSpinVec == 0, "rows must stay 16B aligned");
  static_assert((N / kWorld) % kSpinVec == 0, "a vector must never cross a rank block");
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const uint32_t phase = params.counter[bx].get() & 1;

  PDLTriggerSecondary<kUsePDL>();

  // use the last block to clean up: it flips ITS OWN counter and every one
  // past the grid (work blocks flip [0, num_blocks - 1) themselves)
  if (const auto num_blocks = gridDim.x; bx == num_blocks - 1) {
    [[unlikely]];
    __syncthreads();  // ensure phase is ready for all threads
    for (uint32_t i = num_blocks - 1 + tx; i < params.num_counters; i += blockDim.x) {
      params.counter[i].set(phase ^ 1);
    }
    return void();  // this block is done, no output to write
  }

  // Deliberately NO PDLWaitPrimary: the dependency is carried through data
  const auto half = params.ws_local + phase * params.half_bytes;
  const uint32_t stride = (gridDim.x - 1) * blockDim.x * kSpinVec;
  for (uint32_t elem = (bx * blockDim.x + tx) * kSpinVec; elem < params.num_rows * N; elem += stride) {
    spin_add3_vec<N, kWorld, kHasC>(params, half, elem);
  }
  __syncthreads();
  if (tx == 0) params.counter[bx].set(phase ^ 1);
}

// Gather of a precomputed column slice (the GEMM path for larger M): every
// block multicast-stores its share of this rank's [M, N / world] slice into
// the same dense layout the GEMV producer writes, then runs the consumer above
// over its share of the output. One kernel; the slice GEMM is its PDL primary.

struct GatherParams {
  ConsumerParams consumer;
  uint8_t* ws_mc;  // multicast VA of the push workspace base
  uint32_t rank;
  const bf16_t* local;  // [M, N / world], this rank's slice of x @ weight.T
};

template <uint32_t N, uint32_t kWorld, bool kHasC, bool kUsePDL>
__global__ __launch_bounds__(kMaxSpinBlock) void gather_add3_kernel(const __grid_constant__ GatherParams params) {
  using namespace device;
  constexpr uint32_t kNLocal = N / kWorld;
  static_assert(N % kSpinVec == 0, "rows must stay 16B aligned");
  static_assert(kNLocal % kSpinVec == 0, "a vector must never cross a rank block");
  const auto& consumer = params.consumer;
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();  // the slice GEMM (and the previous push-plane round)
  const uint32_t phase = consumer.counter[bx].get() & 1;

  if (const auto num_blocks = gridDim.x; bx == num_blocks - 1) {
    [[unlikely]];
    PDLTriggerSecondary<kUsePDL>();
    __syncthreads();
    for (uint32_t i = num_blocks - 1 + tx; i < consumer.num_counters; i += blockDim.x) {
      consumer.counter[i].set(phase ^ 1);
    }
    return void();
  }

  const uint32_t stride = (gridDim.x - 1) * blockDim.x * kSpinVec;
  const uint32_t first = (bx * blockDim.x + tx) * kSpinVec;
  // this rank's block of the phase half's dense [world][M][N / world] prefix
  const auto dst = params.ws_mc + phase * consumer.half_bytes +
                   static_cast<size_t>(params.rank) * consumer.num_rows * kNLocal * sizeof(bf16_t);
  for (uint32_t local_elem = first; local_elem < consumer.num_rows * kNLocal; local_elem += stride) {
    uint4 raw = *reinterpret_cast<const uint4*>(params.local + local_elem);
    Lamport::clear_pos_zero(&raw);  // -0.0 in the first element: never all-zero
    ptx::st_multimem_16B(raw, dst + local_elem * sizeof(bf16_t), 0);
  }
  PDLTriggerSecondary<kUsePDL>();

  // every block finishes its own pushes before it polls, so no block waits on
  // a push that cannot be issued
  const auto half = consumer.ws_local + phase * consumer.half_bytes;
  for (uint32_t elem = first; elem < consumer.num_rows * N; elem += stride) {
    spin_add3_vec<N, kWorld, kHasC>(consumer, half, elem);
  }
  __syncthreads();
  if (tx == 0) consumer.counter[bx].set(phase ^ 1);
}

}  // namespace gemm_ag
using host::distributed::CommunicatorRef;

// Host entry points. ``run``: one GEMV instantiation per M in [1, kMaxM]
// (constexpr function-pointer table), then the spin consumer via PDL.
// ``gather``: push and assemble a slice an external GEMM computed.

template <uint32_t K, uint32_t N, uint32_t kWorld, uint32_t kMaxM, bool kUsePDL>
struct GEMMAGKernel {
  using TensorView = tvm::ffi::TensorView;

  // Columns of this rank's slice per producer block; sets the grid to
  // kNLocal / N_SPLIT. Measured on 2x4 GB300 TP8 at bs=1 (three reps each,
  // mean TPOT): 8 -> grid 112, 8.36 ms; 4 -> 224, 8.29 ms; 2 -> 448, 8.21 ms.
  // Standalone GEMV at the same shape: 4.15 / 3.20 / 2.56 us, and 16 -> 4.42 us,
  // so the trend is monotonic and 2 is the floor (the epilogue stores column
  // pairs, so N_SPLIT must stay even). 8 used to be the largest grid that fit
  // the old kNumProducerBlocks <= push.num_blocks bound; the producer now reads
  // a single phase counter, so the grid is free and 112 blocks did not even
  // fill one per SM.
  static constexpr uint32_t N_SPLIT = 2;
  static constexpr uint32_t kNLocal = N / kWorld;
  static constexpr uint32_t kGemvBlock = K / gemm_ag::kVecSize;
  static constexpr uint32_t kNumProducerBlocks = kNLocal / N_SPLIT;
  static_assert(N % kWorld == 0 && kNLocal % N_SPLIT == 0);

  using GemvFn = void (*)(gemm_ag::ProducerParams, const bf16_t*, const bf16_t*);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<GemvFn, kMaxM + 1>{
        nullptr, gemm_ag::gemm_ag_gemv_kernel<K, N, kWorld, I + 1, N_SPLIT, kUsePDL>...};
  }
  static constexpr auto kGemvTable = make_table(std::make_index_sequence<kMaxM>{});

  // Checks shared by both paths; returns the consumer half of the params.
  static gemm_ag::ConsumerParams consumer_params(
      const host::distributed::PushPlaneObj& push,
      uint32_t num_tokens,
      TensorView b,
      std::optional<TensorView> c,
      TensorView out) {
    using namespace host;
    CHECK_HOST(push.world_size == kWorld) << "the kernel is compiled for TP" << kWorld;
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";
    // the dense [world][M][N / world] staging prefix must fit one phase half
    CHECK_HOST(int64_t(num_tokens) * kNLocal * 2 <= push.slot_bytes)
        << "staging slice exceeds the push slot size " << push.slot_bytes;
    CHECK_HOST(2 * kWorld * push.slot_bytes <= std::numeric_limits<uint32_t>::max())
        << "push plane exceeds the 32-bit offset range";
    // one work block per counter slot at most, plus the cleanup block
    CHECK_HOST(push.num_blocks >= 2) << "needs a work block and a counter-update block";
    return gemm_ag::ConsumerParams{
        .ws_local = push.workspaces[push.rank],
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .b = static_cast<const bf16_t*>(b.data_ptr()),
        .c = c.has_value() ? static_cast<const bf16_t*>(c.value().data_ptr()) : nullptr,
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .num_rows = num_tokens,
    };
  }

  // {grid, block}: at most one work block per counter slot plus the cleanup
  // block; widen the block before striding, as each stride is a serial spin.
  static std::pair<uint32_t, uint32_t>
  consumer_shape(const host::distributed::PushPlaneObj& push, uint32_t num_tokens) {
    const uint32_t num_vecs = num_tokens * N / gemm_ag::kSpinVec;
    const uint32_t max_work_blocks = push.num_blocks - 1;
    uint32_t block = gemm_ag::kSpinBlock;
    while (block < gemm_ag::kMaxSpinBlock && host::div_ceil(num_vecs, block) > max_work_blocks) {
      block *= 2;
    }
    return {std::min<uint32_t>(host::div_ceil(num_vecs, block), max_work_blocks) + 1, block};
  }

  static void
  run(CommunicatorRef ref, TensorView x, TensorView weight, TensorView b, std::optional<TensorView> c, TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(b);
    if (c.has_value()) {
      TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(c.value());
    }
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxM);
    const auto params = consumer_params(push, num_tokens, b, c, out);
    // producer: GEMV. Its grid is not bound to the counter array: it reads only
    // counter[0] (see gemm_ag_gemv_kernel).
    const auto producer_params = gemm_ag::ProducerParams{
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .half_bytes = params.half_bytes,
        .rank = push.rank,
    };
    LaunchKernel(kNumProducerBlocks, kGemvBlock, device.unwrap())
        .enable_pdl(kUsePDL)(
            kGemvTable[num_tokens],
            producer_params,
            static_cast<const bf16_t*>(x.data_ptr()),
            static_cast<const bf16_t*>(weight.data_ptr()));

    // consumer: spin + add3
    using gemm_ag::spin_add3_kernel;
    const auto kernel =
        c.has_value() ? spin_add3_kernel<N, kWorld, 1, kUsePDL> : spin_add3_kernel<N, kWorld, 0, kUsePDL>;
    const auto [grid, block] = consumer_shape(push, num_tokens);
    LaunchKernel(grid, block, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }

  static void gather(CommunicatorRef ref, TensorView local, TensorView b, std::optional<TensorView> c, TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({M, kNLocal}).with_dtype<bf16_t>().with_device(device).verify(local);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(b);
    if (c.has_value()) {
      TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(c.value());
    }
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1);
    CHECK_HOST(reinterpret_cast<uintptr_t>(local.data_ptr()) % 16 == 0) << "the slice is read in 16B vectors";
    const auto params = gemm_ag::GatherParams{
        .consumer = consumer_params(push, num_tokens, b, c, out),
        .ws_mc = push.mc_workspace,
        .rank = push.rank,
        .local = static_cast<const bf16_t*>(local.data_ptr()),
    };
    using gemm_ag::gather_add3_kernel;
    const auto kernel =
        c.has_value() ? gather_add3_kernel<N, kWorld, 1, kUsePDL> : gather_add3_kernel<N, kWorld, 0, kUsePDL>;
    const auto [grid, block] = consumer_shape(push, num_tokens);
    LaunchKernel(grid, block, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
