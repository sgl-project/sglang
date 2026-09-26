#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <array>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace sglang {

namespace gemm_ag {

using device::distributed::Counter;

constexpr uint32_t kWorld = 8;                      // TP world size
constexpr uint32_t kVecSize = 32 / sizeof(bf16_t);  // 16 bf16 per 32B vector
constexpr uint32_t kSpinBlock = 128;                // consumer threads per block
constexpr uint32_t kSpinVec = 16 / sizeof(bf16_t);  // 8 bf16 (16B) per consumer thread

// Producer: per-rank column-slice GEMV, multicast store with Lamport markers.

struct ProducerParams {
  uint8_t* ws_mc;       // multicast VA of the push workspace base
  Counter* counter;     // per-block phase counters (READ only here)
  uint32_t half_bytes;  // bytes per phase half (world_size * slot_bytes)
  uint32_t rank;
};

// Tensor-core producer (N_SPLIT = 8, 4 <= M <= 8): A = the block's 8 weight rows,
// B = the M tokens. K is permuted within each 32-wide chunk (same map for A and
// B) so every thread issues 16B loads.
SGL_DEVICE void mma_bf16_m16n8k16(float (&d)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

template <uint32_t K, uint32_t N, uint32_t M, bool kUsePDL>
SGL_DEVICE void
gemm_ag_mma_producer(const ProducerParams& params, const bf16_t* __restrict__ x, const bf16_t* __restrict__ weight) {
  using namespace device;
  constexpr uint32_t kNSplit = 8;
  constexpr uint32_t kNLocal = N / kWorld;
  constexpr uint32_t kWarps = (K / kVecSize) / kWarpThreads;
  constexpr uint32_t kChunks = K / 32;
  static_assert(M <= 8);
  static_assert(kChunks % kWarps == 0, "K chunks must split evenly over the warps");
  constexpr uint32_t kChunksPerWarp = kChunks / kWarps;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % kWarpThreads;
  const uint32_t warp = tx / kWarpThreads;
  const uint32_t g = lane / 4;
  const uint32_t t = lane % 4;
  const bf16_t* w_row = weight + static_cast<int64_t>(params.rank * kNLocal + bx * kNSplit + g) * K;

  // weight prefetch before the PDL wait (input-independent addresses)
  uint4 w_vec[kChunksPerWarp];
#pragma unroll
  for (uint32_t c = 0; c < kChunksPerWarp; ++c) {
    w_vec[c] = *reinterpret_cast<const uint4*>(w_row + (warp * kChunksPerWarp + c) * 32 + 8 * t);
  }

  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = params.counter[0].get() & 1;

  uint4 x_vec[kChunksPerWarp];
#pragma unroll
  for (uint32_t c = 0; c < kChunksPerWarp; ++c) {
    x_vec[c] = g < M ? *reinterpret_cast<const uint4*>(x + g * K + (warp * kChunksPerWarp + c) * 32 + 8 * t)
                     : make_uint4(0, 0, 0, 0);
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (uint32_t c = 0; c < kChunksPerWarp; ++c) {
    mma_bf16_m16n8k16(acc, w_vec[c].x, w_vec[c].y, x_vec[c].x, x_vec[c].y);
    mma_bf16_m16n8k16(acc, w_vec[c].z, w_vec[c].w, x_vec[c].z, x_vec[c].w);
  }

  // D[g][2t, 2t + 1] = out[token 2t, 2t + 1][column g]
  __shared__ float s_part[kWarps][8][kNSplit];
  s_part[warp][2 * t][g] = acc[0];
  s_part[warp][2 * t + 1][g] = acc[1];
  __syncthreads();
  constexpr uint32_t kNumPairs = M * kNSplit / 2;
  if (tx < kNumPairs) {
    const uint32_t m = (2 * tx) / kNSplit;
    const uint32_t n = (2 * tx) % kNSplit;
    float lo = 0.f, hi = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w) {
      lo += s_part[w][m][n];
      hi += s_part[w][m][n + 1];
    }
    const auto pair = cast<bf16x2_t>(fp32x2_t{lo, hi});
    auto bits = *reinterpret_cast<const uint32_t*>(&pair);
    if (bits == 0) bits = 0x8000u;  // -0.0 in the first element: never all-zero
    const uint32_t elem = (params.rank * M + m) * kNLocal + bx * kNSplit + n;
    const auto base = reinterpret_cast<bf16_t*>(params.ws_mc + phase * params.half_bytes);
    ptx::multimem_store_relaxed(reinterpret_cast<uint32_t*>(base + elem), bits);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t K, uint32_t N, uint32_t M, uint32_t N_SPLIT, bool kUsePDL, bool kFP32 = false>
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

  // Note(kpham-sgl): the tensor-core path targets the spec-decode verify batch
  // (4 <= M <= 8). Plain decode (M <= 2) stays on CUDA cores, where tensor cores
  // measured 0.1-0.3 us slower on Rubin.
  if constexpr (N_SPLIT == 8 && !kFP32 && M >= 4 && M <= 8) {
    return gemm_ag_mma_producer<K, N, M, kUsePDL>(params, x, weight);
  }
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
    if constexpr (kFP32) {
      const uint32_t m = (2 * tx) / N_SPLIT;
      const uint32_t n = (2 * tx) % N_SPLIT;
      const uint32_t elem = (params.rank * M + m) * kNLocal + bx * N_SPLIT + n;
      const auto base = reinterpret_cast<uint32_t*>(params.ws_mc + phase * params.half_bytes);
      auto lo = __float_as_uint(packed.x);
      auto hi = __float_as_uint(packed.y);
      // Negative zero is numerically zero but distinct from the empty marker.
      if (lo == 0) lo = 0x80000000u;
      if (hi == 0) hi = 0x80000000u;
      ptx::multimem_store_relaxed(base + elem, lo);
      ptx::multimem_store_relaxed(base + elem + 1, hi);
    } else {
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
  }
  PDLTriggerSecondary<kUsePDL>();
}

// Consumer: Lamport spin + BF16 add3 or FP32 gather, one 16B vector per thread.

struct ConsumerParams {
  uint8_t* ws_local;      // LOCAL VA of the push workspace base (poll + reset)
  Counter* counter;       // per-block phase counters (read + flip)
  uint32_t num_counters;  // full counter array size (PushPlane::num_blocks)
  uint32_t half_bytes;    // bytes per phase half (world_size * slot_bytes)
  const bf16_t* b;        // [M, N]
  const bf16_t* c;        // may be null
  void* out;              // [M, N], bf16 add3 or float gather
  uint32_t num_rows;      // M
};

template <uint32_t N, bool kHasC, bool kUsePDL, bool kFP32 = false>
__global__ void spin_add3_kernel(const __grid_constant__ ConsumerParams params) {
  using namespace device;
  constexpr uint32_t kSpinVec = kFP32 ? 4 : 8;
  using output_t = std::conditional_t<kFP32, float, bf16_t>;
  using vec_t = std::conditional_t<kFP32, AlignedVector<float, 4>, AlignedVector<bf16x2_t, 4>>;
  constexpr uint32_t kNLocal = N / kWorld;
  static_assert(N % kSpinVec == 0, "rows must stay 16B aligned");
  static_assert(kNLocal % kSpinVec == 0, "a vector must never cross a rank block");
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const uint32_t tid = bx * kSpinBlock + tx;
  const uint32_t elem = tid * kSpinVec;  // first element of this thread's vector
  const uint32_t phase = params.counter[bx].get() & 1;

  PDLTriggerSecondary<kUsePDL>();

  // use the last block to clean up: it flips ITS OWN counter and every one
  // past the grid (work blocks flip [0, num_blocks - 1) themselves)
  if (const auto num_blocks = gridDim.x; bx == num_blocks - 1) {
    [[unlikely]];
    __syncthreads();  // ensure phase is ready for all threads
    for (uint32_t i = num_blocks - 1 + tx; i < params.num_counters; i += kSpinBlock) {
      params.counter[i].set(phase ^ 1);
    }
    return void();  // this block is done, no output to write
  }

  // Deliberately NO PDLWaitPrimary: the dependency is carried through data
  if (elem < params.num_rows * N) {
    const auto row = elem / N;
    const auto col = elem % N;
    // out[row, col] lives at half[col / kNLocal][row][col % kNLocal] of the
    // dense [world][M][N / world] prefix of the current phase half
    const auto base = reinterpret_cast<output_t*>(params.ws_local + phase * params.half_bytes);
    const auto src = base + ((col / kNLocal) * params.num_rows + row) * kNLocal + col % kNLocal;
    vec_t b_vec, c_vec;
    if constexpr (!kFP32) {
      b_vec.load(params.b + elem);
      if constexpr (kHasC) c_vec.load(params.c + elem);
    }
    uint4 raw;
    do {
      ptx::ld_relaxed_16B(raw, src, 0);
    } while (raw.x == 0 || raw.y == 0 || raw.z == 0 || raw.w == 0);
    const auto& gathered = *reinterpret_cast<const vec_t*>(&raw);
    if constexpr (kFP32) {
      gathered.store(static_cast<float*>(params.out) + elem);
    } else {
      vec_t out_vec;
#pragma unroll
      for (uint32_t j = 0; j < kSpinVec / 2; ++j) {
        using Trait = DTypeTrait<bf16x2_t>;
        out_vec[j] = Trait::add(gathered[j], b_vec[j]);
        if constexpr (kHasC) out_vec[j] = Trait::add(out_vec[j], c_vec[j]);
      }
      out_vec.store(static_cast<bf16_t*>(params.out) + elem);
    }
    AlignedVector<uint32_t, 4> zero;
    zero.fill(0);
    zero.store(src);
  }
  __syncthreads();
  if (tx == 0) params.counter[bx].set(phase ^ 1);
}

}  // namespace gemm_ag
using host::distributed::CommunicatorRef;

// Host entry point (tiny_gemm style: one GEMV instantiation per M in
// [1, kMaxM] selected through a constexpr function-pointer table, then the
// spin consumer launched with PDL right behind it). Any (K, N) that passes
// the kernels' static_asserts works; Kimi-K3 uses (3584, 7168).

template <uint32_t K, uint32_t N, uint32_t kMaxM, bool kUsePDL, uint32_t kSplitN = 2, bool kFP32 = false>
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
  // SM107 BS1 uses eight columns per block (selected by the Python dispatch).
  static constexpr uint32_t N_SPLIT = kSplitN;
  static constexpr uint32_t kNLocal = N / gemm_ag::kWorld;
  static constexpr uint32_t kGemvBlock = K / gemm_ag::kVecSize;
  static constexpr uint32_t kNumProducerBlocks = kNLocal / N_SPLIT;
  static_assert(kNLocal % N_SPLIT == 0);

  using GemvFn = void (*)(gemm_ag::ProducerParams, const bf16_t*, const bf16_t*);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<GemvFn, kMaxM + 1>{
        nullptr, gemm_ag::gemm_ag_gemv_kernel<K, N, I + 1, N_SPLIT, kUsePDL, kFP32>...};
  }
  static constexpr auto kGemvTable = make_table(std::make_index_sequence<kMaxM>{});

  static void
  run(CommunicatorRef ref,
      TensorView x,
      TensorView weight,
      std::optional<TensorView> b,
      std::optional<TensorView> c,
      TensorView out) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    CHECK_HOST(b.has_value() != kFP32) << "the BF16 add3 epilogue takes b; the FP32 gather does not";
    if (b.has_value()) {
      TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(b.value());
    }
    if (c.has_value()) {
      TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(c.value());
    }
    using output_t = std::conditional_t<kFP32, float, bf16_t>;
    TensorMatcher({M, N}).with_dtype<output_t>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxM);
    CHECK_HOST(push.world_size == gemm_ag::kWorld) << "the kernel is compiled for TP" << gemm_ag::kWorld;
    CHECK_HOST(push.mc_workspace != nullptr) << "requires a multicast-capable push plane";
    CHECK_HOST(int64_t(num_tokens) * kNLocal * sizeof(output_t) <= push.slot_bytes)
        << "staging slice exceeds the push slot size " << push.slot_bytes;
    // The producer grid is no longer bound to the counter array: it reads only
    // counter[0] (see gemm_ag_gemv_kernel). The consumer grid still is.
    CHECK_HOST(push.num_blocks > 0) << "no push blocks available";
    // producer: GEMV
    const auto producer_params = gemm_ag::ProducerParams{
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
    };
    LaunchKernel(kNumProducerBlocks, kGemvBlock, device.unwrap())
        .enable_pdl(kUsePDL)(
            kGemvTable[num_tokens],
            producer_params,
            static_cast<const bf16_t*>(x.data_ptr()),
            static_cast<const bf16_t*>(weight.data_ptr()));

    // consumer: spin + add3
    const auto consumer_params = gemm_ag::ConsumerParams{
        .ws_local = push.workspaces[push.rank],
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .b = b.has_value() ? static_cast<const bf16_t*>(b.value().data_ptr()) : nullptr,
        .c = c.has_value() ? static_cast<const bf16_t*>(c.value().data_ptr()) : nullptr,
        .out = out.data_ptr(),
        .num_rows = num_tokens,
    };
    const auto num_vecs = num_tokens * N / (16 / sizeof(output_t));
    const auto num_consumers = host::div_ceil(num_vecs, gemm_ag::kSpinBlock);
    CHECK_HOST(num_consumers + 1 <= push.num_blocks);
    // use last block to clean up the counter
    const auto num_consumer_blocks = num_consumers + 1;
    using gemm_ag::spin_add3_kernel;
    const auto kernel = c.has_value() ? spin_add3_kernel<N, 1, kUsePDL, kFP32> : spin_add3_kernel<N, 0, kUsePDL, kFP32>;
    host::LaunchKernel(num_consumer_blocks, gemm_ag::kSpinBlock, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, consumer_params);
  }
};

}  // namespace sglang
