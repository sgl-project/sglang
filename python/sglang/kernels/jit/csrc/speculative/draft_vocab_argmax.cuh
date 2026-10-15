// Greedy draft-token selection without gathering the full vocabulary.
//
// pack:   each rank reduces its vocab shard of the bf16 logits to (max, first index
//         of max), NaN read as -1e30 like the split argmax it replaces, then packs
//         the embedding row of that candidate plus (max, global index) into one row.
// select: after an all-gather of the packed rows, every rank picks the largest
//         max (lowest global index on ties), so the chosen token equals the
//         argmax over the gathered logits; it writes the token, advances the
//         position and copies the winner's embedding row for the next draft step.
// The embedding row matches the vocab-parallel lookup plus sum over ranks:
// adding zero rows only turns -0.0 into +0.0, which the copy applies too.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace vocab_argmax {

inline constexpr int kMetaElems = 8;  // packed row tail: max (fp32 bits), index (int32), padding

SGL_DEVICE bool better(float v, int32_t i, float bv, int32_t bi) {
  return v > bv || (v == bv && i < bi);
}

SGL_DEVICE void warp_argmax(float& v, int32_t& i) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, off);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, i, off);
    if (better(ov, oi, v, i)) {
      v = ov;
      i = oi;
    }
  }
}

template <int kThreads>
SGL_DEVICE void block_argmax(float& v, int32_t& i, float* s_v, int32_t* s_i) {
  constexpr int kWarps = kThreads / 32;
  warp_argmax(v, i);
  const int warp = static_cast<int>(threadIdx.x / 32);
  if (device::get_lane_id() == 0) {
    s_v[warp] = v;
    s_i[warp] = i;
  }
  __syncthreads();
  v = s_v[0];
  i = s_i[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    if (better(s_v[w], s_i[w], v, i)) {
      v = s_v[w];
      i = s_i[w];
    }
  }
}

// Map -0.0 to +0.0 (bit pattern 0x8000 -> 0x0000) and keep every other value.
SGL_DEVICE uint32_t clear_negative_zero(uint32_t two_bf16) {
  uint32_t lo = two_bf16 & 0xffffu;
  uint32_t hi = two_bf16 >> 16;
  lo = lo == 0x8000u ? 0u : lo;
  hi = hi == 0x8000u ? 0u : hi;
  return lo | (hi << 16);
}

}  // namespace vocab_argmax

struct VocabArgmaxPackParams {
  const bf16_t* logits;              // [V_local], written by the lm_head GEMM
  const bf16_t* __restrict__ embed;  // [V_local, H] embedding shard
  bf16_t* payload;                   // [H + 8]
  float* part_val;                   // [kBlocks]
  int32_t* part_idx;                 // [kBlocks]
  uint32_t* counter;                 // [1], zero between launches
  int32_t vocab;                     // V_local, multiple of 8
  int32_t vocab_start;               // global id of local row 0
};

struct VocabArgmaxSelectParams {
  const bf16_t* gathered;  // [world, H + 8]
  int64_t* topk_index;     // [1]
  float* topk_p;           // [1]
  int64_t* draft_tokens;   // row 0 of the draft chain, or null
  int64_t* positions;      // [1]
  bf16_t* next_embeds;     // [H]
  int32_t draft_column;
};

/**
 * \brief Shard argmax + candidate embedding pack. kBlocks CTAs scan the shard;
 * the last CTA to finish (atomic counter, fenced) reduces the partials, packs
 * the row and re-arms the counter.
 */
template <int kH, int kBlocks, int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) vocab_argmax_pack_kernel(const VocabArgmaxPackParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  __shared__ float s_v[kThreads / 32];
  __shared__ int32_t s_i[kThreads / 32];
  __shared__ bool s_last;

  PDLWaitPrimary<kUsePDL>();

  float best = -1e30f;
  int32_t bidx = 0x7fffffff;
  const int nvec = p.vocab / 8;
  for (int v = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x); v < nvec;
       v += kBlocks * kThreads) {
    vec_t x;
    x.load(p.logits, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float f = cast<fp32_t>(x[j]);
      f = f == f ? f : -1e30f;
      const int32_t idx = v * 8 + j;
      if (vocab_argmax::better(f, idx, best, bidx)) {
        best = f;
        bidx = idx;
      }
    }
  }
  vocab_argmax::block_argmax<kThreads>(best, bidx, s_v, s_i);
  if (threadIdx.x == 0) {
    p.part_val[blockIdx.x] = best;
    p.part_idx[blockIdx.x] = bidx;
    __threadfence();
    s_last = atomicAdd(p.counter, 1u) == kBlocks - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  float v = -1e30f;
  int32_t i = 0x7fffffff;
  if (threadIdx.x < kBlocks) {
    v = __ldcg(p.part_val + threadIdx.x);
    i = __ldcg(p.part_idx + threadIdx.x);
  }
  __syncthreads();
  vocab_argmax::block_argmax<kThreads>(v, i, s_v, s_i);

  const auto* row = reinterpret_cast<const uint4*>(p.embed + static_cast<int64_t>(i) * kH);
  auto* dst = reinterpret_cast<uint4*>(p.payload);
  for (int c = static_cast<int>(threadIdx.x); c < kH / 8; c += kThreads) {
    uint4 r = row[c];
    r.x = vocab_argmax::clear_negative_zero(r.x);
    r.y = vocab_argmax::clear_negative_zero(r.y);
    r.z = vocab_argmax::clear_negative_zero(r.z);
    r.w = vocab_argmax::clear_negative_zero(r.w);
    dst[c] = r;
  }
  if (threadIdx.x == 0) {
    auto* meta = reinterpret_cast<int32_t*>(p.payload + kH);
    meta[0] = __float_as_int(v);
    meta[1] = p.vocab_start + i;
    meta[2] = 0;
    meta[3] = 0;
    *p.counter = 0u;
  }
}

/** \brief Pick the winner among the gathered rows and publish it. One CTA. */
template <int kH, int kWorld, int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads)
    vocab_argmax_select_kernel(const VocabArgmaxSelectParams __grid_constant__ p) {
  constexpr int kRow = kH + vocab_argmax::kMetaElems;
  device::PDLWaitPrimary<kUsePDL>();
  int win = 0;
  const auto* meta0 = reinterpret_cast<const int32_t*>(p.gathered + kH);
  float bv = __int_as_float(meta0[0]);
  int32_t bi = meta0[1];
#pragma unroll
  for (int r = 1; r < kWorld; ++r) {
    const auto* meta = reinterpret_cast<const int32_t*>(p.gathered + r * kRow + kH);
    const float v = __int_as_float(meta[0]);
    const int32_t i = meta[1];
    if (vocab_argmax::better(v, i, bv, bi)) {
      bv = v;
      bi = i;
      win = r;
    }
  }
  const auto* row = reinterpret_cast<const uint4*>(p.gathered + win * kRow);
  auto* dst = reinterpret_cast<uint4*>(p.next_embeds);
  for (int c = static_cast<int>(threadIdx.x); c < kH / 8; c += kThreads) dst[c] = row[c];
  if (threadIdx.x == 0) {
    p.topk_index[0] = bi;
    p.topk_p[0] = 1.0f;
    if (p.draft_tokens != nullptr) p.draft_tokens[p.draft_column] = bi;
    p.positions[0] = p.positions[0] + 1;
  }
}

namespace vocab_argmax {

SGL_DEVICE void bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          device::ptx::to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(device::ptx::to_shared(bar))
      : "memory");
}

}  // namespace vocab_argmax

struct LmHeadArgmaxParams {
  const bf16_t* x;                    // [H] lm_head input, written by an earlier grid
  const bf16_t* __restrict__ w;       // [V_local, H] lm_head shard
  const bf16_t* __restrict__ embed;   // [V_local, H] embedding shard
  bf16_t* payload;                    // [H + 8]
  float* part_val;                    // [grid]
  int32_t* part_idx;                  // [grid]
  uint32_t* counter;                  // [1], zero between launches
  int32_t vocab;                      // V_local
  int32_t vocab_start;
};

/**
 * \brief lm_head GEMV fused with the shard argmax and the candidate pack.
 * Persistent: one CTA per SM owns a contiguous row range; a producer warp streams
 * kRows rows per stage into a shared-memory ring (bulk copies, issued before the
 * PDL wait since the weights are static); each consumer warp reduces one row of
 * a stage. Logits are rounded to bf16 before the argmax, as when they are stored.
 */
template <int kH, int kRows, int kStages, int kCtasPerSm, bool kUsePDL>
__global__ void __launch_bounds__((kRows + 1) * 32, kCtasPerSm)
    lm_head_argmax_kernel(const LmHeadArgmaxParams __grid_constant__ p) {
  using namespace device;
  static_assert(kRows >= 1 && kRows <= 8, "one consumer warp per row");
  constexpr int kRowBytes = kH * static_cast<int>(sizeof(bf16_t));
  constexpr int kStageBytes = kRows * kRowBytes;
  using cvec_t = AlignedVector<bf16_t, 8>;
  constexpr int kChunks = kH / 8 / static_cast<int>(kWarpThreads);
  static_assert(kH % (8 * kWarpThreads) == 0);

  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ uint64_t full_bar[kStages];
  __shared__ uint64_t empty_bar[kStages];
  __shared__ float s_v[kRows];
  __shared__ int32_t s_i[kRows];
  __shared__ bool s_last;

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int G = static_cast<int>(gridDim.x);
  const int c = static_cast<int>(blockIdx.x);
  const int base = p.vocab / G, rem = p.vocab % G;
  const int r0 = c * base + min(c, rem);
  const int nrows = base + (c < rem ? 1 : 0);
  const int nstages = (nrows + kRows - 1) / kRows;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      ptx::mbar_init(&full_bar[s], 1);
      ptx::mbar_init(&empty_bar[s], kRows);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == kRows) {
    if (lane == 0) {
      int stage = 0;
      uint32_t phase = 0;
      for (int t = 0; t < nstages; ++t) {
        const int rows = min(kRows, nrows - t * kRows);
        ptx::mbar_wait_parity(&empty_bar[stage], phase ^ 1);
        ptx::mbar_arrive_expect_tx(&full_bar[stage], static_cast<uint32_t>(rows) * kRowBytes);
        vocab_argmax::bulk_g2s(
            smem + stage * kStageBytes,
            p.w + static_cast<int64_t>(r0 + t * kRows) * kH,
            static_cast<uint32_t>(rows) * kRowBytes,
            &full_bar[stage]);
        if (++stage == kStages) {
          stage = 0;
          phase ^= 1;
        }
      }
      PDLWaitPrimary<kUsePDL>();
    }
  } else {
    PDLWaitPrimary<kUsePDL>();
    cvec_t xr[kChunks];
#pragma unroll
    for (int k = 0; k < kChunks; ++k) xr[k].load(p.x, k * kWarpThreads + lane);
    float best = -1e30f;
    int32_t bidx = 0x7fffffff;
    int stage = 0;
    uint32_t phase = 0;
    for (int t = 0; t < nstages; ++t) {
      ptx::mbar_wait_parity(&full_bar[stage], phase);
      const int rows = min(kRows, nrows - t * kRows);
      if (warp < rows) {
        const auto* row = reinterpret_cast<const bf16_t*>(smem + stage * kStageBytes) + warp * kH;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < kChunks; ++k) {
          cvec_t w;
          w.load(row, k * kWarpThreads + lane);
#pragma unroll
          for (int j = 0; j < 8; ++j) acc = fmaf(cast<fp32_t>(w[j]), cast<fp32_t>(xr[k][j]), acc);
        }
        acc = warp::reduce_sum(acc);
        float f = cast<fp32_t>(cast<bf16_t>(acc));
        f = f == f ? f : -1e30f;
        const int32_t idx = r0 + t * kRows + warp;
        if (vocab_argmax::better(f, idx, best, bidx)) {
          best = f;
          bidx = idx;
        }
      }
      if (lane == 0) ptx::mbar_arrive(&empty_bar[stage]);
      if (++stage == kStages) {
        stage = 0;
        phase ^= 1;
      }
    }
    if (lane == 0) {
      s_v[warp] = best;
      s_i[warp] = bidx;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = s_v[0];
    int32_t i = s_i[0];
#pragma unroll
    for (int w = 1; w < kRows; ++w) {
      if (vocab_argmax::better(s_v[w], s_i[w], v, i)) {
        v = s_v[w];
        i = s_i[w];
      }
    }
    p.part_val[c] = v;
    p.part_idx[c] = i;
    __threadfence();
    s_last = atomicAdd(p.counter, 1u) == static_cast<uint32_t>(G) - 1;
  }
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();
  if (!s_last) return;
  __threadfence();
  if (warp != 0) return;
  float v = -1e30f;
  int32_t i = 0x7fffffff;
  for (int b = lane; b < G; b += 32) {
    const float bv = __ldcg(p.part_val + b);
    const int32_t bi = __ldcg(p.part_idx + b);
    if (vocab_argmax::better(bv, bi, v, i)) {
      v = bv;
      i = bi;
    }
  }
  vocab_argmax::warp_argmax(v, i);
  const auto* erow = reinterpret_cast<const uint4*>(p.embed + static_cast<int64_t>(i) * kH);
  auto* dst = reinterpret_cast<uint4*>(p.payload);
  for (int k = lane; k < kH / 8; k += 32) {
    uint4 r = erow[k];
    r.x = vocab_argmax::clear_negative_zero(r.x);
    r.y = vocab_argmax::clear_negative_zero(r.y);
    r.z = vocab_argmax::clear_negative_zero(r.z);
    r.w = vocab_argmax::clear_negative_zero(r.w);
    dst[k] = r;
  }
  if (lane == 0) {
    auto* meta = reinterpret_cast<int32_t*>(p.payload + kH);
    meta[0] = __float_as_int(v);
    meta[1] = p.vocab_start + i;
    meta[2] = 0;
    meta[3] = 0;
    *p.counter = 0u;
  }
}

template <int kH, int kRows, int kStages, int kCtasPerSm, bool kUsePDL>
struct LmHeadArgmax {
  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView embed,
      const tvm::ffi::TensorView payload,
      const tvm::ffi::TensorView part_val,
      const tvm::ffi::TensorView part_idx,
      const tvm::ffi::TensorView counter,
      int64_t vocab_start) {
    using namespace host;
    auto V = SymbolicSize{"vocab_local"};
    auto G = SymbolicSize{"grid"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({V, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w).verify(embed);
    TensorMatcher({1, kH + vocab_argmax::kMetaElems})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(payload);
    TensorMatcher({G}).with_dtype<fp32_t>().with_device(device).verify(part_val);
    TensorMatcher({G}).with_dtype<int32_t>().with_device(device).verify(part_idx);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(counter);
    const DLDevice dev = device.unwrap();
    const uint32_t grid = runtime::get_sm_count(dev.device_id) * kCtasPerSm;
    CHECK_HOST(G.unwrap() >= grid) << "partial buffers smaller than the grid";
    const auto params = LmHeadArgmaxParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .embed = static_cast<const bf16_t*>(embed.data_ptr()),
        .payload = static_cast<bf16_t*>(payload.data_ptr()),
        .part_val = static_cast<float*>(part_val.data_ptr()),
        .part_idx = static_cast<int32_t*>(part_idx.data_ptr()),
        .counter = static_cast<uint32_t*>(counter.data_ptr()),
        .vocab = static_cast<int32_t>(V.unwrap()),
        .vocab_start = static_cast<int32_t>(vocab_start),
    };
    constexpr std::size_t kSmem = static_cast<std::size_t>(kStages) * kRows * kH * sizeof(bf16_t);
    auto kernel = lm_head_argmax_kernel<kH, kRows, kStages, kCtasPerSm, kUsePDL>;
    static const bool attr_set = [&] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      return true;
    }();
    (void)attr_set;
    LaunchKernel(grid, (kRows + 1) * 32, dev, kSmem).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <int kH, int kBlocks, int kWorld, bool kUsePDL>
struct DraftVocabArgmax {
  static constexpr int kThreads = 256;
  static_assert(kBlocks <= kThreads, "partials are reduced by one CTA");
  static_assert(kH % 8 == 0);

  static void pack(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView embed,
      const tvm::ffi::TensorView payload,
      const tvm::ffi::TensorView part_val,
      const tvm::ffi::TensorView part_idx,
      const tvm::ffi::TensorView counter,
      int64_t vocab_start) {
    using namespace host;
    auto V = SymbolicSize{"vocab_local"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, V}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({V, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(embed);
    TensorMatcher({1, kH + vocab_argmax::kMetaElems})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(payload);
    TensorMatcher({kBlocks}).with_dtype<fp32_t>().with_device(device).verify(part_val);
    TensorMatcher({kBlocks}).with_dtype<int32_t>().with_device(device).verify(part_idx);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(counter);
    const int64_t vocab = V.unwrap();
    CHECK_HOST(vocab % 8 == 0) << "local vocab " << vocab << " must be a multiple of 8";
    const auto params = VocabArgmaxPackParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .embed = static_cast<const bf16_t*>(embed.data_ptr()),
        .payload = static_cast<bf16_t*>(payload.data_ptr()),
        .part_val = static_cast<float*>(part_val.data_ptr()),
        .part_idx = static_cast<int32_t*>(part_idx.data_ptr()),
        .counter = static_cast<uint32_t*>(counter.data_ptr()),
        .vocab = static_cast<int32_t>(vocab),
        .vocab_start = static_cast<int32_t>(vocab_start),
    };
    LaunchKernel(kBlocks, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(vocab_argmax_pack_kernel<kH, kBlocks, kThreads, kUsePDL>, params);
  }

  static void select(
      const tvm::ffi::TensorView gathered,
      const tvm::ffi::TensorView topk_index,
      const tvm::ffi::TensorView topk_p,
      const tvm::ffi::TensorView draft_tokens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView next_embeds,
      int64_t draft_column) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kWorld * (kH + vocab_argmax::kMetaElems)})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(gathered);
    TensorMatcher({1, 1}).with_dtype<int64_t>().with_device(device).verify(topk_index);
    TensorMatcher({1, 1}).with_dtype<fp32_t>().with_device(device).verify(topk_p);
    TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(positions);
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(next_embeds);
    int64_t* tokens = nullptr;
    if (draft_tokens.numel() > 0) {
      auto S = SymbolicSize{"num_steps"};
      TensorMatcher({1, S}).with_dtype<int64_t>().with_device(device).verify(draft_tokens);
      CHECK_HOST(draft_column >= 0 && draft_column < S.unwrap()) << "draft column out of range";
      tokens = static_cast<int64_t*>(draft_tokens.data_ptr());
    }
    const auto params = VocabArgmaxSelectParams{
        .gathered = static_cast<const bf16_t*>(gathered.data_ptr()),
        .topk_index = static_cast<int64_t*>(topk_index.data_ptr()),
        .topk_p = static_cast<float*>(topk_p.data_ptr()),
        .draft_tokens = tokens,
        .positions = static_cast<int64_t*>(positions.data_ptr()),
        .next_embeds = static_cast<bf16_t*>(next_embeds.data_ptr()),
        .draft_column = static_cast<int32_t>(draft_column),
    };
    LaunchKernel(1, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(vocab_argmax_select_kernel<kH, kWorld, kThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
