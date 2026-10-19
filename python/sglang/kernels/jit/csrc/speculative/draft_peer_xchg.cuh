// Draft-step exchanges between the two TP ranks through peer-mapped (symmetric) memory,
// replacing a multimem all-gather plus a separate consumer kernel.
//
// argmax: the shard argmax + candidate pack of `vocab_argmax_pack_kernel`; the last CTA
//         of a row stores the packed row into the peer's slot, polls its own slot for the
//         peer's row and runs the selection of `vocab_argmax_select_kernel` on the two
//         rows in rank order. Same keys, same comparisons: bitwise equal to pack +
//         all-gather + select.
// fc_sum: each CTA stores its slice of this rank's fp32 fc partial into the peer's slot,
//         polls its own slot for the peer's slice and writes bf16(p0 + p1), the
//         rank-order sum of `_ordered_partial_sum_kernel` (two-term fp32 addition is
//         commutative).
//
// Arrival is detected from the data (Lamport): a slot holds a marker word until the peer's
// store lands, the consumer re-arms the words it read, and slots alternate between two
// halves by a per-row (per-CTA) epoch that both ranks flip in lockstep. Markers never
// collide with data: packed rows carry no -0.0 halves (cleared by the pack) and their
// meta words never equal 0x80008000; fp32 partials are sent with NaNs canonicalized, so
// 0xffffffff never appears (fp32 addition returns the canonical NaN either way).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace peer_xchg {

inline constexpr int kMetaElems = 8;  // same packed-row tail as draft_vocab_argmax.cuh
inline constexpr uint32_t kRowMarker = 0x80008000u;
inline constexpr uint32_t kFp32Marker = 0xffffffffu;
inline constexpr uint32_t kCanonicalNaN = 0x7fffffffu;

SGL_DEVICE uint32_t order_key(float f) {
  uint32_t b = __float_as_uint(f);
  b = b == 0x80000000u ? 0u : b;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

SGL_DEVICE uint32_t value_key(float f, uint32_t nan_key) {
  return f == f ? order_key(f) : nan_key;
}

SGL_DEVICE bool better(uint32_t v, int32_t i, uint32_t bv, int32_t bi) {
  return v > bv || (v == bv && i < bi);
}

SGL_DEVICE void warp_argmax(uint32_t& v, int32_t& i) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint32_t ov = __shfl_xor_sync(0xffffffffu, v, off);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, i, off);
    if (better(ov, oi, v, i)) {
      v = ov;
      i = oi;
    }
  }
}

template <int kThreads>
SGL_DEVICE void block_argmax(uint32_t& v, int32_t& i, uint32_t* s_v, int32_t* s_i) {
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

SGL_DEVICE uint32_t clear_negative_zero(uint32_t two_bf16) {
  uint32_t lo = two_bf16 & 0xffffu;
  uint32_t hi = two_bf16 >> 16;
  lo = lo == 0x8000u ? 0u : lo;
  hi = hi == 0x8000u ? 0u : hi;
  return lo | (hi << 16);
}

SGL_DEVICE bool has_word(const uint4& v, uint32_t w) {
  return v.x == w || v.y == w || v.z == w || v.w == w;
}

SGL_DEVICE uint32_t canonical_nan(uint32_t bits) {
  return (bits & 0x7fffffffu) > 0x7f800000u ? kCanonicalNaN : bits;
}

// A peer that never arrives would hang the GPU; fail loudly instead (~seconds of spinning).
inline constexpr uint32_t kSpinLimit = 1u << 26;

}  // namespace peer_xchg

struct DraftArgmaxXchgParams {
  const bf16_t* logits;              // [M, V_local], written by the lm_head GEMM
  const bf16_t* __restrict__ embed;  // [V_local, H]
  uint32_t* part_key;                // [M, kBlocks]
  int32_t* part_idx;                 // [M, kBlocks]
  uint32_t* counter;                 // [M], zero between launches
  const int64_t* token_map;          // [V_local] vocabulary id of each local row, or null
  bf16_t* slots;                     // this rank's exchange rows [2, max_rows, 2, H + 8]
  bf16_t* peer_slots;                // the peer's exchange rows, same layout
  uint32_t* epoch;                   // [max_rows], local
  int64_t* topk_index;               // [M]
  float* topk_p;                     // [M]
  int64_t* draft_tokens;             // [M, S] or null
  int64_t* positions;                // [M] or null
  int64_t* mrope;                    // [3, >= M] or null
  bf16_t* next_embeds;               // [M, H]
  int32_t vocab;
  int32_t vocab_start;
  uint32_t nan_key;
  int32_t rank;
  int32_t max_rows;
  int32_t draft_column;
  int32_t draft_stride;
  int32_t mrope_stride;
};

/**
 * \brief Shard argmax + pack, peer exchange of the packed rows and selection, one launch.
 * Grid (kBlocks, M); the last CTA of a row (atomic counter) reduces the partials, sends the
 * packed row, waits for the peer's and publishes the winner like the select kernel.
 */
template <int kH, int kBlocks, int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) draft_argmax_xchg_kernel(const DraftArgmaxXchgParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kRow = kH + peer_xchg::kMetaElems;
  constexpr int kRowVecs = kRow / 8;
  static_assert(kRow % 8 == 0);
  __shared__ uint32_t s_v[kThreads / 32];
  __shared__ int32_t s_i[kThreads / 32];
  __shared__ bool s_last;
  __shared__ int s_win;
  __shared__ uint32_t s_bi;

  PDLWaitPrimary<kUsePDL>();

  const int64_t m = blockIdx.y;
  const bf16_t* logits = p.logits + m * p.vocab;
  uint32_t* part_key = p.part_key + m * kBlocks;
  int32_t* part_idx = p.part_idx + m * kBlocks;
  uint32_t* counter = p.counter + m;
  uint32_t best = 0u;
  int32_t bidx = 0x7fffffff;
  const int nvec = p.vocab / 8;
  for (int v = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x); v < nvec;
       v += kBlocks * kThreads) {
    vec_t x;
    x.load(logits, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t key = peer_xchg::value_key(cast<fp32_t>(x[j]), p.nan_key);
      const int32_t idx = v * 8 + j;
      if (peer_xchg::better(key, idx, best, bidx)) {
        best = key;
        bidx = idx;
      }
    }
  }
  peer_xchg::block_argmax<kThreads>(best, bidx, s_v, s_i);
  if (threadIdx.x == 0) {
    part_key[blockIdx.x] = best;
    part_idx[blockIdx.x] = bidx;
    __threadfence();
    s_last = atomicAdd(counter, 1u) == kBlocks - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  uint32_t v = 0u;
  int32_t i = 0x7fffffff;
  if (threadIdx.x < kBlocks) {
    v = __ldcg(part_key + threadIdx.x);
    i = __ldcg(part_idx + threadIdx.x);
  }
  __syncthreads();
  peer_xchg::block_argmax<kThreads>(v, i, s_v, s_i);

  const uint32_t half = p.epoch[m] & 1u;
  const int64_t row_stride = static_cast<int64_t>(2) * kRow;
  const int64_t base = (static_cast<int64_t>(half) * p.max_rows + m) * row_stride;
  bf16_t* own = p.slots + base + static_cast<int64_t>(p.rank) * kRow;           // my row, read locally
  bf16_t* remote = p.peer_slots + base + static_cast<int64_t>(p.rank) * kRow;   // my row, at the peer
  bf16_t* incoming = p.slots + base + static_cast<int64_t>(1 - p.rank) * kRow;  // the peer's row, here

  const auto* erow = reinterpret_cast<const uint4*>(p.embed + static_cast<int64_t>(i) * kH);
  for (int c = static_cast<int>(threadIdx.x); c < kH / 8; c += kThreads) {
    uint4 r = erow[c];
    r.x = peer_xchg::clear_negative_zero(r.x);
    r.y = peer_xchg::clear_negative_zero(r.y);
    r.z = peer_xchg::clear_negative_zero(r.z);
    r.w = peer_xchg::clear_negative_zero(r.w);
    ptx::st_relaxed_16B(r, remote, c);
    ptx::st_global_16B(r, own, c);
  }
  if (threadIdx.x == 0) {
    uint4 meta;
    meta.x = v;
    meta.y = static_cast<uint32_t>(p.token_map != nullptr ? p.token_map[i] : p.vocab_start + i);
    meta.z = 0u;
    meta.w = 0u;
    ptx::st_relaxed_16B(meta, remote, kH / 8);
    ptx::st_global_16B(meta, own, kH / 8);
    *counter = 0u;
  }

  // Wait for every vector of the peer's row; the meta vector lands last only by chance,
  // so each thread polls its own share.
  for (int c = static_cast<int>(threadIdx.x); c < kRowVecs; c += kThreads) {
    uint4 r;
    uint32_t spins = 0;
    do {
      ptx::ld_relaxed_16B(r, incoming, c);
      if (++spins == peer_xchg::kSpinLimit) __trap();
    } while (peer_xchg::has_word(r, peer_xchg::kRowMarker));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const bf16_t* rows = p.slots + base;
    int win = 0;
    uint4 meta;
    ptx::ld_relaxed_16B(meta, rows, kH / 8);
    uint32_t bv = meta.x;
    int32_t bi = static_cast<int32_t>(meta.y);
#pragma unroll
    for (int r = 1; r < 2; ++r) {
      ptx::ld_relaxed_16B(meta, rows + r * kRow, kH / 8);
      const int32_t mi = static_cast<int32_t>(meta.y);
      if (peer_xchg::better(meta.x, mi, bv, bi)) {
        bv = meta.x;
        bi = mi;
        win = r;
      }
    }
    s_win = win;
    s_bi = static_cast<uint32_t>(bi);
  }
  __syncthreads();
  const int win = s_win;
  const int32_t bi = static_cast<int32_t>(s_bi);
  const bf16_t* wrow = p.slots + base + static_cast<int64_t>(win) * kRow;
  auto* dst = reinterpret_cast<uint4*>(p.next_embeds + m * kH);
  for (int c = static_cast<int>(threadIdx.x); c < kH / 8; c += kThreads) {
    uint4 r;
    ptx::ld_relaxed_16B(r, wrow, c);
    dst[c] = r;
  }
  if (threadIdx.x == 0) {
    p.topk_index[m] = bi;
    p.topk_p[m] = 1.0f;
    if (p.draft_tokens != nullptr) p.draft_tokens[m * p.draft_stride + p.draft_column] = bi;
    if (p.positions != nullptr) p.positions[m] = p.positions[m] + 1;
    if (p.mrope != nullptr) {
#pragma unroll
      for (int d = 0; d < 3; ++d)
        p.mrope[d * p.mrope_stride + m] += 1;
    }
  }
  // Re-arm the peer's slot once every thread has read it, then flip this row's half.
  __syncthreads();
  const uint4 marker =
      make_uint4(peer_xchg::kRowMarker, peer_xchg::kRowMarker, peer_xchg::kRowMarker, peer_xchg::kRowMarker);
  for (int c = static_cast<int>(threadIdx.x); c < kRowVecs; c += kThreads)
    ptx::st_global_16B(marker, incoming, c);
  if (threadIdx.x == 0) p.epoch[m] = p.epoch[m] ^ 1u;
}

struct FcXchgSumParams {
  const float* partial;  // [M, N] this rank's fp32 partial
  bf16_t* out;           // [M, N]
  float* slots;          // this rank's slots [2, max_rows, N] (the peer writes them)
  float* peer_slots;     // the peer's slots
  uint32_t* epoch;       // [max_rows, N / (4 * kThreads)], local
  int32_t n;
  int32_t max_rows;
};

/** \brief out = bf16(p_rank0 + p_rank1) with the peer's partial pushed into this rank's slot. */
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) fc_xchg_sum_kernel(const FcXchgSumParams __grid_constant__ p) {
  using namespace device;
  PDLWaitPrimary<kUsePDL>();
  const int64_t m = blockIdx.y;
  const int chunks = static_cast<int>(gridDim.x);
  const int64_t vec = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;  // float4 index in the row
  uint32_t* epoch = p.epoch + m * chunks + blockIdx.x;
  const uint32_t half = *epoch & 1u;
  const int64_t base = (static_cast<int64_t>(half) * p.max_rows + m) * p.n;
  float* remote = p.peer_slots + base;
  float* incoming = p.slots + base;

  uint4 mine = reinterpret_cast<const uint4*>(p.partial + m * p.n)[vec];
  uint4 sent;
  sent.x = peer_xchg::canonical_nan(mine.x);
  sent.y = peer_xchg::canonical_nan(mine.y);
  sent.z = peer_xchg::canonical_nan(mine.z);
  sent.w = peer_xchg::canonical_nan(mine.w);
  ptx::st_relaxed_16B(sent, remote, vec);

  uint4 theirs;
  uint32_t spins = 0;
  do {
    ptx::ld_relaxed_16B(theirs, incoming, vec);
    if (++spins == peer_xchg::kSpinLimit) __trap();
  } while (peer_xchg::has_word(theirs, peer_xchg::kFp32Marker));
  const uint4 marker =
      make_uint4(peer_xchg::kFp32Marker, peer_xchg::kFp32Marker, peer_xchg::kFp32Marker, peer_xchg::kFp32Marker);
  ptx::st_global_16B(marker, incoming, vec);

  // fp32 a + b == b + a, so this is the rank-order sum whichever rank runs it.
  const float s0 = __uint_as_float(mine.x) + __uint_as_float(theirs.x);
  const float s1 = __uint_as_float(mine.y) + __uint_as_float(theirs.y);
  const float s2 = __uint_as_float(mine.z) + __uint_as_float(theirs.z);
  const float s3 = __uint_as_float(mine.w) + __uint_as_float(theirs.w);
  __nv_bfloat162 lo = __floats2bfloat162_rn(s0, s1);
  __nv_bfloat162 hi = __floats2bfloat162_rn(s2, s3);
  uint2 packed;
  packed.x = *reinterpret_cast<uint32_t*>(&lo);
  packed.y = *reinterpret_cast<uint32_t*>(&hi);
  reinterpret_cast<uint2*>(p.out + m * p.n)[vec] = packed;
  __syncthreads();
  if (threadIdx.x == 0) *epoch = half ^ 1u;
}

template <int kH, int kBlocks, bool kUsePDL>
struct DraftArgmaxXchg {
  static constexpr int kThreads = 256;
  static_assert(kBlocks <= kThreads, "partials are reduced by one CTA");
  static_assert(kH % 8 == 0);

  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView embed,
      const tvm::ffi::TensorView part_key,
      const tvm::ffi::TensorView part_idx,
      const tvm::ffi::TensorView counter,
      const tvm::ffi::TensorView token_map,
      const tvm::ffi::TensorView slots,
      int64_t peer_slots,
      const tvm::ffi::TensorView epoch,
      const tvm::ffi::TensorView topk_index,
      const tvm::ffi::TensorView topk_p,
      const tvm::ffi::TensorView draft_tokens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView mrope_positions,
      const tvm::ffi::TensorView next_embeds,
      int64_t vocab_start,
      int64_t nan_key,
      int64_t rank,
      int64_t draft_column) {
    using namespace host;
    constexpr int kRow = kH + peer_xchg::kMetaElems;
    auto M = SymbolicSize{"rows"};
    auto V = SymbolicSize{"vocab_local"};
    auto R = SymbolicSize{"max_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, V}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({V, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(embed);
    TensorMatcher({M, kBlocks}).with_dtype<int32_t>().with_device(device).verify(part_key);
    TensorMatcher({M, kBlocks}).with_dtype<int32_t>().with_device(device).verify(part_idx);
    TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(counter);
    TensorMatcher({2, R, 2, kRow}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(slots);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(epoch);
    TensorMatcher({M, 1}).with_dtype<int64_t>().with_device(device).verify(topk_index);
    TensorMatcher({M, 1}).with_dtype<fp32_t>().with_device(device).verify(topk_p);
    TensorMatcher({M, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(next_embeds);
    const int64_t vocab = V.unwrap();
    CHECK_HOST(vocab % 8 == 0) << "local vocab " << vocab << " must be a multiple of 8";
    CHECK_HOST(M.unwrap() <= R.unwrap()) << "more rows than exchange slots";
    CHECK_HOST(rank == 0 || rank == 1) << "two-rank exchange";
    CHECK_HOST(peer_slots != 0 && peer_slots % 16 == 0) << "peer slots must be a 16-byte aligned pointer";
    const int64_t* map = nullptr;
    if (token_map.numel() > 0) {
      TensorMatcher({V}).with_dtype<int64_t>().with_device(device).verify(token_map);
      map = static_cast<const int64_t*>(token_map.data_ptr());
    }
    int64_t* pos = nullptr;
    if (positions.numel() > 0) {
      TensorMatcher({M}).with_dtype<int64_t>().with_device(device).verify(positions);
      pos = static_cast<int64_t*>(positions.data_ptr());
    }
    int64_t* mrope = nullptr;
    int32_t mrope_stride = 0;
    if (mrope_positions.numel() > 0) {
      auto N = SymbolicSize{"mrope_tokens"};
      auto S = SymbolicSize{"mrope_row_stride"};
      TensorMatcher({3, N}).with_strides({S, 1}).with_dtype<int64_t>().with_device(device).verify(mrope_positions);
      CHECK_HOST(N.unwrap() == M.unwrap()) << "mrope positions must have one column per row";
      mrope = static_cast<int64_t*>(mrope_positions.data_ptr());
      mrope_stride = static_cast<int32_t>(S.unwrap());
    }
    int64_t* tokens = nullptr;
    int32_t stride = 0;
    if (draft_tokens.numel() > 0) {
      auto S = SymbolicSize{"num_steps"};
      TensorMatcher({M, S}).with_dtype<int64_t>().with_device(device).verify(draft_tokens);
      CHECK_HOST(draft_column >= 0 && draft_column < S.unwrap()) << "draft column out of range";
      tokens = static_cast<int64_t*>(draft_tokens.data_ptr());
      stride = static_cast<int32_t>(S.unwrap());
    }
    const auto params = DraftArgmaxXchgParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .embed = static_cast<const bf16_t*>(embed.data_ptr()),
        .part_key = static_cast<uint32_t*>(part_key.data_ptr()),
        .part_idx = static_cast<int32_t*>(part_idx.data_ptr()),
        .counter = static_cast<uint32_t*>(counter.data_ptr()),
        .token_map = map,
        .slots = static_cast<bf16_t*>(slots.data_ptr()),
        .peer_slots = reinterpret_cast<bf16_t*>(peer_slots),
        .epoch = static_cast<uint32_t*>(epoch.data_ptr()),
        .topk_index = static_cast<int64_t*>(topk_index.data_ptr()),
        .topk_p = static_cast<float*>(topk_p.data_ptr()),
        .draft_tokens = tokens,
        .positions = pos,
        .mrope = mrope,
        .next_embeds = static_cast<bf16_t*>(next_embeds.data_ptr()),
        .vocab = static_cast<int32_t>(vocab),
        .vocab_start = static_cast<int32_t>(vocab_start),
        .nan_key = static_cast<uint32_t>(nan_key),
        .rank = static_cast<int32_t>(rank),
        .max_rows = static_cast<int32_t>(R.unwrap()),
        .draft_column = static_cast<int32_t>(draft_column),
        .draft_stride = stride,
        .mrope_stride = mrope_stride,
    };
    LaunchKernel(dim3(kBlocks, static_cast<uint32_t>(M.unwrap())), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(draft_argmax_xchg_kernel<kH, kBlocks, kThreads, kUsePDL>, params);
  }
};

template <bool kUsePDL>
struct FcXchgSum {
  static constexpr int kThreads = 256;

  static void
  run(const tvm::ffi::TensorView partial,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView slots,
      int64_t peer_slots,
      const tvm::ffi::TensorView epoch) {
    using namespace host;
    auto M = SymbolicSize{"rows"};
    auto N = SymbolicSize{"cols"};
    auto R = SymbolicSize{"max_rows"};
    auto C = SymbolicSize{"chunks"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, N}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(partial);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).ensure_alignment(8).verify(out);
    TensorMatcher({2, R, N}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(slots);
    TensorMatcher({R, C}).with_dtype<int32_t>().with_device(device).verify(epoch);
    const int64_t n = N.unwrap();
    CHECK_HOST(n % (4 * kThreads) == 0 && n / (4 * kThreads) == C.unwrap()) << "columns must fill the CTAs";
    CHECK_HOST(M.unwrap() <= R.unwrap()) << "more rows than exchange slots";
    CHECK_HOST(peer_slots != 0 && peer_slots % 16 == 0) << "peer slots must be a 16-byte aligned pointer";
    const auto params = FcXchgSumParams{
        .partial = static_cast<const float*>(partial.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .slots = static_cast<float*>(slots.data_ptr()),
        .peer_slots = reinterpret_cast<float*>(peer_slots),
        .epoch = static_cast<uint32_t*>(epoch.data_ptr()),
        .n = static_cast<int32_t>(n),
        .max_rows = static_cast<int32_t>(R.unwrap()),
    };
    LaunchKernel(dim3(static_cast<uint32_t>(C.unwrap()), static_cast<uint32_t>(M.unwrap())), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(fc_xchg_sum_kernel<kThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
