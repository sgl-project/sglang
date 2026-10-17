// DSv4.1 sparse MLA decode attention, SM100, small batch. KV: V4.1 fp8 SWA cache plus an optional
// V4.1 fp8 or fp4 (e2m1 + per-16 e4m3 scales, 288 B/token) extra cache.
//
// One cluster per query row, one CTA per tile of 64 keys (SWA tiles first, then
// extra tiles), single-block softmax per CTA, partials combined over DSMEM.
//
// V4.1 row: 512 e4m3 values, and in the page's scale rows 16 ue8m0 scales, one
// per 32 values. KV is dequantized to bf16 with the scale applied by the
// conversion itself (exact: e4m3 times a power of two fits bf16), so the MMAs,
// S and P carry no scale bookkeeping. Warp g owns head-dim group g (64 dims);
// lane (r4 = lane / 4, q4 = lane % 4) holds, for keys r4 + 8j, the 8 dims
// [8 q4, 8 q4 + 8) of both 32-dim halves of the group. Needs PTX ISA 9.2
// (CUDA 13.2+) for cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2.
//
// The output is the activation FlashMLA MegaAttention hands DeepGEMM fp8_einsum (wo_a): per
// token and group of 8 heads, 4096 e4m3 in interleave_output_columns order (column
// chunk * 256 + head * 32 + j = dim chunk * 32 + j), and per 32-column block a UE8M0 exponent,
// 4 packed per int32, stored MN-major [G][32][T_al]. The quantized values are the bf16-rounded
// output after the fused inverse RoPE, as the DeepSeek reference stores it; the scale/rounding
// arithmetic is FlashMLA's epilogue bit for bit.
#pragma once

#include <sgl_kernel/bits.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sglang::device::dsv4_decode {

namespace ptx {

using device::ptx::mbar_init;
using device::ptx::to_shared;

SGL_DEVICE uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}
SGL_DEVICE uint32_t cluster_ctarank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
SGL_DEVICE void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// The waits and the expect_tx order the smem reads of what peers st.async'd in, hence local
// copies of device::ptx's with a "memory" clobber.
SGL_DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes) : "memory");
}
SGL_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity)
      : "memory");
}
// 16 B of registers to `dst` in a peer CTA, completing `bytes` on its mbarrier `bar`.
SGL_DEVICE void st_async_16(uint32_t dst, const void* src, uint32_t bar) {
  const auto w = static_cast<const uint32_t*>(src);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "r"(w[0]),
               "r"(w[1]),
               "r"(w[2]),
               "r"(w[3]),
               "r"(bar)
               : "memory");
}
SGL_DEVICE bf16x2_t movmatrix_trans(bf16x2_t x) {
  bf16x2_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
               : "=r"(*reinterpret_cast<uint32_t*>(&y))
               : "r"(*reinterpret_cast<const uint32_t*>(&x)));
  return y;
}
SGL_DEVICE void ldmatrix_x4(const void* smem, void* dst) {
  const auto o = static_cast<uint32_t*>(dst);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
               : "r"(to_shared(smem)));
}
// c (4 f32) += a (4 bf16x2) x (b0, b1).
SGL_DEVICE void mma_bf16(void* acc, const void* a, bf16x2_t b0, bf16x2_t b1) {
  const auto c = static_cast<float*>(acc);
  const auto ar = static_cast<const uint32_t*>(a);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(ar[0]),
        "r"(ar[1]),
        "r"(ar[2]),
        "r"(ar[3]),
        "r"(*reinterpret_cast<const uint32_t*>(&b0)),
        "r"(*reinterpret_cast<const uint32_t*>(&b1)));
}
// 8 e4m3 at `src` (8 B aligned) times 2^(ue8m0 - 127) -> bf16 (exact: a power-of-two scale fits bf16).
SGL_DEVICE AlignedVector<bf16x2_t, 4> e4m3x8_to_bf16x8_scaled(const void* src, uint32_t ue8m0) {
  const uint2 w = *static_cast<const uint2*>(src);
  const uint32_t sf = ue8m0 * 0x0101u;  // the scale of both halves of a pair
  AlignedVector<bf16x2_t, 4> out;
  auto* o = reinterpret_cast<uint32_t*>(out.data());
  asm("{ .reg .b16 a, b, c, d, s, t; mov.b32 {a, b}, %4; mov.b32 {c, d}, %5; mov.b32 {s, t}, %6;"
      " cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, a, s; cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %1, b, s;"
      " cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %2, c, s; cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %3, d, s; }"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
      : "r"(w.x), "r"(w.y), "r"(sf));
  return out;
}
// 8 e2m1 at `src` (4 B aligned; element 2k in the low nibble of byte k) -> bf16, unscaled.
SGL_DEVICE AlignedVector<bf16x2_t, 4> e2m1x8_to_bf16x8(const void* src) {
  const uint32_t w = *static_cast<const uint32_t*>(src);
  AlignedVector<bf16x2_t, 4> out;
  auto* o = reinterpret_cast<uint32_t*>(out.data());
  asm("{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, %4;"
      " cvt.rn.bf16x2.e2m1x2 %0, b0; cvt.rn.bf16x2.e2m1x2 %1, b1;"
      " cvt.rn.bf16x2.e2m1x2 %2, b2; cvt.rn.bf16x2.e2m1x2 %3, b3; }"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
      : "r"(w));
  return out;
}
// Bytes `lo` and `hi` of the 4 e4m3 at `src` (4 B aligned) -> bf16x2.
SGL_DEVICE bf16x2_t e4m3x2_to_bf16x2(const void* src, uint32_t lo, uint32_t hi) {
  const uint32_t two_bytes = __byte_perm(*static_cast<const uint32_t*>(src), 0, lo | (hi << 4));
  bf16x2_t out;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.bf16x2.e4m3x2 %0, lo; }"
      : "=r"(*reinterpret_cast<uint32_t*>(&out))
      : "r"(two_bytes));
  return out;
}
SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE void cp_async_16B(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
// Arrives on `bar` once every cp.async this thread issued so far has landed.
SGL_DEVICE void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(to_shared(bar)) : "memory");
}
SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

}  // namespace ptx

constexpr int kHeads = 16;
constexpr int kDim = 512;
constexpr int kBT = 64;  // kv tile
constexpr int kNumThreads = 256;
constexpr int kDataBytes = 512;
constexpr int kScaleBytes = 16;
constexpr int kUnits = 32;  // output slices of [8 heads, 32 dims]
constexpr int kUnitElems = 8 * 32;
constexpr int kStatFloats = 2 * kHeads;

struct Params {
  const bf16_t* q;
  const uint8_t* kv[2];
  const int32_t* idx[2];
  const int32_t* len[2];
  int64_t q_stride_b, q_stride_h;
  int64_t idx_stride[2];
  int64_t page_stride[2];
  int32_t page_shift[2];  // page sizes are powers of two
  int64_t num_tokens[2];
  const float* sink;
  float scale_log2;
  // Inverse RoPE of the output's last 64 dims, fused: (cos, sin) pairs of row `rope_pos[b]`
  // (int32 elements at stride rope_pos_step: 2 reads an int64's low word), or of row b when
  // rope_pos_step is 0 (a per-row buffer); off when rope_freqs is null. rope_pos is always a
  // readable address, so the position load needs no branch.
  const float* rope_freqs;
  const int32_t* rope_pos;
  int32_t rope_pos_step;
  int64_t rope_stride;
  // Output: e4m3 [B, G, 4096] at element strides (fp8_stride_b, fp8_stride_g),
  // and int32 scales at element strides (sf_stride_b, sf_stride_g, sf_stride_k) for (b, g, k4)
  fp8_e4m3_t* out_fp8;
  int64_t fp8_stride_b, fp8_stride_g;
  uint32_t* out_sf;
  int64_t sf_stride_b, sf_stride_g, sf_stride_k;
};

/**
 * \brief Compile-time shape of one CTA (one 64-key tile) for a cluster of C tiles.
 *
 * \tparam C cluster size: SWA tiles + extra tiles.
 */
template <int C>
struct Trait {
  static constexpr int kSRow = kBT + 8;  // f32, padded
  static constexpr int kPRow = kBT + 8;  // bf16, padded
  // An owner holds units u with u % C == rank; the busiest holds this many.
  static constexpr int kOwnerUnits = (kUnits + C - 1) / C;
  static constexpr int kSwaTiles = 2;                // 128 SWA keys
  static constexpr int kExtraTiles = C - kSwaTiles;  // 0, or 512 indexed keys

  struct Smem {
    union {
      float s[8][16][kSRow];             // [g][head][key] S partials
      bf16_t stage[kUnits][kUnitElems];  // then the bf16 O slices, unit by unit
    };
    // [key][512] e4m3, chunk-swizzled (raw_off); an fp4 tile holds [key][256] e2m1 (raw4_off)
    // followed by [key][32] e4m3 scales.
    uint8_t raw[kBT][kDataBytes];
    uint8_t raw_scale[kBT][kScaleBytes];  // [key][16] ue8m0
    bf16_t p[16][kPRow];                  // [head][key]
    uint32_t valid[kBT];                  // 1 / 0
    float stat[kStatFloats];              // [max, sum][head]
    float wgt[16][kHeads];                // [src][head]
    float stat_recv[16][kStatFloats];     // [src][max, sum][head]
    uint64_t stat_bar, data_bar, stage_bar[4];
    uint64_t pad_[2];
    // [src][slot][256] bf16, the slot stride being this owner's unit count
    bf16_t recv[C * kOwnerUnits * kUnitElems];
  };
  static_assert(sizeof(Smem) <= 113 * 1024, "two CTAs per SM");
};

// Raw rows: 16 B chunk c of key k sits at chunk c ^ 2 (k % 4), so the 8 keys of a
// QK read (2 adjacent chunks each) spread over the 4 32-byte bank groups.
SGL_DEVICE uint32_t raw_off(int key, int chunk) {
  return key * kDataBytes + ((chunk ^ ((key & 3) << 1)) << 4);
}
// V4.1 fp4 extra rows: 256 B of e2m1, 16 B chunk c of key k at chunk c ^ (k % 8).
constexpr int kFp4DataBytes = 256;
constexpr int kFp4ScaleBytes = 32;  // e4m3, one per 16 values
SGL_DEVICE uint32_t raw4_off(int key, int chunk) {
  return key * kFp4DataBytes + ((chunk ^ (key & 7)) << 4);
}
// Unit rows are 32 bf16; bf16x2 slots are XORed with 4 ((h8 / 2) % 4) to spread the stagers' banks.
SGL_DEVICE int unit_off(int h8, int dim) {
  return h8 * 32 + ((((dim >> 1) ^ (((h8 >> 1) & 3) << 2)) << 1) | (dim & 1));
}

// Max over each half warp: two full-warp redux.sync, each with the other half feeding -inf.
// A 16-lane shuffle tree cannot use redux, and redux with a partial mask is much slower.
SGL_DEVICE float half_warp_max(float x, uint32_t lane_id) {
  const bool upper = lane_id >= 16;
  const float lo = warp::reduce_max(upper ? -INFINITY : x);
  const float hi = warp::reduce_max(upper ? x : -INFINITY);
  return upper ? hi : lo;
}

// C: the cluster size = SWA tiles + extra tiles: 2 (128 SWA keys) or 10 (+ 512 indexed keys).
// kExtraFp4: the extra segment is V4.1 fp4 (288 B/token) instead of fp8.
// kFp4Seg: this CTA's tile is from the fp4 extra cache. A compile-time split: a CTA-uniform
// runtime branch inside the unrolled gather / dequant loops cost ~0.5 us of scheduling.
template <int C, bool kFp4Seg>
SGL_DEVICE void attention_body(const Params& p) {
  using T = Trait<C>;
  // A plain byte array: every instantiation shares the one dynamic smem symbol.
  extern __shared__ __align__(128) uint8_t s_buf[];
  auto& smem = *reinterpret_cast<typename T::Smem*>(s_buf);
  const uint32_t tid = threadIdx.x;
  const auto warp_id = tid / 32, lane_id = tid % 32;
  const auto g = warp_id;
  const auto r4 = lane_id / 4, q4 = lane_id % 4;
  const auto rank = ptx::cluster_ctarank();
  const uint32_t b = blockIdx.y;

  const auto stab = &smem.s[0][0][0];
  const auto stage16 = &smem.stage[0][0];
  const auto validtab = smem.valid;
  const auto stat = smem.stat;
  const auto wgt = &smem.wgt[0][0];
  const auto stat_recv = &smem.stat_recv[0][0];
  const auto recv = smem.recv;
  const auto raw = &smem.raw[0][0];
  const auto raw_s = ptx::to_shared(raw);

  if (tid == 0) {
    ptx::mbar_init(&smem.stat_bar, 1);
    ptx::mbar_init(&smem.data_bar, 1);
#pragma unroll
    for (uint32_t st = 0; st < 4; ++st)
      ptx::mbar_init(&smem.stage_bar[st], kNumThreads);
    ptx::fence_mbarrier_init_release_cluster();
  }
  ptx::cluster_arrive_relaxed();
  device::PDLWaitPrimary<true>();
  // Release the dependents right away, so a weight-streaming successor (wo_a) overlaps us.
  device::PDLTriggerSecondary<true>();

  // ---- loads that depend on nothing: this warp's row ids, lengths, sink, Q
  const uint32_t seg = rank < T::kSwaTiles ? 0 : 1;
  // signed: compared against the (signed) lengths
  const int tile_start = int(seg == 0 ? rank : rank - T::kSwaTiles) * kBT;
  const auto idxp = p.idx[seg] + b * p.idx_stride[seg];
  // Warp g gathers the 8 consecutive keys 8g + j into tile positions g + 8j (a relabeling:
  // everything downstream indexes positions), so its ids are two 16 B loads.
  int ids[8];
  {
    AlignedVector<int32_t, 4> id_vec[2];
    id_vec[0].load(idxp + tile_start + 8 * g, 0);
    id_vec[1].load(idxp + tile_start + 8 * g, 1);
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j)
      ids[j] = id_vec[j / 4][j % 4];
  }
  const auto len0_raw = p.len[0][b];
  int len1_raw = 0;
  if constexpr (T::kExtraTiles > 0) len1_raw = p.len[1][b];
  const float sink = p.sink[tid % 16] * 1.4426950408889634f;
  const auto rope_pos = p.rope_pos[b * p.rope_pos_step];
  AlignedVector<bf16x2_t, 4> qraw[4];  // [row r4, r4 + 8][block 2g, 2g + 1]: dims 64 g + 32 blk + 8 q4 + [0, 8)
  {
    const auto qb = p.q + b * p.q_stride_b + g * 64 + 8 * q4;
    __builtin_assume(__isGlobal(qb));  // otherwise the loads below stay generic (LD, not LDG)
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
      qraw[i].load(qb + (r4 + 8 * (i / 2)) * p.q_stride_h + 32 * (i % 2));
  }

  __syncthreads();  // the stage barriers are initialized (the loads above are in flight)
  // ---- gather, via cp.async (invalid ids zero-fill): fp8 rows one warp per row, 16 B per lane;
  // fp4 rows (256 B) half a warp per row. Stage st = positions [16 st, 16 st + 16) = rows
  // 2 st, 2 st + 1 of every warp.
  const auto kv_ptr = p.kv[seg];
  const auto num_tokens = p.num_tokens[seg];
  const auto page_shift = p.page_shift[seg];
  const auto page_stride = p.page_stride[seg];
  __builtin_assume(num_tokens >= 0);  // compiler may use this to optimize 0 <= x && x < num_tokens
  if constexpr (!kFp4Seg) {
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      const auto ok = ids[j] >= 0 && ids[j] < num_tokens;
      const auto id = ok ? ids[j] : 0;
      const auto page = id >> page_shift, slot = id & ((1 << page_shift) - 1);
      const auto base = kv_ptr + int64_t(page) * page_stride;
      const auto key = g + 8 * j;
      ptx::cp_async_16B(raw_s + raw_off(key, lane_id), base + slot * kDataBytes + lane_id * 16, ok ? 16 : 0);
      if (lane_id == j)
        ptx::cp_async_16B(
            ptx::to_shared(&smem.raw_scale[0][0]) + key * kScaleBytes,
            base + (kDataBytes << page_shift) + slot * kScaleBytes,
            ok ? 16 : 0);
      if (j % 2 == 1) ptx::cp_async_mbar_arrive(&smem.stage_bar[j / 2]);
    }
  } else {
#pragma unroll
    for (uint32_t st = 0; st < 4; ++st) {
      const auto jr = 2 * st + lane_id / 16;  // this lane's row of the stage
      const auto idr = lane_id < 16 ? ids[2 * st] : ids[2 * st + 1];
      const auto ok = idr >= 0 && idr < num_tokens;
      const auto id = ok ? idr : 0;
      const auto page = id >> page_shift, slot = id & ((1 << page_shift) - 1);
      const auto base = kv_ptr + int64_t(page) * page_stride;
      const auto key = g + 8 * jr, c = lane_id % 16;
      ptx::cp_async_16B(raw_s + raw4_off(key, c), base + slot * kFp4DataBytes + c * 16, ok ? 16 : 0);
      if (c < 2)  // the row's 32 B of scales
        ptx::cp_async_16B(
            raw_s + kBT * kFp4DataBytes + key * kFp4ScaleBytes + c * 16,
            base + (kFp4DataBytes << page_shift) + slot * kFp4ScaleBytes + c * 16,
            ok ? 16 : 0);
      ptx::cp_async_mbar_arrive(&smem.stage_bar[st]);
    }
  }

  const int len0 = min(len0_raw, T::kSwaTiles * kBT);
  const int len1 = min(len1_raw, T::kExtraTiles * kBT);
  const int seg_len = seg == 0 ? len0 : len1;
  const auto active = tile_start < seg_len;
  if (!active) ptx::cp_async_wait_all();  // nothing consumes the rows, but they must land before exit
  const auto src_active = [&](int r) {
    const auto s = r < T::kSwaTiles ? 0 : 1;
    const auto start = int(s == 0 ? r : r - T::kSwaTiles) * kBT;
    return start < (s == 0 ? len0 : len1);
  };
  const auto my_units = (kUnits - rank + C - 1) / C;
  if (tid == 0) {
    // the active sources are exactly the tiles starting below their segment's length
    const int srcs = (len0 + kBT - 1) / kBT + (len1 + kBT - 1) / kBT;
    // our own contribution is written locally, not pushed
    const int remote = srcs - (active ? 1 : 0);
    ptx::mbar_arrive_expect_tx(&smem.stat_bar, uint32_t(remote * kStatFloats * 4));
    ptx::mbar_arrive_expect_tx(&smem.data_bar, uint32_t(remote * my_units * kUnitElems * 2));
  }
  if (lane_id < 8) {
    const int at = tile_start + int(8 * g + lane_id);
    // lane j wrote ids[j]'s flag; pick it without dynamic register indexing
    int id = ids[0];
#pragma unroll
    for (uint32_t j = 1; j < 8; ++j) {
      id = lane_id == j ? ids[j] : id;
    }
    validtab[g + 8 * lane_id] = (at < seg_len && id >= 0 && id < num_tokens) ? 1u : 0u;
  }

  bf16x2_t kf[8][8];
  if (active) {
    // ---- per stage: this lane's keys r4 + 8j (j = 2 st, 2 st + 1): 8 B from each 32-dim half
    // of the group, -> bf16 with the block scale -> V tile, and S_g = Q_g K_g^T.
    // k-step (blk, s): dims 32 blk + 8 q4 + 4 s + [0, 4); Q is bf16 already.
    const auto srow = stab + (g * 16) * T::kSRow;
    // kf[j][blk * 4 + word]: bf16x2 of dims 32 blk + 8 q4 + 2 word + {0, 1} of position r4 + 8j;
    // kept for PV, which transposes it in registers (no V tile)
#pragma unroll
    for (uint32_t st = 0; st < 4; ++st) {
      ptx::mbar_wait_parity(&smem.stage_bar[st], 0);
#pragma unroll
      for (uint32_t jj = 0; jj < 2; ++jj) {
        const auto j = 2 * st + jj;
        const auto key = r4 + 8 * j;
        if constexpr (!kFp4Seg) {
          AlignedVector<uint8_t, 2> sf;  // ue8m0 of dims 64 g + 32 blk + [0, 32)
          sf.load(&smem.raw_scale[key][2 * g]);
#pragma unroll
          for (uint32_t blk = 0; blk < 2; ++blk) {
            const auto v =
                ptx::e4m3x8_to_bf16x8_scaled(raw + raw_off(key, 4 * g + 2 * blk + q4 / 2) + (q4 % 2) * 8, sf[blk]);
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k)
              kf[j][blk * 4 + k] = v[k];
          }
        } else {
          // e4m3 scales of dims 64g + 32 blk + 8 q4 (one per 16): bytes 4g + q4 / 2 and + 2
          const auto s2 =
              ptx::e4m3x2_to_bf16x2(raw + kBT * kFp4DataBytes + key * kFp4ScaleBytes + 4 * g, q4 / 2, 2 + q4 / 2);
#pragma unroll
          for (uint32_t blk = 0; blk < 2; ++blk) {
            const auto sb = blk == 0 ? __low2bfloat162(s2) : __high2bfloat162(s2);  // that scale, twice
            const auto v = ptx::e2m1x8_to_bf16x8(raw + raw4_off(key, 2 * g + blk) + 4 * q4);
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k)
              kf[j][blk * 4 + k] = __hmul2(v[k], sb);
          }
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (uint32_t blk = 0; blk < 2; ++blk)
#pragma unroll
          for (uint32_t s = 0; s < 2; ++s) {
            const auto &q0 = qraw[blk], &q1 = qraw[2 + blk];  // rows r4, r4 + 8
            const bf16x2_t qa[4] = {q0[2 * s], q1[2 * s], q0[2 * s + 1], q1[2 * s + 1]};
            ptx::mma_bf16(acc, qa, kf[j][blk * 4 + 2 * s], kf[j][blk * 4 + 2 * s + 1]);
          }
        // columns 2 q4, 2 q4 + 1 are keys 8j + 2 q4 + {0, 1}
#pragma unroll
        for (uint32_t rh = 0; rh < 2; ++rh) {
          AlignedVector<fp32_t, 2> sv;
          sv[0] = acc[2 * rh], sv[1] = acc[2 * rh + 1];
          sv.store(srow + (r4 + 8 * rh) * T::kSRow + 8 * j + 2 * q4);
        }
      }
    }
  }
  __syncthreads();
  // (cos, sin) x 2 for this thread's first output slice if it is a RoPE slice (group 7);
  // loaded here, where the position has long arrived, and used at the very end
  const int64_t rope_row = p.rope_pos_step ? int64_t(rope_pos) : int64_t(b);
  AlignedVector<fp32_t, 4> rope;
  rope[0] = 1.f, rope[1] = 0.f, rope[2] = 1.f, rope[3] = 0.f;
  bool rope_on = false;
  if (p.rope_freqs != nullptr) {
    const auto slot = tid / 64, d4 = tid % 8;
    const auto u = slot * C + rank;
    rope_on = u < kUnits && u / 4 == 7 && slot < my_units;
    if (rope_on) rope.load(p.rope_freqs + rope_row * p.rope_stride + (u % 2) * 32 + 4 * d4);
  }

  if (active) {
    // ---- softmax: thread = (head, 4 keys)
    const auto h = tid / 16, key = 4 * (tid % 16);
    float sv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (uint32_t gg = 0; gg < 8; ++gg) {
      AlignedVector<fp32_t, 4> v;
      v.load(stab + (gg * 16 + h) * T::kSRow + key);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        sv[i] += v[i];
    }
    AlignedVector<uint32_t, 4> okv;
    okv.load(validtab + key);
    float mx = -INFINITY;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      sv[i] = okv[i] != 0u ? sv[i] * p.scale_log2 : -INFINITY;
      mx = fmaxf(mx, sv[i]);
    }
    mx = half_warp_max(mx, lane_id);
    if (mx == -INFINITY) mx = 0.f;
    float pr[4], sum = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      pr[i] = okv[i] != 0u ? ptx::ex2_approx(sv[i] - mx) : 0.f;
      sum += pr[i];
    }
    sum = warp::reduce_sum<16>(sum);
    if (tid % 16 == 0) {
      stat[h] = mx;
      stat[16 + h] = sum;
      stat_recv[rank * kStatFloats + h] = mx;
      stat_recv[rank * kStatFloats + 16 + h] = sum;
    }
    const auto ptab = &smem.p[0][0];
    {
      AlignedVector<bf16x2_t, 2> w;
      w[0] = cast<bf16x2_t>(fp32x2_t{pr[0], pr[1]});
      w[1] = cast<bf16x2_t>(fp32x2_t{pr[2], pr[3]});
      // P goes to the PV MMA as bf16, as in FlashMLA (no rounding-residual pass)
      w.store(ptab + h * T::kPRow + key);
    }
  }
  __syncthreads();
  // every peer initialized its barriers long ago; push the stats to all ranks
  ptx::cluster_wait_acquire();
  if (active && tid < C * 8 && tid / 8 != rank) {
    const auto dst_rank = tid / 8, part = tid % 8;
    AlignedVector<fp32_t, 4> v;
    v.load(stat, part);
    ptx::st_async_16(
        ptx::mapa(ptx::to_shared(stat_recv) + (rank * kStatFloats + part * 4) * 4, dst_rank),
        v.data(),
        ptx::mapa(ptx::to_shared(&smem.stat_bar), dst_rank));
  }

  if (active) {
    // ---- O_g = P V_g : 16 heads x 64 dims; n-tile m holds dims 8m + c
    float oacc[8][4];
#pragma unroll
    for (uint32_t n = 0; n < 8; ++n)
      oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;
#pragma unroll
    for (uint32_t kk = 0; kk < 4; ++kk) {
      bf16x2_t pa[4];
      ptx::ldmatrix_x4(&smem.p[lane_id % 16][16 * kk + (lane_id / 16) * 8], pa);
      // n-tile nt = (blk, word): column c holds dim 32 blk + 8 (c / 2) + 2 word + c % 2; the
      // transposed kf registers of positions 16kk + [0, 8) and + [8, 16) are b0 and b1
#pragma unroll
      for (uint32_t nt = 0; nt < 8; ++nt) {
        const auto b0 = ptx::movmatrix_trans(kf[2 * kk][nt]);
        const auto b1 = ptx::movmatrix_trans(kf[2 * kk + 1][nt]);
        ptx::mma_bf16(oacc[nt], pa, b0, b1);
      }
    }
    // ---- O / l -> bf16 staging, unit u = g * 4 + row half * 2 + 32-dim half: [8 heads][32 dims]
    float inv[2];
#pragma unroll
    for (uint32_t rh = 0; rh < 2; ++rh) {
      const auto l = stat[16 + r4 + 8 * rh];
      inv[rh] = l > 0.f ? __fdividef(1.f, l) : 0.f;
    }
    // n-tiles 4 half + [0, 4) are dims 8 q4 + [0, 8) of 32-dim half `half`: one 16 B slot each
    // (unit_off XORs bf16x2 slots by multiples of 4)
#pragma unroll
    for (uint32_t rh = 0; rh < 2; ++rh) {
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        const auto u = g * 4 + rh * 2 + half;
        AlignedVector<bf16x2_t, 4> o;
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) {
          const auto nt = half * 4 + w;
          o[w] = cast<bf16x2_t>(fp32x2_t{oacc[nt][2 * rh] * inv[rh], oacc[nt][2 * rh + 1] * inv[rh]});
        }
        o.store(stage16 + u * kUnitElems + unit_off(r4, 8 * q4));
      }
    }
    // ---- push this warp's 4 units right away: 16 B chunks to each owner's inbox [src][slot]
    __syncwarp();
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto u = g * 4 + i, ch = lane_id;
      const auto o = u % C, slot = u / C;
      AlignedVector<uint32_t, 4> v;
      v.load(stage16 + u * kUnitElems, ch);
      const auto o_units = (kUnits - o + C - 1) / C;
      const auto inbox = smem.recv + (rank * o_units + slot) * kUnitElems;
      if (o == rank) {
        v.store(inbox, ch);
      } else {
        ptx::st_async_16(
            ptx::mapa(ptx::to_shared(inbox) + ch * 16, o), v.data(), ptx::mapa(ptx::to_shared(&smem.data_bar), o));
      }
    }
  }
  // ---- owner: weights from the stats (usually here already), then wait for the data
  ptx::mbar_wait_parity(&smem.stat_bar, 0);
  {
    const auto h = tid / 16, r = tid % 16;  // the 16 sources of a head in 16 consecutive lanes
    const bool live = r < C && src_active(r);
    const auto m = live ? stat_recv[r * kStatFloats + h] : -INFINITY;
    const auto l = live ? stat_recv[r * kStatFloats + 16 + h] : 0.f;
    const bool has = l > 0.f;
    float M = has ? m : -INFINITY;
    M = half_warp_max(M, lane_id);
    const auto sk = warp::broadcast(sink, (lane_id & 16) | h);  // lane h % 16 holds sink[h]
    M = fmaxf(M, sk);
    if (fabsf(M) == INFINITY) M = 0.f;
    const float w = has ? ptx::ex2_approx(m - M) * l : 0.f;
    float den = w;
    den = warp::reduce_sum<16>(den);
    den += ptx::ex2_approx(sk - M);
    const bool zero = !(den > 0.f) || sk == INFINITY;
    wgt[r * kHeads + h] = zero ? 0.f : __fdividef(w, den);
  }
  ptx::mbar_wait_parity(&smem.data_bar, 0);
  __syncthreads();
  for (uint32_t j = tid; j < my_units * 64; j += kNumThreads) {
    const auto slot = j / 64, h8 = (j / 8) % 8, d4 = j % 8;
    const auto u = slot * C + rank;
    const auto head = ((u / 2) % 2) * 8 + h8;
    float4 acc = make_float4(0, 0, 0, 0);
    // branch-free over the C sources, zeroing what is absent
    // (an inactive source's inbox was never written and may hold NaNs)
    const auto rb = recv + slot * kUnitElems + unit_off(h8, 4 * d4);
#pragma unroll
    for (uint32_t r = 0; r < C; ++r) {
      const auto w = wgt[r * kHeads + head];
      AlignedVector<bf16x2_t, 2> x;
      x.load(rb + r * my_units * kUnitElems);
      const bool use = w != 0.f;
      const auto lo = cast<fp32x2_t>(x[0]), hi = cast<fp32x2_t>(x[1]);
      acc.x = use ? fmaf(lo.x, w, acc.x) : acc.x;
      acc.y = use ? fmaf(lo.y, w, acc.y) : acc.y;
      acc.z = use ? fmaf(hi.x, w, acc.z) : acc.z;
      acc.w = use ? fmaf(hi.y, w, acc.w) : acc.w;
    }
    const auto dim = (u / 4) * 64 + (u % 2) * 32 + 4 * d4;
    auto lo = cast<bf16x2_t>(fp32x2_t{acc.x, acc.y});
    auto hi = cast<bf16x2_t>(fp32x2_t{acc.z, acc.w});
    if (p.rope_freqs != nullptr && u / 4 == 7) {
      // inverse rotation of the rounded output, as the standalone RoPE kernel sees it:
      // (re + i im) (c - i s), pairs (dim, dim + 1) and (dim + 2, dim + 3)
      auto f = rope;
      if (j != tid || !rope_on) f.load(p.rope_freqs + rope_row * p.rope_stride + (u % 2) * 32 + 4 * d4);
      const auto a = cast<fp32x2_t>(lo), c = cast<fp32x2_t>(hi);
      lo = cast<bf16x2_t>(fp32x2_t{a.x * f[0] + a.y * f[1], a.y * f[0] - a.x * f[1]});
      hi = cast<bf16x2_t>(fp32x2_t{c.x * f[2] + c.y * f[3], c.y * f[2] - c.x * f[3]});
    }
    {
      // The 8 lanes d4 = 0..7 (lane % 8) hold the 32 dims of one (head, 32-dim chunk) block.
      const auto a = cast<fp32x2_t>(lo), c = cast<fp32x2_t>(hi);
      float am = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(c.x), fabsf(c.y)));
      am = warp::reduce_max<8>(am);
      // FlashMLA fused_norm_rope_attn_rope_cast epilogue: clamp 1e-4, sf = amax * fp32(1/448),
      // e = ceil(log2 sf) from the bits, codes = rn.satfinite e4m3(x * 2^-e), byte e + 127
      am = fmaxf(am, 1e-4f);
      const auto sf = am * (1.0f / 448.0f);
      const uint32_t exp_sf = uint32_t(int32_t((__float_as_uint(sf) - 1u) >> 23) + (1 - 127));
      const auto mul = __uint_as_float((127u - exp_sf) << 23);
      const auto grp = head / 8, chunk = dim / 32;
      // satfinite rn, as FlashMLA's cvt.rn.satfinite.e4m3x2
      AlignedVector<fp8x4_e4m3_t, 1> codes;
      codes[0] = cast<fp8x4_e4m3_t>(fp32x4_t{a.x * mul, a.y * mul, c.x * mul, c.y * mul});
      codes.store(p.out_fp8 + int64_t(b) * p.fp8_stride_b + grp * p.fp8_stride_g + chunk * 256 + h8 * 32 + 4 * d4);
      // A warp is 4 heads h8 % 4 of one unit: scale bytes 4 k4 + h8 % 4, k4 = 2 chunk + h8 / 4.
      uint32_t word = ((exp_sf + 127u) & 0xFFu) << (8 * (lane_id / 8));
      word = warp::reduce_sum<warp::kFullWidth, 8>(word);  // disjoint bytes: the sum is an OR
      if (lane_id == 0)
        p.out_sf[int64_t(b) * p.sf_stride_b + grp * p.sf_stride_g + (2 * chunk + h8 / 4) * p.sf_stride_k] = word;
    }
  }
}

template <int C, bool kExtraFp4>
__global__ __launch_bounds__(kNumThreads, 2) void decode_attention_kernel(const __grid_constant__ Params p) {
  if constexpr (kExtraFp4) {
    if (ptx::cluster_ctarank() >= Trait<C>::kSwaTiles) {
      return attention_body<C, true>(p);
    }
  }
  attention_body<C, false>(p);
}

}  // namespace sglang::device::dsv4_decode

namespace sglang {

template <int C, bool kExtraFp4>
inline void launch_dsv4_decode_attention(DLDevice device, int64_t num_rows, const device::dsv4_decode::Params& p) {
  using namespace host;
  constexpr auto kernel = device::dsv4_decode::decode_attention_kernel<C, kExtraFp4>;
  constexpr size_t kSmemBytes = sizeof(typename device::dsv4_decode::Trait<C>::Smem);
  runtime::init_per_device(device, [] {
    const auto kernel_ptr = reinterpret_cast<const void*>(kernel);
    CHECK_CUDA(::cudaFuncSetAttribute(kernel_ptr, ::cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CHECK_CUDA(::cudaFuncSetAttribute(kernel_ptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    // A PDL successor (wo_a streaming its weights early) can share our SMs only under the same carveout.
    runtime::set_smem_carveout(kernel_ptr, cudaSharedmemCarveoutMaxShared);
  });
  LaunchKernel(dim3(C, num_rows), device::dsv4_decode::kNumThreads, device, kSmemBytes)
      .config({.use_pdl = true, .cluster_dim = dim3{uint32_t(C), 1, 1}})(kernel, p);
}

/**
 * \brief DSv4.1 sparse decode attention for small batches, fp8 (mega) output.
 *
 * \param q             [B, 16, 512] bf16, unit inner stride.
 * \param kv            paged V4.1 fp8 SWA cache [pages, page_size, ...] uint8, 528 B per token:
 *                      a page holds page_size rows of 512 e4m3, then page_size rows of 16 ue8m0.
 * \param indices       [B, 128] int32 SWA token ids (-1: none).
 * \param lengths       [B] int32 valid SWA keys per row.
 * \param sink          [16] f32 attention sink logits.
 * \param out_fp8       [B, 2, 4096] e4m3, unit inner stride (see the file header for the order).
 * \param out_sf        [B, 2, 32] int32 packed UE8M0 scales, any strides.
 * \param softmax_scale softmax scale.
 * \param extra_kv      paged extra cache (c1 / c2 layers), V4.1 fp8 (528 B) or fp4 (288 B) per token;
 *                      none: SWA only.
 * \param extra_indices [B, 512] int32 extra token ids; given exactly when `extra_kv` is.
 * \param extra_lengths [B] int32 valid extra keys per row; given exactly when `extra_kv` is.
 * \param rope_freqs    (cos, sin) pairs [rows, >= 64] f32 for the fused inverse RoPE of each head's
 *                      last 64 dims; none: no inverse RoPE.
 * \param rope_pos      [B] int32 / int64 row of `rope_freqs` per query row; none: row b.
 */
inline void dsv4_decode_attention_sm100(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView lengths,
    tvm::ffi::TensorView sink,
    tvm::ffi::TensorView out_fp8,
    tvm::ffi::TensorView out_sf,
    float softmax_scale,
    tvm::ffi::Optional<tvm::ffi::TensorView> extra_kv,
    tvm::ffi::Optional<tvm::ffi::TensorView> extra_indices,
    tvm::ffi::Optional<tvm::ffi::TensorView> extra_lengths,
    tvm::ffi::Optional<tvm::ffi::TensorView> rope_freqs,
    tvm::ffi::Optional<tvm::ffi::TensorView> rope_pos) {
  using namespace host;
  using namespace device::dsv4_decode;
  constexpr int64_t kFp8TokenBytes = kDataBytes + kScaleBytes;
  constexpr int64_t kFp4TokenBytes = kFp4DataBytes + kFp4ScaleBytes;
  auto B = SymbolicSize{"num_rows"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({B, kHeads, kDim})  //
      .with_strides({-1, -1, 1})
      .with_dtype<bf16_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(q);
  TensorMatcher({B, 128})  //
      .with_strides({-1, 1})
      .with_dtype<int32_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(indices);
  TensorMatcher({B})  //
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(lengths);
  TensorMatcher({kHeads})  //
      .with_dtype<fp32_t>()
      .with_device(device_)
      .verify(sink);
  TensorMatcher({B, kHeads / 8, 8 * kDim})  //
      .with_strides({-1, -1, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(4)
      .verify(out_fp8);
  // FlashMLA's strided layout, [G][32][ceil4(T)]: any strides
  TensorMatcher({B, kHeads / 8, 8 * kDim / 128})  //
      .with_strides({-1, -1, -1})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(out_sf);

  const bool has_extra = extra_kv.has_value();
  CHECK_HOST(extra_indices.has_value() == has_extra && extra_lengths.has_value() == has_extra)
      << has_extra << " " << extra_indices.has_value() << " " << extra_lengths.has_value();
  if (has_extra) {
    TensorMatcher({B, 512})  //
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(extra_indices.value());
    TensorMatcher({B})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(extra_lengths.value());
  }

  Params p{};
  p.q = static_cast<const bf16_t*>(q.data_ptr());
  p.q_stride_b = q.stride(0);
  p.q_stride_h = q.stride(1);
  // Without an extra cache, segment 1 repeats segment 0; the C = 2 kernel never reads it.
  const auto set_segment = [&](int s, tvm::ffi::TensorView cache, tvm::ffi::TensorView ids, tvm::ffi::TensorView lens) {
    TensorMatcher({-1, -1, -1})  // [pages, page_size, bytes per token]
        .with_strides({-1, -1, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(cache);
    const auto bytes_per_token = cache.size(2);
    CHECK_HOST(bytes_per_token == kFp8TokenBytes || (s == 1 && bytes_per_token == kFp4TokenBytes))
        << s << " " << bytes_per_token;
    const auto page_size = static_cast<uint64_t>(cache.size(1));
    CHECK_HOST(is_pow2(page_size)) << page_size;
    CHECK_HOST(cache.stride(0) >= int64_t(page_size) * bytes_per_token) << cache.stride(0) << " " << page_size;
    p.kv[s] = static_cast<const uint8_t*>(cache.data_ptr());
    p.idx[s] = static_cast<const int32_t*>(ids.data_ptr());
    p.len[s] = static_cast<const int32_t*>(lens.data_ptr());
    p.idx_stride[s] = ids.stride(0);
    p.page_stride[s] = cache.stride(0);
    p.page_shift[s] = log2_floor(page_size);
    p.num_tokens[s] = cache.size(0) * int64_t(page_size);
  };
  set_segment(0, kv, indices, lengths);
  if (has_extra) {
    set_segment(1, extra_kv.value(), extra_indices.value(), extra_lengths.value());
  } else {
    set_segment(1, kv, indices, lengths);
  }
  p.sink = static_cast<const float*>(sink.data_ptr());
  p.scale_log2 = softmax_scale * 1.4426950408889634f;

  // Without positions, rope_pos points at any readable int32 with step 0, so the load needs no branch.
  const auto num_rows = B.unwrap();
  p.rope_freqs = nullptr;
  p.rope_pos = p.len[0];
  p.rope_pos_step = 0;
  p.rope_stride = 0;
  if (rope_freqs.has_value()) {
    const auto freqs = rope_freqs.value();
    TensorMatcher({-1, -1})  //
        .with_strides({-1, 1})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(freqs);
    CHECK_HOST(freqs.size(1) >= 64) << freqs.size(1);
    p.rope_freqs = static_cast<const float*>(freqs.data_ptr());
    p.rope_stride = freqs.stride(0);
    if (rope_pos.has_value()) {
      const auto pos = rope_pos.value();
      TensorMatcher({B})  //
          .with_dtype<int32_t, int64_t>()
          .with_device(device_)
          .verify(pos);
      p.rope_pos = static_cast<const int32_t*>(pos.data_ptr());
      p.rope_pos_step = is_type<int64_t>(pos.dtype()) ? 2 : 1;  // an int64 position is read by its low word
    } else {
      CHECK_HOST(freqs.size(0) == num_rows) << freqs.size(0) << " " << num_rows;
    }
  } else {
    CHECK_HOST(!rope_pos.has_value());
  }
  p.out_fp8 = static_cast<fp8_e4m3_t*>(out_fp8.data_ptr());
  p.fp8_stride_b = out_fp8.stride(0);
  p.fp8_stride_g = out_fp8.stride(1);
  p.out_sf = static_cast<uint32_t*>(out_sf.data_ptr());
  p.sf_stride_b = out_sf.stride(0);
  p.sf_stride_g = out_sf.stride(1);
  p.sf_stride_k = out_sf.stride(2);

  // DSv4.1 decode: 128 SWA keys (2 tiles), plus 512 indexed keys (8 tiles) on c1 / c2 layers.
  const auto device = device_.unwrap();
  if (num_rows == 0) return;  // an empty grid is a launch error
  if (!has_extra) {
    launch_dsv4_decode_attention<2, false>(device, num_rows, p);
  } else if (extra_kv.value().size(2) == kFp4TokenBytes) {
    launch_dsv4_decode_attention<10, true>(device, num_rows, p);
  } else {
    launch_dsv4_decode_attention<10, false>(device, num_rows, p);
  }
}

}  // namespace sglang
