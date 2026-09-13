// K3 fused o_proj GEMM + all-reduce for decode (bf16, TP row-parallel).
//
// CONTRACT (per rank r of R):  out[M, 7168] = sum_r x_r[M, K] @ W_r[7168, K]^T
//   bf16 in/out, fp32 accumulate, partials round to bf16 pre-sum (same
//   semantics as the unfused GEMM + bf16 ring AR).
// M in [1, 512], rounded up to a tuned cell {8,16,32,64,128,256,512}; out
// must have `cell` rows — rows [M, cell) are clobbered with zeros.
// Comm plane: pure NVLink P2P (unicast pushes + per-rank flag reductions);
// one-shot AR below the two-shot threshold, two-shot RS+AG above.
// Requires SM100+ with full P2P; tuned on GB300 (sm_103a).

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/cuda_host_adapter.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Local PTX / TMA primitives
// Only what this kernel issues, kept in-file on purpose: these are raw-ISA
// shapes (sm_100+ tcgen05, the cta_group::1 multicast TMA load) that no shared
// sglang header wraps, and splitting them out bought a dozen headers with
// exactly one consumer. Anything cute already provides goes through cute
// (`set_block_rank` below, the tensor-map driver wrapper in `w_maps`).

namespace sglang {

namespace device::ptx {

// ---- generic → shared address conversion (PTX ISA §10.4) --------------------

// ---- cvt: pack 2 fp32 into one bf16x2 (PTX ISA §9.7.9.21) ------------------
//
// PACKED-PAIR ORDERING: `cvt.bf16x2.f32 d, a, b` puts cvt(a) in d's UPPER
// half and cvt(b) in the LOWER half. Read back from little-endian memory as a
// bf16 array, LOWER lands at column i and UPPER at i+1 — so for cells (c0, c1)
// destined for adjacent slots [i, i+1] pass `cvt_pack_f32x2_to<bf16>(c1, c0)`.
struct bf16 {
  using packed2_t = uint32_t;
};

template <typename Dst>
static __device__ __forceinline__ typename Dst::packed2_t cvt_pack_f32x2_to(float a, float b);

template <>
__device__ __forceinline__ uint32_t cvt_pack_f32x2_to<bf16>(float a, float b) {
  uint32_t d;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
  return d;
}

// ---- ldmatrix (PTX ISA §9.7.14.5.15) ---------------------------------------

// Warp-collective load of 4 (8x8) BF16 matrices from smem into mma.sync
// fragments. `row_addr` is this lane's 16-byte-aligned row base: lanes 0-7
// supply matrix 0, 8-15 matrix 1, 16-23 matrix 2, 24-31 matrix 3.
static SGL_DEVICE void ldmatrix_x4_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(row_addr));
}

static SGL_DEVICE void ldmatrix_x2_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(row_addr));
}

// ---- warp-level mma.sync (PTX ISA §9.7.14) ---------------------------------

// D += A*B, bf16 x bf16 -> f32. The warp-register form: co-resides freely (no
// TMEM, no 1-CTA/SM cap), which is why the small-M members use it instead of
// tcgen05.
static __device__ __forceinline__ void
mma_m16n8k16_bf16f32(float4& d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---- tcgen05 matrix descriptors (PTX ISA §9.7.16.4) ------------------------
//
// 0 = K-major (innermost dim is K, the "TN" convention), 1 = MN-major.
enum class Major : uint8_t {
  K = 0,
  MN = 1,
};

enum class F16Type : uint8_t { F16 = 0, BF16 = 1 };  // kind::f16 atype/btype
enum class DType : uint8_t { F16 = 0, F32 = 1, S32 = 2 };

// Smem matrix descriptor. The public PTX spec has errors at bits 46-60; the
// layout below is the one our gate tests verify:
//   0-13:  start_address >> 4      16-29: leading_byte_offset >> 4
//   32-45: stride_byte_offset >> 4 46-47: version (=1, Blackwell — larger
//          shapes such as 128B-swizzle BLOCK_K>16 produce garbage at 0)
//   49-51: base_offset             52: lbo_mode (=0, relative byte offset)
//   61-63: layout_type (0=None, 2=128B, 4=64B, 6=32B)
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc(uint32_t matrix_addr, uint32_t lbo, uint32_t sbo, uint32_t base_offset, int swizzle_bytes) {
  auto enc = [](uint32_t x) -> uint64_t { return (uint64_t)((x & 0x3FFFFu) >> 4); };
  uint8_t code = (swizzle_bytes == 128) ? 2u : (swizzle_bytes == 64) ? 4u : (swizzle_bytes == 32) ? 6u : 0u;
  uint64_t d = 0;
  d |= enc(matrix_addr);                    // bits  0-13
  d |= enc(lbo) << 16;                      // bits 16-29
  d |= enc(sbo) << 32;                      // bits 32-45
  d |= uint64_t(1u) << 46;                  // bits 46-47 = version = 1
  d |= uint64_t(base_offset & 0x7u) << 49;  // bits 49-51
  d |= uint64_t(code & 0x7u) << 61;         // bits 61-63
  return d;
}

// K-major operand (A as (M, K), B as (N, K) in a TN GEMM). `T` is a size proxy
// (uint16_t for BF16/FP16); the dtype semantics live in the instruction
// descriptor. K-major mandates SWIZZLE_BYTES == BLOCK_K * sizeof(T).
template <typename T, int BLOCK_K, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc_k_major(uint32_t addr, uint32_t base_offset = 0) {
  constexpr int K_BYTES = BLOCK_K * int(sizeof(T));
  static_assert(SWIZZLE_BYTES == K_BYTES, "K-major requires swizzle bytes == BLOCK_K * sizeof(T)");
  return mma_smem_desc(addr, /*lbo=*/0u, /*sbo=*/8u * uint32_t(K_BYTES), base_offset, SWIZZLE_BYTES);
}

// Instruction descriptor, kind::f16 (PTX ISA Table 44).
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_f16(
    uint32_t M,
    uint32_t N,
    F16Type a_type = F16Type::BF16,
    F16Type b_type = F16Type::BF16,
    DType d_type = DType::F32,
    Major a_major = Major::K,
    Major b_major = Major::K,
    bool negate_a = false,
    bool negate_b = false) {
  uint32_t d = 0;
  d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
  d |= (static_cast<uint32_t>(a_type) & 0x7u) << 7;
  d |= (static_cast<uint32_t>(b_type) & 0x7u) << 10;
  if (negate_a) d |= 1u << 13;
  if (negate_b) d |= 1u << 14;
  d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
  d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
  d |= ((N >> 3) & 0x3Fu) << 17;
  d |= ((M >> 4) & 0x1Fu) << 24;
  return d;
}

// Cross-CTA arrive with `.release.cta` ordering. A plain cluster arrive carries
// NO release, so prior memory ops (notably a retired `tcgen05.ld` TMEM drain on
// the arriving warp) are not guaranteed visible before a peer warp's `.acquire`
// wait returns — and the peer then overwriting that TMEM is a real race (spec
// §8.8: a release pattern requires `mbarrier.arrive.release`).
//
// `cute::set_block_rank` is the `mapa.shared::cluster` that retargets a local
// smem offset at CTA `cta_rank`; the `.shared::cluster` qualifier on the arrive
// is what makes it cross-CTA (the plain `.shared` form hits the local mbar
// whatever the address bits say).
static SGL_DEVICE void mbar_arrive_cluster_release(uint64_t* bar, uint32_t cta_rank) {
  const uint32_t mapped = cute::set_block_rank(to_shared(bar), cta_rank);
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;" ::"r"(mapped));
}

// ---- cluster / warp sync (PTX ISA §9.7.13, §9.7.4) -------------------------

// Cluster barrier with explicit release/acquire — the publish/observe boundary
// between `mbarrier.init` and any `.shared::cluster` use of those mbars.
static SGL_DEVICE void cluster_sync_rel_acq() {
  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");
}

// True on exactly one lane of the issuing warp — guards single-issuer sites
// (mbar init, TMA issue, MMA issue, TMEM alloc) without gating on lane_id.
static SGL_DEVICE bool elect_one() {
  uint32_t pred;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "elect.sync _|p, 0xffffffff;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}\n"
      : "=r"(pred));
  return pred != 0;
}

static SGL_DEVICE uint32_t cluster_cta_rank() {
  uint32_t rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

// ---- tcgen05 (PTX ISA §9.7.16) ---------------------------------------------
//
// Lifecycle (mandatory order, §9.7.16.7.1): alloc (one warp, n_cols a power of
// 2 in [32, 512], TMEM address written to smem) -> __syncthreads + read taddr
// -> ld/mma -> dealloc -> relinquish before kernel exit.
//
// Each warp can only touch its own 32-lane TMEM band (§9.7.16.8.1): warp 0 ->
// lanes 0-31, warp 1 -> 32-63, and so on.
//
// After an MMA, `tcgen05_commit_arrive` + `mbar_wait_parity` +
// `tcgen05_fence_after_thread_sync` before reading the result with
// `tcgen05_ld_*`; the fence is mandatory or the register reads may see stale
// values even though the mbarrier signaled "MMA done".
static SGL_DEVICE void tcgen05_alloc(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_addr_for_taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

// .32x32b.x8: 8 b32 per lane = 8 TMEM columns. Per-lane 8 FP32 -> 4 bf16x2
// packs = one int4, the natural fit for a BF16 epilogue draining a column band
// with 16-byte smem stores.
static SGL_DEVICE void tcgen05_ld_32x32b_x8(
    uint32_t taddr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    uint32_t& r4,
    uint32_t& r5,
    uint32_t& r6,
    uint32_t& r7) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
      : "r"(taddr));
}

static SGL_DEVICE void tcgen05_ld_32x32b_x8(uint32_t taddr, uint32_t* dst) {
  tcgen05_ld_32x32b_x8(taddr, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
}

// Blocks until this thread's prior TMEM->reg drains retired. The "memory"
// clobber is load-bearing: ptxas lowers the wait to per-load scoreboard waits
// on the dependent register consumers, so an `mbarrier.arrive` that does not
// read the drained registers gets HOISTED above the last drain (observed in
// SASS) and the next tile's MMA reuses TMEM still being read.
static SGL_DEVICE void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Commit prior tcgen05.mma to an mbarrier. Spec accepts `.shared::cluster` or
// no state space, NOT `.shared::cta`.
static SGL_DEVICE void tcgen05_commit_arrive(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(to_shared(bar)));
}

static SGL_DEVICE void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// kind::f16 MMA, cta_group::1. Valid dense shapes: M in {64, 128}, N in
// {8, 16, ..., 256} step 8, K = 16. Off-table shapes are NOT rejected by
// ptxas — they hit cudaErrorIllegalInstruction at runtime.
static SGL_DEVICE void
tcgen05_mma_f16(uint32_t d, uint64_t desc_a, uint64_t desc_b, uint32_t inst_desc_high, uint32_t scale_c) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
      "l"(desc_a),
      "l"(desc_b),
      "r"(inst_desc_high),
      "r"(scale_c));
}

// ---- TMA (PTX ISA §9.7.9.25) -----------------------------------------------
//
// COORDINATE CONVENTION: the tensor map's globalDim is (inner, outer) — dim 0
// is the stride-1 axis. The load calls take (x = inner offset, y = outer
// offset). Mismatch and you load the transposed tile, often with right-looking
// magnitudes but scrambled per-cell pairing.
//
// Completion is mbarrier-based for loads: arm with
// `mbar_arrive_expect_tx(bar, BYTES)` before issuing, then `mbar_wait_parity`.

// Warm the cache line holding the tensor-map descriptor so the first load of
// the persistent loop does not pay the descriptor fetch. `tmap` is a generic
// address into the __grid_constant__ CUtensorMap param; generic addressing
// resolves it to .param (§9.7.9.15).
static SGL_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

// global -> shared::cta 2D tile load.
static SGL_DEVICE void
cp_async_bulk_tensor_2d_load(uint32_t dst_smem, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}

// MULTICAST load, cta_group::1: one leader CTA issues, and every CTA whose bit
// is set in `multicast_mask` receives the same bytes at the same CTA-relative
// smem offset AND its OWN tx-count decrement on its own local mbar at
// `bar`'s offset (spec §9.7.9.25, .cta_group::1 bullet).
//
// Contrast with the cta_group::2 form, which CONSOLIDATES all completion onto
// one CTA's mbar (bit 24 of the mbar address cleared). Here each peer keeps its
// own `expect_tx` accounting and only the leader issues, so the follower's smem
// and mbar are both served by the leader's single DRAM read — do NOT clear bit
// 24, that is the cta_group::2 trick and would mis-route the signal.
static SGL_DEVICE void cp_async_bulk_tensor_2d_load_multicast_cg1(
    uint32_t dst_smem,
    const CUtensorMap* tmap,
    int32_t x,
    int32_t y,
    uint64_t* bar,
    uint16_t multicast_mask = 0b11,
    uint64_t cache_hint = 0x0ULL) {
  const uint32_t mbar_addr = to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::1.shared::cluster.global"
      ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5}], [%2], %3, %6;" ::"r"(dst_smem),
      "l"(tmap),
      "r"(mbar_addr),
      "h"(multicast_mask),
      "r"(x),
      "r"(y),
      "l"(cache_hint)
      : "memory");
}

}  // namespace device::ptx

namespace swz {

// CPU-side 128B-swizzle math. The TMA hardware permutes the (row x col) atom
// layout when it stores a tile into smem, so reading a known cell back applies
// the same permutation: row stride = 128 B = 8 atoms = 64 BF16 cols, and
// smem_atom = logical_atom XOR (r & 7), an 8-row period (PTX ISA §5.5.7,
// Figures 23-37). Verified on B300 (sm_103a) by a load + read-back roundtrip.
//
// Returns the smem column index in BF16 units within the row.
__host__ __device__ inline uint32_t smem_col_128b_bf16(uint32_t r, uint32_t c) {
  return c ^ ((r & 7u) << 3);  // (r & 7) atoms shifted; atom = 8 BF16 cols
}

}  // namespace swz

namespace tmap {

// Thin wrapper over the tensor-map encoder. Coordinate convention: dim 0 is
// the innermost (stride-1) axis, so globalDim = {cols, rows} and globalStrides
// (length rank-1) carries the row stride in BYTES.
//
// K-major MMA feeds require swizzle == BLOCK_K bytes; the smem descriptor
// assumes equality and a smaller inner box is encoder-legal but silently
// corrupts the MMA load.
//
// The encode is the one driver-API call in this file (there is no runtime-API
// tensor-map encoder), so it goes through cutlass's dlopen-based driver wrapper
// like the other JIT kernels that encode tensor maps — that keeps the module
// off `-lcuda`.
inline CUtensorMap encode_tiled_2d(
    void* global_ptr,
    CUtensorMapDataType dtype,
    uint64_t global_rows,
    uint64_t global_cols,
    uint64_t row_stride_bytes,
    uint32_t box_rows,
    uint32_t box_cols,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion promo = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
  cuuint64_t global_dim[2] = {global_cols, global_rows};
  cuuint64_t global_strides[1] = {row_stride_bytes};
  cuuint32_t box_dim[2] = {box_cols, box_rows};
  cuuint32_t element_strides[2] = {1, 1};

  CUtensorMap m{};
  const CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &m,
      dtype,
      /*rank=*/2,
      global_ptr,
      global_dim,
      global_strides,
      box_dim,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      promo,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (res != CUDA_SUCCESS) {
    std::fprintf(
        stderr,
        "gemm_ar: cuTensorMapEncodeTiled failed (%d) at %s:%d — rows=%llu cols=%llu "
        "row_stride=%llu box=%ux%u swizzle=%d\n",
        int(res),
        __FILE__,
        __LINE__,
        (unsigned long long)global_rows,
        (unsigned long long)global_cols,
        (unsigned long long)row_stride_bytes,
        box_rows,
        box_cols,
        int(swizzle));
    std::abort();
  }
  return m;
}

}  // namespace tmap

namespace dense_gemm_mainloop {

// GROUP_N L2-stripe raster: linear tile index -> (bid_m, bid_n), walking the
// output grid in N-stripes of width GROUP_N so the same chunk of B stays
// resident in L2 across all M-rows of the stripe (DeepGEMM's
// get_swizzled_block_idx). `cluster_grid_m` is the M-axis tile count in CLUSTER
// units (= grid_m / CTA_GROUP); `grid_n` is NOT halved by CTA_GROUP. The tail
// stripe is shorter than GROUP_N when grid_n % GROUP_N != 0.
template <int CTA_GROUP, int GROUP_N>
__device__ __forceinline__ int2 group_n_swizzle(int linear, int crank, int cluster_grid_m, int grid_n) {
  if constexpr (CTA_GROUP == 1) {
    const int num_blocks_per_group = cluster_grid_m * GROUP_N;
    const int group_idx = linear / num_blocks_per_group;
    const int first_n = group_idx * GROUP_N;
    const int in_group = linear - group_idx * num_blocks_per_group;
    const int num_n_in_group = grid_n - first_n < GROUP_N ? grid_n - first_n : GROUP_N;
    const int bid_m = in_group / num_n_in_group;
    const int bid_n = first_n + (in_group % num_n_in_group);
    (void)crank;
    return {bid_m, bid_n};
  } else {
    const int c = linear / CTA_GROUP;
    const int r = linear & (CTA_GROUP - 1);
    const int num_blocks_per_group = cluster_grid_m * GROUP_N;
    const int group_idx = c / num_blocks_per_group;
    const int first_n = group_idx * GROUP_N;
    const int in_group = c - group_idx * num_blocks_per_group;
    const int num_n_in_group = grid_n - first_n < GROUP_N ? grid_n - first_n : GROUP_N;
    const int cluster_bid_m = in_group / num_n_in_group;
    const int cluster_bid_n = first_n + (in_group % num_n_in_group);
    const int bid_m = cluster_bid_m * CTA_GROUP + r;
    const int bid_n = cluster_bid_n;
    (void)crank;  // `r` already encodes the intra-cluster lane.
    return {bid_m, bid_n};
  }
}

}  // namespace dense_gemm_mainloop

namespace oproj_ar {

// ---------------------------------------------------------------- constants
#ifndef OPROJ_N       // output dim (columns of W / of out). The
#define OPROJ_N 7168  // default is the Kimi-K3 o_proj shape; other
#endif                // shapes compile via -DOPROJ_N (see asserts).
constexpr int kN = OPROJ_N;
static_assert(
    kN % 256 == 0 && kN >= 256,
    "N must be a multiple of 256 (member tile table: 128-row "
    "strips + BN up to 256); relaxing this needs a BN-table edit");
constexpr int kBK = 64;                       // K per stage (128 B rows → swizzle-128B)
constexpr int kBNRows = 48;                   // B-box rows per stage (6 n8-tiles)
constexpr int kTilesMax = 6;                  // n8-tiles per CTA (5-tile CTAs pad, never push)
constexpr int kCWarps = 6;                    // consumer warp w owns n8-tile w
constexpr int kThreads = (kCWarps + 1) * 32;  // +1 dedicated TMA producer warp
constexpr int kMMax = 512;                    // bs64 — sizes the shared slot layout
constexpr int kRing = 64;                     // epoch flag/gather ring (monotonic values)

enum class Comm { kNone, kMc, kPeer, kMcPull, kTwoShot, kTwoShotPeer };

// Shared-region layout (BYTES, M-independent: sized at kMMax so every arm and
// every bs cell reuses one region). Parity-double-buffered payloads; the flag
// ring lives on its own 2 MB page. Slot reuse across epochs e / e+2 is safe
// with 2 parities because each rank's launch e+1 spin-waits epoch e+1 AFTER
// its own e-reduce (per-rank stream order).
// Slots are TILE-MAJOR ([n8-tile][m][8 cols]), NOT [m][n]: a warp's push for
// one tile is then a CONTIGUOUS 128 B fabric write instead of 32 scattered
// 4-16 B m-strided writes — lane-scatter runs ~0.26x on this fabric
// and the scattered form's ack-drain dominated the bs8
// boundary (stamped 22 us at idle). The reduce un-transposes locally.
constexpr size_t kSlotBytes1 = size_t(kMMax) * kN * 2;  // one [M,N] bf16
constexpr __host__ __device__ size_t slot_off(int parity, int src, int R) {
  return (size_t(parity) * R + src) * kSlotBytes1;
}
constexpr __host__ __device__ size_t pull_off(int parity, int R) {  // [2][M,N] above slots
  return (2 * size_t(R) + parity) * kSlotBytes1;
}
constexpr __host__ __device__ size_t flags_off(int R) {
  const size_t end = (2 * size_t(R) + 2) * kSlotBytes1;
  return (end + (size_t(2) << 20)) & ~((size_t(2) << 20) - 1);
}
// second flag family: "epoch e's slots fully REDUCED", at PER-CTA granularity.
// PDL's wait pairs with the prior grid's TRIGGER (not completion), and with
// 2-CTA/SM residency a fast rank's e+2 pushes can overwrite a slot replica a
// straggling rank's e-reduce still reads — so epoch e's push phase guards on
// done[e-2]. Per-CTA flags (the overwriter of tile t IS every rank's CTA-t)
// keep the publish fully parallel: no second grid-wide gather chain.
constexpr int kMaxCTA = 256;
constexpr int kFams = 7;  // flag/gather/done ring FAMILY per dispatch
                          // CELL {8,16,32,64,128,256,512}. Monotonic ring
                          // targets assume every epoch of a ring came
                          // from the same gridDim (and only two-shot
                          // bumps the boundary-2 ring words), so one fam
                          // per cell removes the host-side ring reset a
                          // cell change needed under the original 3-fam
                          // grid-class split — the reset was a collective
                          // and blocked CUDA-graph capture.
constexpr __host__ __device__ size_t done_off(int R) {
  return flags_off(R) + size_t(kFams) * 512;
}
constexpr __host__ __device__ size_t region_bytes(int R) {
  return done_off(R) + size_t(kFams) * kRing * kMaxCTA * 4 + 512;
}

// PDL (programmatic dependent launch): the NEXT launch on the stream may
// start its feed while THIS grid sits in the boundary spin + reduce; its
// epilogue then griddepcontrol.wait's until this grid fully completes, so
// slots/flags/out stay race-free. Serving overlaps the next layer's kernels
// the same way; an unfused cublas composite cannot cooperate across the
// vendor-kernel boundary.
SGL_DEVICE uint32_t bf2_u32(float2 f) {
  const __nv_bfloat162 p = __float22bfloat162_rn(f);
  return *reinterpret_cast<const uint32_t*>(&p);
}

// ------------------------------------------------------------------ params

template <int R>
struct Params {
  uint8_t* mc_base;                 // MC VA (null when no MC object — kPeer runs)
  uint8_t* uc_base[R];              // per-rank unicast VAs of the shared region
  uint32_t* gather;                 // device-local u32[kRing]
  __nv_bfloat16* out;               // [M,N] local output
  const __nv_bfloat16* partial_in;  // GEMM_ON=false input [M,N]
  // Device-resident per-fam CTA ticket counters. Every CTA takes one ticket
  // at entry and divides by the (family-stable) gridDim to recover the
  // launch epoch. All CTAs have taken their tickets before this grid
  // triggers a PDL successor, so successive launches receive disjoint,
  // contiguous ticket ranges without a separate bump kernel. Device state
  // instead of a launch arg keeps CUDA-graph replays advancing the epoch.
  uint32_t* epoch_base;
  int my_rank;
  int fam;  // ring family (dispatch cell) — see kFams
};

// -------------------------------------------------------------- CTA strips
// kN/8 = 896 n8-tiles over gridDim CTAs: first `rem` CTAs own base+1 tiles.
struct Strip {
  int t0, nt;
  static SGL_DEVICE Strip make(int cta, int ncta) {
    const int kT = kN / 8;
    const int base = kT / ncta, rem = kT % ncta;
    Strip s;
    if (cta < rem) {
      s.nt = base + 1;
      s.t0 = cta * (base + 1);
    } else {
      s.nt = base;
      s.t0 = rem * (base + 1) + (cta - rem) * base;
    }
    return s;
  }
};

// ------------------------------------------------------------------ kernel
// mbar contract (recipes/mbar_handshake_design): full[s] count=1, arrive =
// producer's arrive_expect_tx(B+A bytes) + TMA complete_tx; empty[s] count =
// kCWarps (one elected lane per consumer warp after its last stage read).
// Producer = warp kCWarps lane 0; it never consumes, so the ring never
// self-blocks. Ring reuse distance = S stages, issue j waits empty parity
// ((j-S)/S)&1 — both derived from S.
// C = TMA cluster size for the A feed: ONE leader multicast per stage fills
// all C CTAs' A slots (cp_async_bulk_tensor_2d_load_multicast_cg1: each CTA's
// local full-bar gets its own tx-decrement, so expect_tx is unchanged). Per-CTA
// A bytes drop C-fold — the M-scaling A-tax (LEDGER O1) is A riding every
// CTA's capped TMA pipe. Contract deltas at C>1: the LEADER's empty[s] count
// = kCWarps*C (followers' consumers cluster-arrive it — release variant, so
// their slot reads are performed-before the leader's next multicast write);
// followers skip their own A issue but keep full expect_tx.
template <int M, int K, int R, Comm COMM, bool GEMM_ON, int S, int CH, int C>
__global__ void __launch_bounds__(kThreads) oproj_ar_kernel(
    const __grid_constant__ CUtensorMap w_map,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ Params<R> prm) {
  constexpr int Mp = (M + 15) & ~15;  // mma m16 padding (x buffer padded)
  constexpr int MT = Mp / 16;
  constexpr int KSTEPS = K / (kBK * CH);  // OUTER stages: CH k-chunks batched
  constexpr int kBBytes = kBNRows * kBK * 2, kABytes = Mp * kBK * 2;
  constexpr int kStB = CH * kBBytes, kStA = CH * kABytes;
  static_assert(K % (kBK * CH) == 0);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const Strip strip = Strip::make(blockIdx.x, gridDim.x);
  __shared__ uint32_t cta_epoch;
  if (tid == 0) cta_epoch = atomicAdd(prm.epoch_base + prm.fam, 1u) / uint32_t(gridDim.x);
  __syncthreads();
  const uint32_t epoch = cta_epoch;
  const int parity = int(epoch & 1);
  const int ring = int(epoch % kRing);
  // pinned wait-set: flag VA + monotonic targets resolved before any spin
  const size_t foff = flags_off(R) + size_t(prm.fam) * 512;
  const size_t doff2 = done_off(R) + size_t(prm.fam) * kRing * kMaxCTA * 4;
  uint32_t* const flag_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + foff) + ring;
  uint32_t* const done_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + doff2) + size_t(blockIdx.x);
  uint32_t* const gather_fam = prm.gather + size_t(prm.fam) * 2 * kRing;
  const uint32_t wrap = epoch / kRing + 1;
  const uint32_t flag_target = wrap * R;
  const uint32_t gath_target = wrap * gridDim.x;

  float4 acc[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i)
    acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);

  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* b_st = smem;                     // [S][CH][kBBytes]
  uint8_t* a_st = b_st + size_t(S) * kStB;  // [S][CH][kABytes]
  uint64_t* fullb = reinterpret_cast<uint64_t*>(a_st + size_t(S) * kStA);
  uint64_t* emptyb = fullb + S;

  const uint32_t crank = C > 1 ? device::ptx::cluster_cta_rank() : 0;
  {
    if (tid == 0) {
      device::ptx::prefetch_tensormap(&w_map);
      device::ptx::prefetch_tensormap(&x_map);
#pragma unroll
      for (int s = 0; s < S; ++s) {
        device::ptx::mbar_init(fullb + s, 1);
        device::ptx::mbar_init(emptyb + s, crank == 0 ? kCWarps * C : kCWarps);
      }
    }
    __syncthreads();
    if constexpr (C > 1) device::ptx::cluster_sync_rel_acq();

    if (warp == kCWarps) {
      // ---- producer: the whole K stream, one thread -----------------
      // k-phase rotation: spreads B's DRAM pages across CTAs; fp32
      // accumulation order changes per CTA — a sum, gate-covered.
      // CLUSTER-uniform: all members consume the same A flight.
      const int phase = ((int(blockIdx.x) / C) * KSTEPS) / (int(gridDim.x) / C);
      if (lane == 0) {
        for (int j = 0; j < KSTEPS; ++j) {
          const int slot = j % S;
          const int jj = (j + phase) % KSTEPS;
          if (j >= S) device::ptx::mbar_wait_parity(emptyb + slot, ((j - S) / S) & 1);
          device::ptx::mbar_arrive_expect_tx(fullb + slot, kStB + kStA);
#pragma unroll
          for (int c = 0; c < CH; ++c) {
            device::ptx::cp_async_bulk_tensor_2d_load(
                device::ptx::to_shared(b_st + size_t(slot) * kStB + c * kBBytes),
                &w_map,
                (jj * CH + c) * kBK,
                strip.t0 * 8,
                fullb + slot);
            if constexpr (C > 1) {
              if (crank == 0)
                device::ptx::cp_async_bulk_tensor_2d_load_multicast_cg1(
                    device::ptx::to_shared(a_st + size_t(slot) * kStA + c * kABytes),
                    &x_map,
                    (jj * CH + c) * kBK,
                    0,
                    fullb + slot,
                    uint16_t((1u << C) - 1));
            } else {
              device::ptx::cp_async_bulk_tensor_2d_load(
                  device::ptx::to_shared(a_st + size_t(slot) * kStA + c * kABytes),
                  &x_map,
                  (jj * CH + c) * kBK,
                  0,
                  fullb + slot);
            }
          }
        }
      }
    } else {
      // ---- consumers: warp w owns n8-tile (strip.t0 + w) ------------
      // A fragments load straight from gmem (x is L2-hot and tiny; the
      // LSU pipe is idle here) — the TMA path stays a pure-B stream.
      const int b_row = (lane & 7) + warp * 8;  // row within the B box
      const int b_ka = (lane >> 3) & 1;         // k-atom half (x2)
      const int a_row = lane & 15, a_ka = lane >> 4;
      for (int s = 0; s < KSTEPS; ++s) {
        const int slot = s % S;
        device::ptx::mbar_wait_parity(fullb + slot, (s / S) & 1);
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          const uint32_t b_base = device::ptx::to_shared(b_st + size_t(slot) * kStB + c * kBBytes);
#pragma unroll
          for (int k16 = 0; k16 < kBK / 16; ++k16) {
            uint32_t b0, b1;
            device::ptx::ldmatrix_x2_b16(
                b_base + uint32_t(b_row) * (kBK * 2) + swz::smem_col_128b_bf16(b_row, (k16 * 2 + b_ka) * 8) * 2,
                b0,
                b1);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              uint32_t a0, a1, a2, a3;
              device::ptx::ldmatrix_x4_b16(
                  device::ptx::to_shared(
                      a_st + size_t(slot) * kStA + c * kABytes + uint32_t(mt * 16 + a_row) * (kBK * 2) +
                      swz::smem_col_128b_bf16(a_row, (k16 * 2 + a_ka) * 8) * 2),
                  a0,
                  a1,
                  a2,
                  a3);
              device::ptx::mma_m16n8k16_bf16f32(acc[mt], a0, a1, a2, a3, b0, b1);
            }
          }
        }
        if (lane == 0) {
          device::ptx::mbar_arrive(emptyb + slot);
          if constexpr (C > 1)
            if (crank != 0) device::ptx::mbar_arrive_cluster_release(emptyb + slot, 0);
        }
      }
    }
  }

  // ---- epilogue: push ----------------------------------------------------
  __syncthreads();                      // whole CTA past its smem/feed reads
  device::PDLTriggerSecondary<true>();  // next launch streams weights under our
                                        // push+boundary+reduce (needs 2-CTA/SM
                                        // co-residency: 100% smem carveout + this
                                        // kernel's smem ≤ ~113 KB)
  device::PDLWaitPrimary<true>();       // prior grid reached ITS trigger (k-loop end) — NOT done
  {
    // guard: epoch e-2 (same parity) fully reduced everywhere before we
    // overwrite its slots. Steady-state this is already set (~one hot
    // acquire); it binds only when a boundary straggles.
    if (tid == 0 && epoch >= 2) {
      const uint32_t e2 = epoch - 2;
      const uint32_t tgt = (e2 / kRing + 1) * R;
      while (device::ptx::load_acquire_sys(done_local + size_t(e2 % kRing) * kMaxCTA) < tgt) {
      }
    }
    __syncthreads();
  }
  const int tig = lane & 3, grp = lane >> 2;
  const int n0 = (strip.t0 + warp) * 8;
  const bool own = warp < strip.nt;  // phantom 6th tile / producer never push

  // tile-major slot offset: tile t (global n8 index), row m, byte offset
  auto slot_tm = [&](uint8_t* base, size_t sb, int t, int m) { return base + sb + (size_t(t) * M + m) * 16; };

  {
    const size_t sb = slot_off(parity, prm.my_rank, R);
    if (own) {
      const int t = strip.t0 + warp;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int m = mt * 16 + grp + half * 8;
          if (m < M) {
            const uint32_t v =
                half ? bf2_u32(make_float2(acc[mt].z, acc[mt].w)) : bf2_u32(make_float2(acc[mt].x, acc[mt].y));
            // lanes (grp,tig) of one warp land contiguous: 128 B
            // per (mt,half) group
            const size_t boff = size_t(m) * 16 + tig * 4;
            {
#pragma unroll
              for (int r = 0; r < R; ++r)
                *reinterpret_cast<uint32_t*>(slot_tm(prm.uc_base[r], sb, t, 0) + boff) = v;
            }
          }
        }
      }
    }
  }

  // ---- boundary (gather + flag + per-CTA local-replica spin) -------------
  __syncthreads();
  if (tid == 0) {
    const uint32_t old = device::ptx::atomic_add_acq_rel_gpu(gather_fam + ring, 1);
    if (old + 1 == gath_target) {
      // ONE completing-CTA fence: publishes every CTA's pushes via the
      // acq_rel gather chain. Measured 0.9 us better than per-CTA
      // fences at bs1 (11.04 vs 11.97) — the parallel-drain theory lost.
      device::ptx::fence_release_sys();
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + foff) + ring, 1);
      }
    }
    while (device::ptx::load_acquire_sys(flag_local) < flag_target) {
    }
  }
  __syncthreads();

  // ---- reduce: each CTA finishes its own tiles from the LOCAL replica ----
  const int units = strip.nt * M;
  for (int u = tid; u < units; u += kThreads) {
    const int t = u / M, m = u % M, c0 = (strip.t0 + t) * 8;
    const size_t soff = (size_t(strip.t0 + t) * M + m) * 16;  // tile-major
    uint4 res;
    {
      float2 s[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 v = *reinterpret_cast<const uint4*>(prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
        const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
          s[j].x += f.x;
          s[j].y += f.y;
        }
      }
      res = make_uint4(bf2_u32(s[0]), bf2_u32(s[1]), bf2_u32(s[2]), bf2_u32(s[3]));
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(prm.out) + (size_t(m) * kN + c0) * 2) = res;
  }

  // ---- done-publish: per-CTA, fence-free. The beacon carries no payload:
  // the remote guard only needs its VALUE. This CTA's reduce loads are
  // data-flow-complete (their values fed the out stores) before the
  // syncthreads, so the relaxed beacon cannot pass an unfinished read; a
  // fence here stamped at ~4 us draining against the co-resident feed.
  __syncthreads();
  if (tid == 0) {
    const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
    {
#pragma unroll
      for (int r = 0; r < R; ++r)
        device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + doff2) + off, 1);
    }
  }
}

constexpr int kD3BM = 128, kD3BK = 64;
constexpr __host__ __device__ int d3_ns(int bn) {  // ring depth by stage size
  // deepest ring the smem budget fits, capped at the mbar array size. The
  // old 12/11/4 step form silently ran M>=128 at NS=4 (BN64 stage 24.5 KB
  // fits 9) — a 4-deep ring can't hide DRAM latency and was the dominant
  // M=128 pole (21.4 us vs DG 17.2 at-shape).
  const int stage = kD3BM * kD3BK * 2 + bn * kD3BK * 2;
  const int ns = 220 * 1024 / stage;
  return ns > 12 ? 12 : ns;
}
constexpr int kD3GroupN = 16, kD3KPer = kD3BK / 16;
constexpr int kD3Threads = 8 * 32;  // warps 0,1,4-7 live
constexpr int kD3ABytes = kD3BM * kD3BK * 2;
// per-M BLOCK_N: smaller BN = more tiles = more feed pipes (grid-starvation
// at BN=256 measured M-flat ~30.7 us: 28 CTAs); bigger BN = less A-dup.
// bytes/(min(tiles,152)) + tensor-floor optimum: 64 for M<=128, 128 above.
// re-swept 2026-07-23 at floored d3_ns: M=128 BN64/NS9 16.41 vs BN128/NS6
// 20.50 (ring depth, not BN, was the old 21.4 pole); 256 BN128/NS6 20.51;
// 512 BN256/NS4 30.9 (BN128's 224-on-152 nu-imbalance regressed to 38.9).
constexpr __host__ __device__ int d3_bn(int Mp) {
  return Mp <= 128 ? 64 : (Mp == 256 ? 128 : 256);
}

// drain-order slot offset: unit = (tile t, n-block nb, epi-warp w, lane l),
// 16 B each. Total = num_tiles * (BN/8) * 4 * 32 * 16 = Mpad*N*2 bytes.
SGL_DEVICE size_t d3_slot_off(int t, int nb, int w, int l, int nblk_per_tile) {
  return ((size_t(t) * nblk_per_tile + nb) * 4u + w) * 32u * 16u + size_t(l) * 16u;
}

// SWAP (M<=64): swapAB on the dense ring — A = W (M-slot carries the 7168
// n-rows in 128-row strips, NO small-M padding tax), B = x^T (N-slot = Mp,
// as small as 8). Tiles = 56 n-strips; the drain's lane-rows become n and
// its cols become m — the same drain-order slots work, only the out mapping
// transposes (local scatter, cheap at these payloads).
template <int M, int K, int R, Comm COMM>
__global__ void __launch_bounds__(kD3Threads) oproj_dense_ar_kernel(
    const __grid_constant__ CUtensorMap x_tmap,  // A = x [kMMax, K]
    const __grid_constant__ CUtensorMap w_tmap,  // B = W [kN, K]
    const __grid_constant__ Params<R> prm) {
  constexpr bool SWAP = (M <= 64);
  // two-shot planes share RS/reduce/out-copy; they differ only in the AG +
  // flag transport (kTwoShot = NVLS multimem, kTwoShotPeer = pure P2P).
  constexpr bool k2S = (COMM == Comm::kTwoShot || COMM == Comm::kTwoShotPeer);
  // SWAP = the DG decode recipe (O9h): token-slot tiles of 16/32 m-cols —
  // tiles = 56 strips × (Mp/tok) fills the grid (112 pipes at M≥32; the
  // 56-tile form measured flat ~19.6-20.5, R11) — with a 12-deep ring.
  // UMMA M=128 requires N ≥ 16 step 16 (bs1 pads the token slot to 16).
  constexpr int Mp = SWAP ? (M < 16 ? 16 : ((M + 15) & ~15)) : (M + kD3BM - 1) / kD3BM * kD3BM;
  constexpr int kD3BN = SWAP ? (Mp <= 32 ? 16 : 32) : d3_bn(Mp);
  constexpr int kD3BBytes = kD3BN * kD3BK * 2;
  constexpr int kGridM = SWAP ? kN / kD3BM : Mp / kD3BM;
  constexpr int kGridN = SWAP ? Mp / kD3BN : kN / kD3BN;
  constexpr int kTiles = kGridM * kGridN;
  constexpr int kIters = K / kD3BK;
  constexpr int kNBlk = kD3BN / 8;

  constexpr int kD3NS = d3_ns(kD3BN);
  // tcgen05.alloc column count must be a power of two >= 32
  constexpr int kTmemCols = 2 * kD3BN <= 32    ? 32
                            : 2 * kD3BN <= 64  ? 64
                            : 2 * kD3BN <= 128 ? 128
                            : 2 * kD3BN <= 256 ? 256
                                               : 512;
  extern __shared__ __align__(1024) uint8_t smem_buf[];
  const uint32_t smem_base = device::ptx::to_shared(smem_buf);
  constexpr uint32_t kSmemAOff = 0, kSmemBOff = kD3NS * kD3ABytes;

  __shared__ __align__(8) uint64_t tma_mbars[12];
  __shared__ __align__(8) uint64_t mma_mbars[12];
  __shared__ __align__(8) uint64_t mainloop_mbars[2];
  __shared__ __align__(8) uint64_t epi_mbars[2];
  __shared__ __align__(4) uint32_t s_taddr[1];
  __shared__ uint32_t cta_epoch;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane_id = tid & 31;
  const int bid = int(blockIdx.x), num_bids = int(gridDim.x);
  if (tid == 0) cta_epoch = atomicAdd(prm.epoch_base + prm.fam, 1u) / uint32_t(gridDim.x);
  __syncthreads();
  const uint32_t epoch = cta_epoch;
  const int parity = int(epoch & 1);
  const int ring = int(epoch % kRing);
  const size_t foff = flags_off(R) + size_t(prm.fam) * 512;
  const size_t doff2 = done_off(R) + size_t(prm.fam) * kRing * kMaxCTA * 4;
  uint32_t* const flag_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + foff) + ring;
  uint32_t* const done_local = reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + doff2) + size_t(blockIdx.x);
  uint32_t* const gather_fam = prm.gather + size_t(prm.fam) * 2 * kRing;
  const uint32_t wrap = epoch / kRing + 1;
  const uint32_t flag_target = wrap * R;
  const uint32_t gath_target = wrap * gridDim.x;
  const size_t sb = slot_off(parity, prm.my_rank, R);

  if (warp_id == 0 && device::ptx::elect_one()) {
    for (int i = 0; i < kD3NS; ++i) {
      device::ptx::mbar_init(&tma_mbars[i], 1);
      device::ptx::mbar_init(&mma_mbars[i], 1);
    }
    for (int i = 0; i < 2; ++i) {
      device::ptx::mbar_init(&mainloop_mbars[i], 1);
      device::ptx::mbar_init(&epi_mbars[i], 4 * 32);
    }
  } else if (warp_id == 1) {
    device::ptx::tcgen05_alloc(device::ptx::to_shared(s_taddr), kTmemCols);
  }
  __syncthreads();
  const uint32_t taddr = s_taddr[0];

  constexpr uint32_t i_desc = device::ptx::mma_inst_desc_f16(
      kD3BM,
      kD3BN,
      device::ptx::F16Type::BF16,
      device::ptx::F16Type::BF16,
      device::ptx::DType::F32,
      device::ptx::Major::K,
      device::ptx::Major::K);
  auto tile_mn = [&](int linear) -> int2 {
    return dense_gemm_mainloop::group_n_swizzle<1, kD3GroupN>(linear, 0, kGridM, kGridN);
  };

  // Prefetch the first ring of input-independent weight stages before the
  // PDL dependency. SWAP changes which operand slot contains W, but never
  // changes which tensor map is safe to touch here.
  if (warp_id == 0 && device::ptx::elect_one()) {
    constexpr int kPrefetch = kIters < kD3NS ? kIters : kD3NS;
    const int2 mn = tile_mn(bid);
#pragma unroll
    for (int k = 0; k < kPrefetch; ++k) {
      device::ptx::mbar_arrive_expect_tx(&tma_mbars[k], kD3ABytes + kD3BBytes);
      device::ptx::cp_async_bulk_tensor_2d_load(
          smem_base + (SWAP ? kSmemAOff + k * kD3ABytes : kSmemBOff + k * kD3BBytes),
          &w_tmap,
          k * kD3BK,
          (SWAP ? mn.x * kD3BM : mn.y * kD3BN),
          &tma_mbars[k]);
    }
  }

  // x and all slot traffic remain behind the dependency.
  device::PDLWaitPrimary<true>();

  // done-guard before any slot write (W3; PDL wait pairs with the trigger)
  {
    if (tid == 0 && epoch >= 2) {
      const uint32_t e2 = epoch - 2;
      const uint32_t tgt = (e2 / kRing + 1) * R;
      while (device::ptx::load_acquire_sys(done_local + size_t(e2 % kRing) * kMaxCTA) < tgt) {
      }
    }
    __syncthreads();
  }

  if (warp_id == 0 && device::ptx::elect_one()) {
    // TMA issuer (simple persistent) — verbatim dense_1cta fp8out shape
    int stage = 0, mma_phase = 1;
    for (int t = bid; t < kTiles; t += num_bids) {
      const int2 mn = tile_mn(t);
      for (int k = 0; k < kIters; ++k) {
        device::ptx::mbar_wait_parity(&mma_mbars[stage], mma_phase);
        constexpr bool kDropA = false;
        const bool prefetched = (t == bid && k < kD3NS);
        if (!prefetched) device::ptx::mbar_arrive_expect_tx(&tma_mbars[stage], (kDropA ? 0 : kD3ABytes) + kD3BBytes);
        if constexpr (!kDropA) {
          // In SWAP, A is the prefetched weight; otherwise A is x.
          if (!prefetched || !SWAP)
            device::ptx::cp_async_bulk_tensor_2d_load(
                smem_base + kSmemAOff + stage * kD3ABytes,
                SWAP ? &w_tmap : &x_tmap,
                k * kD3BK,
                mn.x * kD3BM,
                &tma_mbars[stage]);
        }
        // In SWAP, B is x; otherwise B is the prefetched weight.
        if (!prefetched || SWAP)
          device::ptx::cp_async_bulk_tensor_2d_load(
              smem_base + kSmemBOff + stage * kD3BBytes,
              SWAP ? &x_tmap : &w_tmap,
              k * kD3BK,
              mn.y * kD3BN,
              &tma_mbars[stage]);
        if (++stage == kD3NS) {
          stage = 0;
          mma_phase ^= 1;
        }
      }
    }
  } else if (warp_id == 1 && device::ptx::elect_one()) {
    // MMA issuer with 2-stage TMEM ping-pong
    int stage = 0, tma_phase = 0, ml_stage = 0, epi_phase = 1;
    for (int t = bid; t < kTiles; t += num_bids) {
      device::ptx::mbar_wait_parity(&epi_mbars[ml_stage], epi_phase);
      const uint32_t tmem_d = taddr + uint32_t(ml_stage) * kD3BN;
      for (int k = 0; k < kIters; ++k) {
        device::ptx::mbar_wait_parity(&tma_mbars[stage], tma_phase);
        device::ptx::tcgen05_fence_after_thread_sync();
        const uint32_t a_smem = smem_base + kSmemAOff + stage * kD3ABytes;
        const uint32_t b_smem = smem_base + kSmemBOff + stage * kD3BBytes;
#pragma unroll
        for (int k2 = 0; k2 < kD3KPer; ++k2) {
          const uint64_t da = device::ptx::mma_smem_desc_k_major<uint16_t, kD3BK, 128>(a_smem + uint32_t(k2) * 32);
          const uint64_t db = device::ptx::mma_smem_desc_k_major<uint16_t, kD3BK, 128>(b_smem + uint32_t(k2) * 32);
          device::ptx::tcgen05_mma_f16(tmem_d, da, db, i_desc, (k == 0 && k2 == 0) ? 0u : 1u);
        }
        device::ptx::tcgen05_commit_arrive(&mma_mbars[stage]);
        if (++stage == kD3NS) {
          stage = 0;
          tma_phase ^= 1;
        }
      }
      device::ptx::tcgen05_commit_arrive(&mainloop_mbars[ml_stage]);
      ml_stage ^= 1;
      if (ml_stage == 0) epi_phase ^= 1;
    }
  } else if (warp_id >= 4) {
    // epilogue: BF16 drain (dense_1cta idiom) → coalesced comm push
    const int epi_warp = warp_id & 3;
    const uint32_t taddr_lane = uint32_t(epi_warp * 32) << 16;
    int ml_stage = 0, ml_phase = 0;
    for (int t = bid; t < kTiles; t += num_bids) {
      const int2 mn = tile_mn(t);
      device::ptx::mbar_wait_parity(&mainloop_mbars[ml_stage], ml_phase);
      device::ptx::tcgen05_fence_after_thread_sync();
      const uint32_t tmem_d_base = taddr + uint32_t(ml_stage) * kD3BN;
      const int row = mn.x * kD3BM + epi_warp * 32 + lane_id;  // SWAP: n
#pragma unroll 4
      for (int nb = 0; nb < kNBlk; ++nb) {
        const uint32_t taddr_n = tmem_d_base + uint32_t(nb) * 8 + taddr_lane;
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        device::ptx::tcgen05_ld_32x32b_x8(taddr_n, r0, r1, r2, r3, r4, r5, r6, r7);
        device::ptx::tcgen05_wait_ld();
        uint4 v;
        v.x = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r1), __int_as_float(r0));
        v.y = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r3), __int_as_float(r2));
        v.z = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r5), __int_as_float(r4));
        v.w = device::ptx::cvt_pack_f32x2_to<device::ptx::bf16>(__int_as_float(r7), __int_as_float(r6));
        if (SWAP || row < M) {  // SWAP masks pad m-cols below
          {
            const size_t off = sb + d3_slot_off(t, nb, epi_warp, lane_id, kNBlk);
            if constexpr (k2S)
              // RS: unicast to tile-owner only — 1x egress and,
              // spread over the tile loop, absorbed under the
              // mainloop (O9b: one-shot is R x-payload INGRESS-
              // bound; the composite pays its RS serially)
              *reinterpret_cast<uint4*>(prm.uc_base[t % R] + off) = v;
            else {
#pragma unroll
              for (int r = 0; r < R; ++r)
                *reinterpret_cast<uint4*>(prm.uc_base[r] + off) = v;
            }
          }
        }
      }
      (void)device::ptx::mbar_arrive(&epi_mbars[ml_stage]);
      ml_stage ^= 1;
      if (ml_stage == 0) ml_phase ^= 1;
    }
  }
  __syncthreads();
  device::PDLTriggerSecondary<true>();
  if (warp_id == 1) {
    device::ptx::tcgen05_dealloc(taddr, kTmemCols);
    device::ptx::tcgen05_relinquish();
  }

  // ---- boundary (fam rings) — verbatim member-1 contract ----------------
  if (tid == 0) {
    const uint32_t old = device::ptx::atomic_add_acq_rel_gpu(gather_fam + ring, 1);
    if (old + 1 == gath_target) {
      device::ptx::fence_release_sys();
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + foff) + ring, 1);
      }
    }
    while (device::ptx::load_acquire_sys(flag_local) < flag_target) {
    }
  }
  __syncthreads();

  if constexpr (k2S) {
    // ---- owner-reduce + AG: reduce MY tiles, store the result to all
    // replicas' pull region (kTwoShot: one mm.st, fabric replicates;
    // kTwoShotPeer: R unicast stores — (R-1)× the egress, same ingress);
    // then boundary 2 gates the out-copy ---------------------------------
    for (int u = tid + bid * kD3Threads; u < kTiles * kNBlk * 4 * 32; u += num_bids * kD3Threads) {
      const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
      const int t = u / (kNBlk * 128);
      if (t % R != prm.my_rank) continue;  // not my slab
      const int2 mn = tile_mn(t);
      if (!SWAP && mn.x * kD3BM + w * 32 + l >= M) continue;
      const size_t soff = d3_slot_off(t, nb, w, l, kNBlk);
      float2 acc2[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 vv = *reinterpret_cast<const uint4*>(prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
        const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
          acc2[j].x += f.x;
          acc2[j].y += f.y;
        }
      }
      const uint4 res = make_uint4(bf2_u32(acc2[0]), bf2_u32(acc2[1]), bf2_u32(acc2[2]), bf2_u32(acc2[3]));
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          *reinterpret_cast<uint4*>(prm.uc_base[r] + pull_off(parity, R) + soff) = res;
      }
    }
    // boundary 2 (second flag/gather ring words at +256 B / +kRing)
    __syncthreads();
    if (tid == 0) {
      uint32_t* const g2 = gather_fam + kRing;  // AG gather ring
      const uint32_t old = device::ptx::atomic_add_acq_rel_gpu(g2 + ring, 1);
      if (old + 1 == gath_target) {
        device::ptx::fence_release_sys();
        {
#pragma unroll
          for (int r = 0; r < R; ++r)
            device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + foff + 256) + ring, 1);
        }
      }
      while (device::ptx::load_acquire_sys(reinterpret_cast<uint32_t*>(prm.uc_base[prm.my_rank] + foff + 256) + ring) <
             flag_target) {
      }
    }
    __syncthreads();
    // out-copy: every CTA writes its grid-partition of out from the
    // LOCAL reduced replica
    for (int u = tid + bid * kD3Threads; u < kTiles * kNBlk * 4 * 32; u += num_bids * kD3Threads) {
      const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
      const int t = u / (kNBlk * 128);
      const int2 mn = tile_mn(t);
      const int row = mn.x * kD3BM + w * 32 + l;  // SWAP: row = n, full
      if (!SWAP && row >= M) continue;            // range; m masked below
      const uint4 res = *reinterpret_cast<const uint4*>(
          prm.uc_base[prm.my_rank] + pull_off(parity, R) + d3_slot_off(t, nb, w, l, kNBlk));
      if constexpr (SWAP) {
        const uint32_t u4[4] = {res.x, res.y, res.z, res.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int mm = mn.y * kD3BN + nb * 8 + j;
          if (mm < M)
            *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(prm.out) + (size_t(mm) * kN + row) * 2) =
                uint16_t((u4[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu);
        }
      } else {
        *reinterpret_cast<uint4*>(
            reinterpret_cast<uint8_t*>(prm.out) + (size_t(row) * kN + size_t(mn.y) * kD3BN + nb * 8) * 2) = res;
      }
    }
    __syncthreads();
    if (tid == 0) {
      const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
      {
#pragma unroll
        for (int r = 0; r < R; ++r)
          device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + doff2) + off, 1);
      }
    }
    return;
  }

  // ---- reduce: descatter drain-order slots → out[m][n] -------------------
  // unit = (t, nb, w, l): row = mn.x*BM + w*32 + l; cols = mn.y*BN + nb*8.
  const int units = kTiles * kNBlk * 4 * 32;
  for (int u = tid + bid * kD3Threads; u < units; u += num_bids * kD3Threads) {
    const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
    const int t = u / (kNBlk * 128);
    const int2 mn = tile_mn(t);
    const int row = mn.x * kD3BM + w * 32 + l;
    if (!SWAP && row >= M) continue;
    const size_t soff = d3_slot_off(t, nb, w, l, kNBlk);
    uint4 res;
    {
      float2 acc2[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 vv = *reinterpret_cast<const uint4*>(prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
        const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
          acc2[j].x += f.x;
          acc2[j].y += f.y;
        }
      }
      res = make_uint4(bf2_u32(acc2[0]), bf2_u32(acc2[1]), bf2_u32(acc2[2]), bf2_u32(acc2[3]));
    }
    if constexpr (SWAP) {
      const uint32_t u4[4] = {res.x, res.y, res.z, res.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int mm = mn.y * kD3BN + nb * 8 + j;
        if (mm < M)
          *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(prm.out) + (size_t(mm) * kN + row) * 2) =
              uint16_t((u4[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu);
      }
    } else {
      *reinterpret_cast<uint4*>(
          reinterpret_cast<uint8_t*>(prm.out) + (size_t(row) * kN + size_t(mn.y) * kD3BN + nb * 8) * 2) = res;
    }
  }

  // ---- done publish (fence-free beacon, W4) ------------------------------
  __syncthreads();
  if (tid == 0) {
    const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
    {
#pragma unroll
      for (int r = 0; r < R; ++r)
        device::ptx::red_add_relaxed_sys(reinterpret_cast<uint32_t*>(prm.uc_base[r] + doff2) + off, 1);
    }
  }
}

template <int M, int K, int R, Comm COMM>
struct Launcher3 {
  static constexpr bool kSwap = (M <= 64);
  static constexpr int Mp = kSwap ? (M < 16 ? 16 : ((M + 15) & ~15)) : (M + kD3BM - 1) / kD3BM * kD3BM;
  static constexpr int kBN = kSwap ? (Mp <= 32 ? 16 : 32) : d3_bn(Mp);
  static constexpr int kTiles = kSwap ? (kN / kD3BM) * (Mp / kBN) : (Mp / kD3BM) * (kN / kBN);
  static constexpr int kGrid = kTiles < 152 ? kTiles : 152;
  static constexpr size_t kSmem = size_t(d3_ns(kBN)) * (kD3ABytes + kBN * kD3BK * 2);
  static void set_smem_attr() {
    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_dense_ar_kernel<M, K, R, COMM>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_dense_ar_kernel<M, K, R, COMM>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }
  static void
  launch(const CUtensorMap& x_tmap, const CUtensorMap& w_tmap, const Params<R>& prm, cudaStream_t stream, bool pdl) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attr[1];
    int na = 0;
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    cfg.gridDim = dim3(unsigned(kGrid));
    cfg.blockDim = dim3(kD3Threads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = unsigned(na);
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, oproj_dense_ar_kernel<M, K, R, COMM>, x_tmap, w_tmap, prm));
  }
};

template <
    int M,
    int K,
    int R,
    Comm COMM,
    bool GEMM_ON,
    int S = ((M + 15) & ~15) <= 16 ? 6 : (((M + 15) & ~15) == 32 ? 4 : 3),
    int CH = 2,
    int C = 1>  // cluster-multicast A axis: C=2 REFUTED as-built at
                // S=3/4 rings (LEDGER R9 — pair-lockstep pacing beats
                // the halved A bytes); flip here to test siblings
struct Launcher {
  static constexpr int Mp = (M + 15) & ~15;
  static constexpr size_t kSmem = GEMM_ON
                                      ? size_t(S) * CH * (kBNRows * kBK * 2 + Mp * kBK * 2) + 2 * S * sizeof(uint64_t)
                                      : 4096;  // AR-only path never touches the feed ring
  // once per device, before first launch
  static void set_smem_attr() {
    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
    // 100% carveout → the SM smem config fits TWO CTAs (this grid's tail
    // + the next PDL grid's feed); the default config blocks dual
    // residency and with it the whole tail-hiding scheme.
    CHECK_CUDA(cudaFuncSetAttribute(
        oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }
  // pdl=false = the NON-COOPERATIVE-neighbor regime: no programmatic
  // serialization attribute, so successive kernels fully serialize — the
  // AR tail is exposed, as it is in a serving stack whose adjacent kernels
  // don't PDL-cooperate or can't co-reside. In-kernel griddepcontrol ops
  // are no-ops without the attribute; the done-guard is trivially met.
  static void launch(
      const CUtensorMap& w_map,
      const CUtensorMap& x_map,
      const Params<R>& prm,
      int ncta,
      cudaStream_t stream,
      bool pdl) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    if constexpr (C > 1) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = C;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    cfg.gridDim = dim3(unsigned(ncta));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = unsigned(na);
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>, w_map, x_map, prm));
  }
};

}  // namespace oproj_ar

}  // namespace sglang

// ================= sglang tvm-ffi adapter =================

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST

#include <sgl_kernel/utils.cuh>  // For bf16_t, TVMFFIEnvGetStream

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <mutex>
#include <unordered_map>

namespace sglang {

namespace oproj_ar_ffi {

using namespace oproj_ar;
using tvm::ffi::TensorView;

constexpr int kCellList[7] = {8, 16, 32, 64, 128, 256, 512};

inline int cell_of(int m) {
  for (int c : kCellList)
    if (m <= c) return c;
  return -1;
}

template <int K, int R, bool kUsePDL>
struct GemmArKernel {
  static_assert(R >= 2 && R <= 8, "R outside the validated 2..8 range");
  static_assert(K % 128 == 0 && K >= 128, "K must be a multiple of 128");

  static constexpr int kTwoShotMinM = R >= 8 ? 128 : 256;

  // one ring family per dispatch cell (see kFams)
  static int64_t fam_of(int64_t m) {
    const int cell = cell_of(int(m));
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    for (int i = 0; i < 7; ++i)
      if (kCellList[i] == cell) return i;
    return -1;
  }

  static int64_t cell_of_ffi(int64_t m) {
    return cell_of(int(m));
  }
  static int64_t region_nbytes() {
    return int64_t(region_bytes(R));
  }
  static int64_t gather_words() {
    return int64_t(kFams) * 2 * kRing;
  }
  static int64_t num_fams() {
    return kFams;
  }
  static int64_t max_tokens() {
    return kMMax;
  }

  // Per-weight-pointer W tensor maps (encode once; weights are static).
  struct WMaps {
    CUtensorMap w48, w64, w128, w256;
  };

  static const WMaps& w_maps(void* w) {
    static std::unordered_map<void*, WMaps> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(w);
    if (it == cache.end()) {
      auto enc = [&](uint32_t box_rows) {
        return tmap::encode_tiled_2d(
            w, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kN, K, size_t(K) * 2, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
      };
      it = cache.emplace(w, WMaps{enc(kBNRows), enc(64), enc(128), enc(256)}).first;
    }
    return it->second;
  }

  // x tensor map over the caller's [M, K] tensor: global rows = M, TMA
  // zero-fills the [M, cell) padding rows out-of-bounds.
  static CUtensorMap x_map(void* x, int m, uint32_t box_rows) {
    return tmap::encode_tiled_2d(
        x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, uint64_t(m), K, size_t(K) * 2, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
  }

  static void set_smem_attrs_once() {
    static bool done = [] {
      Launcher<8, K, R, Comm::kPeer, true>::set_smem_attr();
      Launcher<16, K, R, Comm::kPeer, true>::set_smem_attr();
      Launcher3<32, K, R, Comm::kPeer>::set_smem_attr();
      Launcher3<64, K, R, Comm::kPeer>::set_smem_attr();
      if constexpr (kTwoShotMinM > 128) Launcher3<128, K, R, Comm::kPeer>::set_smem_attr();
      if constexpr (kTwoShotMinM <= 128) Launcher3<128, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      Launcher3<256, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      Launcher3<512, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      return true;
    }();
    (void)done;
  }

  template <int CELL>
  static void enqueue_cell(const WMaps& wm, void* x, int m, const Params<R>& prm, cudaStream_t stream, bool pdl) {
    if constexpr (CELL <= 16) {
      const CUtensorMap xm = x_map(x, m, 16);
      Launcher<CELL, K, R, Comm::kPeer, true>::launch(wm.w48, xm, prm, kM1Grid_(), stream, pdl);
    } else if constexpr (CELL < kTwoShotMinM) {
      const CUtensorMap xm = x_map(x, m, CELL <= 32 ? 16 : (CELL <= 64 ? 32 : 128));
      const CUtensorMap& wmap = CELL <= 64 ? wm.w128 : wm.w64;
      Launcher3<CELL, K, R, Comm::kPeer>::launch(xm, wmap, prm, stream, pdl);
    } else {
      const CUtensorMap xm = x_map(x, m, 128);
      const CUtensorMap& wmap = CELL == 128 ? wm.w64 : (CELL == 256 ? wm.w128 : wm.w256);
      Launcher3<CELL, K, R, Comm::kTwoShotPeer>::launch(xm, wmap, prm, stream, pdl);
    }
  }

  static constexpr int kM1Grid_() {
    return 152;
  }

  // per-rank UC VAs of the comm region, stashed host-side ONCE at init:
  // per-call CPU-tensor derefs from inside the op are not reliable in every
  // execution context (observed dangling under the sglang scheduler).
  static std::array<uint8_t*, R>& bases_store() {
    static std::array<uint8_t*, R> a{};
    return a;
  }

  static void set_bases(TensorView uc_bases) {
    using namespace host;
    TensorMatcher({R}).with_dtype<int64_t>().verify(uc_bases);
    const int64_t* b = static_cast<const int64_t*>(uc_bases.data_ptr());
    for (int r = 0; r < R; ++r)
      bases_store()[r] = reinterpret_cast<uint8_t*>(b[r]);
  }

  static void
  run(TensorView out,
      TensorView x,
      TensorView w,
      TensorView gather,  // [kFams * 2 * kRing] int32 CUDA, device-local
      TensorView epochs,  // [kFams] int32 CUDA: device-resident CTA ticket counters
      int64_t my_rank) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto CellRows = SymbolicSize{"cell_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({kN, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({CellRows, kN}).with_dtype<bf16_t>().with_device(device).verify(out);

    const int m = int(M.unwrap());
    const int cell = cell_of(m);
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    CHECK_HOST(int64_t(CellRows.unwrap()) == cell)
        << "out must have cell(M)=" << cell << " rows, got " << CellRows.unwrap();
    CHECK_HOST(my_rank >= 0 && my_rank < R);
    CHECK_HOST(bases_store()[0] != nullptr) << "gemm_ar: set_bases not called";
    TensorMatcher({int64_t(kFams) * 2 * kRing}).with_dtype<int32_t>().verify(gather);
    TensorMatcher({kFams}).with_dtype<int32_t>().verify(epochs);

    set_smem_attrs_once();

    const DLDevice dev = device.unwrap();
    const auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    Params<R> prm{};
    prm.mc_base = nullptr;  // pure-P2P plane
    for (int r = 0; r < R; ++r)
      prm.uc_base[r] = bases_store()[r];
    prm.gather = static_cast<uint32_t*>(gather.data_ptr());
    prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.partial_in = nullptr;
    prm.epoch_base = static_cast<uint32_t*>(epochs.data_ptr());
    prm.my_rank = int(my_rank);
    prm.fam = int(fam_of(m));

    const bool pdl = kUsePDL;
    const WMaps& wm = w_maps(w.data_ptr());
    switch (cell) {
      case 8:
        enqueue_cell<8>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 16:
        enqueue_cell<16>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 32:
        enqueue_cell<32>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 64:
        enqueue_cell<64>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 128:
        enqueue_cell<128>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 256:
        enqueue_cell<256>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      default:
        enqueue_cell<512>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
    }
    CHECK_CUDA(cudaGetLastError()) << "gemm_ar launch (cell=" << cell << ")";
  }
};

}  // namespace oproj_ar_ffi

using oproj_ar_ffi::GemmArKernel;

}  // namespace sglang
