// kda_prefill.cu — self-contained KDA prefill forward (inference drop-in for
// the FLA Triton chunk_kda_fwd path). Single-file artifact carrying ONLY the
// shipping default paths — two routes, picked per shape from the launch
// geometry (pick_route):
//   FUSED (long single sequences) —
//     eqlen  (T % 64 == 0, no cu_seqlens):
//       nc >= 4:  kda_fused<GM>  — ONE grid: NP piece-builders (k1 factors +
//                 in-tail running map composition) + NP trailing self-start
//                 chain blocks gated on per-piece flags.
//       nc <  4:  k1_factors_mma<GM> + k2_chain_tc (NP == 1 two-kernel path).
//     varlen (cu_seqlens / ragged T): kda_fused_vl<GM> over a host-built
//       per-sequence piece table (non-tail pieces run the exact eqlen bodies).
//   SEQ0 (many sequences / high H): the tail-free dieted builder
//     k1_tf_builder<GM> / k1_tf_builder_vl<GM> (no composition, 2 CTAs/SM),
//     then ONE chain per (sequence, head) walking from h0 — k2_chain_tc /
//     k2_chain_tc_vl. Its P/u0 differ from the fused route's by design (P4-lo
//     and the block apply, envelope-gated) — see the seq0 section.
// Gate modes (GM): 0 = pre-transformed bf16 glog; 1 = raw softplus,
// 2 = raw safe_gate — both transformed in place off the serial path
// (eqlen AND varlen). RAW and BSIG (orthogonal to GM and to each other, same
// dispatch): RAW = q/k arrive un-normalized, so the caller's l2norm folds into
// the tiles k1 already loads (row_rnorm); BSIG = beta arrives as logits, so
// its sigmoid folds into the beta read.
// Chunk size 64, head dim K == V == 128 fixed.
//
// C++ API (pybind, see kda_prefill.py for the FLA-signature wrapper):
//   kda_prefill_fwd(q, k, v, g, beta, scale, initial_state, cu_seqlens,
//                   use_gate_in_kernel, A_log, dt_bias, safe_gate,
//                   lower_bound, use_qk_l2norm_in_kernel,
//                   use_beta_sigmoid_in_kernel, h_per_chunk, h_v_first)
//                -> (o [T,H,128] bf16, Sf [N,H,128,128] f32)
//   h_per_chunk is an optional preallocated per-chunk state output.
//
// Build (torch cpp_extension JIT):
//   -O3 -std=c++20 -gencode arch=compute_103a,code=sm_103a -use_fast_math
//   -lineinfo, link -lcuda (cuTensorMapEncodeTiled). sm_103a (GB300) only.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <map>
#include <type_traits>
#include <vector>

// Inlined PTX helpers: only the subset the kernels below use, kept local so this
// translation unit compiles standalone.
namespace ptx {

template <typename T>
static __device__ __forceinline__ uint32_t to_shared(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ---- mbarrier ----
static __device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}
static __device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}
static __device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}

// ---- cp.async (per-thread 16 B gmem->smem) ----
static __device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(smem)), "l"(gmem));
}
// ignore-src predicate form: pad rows stage zeros branchlessly
static __device__ __forceinline__ void cp_async_16_zfill(void* smem, const void* gmem, int ignore_src) {
  asm volatile(
      "{\n\t.reg .pred pz;\n\t"
      "setp.ne.b32 pz, %2, 0;\n\t"
      "cp.async.cg.shared.global [%0], [%1], 16, pz;\n\t}" ::"r"(to_shared(smem)),
      "l"(gmem),
      "r"(ignore_src));
}
static __device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
static __device__ __forceinline__ void cp_async_wait_pending(int pending) {
  switch (pending) {
    case 0:
      asm volatile("cp.async.wait_group 0;");
      break;
    default:
      asm volatile("cp.async.wait_group 1;");
      break;
  }
}

// ---- TMA (2D tiled bulk loads) ----
static __device__ __forceinline__ void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}
static __device__ __forceinline__ void
cp_async_bulk_tensor_2d_load(uint32_t dst_smem, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::"
      "complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}

// ---- ldmatrix + warp mma (k1's pair-tile products) ----
static __device__ __forceinline__ void
ldmatrix_x4_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(row_addr));
}
static __device__ __forceinline__ void ldmatrix_x2_b16(uint32_t row_addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(row_addr));
}
static __device__ __forceinline__ void
mma_m16n8k16_bf16f32(float4& d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---- gpu-scoped release/acquire flags (fused-grid piece gating) ----
static __device__ __forceinline__ void red_add_rel_b32(uint32_t* ptr, uint32_t value) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(value));
}
static __device__ __forceinline__ uint32_t ld_acq_b32(const uint32_t* ptr) {
  uint32_t ret;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}
// make generic-proxy global writes visible to the async proxy (consumers
// read the published factor tensors via TMA)
static __device__ __forceinline__ void fence_async_global() {
  asm volatile("fence.proxy.async.global;");
}

// ---- tcgen05 (TMEM lifecycle, ld/st, MMA, fences) ----
static __device__ __forceinline__ void tcgen05_alloc(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_addr_for_taddr), "r"(n_cols));
}
static __device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}
static __device__ __forceinline__ void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
static __device__ __forceinline__ void
tcgen05_st_32x32b_x4(uint32_t taddr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
static __device__ __forceinline__ void tcgen05_st_32x32b_x8(
    uint32_t taddr,
    uint32_t r0,
    uint32_t r1,
    uint32_t r2,
    uint32_t r3,
    uint32_t r4,
    uint32_t r5,
    uint32_t r6,
    uint32_t r7) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
      " {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(taddr),
      "r"(r0),
      "r"(r1),
      "r"(r2),
      "r"(r3),
      "r"(r4),
      "r"(r5),
      "r"(r6),
      "r"(r7));
}
static __device__ __forceinline__ void tcgen05_st_32x32b_x16(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      " {%0, %1, %2, %3, %4, %5, %6, %7,"
      "  %8, %9, %10, %11, %12, %13, %14, %15};" ::"r"(r[0]),
      "r"(r[1]),
      "r"(r[2]),
      "r"(r[3]),
      "r"(r[4]),
      "r"(r[5]),
      "r"(r[6]),
      "r"(r[7]),
      "r"(r[8]),
      "r"(r[9]),
      "r"(r[10]),
      "r"(r[11]),
      "r"(r[12]),
      "r"(r[13]),
      "r"(r[14]),
      "r"(r[15]),
      "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x8(
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
static __device__ __forceinline__ void tcgen05_ld_32x32b_x16(
    uint32_t taddr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    uint32_t& r4,
    uint32_t& r5,
    uint32_t& r6,
    uint32_t& r7,
    uint32_t& r8,
    uint32_t& r9,
    uint32_t& r10,
    uint32_t& r11,
    uint32_t& r12,
    uint32_t& r13,
    uint32_t& r14,
    uint32_t& r15) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      " {%0, %1, %2, %3, %4, %5, %6, %7,"
      "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=r"(r0),
        "=r"(r1),
        "=r"(r2),
        "=r"(r3),
        "=r"(r4),
        "=r"(r5),
        "=r"(r6),
        "=r"(r7),
        "=r"(r8),
        "=r"(r9),
        "=r"(r10),
        "=r"(r11),
        "=r"(r12),
        "=r"(r13),
        "=r"(r14),
        "=r"(r15)
      : "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      " {%0, %1, %2, %3, %4, %5, %6, %7,"
      "  %8, %9, %10, %11, %12, %13, %14, %15,"
      "  %16, %17, %18, %19, %20, %21, %22, %23,"
      "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=r"(r[0]),
        "=r"(r[1]),
        "=r"(r[2]),
        "=r"(r[3]),
        "=r"(r[4]),
        "=r"(r[5]),
        "=r"(r[6]),
        "=r"(r[7]),
        "=r"(r[8]),
        "=r"(r[9]),
        "=r"(r[10]),
        "=r"(r[11]),
        "=r"(r[12]),
        "=r"(r[13]),
        "=r"(r[14]),
        "=r"(r[15]),
        "=r"(r[16]),
        "=r"(r[17]),
        "=r"(r[18]),
        "=r"(r[19]),
        "=r"(r[20]),
        "=r"(r[21]),
        "=r"(r[22]),
        "=r"(r[23]),
        "=r"(r[24]),
        "=r"(r[25]),
        "=r"(r[26]),
        "=r"(r[27]),
        "=r"(r[28]),
        "=r"(r[29]),
        "=r"(r[30]),
        "=r"(r[31])
      : "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
static __device__ __forceinline__ void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
static __device__ __forceinline__ void tcgen05_commit_arrive(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(to_shared(bar)));
}
static __device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}
static __device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}
static __device__ __forceinline__ void
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
// kind::f16 with the A operand sourced from TMEM (packed-bf16 hi/lo A)
static __device__ __forceinline__ void
tcgen05_mma_f16_atmem(uint32_t d, uint32_t a_tmem, uint64_t desc_b, uint32_t inst_desc_high, uint32_t scale_c) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(d),
      "r"(a_tmem),
      "l"(desc_b),
      "r"(inst_desc_high),
      "r"(scale_c));
}

// ---- MMA descriptors (smem matrix desc + instruction desc, Table 44) ----
enum class Major : uint8_t { K = 0, MN = 1 };
enum class F16Type : uint8_t { F16 = 0, BF16 = 1 };
enum class DType : uint8_t { F16 = 0, F32 = 1, S32 = 2 };

__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc(uint32_t matrix_addr, uint32_t lbo, uint32_t sbo, uint32_t base_offset, int swizzle_bytes) {
  auto enc = [](uint32_t x) -> uint64_t { return (uint64_t)((x & 0x3FFFFu) >> 4); };
  uint8_t code = (swizzle_bytes == 128) ? 2u : (swizzle_bytes == 64) ? 4u : (swizzle_bytes == 32) ? 6u : 0u;
  uint64_t d = 0;
  d |= enc(matrix_addr);                    // bits  0-13
  d |= enc(lbo) << 16;                      // bits 16-29
  d |= enc(sbo) << 32;                      // bits 32-45
  d |= uint64_t(1u) << 46;                  // version = 1
  d |= uint64_t(base_offset & 0x7u) << 49;  // bits 49-51
  d |= uint64_t(code & 0x7u) << 61;         // bits 61-63
  return d;
}
template <typename T, int BLOCK_K, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc_k_major(uint32_t addr, uint32_t base_offset = 0) {
  constexpr int K_BYTES = BLOCK_K * int(sizeof(T));
  static_assert(SWIZZLE_BYTES == K_BYTES, "K-major requires swizzle bytes == BLOCK_K * sizeof(T)");
  return mma_smem_desc(addr, /*lbo=*/0u, /*sbo=*/8u * uint32_t(K_BYTES), base_offset, SWIZZLE_BYTES);
}
// MN-major operand (inner = N for B) over swizzle-atom-form smem: adjacent
// K-row groups of 8 at SBO = 8 * SWZ within an MN-chunk, adjacent MN-chunks
// (one SWZ128 TMA box each) at LBO = BLOCK_K * SWZ.
template <typename T, int BLOCK_K, int BLOCK_MN, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc_mn_major(uint32_t addr, uint32_t base_offset = 0) {
  constexpr int BLOCK_MN_BYTES = BLOCK_MN * int(sizeof(T));
  static_assert(
      SWIZZLE_BYTES == 32 || SWIZZLE_BYTES == 64 || SWIZZLE_BYTES == 128, "MN-major requires SWZ in {32, 64, 128}");
  static_assert(BLOCK_MN_BYTES % SWIZZLE_BYTES == 0, "MN-major: BLOCK_MN * sizeof(T) must tile the swizzle atom");
  return mma_smem_desc(
      addr,
      /*lbo=*/uint32_t(BLOCK_K * SWIZZLE_BYTES),
      /*sbo=*/uint32_t(8 * SWIZZLE_BYTES),
      base_offset,
      SWIZZLE_BYTES);
}
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_f16(
    uint32_t M,
    uint32_t N,
    F16Type a_type = F16Type::BF16,
    F16Type b_type = F16Type::BF16,
    DType d_type = DType::F32,
    Major a_major = Major::K,
    Major b_major = Major::K) {
  uint32_t d = 0;
  d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
  d |= (static_cast<uint32_t>(a_type) & 0x7u) << 7;
  d |= (static_cast<uint32_t>(b_type) & 0x7u) << 10;
  d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
  d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
  d |= ((N >> 3) & 0x3Fu) << 17;
  d |= ((M >> 4) & 0x1Fu) << 24;
  return d;
}
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_tf32(
    uint32_t M, uint32_t N, DType d_type = DType::F32, Major a_major = Major::K, Major b_major = Major::K) {
  constexpr uint32_t TF32 = 2u;
  uint32_t d = 0;
  d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
  d |= TF32 << 7;   // atype = TF32 = 2
  d |= TF32 << 10;  // btype = TF32 = 2
  d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
  d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
  d |= ((N >> 3) & 0x3Fu) << 17;
  d |= ((M >> 4) & 0x1Fu) << 24;
  return d;
}
// thin shim keeping the kernel bodies identical to the development source
enum class MmaDenseKind : uint8_t { F16 = 0 };
template <MmaDenseKind KIND>
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_dense(
    uint32_t M,
    uint32_t N,
    F16Type a_type,
    F16Type b_type,
    DType d_type = DType::F32,
    Major a_major = Major::K,
    Major b_major = Major::K) {
  static_assert(KIND == MmaDenseKind::F16);
  return mma_inst_desc_f16(M, N, a_type, b_type, d_type, a_major, b_major);
}

}  // namespace ptx

namespace kda {

constexpr int BT = 64;  // chunk tokens
constexpr int K = 128;  // head dim (qk == v)
using bf16 = __nv_bfloat16;

// balanced piece boundaries (k1's wall = the LONGEST piece: only a
// balanced split minimizes it)
__device__ static inline int piece_c0(int p, int nc, int NP) {
  const int base = nc / NP, rem = nc % NP;
  return base * p + (p < rem ? p : rem);
}
// kind::tf32 with the A operand from TMEM (probe-verified: identity (m,k)->
// (lane,col) map — a fp32 D region reads directly as a K=128 A operand,
// +8-translatable slices, truncation read semantics). B via fp32 SWZ32
// k-major K=8 chunks.
static __device__ __forceinline__ void
mma_tf32_atmem(uint32_t d, uint32_t a_tmem, uint64_t desc_b, uint32_t inst_desc_high, uint32_t scale_c) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(d),
      "r"(a_tmem),
      "l"(desc_b),
      "r"(inst_desc_high),
      "r"(scale_c));
}
// bf16 piece/segment L maps (half the TMA bytes of the former fp32; the c
// maps stay fp32 — bf16 c accumulated past the fla o bar at 16K) land as
// two SWZ128 [128][64] tiles in the TOP half of their 64 KB fp32 slot and
// widen IN PLACE to the SWZ32 fp32 K=8 tiles the tf32 B descs read. fp32
// chunks 8..15 overwrite the pad, so every thread stages all its reads in
// registers behind one barrier (all pad reads precede any overlapping
// write). Callers: 512 threads; thread (vc, ch) owns row vc, cols
// [ch*32, ch*32+32).
static __device__ __forceinline__ void widen_map_tf32(float* slot, const bf16* pad, int vc, int ch) {
  uint32_t w[16];
#pragma unroll
  for (int g = 0; g < 4; ++g) {  // one 16B SWZ128 atom per load
    const int in0 = ch * 32 + g * 8;
    reinterpret_cast<uint4*>(w)[g] =
        *reinterpret_cast<const uint4*>(pad + (in0 >> 6) * (K * 64) + vc * 64 + ((((in0 >> 3) & 7) ^ (vc & 7)) << 3));
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < 8; ++g) {  // 4-col groups: one 16B half-row each
    const int in0 = ch * 32 + g * 4;
    const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[2 * g]));
    const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[2 * g + 1]));
    const int half = ((in0 >> 2) & 1) ^ ((vc >> 2) & 1);
    *reinterpret_cast<float4*>(slot + (in0 >> 3) * 1024 + vc * 8 + half * 4) = float4{f0.x, f0.y, f1.x, f1.y};
  }
}
// Per-kernel smem structs: bodies take them BY REFERENCE so the fused
// dispatcher can union them in one pool (nvcc SUMS per-function static variables).
struct K1Smem {
  bf16 sgb[2][BT][K];  // [chunk-pair slot]: c+1's bf16 gates stage via
                       // cp.async during c's compute phases
  float sgc[BT][K];    // P2a output: this chunk's fp32 gate cumsum
  float sA[BT][BT];
  float sU[BT][2 * K + 4];
  float sb[BT];
  union SPool {
    struct {
      bf16 kw[BT][K + 8];
      bf16 qw[BT][K + 8];
      bf16 kz[10][16][K + 8];
    } a;
    struct {
      __align__(1024) bf16 kdT[K][64];
      __align__(1024) bf16 pTt[K][64];
      __align__(1024) bf16 u0Th[K][64];
      __align__(1024) bf16 u0Tl[K][64];
    } b;
    struct {
      __align__(1024) float sLt[16][128][8];
    } g;  // compose B:
          // fp32 L as SWZ32 k-major K=8 chunks (tf32 smem descs)
    struct {
      bf16 aC_h[BT][56];
      bf16 aC_l[BT][56];  // A cols<48 hi/lo
      bf16 upT_h[2 * K][56];
      bf16 upT_l[2 * K][56];
    } f;
  } sp;
  uint64_t mb_f;
  uint32_t s_taddr;
};
struct ChainSmem {
  __align__(1024) bf16 sPneg[2][2][64][64];  // [buf][ktile]
  __align__(1024) bf16 sKdT[2][128][64];
  __align__(1024) bf16 sQd[2][2][64][64];
  __align__(1024) bf16 sAh[2][64][64];
  __align__(1024) bf16 sAl[2][64][64];
  // u0 chunk tile via TMA riding mb_tma (measured: this stage was memory-
  // latency-bound on u0's 16 fp32 LDGs — a 100MB L2-spilling stream)
  __align__(1024) float sU0[2][BT][K];
  // mb_pre/mb_preC: prefix L/c TMA; mb_cmL/mb_cmC: prefix mma-batch commits
  uint64_t mb_tma, mb_p1, mb_p2, mb_o, mb_pre, mb_preC, mb_cmL, mb_cmC;
  uint32_t s_taddr;
};

// ---------------- RAW input transforms (the fused pre-pass) ----------------
// Two INDEPENDENT conventions, one per fla flag: RAW (q/k arrive
// un-normalized) folds fla's l2norm_fwd(q)/l2norm_fwd(k), BSIG (beta arrives
// as logits) folds its sigmoid(beta) — the caller's separate pre-pass launches
// — into the tiles k1 already loads. Both transforms reproduce the pre-pass's
// BYTES the way GM 1/2 reproduces its glog: fla's rstd is 1/sqrt(sum x^2 +
// 1e-6) with round-to-nearest sqrt/reciprocal (NOT the fast-math rsqrt) over
// an fp32 sum of bf16 squares, and its y lands in bf16, so every read
// re-rounds.
constexpr float L2_EPS = 1e-6f;  // fla.modules.l2norm.l2norm_fwd default

// {k, q} row reciprocal norms for a chunk's 64 token rows, ONE pass for both
// tensors: 512 threads = 8 lanes x 64 rows, 16 columns per lane (two 16 B
// loads), closed by 3 shuffles (a row's lanes share a warp); lane
// (tid & 7) == 0 publishes the row's pair to dst (the two rn sequences stay
// inside that branch — 7 of 8 lanes would throw them away). Pad rows load
// nothing and get rstd(0) — finite, and their values are selected to 0
// downstream anyway.
template <bool VL>
static __device__ __forceinline__ void row_rnorm(
    float2& dst, const bf16* __restrict__ q, const bf16* __restrict__ kk, size_t gbase, int H, int C_act, int tid) {
  const int r = tid >> 3;
  float sk = 0.f, sq = 0.f;
  if (!VL || r < C_act) {
    const size_t o = gbase + (size_t)r * H * K + (tid & 7) * 16;
    const uint4 xb[4] = {
        *reinterpret_cast<const uint4*>(kk + o),
        *reinterpret_cast<const uint4*>(kk + o + 8),
        *reinterpret_cast<const uint4*>(q + o),
        *reinterpret_cast<const uint4*>(q + o + 8)};
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xb);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float2 kf = __bfloat1622float2(x2[e]);
      const float2 qf = __bfloat1622float2(x2[e + 8]);
      sk += kf.x * kf.x + kf.y * kf.y;
      sq += qf.x * qf.x + qf.y * qf.y;
    }
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1) {
    sk += __shfl_xor_sync(0xffffffffu, sk, m);
    sq += __shfl_xor_sync(0xffffffffu, sq, m);
  }
  if ((tid & 7) == 0) dst = float2{__frcp_rn(__fsqrt_rn(sk + L2_EPS)), __frcp_rn(__fsqrt_rn(sq + L2_EPS))};
}
// one element pair through fla's l2norm: x * rstd, rounded to bf16
static __device__ __forceinline__ __nv_bfloat162 l2_bf16(float2 x, float rst) {
  return __floats2bfloat162_rn(x.x * rst, x.y * rst);
}
static __device__ __forceinline__ float2 l2_round(float2 x, float rst) {
  return __bfloat1622float2(l2_bf16(x, rst));
}
// BSIG: beta logits -> fp32 sigmoid, bf16-rounded (the wrapper's own
// torch.sigmoid(blog.float()).to(bfloat16)); BSIG false = beta already
// activated, passed through
template <bool BSIG>
static __device__ __forceinline__ float beta_in(float b) {
  return BSIG ? __bfloat162float(__float2bfloat16(1.f / (1.f + __expf(-b)))) : b;
}

// ---------------- K1: factored-ratio chunk builds --------------------------
// Anchored 16-token sub-blocks: kw[i]=k_i*e^{g_i-a(si)} (<=1), and for each
// ordered pair (si>=sj) kz(si)[j]=k_j*e^{a(si)-g_j} (<=1). A and Aqk become
// plain bf16 mma products; the solve stays fp32 SIMT. 512 threads/CTA.
// VL: varlen instantiation — compile-time-gates every pad-row guard so the
// eqlen (VL=false) kernels keep their exact original code.
// GM: gate mode — 0 reads pre-made bf16 glog; 1/2 (raw softplus/safe_gate)
// stage RAW graw through the same bf16 slot and transform it IN PLACE once
// landed — under the prior chunk's tail mma wait (piece-first / tail-less
// chunks: after their own P1 wait) — so P2a reads all modes as GM == 0.
// RAW: q/k un-normalized — the l2norm pre-pass folds in here (see row_rnorm);
// q/k feed P2b and P2c only. BSIG: beta as logits, its sigmoid folded in at
// the beta read. Independent of each other and of GM.
// When pieceL is non-null the piece's per-chunk affine maps compose IN-TAIL
// (tf32 TMEM chain) and the final (L, c) maps land at pieceL/piecec[pidx].
template <bool VL = false, int GM = 0, bool RAW = false, bool BSIG = false>
__device__ static void k1_body(
    K1Smem& S,
    int job0,
    int njobs,
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    int pidx,
    // varlen piece coords (defaults = eqlen): global chunk c's tokens start
    // at c*BT + tokoff; rows >= tend - t0 are pad (zero-filled on load)
    int nc_tot = 0,
    int tokoff = 0,
    int tend = 0,
    // GM != 0 gate-transform inputs (production lb = -5.0)
    const float* __restrict__ a_log = nullptr,
    const float* __restrict__ dtb = nullptr,
    float lb = 0.f,
    // fused-tail pTt staging TMA: the negated-P global (the W mma's
    // MN-major B source; every pieceL-passing caller supplies it — the
    // NP == 1 k1_factors_mma launch runs pieceL == nullptr, no tail)
    const CUtensorMap* ptm = nullptr) {
  constexpr int SB = 16, NSB = BT / SB;  // 4 sub-blocks
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  auto& sA = S.sA;
  auto& sU = S.sU;
  auto& sb = S.sb;
  auto& sp = S.sp;
  auto& kw = sp.a.kw;
  auto& qw = sp.a.qw;
  auto& kz = sp.a.kz;
  auto& mb_f = S.mb_f;
  auto& s_taddr = S.s_taddr;
  // RAW: the row reciprocal norms park in sU's dead pad columns (the rhs
  // owns [0, 2K); 2K..2K+3 is the row-stride pad, written by nothing) — live
  // from the norm pass to P2c at zero smem cost
  auto rn2 = [&](int i) -> float2& { return *reinterpret_cast<float2*>(&sU[i][2 * K]); };
  if (pieceL) {  // fused-tail tmem/mbar once (re-init of a live mbar is
    // UB; re-alloc after relinquish faults)
    if (tid == 0) ptx::mbar_init(&mb_f, 1);
    if (warp == 0) ptx::tcgen05_alloc(ptx::to_shared(&s_taddr), 512);
  }
  int mbph = 0;       // mb_f phase counter (one wait per committed batch)
  bool pend = false;  // tf32 compose committed, wait deferred a chunk
  float bnext = 0.f;  // next chunk's beta (prefetched a chunk early)
  // deferred compose wait: must land before ANY union write (the compose's
  // B tiles overlay sp); by then the mmas long completed (~free)
  auto pend_wait = [&] {
    if (pend) {
      ptx::mbar_wait_parity(&mb_f, mbph & 1);
      ++mbph;
      pend = false;
    }
  };
  const int nc1 = VL ? nc_tot : T / BT;  // total chunks (varlen: sum nc_s)
  const int tse = VL ? tend : T;         // sequence end token
  // GM != 0 hoist: h is constant across a block's jobs (h-major runs)
  const float ga = GM != 0 ? expf(a_log[job0 / nc1]) : 0.f;
  // GM != 0: in-place transform of a LANDED raw tile -> the same bf16
  // glog bytes GM == 0 stages (fp32 transform + bf16 round == the former
  // P2a fused read, value-identical per element), so P2a reads every
  // mode as GM == 0. Stable softplus subsumes the thr=20 branch
  // (1+e^-20 == 1 in fp32). VL pad rows stay their zfill 0
  // (transform(0) != 0 but the pad algebra needs glog == 0 exactly).
  auto gate_xform = [&](bf16(&gt)[BT][K], int rows) {
    const float* dtr = dtb + (size_t)(job0 / nc1) * K;
    for (int p = tid * 8; p < BT * K; p += blockDim.x * 8) {
      if (VL && p / K >= rows) continue;  // pad rows stay zfill 0
      uint4* gp = reinterpret_cast<uint4*>(&gt[p / K][p % K]);
      uint4 g4 = *gp;
      __nv_bfloat162* g2 = reinterpret_cast<__nv_bfloat162*>(&g4);
      const float4 d0 = *reinterpret_cast<const float4*>(&dtr[p % K]);
      const float4 d1 = *reinterpret_cast<const float4*>(&dtr[p % K + 4]);
      const float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gw = __bfloat1622float2(g2[e]);
        float y[2];
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const float g = (x ? gw.y : gw.x) + dv[2 * e + x];
          y[x] =
              GM == 1 ? -ga * (fmaxf(g, 0.f) + __logf(1.f + __expf(-fabsf(g)))) : lb * (1.f / (1.f + __expf(-ga * g)));
        }
        g2[e] = __floats2bfloat162_rn(y[0], y[1]);
      }
      *gp = g4;
    }
  };
  // persistent job loop: job -> (c = job%nc1, h = job/nc1); the NEXT job's
  // glog stages via cp.async under THIS job's compute (1-deep pipeline)
  for (int sub = 0; sub < njobs; ++sub) {
    const int job = job0 + sub;
    const int c = job % nc1, h = job / nc1, t0 = c * BT + tokoff;
    const int C_act = VL ? min(BT, tse - t0) : BT;  // real rows (tail < BT)
    bf16(&sgb)[BT][K] = S.sgb[sub & 1];
    float (&sg)[BT][K] = S.sgc;
    if (sub == 0) {
      for (int p = tid; p < BT * K; p += blockDim.x)
        sgb[p / K][p % K] =
            !VL || p / K < C_act ? glog[(size_t)(t0 + p / K) * H * K + h * K + p % K] : __float2bfloat16(0.f);
    } else {
      ptx::cp_async_wait_pending(0);
    }
    if (tid < BT)  // beta: chunk 0 loads direct; later chunks read the
      // register prefetched below (its P1 load was an exposed stall)
      sb[tid] = sub == 0 ? (!VL || tid < C_act ? beta_in<BSIG>(beta[(size_t)(t0 + tid) * H + h]) : 0.f) : bnext;
    if (sub + 1 < njobs) {  // stage next job's gates (h-major decode)
      const int jn = job + 1;
      const int t1i = (jn % nc1) * BT + tokoff;
      const size_t t1 = (size_t)t1i;
      const int hn = jn / nc1;
      const int Cn = VL ? min(BT, tse - t1i) : BT;  // next chunk's rows
      bf16* dst = &S.sgb[(sub + 1) & 1][0][0];
      for (int p = tid; p < BT * K / 8; p += blockDim.x) {
        const bf16* src = glog + (t1 + p * 8 / K) * H * K + (size_t)hn * K + (p * 8) % K;
        if constexpr (VL)  // pad rows of a tail chunk stage as zeros
          ptx::cp_async_16_zfill(dst + p * 8, src, p * 8 / K >= Cn);
        else
          ptx::cp_async_16(dst + p * 8, src);
      }
      ptx::cp_async_commit();
      if (tid < BT) bnext = !VL || tid < Cn ? beta_in<BSIG>(beta[(t1 + tid) * H + hn]) : 0.f;
    }
    if constexpr (RAW)  // one row-norm pass per chunk-job, q and k together
      // (rides the gate cp.async; P2b/P2c re-read the same lines out of L1)
      row_rnorm<VL>(rn2(tid >> 3), q, kk, (size_t)t0 * H * K + (size_t)h * K, H, C_act, tid);
    __syncthreads();
    if constexpr (GM != 0)  // tiles no prior tail pre-transformed: the
      // piece-first chunk (and every chunk when the fused tail is absent
      // — no-pieceL callers)
      if (sub == 0 || !pieceL) {
        gate_xform(sgb, C_act);
        __syncthreads();
      }
    {  // split cumsum: 512 threads = 4 x 16-row segments per column
      // (bf16 stage widens on read; the fp32 running sums land in sgc.
      // GM != 0 tiles were transformed in place on landing, so every
      // mode reads pre-made bf16 glog here)
      const int col = tid & (K - 1), r0 = (tid >> 7) * (BT / 4);
      float acc = 0.f;
      for (int r = r0; r < r0 + BT / 4; ++r) {
        acc += __bfloat162float(sgb[r][col]);
        sg[r][col] = acc;
      }
    }
    __syncthreads();
    {  // per-thread column is fixed: the carry-in loads once per segment
      // (the in-place += stores block the compiler from hoisting it)
      const int colc = tid & (K - 1), rc = tid >> 7;
      for (int seg = 1; seg < 4; ++seg) {
        const float carry = sg[seg * (BT / 4) - 1][colc];
#pragma unroll
        for (int it = 0; it < 4; ++it)
          sg[seg * (BT / 4) + rc + it * 4][colc] += carry;
        __syncthreads();
      }
    }
    // anchors a(s) = gamma BEFORE sub-block s (0 for s=0). A thread's
    // column PAIR is fixed across every P2b/P2c position, so the anchors,
    // e^{a(s)} and the chunk-end row hoist to registers; e^{sg} then
    // composes as e^{sg-a(s)}*e^{a(s)} (both factors already anchored
    // quantities; exact for s==0, <=1 extra fp32 rounding else)
    const int colp = (tid & (K / 2 - 1)) * 2;
    float a0v[NSB], a1v[NSB], ea0[NSB], ea1[NSB];
    a0v[0] = a1v[0] = 0.f;
    ea0[0] = ea1[0] = 1.f;
#pragma unroll
    for (int s = 1; s < NSB; ++s) {
      a0v[s] = sg[s * SB - 1][colp];
      a1v[s] = sg[s * SB - 1][colp + 1];
      ea0[s] = __expf(a0v[s]);
      ea1[s] = __expf(a1v[s]);
    }
    // build kw, qw, kappa/rhs, qdec
    {  // kw/qw/qdec/kappa/v + the kdec value (into kz-area staging: the
      // transposed [col][i] global store below keeps both sides coalesced)
      // all of a half-batch's global loads issue FIRST, then compute+store
      bf16(*stg)[K + 4] = reinterpret_cast<bf16(*)[K + 4]>(&kz[0][0][0]);
      // column-PAIRED (bf162 loads/stores: half the transactions, twice
      // the in-flight bytes per MSHR)
      const float gl0 = sg[BT - 1][colp], gl1 = sg[BT - 1][colp + 1];
      // running pointers: a thread's taps advance a FIXED 8 rows, so one
      // add per tensor replaces the per-tap 64-bit gp rebuild (the SASS
      // census's top P2b term, ~9 int ops/load; same addresses bit-exact)
      const size_t gp0 = (size_t)(t0 + tid * 2 / K) * H * K + h * K + tid * 2 % K;
      const size_t gst = (size_t)8 * H * K;  // +8 rows per tap
      const bf16* pk = &kk[gp0];
      const bf16* pq = &q[gp0];
      const bf16* pv = &v[gp0];
      bf16* qd = &qdec[((size_t)c * H + h) * BT * K + (tid >> 6) * K + colp];
      // deferred-compose wait at the last union-independent point: the
      // anchor/pointer setup above reads only sgc/registers; the kw
      // store below is the chunk's first union (sp) write
      pend_wait();
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        float2 kv[4], qv[4], vv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (VL && tid * 2 / K + (hb * 4 + j) * 8 >= C_act) {
            kv[j] = qv[j] = vv[j] = float2{0.f, 0.f};  // pad rows
          } else {
            kv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pk));
            qv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pq));
            vv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pv));
            if constexpr (RAW) {  // l2norm, in fla's bf16 bytes
              const float2 r = rn2((tid >> 6) + (hb * 4 + j) * 8);
              kv[j] = l2_round(kv[j], r.x);
              qv[j] = l2_round(qv[j], r.y);
            }
          }
          pk += gst;
          pq += gst;
          pv += gst;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (tid >> 6) + (hb * 4 + j) * 8;
          const int si = hb * 2 + (j >> 1);  // == i / SB (tid < 512)
          const float ei0 = __expf(sg[i][colp] - a0v[si]);
          const float ei1 = __expf(sg[i][colp + 1] - a1v[si]);
          const float kw0 = kv[j].x * ei0, kw1 = kv[j].y * ei1;
          *reinterpret_cast<__nv_bfloat162*>(&kw[i][colp]) = __floats2bfloat162_rn(kw0, kw1);
          const float qw0 = qv[j].x * ei0 * scale;
          const float qw1 = qv[j].y * ei1 * scale;
          *reinterpret_cast<__nv_bfloat162*>(&qw[i][colp]) = __floats2bfloat162_rn(qw0, qw1);
          const float bi = sb[i];                     // beta folded into the solve rhs
          *reinterpret_cast<float2*>(&sU[i][colp]) =  // kappa
              float2{kw0 * (bi * ea0[si]), kw1 * (bi * ea1[si])};
          *reinterpret_cast<float2*>(&sU[i][K + colp]) = float2{vv[j].x * bi, vv[j].y * bi};
          *reinterpret_cast<__nv_bfloat162*>(qd) = __floats2bfloat162_rn(qw0 * ea0[si], qw1 * ea1[si]);
          qd += 8 * K;  // i advances 8 rows per tap
          *reinterpret_cast<__nv_bfloat162*>(&stg[i][colp]) =
              __floats2bfloat162_rn(kv[j].x * __expf(gl0 - sg[i][colp]), kv[j].y * __expf(gl1 - sg[i][colp + 1]));
        }
      }
      __syncthreads();
      // b64 quads: 4 adjacent tokens of one channel ([col][i] layout).
      // compile-time trip (512 threads, like the tap loop above) => the
      // per-iteration %/÷ and 64-bit address rebuilds become immediates
      const int rr = tid * 4 % BT, cc = tid * 4 / BT;
      bf16* kd = &kdec[((size_t)c * H + h) * BT * K + tid * 4];
#pragma unroll
      for (int n = 0; n < BT * K / 2048; ++n) {
        const int cn = cc + n * 32;
        alignas(8) const __nv_bfloat162 p2[2] = {{stg[rr][cn], stg[rr + 1][cn]}, {stg[rr + 2][cn], stg[rr + 3][cn]}};
        *reinterpret_cast<float2*>(kd + n * 2048) = *reinterpret_cast<const float2*>(p2);
      }
    }
    __syncthreads();
    // kz tiles: a column's pairs (si, sj) share their exponent down si —
    // one anchored base exp e^{a(sj)-sg[j]} per (sj, row), then multiply
    // by the adjacent-anchor gaps f(s) = e^{a(s)-a(s-1)} (the e^{x-a}*e^{a-y}
    // composition in fp32; base tiles bit-exact, derived tiles carry <=3
    // extra fp32 roundings pre-bf16). The 8 dedup'd row loads issue first.
    {
      const int u = tid >> 6;  // row-in-block base; cols = colp pair
      float f0[NSB - 1], f1[NSB - 1];
#pragma unroll
      for (int s = 1; s < NSB; ++s) {
        f0[s - 1] = __expf(a0v[s] - a0v[s - 1]);
        f1[s - 1] = __expf(a1v[s] - a1v[s - 1]);
      }
      __nv_bfloat162 kvz[NSB][2];
      // running pointer: the 8 taps advance a fixed 8 rows (sj*SB + rr*8
      // = 0,8,..,56), so one 64-bit add per tap replaces the per-tap
      // address rebuild (same addresses bit-exact; the P2b diet pattern)
      const bf16* pz = &kk[(size_t)(t0 + u) * H * K + h * K + colp];
#pragma unroll
      for (int sj = 0; sj < NSB; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8;
          kvz[sj][rr] =
              !VL || j < C_act ? *reinterpret_cast<const __nv_bfloat162*>(pz) : __floats2bfloat162_rn(0.f, 0.f);
          if constexpr (RAW)  // l2norm at the load: the tiles below
            // then see the same bytes P2b's kw did
            kvz[sj][rr] = l2_bf16(__bfloat1622float2(kvz[sj][rr]), rn2(j).x);
          pz += (size_t)8 * H * K;
        }
#pragma unroll
      for (int sj = 0; sj < NSB; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8, jj = u + rr * 8;
          const float2 kf = __bfloat1622float2(kvz[sj][rr]);
          // pad rows select 0 OUTRIGHT: their base exponent a(sj) -
          // sg[j] spans block start -> seq end (up to 63 rows of gate
          // mass vs <= 15 for real rows) and 0 * __expf(overflow) = NaN
          const bool real = !VL || j < C_act;
          float v0 = real ? kf.x * __expf(a0v[sj] - sg[j][colp]) : 0.f;
          float v1 = real ? kf.y * __expf(a1v[sj] - sg[j][colp + 1]) : 0.f;
#pragma unroll
          for (int si = sj; si < NSB; ++si) {
            if (si > sj) {
              v0 *= f0[si - 1];
              v1 *= f1[si - 1];
            }
            *reinterpret_cast<__nv_bfloat162*>(&kz[si * (si + 1) / 2 + sj][jj][colp]) = __floats2bfloat162_rn(v0, v1);
          }
        }
    }
    if (tid < K / 4) {  // stored as e^{gC}: compile-time trip (512
      // threads) + one float4 per thread (was a runtime striding loop)
      const float4 g4 = *reinterpret_cast<const float4*>(&sg[BT - 1][tid * 4]);
      *reinterpret_cast<float4*>(&gC[((size_t)c * H + h) * K + tid * 4]) =
          float4{__expf(g4.x), __expf(g4.y), __expf(g4.z), __expf(g4.w)};
    }
    __syncthreads();
    // warps 10-15 are idle through the pair mma: L2-prefetch the inputs of
    // the CTA one wave ahead on this SM (linear bid + 148; same h while the
    // c-range allows) — its P1/P2 loads then hit L2 instead of DRAM
    if (warp >= 10) {
      const int jn = job + 1;  // the next persistent job
      if (jn < nc1 * H && sub + 1 < njobs) {
        const int t1p = (jn % nc1) * BT + tokoff;
        const int Cp = VL ? min(BT, tse - t1p) : BT;  // stop at seq end
        const size_t tb = (size_t)t1p * H * K + (size_t)(jn / nc1) * K;
        const char* pq = reinterpret_cast<const char*>(q + tb);
        const char* pk = reinterpret_cast<const char*>(kk + tb);
        const char* pv = reinterpret_cast<const char*>(v + tb);
        const char* pg = reinterpret_cast<const char*>(glog + tb);
        for (int i = tid - 320; i < Cp; i += 192) {
          const size_t ro = (size_t)i * H * K * 2;
          for (int l = 0; l < 256; l += 128) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pq + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pk + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pv + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pg + ro + l));
          }
        }
      }
    }
    // A and Aqk via mma: each warp does pairs round-robin; per pair:
    // [16x16] = kw_si[16x128] @ kz_pi^T  (and qw for Aqk)
    for (int pi = warp; pi < NSB * (NSB + 1) / 2; pi += 16) {
      int si = 0;
      while ((si + 1) * (si + 2) / 2 <= pi)
        ++si;
      const int sj = pi - si * (si + 1) / 2;
      float4 acck[2], accq[2];
      acck[0] = acck[1] = accq[0] = accq[1] = float4{0, 0, 0, 0};
      const int arow = lane & 15, aka = lane >> 4;
#pragma unroll
      for (int k16 = 0; k16 < K / 16; ++k16) {
        uint32_t a0, a1, a2, a3, q0, q1, q2, q3;
        ptx::ldmatrix_x4_b16(ptx::to_shared(&kw[si * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
        ptx::ldmatrix_x4_b16(ptx::to_shared(&qw[si * SB + arow][k16 * 16 + aka * 8]), q0, q1, q2, q3);
#pragma unroll
        for (int n8 = 0; n8 < 2; ++n8) {
          uint32_t b0, b1;
          ptx::ldmatrix_x2_b16(ptx::to_shared(&kz[pi][(lane & 7) + n8 * 8][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
          ptx::mma_m16n8k16_bf16f32(acck[n8], a0, a1, a2, a3, b0, b1);
          ptx::mma_m16n8k16_bf16f32(accq[n8], q0, q1, q2, q3, b0, b1);
        }
      }
      // scatter A fragments to sA (masked, beta-scaled); Aqk fragments go
      // STRAIGHT to global hi/lo as bf162 pairs (host zero-fills once; a
      // diagonal-straddling pair re-writes the host zero into its
      // jj+1 > ii half — bit-exact vs never-written)
      const int r = lane >> 2, c2 = (lane & 3) * 2;
      const size_t abase = ((size_t)c * H + h) * BT * BT;
#pragma unroll
      for (int n8 = 0; n8 < 2; ++n8) {
        const float vals[4] = {acck[n8].x, acck[n8].y, acck[n8].z, acck[n8].w};
        const float valq[4] = {accq[n8].x, accq[n8].y, accq[n8].z, accq[n8].w};
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {  // fragment row: cols jj, jj+1
          const int ii = si * SB + r + e2 * 8;
          const int jj = sj * SB + n8 * 8 + c2;
          sA[ii][jj] = (jj < ii) ? sb[ii] * vals[e2 * 2] : 0.f;
          sA[ii][jj + 1] = (jj + 1 < ii) ? sb[ii] * vals[e2 * 2 + 1] : 0.f;
          if (jj <= ii) {
            const float q0 = valq[e2 * 2];
            const float q1 = jj + 1 <= ii ? valq[e2 * 2 + 1] : 0.f;
            const __nv_bfloat162 ah{__float2bfloat16(q0), __float2bfloat16(q1)};
            *reinterpret_cast<__nv_bfloat162*>(&aqk_h[abase + ii * BT + jj]) = ah;
            *reinterpret_cast<__nv_bfloat162*>(&aqk_l[abase + ii * BT + jj]) =
                __floats2bfloat162_rn(q0 - __bfloat162float(ah.x), q1 - __bfloat162float(ah.y));
          }
        }
      }
    }
    __syncthreads();
    // (beta pre-folded into the rhs fills)
    pend_wait();  // the sp.f pack below is the next union write
    // A hi/lo operand copy (coupling uses cols < 48 only) — oracle vet:
    // hi/lo tensor-core coupling errs 8.7e-9 at the final state
    for (int p2 = tid; p2 < BT * 24; p2 += blockDim.x) {
      const int p = p2 * 2;
      const int i = p / 48, j = p % 48;
      // pairs never produce sA above the block diagonal and the coupling
      // mma never reads it (A cols k16 < b): pack zeros without the sA
      // round-trip
      const float2 av = j / SB > i / SB ? float2{0.f, 0.f} : *reinterpret_cast<const float2*>(&sA[i][j]);
      const bf16 ah0 = __float2bfloat16(av.x);
      const bf16 ah1 = __float2bfloat16(av.y);
      *reinterpret_cast<__nv_bfloat162*>(&sp.f.aC_h[i][j]) = __nv_bfloat162{ah0, ah1};
      *reinterpret_cast<__nv_bfloat162*>(&sp.f.aC_l[i][j]) =
          __floats2bfloat162_rn(av.x - __bfloat162float(ah0), av.y - __bfloat162float(ah1));
    }
    __syncthreads();
    // Blocked forward solve: cross-block coupling as mma (3 hi/lo products,
    // every warp 2 n8-tiles), triangular block fp32 thread-per-column;
    // solved rows publish transposed hi/lo as the next coupling's B.
#pragma unroll
    for (int b = 0; b < NSB; ++b) {
      if (b) {
        const int arow = lane & 15, aka = lane >> 4;
        float4 acc[2];
        acc[0] = acc[1] = float4{0, 0, 0, 0};
        for (int pr = 0; pr < 3; ++pr) {
          const bf16(*ta)[56] = pr == 1 ? sp.f.aC_l : sp.f.aC_h;
          const bf16(*tb)[56] = pr == 2 ? sp.f.upT_l : sp.f.upT_h;
          for (int k16 = 0; k16 < b; ++k16) {
            uint32_t a0, a1, a2, a3;
            ptx::ldmatrix_x4_b16(ptx::to_shared(&ta[b * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
#pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
              uint32_t b0, b1;
              ptx::ldmatrix_x2_b16(
                  ptx::to_shared(&tb[warp * 16 + n8 * 8 + (lane & 7)][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
              ptx::mma_m16n8k16_bf16f32(acc[n8], a0, a1, a2, a3, b0, b1);
            }
          }
        }
        const int fr = lane >> 2, fc = (lane & 3) * 2;
#pragma unroll
        for (int n8 = 0; n8 < 2; ++n8) {
          const float vals[4] = {acc[n8].x, acc[n8].y, acc[n8].z, acc[n8].w};
#pragma unroll
          for (int e = 0; e < 4; ++e)
            sU[b * SB + fr + (e >> 1) * 8][warp * 16 + n8 * 8 + fc + (e & 1)] -= vals[e];
        }
      }
      __syncthreads();
      if (tid < 2 * K) {
        const int col = tid;
        float r[SB];
#pragma unroll
        for (int i = 0; i < SB; ++i)
          r[i] = sU[b * SB + i][col];
#pragma unroll
        for (int i = 1; i < SB; ++i)
#pragma unroll
          for (int j = 0; j < i; ++j)
            r[i] -= sA[b * SB + i][b * SB + j] * r[j];
#pragma unroll
        for (int i = 0; i < SB; ++i)
          sU[b * SB + i][col] = r[i];
        if (b + 1 < NSB)  // next coupling's B operand (hi/lo^T): 8 rows
                          // pack into ONE 16B store per array (rows are
                          // 112B-strided, so b*SB+i lands 16B-aligned)
#pragma unroll
          for (int i = 0; i < SB; i += 8) {
            alignas(16) __nv_bfloat162 hv[4], lv[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
              const float e0 = r[i + 2 * w], e1 = r[i + 2 * w + 1];
              hv[w] = __nv_bfloat162{__float2bfloat16(e0), __float2bfloat16(e1)};
              lv[w] = __floats2bfloat162_rn(e0 - __bfloat162float(hv[w].x), e1 - __bfloat162float(hv[w].y));
            }
            *reinterpret_cast<uint4*>(&sp.f.upT_h[col][b * SB + i]) = *reinterpret_cast<const uint4*>(hv);
            *reinterpret_cast<uint4*>(&sp.f.upT_l[col][b * SB + i]) = *reinterpret_cast<const uint4*>(lv);
          }
      }
      __syncthreads();
    }
    const size_t base = ((size_t)c * H + h) * BT * K;
    for (int p2 = tid; p2 < BT * K / 2; p2 += blockDim.x) {
      const int p = p2 * 2;
      *reinterpret_cast<__nv_bfloat162*>(&P[base + p]) =
          __floats2bfloat162_rn(-sU[p / K][p % K], -sU[p / K][p % K + 1]);
      *reinterpret_cast<float2*>(&u0[base + p]) = float2{sU[p / K][K + p % K], sU[p / K][K + p % K + 1]};
    }
    if (!pieceL) continue;
    // ---- fused W-form products + IN-TAIL RUNNING COMPOSITION ----
    // Per chunk: W and w0^T products (w0's operands SWAPPED vs the drained
    // form — A=u0T, B=kdT — so its product lands in state orientation), then
    // the piece's running maps compose as a NO-DRAIN tf32 TMEM chain:
    // Lrun/crun are fp32 D pairs ping-ponging between [0,256) and [256,512).
    // The chunk's W/w0 products target the FREE pair; W alone is drained
    // (L = diag(e^gC) - W, staged as fp32 SWZ32 tiles); then Lrun_new =
    // Lrun @ L-tiles overwrites the drained W slot and crun_new = crun @
    // L-tiles accumulates onto the completed w0 product (the c fold).
    // ONE commit+wait per chunk (the products'); the compose commit is
    // waited a chunk LATE (pend_wait, before its B-tile smem is rewritten);
    // piece-end chunks wait in-chunk and drain the final maps to global.
    __syncthreads();  // everyone past the P/u0 reads of sU
    // tail pTt: TMA the L2-hot negated-P copy this CTA stored above — the
    // [tok][kc] global IS the W mma's MN-major B image (probe-verified: SWZ128
    // box == the swz128b staging bytes; 2 boxes = the 2 MN atoms at LBO
    // 8192). Bytes are the former in-smem pack's sign-flipped, so the
    // product lands -W; the L drain folds the sign exactly. The mb_f
    // phase is waited by ALL threads before the W mma.
    if (tid == 0) {
      ptx::fence_async_global();  // P sts -> async proxy (TMA)
      ptx::mbar_arrive_expect_tx(&mb_f, uint32_t(sizeof(sp.b.pTt)));
      ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sp.b.pTt[0][0]), ptm, 0, (c * H + h) * BT, &mb_f);
      ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sp.b.pTt[64][0]), ptm, 64, (c * H + h) * BT, &mb_f);
    }
    const size_t cbase = ((size_t)c * H + h) * BT * K;
#pragma unroll
    for (int hb = 0; hb < 2; ++hb) {  // kdec loads issue first (MLP)
      bf16 kdv[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        kdv[j] = kdec[cbase + tid + (hb * 8 + j) * 512];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int p = tid + (hb * 8 + j) * 512;
        const int kc = p / BT, tok = p % BT;
        // 128B-swizzle placement (16B atoms XOR row): swz128 desc form
        const int tokS = (((tok >> 3) ^ (kc & 7)) << 3) | (tok & 7);
        sp.b.kdT[kc][tokS] = kdv[j];
        const float uv = sU[tok][K + kc];
        const bf16 uh = __float2bfloat16(uv);
        sp.b.u0Th[kc][tokS] = uh;
        sp.b.u0Tl[kc][tokS] = __float2bfloat16(uv - __bfloat162float(uh));
      }
    }
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    const uint32_t taddr = s_taddr;
    const uint32_t idw = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
        128, 128, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);
    // TMA-fed pTt is the [tok][kc] P image: B is MN-major (idesc bit 16)
    const uint32_t idwm = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
        128, 128, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::MN);
    const uint32_t tp = (sub & 1) ? 256u : 0u;  // this chunk's pair
    // pTt TMA landed. ALL threads consume the phase: skipping an open
    // phase would alias the next parity wait against the previous completed
    ptx::mbar_wait_parity(&mb_f, mbph & 1);
    ++mbph;
    if (tid == 0) {
      ptx::tcgen05_fence_after_thread_sync();
      // W-D at tp+[0,128): -W[kc_out][kc_in] = kdecT @ P via MN-major B
#pragma unroll
      for (int k16 = 0; k16 < BT / 16; ++k16) {
        const uint64_t da = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(&sp.b.kdT[0][0]) + k16 * 32);
        const uint64_t db =
            ptx::mma_smem_desc_mn_major<uint16_t, 64, 128, 128>(ptx::to_shared(&sp.b.pTt[0][0]) + k16 * 2048);
        ptx::tcgen05_mma_f16(taddr + tp, da, db, idwm, k16 ? 1u : 0u);
      }
      // w0^T-D at tp+[128,256): w0^T[vc][kc_out] = u0T @ kdT^T (hi + lo
      // A) — this slot IS crun's seed / c-fold addend
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int k16 = 0; k16 < BT / 16; ++k16) {
          const uint64_t da = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(
              ptx::to_shared(half ? &sp.b.u0Tl[0][0] : &sp.b.u0Th[0][0]) + k16 * 32);
          const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(&sp.b.kdT[0][0]) + k16 * 32);
          ptx::tcgen05_mma_f16(taddr + tp + 128u, da, db, idw, (half | k16) ? 1u : 0u);
        }
      ptx::tcgen05_commit_arrive(&mb_f);
    }
    if constexpr (GM != 0)  // next chunk's RAW gates landed long ago (P1
      // cp.async): transform them here, under the W/w0 mma wait, off the
      // serial cumsum path. Each thread rewrites exactly the bytes it
      // staged, so its own-group wait suffices; the next chunk's P2a
      // read is barriers away (P7/P1 syncthreads).
      if (sub + 1 < njobs) {
        ptx::cp_async_wait_pending(0);
        const int t1x = ((job + 1) % nc1) * BT + tokoff;
        gate_xform(S.sgb[(sub + 1) & 1], VL ? min(BT, tse - t1x) : BT);
      }
    ptx::mbar_wait_parity(&mb_f, mbph & 1);
    ++mbph;
    ptx::tcgen05_fence_after_thread_sync();
    const int band = (warp & 3) * 32, ch = warp >> 2;  // 16 warps: 32-col ch
    const uint32_t lane_hi = uint32_t(band) << 16;
    // W drain -> L = diag(e^gC) - W, staged as fp32 SWZ32 K=8-chunk tiles
    // (the tf32 compose's B; b spans exactly slice ch*4+b). SWZ32 row form:
    // a row's two 16B halves swap on (out>>2)&1.
    // The w0 product is NOT drained — it stays in TMEM as crun's addend.
    for (int b = 0; b < 4; ++b) {
      uint32_t r[8];
      ptx::tcgen05_ld_32x32b_x8(
          taddr + lane_hi + (tp + uint32_t(ch * 32 + b * 8)), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
      ptx::tcgen05_wait_ld();
      float lt[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int kin = ch * 32 + b * 8 + j;
        // the mma drained -W (sign-flipped B bytes): a + (-x) == a - x
        // exactly (IEEE), so L is byte-identical to the former pack's
        lt[j] = (band + lane == kin ? gC[((size_t)c * H + h) * K + kin] : 0.f) + __int_as_float(r[j]);
      }
      const bool sw = ((band + lane) >> 2) & 1;
      const float4 lo = float4{lt[0], lt[1], lt[2], lt[3]};
      const float4 hi = float4{lt[4], lt[5], lt[6], lt[7]};
      float4* dst = reinterpret_cast<float4*>(&sp.g.sLt[ch * 4 + b][band + lane][0]);
      dst[0] = sw ? hi : lo;
      dst[1] = sw ? lo : hi;
    }
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();  // sLt complete for the compose's B descs
    {
      const int vc = band + lane;
      auto drain_reg = [&](uint32_t dc, float* r2) {  // fp32 D region -> regs
#pragma unroll
        for (int b2 = 0; b2 < 4; ++b2) {
          uint32_t r[8];
          ptx::tcgen05_ld_32x32b_x8(
              taddr + lane_hi + dc + uint32_t(ch * 32 + b2 * 8), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
          ptx::tcgen05_wait_ld();
#pragma unroll
          for (int j = 0; j < 8; ++j)
            r2[b2 * 8 + j] = __int_as_float(r[j]);
        }
      };
      auto store_maps = [&](const float* lr2, const float* cr2) {
        const size_t sb2 = (size_t)pidx * K * K;
        // both [out][in] row-major. L bf16 (half its prefix TMA bytes;
        // single-bf16 L is the gate-proven original convention): the prefix
        // TMAs SWZ128 tiles and widens to the tf32 SWZ32 B layout
        // (widen_map_tf32). c stays fp32 (SWZ32-TMA'd directly): bf16 c
        // accumulated past the fla o bar at 16K (1.05e-3 vs 7.2e-4).
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          pieceL[sb2 + (size_t)(ch * 32 + j) * K + vc] = __float2bfloat16(lr2[j]);
          piecec[sb2 + (size_t)(ch * 32 + j) * K + vc] = cr2[j];
        }
      };
      const bool piece_end = sub + 1 == njobs;
      if (sub == 0) {
        // seed: crun = the w0^T product (already sitting in tp+128, state-
        // oriented); Lrun = L^T via a transposed (un-swizzling) tile re-read
        alignas(16) float lr[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          const int out = ch * 32 + j;
          lr[j] = sp.g.sLt[vc >> 3][out][(vc & 7) ^ (((out >> 2) & 1) << 2)];
        }
        if (piece_end) {  // 1-chunk piece: the seed IS its map
          alignas(16) float cr[32];
          drain_reg(tp + 128u, cr);
          store_maps(lr, cr);
        } else {
          // st over the drained W slot — safe: the products' wait above
          // retired every mma (st vs in-flight A-reads is UNordered)
#pragma unroll
          for (int c4 = 0; c4 < 32; c4 += 4)
            ptx::tcgen05_st_32x32b_x4(
                taddr + lane_hi + (tp + uint32_t(ch * 32 + c4)),
                __float_as_int(lr[c4]),
                __float_as_int(lr[c4 + 1]),
                __float_as_int(lr[c4 + 2]),
                __float_as_int(lr[c4 + 3]));
          ptx::tcgen05_wait_st();
        }
      } else {
        // no-drain tf32 compose: D(tp) = Lrun @ L-tiles (scale 0 overwrites
        // the drained W), D(tp+128) = crun @ L-tiles + w0 (scale 1 onto the
        // completed product). A = the other pair's fp32 D, read as tf32;
        // no wait here — the commit is consumed by pend_wait / piece end.
        if (tid == 0) {
          const uint32_t idt = ptx::mma_inst_desc_tf32(128, K);
          const uint32_t sp2 = tp ^ 256u;
          for (int st2 = 0; st2 < 2; ++st2)
#pragma unroll
            for (int s = 0; s < 16; ++s)
              mma_tf32_atmem(
                  taddr + tp + uint32_t(st2 * 128),
                  taddr + (sp2 + uint32_t(st2 * 128 + 8 * s)),
                  ptx::mma_smem_desc_k_major<uint32_t, 8, 32>(ptx::to_shared(&sp.g.sLt[s][0][0])),
                  idt,
                  (st2 | s) ? 1u : 0u);
          ptx::tcgen05_commit_arrive(&mb_f);
        }
        if (piece_end) {  // the piece's ONE remaining drain: final maps
          ptx::mbar_wait_parity(&mb_f, mbph & 1);
          ++mbph;
          ptx::tcgen05_fence_after_thread_sync();
          alignas(16) float lr[32], cr[32];
          drain_reg(tp, lr);
          drain_reg(tp + 128u, cr);
          store_maps(lr, cr);
        } else {
          pend = true;
        }
      }
    }
    if (sub + 1 < njobs) __syncthreads();  // slot reuse across jobs
  }  // job loop
  if (pieceL) {
    if (warp == 0) ptx::tcgen05_dealloc(s_taddr, 512);
    ptx::tcgen05_relinquish();
  }
}

template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512) k1_factors_mma(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    int NP) {
  __shared__ K1Smem S;
  // grid = NP pieces per head x H heads; piece p covers len consecutive
  // chunks of ONE head (the running composition needs these runs).
  const int nc1 = T / BT, bid = blockIdx.x;
  const int h = bid / NP, pc = bid % NP;
  const int c00 = piece_c0(pc, nc1, NP);
  const int njobs = piece_c0(pc + 1, nc1, NP) - c00;
  k1_body<false, GM, RAW, BSIG>(
      S,
      h * nc1 + c00,
      njobs,
      q,
      kk,
      v,
      glog,
      beta,
      T,
      H,
      scale,
      P,
      u0,
      kdec,
      qdec,
      aqk_h,
      aqk_l,
      gC,
      pieceL,
      piecec,
      pc * H + h,
      0,
      0,
      0,
      a_log,
      dtb,
      lb);
}
// ---------------- K2: tcgen05 chain (A-from-TMEM, B via TMA) --------------
// Per-piece CTA walks its chunks serially. State S^T lives in registers —
// thread (band = (warp&3)*32, ch = warp>>2) owns S^T[vc = band+lane][kc in
// ch*32 + 0..31] — and is re-staged per chunk as a packed SINGLE-bf16 TMEM
// A region (precision study: the bf16 B operands, not the S read, set the error
// floor — hi/lo S buys <= 1.27x in the kernel ctx). Products (A-from-TMEM,
// B = TMA 128B-swizzled K-major tiles):
//   P1: U^T[vc,tok]  = S^T @ Pneg^T   (M=128 N=64  K=128; +u0 in drain)
//   P2: S^T[vc,kc]  += U^T @ kdecT^T  (M=128 N=128 K=64;  decay in drain)
// o is fused: P3 (o^T += S^T@qdec^T) rides phase 1; P4 (o^T += u^T@Aqk^T)
// rides phase 2. TMEM cols: [0,64) P1-D | [64,192) P2-D | [192,224) A2hi |
// [224,256) A2lo | [256,320) A1 | [384,448) o-D.
// SELF-START: the piece's start state composes from h0 through the prefix
// pieces' (L, c) maps as a NO-DRAIN tf32 TMEM chain, gated on pflags.
template <bool VL = false>  // varlen: pad-row masks compile-time-gated
__device__ static void chain_body(
    ChainSmem& S,
    int h,
    int sg,
    int nseg,
    const CUtensorMap& pneg_map,
    const CUtensorMap& kdt_map,
    const CUtensorMap& qd_map,
    const CUtensorMap& aqh_map,
    const CUtensorMap& aql_map,
    const CUtensorMap& u0f_map,
    const float* __restrict__ gC,
    int n_chunks,
    int H,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    // per-chunk states (nullptr = off): row = GLOBAL chunk c, [c][h][K][V] or
    // [c][h][V][K] (hpc_v_first), fp32 or bf16 (hpc_bf16)
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    const CUtensorMap* sl_map,  // fp32 piece L maps
    const CUtensorMap* sc_map,  // fp32 piece offsets
    const float* __restrict__ h0s,
    const uint32_t* __restrict__ pflags = nullptr,  // fused: piece-done
    // varlen (defaults = eqlen): sg/nseg/n_chunks are SEQUENCE-local; cbase/
    // pbase are the sequence's global chunk/piece bases; o rows live at
    // c*BT + tokoff + tok and rows >= tend are pad (masked, never stored)
    int cbase = 0,
    int pbase = 0,
    int tokoff = 0,
    int tend = 0) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int band = (warp & 3) * 32, ch = warp >> 2;  // 16 warps: ch = quarter
  const int vc = band + lane;
  auto& sPneg = S.sPneg;
  auto& sKdT = S.sKdT;
  auto& sQd = S.sQd;
  auto& sAh = S.sAh;
  auto& sAl = S.sAl;
  auto& sU0 = S.sU0;
  auto& mb_tma = S.mb_tma;
  auto& mb_p1 = S.mb_p1;
  auto& mb_p2 = S.mb_p2;
  auto& mb_o = S.mb_o;
  auto& s_taddr = S.s_taddr;
  if (tid == 0) {
    ptx::mbar_init(&mb_tma, 1);
    ptx::mbar_init(&mb_p1, 1);  // spine: P1 only
    ptx::mbar_init(&mb_p2, 1);  // spine: P2 only
    ptx::mbar_init(&mb_o, 2);   // o-pipe: P3 + P4
    ptx::mbar_init(&S.mb_pre, 1);
    ptx::mbar_init(&S.mb_preC, 1);
    ptx::mbar_init(&S.mb_cmL, 1);
    ptx::mbar_init(&S.mb_cmC, 1);
    ptx::prefetch_tensormap(&pneg_map);
    ptx::prefetch_tensormap(&kdt_map);
  }
  if (warp == 0) ptx::tcgen05_alloc(ptx::to_shared(&s_taddr), 512);
  __syncthreads();
  const uint32_t taddr = s_taddr;
  const uint32_t lane_hi = uint32_t(band) << 16;
  // ascending piece boundaries (shared with the builders)
  const int c0 = cbase + piece_c0(sg, n_chunks, nseg);
  const int per_seg = cbase + piece_c0(sg + 1, n_chunks, nseg) - c0;
  const int tse = VL ? tend : n_chunks * BT;  // sequence end token
  alignas(16) float Sreg[32];
  // SELF-START: compose this piece's start from h0 through pieces
  // 0..sg-1 as a NO-DRAIN tf32 TMEM chain (the fp32 D region IS the
  // next A operand — identity (m,k)->(lane,col) map, same-thread D->A
  // ordered without commit+wait). Per piece, two K=8-sliced mma
  // batches: D_tgt = D_src @ L^T then D_tgt += I @ c^T — the c add
  // rides the mma pipe (a tcgen05.st into a region in-flight mmas
  // read is UNordered). L is stored bf16 (half its TMA bytes): it
  // lands as two SWZ128 tiles in the TOP half of its 64 KB fp32 slot
  // and all threads widen it in place to the SWZ32 fp32 tiles the
  // tf32 descs read (widen_map_tf32); c stays fp32 SWZ32 TMA (bf16 c
  // failed the 16K o gate). Both slots overlay the chunk-loop tiles
  // (exactly 128 KB); slot reuse is gated by per-batch commits WAITED
  // one piece late (mb_cmL/mb_cmC), so TMA overlaps the in-flight
  // mmas. ONE drain at the end. TMEM: Da [0,128) | Db [128,256) |
  // I [256,384) — all reused by the chunk loop only after the final
  // drain.
#pragma unroll
  for (int j = 0; j < 32; ++j)
    Sreg[j] = h0s[((size_t)h * K + (ch * 32 + j)) * K + vc];
  if (sg > 0) {
    float* bL = reinterpret_cast<float*>(&sPneg[0][0][0][0]);
    float* bC = bL + K * K;
    bf16* pL = reinterpret_cast<bf16*>(bL + K * K / 2);
    static_assert(sizeof(S.sPneg) + sizeof(S.sKdT) + sizeof(S.sQd) + sizeof(S.sAh) + sizeof(S.sAl) == 2 * K * K * 4);
#pragma unroll
    for (int c4 = 0; c4 < 32; c4 += 4) {  // I diag + h0 fp32, no pack
      uint32_t iv[4], sv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        iv[e] = __float_as_int(vc == ch * 32 + c4 + e ? 1.f : 0.f);
        sv[e] = __float_as_int(Sreg[c4 + e]);
      }
      ptx::tcgen05_st_32x32b_x4(taddr + lane_hi + uint32_t(256 + ch * 32 + c4), iv[0], iv[1], iv[2], iv[3]);
      ptx::tcgen05_st_32x32b_x4(taddr + lane_hi + uint32_t(ch * 32 + c4), sv[0], sv[1], sv[2], sv[3]);
    }
    ptx::tcgen05_wait_st();
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    const uint32_t idt = ptx::mma_inst_desc_tf32(128, K);
    for (int q2 = 0; q2 < sg; ++q2) {
      if (tid == 0) {
        if (pflags)  // fused grid: trail piece q2's producer
          while (ptx::ld_acq_b32(&pflags[(pbase + q2) * H + h]) < 1u)
            __nanosleep(128);
        if (q2) ptx::mbar_wait_parity(&S.mb_cmL, (q2 - 1) & 1);
        ptx::mbar_arrive_expect_tx(&S.mb_pre, K * K * 2);
        ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(pL), sl_map, 0, ((pbase + q2) * H + h) * K, &S.mb_pre);
        ptx::cp_async_bulk_tensor_2d_load(
            ptx::to_shared(pL) + uint32_t(K * 64 * 2), sl_map, 64, ((pbase + q2) * H + h) * K, &S.mb_pre);
        if (q2) ptx::mbar_wait_parity(&S.mb_cmC, (q2 - 1) & 1);
        ptx::mbar_arrive_expect_tx(&S.mb_preC, K * K * 4);
        for (int s = 0; s < 16; ++s)
          ptx::cp_async_bulk_tensor_2d_load(
              ptx::to_shared(bC) + uint32_t(s * 4096), sc_map, 8 * s, ((pbase + q2) * H + h) * K, &S.mb_preC);
      }
      // L landed: widen (its slot writes are safe — piece q2-1's mmas
      // retired before tid0 issued this TMA, and the widen is ordered
      // after the TMA through mb_pre)
      ptx::mbar_wait_parity(&S.mb_pre, q2 & 1);
      widen_map_tf32(bL, pL, vc, ch);
      ptx::tcgen05_fence_before_thread_sync();
      __syncthreads();
      if (tid == 0) {
        ptx::tcgen05_fence_after_thread_sync();
        const uint32_t src = (q2 & 1) ? 128u : 0u, tgt = 128u - src;
        for (int s = 0; s < 16; ++s)
          mma_tf32_atmem(
              taddr + tgt,
              taddr + src + uint32_t(8 * s),
              ptx::mma_smem_desc_k_major<uint32_t, 8, 32>(ptx::to_shared(bL) + uint32_t(s * 4096)),
              idt,
              s ? 1u : 0u);
        ptx::tcgen05_commit_arrive(&S.mb_cmL);
        ptx::mbar_wait_parity(&S.mb_preC, q2 & 1);  // c: direct fp32
        for (int s = 0; s < 16; ++s)
          mma_tf32_atmem(
              taddr + tgt,
              taddr + uint32_t(256 + 8 * s),
              ptx::mma_smem_desc_k_major<uint32_t, 8, 32>(ptx::to_shared(bC) + uint32_t(s * 4096)),
              idt,
              1u);
        ptx::tcgen05_commit_arrive(&S.mb_cmC);
      }
    }
    if (tid == 0)  // last commit
      ptx::mbar_wait_parity(&S.mb_cmC, (sg - 1) & 1);
    __syncthreads();
    ptx::tcgen05_fence_after_thread_sync();
    const uint32_t fin = ((sg - 1) & 1) ? 0u : 128u;  // last tgt region
#pragma unroll
    for (int b2 = 0; b2 < 4; ++b2) {
      uint32_t r[8];
      ptx::tcgen05_ld_32x32b_x8(
          taddr + lane_hi + fin + uint32_t(ch * 32 + b2 * 8), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
      ptx::tcgen05_wait_ld();
#pragma unroll
      for (int j = 0; j < 8; ++j)
        Sreg[b2 * 8 + j] = __int_as_float(r[j]);  // c folded in-pipe
    }
    __syncthreads();
  }  // sg > 0
  // packed cvt.bf16x2 (per-half rn == the scalar pair; low half = e0)
  auto pack2 = [](float e0, float e1, uint32_t& hi, uint32_t& lo) {
    const __nv_bfloat162 h2 = __floats2bfloat162_rn(e0, e1);
    const __nv_bfloat162 l2 = __floats2bfloat162_rn(e0 - __bfloat162float(h2.x), e1 - __bfloat162float(h2.y));
    hi = *reinterpret_cast<const uint32_t*>(&h2);
    lo = *reinterpret_cast<const uint32_t*>(&l2);
  };
  if (pflags) {  // fused grid: own piece's factors must be complete
    if (tid == 0)
      while (ptx::ld_acq_b32(&pflags[(pbase + sg) * H + h]) < 1u)
        __nanosleep(128);
    __syncthreads();
  }
  auto tma_issue = [&](int cc) {
    const int bu2 = (cc - c0) & 1;
    ptx::mbar_arrive_expect_tx(&mb_tma, 6 * 64 * 64 * 2 + 128 * 64 * 2 + BT * K * 4);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sPneg[bu2][0][0][0]), &pneg_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sPneg[bu2][1][0][0]), &pneg_map, 64, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sKdT[bu2][0][0]), &kdt_map, 0, (cc * H + h) * K, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sQd[bu2][0][0][0]), &qd_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sQd[bu2][1][0][0]), &qd_map, 64, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sAh[bu2][0][0]), &aqh_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sAl[bu2][0][0]), &aql_map, 0, (cc * H + h) * BT, &mb_tma);
    ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sU0[bu2][0][0]), &u0f_map, 0, (cc * H + h) * BT, &mb_tma);
  };
  if (tid == 0 && (!VL || per_seg > 0)) tma_issue(c0);
  // running gC chunk pointer (chunks are H*K apart — same addresses,
  // minus the per-chunk 64-bit rebuild; C4 census diet) + loop-invariant
  // A1 tmem column
  const float* gCp = gC + ((size_t)c0 * H + h) * K + ch * 32;
  const uint32_t a1st = taddr + lane_hi + (256u + ch * 16);
  for (int c = c0; c < c0 + per_seg; ++c) {
    const int buf_idx = (c - c0) & 1;
    // prefetch this chunk's gC early: its latency hides behind A1
    // staging + P1 (u0 rides the chunk TMA into sU0 instead)
    alignas(16) float gv[32];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      reinterpret_cast<float4*>(gv)[j] = reinterpret_cast<const float4*>(gCp)[j];
    gCp += (size_t)H * K;
    // stage A1 = S^T packed single-bf16 (o's S-term reads it too, via
    // P3); no lo pack/st — vet b8 (vet_tf32_step.py): S-read precision
    // is not the error floor, worst kernel-ctx cell 1.27x of hi/lo
    {
      uint32_t hw[16];
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const __nv_bfloat162 h2 = __floats2bfloat162_rn(Sreg[2 * w], Sreg[2 * w + 1]);
        hw[w] = *reinterpret_cast<const uint32_t*>(&h2);
      }
      ptx::tcgen05_st_32x32b_x16(a1st, hw);
    }
    // h[c]: the register state BEFORE this chunk (store-only; the last
    // boundary is final_state's). Dense — every row written; a per-sequence
    // snapshot-index filter would gate this store (sparse follow-up).
    if (hpc) {
      const size_t hb = ((size_t)c * H + h) * K * K;
      if (hpc_v_first) {  // [V,K]: kc contiguous, one run per thread
        const size_t off = hb + (size_t)vc * K + ch * 32;
        if (hpc_bf16) {
          auto* p = reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(hpc) + off);
#pragma unroll
          for (int j = 0; j < 16; ++j)
            p[j] = __floats2bfloat162_rn(Sreg[2 * j], Sreg[2 * j + 1]);
        } else {
          auto* p = reinterpret_cast<float4*>(reinterpret_cast<float*>(hpc) + off);
#pragma unroll
          for (int j = 0; j < 8; ++j)
            p[j] = reinterpret_cast<const float4*>(Sreg)[j];
        }
      } else {  // [K,V]: vc contiguous, coalesced across the warp
        const size_t off = hb + (size_t)(ch * 32) * K + vc;
        if (hpc_bf16) {
          bf16* p = reinterpret_cast<bf16*>(hpc) + off;
#pragma unroll
          for (int j = 0; j < 32; ++j)
            p[j * K] = __float2bfloat16(Sreg[j]);
        } else {
          float* p = reinterpret_cast<float*>(hpc) + off;
#pragma unroll
          for (int j = 0; j < 32; ++j)
            p[j * K] = Sreg[j];
        }
      }
    }
    ptx::tcgen05_wait_st();
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    // ALL threads gate on the chunk TMA: the sU0 reads in the U drain
    // below need the cross-proxy visibility this mbarrier's completion
    // gives its waiters (non-issuer threads previously just idled into
    // the mb_p1 wait from here, so no overlap is lost)
    ptx::mbar_wait_parity(&mb_tma, (c - c0) & 1);
    if (tid == 0 || tid == 32) {  // dual issuers, disjoint D regions
      // phase closed: arm the next chunk into the other buffer
      if (tid == 0 && c + 1 < c0 + per_seg) tma_issue(c + 1);
      ptx::tcgen05_fence_after_thread_sync();
      const uint32_t idesc1 = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
          128, 64, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);
      if (tid == 0) {
#pragma unroll
        for (int k16 = 0; k16 < 8; ++k16) {
          const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(
              ptx::to_shared(&sPneg[buf_idx][k16 >> 2][0][0]) + (k16 & 3) * 32);
          ptx::tcgen05_mma_f16_atmem(taddr, taddr + (256u + k16 * 8), db, idesc1, k16 ? 1u : 0u);
        }
        ptx::tcgen05_commit_arrive(&mb_p1);
      } else {
#pragma unroll
        for (int k16 = 0; k16 < 8; ++k16) {  // P3: o^T = S^T@qdec^T
          const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(
              ptx::to_shared(&sQd[buf_idx][k16 >> 2][0][0]) + (k16 & 3) * 32);
          ptx::tcgen05_mma_f16_atmem(taddr + 384u, taddr + (256u + k16 * 8), db, idesc1, k16 ? 1u : 0u);
        }
        ptx::tcgen05_commit_arrive(&mb_o);
      }
    }
    ptx::mbar_wait_parity(&mb_p1, (c - c0) & 1);  // U-drain: P1 alone
    ptx::tcgen05_fence_after_thread_sync();
    // drain U^T (+u0), pack A2 (one x16 ld: one TMEM
    // round-trip for the 16 columns instead of two x8 trips)
    {
      alignas(16) float Ur[16];
      {
        uint32_t r[16];
        ptx::tcgen05_ld_32x32b_x16(
            taddr + lane_hi + uint32_t(ch * 16),
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
            r[9],
            r[10],
            r[11],
            r[12],
            r[13],
            r[14],
            r[15]);
        ptx::tcgen05_wait_ld();
#pragma unroll
        for (int j = 0; j < 16; ++j)
          Ur[j] = __int_as_float(r[j]) + sU0[buf_idx][ch * 16 + j][vc];
      }
      uint32_t hw[8], lw[8];
#pragma unroll
      for (int w = 0; w < 8; ++w)
        pack2(Ur[2 * w], Ur[2 * w + 1], hw[w], lw[w]);
      ptx::tcgen05_st_32x32b_x8(
          taddr + lane_hi + (192u + ch * 8), hw[0], hw[1], hw[2], hw[3], hw[4], hw[5], hw[6], hw[7]);
      ptx::tcgen05_st_32x32b_x8(
          taddr + lane_hi + (224u + ch * 8), lw[0], lw[1], lw[2], lw[3], lw[4], lw[5], lw[6], lw[7]);
    }
    ptx::tcgen05_wait_st();
    ptx::tcgen05_fence_before_thread_sync();
    __syncthreads();
    if (tid == 0 || tid == 32) {
      ptx::tcgen05_fence_after_thread_sync();
      if (tid == 0) {
        const uint32_t idesc2 = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
            128, 128, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);
        for (int half = 0; half < 2; ++half)
#pragma unroll
          for (int k16 = 0; k16 < 4; ++k16) {
            const uint64_t db =
                ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(&sKdT[buf_idx][0][0]) + k16 * 32);
            ptx::tcgen05_mma_f16_atmem(
                taddr + 64u, taddr + (192u + half * 32 + k16 * 8), db, idesc2, (half | k16) ? 1u : 0u);
          }
        ptx::tcgen05_commit_arrive(&mb_p2);
      } else {
        const uint32_t idesc4 = ptx::mma_inst_desc_dense<ptx::MmaDenseKind::F16>(
            128, 64, ptx::F16Type::BF16, ptx::F16Type::BF16, ptx::DType::F32, ptx::Major::K, ptx::Major::K);
        // P4: o^T += u^T@Aqk^T (u-hi x A-hi, u-hi x A-lo, u-lo x A-hi)
        for (int pr = 0; pr < 3; ++pr) {
          const uint32_t acol = pr == 2 ? 224u : 192u;
          const bf16* ab = pr == 1 ? &sAl[buf_idx][0][0] : &sAh[buf_idx][0][0];
#pragma unroll
          for (int k16 = 0; k16 < 4; ++k16) {
            const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(ptx::to_shared(ab) + k16 * 32);
            ptx::tcgen05_mma_f16_atmem(taddr + 384u, taddr + (acol + k16 * 8), db, idesc4, 1u);
          }
        }
        ptx::tcgen05_commit_arrive(&mb_o);
      }
    }
    ptx::mbar_wait_parity(&mb_p2, (c - c0) & 1);  // S-drain: P2 alone
    ptx::tcgen05_fence_after_thread_sync();
    // drain P2: S = e^{gC} o S + tmem (gv prefetched at chunk top;
    // one x32 ld = the widest single-instruction drain)
    {
      uint32_t r[32];
      ptx::tcgen05_ld_32x32b_x32(taddr + lane_hi + uint32_t(64 + ch * 32), r);
      ptx::tcgen05_wait_ld();
#pragma unroll
      for (int jj = 0; jj < 32; ++jj)
        Sreg[jj] = gv[jj] * Sreg[jj] + __int_as_float(r[jj]);
    }
    // o-pipe drain (P3+P4) rides behind the S update, off the spine
    ptx::mbar_wait_parity(&mb_o, (c - c0) & 1);
    ptx::tcgen05_fence_after_thread_sync();
    {
      uint32_t r[16];
      ptx::tcgen05_ld_32x32b_x16(
          taddr + lane_hi + uint32_t(384 + ch * 16),
          r[0],
          r[1],
          r[2],
          r[3],
          r[4],
          r[5],
          r[6],
          r[7],
          r[8],
          r[9],
          r[10],
          r[11],
          r[12],
          r[13],
          r[14],
          r[15]);
      ptx::tcgen05_wait_ld();
      // running row pointer (rows are H*K apart — same addresses,
      // minus the per-element 64-bit row*H*K rebuild)
      const int row0 = c * BT + tokoff + ch * 16;
      bf16* op = o + ((size_t)row0 * H + h) * K + vc;
      const size_t orow = (size_t)H * K;
#pragma unroll
      for (int j = 0; j < 16; ++j, op += orow)
        if (!VL || row0 + j < tse)  // pad rows: the NEXT seq's
          *op = __float2bfloat16(__int_as_float(r[j]));
    }
    __syncthreads();
  }
  if (sg == nseg - 1)
#pragma unroll
    for (int j = 0; j < 32; ++j)
      Sf[((size_t)h * K + ch * 32 + j) * K + vc] = Sreg[j];
  __syncthreads();
  if (warp == 0) ptx::tcgen05_dealloc(taddr, 512);
  ptx::tcgen05_relinquish();
}
// eqlen NP == 1 fallback: whole chain from h0, no piece maps/flags (the
// sl/sc maps are valid encodes over the workspace but never dereferenced)
__global__ void __launch_bounds__(512) k2_chain_tc(
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap sl_map,
    const __grid_constant__ CUtensorMap sc_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const float* __restrict__ gC,
    const float* __restrict__ h0,
    int n_chunks,
    int H,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first) {
  __shared__ ChainSmem S;
  chain_body(
      S,
      blockIdx.x,
      blockIdx.y,
      gridDim.y,
      pneg_map,
      kdt_map,
      qd_map,
      aqh_map,
      aql_map,
      u0f_map,
      gC,
      n_chunks,
      H,
      o,
      Sf,
      hpc,
      hpc_bf16,
      hpc_v_first,
      &sl_map,
      &sc_map,
      h0);
}

// ---------------- fused grid: chain blocks trail k1's pieces --------------
// bids [0,NP*H) = k1 piece-builders (publish pflags after their stores);
// bids [NP*H, 2*NP*H) = self-start chain blocks gated on the piece flags.
// Chains become resident as builder waves retire and their spins trail the
// (mostly complete) flags — the launch boundary and part of the chain wall
// hide under k1's tail.
template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512) kda_fused(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    int NP,
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const __grid_constant__ CUtensorMap sl_map,
    const __grid_constant__ CUtensorMap sc_map,
    const float* __restrict__ h0,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    uint32_t* __restrict__ pflags) {
  union FusedSmem {
    K1Smem k1;
    ChainSmem chain;
  };
  __shared__ FusedSmem S;
  const int tid = threadIdx.x, bid = blockIdx.x;
  const int nc1 = T / BT;
  if (bid < NP * H) {
    const int h = bid / NP, pc = bid % NP;
    const int c00 = piece_c0(pc, nc1, NP);
    const int njobs = piece_c0(pc + 1, nc1, NP) - c00;
    k1_body<false, GM, RAW, BSIG>(
        S.k1,
        h * nc1 + c00,
        njobs,
        q,
        kk,
        v,
        glog,
        beta,
        T,
        H,
        scale,
        P,
        u0,
        kdec,
        qdec,
        aqk_h,
        aqk_l,
        gC,
        pieceL,
        piecec,
        pc * H + h,
        0,
        0,
        0,
        a_log,
        dtb,
        lb,
        &pneg_map);
    __syncthreads();
    if (tid == 0) {
      ptx::fence_async_global();  // consumers read via TMA
      ptx::red_add_rel_b32(&pflags[pc * H + h], 1u);
    }
  } else {
    const int r = bid - NP * H, h = r % H, sg = r / H;
    chain_body(
        S.chain,
        h,
        sg,
        NP,
        pneg_map,
        kdt_map,
        qd_map,
        aqh_map,
        aql_map,
        u0f_map,
        gC,
        nc1,
        H,
        o,
        Sf,
        hpc,
        hpc_bf16,
        hpc_v_first,
        &sl_map,
        &sc_map,
        h0,
        pflags);
  }
}

// ---------------- varlen fused grid (cu_seqlens + partial tails) ----------
// kda_fused generalized over a host-built PIECE TABLE: sequences pro-rate
// a GLOBAL piece budget by nc_s (seqs already parallelize the grid, a
// per-seq NP over-fills it), capped at the per-seq split
// (min(12, max(1, nc_s/2)) — no empty pieces) and chains compose prefixes
// WITHIN their sequence only. Factor tensors index by GLOBAL chunk
// (cbase_s + local c), maps/pflags by global piece id;
// q/k/v/glog/beta/o stay flat [T,H,K] and tail-chunk pad rows are
// zero-filled on load / masked on store inside the bodies (k1_body C_act
// guards, chain_body o mask). Only TAIL pieces (the one piece holding a
// partial last chunk) need those guards: full pieces dispatch to the exact
// eqlen bodies — t0 = c*BT + tokoff and the global-chunk factor indexing
// are shared, so with T = nc_tot*BT (nc1 == nc_tot) every address matches
// the VL body's.
struct VlPiece {
  int seq;     // sequence index (h0 / Sf row)
  int cbase;   // sequence's first global chunk
  int nc;      // sequence chunk count ceil(len/BT)
  int np;      // pieces in the sequence
  int sg;      // piece index within the sequence
  int tokoff;  // tok0 - cbase*BT: global token row = c*BT + tokoff + i
  int tend;    // sequence end token (pad mask)
  int tail;    // piece holds the partial last chunk (len%BT != 0)
};
template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512) kda_fused_vl(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    bf16* __restrict__ pieceL,
    float* __restrict__ piecec,
    const VlPiece* __restrict__ pieces,
    int npt,
    int nc_tot,
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const __grid_constant__ CUtensorMap sl_map,
    const __grid_constant__ CUtensorMap sc_map,
    const float* __restrict__ h0,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    uint32_t* __restrict__ pflags) {
  union FusedSmem {
    K1Smem k1;
    ChainSmem chain;
  };
  __shared__ FusedSmem S;
  const int tid = threadIdx.x, bid = blockIdx.x;
  if (bid < npt * H) {
    const int h = bid / npt, p = bid % npt;
    const VlPiece* pc = pieces + p;
    const int c00 = pc->cbase + piece_c0(pc->sg, pc->nc, pc->np);
    const int njobs = pc->cbase + piece_c0(pc->sg + 1, pc->nc, pc->np) - c00;
    if (pc->tail)
      k1_body<true, GM, RAW, BSIG>(
          S.k1,
          h * nc_tot + c00,
          njobs,
          q,
          kk,
          v,
          glog,
          beta,
          T,
          H,
          scale,
          P,
          u0,
          kdec,
          qdec,
          aqk_h,
          aqk_l,
          gC,
          pieceL,
          piecec,
          p * H + h,
          nc_tot,
          pc->tokoff,
          pc->tend,
          a_log,
          dtb,
          lb,
          &pneg_map);
    else  // full piece: the eqlen body on T = nc_tot*BT (see VlPiece)
      k1_body<false, GM, RAW, BSIG>(
          S.k1,
          h * nc_tot + c00,
          njobs,
          q,
          kk,
          v,
          glog,
          beta,
          nc_tot * BT,
          H,
          scale,
          P,
          u0,
          kdec,
          qdec,
          aqk_h,
          aqk_l,
          gC,
          pieceL,
          piecec,
          p * H + h,
          nc_tot,
          pc->tokoff,
          pc->tend,
          a_log,
          dtb,
          lb,
          &pneg_map);
    __syncthreads();
    if (tid == 0) {
      ptx::fence_async_global();  // consumers read via TMA
      ptx::red_add_rel_b32(&pflags[p * H + h], 1u);
    }
  } else {
    const int r = bid - npt * H, h = r % H, p = r / H;
    const VlPiece* pc = pieces + p;
    const size_t so = (size_t)pc->seq * H * K * K;  // h0/Sf per sequence
    if (pc->tail)                                   // only the tail piece stores a partial o chunk
      chain_body<true>(
          S.chain,
          h,
          pc->sg,
          pc->np,
          pneg_map,
          kdt_map,
          qd_map,
          aqh_map,
          aql_map,
          u0f_map,
          gC,
          pc->nc,
          H,
          o,
          Sf + so,
          hpc,
          hpc_bf16,
          hpc_v_first,
          &sl_map,
          &sc_map,
          h0 + so,
          pflags,
          pc->cbase,
          p - pc->sg,
          pc->tokoff,
          pc->tend);
    else
      chain_body<false>(
          S.chain,
          h,
          pc->sg,
          pc->np,
          pneg_map,
          kdt_map,
          qd_map,
          aqh_map,
          aql_map,
          u0f_map,
          gC,
          pc->nc,
          H,
          o,
          Sf + so,
          hpc,
          hpc_bf16,
          hpc_v_first,
          &sl_map,
          &sc_map,
          h0 + so,
          pflags,
          pc->cbase,
          p - pc->sg,
          pc->tokoff,
          pc->tend);
  }
}

// ---------------- seq0 route: tail-free builder + whole-seq chains --------
// k1_body with the fused tail DELETED — no piece-map composition, no TMEM, no
// pTt TMA. Nothing composes the maps, so chains cannot self-start: they run
// ONE CTA per (sequence, head) walking that sequence from h0 (NP_chain == 1),
// and the two halves are two launches instead of one fused grid. What the tail
// buys back is smem — every dead range unions (the gate stage over the mma
// operands, the solve pane and packed A tiles over both) — so this builder is
// 102144 B / 64 regs = 2 CTAs/SM against the fused grid's 229376 B / 1. It
// wins wherever the whole-sequence chain depth is short next to the build wall
// (many sequences, or high H); pick_route decides per shape.
// The diet is unconditional here. Every item is factor-exact vs k1_body
// (identical fp32 op sequences, order-preserving remaps, deterministic __expf
// recompute):
//   - gate stage single-buffered (own-chunk arrival, no next-chunk prefetch)
//     with the GM transform folded into P2a's cumsum read
//   - fp32 sA -> packed strict-lower diagonal blocks, and the off-diagonal A
//     fragments are held in registers and scattered straight to the coupling
//     operand (so aC may alias the mma operands)
//   - ONE 128-col rhs pane, rebuilt per pane inside the solve (pane 0 kappa:
//     the held k tap + cumsum recompute; pane 1 v * beta), so it lives in the
//     operand union
//   - sgc unions with upT (upT is born in the solve, after sgc's last read)
//   - the 4 diagonal kz tiles stream through slots 0..3 and the 6 off-diagonal
//     tiles then overwrite them (6 slots, not 10)
// P4-lo is baked in too: the solve's cross-block coupling consumes the hi
// tiles only (the double-bf16 lo corrections are dropped, not reordered) and
// no lo tile is produced. So is the block apply: the in-block substitution is
// X_b = M_bb @ rhs_b over 4 diagonal-only inverses (below) instead of a
// 15-deep fp32 chain. P and u0 therefore DIFFER from k1_body's bytes BY
// DESIGN — this route is envelope-gated (|ours - fp64| <= |fla - fp64|),
// never byte-gated, against the fused one.
// The LSU/shared data pipe is the binding resource on this builder (the only
// counter above 70%), and four items serve it. All four are layout or
// redundancy only — same products, same accumulation order, same bf16 rounds —
// so every factor byte is unchanged:
//   - the coupling shuffle and the sU swizzle below (the block apply's own
//     store/read conflict fix)
//   - the gate tile arrives as ONE [BT][K] bf16 TMA box (SWIZZLE_NONE, so the
//     landed tile is byte-for-byte the row-major staging per-thread cp.async
//     produced). TMA writes shared through its own path, so both halves of the
//     LDGSTS pair leave the pipe. q/k/v are NOT converted: every element is
//     scaled by a decay on the way in, so they land in registers, and staging
//     3 x 16 KB for LDS readback costs the same wavefronts for smem that does
//     not exist here.
//   - the k tile is loaded ONCE: P2b's taps, the kz tiles and the pane-0 rhs
//     rebuild are the SAME 8 bf16x2 values per thread (row u + 8m, column
//     colp, one pad mask), so two of three global reads were pure redundancy
//   - the kdec staging row map (kdec_r) and the 4-wide aqk store (below)
// sU physical column for logical (row, col): the 4-column group index rotates
// by 3 on ODD rows. The apply's X store wants a row-to-bank spread of 8 and
// its B-operand read wants 4, so no row stride serves both; this map makes
// both conflict-free at zero smem cost. Group-preserving, so every 4-aligned
// float2/float4 access stays vectorized (K == 128 == 32 groups).
static __device__ __forceinline__ int su_c(int r, int c) {
  return ((((c >> 2) + 3 * (r & 1)) & 31) << 2) | (c & 3);
}
// kdec staging physical row for logical row r: the 6-bit row index rotated
// right by 2. The repack's 32 lanes each read ONE column down FOUR consecutive
// rows, so lanes step the row by 4 and the [BT][K+4] pitch (132 bf16 = 66 words
// == 2 banks) gives a lane step of 8 banks: 4 banks for 32 lanes. No pitch can
// fix that (4*pitch is always 0 mod 4 banks), so the ROW index has to move.
// Under the rotation a thread's four rows sit 16 apart (16 rows == 0 mod 32
// banks) and consecutive lanes step ONE physical row == 2 banks: one wavefront.
// Both users step r by a multiple of 4 and stay inside one field, so the map is
// affine on each (kdec_r(r + 4n) == kdec_r(r) + kdec_r(4) * n) and costs no
// instruction and no smem — the staging is a dead-window kz-area alias.
static __device__ __forceinline__ constexpr int kdec_r(int r) {
  static_assert(BT == 64, "the map rotates a 6-bit staging row index");
  return (r >> 2) + ((r & 3) << 4);
}
struct K1SmemTF {
  // 128 B: the gate tile's TMA destination (16 B would satisfy the box)
  union __align__(128) Pool {  // disjoint live ranges
    bf16 sgb[BT][K];           // P1 -> P2a
    struct {                   // P2b -> P3 mma
      bf16 kw[BT][K + 8];
      bf16 qw[BT][K + 8];
      bf16 kz[6][16][K + 8];
    } a;
    struct {  // P3 scatter -> P4/P5
      bf16 aC_h[BT][56];
      float sU[BT][K + 4];  // the one rhs pane
    } f;
  } sp;
  union __align__(16) GPool {  // sgc dies before upT is born
    float sgc[BT][K];
    bf16 upT_h[K][56];
  } g;
  float tri[4][120];  // strict-lower 16x16 diag blocks, packed i(i-1)/2+j
  // (I+A_bb)^-1 hi/lo, row-major mma A operands. Own members, NOT the pool:
  // written in P3 while the pool still holds the mma operands, read through
  // both panes. Row stride 24 (48 B, 16B-aligned) spreads the 16 ldmatrix
  // row addresses over all 8 bank segments.
  alignas(16) bf16 Mh[4][16][24];  // ldmatrix needs the 16B row alignment
  alignas(16) bf16 Ml[4][16][24];
  float sb[BT];
  uint64_t mb_in;  // gate-tile TMA arrival (one phase per chunk-job)
};
// RAW only: {k, q} row reciprocal norms. Nothing in the pool is dead across
// their P2b -> P4 pane-0 span (that is what the diet bought), so they cost
// 512 B on top of the 102144 B — 102656 B still holds 2 CTAs/SM — and the
// pre-normed instantiations keep the struct byte-for-byte.
struct K1SmemTFRaw : K1SmemTF {
  float2 rn[BT];
};

// VL: varlen tail piece (the k1_body idiom) — pad rows zero-fill on load so
// every factor row past the seq end is exactly 0; eqlen keeps the original
// code (the guards compile out).
// RAW: q/k un-normalized (see row_rnorm); the one k tap is normalized where it
// is loaded, so all three consumers see the same bytes. BSIG: beta as logits.
template <int GM = 0, bool VL = false, bool RAW = false, bool BSIG = false>
__device__ static void k1_tf_body(
    K1SmemTF& S,
    int job0,
    int njobs,
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const CUtensorMap& glog_map,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    // GM != 0 gate-transform inputs (production lb = -5.0)
    const float* __restrict__ a_log = nullptr,
    const float* __restrict__ dtb = nullptr,
    float lb = 0.f,
    // varlen piece coords (defaults = eqlen): global chunk c's tokens start at
    // c*BT + tokoff; rows >= tend - t0 are pad (zero-filled on load)
    int tokoff = 0,
    int tend = 0,
    float2* __restrict__ rn = nullptr)  // RAW: K1SmemTFRaw::rn
{
  constexpr int SB = 16, NSB = BT / SB;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  auto& kw = S.sp.a.kw;
  auto& qw = S.sp.a.qw;
  auto& kz = S.sp.a.kz;
  auto& aC_h = S.sp.f.aC_h;
  float (&sgc)[BT][K] = S.g.sgc;
  auto& upT_h = S.g.upT_h;
  auto& sU = S.sp.f.sU;  // the pane, in the pool union
  auto& tri = S.tri;
  auto& Mh = S.Mh;
  auto& Ml = S.Ml;
  auto& sb = S.sb;
  const int nc1 = T / BT;
  const int tse = VL ? tend : T;  // sequence end token
  // GM hoist: h is constant across a block's jobs (one piece, one head)
  const float ga = GM != 0 ? expf(a_log[job0 / nc1]) : 0.f;
  // P2a-fold hoist: the cumsum column is thread-fixed (tid & (K-1)) and h is
  // block-constant -> ONE dt_bias scalar covers every job
  const float dtv_xf = GM != 0 ? dtb[(size_t)(job0 / nc1) * K + (tid & (K - 1))] : 0.f;
  if (tid == 0) {
    ptx::mbar_init(&S.mb_in, 1);
    ptx::prefetch_tensormap(&glog_map);
  }
  __syncthreads();
  for (int sub = 0; sub < njobs; ++sub) {
    const int job = job0 + sub;
    const int c = job % nc1, h = job / nc1, t0 = c * BT + tokoff;
    const int C_act = VL ? min(BT, tse - t0) : BT;  // real rows (tail < BT)
    bf16(&sgb)[BT][K] = S.sp.sgb;
    if (sub) __syncthreads();  // pool reuse: prior job's P4/P5 reads done
    if (tid == 0) {            // ONE box: [BT][K] bf16, row-major (SWIZZLE_NONE)
      ptx::mbar_arrive_expect_tx(&S.mb_in, BT * K * 2);
      ptx::cp_async_bulk_tensor_2d_load(ptx::to_shared(&sgb[0][0]), &glog_map, h * K, t0, &S.mb_in);
    }
    if (tid < BT) sb[tid] = !VL || tid < C_act ? beta_in<BSIG>(beta[(size_t)(t0 + tid) * H + h]) : 0.f;
    if constexpr (RAW)  // one row-norm pass per chunk-job, q and k together
      // (rides the gate box; the reads below hit the same L1 lines)
      row_rnorm<VL>(rn[tid >> 3], q, kk, (size_t)t0 * H * K + (size_t)h * K, H, C_act, tid);
    ptx::mbar_wait_parity(&S.mb_in, sub & 1);  // all threads consume the phase
    if constexpr (VL) {
      // rows past the SEQUENCE end are in-bounds for the box (they are the
      // next sequence's tokens), so the zero-fill the cp.async did by
      // predicate happens here; rows past the buffer end arrived as zeros
      for (int p = tid; p < (BT - C_act) * K / 8; p += blockDim.x)
        *reinterpret_cast<uint4*>(&sgb[C_act + p * 8 / K][p * 8 % K]) = uint4{0, 0, 0, 0};
    }
    __syncthreads();
    {  // P2a: split cumsum, 512 threads = 4 x 16-row segments per column
      const int col = tid & (K - 1), r0 = (tid >> 7) * (BT / 4);
      float acc = 0.f;
      for (int r = r0; r < r0 + BT / 4; ++r) {
        // the GM transform folds into the read: identical per-element math
        // INCLUDING the bf16 round k1_body stores through its gate stage
        float gv;
        if constexpr (GM != 0) {
          const float g = __bfloat162float(sgb[r][col]) + dtv_xf;
          const float y =
              GM == 1 ? -ga * (fmaxf(g, 0.f) + __logf(1.f + __expf(-fabsf(g)))) : lb * (1.f / (1.f + __expf(-ga * g)));
          gv = __bfloat162float(__float2bfloat16(y));
        } else {
          gv = __bfloat162float(sgb[r][col]);
        }
        if (VL && r >= C_act) gv = 0.f;  // pad: transform(0) != 0
        acc += gv;
        sgc[r][col] = acc;
      }
    }
    __syncthreads();
    {  // carry propagation
      const int colc = (tid & (K / 2 - 1)) * 2, rc = tid >> 6;
      for (int seg = 1; seg < 4; ++seg) {
        const float2 carry = *reinterpret_cast<const float2*>(&sgc[seg * (BT / 4) - 1][colc]);
#pragma unroll
        for (int it = 0; it < 2; ++it) {
          float2& v2 = *reinterpret_cast<float2*>(&sgc[seg * (BT / 4) + rc + it * 8][colc]);
          v2 = float2{v2.x + carry.x, v2.y + carry.y};
        }
        __syncthreads();
      }
    }
    const int colp = (tid & (K / 2 - 1)) * 2;
    float a0v[NSB], a1v[NSB];  // pair-map anchors (the kz chains keep them)
    float ea0[NSB], ea1[NSB];
    ea0[0] = ea1[0] = 1.f;
    a0v[0] = a1v[0] = 0.f;
#pragma unroll
    for (int s = 1; s < NSB; ++s) {
      a0v[s] = sgc[s * SB - 1][colp];
      a1v[s] = sgc[s * SB - 1][colp + 1];
      ea0[s] = __expf(a0v[s]);
      ea1[s] = __expf(a1v[s]);
    }
    // the ONE k tile: tap m == hb*4+j == 2*sj+rr is row u + 8m, column colp,
    // under one pad mask — so P2b, P2c and the pane-0 rhs all read this
    __nv_bfloat162 kvz[NSB][2];
    {  // P2b: kw/qw/qdec/kdec (the rhs is built later, in the solve)
      bf16(*stg)[K + 4] = reinterpret_cast<bf16(*)[K + 4]>(&kz[0][0][0]);
      const float gl0 = sgc[BT - 1][colp], gl1 = sgc[BT - 1][colp + 1];
      const size_t gp0 = (size_t)(t0 + tid * 2 / K) * H * K + h * K + tid * 2 % K;
      const size_t gst = (size_t)8 * H * K;  // +8 rows per tap
      const bf16* pk = &kk[gp0];
      const bf16* pq = &q[gp0];
      bf16* qd = &qdec[((size_t)c * H + h) * BT * K + (tid >> 6) * K + colp];
      // the staging row map on the tap layout (row == (tid>>6) + 8*tap): it
      // stays affine, so each tap keeps its immediate offset — mapping the
      // materialized row instead costs 3 int ops per tap
      const int sr0 = kdec_r(tid >> 6), srt = kdec_r(8);
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        float2 kv[4], qv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int m = hb * 4 + j;
          if (VL && tid * 2 / K + m * 8 >= C_act) {
            kv[j] = qv[j] = float2{0.f, 0.f};  // pad rows
            kvz[m >> 1][m & 1] = __floats2bfloat162_rn(0.f, 0.f);
          } else {
            kvz[m >> 1][m & 1] = *reinterpret_cast<const __nv_bfloat162*>(pk);
            qv[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pq));
            if constexpr (RAW) {  // l2norm, in fla's bf16 bytes
              const float2 r = rn[(tid >> 6) + m * 8];
              kvz[m >> 1][m & 1] = l2_bf16(__bfloat1622float2(kvz[m >> 1][m & 1]), r.x);
              qv[j] = l2_round(qv[j], r.y);
            }
            kv[j] = __bfloat1622float2(kvz[m >> 1][m & 1]);
          }
          pk += gst;
          pq += gst;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (tid >> 6) + (hb * 4 + j) * 8;
          const int si = hb * 2 + (j >> 1);  // == i / SB (tid < 512)
          const float ei0 = __expf(sgc[i][colp] - a0v[si]);
          const float ei1 = __expf(sgc[i][colp + 1] - a1v[si]);
          const float kw0 = kv[j].x * ei0, kw1 = kv[j].y * ei1;
          *reinterpret_cast<__nv_bfloat162*>(&kw[i][colp]) = __floats2bfloat162_rn(kw0, kw1);
          const float qw0 = qv[j].x * ei0 * scale;
          const float qw1 = qv[j].y * ei1 * scale;
          *reinterpret_cast<__nv_bfloat162*>(&qw[i][colp]) = __floats2bfloat162_rn(qw0, qw1);
          *reinterpret_cast<__nv_bfloat162*>(qd) = __floats2bfloat162_rn(qw0 * ea0[si], qw1 * ea1[si]);
          qd += 8 * K;  // i advances 8 rows per tap
          *reinterpret_cast<__nv_bfloat162*>(&stg[sr0 + srt * (hb * 4 + j)][colp]) =
              __floats2bfloat162_rn(kv[j].x * __expf(gl0 - sgc[i][colp]), kv[j].y * __expf(gl1 - sgc[i][colp + 1]));
        }
      }
      __syncthreads();
      // kdec store from the kz-area staging
      const int rr = tid * 4 % BT, cc = tid * 4 / BT;
      bf16* kd = &kdec[((size_t)c * H + h) * BT * K + tid * 4];
#pragma unroll
      for (int n = 0; n < BT * K / 2048; ++n) {
        const int cn = cc + n * 32;
        alignas(8) const __nv_bfloat162 p2[2] = {
            {stg[kdec_r(rr)][cn], stg[kdec_r(rr + 1)][cn]}, {stg[kdec_r(rr + 2)][cn], stg[kdec_r(rr + 3)][cn]}};
        *reinterpret_cast<float2*>(kd + n * 2048) = *reinterpret_cast<const float2*>(p2);
      }
    }
    __syncthreads();
    // P2c: kz pair tiles + gC
    const int u = tid >> 6;
    float f0[NSB - 1], f1[NSB - 1];
#pragma unroll
    for (int s = 1; s < NSB; ++s) {
      f0[s - 1] = __expf(a0v[s] - a0v[s - 1]);
      f1[s - 1] = __expf(a1v[s] - a1v[s - 1]);
    }
    {  // phase A: the 4 diagonal (base) tiles into slots 0..3
#pragma unroll
      for (int sj = 0; sj < NSB; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8, jj = u + rr * 8;
          const float2 kf = __bfloat1622float2(kvz[sj][rr]);
          // pad rows select 0 OUTRIGHT (the k1_body law): their base
          // exponent spans to the seq end and 0 * __expf(ovfl) = NaN
          const bool real = !VL || j < C_act;
          const float v0 = real ? kf.x * __expf(a0v[sj] - sgc[j][colp]) : 0.f;
          const float v1 = real ? kf.y * __expf(a1v[sj] - sgc[j][colp + 1]) : 0.f;
          *reinterpret_cast<__nv_bfloat162*>(&kz[sj][jj][colp]) = __floats2bfloat162_rn(v0, v1);
        }
    }
    if (tid < K / 4) {  // gC = e^{cumsum at chunk end}
      const float4 g4 = *reinterpret_cast<const float4*>(&sgc[BT - 1][tid * 4]);
      *reinterpret_cast<float4*>(&gC[((size_t)c * H + h) * K + tid * 4]) =
          float4{__expf(g4.x), __expf(g4.y), __expf(g4.z), __expf(g4.w)};
    }
    __syncthreads();
    if (warp >= 10) {  // L2-prefetch the next job's inputs
      const int jn = job + 1;
      if (jn < nc1 * H && sub + 1 < njobs) {
        const int t1p = (jn % nc1) * BT + tokoff;
        const int Cp = VL ? min(BT, tse - t1p) : BT;  // stop at seq end
        const size_t tb = (size_t)t1p * H * K + (size_t)(jn / nc1) * K;
        const char* pq = reinterpret_cast<const char*>(q + tb);
        const char* pk = reinterpret_cast<const char*>(kk + tb);
        const char* pv = reinterpret_cast<const char*>(v + tb);
        const char* pg = reinterpret_cast<const char*>(glog + tb);
        for (int i = tid - 320; i < Cp; i += 192) {
          const size_t ro = (size_t)i * H * K * 2;
          for (int l = 0; l < 256; l += 128) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pq + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pk + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pv + ro + l));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pg + ro + l));
          }
        }
      }
    }
    // P3: A and Aqk via mma — per pair: [16x16] = kw_si @ kz_pi^T (qw for
    // Aqk), one pair per warp. Diagonal pairs run first (slots 0..3), then the
    // off-diagonal tiles overwrite those slots for the second phase.
    auto pair_mma = [&](int si2, int kzslot, float4* acck, float4* accq) {
      acck[0] = acck[1] = accq[0] = accq[1] = float4{0, 0, 0, 0};
      const int arow = lane & 15, aka = lane >> 4;
#pragma unroll
      for (int k16 = 0; k16 < K / 16; ++k16) {
        uint32_t a0, a1, a2, a3, q0, q1, q2, q3;
        ptx::ldmatrix_x4_b16(ptx::to_shared(&kw[si2 * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
        ptx::ldmatrix_x4_b16(ptx::to_shared(&qw[si2 * SB + arow][k16 * 16 + aka * 8]), q0, q1, q2, q3);
#pragma unroll
        for (int n8 = 0; n8 < 2; ++n8) {
          uint32_t b0, b1;
          ptx::ldmatrix_x2_b16(
              ptx::to_shared(&kz[kzslot][(lane & 7) + n8 * 8][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
          ptx::mma_m16n8k16_bf16f32(acck[n8], a0, a1, a2, a3, b0, b1);
          ptx::mma_m16n8k16_bf16f32(accq[n8], q0, q1, q2, q3, b0, b1);
        }
      }
    };
    // Aqk fragments go STRAIGHT to global hi/lo (host zero-fills once; a
    // diagonal-straddling group re-writes the host zero past its diagonal —
    // bit-exact vs never-written).
    // The mma C fragment gives a lane ONE column pair, so a warp's 4 B stores
    // touch 8 rows for 128 B of payload. Lanes l and l^1 hold the SAME rows, so
    // one xor-1 exchange gives each lane four CONSECUTIVE columns: even lanes
    // [own n8=0 | partner n8=0], odd lanes [partner n8=1 | own n8=1]. Same
    // values at the same positions, half the store instructions and half the
    // wavefronts; 32 B/row is the floor, since a warp owns only 16 of a row's
    // 64 columns.
    auto aqk_scatter = [&](int si2, int sj2, const float4* accq) {
      const int r = lane >> 2, c2 = (lane & 3) * 2;
      const size_t abase = ((size_t)c * H + h) * BT * BT;
      const int odd = lane & 1, jj = sj2 * SB + c2 + 6 * odd;
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2) {
        const int ii = si2 * SB + r + e2 * 8, lim = ii - jj;
        const float m0 = e2 ? accq[0].z : accq[0].x;
        const float m1 = e2 ? accq[0].w : accq[0].y;
        const float p0 = e2 ? accq[1].z : accq[1].x;
        const float p1 = e2 ? accq[1].w : accq[1].y;
        // every lane must reach the exchange, so it precedes the mask
        const float s0 = __shfl_xor_sync(0xffffffffu, odd ? m0 : p0, 1);
        const float s1 = __shfl_xor_sync(0xffffffffu, odd ? m1 : p1, 1);
        float v[4] = {odd ? s0 : m0, odd ? s1 : m1, odd ? p0 : s0, odd ? p1 : s1};
        if (lim >= 0) {
#pragma unroll
          for (int e = 1; e < 4; ++e)
            if (lim < e) v[e] = 0.f;
          const __nv_bfloat162 ah[2] = {
              {__float2bfloat16(v[0]), __float2bfloat16(v[1])}, {__float2bfloat16(v[2]), __float2bfloat16(v[3])}};
          *reinterpret_cast<float2*>(&aqk_h[abase + ii * BT + jj]) = *reinterpret_cast<const float2*>(ah);
          const __nv_bfloat162 al[2] = {
              __floats2bfloat162_rn(v[0] - __bfloat162float(ah[0].x), v[1] - __bfloat162float(ah[0].y)),
              __floats2bfloat162_rn(v[2] - __bfloat162float(ah[1].x), v[3] - __bfloat162float(ah[1].y))};
          *reinterpret_cast<float2*>(&aqk_l[abase + ii * BT + jj]) = *reinterpret_cast<const float2*>(al);
        }
      }
    };
    auto tri_scatter = [&](int si2, const float4* acck) {  // si2 == sj2
      const int r = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
      for (int n8 = 0; n8 < 2; ++n8) {
        const float vals[4] = {acck[n8].x, acck[n8].y, acck[n8].z, acck[n8].w};
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          const int il = r + e2 * 8, jl = n8 * 8 + c2;
          const int ii = si2 * SB + il, tb = il * (il - 1) / 2;
          if (jl < il) tri[si2][tb + jl] = sb[ii] * vals[e2 * 2];
          if (jl + 1 < il) tri[si2][tb + jl + 1] = sb[ii] * vals[e2 * 2 + 1];
        }
      }
    };
    float4 offk[2];
    int offsi = -1, offsj = -1;  // held off-diag A fragments
    if (warp < NSB) {            // diag pairs: si == sj == warp, slot = warp
      float4 acck[2], accq[2];
      pair_mma(warp, warp, acck, accq);
      aqk_scatter(warp, warp, accq);
      tri_scatter(warp, acck);
    }
    __syncthreads();
    {  // phase B production: the 6 off-diag tiles (base recompute, bit-same)
#pragma unroll
      for (int sj = 0; sj < NSB - 1; ++sj)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int j = sj * SB + u + rr * 8, jj = u + rr * 8;
          const float2 kf = __bfloat1622float2(kvz[sj][rr]);
          const bool real = !VL || j < C_act;  // 0*__expf(ovfl) = NaN
          float v0 = real ? kf.x * __expf(a0v[sj] - sgc[j][colp]) : 0.f;
          float v1 = real ? kf.y * __expf(a1v[sj] - sgc[j][colp + 1]) : 0.f;
#pragma unroll
          for (int si = sj + 1; si < NSB; ++si) {
            v0 *= f0[si - 1];
            v1 *= f1[si - 1];
            *reinterpret_cast<__nv_bfloat162*>(&kz[si * (si - 1) / 2 + sj][jj][colp]) = __floats2bfloat162_rn(v0, v1);
          }
        }
    }
    __syncthreads();
    // The solve's only serial pole, paid ONCE per chunk-job instead of per
    // pane per block row: M_bb = (I+L_bb)^-1 by fp32 forward substitution, one
    // thread per column of I, 4 blocks in parallel. tri is final at the
    // barrier above and the off-diag-pair mma below occupies only 6 warps, so
    // 15-deep chain rides two warps that phase leaves idle.
    constexpr int MBW = NSB * (NSB - 1) / 2;  // first idle pair-phase warp
    if (tid >= MBW * 32 && tid < MBW * 32 + NSB * SB) {
      const int td = tid - MBW * 32;
      const int b = td >> 4, j = td & 15;
      float m[SB];
#pragma unroll
      for (int i = 0; i < SB; ++i) {
        float s = i == j ? 1.f : 0.f;
        // m[k] == 0 for k < j, so those FMAs are exact no-ops
#pragma unroll
        for (int k2 = 0; k2 < i; ++k2)
          s -= tri[b][i * (i - 1) / 2 + k2] * m[k2];
        m[i] = s;
      }
#pragma unroll
      for (int i = 0; i < SB; ++i) {  // column j down the A operand
        const bf16 hi = __float2bfloat16(m[i]);
        Mh[b][i][j] = hi;
        Ml[b][i][j] = __float2bfloat16(m[i] - __bfloat162float(hi));
      }
    }
    if (warp < NSB * (NSB - 1) / 2) {  // off-diag pairs, slot = warp
      int si2 = 1;
      while (si2 * (si2 + 1) / 2 <= warp)
        ++si2;
      const int sj2 = warp - si2 * (si2 - 1) / 2;
      float4 acck[2], accq[2];
      pair_mma(si2, warp, acck, accq);
      aqk_scatter(si2, sj2, accq);
      offk[0] = acck[0];
      offk[1] = acck[1];
      offsi = si2;
      offsj = sj2;
    }
    __syncthreads();
    // aC operand pack (the coupling uses cols < 48 only)
    if (offsi >= 0) {  // off-diag fragments -> aC direct (jj < ii always)
      const int r = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
      for (int n8 = 0; n8 < 2; ++n8) {
        const float vals[4] = {offk[n8].x, offk[n8].y, offk[n8].z, offk[n8].w};
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          const int ii = offsi * SB + r + e2 * 8;
          const int jj = offsj * SB + n8 * 8 + c2;
          *reinterpret_cast<__nv_bfloat162*>(&aC_h[ii][jj]) =
              __floats2bfloat162_rn(sb[ii] * vals[e2 * 2], sb[ii] * vals[e2 * 2 + 1]);
        }
      }
    }
    for (int p2 = tid; p2 < BT * 24; p2 += blockDim.x) {  // diag/upper cells
      const int p = p2 * 2;
      const int i = p / 48, j = p % 48;
      if (j / SB < i / SB) continue;  // off-diag lower: fragment-scattered
      float2 av{0.f, 0.f};
      if (j / SB == i / SB) {
        const int il = i % SB, jl = j % SB, tb = il * (il - 1) / 2;
        if (jl < il) av.x = tri[i / SB][tb + jl];
        if (jl + 1 < il) av.y = tri[i / SB][tb + jl + 1];
      }
      *reinterpret_cast<__nv_bfloat162*>(&aC_h[i][j]) = __floats2bfloat162_rn(av.x, av.y);
    }
    __syncthreads();
    const size_t base = ((size_t)c * H + h) * BT * K;
    // The rhs panes are born here, over the dead mma operands: pane 0 = kappa
    // -> P, pane 1 = v * beta -> u0. Rebuilt bit-equal (deterministic __expf,
    // same op sequence) and solved in k1_body's per-column order.
    auto rhs_fill = [&](int pane) {  // k1_body's P2b tap map (colp pair)
      const size_t gp0 = (size_t)(t0 + tid * 2 / K) * H * K + h * K + tid * 2 % K;
      const size_t gst = (size_t)8 * H * K;
      const bf16* px = &v[gp0];  // pane 0's k taps are already in kvz
#pragma unroll
      for (int hb = 0; hb < 2; ++hb) {
        __nv_bfloat162 xr[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int m = hb * 4 + j;
          xr[j] = pane == 0 ? kvz[m >> 1][m & 1]
                            : (!VL || tid * 2 / K + m * 8 < C_act ? *reinterpret_cast<const __nv_bfloat162*>(px)
                                                                  : __floats2bfloat162_rn(0.f, 0.f));  // pad rows
          px += gst;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (tid >> 6) + (hb * 4 + j) * 8;
          const int si = hb * 2 + (j >> 1);
          const float2 xv = __bfloat1622float2(xr[j]);
          const float bi = sb[i];
          if (pane == 0) {
            const float ei0 = __expf(sgc[i][colp] - a0v[si]);
            const float ei1 = __expf(sgc[i][colp + 1] - a1v[si]);
            const float kw0 = xv.x * ei0, kw1 = xv.y * ei1;
            *reinterpret_cast<float2*>(&sU[i][su_c(i, colp)]) = float2{kw0 * (bi * ea0[si]), kw1 * (bi * ea1[si])};
          } else {
            *reinterpret_cast<float2*>(&sU[i][su_c(i, colp)]) = float2{xv.x * bi, xv.y * bi};
          }
        }
      }
    };
#pragma unroll 1
    for (int pane = 0; pane < 2; ++pane) {
      rhs_fill(pane);
      __syncthreads();
      auto pu_store4p = [&](int p) {
        const float* r4 = &sU[p / K][su_c(p / K, p % K)];
        if (pane == 0) {
          const __nv_bfloat162 h01 = __floats2bfloat162_rn(-r4[0], -r4[1]);
          const __nv_bfloat162 h23 = __floats2bfloat162_rn(-r4[2], -r4[3]);
          *reinterpret_cast<uint2*>(&P[base + p]) =
              uint2{*reinterpret_cast<const uint32_t*>(&h01), *reinterpret_cast<const uint32_t*>(&h23)};
        } else {
          *reinterpret_cast<float4*>(&u0[base + p]) = *reinterpret_cast<const float4*>(r4);
        }
      };
      // Blocked forward solve: cross-block coupling as mma (hi only, 16
      // warps x ONE n8 tile), then the in-block apply X_b = M_bb @ rhs_b;
      // solved rows publish transposed hi as the next coupling's B. The
      // whole block row is warp-local (coupling, apply and the upT pack
      // touch only cols 8*warp..+7), so the round syncs at warp scope.
#pragma unroll
      for (int b = 0; b < NSB; ++b) {
        float crr[4] = {};  // coupling correction, B-operand order
        if (b) {
          const int arow = lane & 15, aka = lane >> 4;
          float4 acc = float4{0, 0, 0, 0};
          for (int k16 = 0; k16 < b; ++k16) {
            uint32_t a0, a1, a2, a3, b0, b1;
            ptx::ldmatrix_x4_b16(ptx::to_shared(&aC_h[b * SB + arow][k16 * 16 + aka * 8]), a0, a1, a2, a3);
            ptx::ldmatrix_x2_b16(
                ptx::to_shared(&upT_h[warp * 8 + (lane & 7)][k16 * 16 + ((lane >> 3) & 1) * 8]), b0, b1);
            ptx::mma_m16n8k16_bf16f32(acc, a0, a1, a2, a3, b0, b1);
          }
          // C fragment (row fr(+8), col fc(+1)) -> B operand (k fc(+1,
          // +8,+9), col fr): a permutation of the warp's own 16x8 tile,
          // so no smem round trip — 8 shuffles, and the one fp32
          // subtraction moves into the register feeding the apply's B.
          // corr[k][fr] lives in lane 8*(lane&3) + (fr>>1), regs x/y for
          // k < 8 and z/w for k >= 8, picked on fr&1 == the col parity.
          const int fr = lane >> 2;
          const int sla = (lane & 3) * 8 + (fr >> 1), slb = sla + 4;
          const unsigned fm = 0xffffffffu;
          const float ax = __shfl_sync(fm, acc.x, sla);
          const float ay = __shfl_sync(fm, acc.y, sla);
          const float az = __shfl_sync(fm, acc.z, sla);
          const float aw = __shfl_sync(fm, acc.w, sla);
          const float bx = __shfl_sync(fm, acc.x, slb);
          const float by = __shfl_sync(fm, acc.y, slb);
          const float bz = __shfl_sync(fm, acc.z, slb);
          const float bw = __shfl_sync(fm, acc.w, slb);
          crr[0] = (fr & 1) ? ay : ax;  // k = fc
          crr[1] = (fr & 1) ? by : bx;  // k = fc + 1
          crr[2] = (fr & 1) ? aw : az;  // k = fc + 8
          crr[3] = (fr & 1) ? bw : bz;  // k = fc + 9
        }
        {  // X_b = M_bb @ rhs_b, hi/lo 3-pass: ONE mma per pass per warp,
          // all 16 warps. rhs_b comes straight out of fp32 sU — this
          // warp's own 8 columns, which no other warp touches.
          const int arow = lane & 15, aka = lane >> 4;
          const int fr = lane >> 2, fc = (lane & 3) * 2;
          const int bcol = warp * 8 + fr;  // B operand column (n = fr)
          uint32_t ah[4], al[4], bh[2], bl[2];
          ptx::ldmatrix_x4_b16(ptx::to_shared(&Mh[b][arow][aka * 8]), ah[0], ah[1], ah[2], ah[3]);
          ptx::ldmatrix_x4_b16(ptx::to_shared(&Ml[b][arow][aka * 8]), al[0], al[1], al[2], al[3]);
#pragma unroll
          for (int kp = 0; kp < 2; ++kp) {  // k = fc(+1) and fc+8(+9)
            const int rr0 = b * SB + fc + kp * 8;
            float r0 = sU[rr0][su_c(rr0, bcol)];
            float r1 = sU[rr0 + 1][su_c(rr0 + 1, bcol)];
            if (b) {  // the coupling's fp32 subtraction, in-register
              r0 -= crr[kp * 2];
              r1 -= crr[kp * 2 + 1];
            }
            const __nv_bfloat162 hv = __floats2bfloat162_rn(r0, r1);
            const __nv_bfloat162 lv = __floats2bfloat162_rn(r0 - __bfloat162float(hv.x), r1 - __bfloat162float(hv.y));
            bh[kp] = *reinterpret_cast<const uint32_t*>(&hv);
            bl[kp] = *reinterpret_cast<const uint32_t*>(&lv);
          }
          float4 acc = float4{0, 0, 0, 0};
          ptx::mma_m16n8k16_bf16f32(acc, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
          ptx::mma_m16n8k16_bf16f32(acc, al[0], al[1], al[2], al[3], bh[0], bh[1]);
          ptx::mma_m16n8k16_bf16f32(acc, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
          __syncwarp();  // all lanes' rhs_b reads precede the write
          const float vals[4] = {acc.x, acc.y, acc.z, acc.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int xr = b * SB + fr + (e >> 1) * 8;
            const int xc = warp * 8 + fc + (e & 1);
            if (!(e & 1))  // the n-pair explicitly: stays 64-bit
              *reinterpret_cast<float2*>(&sU[xr][su_c(xr, xc)]) = float2{vals[e], vals[e + 1]};
            if (b + 1 < NSB)  // next coupling's B operand (hi^T)
              upT_h[xc][xr] = __float2bfloat16(vals[e]);
          }
          __syncwarp();  // upT_b feeds this warp's next coupling
        }
      }
      // the apply is warp-local, so nothing streamed out under it: one
      // barrier, then the whole pane publishes
      __syncthreads();
      for (int p = tid * 4; p < BT * K; p += blockDim.x * 4)
        pu_store4p(p);
      __syncthreads();  // sU/upT reuse across panes (pool reuse at job end)
    }
  }  // job loop
}

// Piece-major block order: consecutive blocks are the same piece of adjacent
// heads, i.e. the same token rows — their q/k/v/g lines are contiguous, so a
// wave shares L2. h stays constant within a block (the GM hoists need it).
template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512, 2) k1_tf_builder(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const __grid_constant__ CUtensorMap glog_map,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int T,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    int NP) {
  __shared__ std::conditional_t<RAW, K1SmemTFRaw, K1SmemTF> S;
  float2* rn = nullptr;
  if constexpr (RAW) rn = S.rn;
  const int nc1 = T / BT, bid = blockIdx.x;
  const int h = bid % H, pc = bid / H;
  const int c00 = piece_c0(pc, nc1, NP);
  const int njobs = piece_c0(pc + 1, nc1, NP) - c00;
  k1_tf_body<GM, false, RAW, BSIG>(
      S,
      h * nc1 + c00,
      njobs,
      q,
      kk,
      v,
      glog,
      glog_map,
      beta,
      T,
      H,
      scale,
      P,
      u0,
      kdec,
      qdec,
      aqk_h,
      aqk_l,
      gC,
      a_log,
      dtb,
      lb,
      0,
      0,
      rn);
}
template <int GM = 0, bool RAW = false, bool BSIG = false>
__global__ void __launch_bounds__(512, 2) k1_tf_builder_vl(
    const bf16* __restrict__ q,
    const bf16* __restrict__ kk,
    const bf16* __restrict__ v,
    const bf16* __restrict__ glog,
    const __grid_constant__ CUtensorMap glog_map,
    const float* __restrict__ a_log,
    const float* __restrict__ dtb,
    float lb,
    const float* __restrict__ beta,
    int H,
    float scale,
    bf16* __restrict__ P,
    float* __restrict__ u0,
    bf16* __restrict__ kdec,
    bf16* __restrict__ qdec,
    bf16* __restrict__ aqk_h,
    bf16* __restrict__ aqk_l,
    float* __restrict__ gC,
    const VlPiece* __restrict__ pieces,
    int nc_tot) {
  __shared__ std::conditional_t<RAW, K1SmemTFRaw, K1SmemTF> S;
  float2* rn = nullptr;
  if constexpr (RAW) rn = S.rn;
  const int h = blockIdx.x % H;
  const VlPiece* pc = pieces + blockIdx.x / H;
  const int c00 = pc->cbase + piece_c0(pc->sg, pc->nc, pc->np);
  const int njobs = pc->cbase + piece_c0(pc->sg + 1, pc->nc, pc->np) - c00;
  if (pc->tail)  // partial last chunk: pad-row guards live
    k1_tf_body<GM, true, RAW, BSIG>(
        S,
        h * nc_tot + c00,
        njobs,
        q,
        kk,
        v,
        glog,
        glog_map,
        beta,
        nc_tot * BT,
        H,
        scale,
        P,
        u0,
        kdec,
        qdec,
        aqk_h,
        aqk_l,
        gC,
        a_log,
        dtb,
        lb,
        pc->tokoff,
        pc->tend,
        rn);
  else  // full piece: the eqlen body on T = nc_tot*BT (see VlPiece)
    k1_tf_body<GM, false, RAW, BSIG>(
        S,
        h * nc_tot + c00,
        njobs,
        q,
        kk,
        v,
        glog,
        glog_map,
        beta,
        nc_tot * BT,
        H,
        scale,
        P,
        u0,
        kdec,
        qdec,
        aqk_h,
        aqk_l,
        gC,
        a_log,
        dtb,
        lb,
        pc->tokoff,
        pc->tend,
        rn);
}
// varlen seq0 chain: one CTA per (sequence, head) walking the whole sequence
// from h0 (nseg == 1 — no piece maps, no flags). seqs[s] is the whole-sequence
// entry the host appends after the builder piece table.
__global__ void __launch_bounds__(512) k2_chain_tc_vl(
    const __grid_constant__ CUtensorMap pneg_map,
    const __grid_constant__ CUtensorMap kdt_map,
    const __grid_constant__ CUtensorMap qd_map,
    const __grid_constant__ CUtensorMap aqh_map,
    const __grid_constant__ CUtensorMap aql_map,
    const __grid_constant__ CUtensorMap u0f_map,
    const float* __restrict__ gC,
    const float* __restrict__ h0,
    int H,
    bf16* __restrict__ o,
    float* __restrict__ Sf,
    void* __restrict__ hpc,
    bool hpc_bf16,
    bool hpc_v_first,
    const VlPiece* __restrict__ seqs) {
  __shared__ ChainSmem S;
  const int s = blockIdx.x / H, h = blockIdx.x % H;
  const VlPiece* sq = seqs + s;
  const size_t so = (size_t)s * H * K * K;  // h0/Sf per-sequence rows
  if (sq->tail)                             // only a partial last chunk needs the o-store mask
    chain_body<true>(
        S,
        h,
        0,
        1,
        pneg_map,
        kdt_map,
        qd_map,
        aqh_map,
        aql_map,
        u0f_map,
        gC,
        sq->nc,
        H,
        o,
        Sf + so,
        hpc,
        hpc_bf16,
        hpc_v_first,
        nullptr,
        nullptr,
        h0 + so,
        nullptr,
        sq->cbase,
        0,
        sq->tokoff,
        sq->tend);
  else
    chain_body<false>(
        S,
        h,
        0,
        1,
        pneg_map,
        kdt_map,
        qd_map,
        aqh_map,
        aql_map,
        u0f_map,
        gC,
        sq->nc,
        H,
        o,
        Sf + so,
        hpc,
        hpc_bf16,
        hpc_v_first,
        nullptr,
        nullptr,
        h0 + so,
        nullptr,
        sq->cbase,
        0,
        sq->tokoff,
        sq->tend);
}

// Host side: tensor-map encoders, cached workspace, torch entry point.

#define KDA_CU_CHECK(expr)                                                                             \
  do {                                                                                                 \
    CUresult _e = (expr);                                                                              \
    if (_e != CUDA_SUCCESS) {                                                                          \
      const char* _s = nullptr;                                                                        \
      cuGetErrorString(_e, &_s);                                                                       \
      TORCH_CHECK(false, "CUDA driver call failed at ", __FILE__, ":", __LINE__, ": ", _s ? _s : "?"); \
    }                                                                                                  \
  } while (0)

// TMA maps over K1's bf16 outputs (128B swizzle, K-major tiles)
static CUtensorMap enc2d(void* ptr, uint64_t rows, uint64_t cols, uint32_t brows, uint32_t bcols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 2};
  cuuint32_t bdim[2] = {bcols, brows};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
// fp32 c-map tiles for the tf32 chain prefix: SWZ32 k-major chunks
// (box 8 fp32 x 128 rows; 32B swizzle caps the box inner extent at 8 fp32
// -> 16 boxes per 128x128 tile)
static CUtensorMap enc2df(void* ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 4};
  cuuint32_t bdim[2] = {8, 128};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
// fp32 u0 chunk tiles for the chain: plain linear [BT x K] boxes (the
// consumer is per-thread smem loads, not an mma desc, so no swizzle — and
// only swizzled modes cap the box inner extent)
static CUtensorMap enc2dfn(void* ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 4};
  cuuint32_t bdim[2] = {(cuuint32_t)cols, BT};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
// The tail-free builder's gate input: [BT x K] bf16 boxes out of the
// [T x H*K] gate stream. SWIZZLE_NONE, so the landed tile is row-major with a
// 256 B pitch — byte-for-byte the per-thread staging it replaces. rows == the
// ALLOCATED token count, so a tail chunk's rows past the buffer read as zeros.
static CUtensorMap enc2dgb(void* ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstr[1] = {cols * 2};
  cuuint32_t bdim[2] = {K, BT};
  cuuint32_t estr[2] = {1, 1};
  CUtensorMap m{};
  KDA_CU_CHECK(cuTensorMapEncodeTiled(
      &m,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      ptr,
      gdim,
      gstr,
      bdim,
      estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

// Cached workspace, keyed by (device, T, nc_tot, H, npieces, N). The factor
// tensors are torch allocations (caching allocator -> stream-safe reuse,
// freed with the process). The CUDA tensor maps encode over these tensors'
// data pointers, which are stable for the cache entry's lifetime — so the
// maps are encoded ONCE here rather than per call (this IS the
// (pointer, shape) tensor-map cache; a fresh encode is only host-cheap ~us,
// but per-call re-encode is pure waste when the pointers never change).
// NOTE: aqk_h/aqk_l are zero-FILLED once — k1 only ever writes the masked
// lower-triangular cells, the TMA-read upper cells must stay zero.
// The gate map is the exception: g is a CALLER tensor, so its pointer is only
// stable while the allocator hands back the same block. One-entry memo.
struct Workspace {
  torch::Tensor P, u0, kdec, qdec, aqk_h, aqk_l, gC, pieceL, piecec, pflags;
  torch::Tensor h0z;         // zeros initial state (lazy)
  torch::Tensor pieces_dev;  // varlen piece table (lazy)
  std::vector<int64_t> cu;   // piece-table provenance (varlen)
  CUtensorMap pneg_map, kdt_map, qd_map, aqh_map, aql_map, sl_map, sc_map, u0f_map, gin_map;
  const void* gin_ptr = nullptr;  // what gin_map was encoded over
};

// Resident builder-CTA slots on this GPU, per route: the fused grid's builder
// half is NP*H CTAs of the 2*NP*H launch and co-resides 1/SM; the tail-free
// builder is its own NP*H launch at 2/SM. This is what a builder "wave" costs.
static int64_t builder_slots(bool tf, bool varlen) {
  auto query = [](auto fn) {
    int occ = 0;
    AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 512, 0));
    return int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * std::max(1, occ);
  };
  static const int64_t slots[4] = {
      query(kda_fused<0>), query(kda_fused_vl<0>), query(k1_tf_builder<0>), query(k1_tf_builder_vl<0>)};
  return slots[2 * int(tf) + int(varlen)];
}

// (pieces, max chunks in any piece) for a piece budget npb
static std::pair<int64_t, int64_t> piece_plan(const std::vector<int64_t>& cu, int64_t T, int npb, bool varlen) {
  if (!varlen) {
    const int64_t nc = T / BT;
    const int64_t np = std::min<int64_t>(npb, std::max<int64_t>(1, nc / 2));
    return {np, (nc + np - 1) / np};
  }
  int nc_tot = 0;
  for (size_t s = 0; s + 1 < cu.size(); ++s)
    nc_tot += int((cu[s + 1] - cu[s] + BT - 1) / BT);
  int64_t pieces = 0, mx = 0;
  for (size_t s = 0; s + 1 < cu.size(); ++s) {
    const int ncs = int((cu[s + 1] - cu[s] + BT - 1) / BT);
    const int np = std::min(std::min(npb, std::max(1, ncs / 2)), std::max(1, (2 * npb * ncs + nc_tot) / (2 * nc_tot)));
    pieces += np;
    mx = std::max<int64_t>(mx, (ncs + np - 1) / np);
  }
  return {pieces, mx};
}

// Pieces are persistent builders, each walking max_len chunks serially, so the
// build wall is a bin-packing makespan: ceil(builders/slots) waves of max_len,
// plus per-builder composition. Minimizing
//     waves * max_len * slots + builders
// (makespan scaled into builder units + one composition unit per builder)
// reproduces every swept optimum on a 152-SM GB300: 1kx8-H12 -> 24 pieces
// (-17.1% vs the old fixed 12), 8k/16k-H12 -> 12 (one 0.95x-fill wave, already
// optimal), 8k-H96 -> 3, 1kx8-H96 -> 16, mixed6-H96 -> 14 (measured optimum 12,
// +0.3%). Ties break toward fewer pieces, which is why the H12 eqlen cells stay
// on one wave instead of splitting to two.
static int piece_budget(const std::vector<int64_t>& cu, int64_t T, int64_t H, bool varlen, int64_t slots) {
  int64_t best_cost = -1, best_pieces = 0;
  int best_npb = 1;
  for (int npb = 1; npb <= 64; ++npb) {
    const auto [pieces, mx] = piece_plan(cu, T, npb, varlen);
    const int64_t builders = pieces * H;
    const int64_t waves = (builders + slots - 1) / slots;
    const int64_t cost = waves * mx * slots + builders;
    if (best_cost < 0 || cost < best_cost || (cost == best_cost && pieces < best_pieces)) {
      best_cost = cost;
      best_pieces = pieces;
      best_npb = npb;
    }
  }
  return best_npb;
}

// Measured per-unit costs (us) behind the route pick on this shape family: one
// builder chunk-job costs 10.2 us*SM tail-free vs 16.2 fused (that builder also
// composes its piece map in-tail), one chain chunk-step 2.1 us. The tail-free
// figure tracks that builder's measured wall as it improves: 11.7 before the
// block apply, 11.1 after it (-4.9%), 10.2 after the four data-pipe items
// (-8.1% more). Every battery shape keeps its route across that whole range —
// the crossover cell (vl-prod-H12) only gains margin.
constexpr double C_JOB_TF = 10.2, C_JOB_FUSED = 16.2, C_STEP = 2.1;

// Route pick, derived from the launch geometry like the piece budget above: the
// seq0 pair (tail-free builders, then one chain per (sequence, head) walking
// that whole sequence) vs the fused grid. Each wall is max(throughput term, SM
// critical path): the fused grid's chains trail its builders inside one grid,
// so their step time joins its throughput term and only its longest piece's
// chain is exposed, while seq0's chain is a second launch whose depth is the
// LONGEST SEQUENCE. seq0 therefore wins on many-sequence and high-H shapes and
// loses on long single sequences (measured +50% at 8k-H12), which is what this
// reproduces. The margin is the model's own residual at the crossover: on the
// one shape that was ever near it (4 seqs 8192/8192/4096/4096, H12) the model
// priced seq0 at -2.7% against a measured -1.5%, so a flip must be modelled at
// >=2.5% to cover that 1.2-point optimism. Every other battery shape clears the
// guard by >=21 points either way. A single piece has no composition to skip,
// so it stays fused too.
struct Route {
  bool seq0;
  int npb;
};
static Route pick_route(const std::vector<int64_t>& cu, int64_t T, int64_t H, bool varlen) {
  const double sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t nc_tot = 0, depth = 0;
  if (varlen)
    for (size_t s = 0; s + 1 < cu.size(); ++s) {
      const int64_t ncs = (cu[s + 1] - cu[s] + BT - 1) / BT;
      nc_tot += ncs;
      depth = std::max(depth, ncs);
    }
  else
    nc_tot = depth = T / BT;
  const double jobs = double(nc_tot * H) / sms;  // builder jobs per SM
  struct Arm {
    int npb;
    int64_t pieces;
    double wall;
  };
  auto arm = [&](bool tf) {
    const int64_t slots = builder_slots(tf, varlen);
    const int npb = piece_budget(cu, T, H, varlen, slots);
    const auto [pieces, mx] = piece_plan(cu, T, npb, varlen);
    const double c_job = tf ? C_JOB_TF : C_JOB_FUSED;
    const double share =  // builder CTAs sharing one SM
        std::min(double(slots) / sms, std::max(1.0, double(pieces * H) / sms));
    const double build = std::max(jobs * (tf ? c_job : c_job + C_STEP), double(mx) * share * c_job);
    const double chain = (tf ? std::max(double(depth), jobs) : double(mx)) * C_STEP;
    return Arm{npb, pieces, build + chain};
  };
  const Arm tf = arm(true), fused = arm(false);
  if (tf.pieces > 1 && tf.wall <= 0.975 * fused.wall) return {true, tf.npb};
  return {false, fused.npb};
}

static Workspace&
get_workspace(const torch::Device& dev, int64_t T, int64_t nc, int64_t H, int64_t npieces, int64_t N) {
  // guarded by the GIL (single writer); entries live for the process
  static std::map<std::array<int64_t, 6>, Workspace> cache;
  const std::array<int64_t, 6> key{dev.index(), T, nc, H, npieces, N};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  Workspace ws;
  const auto ob = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  const auto of = torch::TensorOptions().dtype(torch::kFloat).device(dev);
  const auto oi = torch::TensorOptions().dtype(torch::kInt).device(dev);
  const int64_t chk = nc * H * BT * K;
  ws.P = torch::empty({chk}, ob);
  ws.u0 = torch::empty({chk}, of);
  ws.kdec = torch::empty({chk}, ob);
  ws.qdec = torch::empty({chk}, ob);
  ws.aqk_h = torch::zeros({nc * H * BT * BT}, ob);
  ws.aqk_l = torch::zeros({nc * H * BT * BT}, ob);
  ws.gC = torch::empty({nc * H * K}, of);
  // L maps bf16 (the chain widens them to tf32); c maps fp32 (bf16 c
  // failed the 16K o gate)
  ws.pieceL = torch::empty({npieces * H * K * K}, ob);
  ws.piecec = torch::empty({npieces * H * K * K}, of);
  ws.pflags = torch::empty({npieces * H}, oi);
  ws.pneg_map = enc2d(ws.P.data_ptr(), (uint64_t)(nc * H * BT), K, BT, 64);
  ws.kdt_map = enc2d(ws.kdec.data_ptr(), (uint64_t)(nc * H * K), BT, K, 64);
  ws.qd_map = enc2d(ws.qdec.data_ptr(), (uint64_t)(nc * H * BT), K, BT, 64);
  ws.aqh_map = enc2d(ws.aqk_h.data_ptr(), (uint64_t)(nc * H * BT), BT, BT, 64);
  ws.aql_map = enc2d(ws.aqk_l.data_ptr(), (uint64_t)(nc * H * BT), BT, BT, 64);
  ws.sl_map = enc2d(ws.pieceL.data_ptr(), (uint64_t)(npieces * H * K), K, K, 64);
  ws.sc_map = enc2df(ws.piecec.data_ptr(), (uint64_t)(npieces * H * K), K);
  ws.u0f_map = enc2dfn(ws.u0.data_ptr(), (uint64_t)(nc * H * BT), K);
  return cache.emplace(key, std::move(ws)).first->second;
}

static_assert(sizeof(VlPiece) == 8 * sizeof(int));

// kda_prefill_fwd — the single forward entry point.
//   q, k, v : [T, H, 128] bf16 contiguous (flat token stream); q/k l2-normed
//             unless use_qk_l2norm_in_kernel
//   g       : [T, H, 128] bf16 — pre-transformed glog (use_gate_in_kernel =
//             false; narrow fp32 -> bf16 in the wrapper) or RAW gate input
//             (use_gate_in_kernel = true)
//   beta    : [T, H] bf16 (or fp32) — widened to fp32 internally; the
//             activated beta, or its LOGITS under use_beta_sigmoid_in_kernel
//   scale   : q scaling (typically 128**-0.5)
//   initial_state : [N, H, 128, 128] fp32 or None (zeros)
//   cu_seqlens    : int32/int64 [N+1] (any device; host values are needed
//             for the piece table — pass a CPU tensor to avoid the D2H sync)
//             or None => single sequence [0, T]
//   use_gate_in_kernel + A_log [H] f32 + dt_bias [H*128] f32 + safe_gate +
//             lower_bound: the production raw-gate convention (safe_gate
//             false => softplus, true => lower_bound * sigmoid)
//   use_qk_l2norm_in_kernel: q/k raw, l2norm(q)/l2norm(k) fused in (fla's
//             l2norm eps/rounding)
//   use_beta_sigmoid_in_kernel: beta raw (logits), sigmoid(beta) fused in.
//             Independent of use_qk_l2norm_in_kernel, both default false —
//             fla's two flags, fla's meanings
//   h_per_chunk : optional PREALLOCATED [nc_tot, H, 128, 128] fp32 or bf16
//             per-chunk state output (nc_tot = sum_n ceil(len_n / 64) = the
//             kernel's own chunk count), h_v_first = store it [V, K] instead
//             of the native [K, V] (the wrapper's state_v_first)
// Returns (o [T, H, 128] bf16, final_state [N, H, 128, 128] fp32).
static std::tuple<torch::Tensor, torch::Tensor> kda_prefill_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    double scale,
    const std::optional<torch::Tensor>& initial_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    bool use_gate_in_kernel,
    const std::optional<torch::Tensor>& A_log,
    const std::optional<torch::Tensor>& dt_bias,
    bool safe_gate,
    double lower_bound,
    bool use_qk_l2norm_in_kernel,
    bool use_beta_sigmoid_in_kernel,
    const std::optional<torch::Tensor>& h_per_chunk,
    bool h_v_first) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(2) == K, "q must be a CUDA [T, H, 128] tensor, got ", q.sizes());
  const int64_t T = q.size(0), H = q.size(1);
  TORCH_CHECK(T >= 1 && H >= 1, "empty input: T=", T, " H=", H);
  for (auto* t : {&q, &k, &v, &g}) {
    TORCH_CHECK(
        t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->sizes() == q.sizes(),
        "q/k/v/g must be contiguous CUDA bf16 [T, H, 128] with "
        "matching shapes; got ",
        t->sizes(),
        " ",
        t->dtype());
  }
  TORCH_CHECK(
      beta.is_cuda() && beta.is_contiguous() && beta.dim() == 2 && beta.size(0) == T && beta.size(1) == H,
      "beta must be contiguous CUDA [T, H], got ",
      beta.sizes());
  TORCH_CHECK(
      beta.scalar_type() == torch::kBFloat16 || beta.scalar_type() == torch::kFloat,
      "beta must be bf16 or fp32, got ",
      beta.dtype());
  const int gm = use_gate_in_kernel ? (safe_gate ? 2 : 1) : 0;
  const float lb = float(lower_bound);
  const float* alog_p = nullptr;
  const float* dtb_p = nullptr;
  if (use_gate_in_kernel) {
    TORCH_CHECK(A_log && dt_bias, "use_gate_in_kernel=True requires A_log and dt_bias");
    TORCH_CHECK(
        A_log->is_cuda() && A_log->is_contiguous() && A_log->scalar_type() == torch::kFloat && A_log->numel() == H,
        "A_log must be contiguous CUDA fp32 [H], got ",
        A_log->sizes(),
        " ",
        A_log->dtype());
    TORCH_CHECK(
        dt_bias->is_cuda() && dt_bias->is_contiguous() && dt_bias->scalar_type() == torch::kFloat &&
            dt_bias->numel() == H * K,
        "dt_bias must be contiguous CUDA fp32 with H*128 "
        "elements, got ",
        dt_bias->sizes(),
        " ",
        dt_bias->dtype());
    alog_p = A_log->data_ptr<float>();
    dtb_p = dt_bias->data_ptr<float>();
  }
  // route: explicit cu_seqlens OR a ragged T runs the varlen grid; an
  // aligned single sequence takes the eqlen fused grid
  std::vector<int64_t> cu;
  if (cu_seqlens) {
    TORCH_CHECK(
        cu_seqlens->dim() == 1 && cu_seqlens->numel() >= 2, "cu_seqlens must be 1-D [N+1], got ", cu_seqlens->sizes());
    const auto cu_cpu =  // D2H sync when given on device
        cu_seqlens->to(torch::kCPU).to(torch::kLong).contiguous();
    const int64_t* p = cu_cpu.data_ptr<int64_t>();
    cu.assign(p, p + cu_cpu.numel());
    TORCH_CHECK(
        cu.front() == 0 && cu.back() == T,
        "cu_seqlens must span [0, T=",
        T,
        "], got [",
        cu.front(),
        ", ",
        cu.back(),
        "]");
    for (size_t s = 0; s + 1 < cu.size(); ++s)
      TORCH_CHECK(
          cu[s] < cu[s + 1],
          "cu_seqlens must be strictly increasing (empty "
          "sequences unsupported) at index ",
          s);
  } else if (T % BT != 0) {
    cu = {0, T};  // ragged single sequence -> varlen grid
  }
  const bool varlen = !cu.empty();
  const int64_t N = varlen ? int64_t(cu.size()) - 1 : 1;
  TORCH_CHECK(!(varlen && N > 1) || cu_seqlens, "internal: synthetic cu is always single-sequence");

  // auto-NP (eqlen): NP = min(npb, max(1, nc/2)). piece table (varlen):
  // that piece budget is GLOBAL (sequences already parallelize the grid;
  // a per-seq NP over-fills it) — pro-rated by nc_s, round-half-up,
  // clamped to the per-seq cap; N=1 reduces exactly to the eqlen NP
  const Route route = pick_route(cu, T, H, varlen);
  const int npb = route.npb;
  std::vector<VlPiece> pieces;
  int64_t nc = 0, npieces = 0;
  if (varlen) {
    int nc_tot = 0;
    for (size_t s = 0; s + 1 < cu.size(); ++s)
      nc_tot += int((cu[s + 1] - cu[s] + BT - 1) / BT);
    std::vector<VlPiece> seqs;  // seq0's chains: one whole-sequence piece
    int nc_run = 0;
    for (size_t s = 0; s + 1 < cu.size(); ++s) {
      const int len = int(cu[s + 1] - cu[s]);
      const int ncs = (len + BT - 1) / BT;
      const int np =
          std::min(std::min(npb, std::max(1, ncs / 2)), std::max(1, (2 * npb * ncs + nc_tot) / (2 * nc_tot)));
      const int tokoff = int(cu[s]) - nc_run * BT;
      for (int sg = 0; sg < np; ++sg)
        pieces.push_back(VlPiece{int(s), nc_run, ncs, np, sg, tokoff, int(cu[s + 1]), sg == np - 1 && len % BT != 0});
      seqs.push_back(VlPiece{int(s), nc_run, ncs, 1, 0, tokoff, int(cu[s + 1]), len % BT != 0});
      nc_run += ncs;
    }
    nc = nc_run;
    npieces = int64_t(pieces.size());
    pieces.insert(pieces.end(), seqs.begin(), seqs.end());
  } else {
    nc = T / BT;
    npieces = std::min<int64_t>(npb, std::max<int64_t>(1, nc / 2));
  }

  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Workspace& ws = get_workspace(q.device(), T, nc, H, npieces, N);

  // initial state (None -> cached zeros; kernels only READ h0)
  const float* h0_p;
  if (initial_state) {
    TORCH_CHECK(
        initial_state->is_cuda() && initial_state->is_contiguous() && initial_state->scalar_type() == torch::kFloat &&
            initial_state->sizes() == torch::IntArrayRef({N, H, int64_t(K), int64_t(K)}),
        "initial_state must be contiguous CUDA fp32 [N=",
        N,
        ", H=",
        H,
        ", 128, 128], got ",
        initial_state->sizes(),
        " ",
        initial_state->dtype());
    h0_p = initial_state->data_ptr<float>();
  } else {
    if (!ws.h0z.defined())
      ws.h0z =
          torch::zeros({N, H, int64_t(K), int64_t(K)}, torch::TensorOptions().dtype(torch::kFloat).device(q.device()));
    h0_p = ws.h0z.data_ptr<float>();
  }

  // outputs (fresh each call — everything else is workspace)
  auto o = torch::empty({T, H, int64_t(K)}, torch::TensorOptions().dtype(torch::kBFloat16).device(q.device()));
  auto Sf =
      torch::empty({N, H, int64_t(K), int64_t(K)}, torch::TensorOptions().dtype(torch::kFloat).device(q.device()));

  // per-chunk states (caller-allocated): row (n, j) = chunk_offset[n] + j is
  // sequence n's state after exactly j*BT tokens (j = 0 == initial_state), so
  // a consumer can only snapshot at multiples of BT == 64. The last boundary
  // is NOT here — it is final_state.
  void* hpc_p = nullptr;
  bool hpc_bf16 = false;
  if (h_per_chunk) {
    const auto& hp = *h_per_chunk;
    TORCH_CHECK(
        hp.is_cuda() && hp.device() == q.device() && hp.is_contiguous() &&
            hp.sizes() == torch::IntArrayRef({nc, H, int64_t(K), int64_t(K)}),
        "h_per_chunk must be a contiguous [nc=",
        nc,
        ", H=",
        H,
        ", 128, 128] tensor on ",
        q.device(),
        ", got ",
        hp.sizes(),
        " on ",
        hp.device());
    TORCH_CHECK(
        hp.scalar_type() == torch::kFloat || hp.scalar_type() == torch::kBFloat16,
        "h_per_chunk must be fp32 or bf16, got ",
        hp.dtype());
    hpc_bf16 = hp.scalar_type() == torch::kBFloat16;
    hpc_p = hp.data_ptr();
  }

  const torch::Tensor beta_f =  // fp32 widen (exact; kernels read fp32)
      beta.scalar_type() == torch::kFloat ? beta : beta.to(torch::kFloat);

  const bf16* q_p = reinterpret_cast<const bf16*>(q.const_data_ptr());
  const bf16* k_p = reinterpret_cast<const bf16*>(k.const_data_ptr());
  const bf16* v_p = reinterpret_cast<const bf16*>(v.const_data_ptr());
  const bf16* g_p = reinterpret_cast<const bf16*>(g.const_data_ptr());
  const float* beta_p = beta_f.data_ptr<float>();
  bf16* o_p = reinterpret_cast<bf16*>(o.data_ptr());
  float* Sf_p = Sf.data_ptr<float>();
  bf16* P_p = reinterpret_cast<bf16*>(ws.P.data_ptr());
  bf16* kdec_p = reinterpret_cast<bf16*>(ws.kdec.data_ptr());
  bf16* qdec_p = reinterpret_cast<bf16*>(ws.qdec.data_ptr());
  bf16* aqh_p = reinterpret_cast<bf16*>(ws.aqk_h.data_ptr());
  bf16* aql_p = reinterpret_cast<bf16*>(ws.aqk_l.data_ptr());
  float* u0_p = ws.u0.data_ptr<float>();
  float* gC_p = ws.gC.data_ptr<float>();
  bf16* pL_p = reinterpret_cast<bf16*>(ws.pieceL.data_ptr());
  float* pc_p = ws.piecec.data_ptr<float>();
  uint32_t* pf_p = reinterpret_cast<uint32_t*>(ws.pflags.data_ptr());
  const float scl = float(scale);

  // (gate mode, q/k raw, beta logits) — the three INDEPENDENT compile-time
  // input conventions the bodies fold in; f takes all three as constants
  auto gm_dispatch = [&](auto&& f) {
    auto bsig = [&](auto gmv, auto rawv) {
      if (use_beta_sigmoid_in_kernel)
        f(gmv, rawv, std::true_type{});
      else
        f(gmv, rawv, std::false_type{});
    };
    auto raw = [&](auto gmv) {
      if (use_qk_l2norm_in_kernel)
        bsig(gmv, std::true_type{});
      else
        bsig(gmv, std::false_type{});
    };
    if (gm == 2)
      raw(std::integral_constant<int, 2>{});
    else if (gm == 1)
      raw(std::integral_constant<int, 1>{});
    else
      raw(std::integral_constant<int, 0>{});
  };

  const VlPiece* pt_p = nullptr;
  if (varlen) {
    // per-cu piece table upload (only when cu changes for this key): the
    // npieces builder pieces, then seq0's N whole-sequence chain entries
    if (ws.cu != cu) {
      static_assert(std::is_standard_layout_v<VlPiece>);
      const auto hp =
          torch::from_blob(pieces.data(), {(npieces + N) * 8}, torch::TensorOptions().dtype(torch::kInt)).clone();
      if (!ws.pieces_dev.defined())
        ws.pieces_dev = torch::empty({(npieces + N) * 8}, torch::TensorOptions().dtype(torch::kInt).device(q.device()));
      ws.pieces_dev.copy_(hp);
      ws.cu = cu;
    }
    pt_p = reinterpret_cast<const VlPiece*>(ws.pieces_dev.const_data_ptr());
  }
  if (route.seq0 && g_p != ws.gin_ptr) {  // only the tail-free builder reads
    ws.gin_map = enc2dgb(const_cast<bf16*>(g_p), (uint64_t)T, (uint64_t)(H * K));
    ws.gin_ptr = g_p;
  }

  // raw modes pass g (RAW graw) as glog: k1 stages it through the same bf16
  // slot and fuses the transform (GM 1/2), on every route
  if (route.seq0 && varlen) {
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      k1_tf_builder_vl<gmv(), rawv(), bsv()><<<int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          ws.gin_map,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          pt_p,
          int(nc));
    });
    k2_chain_tc_vl<<<int(N * H), 512, 0, stream>>>(
        ws.pneg_map,
        ws.kdt_map,
        ws.qd_map,
        ws.aqh_map,
        ws.aql_map,
        ws.u0f_map,
        gC_p,
        h0_p,
        int(H),
        o_p,
        Sf_p,
        hpc_p,
        hpc_bf16,
        h_v_first,
        pt_p + npieces);
  } else if (route.seq0) {
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      k1_tf_builder<gmv(), rawv(), bsv()><<<int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          ws.gin_map,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          int(npieces));
    });
    k2_chain_tc<<<int(H), 512, 0, stream>>>(
        ws.pneg_map,
        ws.kdt_map,
        ws.qd_map,
        ws.aqh_map,
        ws.aql_map,
        ws.sl_map,
        ws.sc_map,
        ws.u0f_map,
        gC_p,
        h0_p,
        int(nc),
        int(H),
        o_p,
        Sf_p,
        hpc_p,
        hpc_bf16,
        h_v_first);
  } else if (varlen) {
    AT_CUDA_CHECK(cudaMemsetAsync(pf_p, 0, size_t(npieces * H) * 4, stream));
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      kda_fused_vl<gmv(), rawv(), bsv()><<<2 * int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          pL_p,
          pc_p,
          pt_p,
          int(npieces),
          int(nc),
          ws.pneg_map,
          ws.kdt_map,
          ws.qd_map,
          ws.aqh_map,
          ws.aql_map,
          ws.u0f_map,
          ws.sl_map,
          ws.sc_map,
          h0_p,
          o_p,
          Sf_p,
          hpc_p,
          hpc_bf16,
          h_v_first,
          pf_p);
    });
  } else if (npieces > 1) {  // eqlen default: one fused trailing grid
    AT_CUDA_CHECK(cudaMemsetAsync(pf_p, 0, size_t(npieces * H) * 4, stream));
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      kda_fused<gmv(), rawv(), bsv()><<<2 * int(npieces * H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          pL_p,
          pc_p,
          int(npieces),
          ws.pneg_map,
          ws.kdt_map,
          ws.qd_map,
          ws.aqh_map,
          ws.aql_map,
          ws.u0f_map,
          ws.sl_map,
          ws.sc_map,
          h0_p,
          o_p,
          Sf_p,
          hpc_p,
          hpc_bf16,
          h_v_first,
          pf_p);
    });
  } else {  // eqlen nc < 4: NP == 1 two-kernel path (no piece maps)
    gm_dispatch([&](auto gmv, auto rawv, auto bsv) {
      k1_factors_mma<gmv(), rawv(), bsv()><<<int(H), 512, 0, stream>>>(
          q_p,
          k_p,
          v_p,
          g_p,
          alog_p,
          dtb_p,
          lb,
          beta_p,
          int(T),
          int(H),
          scl,
          P_p,
          u0_p,
          kdec_p,
          qdec_p,
          aqh_p,
          aql_p,
          gC_p,
          nullptr,
          nullptr,
          1);
    });
    k2_chain_tc<<<int(H), 512, 0, stream>>>(
        ws.pneg_map,
        ws.kdt_map,
        ws.qd_map,
        ws.aqh_map,
        ws.aql_map,
        ws.sl_map,
        ws.sc_map,
        ws.u0f_map,
        gC_p,
        h0_p,
        int(nc),
        int(H),
        o_p,
        Sf_p,
        hpc_p,
        hpc_bf16,
        h_v_first);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, Sf};
}

}  // namespace kda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "kda_prefill_fwd",
      &kda::kda_prefill_fwd,
      "KDA chunked prefill forward (inference): returns (o, final_state)",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("g"),
      py::arg("beta"),
      py::arg("scale"),
      py::arg("initial_state") = std::nullopt,
      py::arg("cu_seqlens") = std::nullopt,
      py::arg("use_gate_in_kernel") = false,
      py::arg("A_log") = std::nullopt,
      py::arg("dt_bias") = std::nullopt,
      py::arg("safe_gate") = false,
      py::arg("lower_bound") = -5.0,
      py::arg("use_qk_l2norm_in_kernel") = false,
      py::arg("use_beta_sigmoid_in_kernel") = false,
      py::arg("h_per_chunk") = std::nullopt,
      py::arg("h_v_first") = false);
}
