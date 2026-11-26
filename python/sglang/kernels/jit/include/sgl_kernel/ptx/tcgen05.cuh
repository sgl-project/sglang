// tcgen05 (SM100 tensor core) PTX wrappers and the UMMA descriptors they take. A kernel picks its CTA group once,
// `using tcgen05 = device::ptx::TCGen05<1>;` in its namespace, then calls `tcgen05::alloc(...)`:
//   - TCGen05Common: what does not depend on the CTA group (fences, TMEM loads / stores, descriptors);
//   - TCGen05<kCtaGroup>: alloc / dealloc, commit, smem -> TMEM copies and the MMA, for cta_group::kCtaGroup.
// The issue instructions (commit, cp, mma) are single-thread: the caller elects the lane. alloc / dealloc /
// relinquish / ld / st / wait are warp-collective (.sync.aligned).

#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/barrier.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace device::ptx {

// Operand plumbing for the TMEM ld / st wrappers below. nvcc rejects symbolic asm operand names, so operands are
// positional: REGS_n is the PTX vector body "%0, ..., %(n-1)", OPS_n(c, r, o) the n constraints c(r[o + i]).
// clang-format off
#define SGL_TCGEN05_REGS_1 "%0"
#define SGL_TCGEN05_REGS_2 SGL_TCGEN05_REGS_1 ", %1"
#define SGL_TCGEN05_REGS_4 SGL_TCGEN05_REGS_2 ", %2, %3"
#define SGL_TCGEN05_REGS_8 SGL_TCGEN05_REGS_4 ", %4, %5, %6, %7"
#define SGL_TCGEN05_REGS_16 SGL_TCGEN05_REGS_8 ", %8, %9, %10, %11, %12, %13, %14, %15"
#define SGL_TCGEN05_REGS_32 SGL_TCGEN05_REGS_16 ", %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
#define SGL_TCGEN05_REGS_64 SGL_TCGEN05_REGS_32 ", %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
#define SGL_TCGEN05_REGS_128 SGL_TCGEN05_REGS_64 ", %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"
#define SGL_TCGEN05_OPS_1(c, r, o) c((r)[o])
#define SGL_TCGEN05_OPS_2(c, r, o) SGL_TCGEN05_OPS_1(c, r, o), SGL_TCGEN05_OPS_1(c, r, (o) + 1)
#define SGL_TCGEN05_OPS_4(c, r, o) SGL_TCGEN05_OPS_2(c, r, o), SGL_TCGEN05_OPS_2(c, r, (o) + 2)
#define SGL_TCGEN05_OPS_8(c, r, o) SGL_TCGEN05_OPS_4(c, r, o), SGL_TCGEN05_OPS_4(c, r, (o) + 4)
#define SGL_TCGEN05_OPS_16(c, r, o) SGL_TCGEN05_OPS_8(c, r, o), SGL_TCGEN05_OPS_8(c, r, (o) + 8)
#define SGL_TCGEN05_OPS_32(c, r, o) SGL_TCGEN05_OPS_16(c, r, o), SGL_TCGEN05_OPS_16(c, r, (o) + 16)
#define SGL_TCGEN05_OPS_64(c, r, o) SGL_TCGEN05_OPS_32(c, r, o), SGL_TCGEN05_OPS_32(c, r, (o) + 32)
#define SGL_TCGEN05_OPS_128(c, r, o) SGL_TCGEN05_OPS_64(c, r, o), SGL_TCGEN05_OPS_64(c, r, (o) + 64)
// "%(n+1)": ld.red's taddr, after the n registers and redval.
#define SGL_TCGEN05_NEXT_2 "%3"
#define SGL_TCGEN05_NEXT_4 "%5"
#define SGL_TCGEN05_NEXT_8 "%9"
#define SGL_TCGEN05_NEXT_16 "%17"
#define SGL_TCGEN05_NEXT_32 "%33"
#define SGL_TCGEN05_NEXT_64 "%65"
#define SGL_TCGEN05_NEXT_128 "%129"
// F(n, args) for each .num; .red requires .num >= .x2.
#define SGL_TCGEN05_FOR_NUM_X2(F, ...) F(2, __VA_ARGS__) F(4, __VA_ARGS__) F(8, __VA_ARGS__) F(16, __VA_ARGS__) F(32, __VA_ARGS__) F(64, __VA_ARGS__) F(128, __VA_ARGS__)
#define SGL_TCGEN05_FOR_NUM(F, ...) F(1, __VA_ARGS__) SGL_TCGEN05_FOR_NUM_X2(F, __VA_ARGS__)
// One `if constexpr (N == n)` per .num; N, r, red and taddr are the enclosing wrapper's.
#define SGL_TCGEN05_LD(n, ...) if constexpr (N == n) asm volatile("tcgen05.ld.sync.aligned.32x32b.x" #n ".b32 {" SGL_TCGEN05_REGS_##n "}, [%" #n "];" : SGL_TCGEN05_OPS_##n("=r", r, 0) : "r"(taddr));
#define SGL_TCGEN05_ST(n, ...) if constexpr (N == n) asm volatile("tcgen05.st.sync.aligned.32x32b.x" #n ".b32 [%" #n "], {" SGL_TCGEN05_REGS_##n "};" :: SGL_TCGEN05_OPS_##n("r", r, 0), "r"(taddr) : "memory");
#define SGL_TCGEN05_LD_RED(n, qual) if constexpr (N == n) asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x" #n qual " {" SGL_TCGEN05_REGS_##n "}, %" #n ", [" SGL_TCGEN05_NEXT_##n "];" : SGL_TCGEN05_OPS_##n("=r", r, 0), "=r"(red) : "r"(taddr));
#define SGL_TCGEN05_LD_RED_BY_OP(type_qual) if constexpr (OP == ReductionOp::MIN) { SGL_TCGEN05_FOR_NUM_X2(SGL_TCGEN05_LD_RED, ".min" type_qual) } else { SGL_TCGEN05_FOR_NUM_X2(SGL_TCGEN05_LD_RED, ".max" type_qual) }
// clang-format on

struct TCGen05Common {
  // ---- ordering ------------------------------------------------------------

  SGL_DEVICE static void fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  }
  SGL_DEVICE static void fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }

  // ---- TMEM <-> registers --------------------------------------------------

  SGL_DEVICE static void wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  }
  SGL_DEVICE static void wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  }
  /// N (1 / 2 / ... / 128) consecutive 32-bit columns of this warp's 32 TMEM lanes (lane quadrant warp_id % 4, in
  /// `taddr` bits [16, 32)) into the register block `dst`; complete after wait_ld().
  template <int N>
  SGL_DEVICE static void ld_32x32b(uint32_t taddr, void* dst) {
    static_assert(N >= 1 && N <= 128 && (N & (N - 1)) == 0);
    const auto r = static_cast<uint32_t*>(dst);
    SGL_TCGEN05_FOR_NUM(SGL_TCGEN05_LD)
  }
  /// ld_32x32b that also returns the per-lane MIN / MAX over the N (>= 2) loaded columns; T is float (optionally
  /// over |x| with kAbs, NaN-propagating with kNaN), uint32_t or int32_t.
  template <int N, ReductionOp OP, typename T = float, bool kAbs = false, bool kNaN = false>
  SGL_DEVICE static T ld_red_32x32b(uint32_t taddr, void* dst) {
    static_assert(N >= 2 && N <= 128 && (N & (N - 1)) == 0);
    static_assert(OP == ReductionOp::MIN || OP == ReductionOp::MAX);
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>);
    // PTX ISA 8.8: sm_103f / sm_110f only, not sm_100.
    static_assert(N > 0 && (SGL_CUDA_ARCH == 1030 || SGL_CUDA_ARCH == 1100), "tcgen05.ld.red needs sm_103 or sm_110");
    const auto r = static_cast<uint32_t*>(dst);
    uint32_t red;
    if constexpr (std::is_same_v<T, uint32_t>) {
      static_assert(!kAbs && !kNaN);
      SGL_TCGEN05_LD_RED_BY_OP(".u32")
    } else if constexpr (std::is_same_v<T, int32_t>) {
      static_assert(!kAbs && !kNaN);
      SGL_TCGEN05_LD_RED_BY_OP(".s32")
    } else if constexpr (kAbs && kNaN) {
      SGL_TCGEN05_LD_RED_BY_OP(".abs.NaN.f32")
    } else if constexpr (kAbs) {
      SGL_TCGEN05_LD_RED_BY_OP(".abs.f32")
    } else if constexpr (kNaN) {
      SGL_TCGEN05_LD_RED_BY_OP(".NaN.f32")
    } else {
      SGL_TCGEN05_LD_RED_BY_OP(".f32")
    }
    if constexpr (std::is_same_v<T, float>) {
      return __uint_as_float(red);
    } else {
      return static_cast<T>(red);
    }
  }
  /// The register block `src` into N (1 / 2 / ... / 128) consecutive 32-bit columns of this warp's 32 TMEM lanes
  /// (lane quadrant warp_id % 4, in `taddr` bits [16, 32)); complete after wait_st().
  template <int N>
  SGL_DEVICE static void st_32x32b(uint32_t taddr, const void* src) {
    static_assert(N >= 1 && N <= 128 && (N & (N - 1)) == 0);
    const auto r = static_cast<const uint32_t*>(src);
    SGL_TCGEN05_FOR_NUM(SGL_TCGEN05_ST)
  }

  // ---- UMMA descriptors ----------------------------------------------------

  /// Shared-memory operand descriptor (addresses and offsets in 16 B units).
  union SmemDesc {
    struct {
      uint64_t start_address : 14;        // [ 0,14)
      uint64_t : 2;                       // [14,16)
      uint64_t leading_byte_offset : 14;  // [16,30) stride between K atoms
      uint64_t : 2;                       // [30,32)
      uint64_t stride_byte_offset : 14;   // [32,46) stride between MN atoms
      uint64_t version : 2;               // [46,48) 1 on Blackwell
      uint64_t : 13;                      // [48,61)
      uint64_t layout_type : 3;           // [61,64) 0: none, 2: 128 B swizzle
    };
    struct {
      uint32_t lo_bits;
      uint32_t hi_bits;
    };
    uint64_t bits;

    /// The operand window slid by `bytes`: one 32-bit add on the low word, as the 14-bit address field never
    /// carries (smem < 256 KiB).
    SGL_DEVICE SmemDesc advance_bytes(uint32_t bytes) const {
      SmemDesc desc = *this;
      desc.lo_bits += bytes >> 4;
      return desc;
    }
  };
  static_assert(sizeof(SmemDesc) == 8);

  SGL_DEVICE static SmemDesc make_smem_desc(uint32_t layout_type, const void* smem, uint32_t sbo_bytes) {
    SmemDesc desc{};
    desc.start_address = to_shared(smem) >> 4;
    desc.stride_byte_offset = sbo_bytes >> 4;
    desc.version = 1;
    desc.layout_type = layout_type;
    return desc;
  }
  /// K-major, 128 B swizzle, 8-row core groups 1024 B apart.
  SGL_DEVICE static SmemDesc make_desc_kmajor_sw128(const void* smem) {
    return make_smem_desc(2, smem, 1024);
  }
  /// One 32-row x 16-byte scale chunk for cp_32x128b_warpx4.
  SGL_DEVICE static SmemDesc make_desc_sf_chunk(const void* smem) {
    return make_smem_desc(0, smem, 128);
  }
  /// Byte offset of 16 B chunk `chunk16` of row `row` in a 128 B-swizzled K-major tile (8-row groups 1024 B apart).
  SGL_DEVICE static uint32_t sw128_off(uint32_t row, uint32_t chunk16) {
    return (row >> 3) * 1024u + (row & 7u) * 128u + ((chunk16 ^ (row & 7u)) << 4);
  }
  /// K-major, 64 B swizzle, 8-row core groups 512 B apart: the TMA SWIZZLE_64B layout of 64 B rows.
  SGL_DEVICE static SmemDesc make_desc_kmajor_sw64(const void* smem) {
    return make_smem_desc(4, smem, 512);
  }
  /// Byte offset of 16 B chunk `chunk16` of row `row` in a 64 B-swizzled K-major tile of 64 B rows.
  SGL_DEVICE static uint32_t sw64_off(uint32_t row, uint32_t chunk16) {
    return row * 64u + ((chunk16 ^ ((row >> 1) & 3u)) << 4);
  }

  /// kind::mxf8f6f4 block-scaled instruction descriptor.
  union MxInstrDesc {
    struct {
      uint32_t : 4;               // [ 0, 4) dense
      uint32_t b_sf_id : 2;       // [ 4, 6) B's scale byte
      uint32_t : 1;               // [ 6, 7)
      uint32_t a_format : 3;      // [ 7,10) 0: e4m3
      uint32_t b_format : 3;      // [10,13) 0: e4m3
      uint32_t : 4;               // [13,17) no negate, both K-major
      uint32_t n_dim : 6;         // [17,23) N >> 3
      uint32_t scale_format : 1;  // [23,24) 1: UE8M0
      uint32_t m_dim : 5;         // [24,29) M >> 4
      uint32_t a_sf_id : 2;       // [29,31) A's scale byte
      uint32_t : 1;               // [31,32)
    };
    uint32_t bits;

    /// The same descriptor reading scale byte `sf_id` (the 32-K block within a 128-K scale group) of A and B.
    SGL_DEVICE MxInstrDesc with_sf_id(uint32_t sf_id) const {
      MxInstrDesc desc = *this;
      desc.a_sf_id = sf_id;
      desc.b_sf_id = sf_id;
      return desc;
    }
  };
  static_assert(sizeof(MxInstrDesc) == 4);

  /// e4m3 x e4m3, f32 accumulate, UE8M0 scales, both K-major; M is the whole MMA's (both CTAs' for cta_group::2).
  template <uint32_t M, uint32_t N>
  SGL_DEVICE static MxInstrDesc make_instr_desc_mxf8() {
    static_assert((M == 64 || M == 128 || M == 256) && N % 8 == 0 && N >= 8 && N <= 256);
    MxInstrDesc desc{};
    desc.n_dim = N >> 3;
    desc.scale_format = 1;
    desc.m_dim = M >> 4;
    return desc;
  }

  /// e2m1 x e2m1, f32 accumulate, UE8M0 scales, both K-major (kind::mxf4 formats: 1 = E2M1).
  template <uint32_t M, uint32_t N>
  SGL_DEVICE static MxInstrDesc make_instr_desc_mxf4() {
    static_assert((M == 64 || M == 128) && N % 8 == 0 && N >= 8 && N <= 256);
    MxInstrDesc desc{};
    desc.a_format = 1;
    desc.b_format = 1;
    desc.n_dim = N >> 3;
    desc.scale_format = 1;
    desc.m_dim = M >> 4;
    return desc;
  }

  /// kind::f8f6f4 (unscaled) instruction descriptor.
  union InstrDesc {
    struct {
      uint32_t : 4;           // [ 0, 4) dense, no saturate
      uint32_t c_format : 2;  // [ 4, 6) 1: f32
      uint32_t : 1;           // [ 6, 7)
      uint32_t a_format : 3;  // [ 7,10) 0: e4m3
      uint32_t b_format : 3;  // [10,13) 0: e4m3
      uint32_t : 4;           // [13,17) no negate, both K-major
      uint32_t n_dim : 6;     // [17,23) N >> 3
      uint32_t : 1;           // [23,24)
      uint32_t m_dim : 5;     // [24,29) M >> 4
      uint32_t : 3;           // [29,32)
    };
    uint32_t bits;
  };
  static_assert(sizeof(InstrDesc) == 4);

  /// e4m3 x e4m3, f32 accumulate, both K-major, no scales.
  template <uint32_t M, uint32_t N>
  SGL_DEVICE static InstrDesc make_instr_desc_f8() {
    static_assert((M == 64 || M == 128 || M == 256) && N % 8 == 0 && N >= 8 && N <= 256);
    InstrDesc desc{};
    desc.c_format = 1;
    desc.n_dim = N >> 3;
    desc.m_dim = M >> 4;
    return desc;
  }
};

#undef SGL_TCGEN05_REGS_1
#undef SGL_TCGEN05_REGS_2
#undef SGL_TCGEN05_REGS_4
#undef SGL_TCGEN05_REGS_8
#undef SGL_TCGEN05_REGS_16
#undef SGL_TCGEN05_REGS_32
#undef SGL_TCGEN05_REGS_64
#undef SGL_TCGEN05_REGS_128
#undef SGL_TCGEN05_OPS_1
#undef SGL_TCGEN05_OPS_2
#undef SGL_TCGEN05_OPS_4
#undef SGL_TCGEN05_OPS_8
#undef SGL_TCGEN05_OPS_16
#undef SGL_TCGEN05_OPS_32
#undef SGL_TCGEN05_OPS_64
#undef SGL_TCGEN05_OPS_128
#undef SGL_TCGEN05_NEXT_2
#undef SGL_TCGEN05_NEXT_4
#undef SGL_TCGEN05_NEXT_8
#undef SGL_TCGEN05_NEXT_16
#undef SGL_TCGEN05_NEXT_32
#undef SGL_TCGEN05_NEXT_64
#undef SGL_TCGEN05_NEXT_128
#undef SGL_TCGEN05_FOR_NUM_X2
#undef SGL_TCGEN05_FOR_NUM
#undef SGL_TCGEN05_LD
#undef SGL_TCGEN05_ST
#undef SGL_TCGEN05_LD_RED
#undef SGL_TCGEN05_LD_RED_BY_OP

template <uint32_t kCtaGroup>
struct TCGen05 : TCGen05Common {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2);

  // ---- TMEM allocation -----------------------------------------------------

  /// Writes the base TMEM address of `n_cols` columns (a power of two >= 32) to `dst` in smem.
  SGL_DEVICE static void alloc(uint32_t* dst, uint32_t n_cols) {
    asm volatile(
        "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(to_shared(dst)),
        "r"(n_cols),
        "n"(kCtaGroup));
  }
  /// Lets the SM's other CTAs allocate: without it a second CTA's alloc never returns.
  SGL_DEVICE static void relinquish_alloc_permit() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::%0.sync.aligned;" ::"n"(kCtaGroup));
  }
  SGL_DEVICE static void dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols), "n"(kCtaGroup));
  }

  // ---- completion ----------------------------------------------------------

  /// Arrives (count 1) on `bar` once every cp / mma this thread issued before it has completed.
  SGL_DEVICE static void commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(to_shared(bar)),
                 "n"(kCtaGroup)
                 : "memory");
  }
  /// The same, arriving on `bar` at the same smem offset in every CTA of `cta_mask` (cluster ranks).
  SGL_DEVICE static void commit_multicast(uint64_t* bar, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
            to_shared(bar)),
        "h"(cta_mask),
        "n"(kCtaGroup)
        : "memory");
  }

  // ---- smem -> TMEM copies (`sdesc`: an SmemDesc of the source) ------------

  SGL_DEVICE static void cp_128x256b(uint32_t taddr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::%2.128x256b [%0], %1;" ::"r"(taddr), "l"(sdesc), "n"(kCtaGroup) : "memory");
  }
  SGL_DEVICE static void cp_128x128b(uint32_t taddr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::%2.128x128b [%0], %1;" ::"r"(taddr), "l"(sdesc), "n"(kCtaGroup) : "memory");
  }
  SGL_DEVICE static void cp_4x256b(uint32_t taddr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::%2.4x256b [%0], %1;" ::"r"(taddr), "l"(sdesc), "n"(kCtaGroup) : "memory");
  }
  /// 32 rows x 16 B, replicated into all 4 lane quadrants: the block-scale layout of the MMA.
  SGL_DEVICE static void cp_32x128b_warpx4(uint32_t taddr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::%2.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc), "n"(kCtaGroup)
                 : "memory");
  }

  // ---- block-scaled MMA (kind::mxf8f6f4, UE8M0 scales in TMEM) -------------
  // D [+]= A x B; accumulate == 0 overwrites D. `idesc` is an MxInstrDesc.

  /// A from TMEM, B from smem.
  SGL_DEVICE static void mma_mxf8_ts(
      uint32_t taddr_d,
      uint32_t taddr_a,
      uint64_t bdesc,
      uint32_t idesc,
      uint32_t taddr_sfa,
      uint32_t taddr_sfb,
      uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::%7.kind::mxf8f6f4.block_scale [%0], [%1], %2, %3, [%4], [%5], p;\n\t}" ::"r"(taddr_d),
        "r"(taddr_a),
        "l"(bdesc),
        "r"(idesc),
        "r"(taddr_sfa),
        "r"(taddr_sfb),
        "r"(accumulate),
        "n"(kCtaGroup)
        : "memory");
  }
  // ---- unscaled MMA (kind::f8f6f4) ----------------------------------------

  /// D [+]= A x B with A and B from smem; `idesc` is an InstrDesc.
  SGL_DEVICE static void
  mma_f8_ss(uint32_t taddr_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::%5.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(taddr_d),
        "l"(adesc),
        "l"(bdesc),
        "r"(idesc),
        "r"(accumulate),
        "n"(kCtaGroup)
        : "memory");
  }

  /// A and B from smem.
  SGL_DEVICE static void mma_mxf8_ss(
      uint32_t taddr_d,
      uint64_t adesc,
      uint64_t bdesc,
      uint32_t idesc,
      uint32_t taddr_sfa,
      uint32_t taddr_sfb,
      uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::%7.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(taddr_d),
        "l"(adesc),
        "l"(bdesc),
        "r"(idesc),
        "r"(taddr_sfa),
        "r"(taddr_sfb),
        "r"(accumulate),
        "n"(kCtaGroup)
        : "memory");
  }

  // ---- block-scaled MMA (kind::mxf4, UE8M0 scales per 32 elements) --------

  /// e2m1 x e2m1 with A and B from smem (`.block32`); `idesc` from make_instr_desc_mxf4.
  SGL_DEVICE static void mma_mxf4_ss(
      uint32_t taddr_d,
      uint64_t adesc,
      uint64_t bdesc,
      uint32_t idesc,
      uint32_t taddr_sfa,
      uint32_t taddr_sfb,
      uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::%7.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(taddr_d),
        "l"(adesc),
        "l"(bdesc),
        "r"(idesc),
        "r"(taddr_sfa),
        "r"(taddr_sfb),
        "r"(accumulate),
        "n"(kCtaGroup)
        : "memory");
  }
};

}  // namespace device::ptx

}  // namespace sglang
