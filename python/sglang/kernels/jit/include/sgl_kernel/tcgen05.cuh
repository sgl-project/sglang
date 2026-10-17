// tcgen05 (SM100 tensor core) PTX wrappers and the UMMA descriptors they take. A kernel picks its CTA group once,
// `using tcgen05 = device::ptx::TCGen05<1>;` in its namespace, then calls `tcgen05::alloc(...)`:
//   - TCGen05Common: what does not depend on the CTA group (fences, TMEM loads, descriptors);
//   - TCGen05<kCtaGroup>: alloc / dealloc, commit, smem -> TMEM copies and the MMA, for cta_group::kCtaGroup.
// The issue instructions (commit, cp, mma) are single-thread: the caller elects the lane. alloc / dealloc /
// relinquish / ld / wait are warp-collective (.sync.aligned).

#pragma once

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace device::ptx {

struct TCGen05Common {
  // ---- ordering ------------------------------------------------------------

  SGL_DEVICE static void fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  }
  SGL_DEVICE static void fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }

  // ---- TMEM -> registers ---------------------------------------------------

  SGL_DEVICE static void wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  }
  /// N (1 / 2 / ... / 128) consecutive 32-bit columns of this warp's 32 TMEM lanes (lane quadrant warp_id % 4, in
  /// `taddr` bits [16, 32)) into the register block `dst`; complete after wait_ld().
  template <int N>
  SGL_DEVICE static void ld_32x32b(uint32_t taddr, void* dst) {
    static_assert(N >= 1 && N <= 128 && (N & (N - 1)) == 0);
    const auto r = static_cast<uint32_t*>(dst);
    if constexpr (N == 1) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r[0]) : "r"(taddr));
    } else if constexpr (N == 2) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(taddr));
    } else if constexpr (N == 4) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                   : "r"(taddr));
    } else if constexpr (N == 8) {
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                   : "r"(taddr));
    } else if constexpr (N == 16) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15}, [%16];"
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
            "=r"(r[15])
          : "r"(taddr));
    } else if constexpr (N == 32) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
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
    } else if constexpr (N == 64) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
          "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
          "%57, %58, %59, %60, %61, %62, %63}, [%64];"
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
            "=r"(r[31]),
            "=r"(r[32]),
            "=r"(r[33]),
            "=r"(r[34]),
            "=r"(r[35]),
            "=r"(r[36]),
            "=r"(r[37]),
            "=r"(r[38]),
            "=r"(r[39]),
            "=r"(r[40]),
            "=r"(r[41]),
            "=r"(r[42]),
            "=r"(r[43]),
            "=r"(r[44]),
            "=r"(r[45]),
            "=r"(r[46]),
            "=r"(r[47]),
            "=r"(r[48]),
            "=r"(r[49]),
            "=r"(r[50]),
            "=r"(r[51]),
            "=r"(r[52]),
            "=r"(r[53]),
            "=r"(r[54]),
            "=r"(r[55]),
            "=r"(r[56]),
            "=r"(r[57]),
            "=r"(r[58]),
            "=r"(r[59]),
            "=r"(r[60]),
            "=r"(r[61]),
            "=r"(r[62]),
            "=r"(r[63])
          : "r"(taddr));
    } else if constexpr (N == 128) {
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
          "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
          "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
          "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, "
          "%78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, "
          "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, "
          "%117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
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
            "=r"(r[31]),
            "=r"(r[32]),
            "=r"(r[33]),
            "=r"(r[34]),
            "=r"(r[35]),
            "=r"(r[36]),
            "=r"(r[37]),
            "=r"(r[38]),
            "=r"(r[39]),
            "=r"(r[40]),
            "=r"(r[41]),
            "=r"(r[42]),
            "=r"(r[43]),
            "=r"(r[44]),
            "=r"(r[45]),
            "=r"(r[46]),
            "=r"(r[47]),
            "=r"(r[48]),
            "=r"(r[49]),
            "=r"(r[50]),
            "=r"(r[51]),
            "=r"(r[52]),
            "=r"(r[53]),
            "=r"(r[54]),
            "=r"(r[55]),
            "=r"(r[56]),
            "=r"(r[57]),
            "=r"(r[58]),
            "=r"(r[59]),
            "=r"(r[60]),
            "=r"(r[61]),
            "=r"(r[62]),
            "=r"(r[63]),
            "=r"(r[64]),
            "=r"(r[65]),
            "=r"(r[66]),
            "=r"(r[67]),
            "=r"(r[68]),
            "=r"(r[69]),
            "=r"(r[70]),
            "=r"(r[71]),
            "=r"(r[72]),
            "=r"(r[73]),
            "=r"(r[74]),
            "=r"(r[75]),
            "=r"(r[76]),
            "=r"(r[77]),
            "=r"(r[78]),
            "=r"(r[79]),
            "=r"(r[80]),
            "=r"(r[81]),
            "=r"(r[82]),
            "=r"(r[83]),
            "=r"(r[84]),
            "=r"(r[85]),
            "=r"(r[86]),
            "=r"(r[87]),
            "=r"(r[88]),
            "=r"(r[89]),
            "=r"(r[90]),
            "=r"(r[91]),
            "=r"(r[92]),
            "=r"(r[93]),
            "=r"(r[94]),
            "=r"(r[95]),
            "=r"(r[96]),
            "=r"(r[97]),
            "=r"(r[98]),
            "=r"(r[99]),
            "=r"(r[100]),
            "=r"(r[101]),
            "=r"(r[102]),
            "=r"(r[103]),
            "=r"(r[104]),
            "=r"(r[105]),
            "=r"(r[106]),
            "=r"(r[107]),
            "=r"(r[108]),
            "=r"(r[109]),
            "=r"(r[110]),
            "=r"(r[111]),
            "=r"(r[112]),
            "=r"(r[113]),
            "=r"(r[114]),
            "=r"(r[115]),
            "=r"(r[116]),
            "=r"(r[117]),
            "=r"(r[118]),
            "=r"(r[119]),
            "=r"(r[120]),
            "=r"(r[121]),
            "=r"(r[122]),
            "=r"(r[123]),
            "=r"(r[124]),
            "=r"(r[125]),
            "=r"(r[126]),
            "=r"(r[127])
          : "r"(taddr));
    }
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
};

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
};

}  // namespace device::ptx

}  // namespace sglang
