/*
 * MXFP4A8 support for the CUTLASS w4a8 mixed-input grouped GEMM.
 *
 * Provides the missing `NumericArrayConverter<float_e4m3_t, float_e2m1_t, N>`
 * specialization used by the DirectConvert path (LayoutAwareConvert ->
 * universal LayoutAwareConvertImpl -> NumericArrayConverter). CUTLASS commit
 * 57e3cfb ships an int4b_t->e4m3 converter and an e2m1->fp16/fp32 converter,
 * but no e2m1->e4m3 one, which is exactly what the SM90 fp8xfp8 GMMA needs.
 *
 * The E2M1 (weight) magnitudes {0,.5,1,1.5,2,3,4,6} are all exactly
 * representable in E4M3, so this is a lossless 8-value LUT identical in
 * structure to the existing int4->e4m3 prmt LUT; only the candidate bytes
 * differ. The candidate constants below were derived and exhaustively
 * verified against a bit-exact golden model (all 16 nibbles).
 *
 * NOTE: this header only ADDS a new specialization; it does not touch the
 * int4b_t->e4m3 path, so int4a8 stays bit-identical.
 */
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

#if !defined(SGL_KERNEL_USE_CUTLASS_BUILTIN_E2M1_CONVERTER)

/// Partial specialization for Array<cutlass::float_e4m3_t, N> <= Array<cutlass::float_e2m1_t, N>
///
/// Mirrors NumericArrayConverter<float_e4m3_t, int4b_t, N> (numeric_conversion.h),
/// but with:
///   - E2M1 candidate bytes (sign-magnitude, so neg = -pos index-aligned), and
///   - E2M1 sign/index bit extraction (identical bit positions to int4).
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, cutlass::float_e2m1_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<cutlass::float_e2m1_t, N>;

  static FloatRoundStyle const round_style = Round;

 private:
  using result_type_packed_8 = Array<cutlass::float_e4m3_t, 8>;
  using result_type_packed_4 = Array<cutlass::float_e4m3_t, 4>;
  using source_type_packed_8 = Array<cutlass::float_e2m1_t, 8>;
  using source_type_packed_4 = Array<cutlass::float_e2m1_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::float_e4m3_t, cutlass::float_e2m1_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses a lookup table to convert e2m1 -> e4m3.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE static PackedResultType packed_convert(PackedSrcType const& source) {
    static_assert(
        (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
         platform::is_same<PackedResultType, result_type_packed_4>::value) ||
            (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
             platform::is_same<PackedResultType, result_type_packed_8>::value),
        "Invalid PackedSrcType/PackedResultType must be 4 or 8 to use private convert dispatch.");

    // Hold FP8 outputs in reg. We need 1 reg for every 4 outputs.
    cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)> r;

    // View the input as reg
    uint32_t reg = to_reg(source);

    // Determines if to get from the positive or negative candidates.
    // E2M1 sign is the MSB of each 4-bit nibble, same position as int4.
    uint32_t sign = (reg & 0x88888888) >> 1;

    // Ignore sign bit when indexing into LUT. The low 3 bits (exp<<1 | mant)
    // form the magnitude index 0..7 into {0,.5,1,1.5,2,3,4,6}.
    uint32_t lut_idx = (reg & 0x77777777);

    // Signed is OR'd with 0x32103210 to find the correct value in the LUT.
    const uint32_t final_prmt_base = 0x32103210;

    // E2M1 -> E4M3 candidate bytes (verified by golden model, all 16 nibbles):
    //   +mag idx 0..3 = { 0, .5, 1, 1.5 }  -> {0x00,0x30,0x38,0x3C}
    static constexpr uint32_t POS_E4M3s_REG1 = 0x3C383000;
    //   +mag idx 4..7 = { 2, 3, 4, 6 }      -> {0x40,0x44,0x48,0x4C}
    static constexpr uint32_t POS_E4M3s_REG2 = 0x4C484440;
    //   -mag idx 0..3 = { -0, -.5, -1, -1.5}-> {0x80,0xB0,0xB8,0xBC}
    static constexpr uint32_t NEG_E4M3s_REG1 = 0xBCB8B080;
    //   -mag idx 4..7 = { -2, -3, -4, -6 }  -> {0xC0,0xC4,0xC8,0xCC}
    static constexpr uint32_t NEG_E4M3s_REG2 = 0xCCC8C4C0;

    const int iters = PackedSrcType::kElements / 4;
#pragma unroll
    for (int ii = 0; ii < iters; ++ii, lut_idx >>= 16, sign >>= 16) {
      uint32_t final_prmt_idx = final_prmt_base | sign;

      // Select both positive and negative candidates via prmt using the
      // magnitude index, then use the sign bit to pick the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 pos_f8s, neg_f8s;\n"
          "  prmt.b32 pos_f8s, %1, %2, %5;\n"
          "  prmt.b32 neg_f8s, %3, %4, %5;\n"
          "  prmt.b32 %0, pos_f8s, neg_f8s, %6;\n"
          "}\n"
          : "=r"(r[ii])
          : "n"(POS_E4M3s_REG1),
            "n"(POS_E4M3s_REG2),
            "n"(NEG_E4M3s_REG1),
            "n"(NEG_E4M3s_REG2),
            "r"(lut_idx),
            "r"(final_prmt_idx));
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

 public:
  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::
        convert<ConverterType, result_type_packed_8, source_type_packed_8, result_type_packed_4, source_type_packed_4>(
            result, source);
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const {
    return convert(s);
  }
};

#endif  // !defined(SGL_KERNEL_USE_CUTLASS_BUILTIN_E2M1_CONVERTER)

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
