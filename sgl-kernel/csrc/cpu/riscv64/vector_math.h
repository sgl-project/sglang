#pragma once

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

#include <limits>

// Polynomial approximation: exp(x) = 2^(x*log2(e)
constexpr float RVV_EXP_C0 = 1.0f;
constexpr float RVV_EXP_C1 = 0.69314718056f;
constexpr float RVV_EXP_C2 = 0.24022650695f;
constexpr float RVV_EXP_C3 = 0.05550410866f;
constexpr float RVV_EXP_C4 = 0.00961812910f;
constexpr float RVV_EXP_C5 = 0.00133335581f;
constexpr float RVV_LOG2_E = 1.44269504089f;

// Generic Exp template for LMUL={1,2,4,8} — all four specializations are provided.
template <int LMUL>
struct RVVExpImpl;

// Specialization: LMUL=1
template <>
struct RVVExpImpl<1> {
  using VFloat = vfloat32m1_t;
  using VInt = vint32m1_t;

  static inline VFloat compute(VFloat vx, size_t vl) {
    vx = __riscv_vfmax_vf_f32m1(vx, -87.0f, vl);  // Clamp to avoid denormals
    vx = __riscv_vfmin_vf_f32m1(vx, 88.0f, vl);

    VFloat vz = __riscv_vfmul_vf_f32m1(vx, RVV_LOG2_E, vl);
    // vfcvt_x_f uses fcsr.frm (assumes RNE = default rounding mode)
    VInt vn_int = __riscv_vfcvt_x_f_v_i32m1(vz, vl);
    VFloat vn = __riscv_vfcvt_f_x_v_f32m1(vn_int, vl);
    VFloat vf = __riscv_vfsub_vv_f32m1(vz, vn, vl);

    // Horner's method: ((((C5*f + C4)*f + C3)*f + C2)*f + C1)*f + C0
    // vfmadd(vd, vs1, vs2) = vd * vs1 + vs2 — vd is the multiplicand
    VFloat vC0 = __riscv_vfmv_v_f_f32m1(RVV_EXP_C0, vl);
    VFloat vC1 = __riscv_vfmv_v_f_f32m1(RVV_EXP_C1, vl);
    VFloat vC2 = __riscv_vfmv_v_f_f32m1(RVV_EXP_C2, vl);
    VFloat vC3 = __riscv_vfmv_v_f_f32m1(RVV_EXP_C3, vl);
    VFloat vC4 = __riscv_vfmv_v_f_f32m1(RVV_EXP_C4, vl);

    VFloat poly = __riscv_vfmv_v_f_f32m1(RVV_EXP_C5, vl);
    poly = __riscv_vfmadd_vv_f32m1(poly, vf, vC4, vl);  // C5*f + C4
    poly = __riscv_vfmadd_vv_f32m1(poly, vf, vC3, vl);  // (C5*f+C4)*f + C3
    poly = __riscv_vfmadd_vv_f32m1(poly, vf, vC2, vl);
    poly = __riscv_vfmadd_vv_f32m1(poly, vf, vC1, vl);
    poly = __riscv_vfmadd_vv_f32m1(poly, vf, vC0, vl);

    VInt v_exp = __riscv_vadd_vx_i32m1(vn_int, 127, vl);
    v_exp = __riscv_vsll_vx_i32m1(v_exp, 23, vl);
    VFloat v_pow2n = __riscv_vreinterpret_v_i32m1_f32m1(v_exp);

    return __riscv_vfmul_vv_f32m1(poly, v_pow2n, vl);
  }
};

// Specialization: LMUL=2
template <>
struct RVVExpImpl<2> {
  using VFloat = vfloat32m2_t;
  using VInt = vint32m2_t;

  static inline VFloat compute(VFloat vx, size_t vl) {
    vx = __riscv_vfmax_vf_f32m2(vx, -87.0f, vl);  // Clamp to avoid denormals
    vx = __riscv_vfmin_vf_f32m2(vx, 88.0f, vl);

    VFloat vz = __riscv_vfmul_vf_f32m2(vx, RVV_LOG2_E, vl);
    // vfcvt_x_f uses fcsr.frm (assumes RNE = default rounding mode)
    VInt vn_int = __riscv_vfcvt_x_f_v_i32m2(vz, vl);
    VFloat vn = __riscv_vfcvt_f_x_v_f32m2(vn_int, vl);
    VFloat vf = __riscv_vfsub_vv_f32m2(vz, vn, vl);

    // Horner's method: ((((C5*f + C4)*f + C3)*f + C2)*f + C1)*f + C0
    VFloat vC0 = __riscv_vfmv_v_f_f32m2(RVV_EXP_C0, vl);
    VFloat vC1 = __riscv_vfmv_v_f_f32m2(RVV_EXP_C1, vl);
    VFloat vC2 = __riscv_vfmv_v_f_f32m2(RVV_EXP_C2, vl);
    VFloat vC3 = __riscv_vfmv_v_f_f32m2(RVV_EXP_C3, vl);
    VFloat vC4 = __riscv_vfmv_v_f_f32m2(RVV_EXP_C4, vl);

    VFloat poly = __riscv_vfmv_v_f_f32m2(RVV_EXP_C5, vl);
    poly = __riscv_vfmadd_vv_f32m2(poly, vf, vC4, vl);  // C5*f + C4
    poly = __riscv_vfmadd_vv_f32m2(poly, vf, vC3, vl);  // (C5*f+C4)*f + C3
    poly = __riscv_vfmadd_vv_f32m2(poly, vf, vC2, vl);
    poly = __riscv_vfmadd_vv_f32m2(poly, vf, vC1, vl);
    poly = __riscv_vfmadd_vv_f32m2(poly, vf, vC0, vl);

    VInt v_exp = __riscv_vadd_vx_i32m2(vn_int, 127, vl);
    v_exp = __riscv_vsll_vx_i32m2(v_exp, 23, vl);
    VFloat v_pow2n = __riscv_vreinterpret_v_i32m2_f32m2(v_exp);

    return __riscv_vfmul_vv_f32m2(poly, v_pow2n, vl);
  }
};

// Specialization: LMUL=4
template <>
struct RVVExpImpl<4> {
  using VFloat = vfloat32m4_t;
  using VInt = vint32m4_t;

  static inline VFloat compute(VFloat vx, size_t vl) {
    vx = __riscv_vfmax_vf_f32m4(vx, -87.0f, vl);  // Clamp to avoid denormals
    vx = __riscv_vfmin_vf_f32m4(vx, 88.0f, vl);

    VFloat vz = __riscv_vfmul_vf_f32m4(vx, RVV_LOG2_E, vl);
    // vfcvt_x_f uses fcsr.frm (assumes RNE = default rounding mode)
    VInt vn_int = __riscv_vfcvt_x_f_v_i32m4(vz, vl);
    VFloat vn = __riscv_vfcvt_f_x_v_f32m4(vn_int, vl);
    VFloat vf = __riscv_vfsub_vv_f32m4(vz, vn, vl);

    // Horner's method: ((((C5*f + C4)*f + C3)*f + C2)*f + C1)*f + C0
    // vfmadd(vd, vs1, vs2) = vd * vs1 + vs2 — vd is the multiplicand
    VFloat vC0 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C0, vl);
    VFloat vC1 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C1, vl);
    VFloat vC2 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C2, vl);
    VFloat vC3 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C3, vl);
    VFloat vC4 = __riscv_vfmv_v_f_f32m4(RVV_EXP_C4, vl);

    VFloat poly = __riscv_vfmv_v_f_f32m4(RVV_EXP_C5, vl);
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC4, vl);  // C5*f + C4
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC3, vl);  // (C5*f+C4)*f + C3
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC2, vl);
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC1, vl);
    poly = __riscv_vfmadd_vv_f32m4(poly, vf, vC0, vl);

    VInt v_exp = __riscv_vadd_vx_i32m4(vn_int, 127, vl);
    v_exp = __riscv_vsll_vx_i32m4(v_exp, 23, vl);
    VFloat v_pow2n = __riscv_vreinterpret_v_i32m4_f32m4(v_exp);

    return __riscv_vfmul_vv_f32m4(poly, v_pow2n, vl);
  }
};

// Specialization: LMUL=8
template <>
struct RVVExpImpl<8> {
  using VFloat = vfloat32m8_t;
  using VInt = vint32m8_t;

  static inline VFloat compute(VFloat vx, size_t vl) {
    vx = __riscv_vfmax_vf_f32m8(vx, -87.0f, vl);
    vx = __riscv_vfmin_vf_f32m8(vx, 88.0f, vl);

    VFloat vz = __riscv_vfmul_vf_f32m8(vx, RVV_LOG2_E, vl);
    // vfcvt_x_f uses fcsr.frm (assumes RNE = default rounding mode)
    VInt vn_int = __riscv_vfcvt_x_f_v_i32m8(vz, vl);
    VFloat vn = __riscv_vfcvt_f_x_v_f32m8(vn_int, vl);
    VFloat vf = __riscv_vfsub_vv_f32m8(vz, vn, vl);

    // Horner's method: ((((C5*f + C4)*f + C3)*f + C2)*f + C1)*f + C0
    // vfmadd(vd, vs1, vs2) = vd * vs1 + vs2 — vd is the multiplicand
    VFloat vC0 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C0, vl);
    VFloat vC1 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C1, vl);
    VFloat vC2 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C2, vl);
    VFloat vC3 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C3, vl);
    VFloat vC4 = __riscv_vfmv_v_f_f32m8(RVV_EXP_C4, vl);

    VFloat poly = __riscv_vfmv_v_f_f32m8(RVV_EXP_C5, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC4, vl);  // C5*f + C4
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC3, vl);  // (C5*f+C4)*f + C3
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC2, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC1, vl);
    poly = __riscv_vfmadd_vv_f32m8(poly, vf, vC0, vl);

    VInt v_exp = __riscv_vadd_vx_i32m8(vn_int, 127, vl);
    v_exp = __riscv_vsll_vx_i32m8(v_exp, 23, vl);
    VFloat v_pow2n = __riscv_vreinterpret_v_i32m8_f32m8(v_exp);

    return __riscv_vfmul_vv_f32m8(poly, v_pow2n, vl);
  }
};

// Wrapper functions (delegate to template implementations)
inline vfloat32m1_t vfexp_f32m1(vfloat32m1_t vx, size_t vl) {
  return RVVExpImpl<1>::compute(vx, vl);
}

inline vfloat32m2_t vfexp_f32m2(vfloat32m2_t vx, size_t vl) {
  return RVVExpImpl<2>::compute(vx, vl);
}

inline vfloat32m4_t vfexp_f32m4(vfloat32m4_t vx, size_t vl) {
  return RVVExpImpl<4>::compute(vx, vl);
}

inline vfloat32m8_t vfexp_f32m8(vfloat32m8_t vx, size_t vl) {
  return RVVExpImpl<8>::compute(vx, vl);
}

// Fast reciprocal: ~1/vd via vfrec7 + one Newton-Raphson step (~14-bit accuracy).
// NR: r <- r * (2 - d * r).  Safe for any d > 0 (which holds for all our use sites).
// NOT suitable for full FP32 precision (24-bit); sufficient for fp16/bf16 outputs only.
inline vfloat32m1_t vrec_f32m1(vfloat32m1_t vd, size_t vl) {
  vfloat32m1_t vr = __riscv_vfrec7_v_f32m1(vd, vl);
  vfloat32m1_t vdr = __riscv_vfmul_vv_f32m1(vd, vr, vl);
  vfloat32m1_t vcorr = __riscv_vfrsub_vf_f32m1(vdr, 2.0f, vl);  // 2 - d*r
  return __riscv_vfmul_vv_f32m1(vr, vcorr, vl);
}

inline vfloat32m2_t vrec_f32m2(vfloat32m2_t vd, size_t vl) {
  vfloat32m2_t vr = __riscv_vfrec7_v_f32m2(vd, vl);
  vfloat32m2_t vdr = __riscv_vfmul_vv_f32m2(vd, vr, vl);
  vfloat32m2_t vcorr = __riscv_vfrsub_vf_f32m2(vdr, 2.0f, vl);  // 2 - d*r
  return __riscv_vfmul_vv_f32m2(vr, vcorr, vl);
}

inline vfloat32m4_t vrec_f32m4(vfloat32m4_t vd, size_t vl) {
  vfloat32m4_t vr = __riscv_vfrec7_v_f32m4(vd, vl);
  vfloat32m4_t vdr = __riscv_vfmul_vv_f32m4(vd, vr, vl);
  vfloat32m4_t vcorr = __riscv_vfrsub_vf_f32m4(vdr, 2.0f, vl);  // 2 - d*r
  return __riscv_vfmul_vv_f32m4(vr, vcorr, vl);
}

inline vfloat32m8_t vrec_f32m8(vfloat32m8_t vd, size_t vl) {
  vfloat32m8_t vr = __riscv_vfrec7_v_f32m8(vd, vl);
  vfloat32m8_t vdr = __riscv_vfmul_vv_f32m8(vd, vr, vl);
  vfloat32m8_t vcorr = __riscv_vfrsub_vf_f32m8(vdr, 2.0f, vl);
  return __riscv_vfmul_vv_f32m8(vr, vcorr, vl);
}

// tanh(x) = (e^2x - 1) / (e^2x + 1).  Clamped to ±9: at |x|=9, FP32 tanh ≈ 1-6e-8
// (within 1 ULP of 1.0). FP32 saturates to exactly ±1.0 at |x|>=10; ±9 is a
// conservative bound that also keeps exp(2x) < 65M, avoiding approximation breakdown.
inline vfloat32m1_t vftanh_f32m1(vfloat32m1_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m1(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m1(vx, 9.0f, vl);
  vfloat32m1_t v2x = __riscv_vfmul_vf_f32m1(vx, 2.0f, vl);
  vfloat32m1_t vex = vfexp_f32m1(v2x, vl);
  vfloat32m1_t v_num = __riscv_vfsub_vf_f32m1(vex, 1.0f, vl);
  vfloat32m1_t v_denom = __riscv_vfadd_vf_f32m1(vex, 1.0f, vl);
  return __riscv_vfmul_vv_f32m1(v_num, vrec_f32m1(v_denom, vl), vl);
}

inline vfloat32m2_t vftanh_f32m2(vfloat32m2_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m2(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m2(vx, 9.0f, vl);
  vfloat32m2_t v2x = __riscv_vfmul_vf_f32m2(vx, 2.0f, vl);
  vfloat32m2_t vex = vfexp_f32m2(v2x, vl);
  vfloat32m2_t v_num = __riscv_vfsub_vf_f32m2(vex, 1.0f, vl);
  vfloat32m2_t v_denom = __riscv_vfadd_vf_f32m2(vex, 1.0f, vl);
  return __riscv_vfmul_vv_f32m2(v_num, vrec_f32m2(v_denom, vl), vl);
}

inline vfloat32m4_t vftanh_f32m4(vfloat32m4_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m4(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m4(vx, 9.0f, vl);
  vfloat32m4_t v2x = __riscv_vfmul_vf_f32m4(vx, 2.0f, vl);
  vfloat32m4_t vex = vfexp_f32m4(v2x, vl);
  vfloat32m4_t v_num = __riscv_vfsub_vf_f32m4(vex, 1.0f, vl);
  vfloat32m4_t v_denom = __riscv_vfadd_vf_f32m4(vex, 1.0f, vl);
  return __riscv_vfmul_vv_f32m4(v_num, vrec_f32m4(v_denom, vl), vl);
}

inline vfloat32m8_t vftanh_f32m8(vfloat32m8_t vx, size_t vl) {
  vx = __riscv_vfmax_vf_f32m8(vx, -9.0f, vl);
  vx = __riscv_vfmin_vf_f32m8(vx, 9.0f, vl);
  vfloat32m8_t v2x = __riscv_vfmul_vf_f32m8(vx, 2.0f, vl);
  vfloat32m8_t vex = vfexp_f32m8(v2x, vl);
  vfloat32m8_t v_num = __riscv_vfsub_vf_f32m8(vex, 1.0f, vl);
  vfloat32m8_t v_denom = __riscv_vfadd_vf_f32m8(vex, 1.0f, vl);
  return __riscv_vfmul_vv_f32m8(v_num, vrec_f32m8(v_denom, vl), vl);
}

// Polynomial approximation: erf(x) max error 1.5e-7
// erf(x) = sign(x) * (1 - poly(t) * exp(-x^2))
// where t = 1 / (1 + 0.3275911 * |x|)
// poly(t) = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t  (Horner form)
// erf saturates to ±1 for |x| >= 4; |x| is clamped to 4 to avoid exp underflow.
constexpr float RVV_ERF_P = 0.3275911f;
constexpr float RVV_ERF_A1 = 0.254829592f;
constexpr float RVV_ERF_A2 = -0.284496736f;
constexpr float RVV_ERF_A3 = 1.421413741f;
constexpr float RVV_ERF_A4 = -1.453152027f;
constexpr float RVV_ERF_A5 = 1.061405429f;

inline vfloat32m4_t vferf_f32m4(vfloat32m4_t vx, size_t vl) {
  // |x|, clamped to 4 (erf saturates to ±1 beyond this)
  vfloat32m4_t vabs = __riscv_vfabs_v_f32m4(vx, vl);
  vabs = __riscv_vfmin_vf_f32m4(vabs, 4.0f, vl);

  // t = 1 / (1 + p * |x|)
  vfloat32m4_t vone = __riscv_vfmv_v_f_f32m4(1.0f, vl);
  vfloat32m4_t vdenom = __riscv_vfmacc_vf_f32m4(vone, RVV_ERF_P, vabs, vl);
  vfloat32m4_t vt = vrec_f32m4(vdenom, vl);  // ~1/(1 + p*|x|), replaces vfdiv

  // Horner's method: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
  vfloat32m4_t vpoly = __riscv_vfmv_v_f_f32m4(RVV_ERF_A5, vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A4, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A3, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A2, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m4(vpoly, vt, __riscv_vfmv_v_f_f32m4(RVV_ERF_A1, vl), vl);
  vpoly = __riscv_vfmul_vv_f32m4(vpoly, vt, vl);  // final *t gives a1*t + ... + a5*t^5

  // exp(-x^2)
  vfloat32m4_t vx2 = __riscv_vfmul_vv_f32m4(vabs, vabs, vl);
  vfloat32m4_t vexp = vfexp_f32m4(__riscv_vfneg_v_f32m4(vx2, vl), vl);

  // result = 1 - poly * exp(-x^2); vfnmsac(vd, vs1, vs2) = vd - vs1 * vs2
  vfloat32m4_t vresult = __riscv_vfnmsac_vv_f32m4(vone, vpoly, vexp, vl);

  // Apply sign: erf(x) has the same sign as x.
  // vfsgnjx(vd, vs) = vd with sign = sign(vd) XOR sign(vs).
  // vresult >= 0 (sign bit = 0), so result sign = sign(vx).
  return __riscv_vfsgnjx_vv_f32m4(vresult, vx, vl);
}

inline vfloat32m8_t vferf_f32m8(vfloat32m8_t vx, size_t vl) {
  vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vx, vl);
  vabs = __riscv_vfmin_vf_f32m8(vabs, 4.0f, vl);

  vfloat32m8_t vone = __riscv_vfmv_v_f_f32m8(1.0f, vl);
  vfloat32m8_t vdenom = __riscv_vfmacc_vf_f32m8(vone, RVV_ERF_P, vabs, vl);
  vfloat32m8_t vt = vrec_f32m8(vdenom, vl);  // ~1/(1 + p*|x|), replaces vfdiv

  vfloat32m8_t vpoly = __riscv_vfmv_v_f_f32m8(RVV_ERF_A5, vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A4, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A3, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A2, vl), vl);
  vpoly = __riscv_vfmadd_vv_f32m8(vpoly, vt, __riscv_vfmv_v_f_f32m8(RVV_ERF_A1, vl), vl);
  vpoly = __riscv_vfmul_vv_f32m8(vpoly, vt, vl);

  vfloat32m8_t vx2 = __riscv_vfmul_vv_f32m8(vabs, vabs, vl);
  vfloat32m8_t vexp = vfexp_f32m8(__riscv_vfneg_v_f32m8(vx2, vl), vl);

  vfloat32m8_t vresult = __riscv_vfnmsac_vv_f32m8(vone, vpoly, vexp, vl);
  return __riscv_vfsgnjx_vv_f32m8(vresult, vx, vl);
}

// Reduction Wrappers for Different register configurations
inline float reduce_sum_f32m4(vfloat32m4_t v_acc, size_t vl_max) {
  vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t vred = __riscv_vfredusum_vs_f32m4_f32m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f32m1_f32(vred);
}

inline float reduce_sum_f32m1(vfloat32m1_t v_acc, size_t vl_max) {
  vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(v_acc, vzero, vl_max);
  return __riscv_vfmv_f_s_f32m1_f32(vred);
}

// Widening Multiply-Accumulate Support

// FP16 Vector-Vector -> FP32 Accumulator
inline vfloat32m4_t vfwmacc_f16_to_f32m4(vfloat32m4_t vd, vfloat16m2_t vs1, vfloat16m2_t vs2, size_t vl) {
#if defined(__riscv_zvfh)
  return __riscv_vfwmacc_vv_f32m4_tu(vd, vs1, vs2, vl);
#else
  vfloat32m4_t vs1_f32 = __riscv_vfwcvt_f_f_v_f32m4(vs1, vl);
  vfloat32m4_t vs2_f32 = __riscv_vfwcvt_f_f_v_f32m4(vs2, vl);
  return __riscv_vfmacc_vv_f32m4_tu(vd, vs1_f32, vs2_f32, vl);
#endif
}

// FP16 Scalar-Vector -> FP32 Accumulator
inline vfloat32m4_t vfwmacc_f16_scalar_to_f32m4(vfloat32m4_t vd, _Float16 scalar, vfloat16m2_t vs2, size_t vl) {
#if defined(__riscv_zvfh)
  return __riscv_vfwmacc_vf_f32m4_tu(vd, scalar, vs2, vl);
#else
  vfloat32m4_t vs2_f32 = __riscv_vfwcvt_f_f_v_f32m4(vs2, vl);
  return __riscv_vfmacc_vf_f32m4_tu(vd, (float)scalar, vs2_f32, vl);
#endif
}

#endif  // CPU_CAPABILITY_RVV
