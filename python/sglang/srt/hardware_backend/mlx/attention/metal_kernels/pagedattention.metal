// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

// Portions of this file are adapted from the vLLM project
// (https://github.com/vllm-project/vllm)
// Licensed under the Apache License 2.0
// Copyright contributors to the vLLM project

#include "utils.metal"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// ========================================== Generic vector types

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE> struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T> struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B> inline Acc mul(A a, B b);

template <typename T> inline float sum(T v);

template <typename T> inline float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T> inline float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// FP32 vector data types.
struct Float8_ {
  float4 x;
  float4 y;
};

template <> struct Vec<float, 1> {
  using Type = float;
};
template <> struct Vec<float, 2> {
  using Type = float2;
};
template <> struct Vec<float, 4> {
  using Type = float4;
};
template <> struct Vec<float, 8> {
  using Type = Float8_;
};

template <> struct FloatVec<float> {
  using Type = float;
};
template <> struct FloatVec<float2> {
  using Type = float2;
};
template <> struct FloatVec<float4> {
  using Type = float4;
};
template <> struct FloatVec<Float8_> {
  using Type = Float8_;
};

template <> inline float mul(float a, float b) { return a * b; }

template <> inline float2 mul(float2 a, float2 b) { return a * b; }

template <> inline float4 mul(float4 a, float4 b) { return a * b; }

template <> inline Float8_ mul(Float8_ a, Float8_ b) {
  Float8_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float sum(float a) { return a; }

template <> inline float sum(float2 a) { return a.x + a.y; }

template <> inline float sum(float4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Float8_ a) { return sum(a.x) + sum(a.y); }

inline Float8_ fma(Float8_ a, Float8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread float &dst, float src) { dst = src; }
inline void from_float(thread float2 &dst, float2 src) { dst = src; }
inline void from_float(thread float4 &dst, float4 src) { dst = src; }
inline void from_float(thread Float8_ &dst, Float8_ src) { dst = src; }

// BF16 vector data types.
// #if defined(__HAVE_BFLOAT__)

// struct Bfloat8_ {
//   bfloat4 x;
//   bfloat4 y;
// };

// template<>
// struct Vec<bfloat, 1> {
//   using Type = bfloat;
// };
// template<>
// struct Vec<bfloat, 2> {
//   using Type = bfloat2;
// };
// template<>
// struct Vec<bfloat, 4> {
//   using Type = bfloat4;
// };
// template<>
// struct Vec<bfloat, 8> {
//   using Type = Bfloat8_;
// };

// template<>
// struct FloatVec<bfloat> {
//   using Type = float;
// };
// template<>
// struct FloatVec<bfloat2> {
//   using Type = float2;
// };
// template<>
// struct FloatVec<bfloat4> {
//   using Type = float4;
// };
// template<>
// struct FloatVec<Bfloat8_> {
//   using Type = Float8_;
// };

// template<>
// inline float mul(bfloat a, bfloat b) {
//   return (float)a * (float)b;
// }
// template<>
// inline bfloat mul(bfloat a, bfloat b) {
//   return a*b;
// }

// template<>
// inline float2 mul(bfloat2 a, bfloat2 b) {
//   return (float2)a * (float2)b;
// }
// template<>
// inline bfloat2 mul(bfloat2 a, bfloat2 b) {
//   return a * b;
// }

// template<>
// inline float4 mul(bfloat4 a, bfloat4 b) {
//   return (float4)a * (float4)b;
// }
// template<>
// inline bfloat4 mul(bfloat4 a, bfloat4 b) {
//   return a * b;
// }

// template<>
// inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Float8_ c;
//   c.x = mul<float4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<float4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }
// template<>
// inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Bfloat8_ c;
//   c.x = mul<bfloat4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<bfloat4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }

// template<>
// inline float sum(bfloat a) {
//   return (float)a;
// }

// template<>
// inline float sum(bfloat2 a) {
//   return (float)a.x + (float)a.y;
// }

// template<>
// inline float sum(bfloat4 a) {
//   return sum(a.x) + sum(a.y);
// }

// template<>
// inline float sum(Bfloat8_ a) {
//   return sum(a.x) + sum(a.y);
// }

// inline float fma(bfloat a, bfloat b, float c) {
//   return (float)a * (float)b + c;
// }

// inline float2 fma(bfloat2 a, bfloat2 b, float2 c) {
//   return (float2)a * (float2)b + c;
// }

// inline float4 fma(bfloat4 a, bfloat4 b, float4 c) {
//   return (float4)a * (float4)b + c;
// }

// inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
//   Float8_ res;
//   res.x = fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = fma((float4)a.y, (float4)b.y, (float4)c.y);
//   return res;
// }
// inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
//   Bfloat8_ res;
//   res.x = (bfloat4)fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = (bfloat4)fma((float4)a.y, (float4)b.x, (float4)c.y);
//   return c;
// }

// inline void from_float(thread bfloat& dst, float src) {
//   dst = static_cast<bfloat>(src);
// }
// inline void from_float(thread bfloat2& dst, float2 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
// }
// inline void from_float(thread bfloat4& dst, float4 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
//   dst.z = static_cast<bfloat>(src.z);
//   dst.w = static_cast<bfloat>(src.w);
// }
// inline void from_float(thread Bfloat8_& dst, Float8_ src) {
//   bfloat4 x;
//   bfloat4 y;
//   from_float(x, src.x);
//   from_float(y, src.y);
//   dst.x = x;
//   dst.y = y;
// }

// #else

struct Bfloat2_ {
  bfloat16_t x;
  bfloat16_t y;
};

struct Bfloat4_ {
  Bfloat2_ x;
  Bfloat2_ y;
};

struct Bfloat8_ {
  Bfloat4_ x;
  Bfloat4_ y;
};

template <> struct Vec<bfloat16_t, 1> {
  using Type = bfloat16_t;
};
template <> struct Vec<bfloat16_t, 2> {
  using Type = Bfloat2_;
};
template <> struct Vec<bfloat16_t, 4> {
  using Type = Bfloat4_;
};
template <> struct Vec<bfloat16_t, 8> {
  using Type = Bfloat8_;
};

template <> struct FloatVec<bfloat16_t> {
  using Type = float;
};
template <> struct FloatVec<Bfloat2_> {
  using Type = float2;
};
template <> struct FloatVec<Bfloat4_> {
  using Type = float4;
};
template <> struct FloatVec<Bfloat8_> {
  using Type = Float8_;
};

template <> inline float mul(bfloat16_t a, bfloat16_t b) {
  return (float)a * (float)b;
}
template <> inline bfloat16_t mul(bfloat16_t a, bfloat16_t b) { return a * b; }

template <> inline float2 mul(Bfloat2_ a, Bfloat2_ b) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f;
}
template <> inline Bfloat2_ mul(Bfloat2_ a, Bfloat2_ b) {
  Bfloat2_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float4 mul(Bfloat4_ a, Bfloat4_ b) {
  float2 x = mul<float2, Bfloat2_, Bfloat2_>(a.x, b.x);
  float2 y = mul<float2, Bfloat2_, Bfloat2_>(a.y, b.y);
  float4 c;
  c.x = x.x;
  c.y = x.y;
  c.z = y.x;
  c.w = y.y;
  return c;
}
template <> inline Bfloat4_ mul(Bfloat4_ a, Bfloat4_ b) {
  Bfloat4_ c;
  c.x = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.x, b.x);
  c.y = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.y, b.y);
  return c;
}

template <> inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Float8_ c;
  c.x = mul<float4, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<float4, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}
template <> inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Bfloat8_ c;
  c.x = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}

template <> inline float sum(bfloat16_t a) { return (float)a; }

template <> inline float sum(Bfloat2_ a) { return (float)a.x + (float)a.y; }

template <> inline float sum(Bfloat4_ a) { return sum(a.x) + sum(a.y); }

template <> inline float sum(Bfloat8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(bfloat16_t a, bfloat16_t b, float c) {
  return (float)a * (float)b + c;
}
inline bfloat16_t fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
  return a * b + c;
}

inline float2 fma(Bfloat2_ a, Bfloat2_ b, float2 c) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f + c;
}
inline Bfloat2_ fma(Bfloat2_ a, Bfloat2_ b, Bfloat2_ c) {
  Bfloat2_ res;
  res.x = a.x * b.x + c.x;
  res.y = a.y * b.y + c.y;
  return res;
}

inline float4 fma(Bfloat4_ a, Bfloat4_ b, float4 c) {
  float4 res;
  res.x = fma(a.x.x, b.x.x, c.x);
  res.y = fma(a.x.y, b.x.y, c.y);
  res.z = fma(a.y.x, b.y.x, c.z);
  res.w = fma(a.y.y, b.y.y, c.w);
  return res;
}
inline Bfloat4_ fma(Bfloat4_ a, Bfloat4_ b, Bfloat4_ c) {
  Bfloat4_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
  Bfloat8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread bfloat16_t &dst, float src) {
  dst = static_cast<bfloat16_t>(src);
}
inline void from_float(thread Bfloat2_ &dst, float2 src) {
  dst.x = static_cast<bfloat16_t>(src.x);
  dst.y = static_cast<bfloat16_t>(src.y);
}
inline void from_float(thread Bfloat4_ &dst, float4 src) {
  dst.x.x = static_cast<bfloat16_t>(src.x);
  dst.x.y = static_cast<bfloat16_t>(src.y);
  dst.y.x = static_cast<bfloat16_t>(src.z);
  dst.y.y = static_cast<bfloat16_t>(src.w);
}
inline void from_float(thread Bfloat8_ &dst, Float8_ src) {
  Bfloat4_ x;
  Bfloat4_ y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// #endif

// ========================================== FP8 (uchar) vector data types.

// 8‑lane uchar vector – Metal only provides up to uchar4, so build our own.
struct Uchar8_ {
  uchar4 x;
  uchar4 y;
};

// Vec specialisations so Vec<uchar, N>::Type resolves correctly.
template <> struct Vec<uchar, 1> {
  using Type = uchar;
};
template <> struct Vec<uchar, 2> {
  using Type = uchar2;
};
template <> struct Vec<uchar, 4> {
  using Type = uchar4;
};
template <> struct Vec<uchar, 8> {
  using Type = Uchar8_;
};

// FP16 vector data types.
struct Half8_ {
  half4 x;
  half4 y;
};

template <> struct Vec<half, 1> {
  using Type = half;
};
template <> struct Vec<half, 2> {
  using Type = half2;
};
template <> struct Vec<half, 4> {
  using Type = half4;
};
template <> struct Vec<half, 8> {
  using Type = Half8_;
};

template <> struct FloatVec<half> {
  using Type = float;
};
template <> struct FloatVec<half2> {
  using Type = float2;
};
template <> struct FloatVec<half4> {
  using Type = float4;
};
template <> struct FloatVec<Half8_> {
  using Type = Float8_;
};

template <> inline float mul(half a, half b) { return (float)a * (float)b; }
template <> inline half mul(half a, half b) { return a * b; }

template <> inline float2 mul(half2 a, half2 b) {
  return (float2)a * (float2)b;
}
template <> inline half2 mul(half2 a, half2 b) { return a * b; }

template <> inline float4 mul(half4 a, half4 b) {
  return (float4)a * (float4)b;
}
template <> inline half4 mul(half4 a, half4 b) { return a * b; }

template <> inline Float8_ mul(Half8_ a, Half8_ b) {
  float4 x = mul<float4, half4, half4>(a.x, b.x);
  float4 y = mul<float4, half4, half4>(a.y, b.y);
  Float8_ c;
  c.x = x;
  c.y = y;
  return c;
}
template <> inline Half8_ mul(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = mul<half4, half4, half4>(a.x, b.x);
  c.y = mul<half4, half4, half4>(a.y, b.y);
  return c;
}

template <> inline float sum(half a) { return (float)a; }

template <> inline float sum(half2 a) { return (float)a.x + (float)a.y; }

template <> inline float sum(half4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Half8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(half a, half b, float c) { return (float)a * (float)b + c; }

inline float2 fma(half2 a, half2 b, float2 c) {
  return (float2)a * (float2)b + c;
}

inline float4 fma(half4 a, half4 b, float4 c) {
  return (float4)a * (float4)b + c;
}

inline Float8_ fma(Half8_ a, Half8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Half8_ fma(Half8_ a, Half8_ b, Half8_ c) {
  Half8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread half &dst, float src) {
  dst = static_cast<half>(src);
}
inline void from_float(thread half2 &dst, float2 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
}
inline void from_float(thread half4 &dst, float4 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
  dst.z = static_cast<half>(src.z);
  dst.w = static_cast<half>(src.w);
}
inline void from_float(thread Half8_ &dst, Float8_ src) {
  half4 x;
  half4 y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// General case: not uchar
template <typename T> inline constexpr bool is_uchar() { return false; }

// Specialization: T is uchar
template <> inline constexpr bool is_uchar<uchar>() { return true; }

// Generic fallback – will fail to compile if a required specialisation is
// missing.
template <typename Vec, typename Quant_vec>
inline Vec fp8_convert(const thread Quant_vec &, float scale) {
  static_assert(sizeof(Vec) == 0, "Missing fp8_convert specialisation");
}

// ========================================== FP8 -> float/half/bfloat
inline float __dequant_single(uchar v, float scale) {
  // Dummy implementation since we don't have float8.metal
  return 0.0f; // TODO
}

// ---- 1‑lane ----
template <>
inline float fp8_convert<float, uchar>(const thread uchar &in, float scale) {
  return __dequant_single(in, scale);
}
template <>
inline half fp8_convert<half, uchar>(const thread uchar &in, float scale) {
  return half(__dequant_single(in, scale));
}
template <>
inline bfloat16_t fp8_convert<bfloat16_t, uchar>(const thread uchar &in,
                                                 float scale) {
  return bfloat16_t(__dequant_single(in, scale));
}

// ---- 2‑lane ----
template <>
inline float2 fp8_convert<float2, uchar2>(const thread uchar2 &in,
                                          float scale) {
  return float2(__dequant_single(in.x, scale), __dequant_single(in.y, scale));
}
template <>
inline half2 fp8_convert<half2, uchar2>(const thread uchar2 &in, float scale) {
  half2 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  return out;
}
template <>
inline Bfloat2_ fp8_convert<Bfloat2_, uchar2>(const thread uchar2 &in,
                                              float scale) {
  Bfloat2_ out;
  out.x = bfloat16_t(__dequant_single(in.x, scale));
  out.y = bfloat16_t(__dequant_single(in.y, scale));
  return out;
}

// ---- 4‑lane ----
template <>
inline float4 fp8_convert<float4, uchar4>(const thread uchar4 &in,
                                          float scale) {
  return float4(__dequant_single(in.x, scale), __dequant_single(in.y, scale),
                __dequant_single(in.z, scale), __dequant_single(in.w, scale));
}
template <>
inline half4 fp8_convert<half4, uchar4>(const thread uchar4 &in, float scale) {
  half4 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  out.z = half(__dequant_single(in.z, scale));
  out.w = half(__dequant_single(in.w, scale));
  return out;
}
template <>
inline Bfloat4_ fp8_convert<Bfloat4_, uchar4>(const thread uchar4 &in,
                                              float scale) {
  Bfloat4_ out;
  out.x.x = bfloat16_t(__dequant_single(in.x, scale));
  out.x.y = bfloat16_t(__dequant_single(in.y, scale));
  out.y.x = bfloat16_t(__dequant_single(in.z, scale));
  out.y.y = bfloat16_t(__dequant_single(in.w, scale));
  return out;
}

// ---- 8‑lane ----
template <>
inline Float8_ fp8_convert<Float8_, Uchar8_>(const thread Uchar8_ &in,
                                             float scale) {
  Float8_ out;
  out.x =
      float4(__dequant_single(in.x.x, scale), __dequant_single(in.x.y, scale),
             __dequant_single(in.x.z, scale), __dequant_single(in.x.w, scale));
  out.y =
      float4(__dequant_single(in.y.x, scale), __dequant_single(in.y.y, scale),
             __dequant_single(in.y.z, scale), __dequant_single(in.y.w, scale));
  return out;
}
template <>
inline Half8_ fp8_convert<Half8_, Uchar8_>(const thread Uchar8_ &in,
                                           float scale) {
  Half8_ out;
  out.x = half4(half(__dequant_single(in.x.x, scale)),
                half(__dequant_single(in.x.y, scale)),
                half(__dequant_single(in.x.z, scale)),
                half(__dequant_single(in.x.w, scale)));
  out.y = half4(half(__dequant_single(in.y.x, scale)),
                half(__dequant_single(in.y.y, scale)),
                half(__dequant_single(in.y.z, scale)),
                half(__dequant_single(in.y.w, scale)));
  return out;
}
template <>
inline Bfloat8_ fp8_convert<Bfloat8_, Uchar8_>(const thread Uchar8_ &in,
                                               float scale) {
  Bfloat8_ out;
  // first 4
  out.x.x.x = bfloat16_t(__dequant_single(in.x.x, scale));
  out.x.x.y = bfloat16_t(__dequant_single(in.x.y, scale));
  out.x.y.x = bfloat16_t(__dequant_single(in.x.z, scale));
  out.x.y.y = bfloat16_t(__dequant_single(in.x.w, scale));
  // second 4
  out.y.x.x = bfloat16_t(__dequant_single(in.y.x, scale));
  out.y.x.y = bfloat16_t(__dequant_single(in.y.y, scale));
  out.y.y.x = bfloat16_t(__dequant_single(in.y.z, scale));
  out.y.y.y = bfloat16_t(__dequant_single(in.y.w, scale));
  return out;
}

// ========================================== Dot product utilities

// TODO(EricLBuehler): optimize with vectorization
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  using A_vec = typename FloatVec<Vec>::Type;
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += simd_shuffle_xor(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE> struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(const threadgroup Vec (&q)[N],
                          const thread Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// ========================================== Block sum utility

// Utility function for attention softmax.
template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(threadgroup float *red_smem, float sum, uint simd_tid,
                       uint simd_lid) {
  // Compute the sum per simdgroup.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Simd leaders store the data to shared memory.
  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }

  // Make sure the data is in shared memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The warps compute the final sums.
  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  }

  // Parallel reduction inside the simd group.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Broadcast to other threads.
  return simd_shuffle(sum, 0);
}

// ========================================== Paged Attention kernel

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// Binary search to find which sequence a global query token belongs to.
//
// In varlen (ragged-batch) attention, queries from multiple sequences are
// packed contiguously into a flat array:
//   q[0..q_len_0-1]  → seq 0,  q[q_len_0..q_len_0+q_len_1-1]  → seq 1, ...
// The kernel launches one threadgroup per (head, query_token) in a flat grid.
// Each threadgroup needs to discover which sequence it belongs to so it can
// look up the correct block_table row, kv_len, and causal mask boundary.
//
// This is the same approach used by the upstream vLLM unified Triton kernel
// (triton_unified_attention.py:find_seq_idx) and FlashAttention's varlen API.
//
// cu_seqlens_q is sorted ascending: [0, q_len_0, q_len_0+q_len_1, ...].
// Returns seq_idx such that cu_seqlens_q[seq_idx] <= q_token_idx < cu_seqlens_q[seq_idx+1].
inline int find_seq_idx(const device int32_t *cu_seqlens_q,
                        int q_token_idx, int num_seqs) {
  int lo = 0, hi = num_seqs;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    if (cu_seqlens_q[mid] <= q_token_idx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

constant bool use_partitioning [[function_constant(10)]];
constant bool use_alibi [[function_constant(20)]];
constant bool use_fp8_scales [[function_constant(30)]];
constant bool use_sinks [[function_constant(40)]];

template <typename T, typename CACHE_T, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int NUM_SIMD_LANES, int PARTITION_SIZE = 0>
[[kernel]] void paged_attention(
    device float *exp_sums
    [[buffer(0), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device float *max_logits
    [[buffer(1), function_constant(use_partitioning)]], // [num_seqs, num_heads,
                                                        // max_num_partitions]
    device T *out
    [[buffer(2)]], // [num_seqs, num_heads, max_num_partitions, head_size]
    device const T *q [[buffer(3)]], // [num_seqs, num_heads, head_size]
    device const CACHE_T *k_cache
    [[buffer(4)]], // [num_blocks, block_size, num_kv_heads, head_size]
    device const CACHE_T *v_cache
    [[buffer(5)]], // [num_blocks, block_size, num_kv_heads, head_size]
    const device float *__restrict__ k_scale
    [[buffer(6), function_constant(use_fp8_scales)]], // [1]
    const device float *__restrict__ v_scale
    [[buffer(7), function_constant(use_fp8_scales)]], // [1]
    const constant int &num_kv_heads [[buffer(8)]],   // [num_heads]
    const constant float &scale [[buffer(9)]],
    const constant float &softcapping [[buffer(10)]],
    device const uint32_t *block_tables
    [[buffer(11)]], // [num_seqs, max_num_blocks_per_seq]
    device const uint32_t *context_lens [[buffer(12)]], // [num_seqs]
    const constant int &max_num_blocks_per_seq [[buffer(13)]],
    device const float *alibi_slopes
    [[buffer(14), function_constant(use_alibi)]], // [num_heads]
    const constant int &q_stride [[buffer(15)]],
    const constant int &kv_block_stride [[buffer(16)]],
    const constant int &kv_head_stride [[buffer(17)]],
    const device float *sinks
    [[buffer(18), function_constant(use_sinks)]], // [num_heads]
    device const int32_t *cu_seqlens_q [[buffer(19)]],  // [num_seqs + 1]
    const constant int &num_seqs [[buffer(20)]],
    const constant int &sliding_window [[buffer(21)]],  // -1 = disabled
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Varlen: each threadgroup handles one query token.
  // Use binary search on cu_seqlens_q to find which sequence it belongs to.
  const int q_token_idx = threadgroup_position_in_grid.y;
  const int seq_idx = find_seq_idx(cu_seqlens_q, q_token_idx, num_seqs);
  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int q_pos_in_seq = q_token_idx - q_seq_start;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = thread_position_in_threadgroup.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const uint32_t context_len = context_lens[seq_idx];  // total KV length for this seq

  // Causal: this query token can attend to KV positions [0, effective_context_len).
  const int effective_context_len = (int)context_len - q_len + q_pos_in_seq + 1;
  if (effective_context_len <= 0) {
    // No KV tokens to attend to. Caller guarantees out is zero-initialized.
    return;
  }

  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= effective_context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(effective_context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = !use_alibi ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<CACHE_T, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in the
  // group has 0, 4, 8, ... th vectors of the query, and the second thread has
  // 1, 5, 9, ... th vectors of the query, and so on.
  const device T *q_ptr = q + q_token_idx * q_stride + head_idx * HEAD_SIZE;
  threadgroup Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const device Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Workspace for cross-warp reduction of online softmax state.
  // Layout: [NUM_WARPS] for m, [NUM_WARPS] for l, then HEAD_SIZE floats
  // for the tree reduction of O accumulators.
  threadgroup float red_smem[2 * NUM_WARPS];

  // Token stride within a block: num_kv_heads * HEAD_SIZE
  const int kv_token_stride = num_kv_heads * HEAD_SIZE;

  // ========== Online softmax: per-warp state ==========
  // Each warp maintains its own running (m, l, O) across its KV blocks.
  // m = running max of QK scores (for numerical stability)
  // l = running sum of exp(score - m) (softmax denominator)
  // O = running unnormalized output accumulator
  float warp_m = -FLT_MAX;  // running max
  float warp_l = 0.f;       // running exp sum

  constexpr int V_ELEMS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_SIMD_LANES);
  float v_accs[V_ELEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
    v_accs[i] = 0.f;
  }

  // ========== Fused QK + online softmax + V accumulation ==========
  // Each warp processes a subset of KV blocks. Within each block, we:
  //   1. Compute QK dot products (same thread-group mechanism as v1)
  //   2. Collect scores for this block into a per-warp buffer
  //   3. Update online softmax state (m, l) and rescale O
  //   4. Accumulate weighted V into O
  //
  // The scores for one KV block (up to BLOCK_SIZE floats per warp) are
  // stored in threadgroup memory. Each warp uses its own slice so there
  // are no conflicts between warps.
  threadgroup float *warp_scores =
      reinterpret_cast<threadgroup float *>(shared_mem) +
      warp_idx * BLOCK_SIZE;

  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // --- Step 1: Compute QK scores for this block ---
    // The QK dot product uses "thread groups" (sub-warp groups) where
    // each thread group cooperatively computes the dot product for one
    // KV token. Only thread_group_offset==0 holds the final score.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const device CACHE_T *k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            physical_block_offset * kv_token_stride +
            kv_head_idx * kv_head_stride;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;

        if constexpr (is_uchar<CACHE_T>()) {
          Quant_vec k_vec_quant = *reinterpret_cast<const device Quant_vec *>(
              k_ptr + vec_idx * VEC_SIZE);
          k_vecs[j] = fp8_convert<K_vec, Quant_vec>(k_vec_quant, *k_scale);
        } else {
          k_vecs[j] = *reinterpret_cast<const device K_vec *>(
              k_ptr + vec_idx * VEC_SIZE);
        }
      }

      float qk = scale * Qk_dot<T, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);

      if (softcapping > 0.0f) {
        qk = tanh(qk / softcapping) * softcapping;
      }

      qk +=
          (alibi_slope != 0) ? alibi_slope * (token_idx - effective_context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Causal mask: only attend to KV positions < effective_context_len.
        bool mask = token_idx >= effective_context_len;
        // Sliding window mask: skip positions too far in the past.
        if (sliding_window >= 0) {
          mask = mask || (token_idx < effective_context_len - sliding_window);
        }
        warp_scores[physical_block_offset] = mask ? -FLT_MAX : qk;
      }
    }

    // Ensure all scores for this block are written before reading them.
    // Each warp writes/reads its own warp_scores slice, so we only need
    // intra-SIMD-group visibility. Using simdgroup_barrier (not
    // threadgroup_barrier) because warps may have different iteration
    // counts in the warp-strided loop — threadgroup_barrier would be UB.
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 2: Online softmax update ---
    // Find block-local max (only lane 0 of each thread group has scores,
    // but for V accumulation ALL lanes need the weights, so we broadcast).
    // Valid tokens in this block:
    const int block_start_token = block_idx * BLOCK_SIZE;
    const int block_valid_tokens =
        MIN(BLOCK_SIZE, effective_context_len - block_start_token);

    // Find max score in this block (all lanes participate for speed).
    float block_max = -FLT_MAX;
    for (int t = lane; t < block_valid_tokens; t += NUM_SIMD_LANES) {
      block_max = max(block_max, warp_scores[t]);
    }
    // Reduce max within the warp.
#pragma unroll
    for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
      block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    // Compute correction factor to rescale previous state.
    float new_m = max(warp_m, block_max);
    // NaN-safe: if new_m is still -inf (all masked), clamp to 0.
    if (new_m == -FLT_MAX) new_m = 0.f;

    float old_correction = exp(warp_m - new_m);
    // If warp_m was -FLT_MAX (first iteration), correction = 0, which
    // correctly zeroes out the (already zero) previous O and l.
    if (warp_m == -FLT_MAX) old_correction = 0.f;

    // Rescale running state.
#pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
      v_accs[i] *= old_correction;
    }
    warp_l *= old_correction;
    warp_m = new_m;

    // --- Step 3: Compute exp weights and accumulate V ---
    // For each valid token in this block, compute its softmax weight
    // (unnormalized) and accumulate into O.
    for (int tok = 0; tok < block_valid_tokens; tok++) {
      const float score = warp_scores[tok];
      const float w = exp(score - warp_m);
      warp_l += w;

      // Load V and accumulate: O += w * V
      const device CACHE_T *v_ptr =
          v_cache + physical_block_number * kv_block_stride +
          tok * kv_token_stride +
          kv_head_idx * kv_head_stride;

#pragma unroll
      for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
        const int d = lane + i * NUM_SIMD_LANES;
        if (d < HEAD_SIZE) {
          float v_val;
          if constexpr (is_uchar<CACHE_T>()) {
            v_val = fp8_e4m3_to_float(v_ptr[d]) * (*v_scale);
          } else {
            v_val = float(v_ptr[d]);
          }
          v_accs[i] += w * v_val;
        }
      }
    }

    // Barrier before next iteration reuses warp_scores.
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Ensure all warps have finished the KV loop before reusing shared_mem
  // for the merge. Without this barrier, early-exiting warps could write
  // merge state into shared_mem regions still used as warp_scores by
  // slower warps (the memory aliases).
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ========== Cross-warp merge of online softmax state ==========
  // Each warp has its own (warp_m, warp_l, v_accs[]). We merge by having
  // all warps write their state to shared memory, then warp 0 reads and
  // merges sequentially. Simple and barrier-safe (all barriers are
  // reached by all threads in the threadgroup).

  // For non-partitioned mode, include the sink in each warp's state.
  if (!USE_PARTITIONING && use_sinks) {
    float sink_val = sinks[head_idx];
    float new_m = max(warp_m, sink_val);
    float old_corr = (warp_m == -FLT_MAX) ? 0.f : exp(warp_m - new_m);
#pragma unroll
    for (int i = 0; i < V_ELEMS_PER_THREAD; i++) {
      v_accs[i] *= old_corr;
    }
    warp_l = warp_l * old_corr + exp(sink_val - new_m);
    warp_m = new_m;
  }

  // Shared memory layout for merge:
  //   merge_m[NUM_WARPS]: per-warp max values
  //   merge_l[NUM_WARPS]: per-warp exp sums
  //   merge_O[NUM_WARPS * HEAD_SIZE]: per-warp O accumulators
  threadgroup float *merge_m =
      reinterpret_cast<threadgroup float *>(shared_mem);
  threadgroup float *merge_l = merge_m + NUM_WARPS;
  threadgroup float *merge_O =
      reinterpret_cast<threadgroup float *>(shared_mem) + 2 * NUM_WARPS;

  // All warps write their state to shared memory.
  if (lane == 0) {
    merge_m[warp_idx] = warp_m;
    merge_l[warp_idx] = warp_l;
  }
  // Each warp writes its O accumulator.
  {
    threadgroup float *my_O = &merge_O[warp_idx * HEAD_SIZE];
#pragma unroll
    for (int j = 0; j < V_ELEMS_PER_THREAD; j++) {
      const int d = lane + j * NUM_SIMD_LANES;
      if (d < HEAD_SIZE) {
        my_O[d] = v_accs[j];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Warp 0 reads all warp states and merges sequentially.
  if (warp_idx == 0) {
    // Start with warp 0's state (already in v_accs, warp_m, warp_l).
    // Merge warps 1..NUM_WARPS-1 into it.
    for (int w = 1; w < NUM_WARPS; w++) {
      float other_m = merge_m[w];
      float other_l = merge_l[w];

      // Skip warps that processed no blocks (m == -FLT_MAX).
      if (other_m == -FLT_MAX && other_l == 0.f) continue;

      float new_m = max(warp_m, other_m);
      if (new_m == -FLT_MAX) new_m = 0.f;

      float my_corr = (warp_m == -FLT_MAX) ? 0.f : exp(warp_m - new_m);
      float other_corr = (other_m == -FLT_MAX) ? 0.f : exp(other_m - new_m);

      const threadgroup float *other_O = &merge_O[w * HEAD_SIZE];
#pragma unroll
      for (int j = 0; j < V_ELEMS_PER_THREAD; j++) {
        const int d = lane + j * NUM_SIMD_LANES;
        if (d < HEAD_SIZE) {
          v_accs[j] = v_accs[j] * my_corr + other_O[d] * other_corr;
        }
      }
      warp_m = new_m;
      warp_l = warp_l * my_corr + other_l * other_corr;
    }

    // For partitioned mode, persist the merged partition statistics for the
    // reduce kernel. These must match the normalized tmp_out written below.
    if (USE_PARTITIONING && thread_idx == 0 && use_partitioning) {
      device float *max_logits_ptr =
          max_logits + q_token_idx * num_heads * max_num_partitions +
          head_idx * max_num_partitions + partition_idx;
      *max_logits_ptr = warp_m;
      device float *exp_sums_ptr = exp_sums +
                                   q_token_idx * num_heads * max_num_partitions +
                                   head_idx * max_num_partitions + partition_idx;
      *exp_sums_ptr = warp_l;
    }

    // Final normalization: O = O / l
    const float inv_l = 1.f / (warp_l + 1e-6f);

    device T *out_ptr =
        out + q_token_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int j = 0; j < V_ELEMS_PER_THREAD; j++) {
      const int d = lane + j * NUM_SIMD_LANES;
      if (d < HEAD_SIZE) {
        *(out_ptr + d) = T(v_accs[j] * inv_l);
      }
    }
  }
}

template <typename T, int HEAD_SIZE, int NUM_THREADS, int NUM_SIMD_LANES,
          int PARTITION_SIZE = 0>
[[kernel]] void paged_attention_v2_reduce(
    device T *out [[buffer(0)]], const device float *exp_sums [[buffer(1)]],
    const device float *max_logits [[buffer(2)]],
    const device T *tmp_out [[buffer(3)]],
    device uint32_t *context_lens [[buffer(4)]],
    const constant int &max_num_partitions [[buffer(5)]],
    const device float *sinks
    [[buffer(6), function_constant(use_sinks)]], // [num_heads]
    device const int32_t *cu_seqlens_q [[buffer(7)]],  // [num_seqs + 1]
    const constant int &num_seqs [[buffer(8)]],
    threadgroup char *shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int num_heads = threadgroups_per_grid.x;
  const int head_idx = threadgroup_position_in_grid.x;
  // Varlen: grid.y is q_token_idx (one per query token), not seq_idx.
  const int q_token_idx = threadgroup_position_in_grid.y;
  const int seq_idx = find_seq_idx(cu_seqlens_q, q_token_idx, num_seqs);
  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int q_pos_in_seq = q_token_idx - q_seq_start;
  const uint32_t context_len = context_lens[seq_idx];
  const int effective_context_len = (int)context_len - q_len + q_pos_in_seq + 1;
  const int num_partitions = DIVIDE_ROUND_UP(effective_context_len, PARTITION_SIZE);
  if (num_partitions == 1 && !use_sinks) {
    // No need to reduce. Only copy tmp_out to out.
    device T *out_ptr =
        out + q_token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const device T *tmp_out_ptr =
        tmp_out + q_token_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
         i += threads_per_threadgroup.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  // Workspace for reduction.
  threadgroup float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  threadgroup float *shared_max_logits =
      reinterpret_cast<threadgroup float *>(shared_mem);
  const device float *max_logits_ptr =
      max_logits + q_token_idx * num_heads * max_num_partitions +
      head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = max(max_logit, l);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = simd_shuffle(max_logit, 0);

  // Include the sink in the global max before rescaling.
  if (use_sinks) {
    max_logit = max(max_logit, sinks[head_idx]);
  }

  // Load rescaled exp sums to shared memory.
  threadgroup float *shared_exp_sums = reinterpret_cast<threadgroup float *>(
      shared_mem + sizeof(float) * num_partitions);
  const device float *exp_sums_ptr = exp_sums +
                                     q_token_idx * num_heads * max_num_partitions +
                                     head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * exp(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(
      &red_smem[NUM_WARPS], global_exp_sum, simd_tid, simd_lid);

  // Include the sink in the global exp sum.
  if (use_sinks) {
    global_exp_sum += exp(sinks[head_idx] - max_logit);
  }

  const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const device T *tmp_out_ptr =
      tmp_out + q_token_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  device T *out_ptr =
      out + q_token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
       i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    out_ptr[i] = T(acc);
  }
}

#define instantiate_paged_attention_inner(type, cache_type, head_size,         \
                                          block_size, num_threads,             \
                                          num_simd_lanes, partition_size)      \
  template [[host_name("paged_attention_" #type "_cache_" #cache_type          \
                       "_hs" #head_size "_bs" #block_size "_nt" #num_threads   \
                       "_nsl" #num_simd_lanes                                  \
                       "_ps" #partition_size)]] [[kernel]] void                \
  paged_attention<type, cache_type, head_size, block_size, num_threads,        \
                  num_simd_lanes, partition_size>(                             \
      device float *exp_sums                                                   \
      [[buffer(0), function_constant(use_partitioning)]],                      \
      device float *max_logits                                                 \
      [[buffer(1), function_constant(use_partitioning)]],                      \
      device type *out [[buffer(2)]], device const type *q [[buffer(3)]],      \
      device const cache_type *k_cache [[buffer(4)]],                          \
      device const cache_type *v_cache [[buffer(5)]],                          \
      device const float *k_scale                                              \
      [[buffer(6), function_constant(use_fp8_scales)]],                        \
      device const float *v_scale                                              \
      [[buffer(7), function_constant(use_fp8_scales)]],                        \
      const constant int &num_kv_heads [[buffer(8)]],                          \
      const constant float &scale [[buffer(9)]],                               \
      const constant float &softcapping [[buffer(10)]],                        \
      device const uint32_t *block_tables [[buffer(11)]],                      \
      device const uint32_t *context_lens [[buffer(12)]],                      \
      const constant int &max_num_blocks_per_seq [[buffer(13)]],               \
      device const float *alibi_slopes                                         \
      [[buffer(14), function_constant(use_alibi)]],                            \
      const constant int &q_stride [[buffer(15)]],                             \
      const constant int &kv_block_stride [[buffer(16)]],                      \
      const constant int &kv_head_stride [[buffer(17)]],                       \
      const device float *sinks [[buffer(18), function_constant(use_sinks)]],  \
      device const int32_t *cu_seqlens_q [[buffer(19)]],                       \
      const constant int &num_seqs [[buffer(20)]],                             \
      const constant int &sliding_window [[buffer(21)]],                       \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_v2_reduce_inner(                           \
    type, head_size, num_threads, num_simd_lanes, partition_size)              \
  template [[host_name("paged_attention_v2_reduce_" #type "_hs" #head_size     \
                       "_nt" #num_threads "_nsl" #num_simd_lanes               \
                       "_ps" #partition_size)]] [[kernel]] void                \
  paged_attention_v2_reduce<type, head_size, num_threads, num_simd_lanes,      \
                            partition_size>(                                   \
      device type * out [[buffer(0)]],                                         \
      const device float *exp_sums [[buffer(1)]],                              \
      const device float *max_logits [[buffer(2)]],                            \
      const device type *tmp_out [[buffer(3)]],                                \
      device uint32_t *context_lens [[buffer(4)]],                             \
      const constant int &max_num_partitions [[buffer(5)]],                    \
      const device float *sinks [[buffer(6), function_constant(use_sinks)]],   \
      device const int32_t *cu_seqlens_q [[buffer(7)]],                        \
      const constant int &num_seqs [[buffer(8)]],                              \
      threadgroup char *shared_mem [[threadgroup(0)]],                         \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],     \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                   \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
      uint3 threads_per_threadgroup [[threads_per_threadgroup]],               \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_heads(                                     \
    type, cache_type, block_size, num_threads, num_simd_lanes, partition_size) \
  instantiate_paged_attention_inner(type, cache_type, 64, block_size,          \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, cache_type, 80, block_size,          \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, cache_type, 96, block_size,          \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, cache_type, 112, block_size,         \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, cache_type, 128, block_size,         \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, cache_type, 192, block_size,         \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);                           \
  instantiate_paged_attention_inner(type, cache_type, 256, block_size,         \
                                    num_threads, num_simd_lanes,               \
                                    partition_size);

#define instantiate_paged_attention_v2_reduce_heads(                           \
    type, num_threads, num_simd_lanes, partition_size)                         \
  instantiate_paged_attention_v2_reduce_inner(type, 64, num_threads,           \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 80, num_threads,           \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 96, num_threads,           \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 112, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 128, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 192, num_threads,          \
                                              num_simd_lanes, partition_size); \
  instantiate_paged_attention_v2_reduce_inner(type, 256, num_threads,          \
                                              num_simd_lanes, partition_size);

#define instantiate_paged_attention_block_size(type, cache_type, num_threads,  \
                                               num_simd_lanes, partition_size) \
  instantiate_paged_attention_heads(type, cache_type, 1, num_threads,          \
                                    num_simd_lanes, partition_size);           \
  instantiate_paged_attention_heads(type, cache_type, 8, num_threads,          \
                                    num_simd_lanes, partition_size);           \
  instantiate_paged_attention_heads(type, cache_type, 16, num_threads,         \
                                    num_simd_lanes, partition_size);           \
  instantiate_paged_attention_heads(type, cache_type, 32, num_threads,         \
                                    num_simd_lanes, partition_size);

// TODO: tune num_threads = 256
// NOTE: partition_size = 0
#define instantiate_paged_attention_v1(type, cache_type, num_simd_lanes)       \
  instantiate_paged_attention_block_size(type, cache_type, 256,                \
                                         num_simd_lanes, 0);

// TODO: tune num_threads = 256
// NOTE: partition_size = VLLM_METAL_PARTITION_SIZE
#define instantiate_paged_attention_v2(type, cache_type, num_simd_lanes)       \
  instantiate_paged_attention_block_size(type, cache_type, 256,                \
                                         num_simd_lanes,                        \
                                         VLLM_METAL_PARTITION_SIZE);

// TODO: tune num_threads = 256
// NOTE: partition_size = VLLM_METAL_PARTITION_SIZE
#define instantiate_paged_attention_v2_reduce(type, num_simd_lanes)            \
  instantiate_paged_attention_v2_reduce_heads(                                  \
      type, 256, num_simd_lanes, VLLM_METAL_PARTITION_SIZE);

instantiate_paged_attention_v1(float, float, 32);
instantiate_paged_attention_v1(bfloat16_t, bfloat16_t, 32);
instantiate_paged_attention_v1(half, half, 32);

// instantiate_paged_attention_v1(float, uchar, 32);
// instantiate_paged_attention_v1(bfloat16_t, uchar, 32);
// instantiate_paged_attention_v1(half, uchar, 32);

instantiate_paged_attention_v2_reduce(float, 32);
instantiate_paged_attention_v2_reduce(bfloat16_t, 32);
instantiate_paged_attention_v2_reduce(half, 32);

instantiate_paged_attention_v2(float, float, 32);
instantiate_paged_attention_v2(bfloat16_t, bfloat16_t, 32);
instantiate_paged_attention_v2(half, half, 32);

// instantiate_paged_attention_v2(float, uchar, 32);
// instantiate_paged_attention_v2(bfloat16_t, uchar, 32);
// instantiate_paged_attention_v2(half, uchar, 32);