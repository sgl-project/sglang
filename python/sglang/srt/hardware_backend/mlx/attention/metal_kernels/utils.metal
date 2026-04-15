// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

#include "float8.metal"
#include <metal_stdlib>

using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype)    \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {              \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)         \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }                                                                            \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                         \
  bfloat_binop_base(_op_, _operator_, _MLX_BFloat16, _MLX_BFloat16,            \
                    _MLX_BFloat16, float);                                     \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                  \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                   \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);       \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                                    \
  bfloat_binop_base(__op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16,  \
                    float);                                                    \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);               \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);                \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);            \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space)      \
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \
      addr_space _MLX_BFloat16 &lhs, itype rhs) {                              \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }                                                                            \
  constexpr METAL_FUNC addr_space itype &__operator__(addr_space itype &lhs,   \
                                                      _MLX_BFloat16 rhs) {     \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype)       \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);               \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);               \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                                               \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype);                   \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype);                   \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype);                   \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space)             \
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \
      addr_space _MLX_BFloat16 &lhs, _MLX_BFloat16 rhs) {                      \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__)              \
  bfloat_inplace_op_helper(__op__, __operator__, device);                      \
  bfloat_inplace_op_helper(__op__, __operator__, thread);                      \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif