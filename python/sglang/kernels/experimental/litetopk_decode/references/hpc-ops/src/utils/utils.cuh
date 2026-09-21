// Copyright (C) 2026 Tencent.

#ifndef SRC_UTILS_UTILS_CUH_
#define SRC_UTILS_UTILS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"
#include "src/utils/utils.h"

namespace hpc {

// ============================
//    Debug Utility
// ============================

// print_type<T> _; // it will generate a compile time error with type T.
template <typename T>
struct print_type;

__device__ __forceinline__ void brkpt() { asm volatile("brkpt;" ::); }

// ============================
//    Load/Store(vectorized)
// ============================
template <typename T, int N>
struct vec_t {
  T data[N];

  using type = T;
  static constexpr int num = N;
  static constexpr int kNum = N;

  __device__ __forceinline__ constexpr T &operator[](int idx) { return data[idx]; }

  __device__ __forceinline__ constexpr const T &operator[](int idx) const { return data[idx]; }
};

template <typename T, int N, int... Dims>
struct traits_vec_t;

template <typename T, int N, int Dim>
struct traits_vec_t<T, N, Dim> {
  static_assert(N == Dim, "dimension mismatch");
  using type = vec_t<T, Dim>;
};

template <typename T, int N, int First, int... Rest>
struct traits_vec_t<T, N, First, Rest...> {
  static_assert(N % First == 0, "first dimension must divide total size");
  using inner_type = typename traits_vec_t<T, N / First, Rest...>::type;
  using type = vec_t<inner_type, First>;
};

template <typename T, int N>
__device__ __forceinline__ constexpr int size(vec_t<T, N> &v) {
  return N;
}

template <int... Dims, typename T, int N>
__device__ __forceinline__ constexpr auto &reshape(vec_t<T, N> &v) {
  constexpr int num_elements = (Dims * ...);

  using ResultType = typename traits_vec_t<T, N, Dims...>::type;
  return *reinterpret_cast<ResultType *>(&v);
}

template <typename U, typename T, int N>
__device__ __forceinline__ constexpr auto to(const vec_t<T, N> &v) {
  if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat16>) {
    using V = vec_t<__nv_bfloat16, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __float2bfloat16(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half2>) {
    using V = vec_t<U, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __float22half2_rn(*reinterpret_cast<const float2 *>(&v[2 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<U, float>) {
    using V = vec_t<float, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __bfloat162float(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat162> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = __bfloat1622float2(v[i]);
      o[i * 2 + 0] = y.x;
      o[i * 2 + 1] = y.y;
    }
    return o;
  } else if constexpr (std::is_same_v<T, __half2> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = __half22float2(v[i]);
      o[i * 2 + 0] = y.x;
      o[i * 2 + 1] = y.y;
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_fp8x4_e4m3>) {
    static_assert(N % 4 == 0, "N % 4 must be 0");
    using V = vec_t<__nv_fp8x4_e4m3, N / 4>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 4; ++i) {
      o[i] = __nv_fp8x4_e4m3(*reinterpret_cast<const float4 *>(&v[4 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_fp8x4_e4m3> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 4>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = static_cast<float4>(v[i]);
      o[i * 4 + 0] = y.x;
      o[i * 4 + 1] = y.y;
      o[i * 4 + 2] = y.z;
      o[i * 4 + 3] = y.w;
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3> && std::is_same_v<U, float>) {
    using V = vec_t<float, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = static_cast<float>(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat162> && std::is_same_v<U, __nv_fp8x4_e4m3>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__nv_fp8x4_e4m3, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __nv_fp8x4_e4m3(*(reinterpret_cast<__nv_bfloat162 *>(&v[2 * i])),
                             *(reinterpret_cast<__nv_bfloat162 *>(&v[2 * i + 1])));
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat162>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__nv_bfloat162, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __float22bfloat162_rn(*reinterpret_cast<const float2 *>(&v[2 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, U>) {
    return v;
  }
}

template <typename T, int N>
__device__ __forceinline__ constexpr auto load(const void *ptr) {
  using V = vec_t<T, N>;
  V v;

  constexpr int kBytes = sizeof(T) * N;

  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "not support for T x N");

  if constexpr (kBytes == 1) {
    using L = uint8_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 2) {
    using L = uint16_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 4) {
    using L = uint32_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 8) {
    using L = uint64_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 16) {
    using L = uint4;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  }

  return v;
}

template <typename T, int N>
__device__ __forceinline__ constexpr void store(void *ptr, const vec_t<T, N> &v) {
  using V = vec_t<T, N>;

  constexpr int kBytes = sizeof(T) * N;

  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "not support for T x N");

  if constexpr (kBytes == 1) {
    using S = uint8_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 2) {
    using S = uint16_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 4) {
    using S = uint32_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 8) {
    using S = uint64_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 16) {
    using S = uint4;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  }

  return;
}

template <typename T, typename... Args>
__device__ __forceinline__ constexpr void store(void *ptr, T val, Args... vals) {
  constexpr int N = sizeof...(Args);
  using V = vec_t<T, 1 + N>;

  static_assert((std::is_same_v<Args, T> && ...), "all vals must be type of T");

  V v;
  int idx = 0;
  (reinterpret_cast<T *>(&v))[idx++] = val;
  (((reinterpret_cast<T *>(&v))[idx++] = vals), ...);

  store(ptr, v);
}

// ============================
//    Multimem Load/Store Ops
// ============================

template <typename T, int N>
__device__ __forceinline__ auto multi_load_reduce_add(const void *ptr) {
  if constexpr (std::is_same_v<T, __nv_bfloat162> && N == 4) {
    using V = vec_t<T, N>;
    V v;
    auto *l = reinterpret_cast<uint4 *>(&v);
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2"
        " {%0,%1,%2,%3}, [%4];"
        : "=r"(l->x), "=r"(l->y), "=r"(l->z), "=r"(l->w)
        : "l"(ptr)
        : "memory");
    return v;
  }
}

template <typename T, int N>
__device__ __forceinline__ void multi_store(void *mc_ptr, const vec_t<T, N> &v) {
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 8 || kBytes == 16, "not support for T x N");

  if constexpr (kBytes == 8) {
    const uint2 *s = reinterpret_cast<const uint2 *>(&v);
    asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};"
                 :
                 : "l"(mc_ptr), "r"(s->x), "r"(s->y)
                 : "memory");
  } else if constexpr (kBytes == 16) {
    const uint4 *s = reinterpret_cast<const uint4 *>(&v);
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(mc_ptr), "r"(s->x), "r"(s->y), "r"(s->z), "r"(s->w)
                 : "memory");
  }
}

// ============================
//       Fast Math API
// ============================

__device__ __forceinline__ float expf_ftz(float x) {
  // e^x = (2^m)^x
  // e = 2^m
  // m = lg2(e)
  // m = 1.4426950408889634

  const float m = 1.4426950408889634f;
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x * m));
  return r;
}

__device__ __forceinline__ float exp2f_ftz(float x) {
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float log2f_ftz(float x) {
  float r;
  asm volatile("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float logf_ftz(float x) {
  // log(x) = lg2(x)log(2)
  // m = log(2) = 0.6931471805599453

  const float m = 0.6931471805599453f;
  float r;
  asm volatile("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r * m;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

// y = 1 / (1 + e^(-x))
__device__ __forceinline__ float sigmoid(float x) { return rcpf_ftz(1.f + expf_ftz(-x)); }
__device__ __forceinline__ float sqrt_ftz(float x) {
  float r;
  asm volatile("sqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

// y = max(0, x)
__device__ __forceinline__ float relu(float x) { return fmaxf(0, x); }

// y = x / (1 + e^(-x))
__device__ __forceinline__ float silu(float x) { return x * rcpf_ftz(1.f + expf_ftz(-x)); }

// y = log(1 + exp(x))
__device__ __forceinline__ float softplus(float x) { return logf_ftz(1.f + expf_ftz(x)); }

__device__ __forceinline__ float rsqrtf_ftz(float in) {
  float out;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(out) : "f"(in));
  return out;
}

__device__ __forceinline__ float warp_reduce_sum_down(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x += __shfl_down_sync(0xFFFFFFFF, x, ioffset);
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_max_down(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x = fmaxf(x, __shfl_down_sync(0xFFFFFFFF, x, ioffset));
  }

  return x;
}

__device__ __forceinline__ float half_warp_reduce_max_down(float x) {
  const int width = 16;

#pragma unroll
  for (int ioffset = width / 2; ioffset >= 1; ioffset /= 2) {
    x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFF, x, ioffset, width));
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_sum_xor(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x += __shfl_xor_sync(0xFFFFFFFF, x, ioffset);
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_max_xor(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, ioffset), x);
  }

  return x;
}

__device__ __forceinline__ float warp_4lane_reduce_max_xor(float x) {
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 1), x);
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 2), x);

  return x;
}

__device__ __forceinline__ float warp_4lane_reduce_sum_xor(float x) {
  x += __shfl_xor_sync(0xFFFFFFFF, x, 1);
  x += __shfl_xor_sync(0xFFFFFFFF, x, 2);

  return x;
}

__device__ __forceinline__ float warp_8lane_stride4_reduce_max_xor(float x) {
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 4), x);
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 8), x);
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 16), x);

  return x;
}

__device__ __forceinline__ float warp_8lane_stride4_reduce_sum_xor(float x) {
  x += __shfl_xor_sync(0xFFFFFFFF, x, 4);
  x += __shfl_xor_sync(0xFFFFFFFF, x, 8);
  x += __shfl_xor_sync(0xFFFFFFFF, x, 16);

  return x;
}

// ============================
//      Attention Utility
// ============================

template <typename Tensor>
__device__ __forceinline__ constexpr auto retile_fragment(Tensor &&tensor) {
  using namespace cute;  // NOLINT

  constexpr int R = decltype(tensor.layout())::rank;
  static_assert(R >= 3, "rank must geater than or equal to 3");

  auto thr_vmk = append<3>(flatten(select<0>(tensor.layout())));
  auto tile_mk = take<1, R>(tensor.layout());

  auto m_layout =
      coalesce(make_layout(make_shape(get<1>(thr_vmk.shape()), get<0>(tile_mk.shape())),
                           make_stride(get<1>(thr_vmk.stride()), get<0>(tile_mk.stride()))));
  auto k_layout = coalesce(make_layout(
      make_shape(get<0>(thr_vmk.shape()), get<2>(thr_vmk.shape()), get<1>(tile_mk.shape())),
      make_stride(get<0>(thr_vmk.stride()), get<2>(thr_vmk.stride()), get<1>(tile_mk.stride()))));

  if constexpr (R == Int<3>{}) {
    return make_tensor(static_cast<Tensor &&>(tensor).data(), make_layout(m_layout, k_layout));
  } else {
    auto r_layout = take<3, R>(make_layout(tensor.shape(), tensor.stride()));
    auto mkr_layout = make_layout(m_layout, k_layout, r_layout);
    auto t = make_tensor(static_cast<Tensor &&>(tensor).data(), mkr_layout);
    return t(_, _, repeat<R - Int<3>{}>(_));
  }
}

// STensor shape is (M, N) and row major
// Dtensor shape is (M, N) and column major
template <typename STensor, typename DTensor>
__device__ __forceinline__ constexpr void smem_trans_and_interleave0189_mn_mn(const STensor &sV,
                                                                              DTensor &&sVt,
                                                                              int ilane) {
  using namespace cute;  // NOLINT
  using T = typename STensor::value_type;
  using Layout = typename STensor::layout_type;

  static_assert(sizeof(T) == 1, "sV T must be 1 byte dtype");
  static_assert(STensor::rank == 2, "sV rank support rank 2");
  constexpr int kM = shape<0>(Layout{});
  constexpr int kN = shape<1>(Layout{});

  static_assert(((kM == 16) && (kN == 32)) || ((kM == 32) && (kN == 16)),
                "sV shape must be 16x32 or 32x16");

  auto tile = make_tile(Int<8>{}, Int<16>{});
  auto vtile = make_tile(Int<16>{}, Int<16>{});
  auto sV_tile = flat_divide(sV, tile);
  auto sVt_tile = flat_divide(sVt, vtile);

  auto m = size<2>(sV_tile);
  auto n = size<3>(sV_tile);

  auto mt = size<2>(sVt_tile);
  auto nt = size<3>(sVt_tile);

  auto s2r_tiled_copy =
      make_tiled_copy(Copy_Atom<SM75_U16x8_LDSM_T, T>{},
                      make_layout(make_shape(Int<4>{}, Int<8>{}, Int<1>{}, Int<1>{}),
                                  make_stride(Int<1>{}, Int<4>{}, Int<0>{}, Int<0>{})),
                      make_layout(make_shape(Int<2>{}, Int<2>{}, m, n),
                                  make_stride(Int<2>{}, Int<1>{}, Int<4>{}, Int<4>{} * m)));

  auto r2s_tiled_copy =
      make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, T>{},
                      make_layout(make_shape(Int<4>{}, Int<8>{}, Int<1>{}, Int<1>{}),
                                  make_stride(Int<1>{}, Int<4>{}, Int<0>{}, Int<0>{})),
                      make_layout(make_shape(Int<4>{}, Int<2>{}, mt, nt),
                                  make_stride(Int<1>{}, Int<4>{}, Int<8>{}, Int<8>{})));

  auto s2r_thr_copy = s2r_tiled_copy.get_slice(ilane);
  auto tVs4r = s2r_thr_copy.partition_S(sV_tile);
  auto tVr4s = make_fragment_like(s2r_thr_copy.partition_D(sV_tile));
  auto v = recast<uint32_t>(coalesce(tVr4s));

  auto r2s_thr_copy = r2s_tiled_copy.get_slice(ilane);
  auto tVtr4s = make_fragment_like(r2s_thr_copy.partition_S(sVt_tile));
  auto tVts4r = r2s_thr_copy.partition_D(sVt_tile);
  auto vt = recast<uint32_t>(coalesce(tVtr4s));

  cute::copy(s2r_tiled_copy, tVs4r, tVr4s);

  vt(0) = __byte_perm(v(0), v(1), 0x6420);
  vt(1) = __byte_perm(v(0), v(1), 0x7531);

  vt(2) = __byte_perm(v(2), v(3), 0x6420);
  vt(3) = __byte_perm(v(2), v(3), 0x7531);

  cute::copy(r2s_tiled_copy, tVtr4s, tVts4r);
}

template <typename STensor, typename DTensor>
__device__ __forceinline__ constexpr void smem_trans_and_interleave0189_mn_nm(const STensor &sV,
                                                                              DTensor &&sVtp,
                                                                              int ilane) {
  auto sVt = make_tensor(sVtp.data(), cute::select<1, 0>(sVtp.layout()));
  smem_trans_and_interleave0189_mn_mn(sV, sVt, ilane);
}

template <typename STensor, typename DTensor>
__device__ __forceinline__ constexpr void smem_trans_and_interleave0189_nm_nm(const STensor &sVp,
                                                                              DTensor &&sVtp,
                                                                              int ilane) {
  auto sV = make_tensor(sVp.data(), cute::select<1, 0>(sVp.layout()));
  auto sVt = make_tensor(sVtp.data(), cute::select<1, 0>(sVtp.layout()));
  smem_trans_and_interleave0189_mn_mn(sV, sVt, ilane);
}

// ============================
//    Atom CAS Primitives
// ============================

__device__ __forceinline__ int atom_cas_relaxed(uint32_t *addr, uint32_t compare, uint32_t value) {
  uint32_t old_val;
  asm volatile("atom.global.relaxed.sys.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old_val)
               : "l"(addr), "r"(compare), "r"(value)
               : "memory");
  return old_val;
}

__device__ __forceinline__ int atom_cas_release(uint32_t *addr, uint32_t compare, uint32_t value) {
  uint32_t old_val;
  asm volatile("atom.global.release.sys.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old_val)
               : "l"(addr), "r"(compare), "r"(value)
               : "memory");
  return old_val;
}

__device__ __forceinline__ int atom_cas_acquire(uint32_t *addr, uint32_t compare, uint32_t value) {
  uint32_t old_val;
  asm volatile("atom.global.acquire.sys.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old_val)
               : "l"(addr), "r"(compare), "r"(value)
               : "memory");
  return old_val;
}

__device__ __forceinline__ void put_signal_relaxed(uint32_t *addr) {
  while (atom_cas_relaxed(addr, 0, 1) != 0) {
  }
}

__device__ __forceinline__ void put_signal_release(uint32_t *addr) {
  while (atom_cas_release(addr, 0, 1) != 0) {
  }
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t *addr) {
  while (atom_cas_relaxed(addr, 1, 0) != 1) {
  }
}

__device__ __forceinline__ void wait_signal_acquire(uint32_t *addr) {
  while (atom_cas_acquire(addr, 1, 0) != 1) {
  }
}

// ================================
//    Memory-order-aware LD/ST Primitives
// ================================

__device__ int __forceinline__ load_global_volatile(int *ptr) {
  int val;
  asm volatile("ld.volatile.global.s32 {%0}, [%1];\n" : "=r"(val) : "l"(ptr));
  return val;
}

// ================================
//    Synchronization Primitives
// ================================

__device__ __forceinline__ void syncwarpgroup(int barrier_id) {
  asm volatile("barrier.cta.sync %0, 128;\n" ::"r"(barrier_id) : "memory");
}

template <int N>
__device__ __forceinline__ void bar_sync(int barrier_id) {
  asm volatile("barrier.cta.sync %0, %1;\n" ::"r"(barrier_id), "n"(N) : "memory");
}

__device__ __forceinline__ void fence_async_global() {
  asm volatile("fence.proxy.async.global;\n");
}

__device__ __forceinline__ void set_barrier_transaction_bytes_cluster(uint64_t &smem_bar,
                                                                      uint32_t transaction_bytes,
                                                                      uint32_t cta_id = 0) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_bar);
  asm volatile(
      "{\n\t"
      ".reg .b32 remAddr32;\n\t"
      "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
      "mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %2;\n\t"
      "}"
      :
      : "r"(smem_addr), "r"(cta_id), "r"(transaction_bytes));
}

__device__ __forceinline__ void arrive_cluster_barrier(uint64_t &smem_bar, uint32_t cta_id = 0) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_bar);
  asm volatile(
      "{\n\t"
      ".reg .b32 remAddr32;\n\t"
      "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
      "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
      "}"
      :
      : "r"(smem_addr), "r"(cta_id));
}

__device__ __forceinline__ void cluster_relaxed_sync() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

__device__ __forceinline__ uint32_t test_barrier(uint64_t &bar, int phase) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(&bar);
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t"
      "}\n"
      : "=r"(ok)
      : "r"(addr), "r"(phase));
  return ok;
}

__device__ __forceinline__ void cp_async_16b(void *smem_dst, const void *gmem_src, int src_size) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gmem_src),
               "r"(src_size));
}

__device__ __forceinline__ void find_next_block(uint4 *clc_response, uint64_t *mbar) {
  uint32_t mbar_ptr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void *>(mbar));
  uint32_t clc_resp_ptr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void *>(clc_response));
  asm volatile(
      "{\n\t"
      "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::"
      "cluster::all.b128 [%0], [%1];\n\t"
      "}\n"
      :
      : "r"(clc_resp_ptr), "r"(mbar_ptr));
}

__device__ __forceinline__ auto get_next_block(uint4 *clc_response) {
  uint32_t clc_resp_ptr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void *>(clc_response));
  uint32_t iblock;
  uint32_t valid;
  asm volatile(
      "{\n"
      ".reg .pred p1;\n\t"
      ".reg .b128 clc_result;\n\t"
      "ld.shared.b128 clc_result, [%2];\n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"
      "selp.u32 %1, 1, 0, p1;\n\t"
      "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_result;\n\t"
      "fence.proxy.async.shared::cta;\n\t"
      "}\n"
      : "=r"(iblock), "=r"(valid)
      : "r"(clc_resp_ptr)
      : "memory");

  return cute::make_tuple(iblock, valid);
}

}  // namespace hpc

namespace cute {

template <class... Args, class ThrLayout, class ValLayout, class Tiler>
CUTE_HOST_DEVICE auto make_tiled_copy(Copy_Atom<Args...> const &copy_atom,
                                      ThrLayout const &thr_layout, ValLayout const &val_layout,
                                      Tiler const &tiler) {
  // Take the raked_products to compute the Layout_MN
  // (M,N) -> (thr_idx, val_idx)
  auto layout_mn = raked_product(thr_layout, val_layout);
  // (thr_idx, val_idx) -> (M,N)
  auto layout_tv =
      right_inverse(layout_mn).with_shape(make_shape(size(thr_layout), size(val_layout)));

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}

}  // namespace cute

#endif  // SRC_UTILS_UTILS_CUH_
