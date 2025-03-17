#pragma once

#include <ATen/native/CPUBlas.h>

// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// block size for AMX gemm
constexpr int block_size_m() { return 2 * TILE_M; }
constexpr int block_size_n() { return 2 * TILE_N; }

//template <typename T> constexpr int vnni_blk();
//template <> constexpr int vnni_blk<at::BFloat16>() { return 2; }
//template <> constexpr int vnni_blk<at::Half>() { return 2; }
//template <> constexpr int vnni_blk<int8_t>() { return 4; }

// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

// adjust leading dimension size for K
template <typename T>
inline int get_row_size(int K) {
  return K;
}

template <>
inline int get_row_size<int8_t>(int K) {
  return K + sizeof(int32_t);
}

// pack weight to vnni format
at::Tensor convert_weight_packed(at::Tensor& weight);

// TODO: debug print, remove me later
inline void print_16x32i(const __m512i x) {
  int32_t a[16];
  _mm512_storeu_si512((__m512i *)a, x);

  for (int i = 0; i < 16; i++){
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

inline void print_16x32(const __m512 x) {
  float a[16];
  _mm512_storeu_ps((__m512 *)a, x);

  for (int i = 0; i < 16; i++){
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}


inline void print_32x8u(const __m256i x) {
  uint8_t a[32];
  _mm256_storeu_si256((__m256i *)a, x);

  for (int i = 0; i < 32; ++i) {
    std::cout << int32_t(a[i]) << " ";
  }
  std::cout << std::endl;
}
