#pragma once

#include <ATen/native/CPUBlas.h>

// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 2
#define TILE_SIZE 512

// block size for AMX gemm
constexpr int block_size_m() { return 2 * TILE_M; }
constexpr int block_size_n() { return 2 * TILE_N; }

// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

// pack weight to vnni format
at::Tensor convert_weight_packed(at::Tensor& weight);
