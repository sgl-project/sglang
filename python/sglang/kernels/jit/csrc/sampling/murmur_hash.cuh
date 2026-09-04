#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/// \brief Rotate a 32-bit value left by `r` bits.
SGL_DEVICE uint32_t murmur3_rotl32(uint32_t x, uint32_t r) {
  return (x << r) | (x >> (32u - r));
}

/// \brief Mix one 32-bit key into the MurmurHash3 accumulator.
///
/// Mirrors the reference MurmurHash3 x86_32 body: the accumulator and key
/// both go through the k*^= rotations and the len-4 mixing step.
SGL_DEVICE uint32_t murmur3_mix32(uint32_t h, uint32_t k) {
  k *= 0xCC9E2D51u;
  k = murmur3_rotl32(k, 15u);
  k *= 0x1B873593u;
  h ^= k;
  h = murmur3_rotl32(h, 13u);
  h = h * 5u + 0xE6546B64u;
  return h;
}

/// \brief Final avalanche pass of MurmurHash3 x86_32.
SGL_DEVICE uint32_t murmur3_fmix32(uint32_t h) {
  h ^= h >> 16u;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13u;
  h *= 0xC2B2AE35u;
  h ^= h >> 16u;
  return h;
}

/// \brief Shared launch parameters for the MurmurHash32 kernel.
///
/// `positions` and `col_indices` are stored as `void*` and read through the
/// templated element type so every 32-bit integer dtype (signed or unsigned,
/// 32- or 64-bit) truncates to `uint32_t` exactly the way the reference
/// Triton kernel's `.to(tl.uint32)` does.
struct MurmurHashParams {
  const uint64_t* __restrict__ seed;     // [n] row seeds
  const void* __restrict__ positions;    // [n] per-row positions, truncated to u32
  const void* __restrict__ col_indices;  // [m] per-column indices, truncated to u32
  uint32_t* __restrict__ out;            // [n * m]
  uint32_t m;                            // number of columns
};

/// \brief MurmurHash3 x86_32 over seed, position, and column index.
///
/// Treats the 64-bit seed, 32-bit position, and 32-bit column index as four
/// 4-byte blocks, bit-blends them through `murmur3_mix32`, then finalizes
/// with length 16. Bit-identical to `sglang.kernels.ops.sampling.murmur_hash`
/// Triton reference.
///
/// \tparam TPos Element type of `positions` (int32/int64/uint32/uint64)
/// \tparam TCol Element type of `col_indices` (int32/int64/uint32/uint64)
template <typename TPos, typename TCol>
__global__ void murmur_hash32_kernel(const __grid_constant__ MurmurHashParams params) {
  // Rows on grid.x (batch can be large; grid.x spans 2^31-1), column blocks on
  // grid.y. Threads in a warp keep consecutive `col`, so `out[row * m + col]`
  // stays coalesced.
  const uint32_t row = blockIdx.x;
  const uint32_t col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= params.m) return;

  const uint64_t seed = params.seed[row];
  const uint32_t pos = static_cast<uint32_t>(static_cast<const TPos*>(params.positions)[row]);
  const uint32_t col_val = static_cast<uint32_t>(static_cast<const TCol*>(params.col_indices)[col]);

  uint32_t h = murmur3_mix32(0u, static_cast<uint32_t>(seed & 0xFFFFFFFFull));
  h = murmur3_mix32(h, static_cast<uint32_t>(seed >> 32u));
  h = murmur3_mix32(h, pos);
  h = murmur3_mix32(h, col_val);
  h ^= 16u;
  h = murmur3_fmix32(h);

  params.out[static_cast<uint64_t>(row) * params.m + col] = h;
}

/// \brief Host launcher for `murmur_hash32_kernel`.
///
/// \tparam TPos Element type of `positions` (int32/int64/uint32/uint64)
/// \tparam TCol Element type of `col_indices` (int32/int64/uint32/uint64)
template <typename TPos, typename TCol>
struct MurmurHashKernel {
  static constexpr uint32_t kBlockSize = 256u;

  static void launch(
      const tvm::ffi::TensorView seed,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView col_indices,
      const tvm::ffi::TensorView out) {
    using namespace host;

    auto N = SymbolicSize{"num_rows"};
    auto M = SymbolicSize{"num_cols"};
    auto Total = SymbolicSize{"numel"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N}).with_dtype<uint64_t>().with_device(device_).verify(seed);
    TensorMatcher({N}).with_dtype<TPos>().with_device(device_).verify(positions);
    TensorMatcher({M}).with_dtype<TCol>().with_device(device_).verify(col_indices);
    TensorMatcher({Total}).with_dtype<uint32_t>().with_device(device_).verify(out);

    const auto n = N.unwrap();
    const auto m = M.unwrap();
    CHECK_HOST(n * m == Total.unwrap()) << "out numel must equal n * m";
    if (n == 0 || m == 0) return;

    // grid.x carries one block per row; grid.y one block per kBlockSize columns.
    CHECK_HOST(n <= 2147483647LL) << "num_rows " << n << " exceeds CUDA grid.x limit (2^31-1)";
    const auto col_blocks = div_ceil(static_cast<uint32_t>(m), kBlockSize);
    CHECK_HOST(col_blocks <= 65535u) << "num_cols too large for CUDA grid.y (" << col_blocks << " > 65535 blocks)";

    const auto params = MurmurHashParams{
        .seed = static_cast<const uint64_t*>(seed.data_ptr()),
        .positions = positions.data_ptr(),
        .col_indices = col_indices.data_ptr(),
        .out = static_cast<uint32_t*>(out.data_ptr()),
        .m = static_cast<uint32_t>(m),
    };
    const auto grid = dim3(static_cast<uint32_t>(n), col_blocks);
    LaunchKernel(grid, kBlockSize, device_.unwrap())(murmur_hash32_kernel<TPos, TCol>, params);
  }
};

}  // namespace sglang
