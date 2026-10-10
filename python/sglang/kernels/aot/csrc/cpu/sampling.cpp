#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>

#include "common.h"

namespace {
inline uint32_t rotl32(uint32_t x, int r) {
  return (x << r) | (x >> (32 - r));
}

inline uint32_t fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;

  return h;
}

inline uint32_t murmur3_mix(uint32_t h, uint32_t k) {
  k *= 0xCC9E2D51u;
  k = rotl32(k, 15);
  k *= 0x1B873593u;

  h ^= k;
  h = rotl32(h, 13);
  h = h * 5u + 0xE6546B64u;

  return h;
}

template <typename pos_t>
void murmur_hash32_kernel_impl(
    const uint64_t* seed_ptr,
    const pos_t* positions_ptr,
    const int64_t* col_indices_ptr,
    uint32_t* output_ptr,
    int64_t n,
    int64_t m) {
  const int64_t total = n * m;

  at::parallel_for(0, total, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t row = idx / m;
      const int64_t col_idx = idx % m;
      const uint64_t seed = seed_ptr[row];

      const uint32_t pos = static_cast<uint32_t>(positions_ptr[row]);

      const uint32_t col = static_cast<uint32_t>(col_indices_ptr[col_idx]);

      // Split 64-bit seed into two 32-bit blocks.
      const uint32_t seed_low = static_cast<uint32_t>(seed);

      const uint32_t seed_high = static_cast<uint32_t>(seed >> 32);

      uint32_t h = 0;

      // Process seed_low
      h = murmur3_mix(h, seed_low);

      // Process seed_high
      h = murmur3_mix(h, seed_high);

      // position
      h = murmur3_mix(h, pos);

      // column index
      h = murmur3_mix(h, col);

      h ^= 16u;

      h = fmix32(h);

      output_ptr[idx] = h;
    }
  });
}

}  // namespace

at::Tensor murmur_hash32_cpu(const at::Tensor& seed, const at::Tensor& positions, const at::Tensor& col_indices) {
  CHECK_INPUT(seed);
  CHECK_INPUT(positions);
  CHECK_INPUT(col_indices);

  CHECK_DIM(1, seed);
  CHECK_DIM(1, positions);
  CHECK_DIM(1, col_indices);
  TORCH_CHECK(seed.size(0) == positions.size(0), "seed and positions must have the same length");

  TORCH_CHECK(seed.scalar_type() == at::kUInt64, "seed must have dtype torch.uint64");

  TORCH_CHECK(
      positions.scalar_type() == at::kLong || positions.scalar_type() == at::kUInt64,
      "positions must have dtype torch.int64 or torch.uint64");

  TORCH_CHECK(col_indices.scalar_type() == at::kLong, "col_indices must have dtype torch.int64");

  const int64_t n = seed.size(0);
  const int64_t m = col_indices.size(0);

  auto output = at::empty({n, m}, seed.options().dtype(at::kUInt32));

  const uint64_t* seed_ptr = seed.data_ptr<uint64_t>();

  const int64_t* col_indices_ptr = col_indices.data_ptr<int64_t>();

  uint32_t* output_ptr = output.data_ptr<uint32_t>();

  if (n == 0 || m == 0) {
    return output;
  }

  AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::UInt64, positions.scalar_type(), "murmur_hash32_cpu", [&] {
    murmur_hash32_kernel_impl(seed_ptr, positions.data_ptr<scalar_t>(), col_indices_ptr, output_ptr, n, m);
  });

  return output;
}
