#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "common.h"
#include "vec.h"

namespace {

// NaN counts as the maximum and ties take the lowest index, same as torch.argmax.
void argmax_float_kernel_impl(
    int64_t* __restrict__ out, const float* __restrict__ x, int64_t rows, int64_t cols, int64_t stride) {
  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = Vec::size();
  constexpr int64_t kChunk = 4096;
  const int64_t num_chunks = div_up(cols, kChunk);

  std::vector<float> chunk_max(rows * num_chunks);

  at::parallel_for(0, rows * num_chunks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const int64_t row = t / num_chunks;
      const int64_t c = t % num_chunks;
      const float* p = x + row * stride + c * kChunk;
      const int64_t n = std::min(kChunk, cols - c * kChunk);

      Vec vmax(-std::numeric_limits<float>::infinity());
      int64_t i = 0;
      for (; i <= n - kVecSize; i += kVecSize) {
        vmax = at::vec::maximum(vmax, Vec::loadu(p + i));
      }
      float m = at::vec::vec_reduce_all([](Vec& a, Vec& b) { return at::vec::maximum(a, b); }, vmax);
      for (; i < n; ++i) {
        if (!std::isnan(m) && (std::isnan(p[i]) || p[i] > m)) {
          m = p[i];
        }
      }
      chunk_max[t] = m;
    }
  });

  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      int64_t best_chunk = 0;
      float best = chunk_max[row * num_chunks];
      for (int64_t c = 1; c < num_chunks && !std::isnan(best); ++c) {
        const float v = chunk_max[row * num_chunks + c];
        if (std::isnan(v) || v > best) {
          best = v;
          best_chunk = c;
        }
      }
      const float* p = x + row * stride + best_chunk * kChunk;
      const int64_t n = std::min(kChunk, cols - best_chunk * kChunk);
      int64_t idx = 0;
      while (idx < n && (std::isnan(best) ? !std::isnan(p[idx]) : p[idx] != best)) {
        ++idx;
      }
      out[row] = best_chunk * kChunk + idx;
    }
  });
}

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

// logits: [rows, vocab] float32, last dim contiguous
at::Tensor argmax_cpu(const at::Tensor& logits) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(logits);
  CHECK_DIM(2, logits);
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "argmax_cpu: expect float32 logits, got ", logits.scalar_type());
  const int64_t rows = logits.size(0);
  const int64_t cols = logits.size(1);
  TORCH_CHECK(cols > 0, "argmax_cpu: expect non-empty rows");

  auto out = at::empty({rows}, logits.options().dtype(at::kLong));
  if (rows > 0) {
    argmax_float_kernel_impl(out.data_ptr<int64_t>(), logits.data_ptr<float>(), rows, cols, logits.stride(0));
  }
  return out;
}

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
