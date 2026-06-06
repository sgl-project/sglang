#include <cmath>
#include <cstdint>
#include <limits>

#include "common.h"

namespace {

constexpr int32_t BITS_PER_BLOCK = 32;

template <typename scalar_t>
void apply_token_bitmask_inplace_impl(
    scalar_t* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    int64_t batch_size,
    int64_t vocab_size,
    int64_t logits_stride,
    int64_t bitmask_stride) {
  const scalar_t neg_inf = -std::numeric_limits<float>::infinity();

  at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      scalar_t* __restrict__ logits_row = logits + b * logits_stride;
      const int32_t* __restrict__ bitmask_row = bitmask + b * bitmask_stride;

      // Process 32 tokens at a time (one bitmask word)
      int64_t num_words = (vocab_size + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
      for (int64_t w = 0; w < num_words; ++w) {
        int32_t mask_word = bitmask_row[w];

        // Fast path: if all bits are set, skip this block
        if (mask_word == -1) {  // all bits 1 in two's complement
          continue;
        }

        // Fast path: if no bits are set, fill entire block with -inf
        int64_t base = w * BITS_PER_BLOCK;
        int64_t block_end = std::min(base + BITS_PER_BLOCK, vocab_size);

        if (mask_word == 0) {
          for (int64_t t = base; t < block_end; ++t) {
            logits_row[t] = neg_inf;
          }
          continue;
        }

        // General case: check each bit
#pragma GCC unroll 8
        for (int64_t t = base; t < block_end; ++t) {
          int32_t bit_idx = static_cast<int32_t>(t - base);
          if (!((mask_word >> bit_idx) & 1)) {
            logits_row[t] = neg_inf;
          }
        }
      }
    }
  });
}

}  // anonymous namespace

// logits  : {batch_size, vocab_size}, in-place modified
// bitmask : {batch_size, ceil(vocab_size / 32)}, int32
void apply_token_bitmask_inplace_cpu(at::Tensor& logits, at::Tensor& bitmask) {
  CHECK_DIM(2, logits);
  CHECK_DIM(2, bitmask);
  CHECK_INPUT(logits);
  CHECK_INPUT(bitmask);
  CHECK_EQ(bitmask.scalar_type(), at::kInt);

  int64_t batch_size = logits.size(0);
  int64_t vocab_size = logits.size(1);
  int64_t logits_stride = logits.stride(0);
  int64_t bitmask_stride = bitmask.stride(0);

  CPU_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "apply_token_bitmask_inplace_cpu", [&] {
    apply_token_bitmask_inplace_impl<scalar_t>(
        logits.data_ptr<scalar_t>(),
        bitmask.data_ptr<int32_t>(),
        batch_size,
        vocab_size,
        logits_stride,
        bitmask_stride);
  });
}
