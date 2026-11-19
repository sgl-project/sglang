#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/// \brief An int32 or int64 tensor. The width is a runtime field: callers mix
/// widths per tensor, and this kernel is too small to pay for one instantiation
/// per combination.
struct DFlashIntTensor {
  void* ptr;
  int64_t row_stride;
  bool is_int64;
};

struct DFlashSimulatedAcceptParams {
  DFlashIntTensor candidates;
  int64_t candidates_col_stride;
  DFlashIntTensor target_bonus_tokens;  // ptr is null in fixed mode
  DFlashIntTensor num_correct_drafts;
  DFlashIntTensor num_accept_tokens;
  DFlashIntTensor bonus_tokens;
  DFlashIntTensor out_tokens;
  int64_t out_tokens_col_stride;
  DFlashIntTensor prefix_lens;  // both ptrs are null when new_seq_lens is not written
  DFlashIntTensor new_seq_lens;
  uint32_t num_elements;
  uint32_t block_size;
  int64_t forced_num_accept_tokens;
  int64_t fixed_token;
};

SGL_DEVICE int64_t dflash_load(const DFlashIntTensor& t, int64_t offset) {
  return t.is_int64 ? static_cast<const int64_t*>(t.ptr)[offset] : static_cast<const int32_t*>(t.ptr)[offset];
}

SGL_DEVICE void dflash_store(const DFlashIntTensor& t, int64_t offset, int64_t value) {
  if (t.is_int64) {
    static_cast<int64_t*>(t.ptr)[offset] = value;
  } else {
    static_cast<int32_t*>(t.ptr)[offset] = static_cast<int32_t>(value);
  }
}

/// \brief The value `t` holds after storing `value`, as a torch copy into its dtype wraps it.
SGL_DEVICE int64_t dflash_wrap(const DFlashIntTensor& t, int64_t value) {
  return t.is_int64 ? value : static_cast<int64_t>(static_cast<int32_t>(value));
}

/// \brief One thread per `out_tokens` element; column 0 also writes its row's lengths and bonus.
__global__ void dflash_simulated_accept_kernel(const __grid_constant__ DFlashSimulatedAcceptParams p) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= p.num_elements) return;
  const int64_t row = idx / p.block_size;
  const int64_t col = idx % p.block_size;
  const int64_t n = p.forced_num_accept_tokens;
  const bool real_draft_tokens = p.target_bonus_tokens.ptr != nullptr;

  // Two casts, as in the eager path: target -> bonus dtype -> out dtype.
  const int64_t bonus =
      real_draft_tokens
          ? dflash_wrap(p.bonus_tokens, dflash_load(p.target_bonus_tokens, row * p.target_bonus_tokens.row_stride))
          : p.fixed_token;
  int64_t token = 0;
  if (!real_draft_tokens || col == n - 1) {
    token = bonus;
  } else if (col < n - 1) {
    token = dflash_load(p.candidates, row * p.candidates.row_stride + (col + 1) * p.candidates_col_stride);
  }
  dflash_store(p.out_tokens, row * p.out_tokens.row_stride + col * p.out_tokens_col_stride, token);

  if (col == 0) {
    dflash_store(p.num_correct_drafts, row * p.num_correct_drafts.row_stride, n - 1);
    dflash_store(p.num_accept_tokens, row * p.num_accept_tokens.row_stride, n);
    dflash_store(p.bonus_tokens, row * p.bonus_tokens.row_stride, bonus);
    if (p.new_seq_lens.ptr != nullptr) {
      const int64_t prefix_len = dflash_load(p.prefix_lens, row * p.prefix_lens.row_stride);
      dflash_store(p.new_seq_lens, row * p.new_seq_lens.row_stride, dflash_wrap(p.prefix_lens, prefix_len + n));
    }
  }
}

/**
 * \brief Write a forced DFlash acceptance of `n = forced_num_accept_tokens`
 *        (bonus included) for every request in one launch.
 *
 * Every tensor is int32 or int64, on one GPU, with any strides.
 *
 * \param candidates          `[bs, block_size]` verify tokens; the drafts are columns 1..n-1.
 * \param target_bonus_tokens `[bs]` target token at the forced row; absent in fixed mode.
 * \param num_correct_drafts  `[bs]` output, set to `n - 1`.
 * \param num_accept_tokens   `[bs]` output, set to `n`.
 * \param bonus_tokens        `[bs]` output: the target token, or `fixed_token` in fixed mode.
 * \param out_tokens          `[bs, block_size]` output: drafts, the bonus at column `n - 1`, then zeros;
 *                            every column is `fixed_token` in fixed mode.
 * \param prefix_lens         `[bs]`; read only when `new_seq_lens` is given.
 * \param new_seq_lens        `[bs]` output `prefix_lens + n`; optional.
 * \param forced_num_accept_tokens `n`, in `[1, block_size]`.
 * \param fixed_token         Token written in fixed mode.
 */
inline void dflash_simulated_accept(
    tvm::ffi::TensorView candidates,
    tvm::ffi::Optional<tvm::ffi::TensorView> target_bonus_tokens,
    tvm::ffi::TensorView num_correct_drafts,
    tvm::ffi::TensorView num_accept_tokens,
    tvm::ffi::TensorView bonus_tokens,
    tvm::ffi::TensorView out_tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> prefix_lens,
    tvm::ffi::Optional<tvm::ffi::TensorView> new_seq_lens,
    int64_t forced_num_accept_tokens,
    int64_t fixed_token) {
  using namespace host;
  SymbolicSize batch_size{"batch_size"}, block_size{"block_size"};
  SymbolicDevice device;
  device.set_options<kDLGPU>();

  auto matrix = [&](tvm::ffi::TensorView t, int64_t& col_stride) {
    SymbolicSize row_stride_sym{"row_stride"}, col_stride_sym{"col_stride"};
    TensorMatcher({batch_size, block_size})
        .with_strides({row_stride_sym, col_stride_sym})
        .with_dtype<int32_t, int64_t>()
        .with_device(device)
        .verify(t);
    col_stride = col_stride_sym.unwrap();
    return DFlashIntTensor{t.data_ptr(), row_stride_sym.unwrap(), t.dtype().bits == 64};
  };
  auto vector = [&](tvm::ffi::TensorView t) {
    SymbolicSize row_stride_sym{"row_stride"};
    TensorMatcher({batch_size})
        .with_strides({row_stride_sym})
        .with_dtype<int32_t, int64_t>()
        .with_device(device)
        .verify(t);
    return DFlashIntTensor{t.data_ptr(), row_stride_sym.unwrap(), t.dtype().bits == 64};
  };
  constexpr auto kAbsent = DFlashIntTensor{nullptr, 0, false};

  DFlashSimulatedAcceptParams p{};
  p.candidates = matrix(candidates, p.candidates_col_stride);
  p.out_tokens = matrix(out_tokens, p.out_tokens_col_stride);
  p.target_bonus_tokens = target_bonus_tokens.has_value() ? vector(target_bonus_tokens.value()) : kAbsent;
  p.num_correct_drafts = vector(num_correct_drafts);
  p.num_accept_tokens = vector(num_accept_tokens);
  p.bonus_tokens = vector(bonus_tokens);
  if (new_seq_lens.has_value()) {
    CHECK_HOST(prefix_lens.has_value()) << "dflash_simulated_accept: new_seq_lens needs prefix_lens";
    p.prefix_lens = vector(prefix_lens.value());
    p.new_seq_lens = vector(new_seq_lens.value());
  } else {
    p.prefix_lens = p.new_seq_lens = kAbsent;
  }

  const int64_t bs = batch_size.unwrap();
  const int64_t block = block_size.unwrap();
  CHECK_HOST(1 <= forced_num_accept_tokens && forced_num_accept_tokens <= block)
      << "dflash_simulated_accept: forced_num_accept_tokens " << forced_num_accept_tokens << " not in [1, " << block
      << "]";
  CHECK_HOST(bs * block <= INT32_MAX) << "dflash_simulated_accept: too many elements";
  if (bs == 0) return;
  p.num_elements = static_cast<uint32_t>(bs * block);
  p.block_size = static_cast<uint32_t>(block);
  p.forced_num_accept_tokens = forced_num_accept_tokens;
  p.fixed_token = fixed_token;

  // Arbitrary; the kernel is launch-bound at every realistic size.
  constexpr uint32_t kBlockSize = 128;
  LaunchKernel(div_ceil(p.num_elements, kBlockSize), kBlockSize, device.unwrap())(dflash_simulated_accept_kernel, p);
}

}  // namespace sglang
