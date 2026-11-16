/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <sgl_kernel/bits.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "fast_hadamard_transform_special.h"
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace sglang {

using ::HadamardParamsBase;

template <int kNThreads_, int kLogN_, typename input_t_>
struct FastHadamardKernelTraits {
  using input_t = input_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kLogN = kLogN_;
  static constexpr int N = 1 << kLogN;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  static constexpr int kNChunks = N / (kNElts * kNThreads);
  static constexpr int kSmemExchangeSize = (N * 4) < (32 * 1024) ? (N * 4) : (32 * 1024);
  static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
  static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
  static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNThreads_, int kLogN_, int kMultiple, int kMaxDim, int kMaxSmem, typename input_t_>
struct FastHadamardMNKernelTraits {
  using input_t = input_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kLogN = kLogN_;
  static constexpr int N = (1 << kLogN) * kMultiple;
  static_assert(N <= kMaxDim);
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = 4;
  static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  static constexpr int kNChunks = N / (kNElts * kNThreads);
  static_assert(kNChunks == kMultiple);
  static constexpr int kSmemExchangeSize = (N * 4) < kMaxSmem ? (N * 4) : kMaxSmem;
  static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
  static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
  static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_12(float x[kNChunks][12]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_12(x[c]);
  }
}

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_20(float x[kNChunks][20]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_20(x[c]);
  }
}

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_28(float x[kNChunks][28]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_28(x[c]);
  }
}

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_40(float x[kNChunks][40]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_40(x[c]);
  }
}

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void fast_hadamard_transform_kernel(HadamardParamsBase params) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(Ktraits::kNElts);
  static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");

  constexpr int kWarpSize = kNThreads < 32 ? kNThreads : 32;
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");

  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

  constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
  constexpr int kNExchanges = kNChunks / kChunksPerExchange;
  static_assert(kNExchanges * kChunksPerExchange == kNChunks);

  extern __shared__ char smem_[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_);

  const int batch_id = static_cast<int>(blockIdx.x);
  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride;

  float x_vals[kNChunks][kNElts];
  load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

  hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
    hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals_transposed[i][c] = x_vals[c][i];
      }
    }

    if constexpr (kNChunks == 12) {
      hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 20) {
      hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 28) {
      hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 40) {
      hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
    } else {
      constexpr int kLogNChunks = cilog2(kNChunks);
      static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
      hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[c][i] = x_vals_transposed[i][c];
      }
    }
  }

  store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale);
}

// Upstream's launch shapes as compile-time tables.
// Threads per row of the power-of-two sizes, indexed by kLogN (3..15).
inline constexpr int kPow2NumThreads[] = {0, 0, 0, 1, 2, 4, 8, 16, 32, 32, 128, 256, 256, 256, 256, 256};
// (multiple, max dim, max exchange smem bytes) of the multiple x 2^kLogN sizes.
inline constexpr std::tuple<int, int, int> kMNConfigs[] = {
    {12, 12 * 1024, 24 * 1024},
    {20, 20 * 1024, 40 * 1024},
    {28, 28 * 1024, 28 * 1024},
    {40, 40 * 1024, 40 * 1024},
};

constexpr auto find_mn_config(int multiple) -> std::tuple<int, int, int> {
  for (const auto& config : kMNConfigs) {
    if (std::get<0>(config) == multiple) return config;
  }
  return {0, 0, 0};
}

// One module per (multiple, log_N, dtype), chosen in kernels/ops/quantization/hadamard.py:
// a row of dim elements is transformed as kMultiple x 2^kLogN, dim <= kMultiple << kLogN.
template <int kMultiple, int kLogN, typename DType>
struct HadamardKernel {
  static constexpr bool kPow2 = kMultiple == 1;
  static constexpr auto kMN = find_mn_config(kMultiple);
  static_assert(kPow2 || std::get<0>(kMN) == kMultiple, "hadamard: unsupported multiple");
  static_assert(
      kPow2 ? (3 <= kLogN && kLogN <= 15) : (2 <= kLogN && kLogN <= 10),
      "hadamard: dim not supported, at most 2^15 (power of two) or multiple x 2^10");

  static constexpr int kNumThreads = kPow2 ? kPow2NumThreads[kLogN] : 1 << (kLogN - 2);
  using Ktraits = std::conditional_t<
      kPow2,
      FastHadamardKernelTraits<kNumThreads, kLogN, DType>,
      FastHadamardMNKernelTraits<kNumThreads, kLogN, kMultiple, std::get<1>(kMN), std::get<2>(kMN), DType>>;
  static constexpr auto kernel = fast_hadamard_transform_kernel<Ktraits>;
  // Row granularity; the Python side pads to it (pad_multiple in hadamard.py).
  static constexpr int64_t kDimAlign = kPow2 ? 8 : 4 * kMultiple;

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    using namespace host;

    auto N = SymbolicSize{"batch"};
    auto D = SymbolicSize{"dim"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({N, D})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(x);
    TensorMatcher({N, D})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(out);

    const auto batch = N.unwrap();
    const auto dim = D.unwrap();
    CHECK_HOST(dim % kDimAlign == 0 && log2_ceil(static_cast<uint64_t>(dim / kMultiple)) == kLogN)
        << "hadamard: dim " << dim << " does not fit the " << kMultiple << " x 2^" << kLogN << " module";

    if constexpr (Ktraits::kSmemSize >= 48 * 1024) {
      runtime::set_max_dynamic_smem_per_device<kernel>(device.unwrap(), Ktraits::kSmemSize);
    }
    const auto params = HadamardParamsBase{
        .batch = static_cast<int>(batch),
        .dim = static_cast<int>(dim),
        .log_N = kLogN,
        .x_batch_stride = x.stride(0),
        .out_batch_stride = out.stride(0),
        .scale = scale,
        .x_ptr = const_cast<void*>(x.data_ptr()),
        .out_ptr = const_cast<void*>(out.data_ptr()),
    };
    LaunchKernel(static_cast<uint32_t>(batch), kNumThreads, device.unwrap(), Ktraits::kSmemSize)(kernel, params);
  }
};

}  // namespace sglang
