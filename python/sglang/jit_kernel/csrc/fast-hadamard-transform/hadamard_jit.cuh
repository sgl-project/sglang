/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "fast_hadamard_transform_special.h"
#include "static_switch.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

using ::bf16_t;
using ::fp16_t;
using ::HadamardParamsBase;

constexpr inline int ceil_log2(int val) {
  int log = 0;
  int p = 1;
  while (p < val) {
    p <<= 1;
    ++log;
  }
  return log;
}

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

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard12NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 12, 12 * 1024, 24 * 1024, input_t_>;

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard20NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 20, 20 * 1024, 40 * 1024, input_t_>;

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard28NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 28, 28 * 1024, 28 * 1024, input_t_>;

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard40NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 40, 40 * 1024, 40 * 1024, input_t_>;

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

template <typename Ktraits>
inline void set_max_dynamic_smem() {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  if constexpr (kSmemSize >= 48 * 1024) {
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
  }
}

template <typename Ktraits>
inline void launch_kernel(HadamardParamsBase& params, DLDevice device) {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  set_max_dynamic_smem<Ktraits>();
  auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
  host::LaunchKernel(dim3(params.batch), dim3(Ktraits::kNThreads), device, kSmemSize)(kernel, params);
  host::RuntimeDeviceCheck();
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamardKernelTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 3) {
    fast_hadamard_transform_launch<1, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_launch<2, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_launch<4, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_launch<8, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_launch<16, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_launch<32, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_launch<32, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_launch<128, 10, input_t>(params, device);
  } else if (params.log_N == 11) {
    fast_hadamard_transform_launch<256, 11, input_t>(params, device);
  } else if (params.log_N == 12) {
    fast_hadamard_transform_launch<256, 12, input_t>(params, device);
  } else if (params.log_N == 13) {
    fast_hadamard_transform_launch<256, 13, input_t>(params, device);
  } else if (params.log_N == 14) {
    fast_hadamard_transform_launch<256, 14, input_t>(params, device);
  } else if (params.log_N == 15) {
    fast_hadamard_transform_launch<256, 15, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_12N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard12NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_12N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_12N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_12N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_12N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_12N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_12N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_12N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_12N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_12N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_12N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_12N: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_20N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard20NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_20N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_20N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_20N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_20N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_20N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_20N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_20N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_20N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_20N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_20N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_20N: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_28N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard28NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_28N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_28N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_28N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_28N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_28N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_28N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_28N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_28N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_28N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_28N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_28N: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_40N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard40NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_40N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_40N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_40N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_40N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_40N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_40N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_40N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_40N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_40N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_40N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_40N: unsupported log_N=", params.log_N);
  }
}

inline void set_hadamard_params(
    HadamardParamsBase& params,
    int64_t batch,
    int64_t dim,
    int64_t multiple,
    const tvm::ffi::TensorView x,
    const tvm::ffi::TensorView out,
    float scale) {
  std::memset(&params, 0, sizeof(params));
  params.batch = static_cast<int>(batch);
  params.dim = static_cast<int>(dim);
  params.log_N = ceil_log2(static_cast<int>(dim / multiple));
  params.x_ptr = const_cast<void*>(x.data_ptr());
  params.out_ptr = const_cast<void*>(out.data_ptr());
  params.x_batch_stride = x.stride(0);
  params.out_batch_stride = out.stride(0);
  params.scale = scale;
}

template <int kMultiple, typename DType>
inline void run_hadamard(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
  using namespace host;

  auto N = SymbolicSize{"batch"};
  auto D = SymbolicSize{"dim"};
  auto SX = SymbolicSize{"x_batch_stride"};
  auto SO = SymbolicSize{"out_batch_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({N, D}).with_strides({SX, 1}).with_dtype<DType>().with_device(device).verify(x);
  TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device).verify(out);

  const int64_t batch = N.unwrap();
  const int64_t dim = D.unwrap();

  RuntimeCheck(dim % kMultiple == 0, "hadamard: dim must be divisible by ", kMultiple);

  HadamardParamsBase params;
  set_hadamard_params(params, batch, dim, kMultiple, x, out, scale);

  if constexpr (kMultiple == 1) {
    RuntimeCheck(dim % 8 == 0, "fast_hadamard_transform only supports hidden dim divisible by 8");
    RuntimeCheck(dim <= 32768, "fast_hadamard_transform only supports hidden dim <= 32768");
    fast_hadamard_transform_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 12) {
    RuntimeCheck(dim % (4 * 12) == 0, "fast_hadamard_transform_12N only supports hidden dim divisible by 48");
    RuntimeCheck(dim <= 12 * 1024, "fast_hadamard_transform_12N only supports hidden dim <= 12288");
    fast_hadamard_transform_12N_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 20) {
    RuntimeCheck(dim % (4 * 20) == 0, "fast_hadamard_transform_20N only supports hidden dim divisible by 80");
    RuntimeCheck(dim <= 20 * 1024, "fast_hadamard_transform_20N only supports hidden dim <= 20480");
    fast_hadamard_transform_20N_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 28) {
    RuntimeCheck(dim % (4 * 28) == 0, "fast_hadamard_transform_28N only supports hidden dim divisible by 112");
    RuntimeCheck(dim <= 28 * 1024, "fast_hadamard_transform_28N only supports hidden dim <= 28672");
    fast_hadamard_transform_28N_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 40) {
    RuntimeCheck(dim % (4 * 40) == 0, "fast_hadamard_transform_40N only supports hidden dim divisible by 160");
    RuntimeCheck(dim <= 40 * 1024, "fast_hadamard_transform_40N only supports hidden dim <= 40960");
    fast_hadamard_transform_40N_cuda<DType>(params, device.unwrap());
  } else {
    Panic("Unsupported multiple");
  }
}

template <typename DType>
struct HadamardKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<1, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard12NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<12, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard20NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<20, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard28NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<28, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard40NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<40, DType>(x, out, scale);
  }
};

}  // namespace
