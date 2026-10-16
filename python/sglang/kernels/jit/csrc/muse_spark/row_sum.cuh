// Row sum over [N, K] fp32, bitwise-identical to `sglang.kernels.ops.muse_spark.triton_sum`.
//
// The point of that Triton kernel is a reduction order fixed by the compiler
// rather than by the CUDA runtime version, so this kernel reproduces tl.sum's
// pairwise tree (x0+x1, x2+x3, ... then pairs of pairs) exactly. One thread
// per row; K is padded to a power of two with zeros, as tl.sum does.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace muse_spark {

struct RowSumParams {
  const float* __restrict__ in;
  float* __restrict__ out;
  int num_rows;
  int num_cols;
  int64_t stride_n;
};

template <int kBlockK, int kThreads, bool kUsePDL>
__launch_bounds__(kThreads) __global__ void row_sum_kernel(const RowSumParams __grid_constant__ params) {
  using namespace device;

  PDLTriggerSecondary<kUsePDL>();

  const int row = blockIdx.x * kThreads + threadIdx.x;
  if (row >= params.num_rows) return;

  PDLWaitPrimary<kUsePDL>();

  const float* __restrict__ src = params.in + static_cast<int64_t>(row) * params.stride_n;
  float acc[kBlockK];
#pragma unroll
  for (int k = 0; k < kBlockK; ++k)
    acc[k] = (k < params.num_cols) ? src[k] : 0.0f;

  // tl.sum's butterfly: shfl.sync.bfly with masks kBlockK/2 ... 1, accumulating
  // `own + partner`.
#pragma unroll
  for (int span = kBlockK / 2; span >= 1; span /= 2)
#pragma unroll
    for (int k = 0; k < span; ++k)
      acc[k] += acc[k + span];

  params.out[row] = acc[0];
}

}  // namespace muse_spark

template <int kBlockK, int kThreads, bool kUsePDL>
void muse_spark_row_sum(
    tvm::ffi::TensorView input,  // [N, K] fp32
    tvm::ffi::TensorView out) {  // [N] fp32
  using namespace host;

  SymbolicSize N{"num_rows"};
  SymbolicSize K{"num_cols"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N, K}).with_dtype<fp32_t>().with_device(device_).verify(input);
  TensorMatcher({N}).with_dtype<fp32_t>().with_device(device_).verify(out);

  const int num_rows = static_cast<int>(N.unwrap());
  const int num_cols = static_cast<int>(K.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(num_cols <= kBlockK, "num_cols ", num_cols, " exceeds compiled kBlockK ", kBlockK);
  RuntimeCheck(input.stride(1) == 1, "input must be row-contiguous");
  if (num_rows == 0) return;

  const muse_spark::RowSumParams params{
      .in = static_cast<const float*>(input.data_ptr()),
      .out = static_cast<float*>(out.data_ptr()),
      .num_rows = num_rows,
      .num_cols = num_cols,
      .stride_n = input.stride(0),
  };

  const dim3 grid(static_cast<unsigned>((num_rows + kThreads - 1) / kThreads));
  LaunchKernel(grid, kThreads, device, 0)
      .enable_pdl(kUsePDL)(muse_spark::row_sum_kernel<kBlockK, kThreads, kUsePDL>, params);
}

}  // namespace sglang
