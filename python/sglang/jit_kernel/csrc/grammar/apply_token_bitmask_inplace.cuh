// Adapted from xgrammar:
// https://github.com/mlc-ai/xgrammar/blob/v0.1.18/python/xgrammar/kernels/apply_token_bitmask_inplace_cuda.cu
//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int32_t BITS_PER_BLOCK = 32;
constexpr int32_t THREADS_PER_THREAD_BLOCK = 256;

template <typename T>
__device__ T NegativeInfinity() {
  return static_cast<T>(-INFINITY);
}

template <>
__device__ fp16_t NegativeInfinity<fp16_t>() {
  return __float2half(-INFINITY);
}

template <>
__device__ bf16_t NegativeInfinity<bf16_t>() {
  return __float2bfloat16(-INFINITY);
}

template <typename T, typename PackedT>
__device__ PackedT PackedNegativeInfinity() {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  T packed[kAlignment];
#pragma unroll
  for (int i = 0; i < kAlignment; i++) {
    packed[i] = NegativeInfinity<T>();
  }
  return *reinterpret_cast<PackedT*>(packed);
}

template <typename T, typename PackedT, int kBitsPerThread>
__global__ void __launch_bounds__(THREADS_PER_THREAD_BLOCK) LogitsBitmaskKernel(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  constexpr uint32_t kPackedMask = (1 << kAlignment) - 1;

  const int batch_idx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

  const int block_offset = blockIdx.x * THREADS_PER_THREAD_BLOCK * kBitsPerThread;
  T* logits_gmem_ptr = logits + batch_idx * logits_stride + block_offset;
  const int32_t* bitmask_gmem_ptr =
      bitmask + batch_idx * bitmask_stride + block_offset / BITS_PER_BLOCK;
  const int bitmask_inner_idx = threadIdx.x % (BITS_PER_BLOCK / kAlignment);
  T logits_reg[kAlignment];

#pragma unroll
  for (int offset = threadIdx.x * kAlignment;
       offset < THREADS_PER_THREAD_BLOCK * kBitsPerThread;
       offset += THREADS_PER_THREAD_BLOCK * kAlignment) {
    if (block_offset + offset >= vocab_size) {
      break;
    }

    const uint32_t bitmask_val =
        (~bitmask_gmem_ptr[offset / BITS_PER_BLOCK] >> (bitmask_inner_idx * kAlignment)) &
        kPackedMask;

    if (bitmask_val == 0) {
      continue;
    }

    if (bitmask_val == kPackedMask) {
      *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) =
          PackedNegativeInfinity<T, PackedT>();
      continue;
    }

    *reinterpret_cast<PackedT*>(logits_reg) =
        *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset);
#pragma unroll
    for (int i = 0; i < kAlignment; i++) {
      if (((bitmask_val >> i) & 1)) {
        logits_reg[i] = NegativeInfinity<T>();
      }
    }
    *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) =
        *reinterpret_cast<PackedT*>(logits_reg);
  }
}

template <std::integral T, std::integral U>
constexpr auto CeilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T, typename PackedT>
void DispatchToBitsPerThread(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride,
    int32_t num_rows,
    DLDevice device) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  const int32_t num_blocks_per_row =
      CeilDiv(2048 / THREADS_PER_THREAD_BLOCK * 128, num_rows);
  const int32_t num_bits_per_thread =
      CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * num_blocks_per_row);

  const dim3 block(THREADS_PER_THREAD_BLOCK);

  if (num_bits_per_thread <= 4 && kAlignment <= 4) {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 4), num_rows);
    host::LaunchKernel(grid, block, device)(
        LogitsBitmaskKernel<T, PackedT, 4>,
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  } else if (num_bits_per_thread <= 8 && kAlignment <= 8) {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 8), num_rows);
    host::LaunchKernel(grid, block, device)(
        LogitsBitmaskKernel<T, PackedT, 8>,
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  } else if (num_bits_per_thread <= 16 && kAlignment <= 16) {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 16), num_rows);
    host::LaunchKernel(grid, block, device)(
        LogitsBitmaskKernel<T, PackedT, 16>,
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  } else {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 32), num_rows);
    host::LaunchKernel(grid, block, device)(
        LogitsBitmaskKernel<T, PackedT, 32>,
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  }
}

template <typename T>
void DispatchToPackedT(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride,
    int32_t num_rows,
    DLDevice device) {
  if (logits_stride % (sizeof(float4) / sizeof(T)) == 0) {
    DispatchToBitsPerThread<T, float4>(
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride,
        num_rows, device);
  } else {
    DispatchToBitsPerThread<T, T>(
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride,
        num_rows, device);
  }
}

template <typename T>
void apply_token_bitmask_inplace(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bitmask,
    tvm::ffi::TensorView indices,
    int64_t num_rows,
    int64_t use_indices) {
  using namespace host;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  const int ndim_logits = logits.dim();
  RuntimeCheck(ndim_logits == 1 || ndim_logits == 2, "logits must be 1D or 2D");

  const int32_t logits_rows =
      ndim_logits == 2 ? static_cast<int32_t>(logits.size(0)) : 1;
  const int32_t logits_cols =
      ndim_logits == 2 ? static_cast<int32_t>(logits.size(1))
                       : static_cast<int32_t>(logits.size(0));

  const int ndim_bitmask = bitmask.dim();
  RuntimeCheck(ndim_bitmask == 1 || ndim_bitmask == 2, "bitmask must be 1D or 2D");

  const int32_t bitmask_rows =
      ndim_bitmask == 2 ? static_cast<int32_t>(bitmask.size(0)) : 1;
  const int32_t bitmask_cols =
      ndim_bitmask == 2 ? static_cast<int32_t>(bitmask.size(1))
                        : static_cast<int32_t>(bitmask.size(0));

  device.verify(logits.device());
  device.verify(bitmask.device());
  RuntimeCheck(logits.is_contiguous(), "logits must be contiguous");
  RuntimeCheck(bitmask.is_contiguous(), "bitmask must be contiguous");

  const int32_t vocab_size =
      std::min(logits_cols, bitmask_cols * BITS_PER_BLOCK);
  const int32_t rows = static_cast<int32_t>(num_rows);
  const int32_t* indices_ptr =
      use_indices ? static_cast<const int32_t*>(indices.data_ptr()) : nullptr;

  DispatchToPackedT<T>(
      static_cast<T*>(logits.data_ptr()),
      static_cast<const int32_t*>(bitmask.data_ptr()),
      indices_ptr,
      vocab_size,
      logits_cols,
      bitmask_cols,
      rows,
      device.unwrap());
}

}  // namespace
