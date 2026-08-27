/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 Rong Shuo.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// mext_r1 (ratio 1.0 M-extension) NVFP4 activation quantization.
//
// Ported from ResInfer csrc/nvfp4/src/fp4/nvfp4_quant_kernels.cu (the mext_r1
// subset), rehosted on the sglang JIT tvm-ffi conventions. The k_ext masked
// quantization lives in a separate header (added with the k_ext linear path).
//
// Contract (mirrors the reference torch op):
//   input     [M, K] fp16/bf16, K % 16 == 0
//   input_sf  float32, 1 element (single global scale reciprocal)
//   output    uint8, >= 2 * output_M * K / 2 bytes (row-doubled packed FP4)
//   output_sf uint8 (fp8-e4m3 bytes), swizzled 128x4 tiled layout sized for
//             (2 * output_M, K) -- or (output_M, 2K) for concat_k
//   output_M  == M; derived from input.shape inside the op, never passed (a
//             scalar op-arg desyncs from the buffer under torch.compile
//             specialization)
//
// layout_mode: 0=concat, 1=row_pair, 3=concat_k
// elts_mode:   0=auto (pack16 on cc 10.x, else pack8), 8, 16

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include "residue_nvfp4_utils.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {
namespace residue_nvfp4 {

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, RESIDUE_NVFP4_BLOCKS_PER_SM(512)) cvt_fp16_to_fp4_mext_r1_pack8(
    int32_t inputRows,
    int32_t outputRows,
    int32_t numCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    int32_t layoutMode) {
  using PackedVec8 = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec8) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int colGroups = numCols / CVT_FP4_ELTS_PER_THREAD;
  if (colIdx >= colGroups) return;

  for (int rowIdx = blockIdx.x; rowIdx < outputRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < inputRows;
    int32_t baseRow = compute_mext_r1_base_row(rowIdx, outputRows, layoutMode);
    int32_t residueRow = compute_mext_r1_residue_row(rowIdx, outputRows, layoutMode);

    PackedVec8 in_vec{};
    if (validRow) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      in_vec = reinterpret_cast<PackedVec8 const*>(in)[inOffset];
    }

    // kConcatK (mode 3): one output row per token, [M, 2K] geometry. Base
    // lands at (row, col), residue at (row, col + K); both data and SF use
    // the canonical addressing of the (M, 2K) grid, so the doubled-K operand
    // is indistinguishable from a stock quantization of a 2K-wide input.
    int32_t colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
    bool kConcatK = layoutMode == 3;
    int64_t outOffset = kConcatK ? (int64_t)rowIdx * (2 * colsPerRow) + colIdx : (int64_t)baseRow * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    auto sf_out = kConcatK ? cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
                                 rowIdx, colIdx, 2 * numCols, SFout)
                           : cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
                                 baseRow, colIdx, numCols, SFout);

    int64_t residueOutOffset =
        kConcatK ? (int64_t)rowIdx * (2 * colsPerRow) + colsPerRow + colIdx : (int64_t)residueRow * colsPerRow + colIdx;
    uint32_t* residue_data_out = out + residueOutOffset;

    auto residue_sf_out = kConcatK ? cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
                                         rowIdx, colIdx + colsPerRow, 2 * numCols, SFout)
                                   : cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
                                         residueRow, colIdx, numCols, SFout);

    out_pos =
        cvt_warp_fp16_to_fp4_mext_r1_fast<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out, residue_data_out, residue_sf_out);
  }
}

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, RESIDUE_NVFP4_BLOCKS_PER_SM(512)) cvt_fp16_to_fp4_mext_r1_pack16(
    int32_t inputRows,
    int32_t outputRows,
    int32_t numCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    int32_t layoutMode) {
  using PackedVec8 = PackedVec<Type>;
  constexpr int CVT_FP4_NUM_THREADS_PER_SF = 1;

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int colGroups = numCols / 16;
  if (colIdx >= colGroups) return;

  for (int rowIdx = blockIdx.x; rowIdx < outputRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < inputRows;
    int32_t baseRow = compute_mext_r1_base_row(rowIdx, outputRows, layoutMode);
    int32_t residueRow = compute_mext_r1_residue_row(rowIdx, outputRows, layoutMode);

    PackedVec8 lo_vec{};
    PackedVec8 hi_vec{};
    if (validRow) {
      // One 256-bit load covers the thread's entire 16-element SF vector.
      load_packed_vec16(in + static_cast<int64_t>(rowIdx) * numCols + colIdx * 16, lo_vec, hi_vec);
    }

    int64_t outOffset = baseRow * (numCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(baseRow, colIdx, numCols, SFout);

    int64_t residueOutOffset = residueRow * (numCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
    uint32_t* residue_data_out = out + residueOutOffset;

    auto residue_sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(residueRow, colIdx, numCols, SFout);

    u32x2 packed = cvt_warp_fp16_to_fp4_mext_r1_fast16<Type, UE8M0_SF>(
        lo_vec, hi_vec, SFScaleVal, sf_out, residue_data_out, residue_sf_out);
    // Single 64-bit store; outOffset is even because numCols % 16 == 0.
    *reinterpret_cast<uint2*>(out + outOffset) = make_uint2(packed.lo, packed.hi);
  }
}

enum class MExtR1EltsMode : int {
  kAuto = 0,
  kElts8 = 8,
  kElts16 = 16,
};

enum class MExtR1LayoutMode : int {
  kConcat = 0,
  kRowPair = 1,
  // K-concat: ONE output row per token, [M, 2K] geometry -- base in element
  // cols [0, K), residue in [K, 2K).
  kConcatK = 3,
};

inline bool use_pack16_for_mode(int elts_mode, int cc_major) {
  switch (static_cast<MExtR1EltsMode>(elts_mode)) {
    case MExtR1EltsMode::kElts8:
      return false;
    case MExtR1EltsMode::kElts16:
      return true;
    case MExtR1EltsMode::kAuto:
    default:
      // B200-measured: pack16 >= pack8 at every shape except M=1/K=4096
      // (launch-floor regime) and 1.36-1.49x faster at prefill sizes. SM120
      // decode measured pack8 faster, so only datacenter Blackwell (cc 10.x)
      // defaults to 16.
      return cc_major == 10;
  }
}

inline int normalize_mext_r1_layout_mode(int layout_mode) {
  switch (static_cast<MExtR1LayoutMode>(layout_mode)) {
    case MExtR1LayoutMode::kRowPair:
      return static_cast<int>(MExtR1LayoutMode::kRowPair);
    case MExtR1LayoutMode::kConcatK:
      return static_cast<int>(MExtR1LayoutMode::kConcatK);
    case MExtR1LayoutMode::kConcat:
    default:
      return static_cast<int>(MExtR1LayoutMode::kConcat);
  }
}

template <typename T>
void invokeFP4QuantizationMExtR1(
    int input_m,
    int output_m,
    int n,
    T const* input,
    float const* SFScale,
    uint32_t* output,
    uint32_t* SFOuput,
    bool useUE8M0,
    int elts_mode,
    int layout_mode,
    int multiProcessorCount,
    int ccMajor,
    cudaStream_t stream) {
  bool pack16 = use_pack16_for_mode(elts_mode, ccMajor);
  int normalized_layout_mode = normalize_mext_r1_layout_mode(layout_mode);
  // kConcatK is implemented in the pack8 kernel only; pack8 is correct on
  // every arch (pack16 is a cc10.x throughput default, not a semantic).
  if (normalized_layout_mode == static_cast<int>(MExtR1LayoutMode::kConcatK)) {
    pack16 = false;
  }
  int colGroups = n / (pack16 ? 16 : CVT_FP4_ELTS_PER_THREAD);
  colGroups = std::max(colGroups, 1);

  // Decode M-extension is intentionally small-M. Split the K dimension into
  // multiple CTAs so M=1 does not collapse to one row CTA.
  int desiredGridY = std::max(1, (computeEffectiveRows(output_m) + std::max(output_m, 1) - 1) / std::max(output_m, 1));
  desiredGridY = std::min(desiredGridY, colGroups);
  int numThreads = (colGroups + desiredGridY - 1) / desiredGridY;
  numThreads = std::max(round_up_int(numThreads, 32), 32);
  dim3 block(std::min(numThreads, 512));
  int const numBlocksPerSM = runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(
      std::min(int(output_m), multiProcessorCount * numBlocksPerSM),
      (colGroups + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));

  auto launch = [&](auto kernel) {
    kernel<<<grid, block, 0, stream>>>(input_m, output_m, n, input, SFScale, output, SFOuput, normalized_layout_mode);
    cudaError_t status = cudaGetLastError();
    host::RuntimeCheck(
        status == cudaSuccess, "residue nvfp4 mext_r1 quant kernel launch failed: ", cudaGetErrorString(status));
  };

  if (useUE8M0) {
    if (pack16) {
      launch(cvt_fp16_to_fp4_mext_r1_pack16<T, true>);
    } else {
      launch(cvt_fp16_to_fp4_mext_r1_pack8<T, true>);
    }
  } else {
    if (pack16) {
      launch(cvt_fp16_to_fp4_mext_r1_pack16<T, false>);
    } else {
      launch(cvt_fp16_to_fp4_mext_r1_pack8<T, false>);
    }
  }
}

}  // namespace residue_nvfp4

template <typename DType>
void scaled_fp4_quant_mext_r1(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView output_sf,
    tvm::ffi::TensorView input_sf,
    int64_t elts_mode,
    int64_t layout_mode) {
  using namespace host;
  namespace rn = residue_nvfp4;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto M = SymbolicSize{"m"};
  auto K = SymbolicSize{"k"};

  TensorMatcher({M, K})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);

  const auto m = static_cast<int32_t>(M.unwrap());
  const auto k = static_cast<int32_t>(K.unwrap());
  RuntimeCheck(k % 16 == 0, "residue nvfp4 mext_r1: K must be a multiple of 16, got ", k);

  // output_m is derived HERE from the input row count -- it is NOT an op
  // argument. As a scalar op-arg,
  // torch.compile specialized it to a constant while the caller's output
  // buffer stayed symbolic in M, desyncing the two under CUDA-graph capture.
  const int64_t output_m = static_cast<int64_t>(m);

  auto OutLen = SymbolicSize{"out_len"};
  auto SfLen = SymbolicSize{"sf_len"};
  TensorMatcher({OutLen}).with_dtype<uint8_t>().with_device(device).verify(output);
  TensorMatcher({SfLen}).with_dtype<uint8_t>().with_device(device).verify(output_sf);
  RuntimeCheck(
      output.numel() >= 2 * output_m * k / 2,
      "residue nvfp4 mext_r1: output tensor too small for the derived output_m");

  // Swizzled SF layout size: 128-row x 4-SF tiles over the OUTPUT geometry
  // ((2*output_m, k) row-doubled, or (output_m, 2k) for concat_k -- both need
  // the same byte count).
  const int64_t num_m_tiles = (2 * output_m + 127) / 128;
  const int64_t num_k_tiles = (k + 63) / 64;
  RuntimeCheck(
      output_sf.numel() >= num_m_tiles * num_k_tiles * 512,
      "residue nvfp4 mext_r1: output_sf tensor too small for the swizzled layout");

  auto InSfLen = SymbolicSize{"in_sf_len"};
  TensorMatcher({InSfLen}).with_dtype<float>().with_device(device).verify(input_sf);
  RuntimeCheck(input_sf.numel() == 1, "residue nvfp4 mext_r1: input_sf must contain exactly 1 element");

  const auto dl_device = device.unwrap();
  const auto info = rn::query_device_info(dl_device.device_id);
  const auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dl_device.device_type, dl_device.device_id));

  constexpr bool useUE8M0 = false;

  rn::invokeFP4QuantizationMExtR1(
      m,
      static_cast<int>(output_m),
      k,
      static_cast<DType const*>(input.data_ptr()),
      static_cast<float const*>(input_sf.data_ptr()),
      reinterpret_cast<uint32_t*>(output.data_ptr()),
      reinterpret_cast<uint32_t*>(output_sf.data_ptr()),
      useUE8M0,
      static_cast<int>(elts_mode),
      static_cast<int>(layout_mode),
      info.multi_processor_count,
      info.cc_major,
      stream);
}

}  // namespace sglang
