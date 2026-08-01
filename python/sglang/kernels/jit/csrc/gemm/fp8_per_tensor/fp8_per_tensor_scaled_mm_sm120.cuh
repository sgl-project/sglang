/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "fp8_per_tensor_rowwise_c3x.cuh"

// M-bucketed tile-shape dispatch, ported from vLLM's sm120_fp8_config_{M16,M32,M64,default}
// (csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh).
// The AOT kernel this replaces always used the single 128x128x128 default tile
// regardless of M, which wastes most of a 128-row tile on decode-shaped batches.
//
// The M16/M32 buckets need an explicit small EpilogueTile: EpilogueTileAuto
// selects for the 128-wide cooperative tiles and does not fit a 16/32-row CTA.
// SM120's cooperative kernel requires tile M >= 128, so every sub-128 bucket
// runs the pingpong schedule. Cluster is 1x1x1 throughout -- SM120 restricts
// programmatic multicast for these schedules.
//
// Measured on RTX PRO 6000 (cold-L2 CUPTI, CUDA graph) against the single-tile
// AOT kernel, over FP8 attention-projection shapes of two LLM checkpoints
// (hidden 6656 and 5120) at TP1 and TP2:
// 1.55x geomean for M<=256 (up to 4.5x at M=1 on narrow-N KV projections),
// 1.00x above it, where this dispatch falls through to the same 128x128x128
// tile the AOT kernel always used. The separate M16 bucket is worth keeping on
// its own: dropping it and letting M<=16 use the M32 tile costs 1.7% geomean.
//
// 128 is the largest usable tile M here. 256x128x128 does not compile -- at 48KB
// of A+B smem per stage it cannot fit the 2 stages sm120_mma_tma.hpp requires.
// 256x128x64 does fit and is correct, but measured 0.89x geomean against the
// 128x128x128 tile over M in [257, 4096] (0.71x at M=257), so the halved K tile
// costs more than the taller M tile wins. CUTLASS itself never generates a
// 256-row SM120 tile: PermTileM is capped at min(TileM, 128) and every config
// in cutlass_library's SM120 generators is 128 or 64 rows.
//
// The M<=16 bucket is occupancy-bound, not bandwidth-bound: it launches exactly
// ceil(N/64) CTAs, so on 188 SMs a narrow-N projection (N=128/256) runs 2-4 CTAs
// at ~5% of the ~1400 GB/s this part sustains, while N>=14336 fills the machine
// and reaches ~71%. Three ways out were measured and rejected:
//   * tile N=16 does not build -- a 16-wide epilogue trips "TiledCopy uses too
//     few vals for selected CopyAtom", and EpilogueTileAuto trips "EPI_TILE_N
//     must divide CTA_N". 32 is the floor.
//   * tile N=32 doubles the CTAs and is 1.17x on N=128/256, but regresses to
//     0.93x once ceil(N/32) overshoots 188 (N>=6656); 1.02x geomean overall.
//     Worth revisiting only behind an N-aware bucket.
//   * split-K cannot be reached from here at all: sub-128 tile M forces the
//     pingpong schedule, and pingpong static_asserts "Ping-pong kernel does not
//     currently support stream-K scheduler." Going cooperative to unlock it
//     costs more than the split wins -- tile M=128 + StreamK measured 0.70x,
//     and with an explicit splits=8 it was 0.57x.
template <typename OutType, bool WithBias>
void sm120_fp8_pertensor_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  const int m = a.size(0);
  using ArchTag = cutlass::arch::Sm120;
  using ClusterShape = Shape<_1, _1, _1>;
  using PingpongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueScheduleAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueTileAuto = cutlass::epilogue::collective::EpilogueTileAuto;

  if (m <= 16) {
    using GemmM16 = JitGemmFp8RowwiseC3x<
        ArchTag,
        OutType,
        Shape<_16, _64, _128>,
        ClusterShape,
        PingpongSchedule,
        EpilogueScheduleAuto,
        Shape<_16, _32>,
        WithBias>;
    return launch_c3x_fp8_rowwise_scaled_mm<GemmM16>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (m <= 32) {
    using GemmM32 = JitGemmFp8RowwiseC3x<
        ArchTag,
        OutType,
        Shape<_32, _64, _128>,
        ClusterShape,
        PingpongSchedule,
        EpilogueScheduleAuto,
        Shape<_32, _32>,
        WithBias>;
    return launch_c3x_fp8_rowwise_scaled_mm<GemmM32>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (m <= 256) {
    using GemmM64 = JitGemmFp8RowwiseC3x<
        ArchTag,
        OutType,
        Shape<_64, _64, _128>,
        ClusterShape,
        PingpongSchedule,
        EpilogueScheduleAuto,
        EpilogueTileAuto,
        WithBias>;
    return launch_c3x_fp8_rowwise_scaled_mm<GemmM64>(out, a, b, scales_a, scales_b, bias, stream);
  }

  using GemmDefault = JitGemmFp8RowwiseC3x<
      ArchTag,
      OutType,
      Shape<_128, _128, _128>,
      ClusterShape,
      cutlass::gemm::collective::KernelScheduleAuto,
      EpilogueScheduleAuto,
      EpilogueTileAuto,
      WithBias>;
  return launch_c3x_fp8_rowwise_scaled_mm<GemmDefault>(out, a, b, scales_a, scales_b, bias, stream);
}

template <typename OutType>
void sm120_fp8_pertensor_dispatch_bias(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  if (bias.has_value()) {
    return sm120_fp8_pertensor_dispatch_shape<OutType, true>(out, a, b, scales_a, scales_b, bias, stream);
  }
  return sm120_fp8_pertensor_dispatch_shape<OutType, false>(out, a, b, scales_a, scales_b, bias, stream);
}
