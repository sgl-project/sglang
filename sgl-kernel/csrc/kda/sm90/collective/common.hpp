// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cutlass/kernel_hardware_info.h>

#include <cute/tensor.hpp>

namespace kda::sm90::collective {

using namespace cute;

template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  constexpr int rA = decltype(rank(tA))::value;
  constexpr int rB = decltype(rank(tB))::value;
  constexpr int rC = decltype(rank(tC))::value;
  if constexpr (rA == 2 && rB == 2 && rC == 1) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < size<1>(tA); k_block++) {
      cute::gemm(atom, tA(_, k_block), tB(_, k_block), tC);
      atom.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    static_assert(rA == 3 && rB == 3 && rC == 3);
    CUTE_UNROLL
    for (int k_block = 0; k_block < size<2>(tA); k_block++) {
      cute::gemm(atom, tA(_, _, k_block), tB(_, _, k_block), tC);
      atom.accumulate_ = GMMA::ScaleOut::One;
    }
  }
}

template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  atom.accumulate_ = GMMA::ScaleOut::Zero;
  gemm_reset_zero_acc(atom, tA, tB, tC);
}

template <
    template <cute::GMMA::Major, cute::GMMA::Major, cute::GMMA::ScaleIn, cute::GMMA::ScaleIn> class Primitive,
    cute::GMMA::Major tA,
    cute::GMMA::Major tB,
    cute::GMMA::ScaleIn sA,
    cute::GMMA::ScaleIn sB>
CUTE_DEVICE constexpr auto convert_to_gmma_rs(cute::MMA_Atom<Primitive<tA, tB, sA, sB>> const& tiled_mma) {
  using Atom = cute::MMA_Atom<Primitive<tA, tB, sA, sB>>;
  using ElementA = typename Atom::ValTypeA;
  using ElementB = typename Atom::ValTypeB;
  using ElementC = typename Atom::ValTypeC;
  using Shape_MNK = typename Atom::Shape_MNK;
  using RS = decltype(cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC, Shape_MNK, tA, tB, sA, sB>());
  return cute::MMA_Atom<RS>{};
}

template <
    template <cute::GMMA::ScaleIn, cute::GMMA::ScaleIn> class Primitive,
    cute::GMMA::ScaleIn sA,
    cute::GMMA::ScaleIn sB>
CUTE_DEVICE constexpr auto convert_to_gmma_rs(cute::MMA_Atom<Primitive<sA, sB>> const& tiled_mma) {
  using Atom = cute::MMA_Atom<Primitive<sA, sB>>;
  using ElementA = typename Atom::ValTypeA;
  using ElementB = typename Atom::ValTypeB;
  using ElementC = typename Atom::ValTypeC;
  using Shape_MNK = typename Atom::Shape_MNK;
  constexpr auto tA = cute::GMMA::Major::K;
  constexpr auto tB = cute::GMMA::Major::K;
  using RS = decltype(cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC, Shape_MNK, tA, tB, sA, sB>());
  return cute::MMA_Atom<RS>{};
}

template <class Atom, class... Args>
CUTE_DEVICE constexpr auto convert_to_gmma_rs(cute::TiledMMA<Atom, Args...> const& tiled_mma) {
  return cute::TiledMMA<decltype(convert_to_gmma_rs(Atom{})), Args...>{};
}

template <typename CLayout, typename AValueShape>
CUTE_DEVICE constexpr auto convert_c_layout_to_a_layout(CLayout const& c, AValueShape const& a) {
  return make_layout(
      make_shape(a, shape<1>(c), make_shape(shape<2>(c), size<0>(c) / size(a))),
      make_stride(stride<0>(c), stride<1>(c), make_stride(stride<2>(c), size<2>(a) * stride<0, 2>(c))));
}

template <class Layout, class Stages = _1>
CUTE_DEVICE constexpr auto unstage_smem_layout(Layout const& layout, Stages stages = {}) {
  return composition(layout, make_tuple(_, _, make_layout(stages)));
}

template <class Element, class Accumulator, class OperandLayout_TV>
CUTE_DEVICE auto make_acc_into_op(Accumulator const& acc, OperandLayout_TV const& operand_layout_tv) {
  Tensor operand = make_fragment_like<Element>(convert_c_layout_to_a_layout(acc.layout(), shape<1>(operand_layout_tv)));
  Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());

  cute::copy(acc, operand_as_acc);

  if constexpr (sizeof(Element) == 1) {
    // 00 11 22 33 00 11 22 33 acc layout
    // 00 00 11 11 22 22 33 33 operand layout
    // BB AA AA BB AA BB BB AA conflict-free exchange pattern
    //                         16-bit exchange; so process two at a time potentially
    int tid = threadIdx.x % 4;
    auto values_u32 = recast<uint32_t>(operand);

    CUTE_UNROLL
    for (int n = 0; n < size<1>(values_u32); n++) {
      CUTE_UNROLL
      for (int k = 0; k < size<2>(values_u32); k++) {
        CUTE_UNROLL
        for (int ii = 0; ii < 8; ii += 4) {
          uint32_t values_tmp_0 = values_u32(ii / 2 + 0, n, k);
          uint32_t values_tmp_1 = values_u32(ii / 2 + 1, n, k);

          // step A:
          // t 1 v 0 -> t 0 v 1
          // t 2 v 0 -> t 1 v 0
          // t 0 v 1 -> t 2 v 0
          // t 3 v 1 -> t 3 v 1

          int v_to_send = tid == 1 || tid == 2 ? 0 : 1;
          int v_to_recv = v_to_send;
          int t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF;

          uint32_t values_tmp_a = v_to_send == 0 ? values_tmp_0 : values_tmp_1;

          values_tmp_a = __shfl_sync(0xFFFFFFFF, values_tmp_a, t_to_recv_from, 4);

          // step B:
          // t 0 v 0 -> t 0 v 0
          // t 3 v 0 -> t 1 v 1
          // t 1 v 1 -> t 2 v 1
          // t 2 v 1 -> t 3 v 0

          v_to_send = 1 - v_to_send;
          v_to_recv = 1 - v_to_recv;
          t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF;

          uint32_t values_tmp_b = v_to_send == 0 ? values_tmp_0 : values_tmp_1;

          values_tmp_b = __shfl_sync(0xFFFFFFFF, values_tmp_b, t_to_recv_from, 4);

          values_u32(ii / 2 + 0, n, k) = __byte_perm(values_tmp_a, values_tmp_b, v_to_send == 0 ? 0x1054 : 0x5410);
          values_u32(ii / 2 + 1, n, k) = __byte_perm(values_tmp_a, values_tmp_b, v_to_send == 0 ? 0x3276 : 0x7632);
        }
      }
    }
  }

  return operand;
}

// Convert float register values from BF16 MMA operand A layout to TF32 MMA operand A layout.
//
// Both SM80_16x8x8_F32BF16BF16F32_TN and SM80_16x8x8_F32TF32TF32F32_TN have the same
// per-thread fragment shape: ((_2,_2),_1,_4):((_1,_2),_0,_4) → 16 values per thread.
// But they map different (M, K) positions to each thread.
//
// BF16 LayoutA_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
//   t0 = tid % 4, t1 = tid / 4 (note: Shape<_4,_8> → colmajor → t0 = tid%4)
//   v_flat = v0 + v1*2: v0 → K offset (stride 16), v1 → M offset (stride 8)
//   Thread t0 holds K = {2*t0, 2*t0+1} (consecutive K per thread)
//
// TF32 LayoutA_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
//   v_flat = v0 + v1*2: v0 → M offset (stride 8), v1 → K offset (stride 64)
//   Thread t0 holds K = {t0, t0+4} (stride-4 K per thread)
//
// Algorithm (two-phase shuffle):
//   For each v0_tf32 (M selector in TF32), the source data is at v1_bf16 = v0_tf32
//   in BF16 layout. Both v1_tf32 outputs need the same BF16 value index but from
//   different source threads.
//
//   Phase 1: shuffle bf16_frag[idx0] (v0_bf16=0) from source thread
//   Phase 2: shuffle bf16_frag[idx1] (v0_bf16=1) from source thread
//   Then select phase1 or phase2 result based on t0%2.
//
// Supports BK=32 (NumKAtoms=4, 16 values) and BK=64 (NumKAtoms=8, 32 values).
// The shuffle pattern is identical per k-atom; only the loop count changes.
//
// @param frag_A  Float tensor with NumKAtoms*4 values in BF16 MMA A layout, converted
//                in-place to TF32 MMA A layout. Values remain as float; the TF32 MMA
//                hardware will truncate mantissa bits automatically during execution.
//                Caller uses recast<DstType>(frag_A) to obtain a typed view if needed.
// @param local_thread_idx  Thread index within the MMA tile (0..63)
template <int NumKAtoms = 4, class FragA>
CUTE_DEVICE void convert_bf16_to_tf32_operandA_layout(FragA& frag_A, int local_thread_idx) {
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragA>>::value);
  static_assert(NumKAtoms == 4 || NumKAtoms == 8, "Only BK=32 (4 k-atoms) and BK=64 (8 k-atoms) supported");
  static_assert(decltype(size(frag_A))::value == NumKAtoms * 4);
  // Fragment must hold float values (gated results are already in float).
  // MMA hardware will truncate to tf32 precision automatically.
  using ElemType = typename cute::remove_cvref_t<FragA>::value_type;
  static_assert(cute::is_same_v<ElemType, float>, "Fragment must be float; tf32 truncation is done by MMA hw");

  int tid = local_thread_idx % 32;  // lane within warp
  int t0 = tid % 4;
  bool sel_odd = (t0 & 1);  // t0%2: selects v0_bf16=1 result

  // Source lane for v1_tf32=0: t0_src = t0/2, lane = t0_src + (tid & ~3)
  // Source lane for v1_tf32=1: t0_src = t0/2+2, lane = (t0/2+2) + (tid & ~3)
  int src_lane_lo = (t0 / 2) + (tid & ~3);
  int src_lane_hi = (t0 / 2 + 2) + (tid & ~3);

  // Process NumKAtoms k-iterations, each with 4 values: [4j+0, 4j+1, 4j+2, 4j+3]
  // BF16 fragment layout per k-iter: (v0_bf16=0,v1_bf16=0), (v0_bf16=1,v1_bf16=0),
  //                                   (v0_bf16=0,v1_bf16=1), (v0_bf16=1,v1_bf16=1)
  // TF32 output layout per k-iter:   (v0_tf32=0,v1_tf32=0), (v0_tf32=1,v1_tf32=0),
  //                                   (v0_tf32=0,v1_tf32=1), (v0_tf32=1,v1_tf32=1)
  CUTE_UNROLL
  for (int j = 0; j < NumKAtoms; j++) {
    // Read all 4 input values for this k-iter before writing any output,
    // to avoid read-after-write hazard (in-place update).
    float in0 = frag_A(0 + 4 * j);  // v0_bf16=0, v1_bf16=0
    float in1 = frag_A(1 + 4 * j);  // v0_bf16=1, v1_bf16=0
    float in2 = frag_A(2 + 4 * j);  // v0_bf16=0, v1_bf16=1
    float in3 = frag_A(3 + 4 * j);  // v0_bf16=1, v1_bf16=1

    // For v0_tf32=0: M is selected by v1_bf16=0, so source values are (in0, in1)
    // For v0_tf32=1: M is selected by v1_bf16=1, so source values are (in2, in3)
    float out_vals[4];
    CUTE_UNROLL
    for (int v0_tf32 = 0; v0_tf32 < 2; v0_tf32++) {
      float val0 = (v0_tf32 == 0) ? in0 : in2;  // v0_bf16=0 at chosen v1_bf16
      float val1 = (v0_tf32 == 0) ? in1 : in3;  // v0_bf16=1 at chosen v1_bf16

      // Shuffle to get values from source threads
      float recv0_lo = __shfl_sync(0xFFFFFFFF, val0, src_lane_lo);
      float recv1_lo = __shfl_sync(0xFFFFFFFF, val1, src_lane_lo);
      float recv0_hi = __shfl_sync(0xFFFFFFFF, val0, src_lane_hi);
      float recv1_hi = __shfl_sync(0xFFFFFFFF, val1, src_lane_hi);

      // Select based on t0%2: even → v0_bf16=0, odd → v0_bf16=1
      out_vals[v0_tf32 + 0] = sel_odd ? recv1_lo : recv0_lo;  // v1_tf32=0
      out_vals[v0_tf32 + 2] = sel_odd ? recv1_hi : recv0_hi;  // v1_tf32=1
    }

    // Write all 4 output values
    frag_A(0 + 4 * j) = out_vals[0];
    frag_A(1 + 4 * j) = out_vals[1];
    frag_A(2 + 4 * j) = out_vals[2];
    frag_A(3 + 4 * j) = out_vals[3];
  }
}

// Convert float register values from BF16 MMA operand B layout to TF32 MMA operand B layout.
//
// BF16 LayoutB_TV: ((_4,_8),_2):((_16,_1),_8)
//   Thread t0 holds K = {2*t0, 2*t0+1}. v selects K offset (consecutive).
//
// TF32 LayoutB_TV: ((_4,_8),_2):((_8,_1),_32)
//   Thread t0 holds K = {t0, t0+4}. v selects K offset (stride-4).
//
// Same two-phase shuffle approach as operand A, but B has only 2 values per atom
// (no M dimension in the value index).
//
// Supports BK=32 (NumKAtoms=4, 8 values) and BK=64 (NumKAtoms=8, 16 values).
//
// @param frag_B  Float tensor with NumKAtoms*2 values in BF16 MMA B layout, converted
//                in-place. Values remain as float; TF32 MMA hardware truncates automatically.
//                Caller uses recast<DstType>(frag_B) to obtain a typed view if needed.
// @param local_thread_idx  Thread index within the MMA tile (0..63)
template <int NumKAtoms = 4, class FragB>
CUTE_DEVICE void convert_bf16_to_tf32_operandB_layout(FragB& frag_B, int local_thread_idx) {
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragB>>::value);
  static_assert(NumKAtoms == 4 || NumKAtoms == 8, "Only BK=32 (4 k-atoms) and BK=64 (8 k-atoms) supported");
  static_assert(decltype(size(frag_B))::value == NumKAtoms * 2);
  // Fragment must hold float values; MMA hardware truncates to tf32 automatically.
  using ElemType = typename cute::remove_cvref_t<FragB>::value_type;
  static_assert(cute::is_same_v<ElemType, float>, "Fragment must be float; tf32 truncation is done by MMA hw");

  int tid = local_thread_idx % 32;
  int t0 = tid % 4;
  bool sel_odd = (t0 & 1);

  int src_lane_lo = (t0 / 2) + (tid & ~3);
  int src_lane_hi = (t0 / 2 + 2) + (tid & ~3);

  // Process NumKAtoms k-iterations, each with 2 values: [2j, 2j+1]
  CUTE_UNROLL
  for (int j = 0; j < NumKAtoms; j++) {
    int idx0 = 2 * j;      // BF16 v=0, K = 2*t0
    int idx1 = 2 * j + 1;  // BF16 v=1, K = 2*t0+1

    float val0 = frag_B(idx0);
    float val1 = frag_B(idx1);

    // v_tf32=0: need K=t0 from src_t0=t0/2
    float recv0_lo = __shfl_sync(0xFFFFFFFF, val0, src_lane_lo);
    float recv1_lo = __shfl_sync(0xFFFFFFFF, val1, src_lane_lo);
    // v_tf32=1: need K=t0+4 from src_t0=t0/2+2
    float recv0_hi = __shfl_sync(0xFFFFFFFF, val0, src_lane_hi);
    float recv1_hi = __shfl_sync(0xFFFFFFFF, val1, src_lane_hi);

    frag_B(idx0) = sel_odd ? recv1_lo : recv0_lo;
    frag_B(idx1) = sel_odd ? recv1_hi : recv0_hi;
  }
}

// Broadcast row 0 from a BF16 MMA operand A fragment and output directly
// into a BF16 MMA operand B fragment.
//
// Combines broadcast_row0 + extract_A_to_B into one step, avoiding the
// intermediate 16-float operand A broadcast tensor (saves 8 float registers).
//
// Since g_first is broadcast (all M rows identical), operand B only needs
// the K-dimension values. We shuffle v1=0 values from the t1=0 thread
// and output the 8-value B fragment directly.
//
// BF16 A layout: t0 = tid % 4, t1 = tid / 4. Row 0 at t1=0, v1=0.
//   frag_A(4j+0) = (v0=0, v1=0), frag_A(4j+1) = (v0=1, v1=0)
// BF16 B layout: frag_B(2j+0) = v=0, frag_B(2j+1) = v=1
// Both have K = {2*t0, 2*t0+1} at same positions.
//
// Supports BK=32 (NumKAtoms=4) and BK=64 (NumKAtoms=8).
//
// Cost: 2 shuffles per k-iter × NumKAtoms k-iters.
// Saves: NumKAtoms*4-float intermediate tensor (vs broadcast_row0 + extract).
//
// @param frag_A       Input: alpha[m, k] in BF16 MMA A layout (NumKAtoms*4 values)
// @param frag_B_first Output: alpha[0, k] in BF16 MMA B layout (NumKAtoms*2 values)
// @param local_thread_idx  Thread index within the MMA tile (0..63)
template <int NumKAtoms = 4, class FragA, class FragB>
CUTE_DEVICE void
broadcast_row0_operandA_to_operandB_bf16_layout(FragA const& frag_A, FragB& frag_B_first, int local_thread_idx) {
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragA>>::value);
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragB>>::value);
  static_assert(NumKAtoms == 4 || NumKAtoms == 8, "Only BK=32 (4 k-atoms) and BK=64 (8 k-atoms) supported");
  static_assert(decltype(size(frag_A))::value == NumKAtoms * 4);
  static_assert(decltype(size(frag_B_first))::value == NumKAtoms * 2);

  int tid = local_thread_idx % 32;  // lane within warp
  // Row 0 is at t1=0. In BF16 A layout, t0 = tid % 4, t1 = tid / 4.
  // Source lane: same t0, t1=0 → src = tid % 4.
  int src_lane = tid % 4;

  CUTE_UNROLL
  for (int j = 0; j < NumKAtoms; j++) {
    // Shuffle v1=0 values from thread with t1=0 (row 0 holder)
    // frag_A(4j+0) = (v0=0, v1=0) → K=2*t0
    // frag_A(4j+1) = (v0=1, v1=0) → K=2*t0+1
    auto val0 = __shfl_sync(0xFFFFFFFF, frag_A(4 * j + 0), src_lane);
    auto val1 = __shfl_sync(0xFFFFFFFF, frag_A(4 * j + 1), src_lane);

    // Output directly into B layout: frag_B(2j) = K_lo, frag_B(2j+1) = K_hi
    frag_B_first(2 * j + 0) = val0;
    frag_B_first(2 * j + 1) = val1;
  }
}

// Broadcast row 0 across all M rows in a BF16 MMA operand A fragment.
//
// Given frag_A holding alpha[m, k] per thread, produces frag_A_first holding alpha[0, k]
// for all m (broadcast). This eliminates a redundant S2R load of g_first from shared memory.
//
// BF16 LayoutA_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
//   tid decomposition: t0 = tid % 4, t1 = tid / 4
//   m = t1 + v1*8 (v1 selects M row within thread)
//   Row 0 is held by threads with t1=0, at v1=0 positions.
//
// Algorithm:
//   For each k-iter, shuffle v1=0 values from thread (tid % 4) (same t0, t1=0),
//   then replicate to v1=1 positions.
//
// Supports BK=32 (NumKAtoms=4) and BK=64 (NumKAtoms=8).
//
// Cost: 2 shuffles per k-iter × NumKAtoms k-iters.
//
// @param frag_A       Input: alpha[m, k] in BF16 MMA A layout (NumKAtoms*4 values)
// @param frag_A_first Output: alpha[0, k] broadcast in BF16 MMA A layout (NumKAtoms*4 values)
// @param local_thread_idx  Thread index within the MMA tile (0..63)
template <int NumKAtoms = 4, class FragA, class FragAFirst>
CUTE_DEVICE void
broadcast_row0_operandA_bf16_layout(FragA const& frag_A, FragAFirst& frag_A_first, int local_thread_idx) {
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragA>>::value);
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragAFirst>>::value);
  static_assert(NumKAtoms == 4 || NumKAtoms == 8, "Only BK=32 (4 k-atoms) and BK=64 (8 k-atoms) supported");
  static_assert(decltype(size(frag_A))::value == NumKAtoms * 4);
  static_assert(decltype(size(frag_A_first))::value == NumKAtoms * 4);

  int tid = local_thread_idx % 32;  // lane within warp
  // Row 0 is at t1=0 → tid % 4 (keep same t0, set t1=0)
  int src_lane = tid % 4;

  CUTE_UNROLL
  for (int j = 0; j < NumKAtoms; j++) {
    // v1=0 positions hold m=t1, v1=1 positions hold m=t1+8
    // We want m=0, which is at t1=0, v1=0 → indices 4j+0 and 4j+1
    auto val0 = __shfl_sync(0xFFFFFFFF, frag_A(4 * j + 0), src_lane);  // alpha[0, 2*t0] from t1=0
    auto val1 = __shfl_sync(0xFFFFFFFF, frag_A(4 * j + 1), src_lane);  // alpha[0, 2*t0+1] from t1=0

    // Broadcast to both v1=0 and v1=1 (same value, different M rows)
    frag_A_first(4 * j + 0) = val0;  // v0=0, v1=0
    frag_A_first(4 * j + 1) = val1;  // v0=1, v1=0
    frag_A_first(4 * j + 2) = val0;  // v0=0, v1=1 (broadcast)
    frag_A_first(4 * j + 3) = val1;  // v0=1, v1=1 (broadcast)
  }
}

// Extract BF16 MMA operand B fragment from a BF16 MMA operand A fragment,
// for data that is **broadcast across M rows** (e.g., g_first = g[row=0, :]).
//
// When the source data is broadcast (identical for all M rows), the K-dimension
// mapping is the same in both A and B BF16 MMA layouts:
//   A: thread t0 holds K = {2*t0, 2*t0+1} at v0={0,1}, with v1 selecting M row
//   B: thread t0 holds K = {2*t0, 2*t0+1} at v={0,1}
//
// Since M rows are identical (broadcast), we can simply pick v1=0 from A:
//   frag_B(2j + 0) = frag_A(4j + 0)   // v0_bf16=0 → K=2*t0
//   frag_B(2j + 1) = frag_A(4j + 1)   // v0_bf16=1 → K=2*t0+1
//
// Supports BK=32 (NumKAtoms=4) and BK=64 (NumKAtoms=8).
//
// This avoids a redundant S2R load from shared memory.
// No warp shuffles needed — purely register-local extraction.
//
// @param frag_A  Float tensor with NumKAtoms*4 values in BF16 MMA A layout (broadcast data)
// @param frag_B  Float tensor with NumKAtoms*2 values in BF16 MMA B layout (output)
template <int NumKAtoms = 4, class FragA, class FragB>
CUTE_DEVICE void extract_broadcast_operandA_to_operandB_bf16_layout(FragA const& frag_A, FragB& frag_B) {
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragA>>::value);
  static_assert(cute::is_tensor<cute::remove_cvref_t<FragB>>::value);
  static_assert(NumKAtoms == 4 || NumKAtoms == 8, "Only BK=32 (4 k-atoms) and BK=64 (8 k-atoms) supported");
  static_assert(decltype(size(frag_A))::value == NumKAtoms * 4);
  static_assert(decltype(size(frag_B))::value == NumKAtoms * 2);

  CUTE_UNROLL
  for (int j = 0; j < NumKAtoms; j++) {
    // A layout per k-iter: [4j+0]=(v0=0,v1=0), [4j+1]=(v0=1,v1=0), [4j+2]=(v0=0,v1=1), [4j+3]=(v0=1,v1=1)
    // B layout per k-iter: [2j+0]=v=0, [2j+1]=v=1
    // For broadcast data, v1 doesn't matter, so pick v1=0:
    frag_B(2 * j + 0) = frag_A(4 * j + 0);  // K = 2*t0
    frag_B(2 * j + 1) = frag_A(4 * j + 1);  // K = 2*t0+1
  }
}

}  // namespace kda::sm90::collective
