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

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>

#include <cute/tensor.hpp>

#include "kerutils/common/cute_ext.hpp"

namespace kerutils {

using namespace cute;

template <int dim, typename Layout>
constexpr bool is_contiguous(Layout&& layout) {
  auto dim_layout = get<dim>(layout);
  if constexpr (rank(dim_layout) == 0) {
    return stride(dim_layout) == 1;
  } else {
    return stride<0>(dim_layout) == 1;
  }
}

namespace detail::SM80 {

// ============================================================================
// convert_c_layout_to_a_layout — Compile-time layout reshape: C-fragment → A-fragment
// ============================================================================
// The C-fragment (accumulator) layout is (FragAtomC, MMA_M, MMA_N).
// The A-fragment (operand A) layout is  (FragAtomA, MMA_M, MMA_K).
//
// When FragAtomA > FragAtomC (e.g. SM80_16x8x16 where A has 8 vals/atom but
// C has 4 vals/atom, ratio=2), the N-dimension is split: MMA_N → (ratio, MMA_N/ratio),
// and the "ratio" factor is folded into the value (FragAtom) dimension:
//   C: ((2,2), MMA_M, MMA_N) → ((2,2,ratio), MMA_M, MMA_N/ratio)
//
// For SM80_16x8x8 TF32 (used in the inverse), FragAtomA == FragAtomC == 4,
// so ratio=1 and this function returns the C-layout unchanged. The function is
// still called to maintain a uniform code path and to reinterpret the fragment's
// rank-3 indexing in A-operand terms for the per-k-atom shuffle loop.
//
template <typename CLayout, typename TiledMMA>
CUTE_DEVICE constexpr auto convert_c_layout_to_a_layout(CLayout const& c, TiledMMA const& tiled_mma) {
  constexpr auto c_frag_atom_size = size<0>(CLayout{});  // values per C atom (4 for SM80_16x8)
  constexpr auto a_frag_atom_size = size<1>(typename TiledMMA::AtomLayoutA_TV{});  // values per A atom
  static_assert(a_frag_atom_size % c_frag_atom_size == 0);
  constexpr auto ratio = a_frag_atom_size / c_frag_atom_size;  // 1 for 16x8x8, 2 for 16x8x16
  if constexpr (ratio == 1) {
    // SM80_16x8x8: C and A have the same value count per atom → no reshape needed
    return CLayout{};
  } else {
    // SM80_16x8x16: A atom has 2× more values than C atom.
    // Split N-dimension by ratio, fold the extra factor into the value dimension:
    //   C: (FragAtom, MMA_M, MMA_N)
    //   → logical_divide by (_, _, ratio): (FragAtom, MMA_M, (ratio, MMA_N/ratio))
    //   → flatten first + third[0]: ((FragAtom, ratio), MMA_M, MMA_N/ratio)
    constexpr auto tiler = make_shape(_, _, Int<ratio>{});      // keep FragAtom and MMA_M, split MMA_N
    constexpr auto divided = logical_divide(CLayout{}, tiler);  // (FragAtom, MMA_M, (ratio, MMA_N/ratio))

    return make_layout(flatten(make_layout(get<0>(divided), get<2, 0>(divided))), get<1>(divided), get<2, 1>(divided));
  }
}

// ============================================================================
// make_acc_into_op — Generic acc→operand conversion (FP16 path only)
// ============================================================================
// Works for FP16 MMA because C-layout and A-layout have the SAME thread-to-K
// mapping ({2*t0, 2*t0+1} for both). A simple in-thread element copy suffices.
// For TF32 MMA, thread-to-K mappings DIFFER between C and A layouts, so this
// function is INCORRECT for TF32. Use convert_fp32_acc_to_tf32_operandA_layout
// instead, which performs cross-thread __shfl_sync shuffles.
template <class Element, class Accumulator, class TiledMMA>
CUTE_DEVICE auto make_acc_into_op(Accumulator const& acc, TiledMMA const& tiled_mma) {
  Tensor operand = make_fragment_like<Element>(convert_c_layout_to_a_layout(acc.layout(), tiled_mma));
  Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());
  cute::copy(acc, operand_as_acc);
  return operand;
}

// ============================================================================
// convert_fp32_acc_to_tf32_operandA_layout — Cross-thread shuffle: C-layout → A-layout (TF32)
// ============================================================================
// Converts an MMA accumulator fragment (C-layout, float) into an A-operand fragment
// (A-layout, float or tf32) via warp-level __shfl_sync shuffles.
//
// Why this is needed:
//   The SM80 TF32 MMA (16x8x8) has different thread-value (TV) mappings for
//   C-layout vs A-layout. Specifically, the K-dimension ownership differs:
//     C-layout: t0 selects M-row group (stride 32), v0/v1 select M sub-rows
//     A-layout: t0 selects K-column pair {t0, t0+4}, v0 selects M-half, v1 selects K-half
//   This means values must be redistributed ACROSS threads (not just within a thread).
//
// Algorithm overview (per k-atom, 4 values each):
//   1. Read all 4 src values into locals (avoid read-after-write hazard)
//   2. For each M-half (v0_tf32 ∈ {0,1}):
//      a. Select the 2 source values for this M-half (maps to v1_bf16 in C-layout)
//      b. Shuffle from source threads: src_lane = t0/2 (K-lo) or t0/2+2 (K-hi)
//      c. Select v0_bf16=0 or v0_bf16=1 based on t0 parity (even/odd)
//   3. Write 4 output values (with optional TF32 rounding)
//
// Cost: 8 __shfl_sync per k-atom (2 M-halves × 4 shuffles each).
//
// Template parameters:
//   NumKAtoms   — number of K-atoms to process (1, 2, or 4)
//   RoundingTF32 — if true, output is tfloat32_t with explicit rounding;
//                  if false, output is float and MMA hardware truncates
template <int NumKAtoms = 4, bool RoundingTF32 = false, class FragSrc, class FragDst, class TiledMMA>
CUTE_DEVICE void convert_fp32_acc_to_tf32_operandA_layout(
    const FragSrc& frag_src, FragDst& frag_dst, TiledMMA const& tiled_mma, int lane_id) {
  using ElemSrc = typename cute::remove_cvref_t<FragSrc>::value_type;
  using ElemDst = typename cute::remove_cvref_t<FragDst>::value_type;
  static_assert(cute::is_same_v<ElemSrc, float>, "Fragment input must be float with acc data dtype");
  static_assert(
      RoundingTF32 || cute::is_same_v<ElemDst, float>,
      "Fragment must be float with no rounding; tf32 truncation is done by MMA hardware");
  static_assert(
      !RoundingTF32 || cute::is_same_v<ElemDst, cutlass::tfloat32_t>,
      "Fragment must be tfloat32 with rounding; tf32 truncation is done manually");

  // Reinterpret the source fragment with A-layout indexing.
  // For SM80_16x8x8 TF32, ratio=1 so this is effectively a no-op (same shape).
  // For SM80_16x8x16, this would reshape (4, M, N) → (8, M, N/2) to align k-atoms.
  auto frag_src_cvt = make_tensor(frag_src.data(), convert_c_layout_to_a_layout(frag_src.layout(), tiled_mma));

  // Thread decomposition within a warp of 32 threads.
  // tid = t0 + 4*t1, where t0 ∈ [0,4) indexes K-groups, t1 ∈ [0,8) indexes M-rows.
  int tid = lane_id;
  int t0 = tid % 4;
  // Parity of t0 determines which v0_bf16 value to select after shuffle:
  //   even t0 → needs v0_bf16=0 (K = 2*(t0/2) = t0)
  //   odd  t0 → needs v0_bf16=1 (K = 2*(t0/2)+1 = t0)
  bool sel_odd = (t0 & 1);

  // Compute source lane IDs for the K-dimension shuffle.
  // In C-layout, thread with t0_src holds K = {2*t0_src, 2*t0_src+1}.
  // We need K = t0 (from t0_src = t0/2) and K = t0+4 (from t0_src = t0/2+2).
  // The upper bits of lane_id (t1 portion) stay the same: (tid & ~3).
  int src_lane_lo = (t0 / 2) + (tid & ~3);      // source for K = t0     (v1_tf32=0)
  int src_lane_hi = (t0 / 2 + 2) + (tid & ~3);  // source for K = t0 + 4 (v1_tf32=1)

  // Process each k-atom independently. Each atom has 4 values indexed as:
  //   C-layout: [4j+0]=(v0=0,v1=0), [4j+1]=(v0=1,v1=0), [4j+2]=(v0=0,v1=1), [4j+3]=(v0=1,v1=1)
  //     where v0 selects K sub-position, v1 selects M sub-row
  //   A-layout: [4j+0]=(v0=0,v1=0), [4j+1]=(v0=1,v1=0), [4j+2]=(v0=0,v1=1), [4j+3]=(v0=1,v1=1)
  //     where v0 selects M half (0 or +8), v1 selects K half (lo or hi)
  CUTE_UNROLL
  for (int j = 0; j < NumKAtoms; j++) {
    // Step 1: Read all 4 input values BEFORE writing any output.
    // This prevents read-after-write hazard when src and dst share storage
    // (output positions overlap with input positions within the same k-atom).
    float in0 = frag_src_cvt(0 + 4 * j);  // C-layout: (v0=0, v1=0) → K-lo, M-row-0
    float in1 = frag_src_cvt(1 + 4 * j);  // C-layout: (v0=1, v1=0) → K-hi, M-row-0
    float in2 = frag_src_cvt(2 + 4 * j);  // C-layout: (v0=0, v1=1) → K-lo, M-row-1
    float in3 = frag_src_cvt(3 + 4 * j);  // C-layout: (v0=1, v1=1) → K-hi, M-row-1

    // Step 2: Process each M-half (v0_tf32) independently.
    // v0_tf32=0 (M-half 0) needs data from v1_bf16=0 → (in0, in1)
    // v0_tf32=1 (M-half 1) needs data from v1_bf16=1 → (in2, in3)
    float out_vals[4];
    CUTE_UNROLL
    for (int v0_tf32 = 0; v0_tf32 < 2; v0_tf32++) {
      // Select the two source values for this M-half.
      // In C-layout, v1 selects M-row; in A-layout, v0 selects M-half.
      // So v0_tf32 maps to v1_bf16 for M-row selection.
      float val0 = (v0_tf32 == 0) ? in0 : in2;  // v0_bf16=0 at chosen M-half
      float val1 = (v0_tf32 == 0) ? in1 : in3;  // v0_bf16=1 at chosen M-half

      // Step 3: Cross-thread shuffle for K-dimension remapping.
      // Fetch both v0_bf16 values (=0 and =1) from the source threads,
      // then select the correct one based on t0 parity.
      float recv0_lo = __shfl_sync(0xFFFFFFFF, val0, src_lane_lo);  // v0_bf16=0 from K-lo source
      float recv1_lo = __shfl_sync(0xFFFFFFFF, val1, src_lane_lo);  // v0_bf16=1 from K-lo source
      float recv0_hi = __shfl_sync(0xFFFFFFFF, val0, src_lane_hi);  // v0_bf16=0 from K-hi source
      float recv1_hi = __shfl_sync(0xFFFFFFFF, val1, src_lane_hi);  // v0_bf16=1 from K-hi source

      // Step 4: Select correct value based on t0 parity.
      // Even t0 needs v0_bf16=0 (recv0), odd t0 needs v0_bf16=1 (recv1).
      out_vals[v0_tf32 + 0] = sel_odd ? recv1_lo : recv0_lo;  // v1_tf32=0 (K-lo half)
      out_vals[v0_tf32 + 2] = sel_odd ? recv1_hi : recv0_hi;  // v1_tf32=1 (K-hi half)
    }

    // Step 5: Write output values in A-layout order.
    // If RoundingTF32: explicit cast to tfloat32_t applies rounding (x += 0x1000u).
    // If !RoundingTF32: store as float; MMA hardware truncates to TF32 precision.
    if constexpr (RoundingTF32) {
      frag_dst(0 + 4 * j) = (ElemDst)out_vals[0];  // (v0_tf32=0, v1_tf32=0) → M-half-0, K-lo
      frag_dst(1 + 4 * j) = (ElemDst)out_vals[1];  // (v0_tf32=1, v1_tf32=0) → M-half-1, K-lo
      frag_dst(2 + 4 * j) = (ElemDst)out_vals[2];  // (v0_tf32=0, v1_tf32=1) → M-half-0, K-hi
      frag_dst(3 + 4 * j) = (ElemDst)out_vals[3];  // (v0_tf32=1, v1_tf32=1) → M-half-1, K-hi
    } else {
      frag_dst(0 + 4 * j) = out_vals[0];
      frag_dst(1 + 4 * j) = out_vals[1];
      frag_dst(2 + 4 * j) = out_vals[2];
      frag_dst(3 + 4 * j) = out_vals[3];
    }
  }
}

}  // namespace detail::SM80

template <class Element, bool GarbageFilledDiagonal, bool GarbageFilledUpperTriangular, bool RoundingTF32 = false>
struct CollectiveInverseTF32 {
  static_assert(std::is_same_v<Element, cutlass::tfloat32_t>);
  using ElementView = std::conditional_t<RoundingTF32, Element, float>;

  CUTE_DEVICE
  CollectiveInverseTF32(int wg_sync_named_barrier_id) : wg_sync_named_barrier_id_(wg_sync_named_barrier_id) {}

  template <typename TensorT>
  CUTE_DEVICE void compute(TensorT&& sT) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 64);
    static_assert(size<1>(L) == 64);

    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    auto sT_view = recast<ElementView>(sT);
    auto t16X16sT = flat_divide(sT_view, Shape<_16, _16>{});

    if (thread_idx < 64) {  // compute 16x16 inverse on diagnal directly
      compute_diagonal_inverse_NxN<16>(t16X16sT(_, _, thread_idx / 16, thread_idx / 16), thread_idx % 16);
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);

    auto t32X32sT = flat_divide(sT_view, Shape<_32, _32>{});
    if (thread_idx < 64) {  // two warps for 16x16 -> 32x32
      blockwise_diagonal_inversed_16x16_to_32x32(t32X32sT(_, _, thread_idx / 32, thread_idx / 32));
    }

    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);
    // one warpgroup for 32x32 -> 64x64
    blockwise_diagonal_inversed_32x32_to_64x64(sT_view);
  }

 private:
  template <int N, typename TensorT>
  CUTE_DEVICE void compute_diagonal_inverse_NxN(TensorT&& mat, int tid_in_group) {  // group_size = N
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == N);
    static_assert(size<1>(L) == N);

    using ElementCompute = float;

    using CopyOp = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementView>;

    auto load_row = [&](int y) {
      auto row = make_tensor<ElementView>(Shape<Int<N>>{});
      copy(CopyOp{}, std::forward<TensorT>(mat)(y, _), row);

      auto row_cvt = [&]() {
        if constexpr (RoundingTF32) {
          return make_tensor_like<ElementCompute>(row);
        } else {
          return row;
        }
      }();
      if constexpr (RoundingTF32) {
        copy(row, row_cvt);
      }

      if constexpr (GarbageFilledDiagonal || GarbageFilledUpperTriangular) {
        CUTE_UNROLL
        for (int i = 0; i < N; ++i) {
          row_cvt(i) = i == y ? 1.0f : (i > y ? 0.0f : row_cvt(i));
        }
      }
      return row_cvt;
    };

    auto store_row = [&](int y, auto row) {
      auto row_cvt = [&]() {
        if constexpr (RoundingTF32) {
          return make_tensor_like<ElementView>(row);
        } else {
          return row;
        }
      }();
      if constexpr (RoundingTF32) {
        copy(row, row_cvt);
      }
      copy(CopyOp{}, row_cvt, std::forward<TensorT>(mat)(y, _));
    };

    auto row = load_row(tid_in_group);
#define LOAD(y, x) __shfl_sync(0xffffffff, row(x), y, N)

    CUTE_UNROLL
    for (int src_row = 0; src_row < N - 1; ++src_row) {  // idx of src row to eliminate
      auto row_scale = -row(src_row);                    // scale the src row
      CUTE_UNROLL
      for (int i = 0; i < src_row; ++i) {
        auto src_row_value = LOAD(src_row, i);
        row(i) = tid_in_group > src_row ? row_scale * src_row_value + row(i) : row(i);
      }
      row(src_row) = tid_in_group > src_row ? row_scale : row(src_row);
    }

#undef LOAD

    store_row(tid_in_group, row);
  }

  /*
  blockwise inverse has relation as follows
  inv(| A 0 |)     = |          inv(A)       0  |
      | C D |        | -inv(D)C inv(A)   inv(D) |
  */

  // not used
  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_8x8_to_16x16(TensorT&& mat) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 16);
    static_assert(size<1>(L) == 16);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    auto mat_8x8_2x2 = flat_divide(std::forward<TensorT>(mat), Shape<_8, _8>{});
    using MMA = SM80_16x8x8_F32TF32TF32F32_TN;
    using TiledMMA = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _8, _8>{}));

    using CopyOpTF32 = AutoVectorizingCopyWithAssumedAlignment<128>;
    using CopyOpD_S2R = CopyOpTF32;
    using CopyOpC_S2R = CopyOpTF32;
    using CopyOpA_S2R = CopyOpTF32;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = CopyOpTF32;
#else
    using CopyOpO_R2S = UniversalCopy<ElementView, ElementView>;
#endif

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma = TiledMMA{};
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, ElementView>{}, tiled_mma);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, ElementView>{}, tiled_mma);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, ElementView>{}, tiled_mma);
    auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, ElementView>{}, tiled_mma);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_copy = O_tiled_copy.get_thread_slice(lane_id);

    Tensor sDinv = mat_8x8_2x2(_, _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_8x8_2x2(_, _, _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_8x8_2x2(_, _, _0{}, _0{}));
    Tensor sO = mat_8x8_2x2(_, _, _1{}, _0{});

    Tensor sDinv_m_bcast = make_tensor(sDinv.data(), logical_product(sDinv.layout(), Tile<Layout<_2, _0>>{}));
    Tensor sO_m_bcast = make_tensor(sO.data(), logical_product(sO.layout(), Tile<Layout<_2, _0>>{}));

    Tensor tOrDinv = make_fragment_like<ElementView>(partition_shape_A(tiled_mma, Shape<_16, _8>{}));
    Tensor tOrC = make_fragment_like<ElementView>(partition_shape_B(tiled_mma, Shape<_8, _8>{}));
    // Tensor tOrC    = thr_mma.partition_fragment_B(sC);
    Tensor tOrAinv = make_fragment_like<ElementView>(partition_shape_B(tiled_mma, Shape<_8, _8>{}));
    // Tensor tOrAinv = thr_mma.partition_fragment_B(sAinv);

    Tensor tDCrDC = partition_fragment_C(tiled_mma, Shape<_16, _8>{});  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma, Shape<_16, _8>{});    // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv_m_bcast);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);
    Tensor tOsO = O_thr_copy.partition_D(sO_m_bcast);
    Tensor tOrO_cv = O_thr_copy.retile_S(tOrO);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv, tOrDinv_cv);
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    auto tOrDinv_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrDinv);
      } else {
        return tOrDinv;
      }
    }();
    auto tOrC_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrC);
      } else {
        return tOrC;
      }
    }();
    gemm(tiled_mma, tOrDinv_mma, tOrC_mma, tDCrDC);
    transform(tDCrDC, [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = make_fragment_like<ElementView>(partition_shape_A(tiled_mma, Shape<_16, _8>{}));
    // Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma);
    detail::SM80::convert_fp32_acc_to_tf32_operandA_layout<1, RoundingTF32>(tDCrDC, tOrDC, tiled_mma, lane_id);
    auto tOrDC_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrDC);
      } else {
        return tOrDC;
      }
    }();

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);

    clear(tOrO);
    auto tOrAinv_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrAinv);
      } else {
        return tOrAinv;
      }
    }();
    gemm(tiled_mma, tOrDC_mma, tOrAinv_mma, tOrO);

    if constexpr (!RoundingTF32) {
      // no need for conversion
      copy(O_tiled_copy, tOrO_cv, tOsO);
    } else {
      auto tOrO_cv_cvt = make_tensor_like<Element>(tOrO_cv);
      transform(tOrO_cv, tOrO_cv_cvt, [](auto v) { return Element(v); });
      copy(O_tiled_copy, tOrO_cv_cvt, tOsO);
    }
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_16x16_to_32x32(TensorT&& mat) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 32);
    static_assert(size<1>(L) == 32);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    // row-major
    constexpr bool is_col_major = is_contiguous<0>(L);

    using TileShape = Shape<_16, _16, _16>;
    auto mat_16x16_2x2 = flat_divide(std::forward<TensorT>(mat), select<0, 1>(TileShape{}));

    using MMA = SM80_16x8x8_F32TF32TF32F32_TN;
    using TiledMMA = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, TileShape{}));

    using CopyOpTF32 = AutoVectorizingCopyWithAssumedAlignment<128>;
    using CopyOpD_S2R = CopyOpTF32;
    using CopyOpC_S2R = CopyOpTF32;
    using CopyOpA_S2R = CopyOpTF32;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = CopyOpTF32;
#else
    using CopyOpO_R2S = UniversalCopy<ElementView, ElementView>;
#endif

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma = TiledMMA{};
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, ElementView>{}, tiled_mma);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, ElementView>{}, tiled_mma);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, ElementView>{}, tiled_mma);
    auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, ElementView>{}, tiled_mma);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_copy = O_tiled_copy.get_thread_slice(lane_id);

    Tensor sDinv = mat_16x16_2x2(_, _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_16x16_2x2(_, _, _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_16x16_2x2(_, _, _0{}, _0{}));
    Tensor sO = mat_16x16_2x2(_, _, _1{}, _0{});

    Tensor tOrDinv = make_fragment_like<ElementView>(partition_shape_A(tiled_mma, select<0, 2>(TileShape{})));
    Tensor tOrC = make_fragment_like<ElementView>(partition_shape_B(tiled_mma, select<1, 2>(TileShape{})));
    Tensor tOrAinv = make_fragment_like<ElementView>(partition_shape_B(tiled_mma, select<1, 2>(TileShape{})));

    Tensor tDCrDC = partition_fragment_C(tiled_mma, select<0, 1>(TileShape{}));  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 1>(TileShape{}));    // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);
    Tensor tOsO = O_thr_copy.partition_D(sO);
    Tensor tOrO_cv = O_thr_copy.retile_S(tOrO);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv, tOrDinv_cv);
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    auto tOrDinv_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrDinv);
      } else {
        return tOrDinv;
      }
    }();
    auto tOrC_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrC);
      } else {
        return tOrC;
      }
    }();
    gemm(tiled_mma, tOrDinv_mma, tOrC_mma, tDCrDC);
    transform(tDCrDC, [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = make_fragment_like<ElementView>(partition_shape_A(tiled_mma, Shape<_16, _16>{}));
    // Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma);
    detail::SM80::convert_fp32_acc_to_tf32_operandA_layout<2, RoundingTF32>(tDCrDC, tOrDC, tiled_mma, lane_id);
    auto tOrDC_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrDC);
      } else {
        return tOrDC;
      }
    }();

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);
    clear(tOrO);

    auto tOrAinv_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrAinv);
      } else {
        return tOrAinv;
      }
    }();
    gemm(tiled_mma, tOrDC_mma, tOrAinv_mma, tOrO);

    if constexpr (!RoundingTF32) {
      // no need for conversion: MMA hardware truncates float→tf32
      copy(O_tiled_copy, tOrO_cv, tOsO);
    } else {
      // Explicit rounding: convert float→tfloat32_t before store to SMEM
      auto tOrO_cv_cvt = make_tensor_like<Element>(tOrO_cv);
      transform(tOrO_cv, tOrO_cv_cvt, [](auto v) { return Element(v); });
      copy(O_tiled_copy, tOrO_cv_cvt, tOsO);
    }
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_32x32_to_64x64(TensorT&& mat) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 64);
    static_assert(size<1>(L) == 64);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    auto mat_32x32_2x2 = flat_divide(std::forward<TensorT>(mat), select<0, 1>(Shape<_32, _32>{}));
    auto mat_16x2X16x2_2x2 = logical_divide(mat_32x32_2x2, Shape<_16, _16>{});

    using MMA = SM80_16x8x8_F32TF32TF32F32_TN;
    using TiledMMA1 = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _16, _32>{}));
    using TiledMMA2 = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _32, _16>{}));

    using CopyOpTF32 = AutoVectorizingCopyWithAssumedAlignment<128>;
    using CopyOpD_S2R = CopyOpTF32;
    using CopyOpC_S2R = CopyOpTF32;
    using CopyOpA_S2R = CopyOpTF32;
    using CopyOpO_S2R = CopyOpTF32;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = CopyOpTF32;
#else
    using CopyOpO_R2S = UniversalCopy<ElementView, ElementView>;
#endif

    int warp_id_in_wg =
        cutlass::canonical_warp_idx() - cutlass::NumWarpsPerWarpGroup * cutlass::canonical_warp_group_idx();
    int x = warp_id_in_wg / 2;
    int y = warp_id_in_wg % 2;

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma1 = TiledMMA1{};
    auto thr_mma1 = tiled_mma1.get_thread_slice(lane_id);

    auto tiled_mma2 = TiledMMA2{};
    auto thr_mma2 = tiled_mma2.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, ElementView>{}, tiled_mma1);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, ElementView>{}, tiled_mma1);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, ElementView>{}, tiled_mma2);
    auto O_tiled_s2r = make_tiled_copy_C(Copy_Atom<CopyOpO_S2R, ElementView>{}, tiled_mma2);
    auto O_tiled_r2s = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, ElementView>{}, tiled_mma2);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_s2r = O_tiled_s2r.get_thread_slice(lane_id);
    auto O_thr_r2s = O_tiled_r2s.get_thread_slice(lane_id);

    Tensor sDinv = mat_16x2X16x2_2x2(make_coord(_, y), _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_16x2X16x2_2x2(_, make_coord(_, x), _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_16x2X16x2_2x2(make_coord(_, x), _, _0{}, _0{}));  // NOTE: not y!
    Tensor sO = mat_16x2X16x2_2x2(make_coord(_, y), _, _1{}, _0{});  // needs cross-warp reduction

    Tensor tOrDinv = make_fragment_like<ElementView>(partition_shape_A(tiled_mma1, Shape<_16, _32>{}));
    Tensor tOrC = make_fragment_like<ElementView>(partition_shape_B(tiled_mma1, Shape<_16, _32>{}));
    Tensor tOrAinv = make_fragment_like<ElementView>(partition_shape_B(tiled_mma2, Shape<_32, _16>{}));

    Tensor tDCrDC = partition_fragment_C(tiled_mma1, Shape<_16, _16>{});  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma2, Shape<_16, _32>{});    // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv, tOrDinv_cv);
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    auto tOrDinv_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrDinv);
      } else {
        return tOrDinv;
      }
    }();
    auto tOrC_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrC);
      } else {
        return tOrC;
      }
    }();
    gemm(tiled_mma1, tOrDinv_mma, tOrC_mma, tDCrDC);
    transform(tDCrDC, [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = make_fragment_like<ElementView>(partition_shape_A(tiled_mma2, Shape<_16, _16>{}));
    detail::SM80::convert_fp32_acc_to_tf32_operandA_layout<2, RoundingTF32>(tDCrDC, tOrDC, tiled_mma2, lane_id);
    auto tOrDC_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrDC);
      } else {
        return tOrDC;
      }
    }();

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);
    clear(tOrO);

    auto tOrAinv_mma = [&]() {
      if constexpr (!RoundingTF32) {
        return recast<Element>(tOrAinv);
      } else {
        return tOrAinv;
      }
    }();
    gemm(tiled_mma2, tOrDC_mma, tOrAinv_mma, tOrO);

    auto tOrO_cvt = [&]() {
      if constexpr (!RoundingTF32) {
        return tOrO;
      } else {
        return make_tensor_like<Element>(tOrO);
      }
    }();
    if constexpr (RoundingTF32) {
      transform(tOrO, tOrO_cvt, [](auto v) { return Element(v); });
    }

    // ensure tOsC consumed, tOsC and tOsO are the same buffer
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);

    Tensor tOsO = O_thr_r2s.partition_D(sO);
    Tensor tOrO_cvt_cv = O_thr_r2s.retile_S(tOrO_cvt);
    if (x == 0) {
      copy(O_tiled_r2s, tOrO_cvt_cv, tOsO);
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);
    if (x == 1) {
      Tensor tOrO_red = make_tensor_like(tOrO_cvt);
      Tensor tOsO_s = O_thr_s2r.partition_S(sO);
      Tensor tOrO_red_cv = O_thr_s2r.retile_D(tOrO_red);
      copy(O_tiled_s2r, tOsO_s, tOrO_red_cv);
      transform(tOrO_cvt, tOrO_red, tOrO_cvt, [](auto a, auto b) { return a + b; });
      copy(O_tiled_r2s, tOrO_cvt_cv, tOsO);
    }
  }

 private:
  int wg_sync_named_barrier_id_;
};

// Adapted from
// https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/flat/ampere/collective/flat_collective_inverse.hpp
template <class Element, bool GarbageFilledDiagonal, bool GarbageFilledUpperTriangular>
struct CollectiveInverse {
  // FIXME: precision is not good due to half
  static_assert(std::is_same_v<Element, half> || std::is_same_v<Element, cutlass::half_t>, "only half is implemented");

  CUTE_DEVICE
  CollectiveInverse(int wg_sync_named_barrier_id) : wg_sync_named_barrier_id_(wg_sync_named_barrier_id) {}

  template <typename TensorT>
  CUTE_DEVICE void compute(TensorT&& sT) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 64);
    static_assert(size<1>(L) == 64);

    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    auto t8X8sT = flat_divide(sT, Shape<_8, _8>{});
    if (thread_idx < 64) {  // compute 8x8 inverse on diagnal directly
      compute_diagonal_inverse_NxN<8>(t8X8sT(_, _, thread_idx / 8, thread_idx / 8), thread_idx % 8);
    }

    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);

    auto t16X16sT = flat_divide(sT, Shape<_16, _16>{});
    // four warps for 8x8 -> 16x16
    blockwise_diagonal_inversed_8x8_to_16x16(t16X16sT(_, _, thread_idx / 32, thread_idx / 32));

    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);

    auto t32X32sT = flat_divide(sT, Shape<_32, _32>{});
    if (thread_idx < 64) {  // two warps for 16x16 -> 32x32
      blockwise_diagonal_inversed_16x16_to_32x32(t32X32sT(_, _, thread_idx / 32, thread_idx / 32));
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);

    // one warpgroup for 32x32 -> 64x64
    blockwise_diagonal_inversed_32x32_to_64x64(sT);
  }

 private:
  template <int N, typename TensorT>
  CUTE_DEVICE void compute_diagonal_inverse_NxN(TensorT&& mat, int tid_in_group) {  // group_size = N
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == N);
    static_assert(size<1>(L) == N);

    using ElementCompute = float;

    using CopyOp = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(Element) * 8 * N>, Element>;

    auto load_row = [&](int y) {
      auto row = make_tensor<Element>(Shape<Int<N>>{});
      copy(CopyOp{}, std::forward<TensorT>(mat)(y, _), row);

      auto row_cvt = make_tensor_like<ElementCompute>(row);
      copy(row, row_cvt);

      if constexpr (GarbageFilledDiagonal || GarbageFilledUpperTriangular) {
        CUTE_UNROLL
        for (int i = 0; i < N; ++i) {
          row_cvt(i) = i == y ? 1.0f : (i > y ? 0.0f : row_cvt(i));
        }
      }
      return row_cvt;
    };

    auto store_row = [&](int y, auto row) {
      auto row_cvt = make_tensor_like<Element>(row);
      copy(row, row_cvt);
      copy(CopyOp{}, row_cvt, std::forward<TensorT>(mat)(y, _));
    };

    auto row = load_row(tid_in_group);
#define LOAD(y, x) __shfl_sync(0xffffffff, row(x), y, N)

    CUTE_UNROLL
    for (int src_row = 0; src_row < N - 1; ++src_row) {  // idx of src row to eliminate
      auto row_scale = -row(src_row);                    // scale the src row
      CUTE_UNROLL
      for (int i = 0; i < src_row; ++i) {
        auto src_row_value = LOAD(src_row, i);
        row(i) = tid_in_group > src_row ? row_scale * src_row_value + row(i) : row(i);
      }
      row(src_row) = tid_in_group > src_row ? row_scale : row(src_row);
    }

#undef LOAD

    store_row(tid_in_group, row);
  }

  /*
  blockwise inverse has relation as follows
  inv(| A 0 |)     = |          inv(A)       0  |
      | C D |        | -inv(D)C inv(A)   inv(D) |
  */

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_4x4_to_8x8(TensorT&& mat) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 8);
    static_assert(size<1>(L) == 8);
    auto mat_NxN_2x2 = flat_divide(std::forward<TensorT>(mat), Shape<_4, _4>{});

    // FIXME: implement
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_8x8_to_16x16(TensorT&& mat) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 16);
    static_assert(size<1>(L) == 16);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    auto mat_8x8_2x2 = flat_divide(std::forward<TensorT>(mat), Shape<_8, _8>{});
    using MMA = SM80_16x8x8_F32F16F16F32_TN;
    using TiledMMA = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _8, _8>{}));

    using CopyOpD_S2R = std::conditional_t<is_col_major, SM75_U16x2_LDSM_T, SM75_U32x1_LDSM_N>;
    using CopyOpC_S2R = std::conditional_t<is_col_major, SM75_U32x1_LDSM_N, SM75_U16x2_LDSM_T>;
    using CopyOpA_S2R = std::conditional_t<is_col_major, SM75_U32x1_LDSM_N, SM75_U16x2_LDSM_T>;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = std::conditional_t<is_col_major, SM90_U16x2_STSM_T, SM90_U32x1_STSM_N>;
#else
    using CopyOpO_R2S = UniversalCopy<Element, Element>;
#endif

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma = TiledMMA{};
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, Element>{}, tiled_mma);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, Element>{}, tiled_mma);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, Element>{}, tiled_mma);
    auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, Element>{}, tiled_mma);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_copy = O_tiled_copy.get_thread_slice(lane_id);

    Tensor sDinv = mat_8x8_2x2(_, _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_8x8_2x2(_, _, _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_8x8_2x2(_, _, _0{}, _0{}));
    Tensor sO = mat_8x8_2x2(_, _, _1{}, _0{});

    Tensor sDinv_m_bcast = make_tensor(sDinv.data(), logical_product(sDinv.layout(), Tile<Layout<_2, _0>>{}));
    Tensor sO_m_bcast = make_tensor(sO.data(), logical_product(sO.layout(), Tile<Layout<_2, _0>>{}));

    Tensor tOrDinv = make_fragment_like<Element>(partition_shape_A(tiled_mma, Shape<_16, _8>{}));
    Tensor tOrC = thr_mma.partition_fragment_B(sC);
    Tensor tOrAinv = thr_mma.partition_fragment_B(sAinv);

    Tensor tDCrDC = partition_fragment_C(tiled_mma, Shape<_16, _8>{});  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma, Shape<_16, _8>{});    // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv_m_bcast);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);
    Tensor tOsO = O_thr_copy.partition_D(sO_m_bcast);
    Tensor tOrO_cv = O_thr_copy.retile_S(tOrO);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv(make_coord(_, _0{}), _, _), tOrDinv_cv(make_coord(_, _0{}), _, _));
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    gemm(tiled_mma, tOrDinv, tOrC, tDCrDC);
    transform(tDCrDC(make_coord(_, _0{}), _, _), [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma);

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);

    clear(tOrO);
    gemm(tiled_mma, tOrDC, tOrAinv, tOrO);

    auto tOrO_cv_cvt = make_tensor_like<Element>(tOrO_cv(make_coord(_, _0{}), _, _));
    transform(tOrO_cv(make_coord(_, _0{}), _, _), tOrO_cv_cvt, [](auto v) { return Element(v); });
    copy(O_tiled_copy, tOrO_cv_cvt, tOsO(make_coord(_, _0{}), _, _));
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_16x16_to_32x32(TensorT&& mat) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 32);
    static_assert(size<1>(L) == 32);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    using TileShape = Shape<_16, _16, _16>;
    auto mat_16x16_2x2 = flat_divide(std::forward<TensorT>(mat), select<0, 1>(TileShape{}));

    using MMA = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMMA = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, TileShape{}));

    using CopyOpD_S2R = std::conditional_t<is_col_major, SM75_U16x4_LDSM_T, SM75_U32x2_LDSM_N>;
    using CopyOpC_S2R = std::conditional_t<is_col_major, SM75_U32x2_LDSM_N, SM75_U16x4_LDSM_T>;
    using CopyOpA_S2R = std::conditional_t<is_col_major, SM75_U32x2_LDSM_N, SM75_U16x4_LDSM_T>;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = std::conditional_t<is_col_major, SM90_U16x4_STSM_T, SM90_U32x2_STSM_N>;
#else
    using CopyOpO_R2S = UniversalCopy<Element, Element>;
#endif

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma = TiledMMA{};
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, Element>{}, tiled_mma);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, Element>{}, tiled_mma);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, Element>{}, tiled_mma);
    auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, Element>{}, tiled_mma);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_copy = O_tiled_copy.get_thread_slice(lane_id);

    Tensor sDinv = mat_16x16_2x2(_, _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_16x16_2x2(_, _, _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_16x16_2x2(_, _, _0{}, _0{}));
    Tensor sO = mat_16x16_2x2(_, _, _1{}, _0{});

    Tensor tOrDinv = thr_mma.partition_fragment_A(sDinv);
    Tensor tOrC = thr_mma.partition_fragment_B(sC);
    Tensor tOrAinv = thr_mma.partition_fragment_B(sAinv);

    Tensor tDCrDC = partition_fragment_C(tiled_mma, select<0, 1>(TileShape{}));  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 1>(TileShape{}));    // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);
    Tensor tOsO = O_thr_copy.partition_D(sO);
    Tensor tOrO_cv = O_thr_copy.retile_S(tOrO);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv, tOrDinv_cv);
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    gemm(tiled_mma, tOrDinv, tOrC, tDCrDC);
    transform(tDCrDC, [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma);

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);
    clear(tOrO);
    gemm(tiled_mma, tOrDC, tOrAinv, tOrO);

    auto tOrO_cv_cvt = make_tensor_like<Element>(tOrO_cv);
    transform(tOrO_cv, tOrO_cv_cvt, [](auto v) { return Element(v); });
    copy(O_tiled_copy, tOrO_cv_cvt, tOsO);
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_32x32_to_64x64(TensorT&& mat) {
    constexpr auto L = typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 64);
    static_assert(size<1>(L) == 64);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    auto mat_32x32_2x2 = flat_divide(std::forward<TensorT>(mat), select<0, 1>(Shape<_32, _32>{}));
    auto mat_16x2X16x2_2x2 = logical_divide(mat_32x32_2x2, Shape<_16, _16>{});

    using MMA = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMMA1 = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _16, _32>{}));
    using TiledMMA2 = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _32, _16>{}));

    using CopyOpD_S2R = std::conditional_t<is_col_major, SM75_U16x8_LDSM_T, SM75_U32x4_LDSM_N>;
    using CopyOpC_S2R = std::conditional_t<is_col_major, SM75_U32x4_LDSM_N, SM75_U16x8_LDSM_T>;
    using CopyOpA_S2R = std::conditional_t<is_col_major, SM75_U32x2_LDSM_N, SM75_U16x4_LDSM_T>;
    using CopyOpO_S2R = std::conditional_t<is_col_major, SM75_U16x8_LDSM_T, SM75_U32x4_LDSM_N>;
    using CopyOpO_S2R = std::conditional_t<is_col_major, SM75_U16x8_LDSM_T, SM75_U32x4_LDSM_N>;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = std::conditional_t<is_col_major, SM90_U16x8_STSM_T, SM90_U32x4_STSM_N>;
#else
    using CopyOpO_R2S = UniversalCopy<Element, Element>;
#endif

    int warp_id_in_wg =
        cutlass::canonical_warp_idx() - cutlass::NumWarpsPerWarpGroup * cutlass::canonical_warp_group_idx();
    int x = warp_id_in_wg / 2;
    int y = warp_id_in_wg % 2;

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma1 = TiledMMA1{};
    auto thr_mma1 = tiled_mma1.get_thread_slice(lane_id);

    auto tiled_mma2 = TiledMMA2{};
    auto thr_mma2 = tiled_mma2.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, Element>{}, tiled_mma1);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, Element>{}, tiled_mma1);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, Element>{}, tiled_mma2);
    auto O_tiled_s2r = make_tiled_copy_C(Copy_Atom<CopyOpO_S2R, Element>{}, tiled_mma2);
    auto O_tiled_r2s = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, Element>{}, tiled_mma2);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_s2r = O_tiled_s2r.get_thread_slice(lane_id);
    auto O_thr_r2s = O_tiled_r2s.get_thread_slice(lane_id);

    Tensor sDinv = mat_16x2X16x2_2x2(make_coord(_, y), _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_16x2X16x2_2x2(_, make_coord(_, x), _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_16x2X16x2_2x2(make_coord(_, x), _, _0{}, _0{}));  // NOTE: not y!
    Tensor sO = mat_16x2X16x2_2x2(make_coord(_, y), _, _1{}, _0{});  // needs cross-warp reduction

    Tensor tOrDinv = thr_mma1.partition_fragment_A(sDinv);
    Tensor tOrC = thr_mma1.partition_fragment_B(sC);
    Tensor tOrAinv = thr_mma2.partition_fragment_B(sAinv);

    Tensor tDCrDC = partition_fragment_C(tiled_mma1, Shape<_16, _16>{});  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma2, Shape<_16, _32>{});    // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv, tOrDinv_cv);
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    gemm(tiled_mma1, tOrDinv, tOrC, tDCrDC);
    transform(tDCrDC, [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma2);

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);
    clear(tOrO);
    gemm(tiled_mma2, tOrDC, tOrAinv, tOrO);

    auto tOrO_cvt = make_tensor_like<Element>(tOrO);
    transform(tOrO, tOrO_cvt, [](auto v) { return Element(v); });

    // ensure tOsC consumed, tOsC and tOsO are the same buffer
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);

    Tensor tOsO = O_thr_r2s.partition_D(sO);
    Tensor tOrO_cvt_cv = O_thr_r2s.retile_S(tOrO_cvt);
    if (x == 0) {
      copy(O_tiled_r2s, tOrO_cvt_cv, tOsO);
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, wg_sync_named_barrier_id_);
    if (x == 1) {
      Tensor tOrO_red = make_tensor_like(tOrO_cvt);
      Tensor tOsO_s = O_thr_s2r.partition_S(sO);
      Tensor tOrO_red_cv = O_thr_s2r.retile_D(tOrO_red);
      copy(O_tiled_s2r, tOsO_s, tOrO_red_cv);
      transform(tOrO_cvt, tOrO_red, tOrO_cvt, [](auto a, auto b) { return a + b; });
      copy(O_tiled_r2s, tOrO_cvt_cv, tOsO);
    }
  }

 private:
  int wg_sync_named_barrier_id_;
};

}  // namespace kerutils
