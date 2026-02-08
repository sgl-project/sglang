/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Problem visitor for grouped Rank2K operations.

    This problem visitor is specialized for Rank2K operations, for which matrix C is upper/lower
    triangular. Using a problem visitor designed for GEMMs for Rank2K problems is inefficient
    because threadblocks will be frequently assigned to tiles that exit early (e.g., due to
    being assigned to a tile in the upper-triangular portion of a lower-triangular problem).
    This can lead to load imbalance among threadblocks, as the GEMM-based scheduler
    assigns all threadblocks to nearly the same number of tiles, regardless of whether
    those tiles exit early.

    Consider an example of a group of four Rank2Ks with matrix C consisting of a grid of 2x2 tiles.
    Consider a grid of 8 threadblocks. The default GEMM scheduler will assign threadblocks to
    tiles in the following order:
        Rank2K 0      Rank2K 1       Rank2K 2      Rank2K 3
          0  1          4  5           0  1          4  5
          2  3          6  7           2  3          6  7
    Assuming that the problems are lower triangular, blocks 1 and 5 are continuously assigned
    to inactive tiles.

    This problem visitor aims to assign threadblocks to only those tiles which are in the
    upper/lower triangular portion of a given problem. Using the example above, the resulting
    assignment would be:
        Rank2K 0      Rank2K 1       Rank2K 2      Rank2K 3
          0  -          3  -           6  -          1  -
          1  2          4  5           7  0          2  3

    Achieving the schedule above requires a mapping from threadblock ID to tile coordinates (i, j).
    We will illustrate this by mapping on a lower-triangular matrix with a 3x3 grid. We first
    calculate row and column indices assuming one-indexed rows, tiles, and threadblock IDs, and
    then subtract one to convert to zero-indexed.
                      Col 1   Col 2   Col 3
                     ----------------------
              Row 1 |   1      -       -
              Row 2 |   2      3       -
              Row 3 |   4      5       6

    We next outline this mapping, borrowing from: https://stackoverflow.com/a/40954159

    Calculating row i given threadblock ID t
    ----------------------------------------
    For a given row i, all threadblock IDs t in that row satisfy the following:
          t <= 1 + 2 + 3 + ... + (i-1) + i

    The closed-form equation for the right-hand side is: i(i+1)/2.
    Using this, we can solve for i given t:
          t  <= i(i+1)/2
          2t <= i^2 + i
          2t <= i^2 + i + 0.25 - 0.25
          2t + 0.25 <= i^2 + i + 0.25
          2t + 0.25 <= (i + 0.5)^2
          sqrt(2t + 0.25) - 0.5 <= i

    To account for fractional values, we set:
          i = ceil(sqrt(2t + 0.25) - 0.5)

    To turn this into a zero-indexed row and work with zero-indexed t, we perform:
          i = ceil(sqrt(2(t+1) + 0.25) - 0.5) - 1
            = ceil(sqrt(2t + 2.25) - 0.5) - 1

    Calculating column j given threadblock ID t and row i
    -----------------------------------------------------
    For a given row i, all threadblock IDs t in that row also satisfy the following:
          t > 1 + 2 + 3 + ... + (i-2) + (i-1)
      --> t > i(i-1)/2

    Threadblock IDs within a given row are sequential, so the one-indexed column ID
    for one-indexed threadblock ID t and row i is:
          j = t - (i(i-1)/2)

    The zero-indexed version becomes:
          j = (t+1) - (i(i+1)/2) -1
            = t - (i(i+1)/2)

    Accounting for non-square grids
    -------------------------------
    Though the overall output problem size for Rank2K problems is guranteed to be square, the
    grids used in computing may not be square due to using non-square threadblock shapes. For
    example, a threadblock shape of 64x32 operating on a problem of output size 128x128 would
    result in a grid of 2x4 tiles.

    This case can be handled by noting that the output resembles a square grid of 2x2 "macro tiles"
    each of which contains 2 "true tiles." We can thus first map a threadblock ID to its "macro tile"
    using the equations above, and then map it to the "true tile" within its "macro tile." In the example
    of a 2x4 grid, this mapping would look as follows:
        "Macro grid"           "True grid"
       {0, 1}    -            0   1   -   -
       {2, 3}  {4, 5}         2   3   4   5

    A zero-indexed threadblock ID t is mapped to its "macro tile ID" t_macro as:
      t_macro = t // r
    Where r is the ratio of the maximum dimension of the grid to the minimum dimension of the grid
    (i.e., r = 4 / 2 = 2 in the previous example).

    One uses t_macro and the calculations above to find the row and column in the square matrix to
    obtain i_macro and j_macro (zero-indexed). The mapping from (i_macro, j_macro) --> (i, j)
    is simply the following:
        if (ThreadblockShape::M > ThreadblockShape::N):
            r = ThreadblockShape::M / ThreadblockShape::N
            i = i_macro
            j = (j_macro * r) + (t % r)
        elif (ThreadblockShape::M < ThreadblockShape::N):
            r = ThreadblockShape::N / ThreadblockShape::M
            i = (i_macro * r) + (t % r)
            j = j_macro
        else:
            i = i_macro
            j = j_macro

    Handling cases with grid dimensions that aren't multiples of eachother
    ----------------------------------------------------------------------
    Even though threadblock shapes M and N are typically multiples of one another, the grid
    for a given problem may not have dimensions of the same ratio as that of the threadblock.
    For example, a problem of size 132x132 using a threadblock of shape 64x32 will result
    in a grid of 3x5 tiles. In this case, there is not an integer number of "true tiles"
    per "macro tile."

    When this scenario arises, we simply pad the larger dimension of the grid such that
    there are an integer number of "true tiles" per "macro tile." Thus, the 3x5 grid in
    the example above will be treated as a 3x6 grid. Row and column positions for each
    tile are calculated as above. Any threadblocks that map to tiles that are outside the
    problem range or upper/lower triangular portion (e.g., (2, 5)) will exit early from
    this problem and may proceed to the next problem in the group.

    Handling upper-triangular matrices
    ----------------------------------
    The only modification needed for upper-triangular matrices is to swap i_macro and j_macro
    in the calculations above.
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/gemm/kernel/grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

namespace detail {
/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Helpers for calculating offsets for Rank2K problem visitor. These helpers specifically pertain
// to the conversion from "macro tiles" to "true tiles" in the description above.
//
template <
  typename ThreadblockShape,
  typename Enable = void
>
struct Rank2KGroupedProblemVisitorOffsetHelper;

// Partial specialization for the case where threadblock shape M > threadblock shape N
template <
  typename ThreadblockShape
>
struct Rank2KGroupedProblemVisitorOffsetHelper<
    ThreadblockShape,
    typename platform::enable_if< (ThreadblockShape::kM > ThreadblockShape::kN) >::type
> {
  static_assert(ThreadblockShape::kM % ThreadblockShape::kN == 0,
             "Rank2KGroupedProblemVisitor with threadblock shape M > threadblock shape N "
             "requires that threadblock shape M be a multiple of threadblock shape N.");

  static int32_t const kThreadblockSkewRatio = ThreadblockShape::kM / ThreadblockShape::kN;

  CUTLASS_HOST_DEVICE
  static int32_t min_dim(cutlass::gemm::GemmCoord grid) {
    return grid.m();
  }

  CUTLASS_HOST_DEVICE
  static int32_t macro_row_to_row(int32_t row, int32_t threadblock_id) {
    return row;
  }

  CUTLASS_HOST_DEVICE
  static int32_t macro_col_to_col(int32_t col, int32_t threadblock_id) {
    return (col * kThreadblockSkewRatio) + (threadblock_id % kThreadblockSkewRatio);
  }
};

// Partial specialization for the case where threadblock shape M < threadblock shape N
template <
  typename ThreadblockShape
>
struct Rank2KGroupedProblemVisitorOffsetHelper<
    ThreadblockShape,
    typename platform::enable_if< (ThreadblockShape::kM < ThreadblockShape::kN) >::type
> {

  static_assert(ThreadblockShape::kN % ThreadblockShape::kM == 0,
             "Rank2KGroupedProblemVisitor with threadblock shape M < threadblock shape N "
             "requires that threadblock shape N be a multiple of threadblock shape M.");

  static int32_t const kThreadblockSkewRatio = ThreadblockShape::kN / ThreadblockShape::kM;

  CUTLASS_HOST_DEVICE
  static int32_t min_dim(cutlass::gemm::GemmCoord grid) {
    return grid.n();
  }

  CUTLASS_HOST_DEVICE
  static int32_t macro_row_to_row(int32_t row, int32_t threadblock_id) {
    return (row * kThreadblockSkewRatio) + (threadblock_id % kThreadblockSkewRatio);
  }

  CUTLASS_HOST_DEVICE
  static int32_t macro_col_to_col(int32_t col, int32_t threadblock_id) {
    return col;
  }
};

// Partial specialization for the case where threadblock shape M == threadblock shape N
// In this case, macro tiles are equivalent to true tiles, so the conversions are
// identity functions.
template <
  typename ThreadblockShape
>
struct Rank2KGroupedProblemVisitorOffsetHelper<
    ThreadblockShape,
    typename platform::enable_if< (ThreadblockShape::kM == ThreadblockShape::kN) >::type
> {

  static int32_t const kThreadblockSkewRatio = 1;

  CUTLASS_HOST_DEVICE
  static int32_t min_dim(cutlass::gemm::GemmCoord grid) {
    return grid.m();
  }

  CUTLASS_HOST_DEVICE
  static int32_t macro_row_to_row(int32_t row, int32_t threadblock_id) {
    return row;
  }

  CUTLASS_HOST_DEVICE
  static int32_t macro_col_to_col(int32_t col, int32_t threadblock_id) {
    return col;
  }
};

// Helper for correctly representing problem sizes in grouped kernels 
template <typename ThreadblockShape>
struct Rank2KGroupedProblemSizeHelper {
  using OffsetHelper = Rank2KGroupedProblemVisitorOffsetHelper<ThreadblockShape>;

  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem) {
    return cutlass::gemm::GemmCoord(
      ((problem.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM),
      ((problem.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN),
      1);
  }

  CUTLASS_HOST_DEVICE
  static int32_t tile_count(const cutlass::gemm::GemmCoord& grid) {
    // Return the number of tiles at or below the diagonal (or at and above
    // for mode kUpper). We do this by first calculating this value assuming
    // we have a square matrix of tiles of size `dim x dim` where `dim` is the
    // minimum among {grid.m(), grid.n()}. We then multiply the resulting value
    // by OffsetHelper::kThreadblockSkewRatio to account for cases in which there
    // are more tiles in one dimension than the other.
    int32_t dim = OffsetHelper::min_dim(grid);
    int32_t tiles_on_diagonal = dim;
    int32_t tiles_below_diagonal = ((dim * (dim - 1)) / 2);
    return (tiles_on_diagonal + tiles_below_diagonal) * OffsetHelper::kThreadblockSkewRatio;
  }

  CUTLASS_HOST_DEVICE
  static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem) {}
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Default problem visitor for fill modes kUpper and kLower.
//
template <typename ThreadblockShape,
          GroupScheduleMode GroupScheduleMode_,
          int PrefetchTileCount,
          int ThreadCount,
          cutlass::FillMode FillModeC>
struct Rank2KGroupedProblemVisitor : public GroupedProblemVisitor<
                                              detail::Rank2KGroupedProblemSizeHelper<ThreadblockShape>,
                                              ThreadblockShape,
                                              GroupScheduleMode_,
                                              PrefetchTileCount,
                                              ThreadCount> {

  static cutlass::FillMode const kFillModeC = FillModeC;

  static_assert(kFillModeC == cutlass::FillMode::kLower || kFillModeC == cutlass::FillMode::kUpper,
              "Default Rank2KGroupedProblemVisitor requires fill mode of kLower or kUpper.");

  using ProblemSizeHelper = detail::Rank2KGroupedProblemSizeHelper<ThreadblockShape>;
  using Base = GroupedProblemVisitor<ProblemSizeHelper,
                                     ThreadblockShape,
                                     GroupScheduleMode_,
                                     PrefetchTileCount,
                                     ThreadCount>;
  using OffsetHelper = typename ProblemSizeHelper::OffsetHelper;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;

  //
  // Methods
  //
  CUTLASS_DEVICE
  Rank2KGroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, shared_storage_, block_idx)
  {}

  CUTLASS_DEVICE
  cutlass::gemm::GemmCoord threadblock_offset(int32_t threadblock_id) const {
    int32_t macro_id = threadblock_id / OffsetHelper::kThreadblockSkewRatio;
    int32_t macro_row = ceil(cutlass::fast_sqrt((2*macro_id) + 2.25) - 0.5) - 1;
    int32_t macro_col = macro_id - (((macro_row+1) * macro_row)/2);

    if (kFillModeC == cutlass::FillMode::kUpper) {
      cutlass::swap(macro_row, macro_col);
    }

    int32_t row = OffsetHelper::macro_row_to_row(macro_row, threadblock_id);
    int32_t col = OffsetHelper::macro_col_to_col(macro_col, threadblock_id);

    return cutlass::gemm::GemmCoord(row, col, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
