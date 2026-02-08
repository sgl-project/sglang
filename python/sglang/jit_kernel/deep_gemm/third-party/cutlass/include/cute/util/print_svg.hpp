/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <cute/config.hpp>           // CUTE_HOST_DEVICE

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/layout.hpp>
#include <cute/tensor_impl.hpp>

namespace cute
{

////////////////////////////////
// Common SVG Color utilities //
////////////////////////////////

struct TSVGColor_White {
  CUTE_HOST_DEVICE char const*
  operator()(int idx) const {
    return "255,255,255";
  }
};

struct TSVGColor_BWx8 {
  CUTE_HOST_DEVICE char const*
  operator()(int idx) const {
    static char const* color_map[8] = {"255,255,255", "230,230,230", "205,205,205", "180,180,180",
                                       "155,155,155", "130,130,130", "105,105,105", "080,080,080"};
    return color_map[idx % 8];
  }
};

struct SVGColor_TV {
  CUTE_HOST_DEVICE char const*
  operator()(int tid, int vid) const {
    static char const* color_map[8] = {"175,175,255", "175,255,175", "255,255,175", "255,175,175",
                                       "210,210,255", "210,255,210", "255,255,210", "255,210,210"};
    return color_map[tid % 8];
  }
};

/////////////////////
// MMA Atom to SVG //
/////////////////////

namespace detail {

template <class LayoutC, class LayoutA, class LayoutB, class Tile_MNK,
          class SVGColorFn = SVGColor_TV>
CUTE_HOST_DEVICE
void
print_svg_mma(LayoutC const& C,
              LayoutA const& A,
              LayoutB const& B,
              Tile_MNK const& tile_mnk,
              SVGColorFn color = {})  // lambda(tid,vid) -> SVG color string
{
  CUTE_STATIC_ASSERT_V(rank(C) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(A) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(B) == Int<2>{});

  auto [M, N, K] = product_each(shape(tile_mnk));

  int cell_size = 20;

  int page_width  = (K + N + 2) * cell_size;
  int page_height = (K + M + 2) * cell_size;

  // Commented print
  printf("<!--  Tile: "); print(tile_mnk); printf("  -->\n");
  printf("<!--     A: "); print(A);        printf("  -->\n");
  printf("<!--     B: "); print(B);        printf("  -->\n");
  printf("<!--     C: "); print(C);        printf("  -->\n");

  // SVG Header
  printf("<svg width=\"100%%\" height=\"100%%\" viewBox=\"0 0 %d %d\" "
         "preserveAspectRatio=\"xMidYMid meet\" "
         "xmlns=\"http://www.w3.org/2000/svg\">\n",
         page_width, page_height);

  Tensor filled = make_tensor<bool>(make_shape(M, N, K));
  clear(filled);

  // --- Draw C ---
  for (int tid = 0; tid < size<0>(C); ++tid) {
    for (int vid = 0; vid < size<1>(C); ++vid) {
      auto [m, n] = C(tid, vid);
      if (!filled(m, n, 0)) {
        filled(m, n, 0) = true;

        int x = (n + K + 2) * cell_size;
        int y = (m + K + 2) * cell_size;

        printf("<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
              "fill=\"rgb(%s)\" stroke=\"black\"/>\n",
              x, y, cell_size, cell_size, color(tid,vid));
        printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
              "alignment-baseline=\"central\" font-size=\"8\">T%d</text>\n",
              x + cell_size/2, y + 1*cell_size/4, tid);
        printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
              "alignment-baseline=\"central\" font-size=\"8\">V%d</text>\n",
              x + cell_size/2, y + 3*cell_size/4, vid);
      }
    }
  }

  clear(filled);

  // --- Draw A ---
  for (int tid = 0; tid < size<0>(A); ++tid) {
    for (int vid = 0; vid < size<1>(A); ++vid) {
      auto [m, k] = A(tid, vid);
      if (!filled(m, 0, k)) {
        filled(m, 0, k) = true;

        int x = (k     + 1) * cell_size;
        int y = (m + K + 2) * cell_size;

        printf("<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
              "fill=\"rgb(%s)\" stroke=\"black\" />\n",
              x, y, cell_size, cell_size, color(tid,vid));
        printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
              "alignment-baseline=\"central\" font-size=\"8\">T%d</text>\n",
              x + cell_size/2, y + 1*cell_size/4, tid);
        printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
              "alignment-baseline=\"central\" font-size=\"8\">V%d</text>\n",
              x + cell_size/2, y + 3*cell_size/4, vid);
      }
    }
  }

  // A labels
  for (int m =  0, k = -1; m < M; ++m) {
    int x = (k     + 1) * cell_size;
    int y = (m + K + 2) * cell_size;
    printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
           "alignment-baseline=\"central\" font-size=\"12\">%d</text>\n",
           x + cell_size/2, y + cell_size/2, m);
  }
  for (int m = -1, k =  0; k < K; ++k) {
    int x = (k     + 1) * cell_size;
    int y = (m + K + 2) * cell_size;
    printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
           "alignment-baseline=\"central\" font-size=\"12\">%d</text>\n",
           x + cell_size/2, y + cell_size/2, k);
  }

  clear(filled);

  // --- Draw B ---
  for (int tid = 0; tid < size<0>(B); ++tid) {
    for (int vid = 0; vid < size<1>(B); ++vid) {
      auto [n, k] = B(tid, vid);
      if (!filled(0, n, k)) {
        filled(0, n, k) = true;

        int x = (n + K + 2) * cell_size;
        int y = (k     + 1) * cell_size;

        printf("<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
              "fill=\"rgb(%s)\" stroke=\"black\" />\n",
              x, y, cell_size, cell_size, color(tid,vid));
        printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
              "alignment-baseline=\"central\" font-size=\"8\">T%d</text>\n",
              x + cell_size/2, y + 1*cell_size/4, tid);
        printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
              "alignment-baseline=\"central\" font-size=\"8\">V%d</text>\n",
              x + cell_size/2, y + 3*cell_size/4, vid);
      }
    }
  }

  // B labels
  for (int n =  0, k = -1; n < N; ++n) {
    int x = (n + K + 2) * cell_size;
    int y = (k     + 1) * cell_size;
    printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
           "alignment-baseline=\"central\" font-size=\"12\">%d</text>\n",
           x + cell_size/2, y + cell_size/2, n);
  }
  for (int n = -1, k =  0; k < K; ++k) {
    int x = (n + K + 2) * cell_size;
    int y = (k     + 1) * cell_size;
    printf("<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
           "alignment-baseline=\"central\" font-size=\"12\">%d</text>\n",
           x + cell_size/2, y + cell_size/2, k);
  }

  // SVG footer
  printf("</svg>\n");
}

} // end namespace detail

// MMA Atom to SVG
template <class... Args, class SVGColorFn = SVGColor_TV>
CUTE_HOST_DEVICE
void
print_svg(MMA_Atom<Args...> const& mma_atom,
          SVGColorFn color = {})             // lambda(thr_idx,val_idx) -> svg color string
{
  print_svg(make_tiled_mma(mma_atom));
}

// TiledMMA to SVG
template <class... Args, class SVGColorFn = SVGColor_TV>
CUTE_HOST_DEVICE
void
print_svg(TiledMMA<Args...> const& mma,
          SVGColorFn color = {})             // lambda(thr_idx,val_idx) -> svg color string
{
  auto tile_mnk = tile_shape(mma);

  Tensor refC = make_identity_tensor(select<0,1>(tile_mnk));
  Tensor tensorC_TV = composition(refC, mma.get_layoutC_TV());

  Tensor refA = make_identity_tensor(select<0,2>(tile_mnk));
  Tensor tensorA_TV = composition(refA, mma.get_layoutA_TV());

  Tensor refB = make_identity_tensor(select<1,2>(tile_mnk));
  Tensor tensorB_TV = composition(refB, mma.get_layoutB_TV());

  detail::print_svg_mma(tensorC_TV, tensorA_TV, tensorB_TV, tile_mnk, color);
}

} // end namespace cute
