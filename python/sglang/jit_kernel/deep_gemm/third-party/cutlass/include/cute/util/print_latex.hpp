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

///////////////////////////////////////
// Common LaTeX TikZ Color utilities //
///////////////////////////////////////

struct TikzColor_White {
  CUTE_HOST_DEVICE char const*
  operator()(int idx) const {
    return "white";
  }
};

struct TikzColor_BWx8 {
  CUTE_HOST_DEVICE char const*
  operator()(int idx) const {
    static char const* color_map[8] = {"black!00", "black!40", "black!20", "black!60",
                                       "black!10", "black!50", "black!30", "black!70"};
    return color_map[idx % 8];
  }
};

struct TikzColor_TV {
  CUTE_HOST_DEVICE char const*
  operator()(int tid, int vid) const {
    static char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                                       "{rgb,255:red,175;green,255;blue,175}",
                                       "{rgb,255:red,255;green,255;blue,175}",
                                       "{rgb,255:red,255;green,175;blue,175}",
                                       "{rgb,255:red,210;green,210;blue,255}",
                                       "{rgb,255:red,210;green,255;blue,210}",
                                       "{rgb,255:red,255;green,255;blue,210}",
                                       "{rgb,255:red,255;green,210;blue,210}"};
    return color_map[tid % 8];
  }
};

/////////////////////////////
// Layout 2D to LaTeX TikZ //
/////////////////////////////

template <class LayoutA, class TikzColorFn = TikzColor_BWx8>
CUTE_HOST_DEVICE
void
print_latex(LayoutA const& layout_a,   // (m,n) -> idx
            TikzColorFn color = {})    // lambda(idx) -> tikz color string
{
  CUTE_STATIC_ASSERT_V(rank(layout_a) <= Int<2>{});
  auto layout = append<2>(layout_a, Layout<_1,_0>{});

  // Commented print(layout)
  printf("%% Layout: "); print(layout); printf("\n");
  // Header
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  auto [M, N] = product_each(shape(layout));

  // Layout
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int idx = layout(m,n);
      printf("\\node[fill=%s] at (%d,%d) {%d};\n",
             color(idx), m, n, idx);
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n",
         int(M), int(N));
  // Labels
  for (int m =  0, n = -1; m < M; ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, m);
  }
  for (int m = -1, n =  0; n < N; ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, n);
  }

  // Footer
  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

template <class SwizzleFn, int B, class Layout, class TikzColorFn = TikzColor_BWx8>
CUTE_HOST_DEVICE
void
print_latex(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout,
            TikzColorFn color = {})    // lambda(idx) -> tikz color string)
{
  print_latex(as_position_independent_swizzle_layout(layout), color);
}

///////////////////////////////
// LayoutTV 2D to LaTeX TikZ //
///////////////////////////////

template <class LayoutTV, class Tile_MN,
          class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex_tv(LayoutTV const& layout_tv,   // (t,v) -> m,n coord
               Tile_MN  const& tile_mn,     // (M,N)
               TikzColorFn     color = {})  // (t,v) -> color
{
  CUTE_STATIC_ASSERT_V(rank(layout_tv) == Int<2>{});

  // Commented prints
  printf("%% Layout TV: "); print(layout_tv); printf("\n");
  // Header
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  auto [M, N] = product_each(shape(tile_mn));
  Tensor filled = make_tensor<bool>(make_shape(M, N));
  clear(filled);

  // Layout
  for (int tid = 0; tid < size<0>(layout_tv); ++tid) {
    for (int vid = 0; vid < size<1>(layout_tv); ++vid) {
      auto [m, n] = layout_tv(tid, vid);
      if (not filled(m, n)) {
        filled(m, n) = true;
        printf("\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
                color(tid, vid),
                int(m), int(n),
                tid, vid);
      }
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n", int(M), int(N));
  // Labels
  for (int m = 0, n = -1; m < M; ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, m);
  }
  for (int n = 0, m = -1; n < N; ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, n);
  }
  // Footer
  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

////////////////////////////
// MMA Atom to LaTeX TikZ //
////////////////////////////

namespace detail {

template <class LayoutC, class LayoutA, class LayoutB, class Tile_MNK,
          class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex_mma(LayoutC const& C,         // (tid,vid) -> (m,n) coord
                LayoutA const& A,         // (tid,vid) -> (m,k) coord
                LayoutB const& B,         // (tid,vid) -> (n,k) coord
                Tile_MNK const& tile_mnk, // (M,N,K)
                TikzColorFn color = {})   // lambda(tid,vid) -> tikz color string
{
  CUTE_STATIC_ASSERT_V(rank(C) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(A) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(B) == Int<2>{});

  // Commented prints
  printf("%% LayoutC: "); print(C); printf("\n");
  printf("%% LayoutA: "); print(A); printf("\n");
  printf("%% LayoutB: "); print(B); printf("\n");
  // Header
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  auto [M, N, K] = product_each(shape(tile_mnk));
  Tensor filled = make_tensor<bool>(make_shape(M, N, K));
  clear(filled);

  // C starting at 0,0
  for (int tid = 0; tid < size<0>(C); ++tid) {
    for (int vid = 0; vid < size<1>(C); ++vid) {
      auto [m, n] = C(tid, vid);
      if (not filled(m, n, 0)) {
        filled(m, n, 0) = true;
        printf("\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
               color(tid, vid),
               int(m), int(n),
               tid, vid);
      }
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
          0, 0, int(M), int(N));

  clear(filled);

  // A starting at 0,-K-1
  for (int tid = 0; tid < size<0>(A); ++tid) {
    for (int vid = 0; vid < size<1>(A); ++vid) {
      auto [m, k] = A(tid, vid);
      if (not filled(m, 0, k)) {
        filled(m, 0, k) = true;
        printf("\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
               color(tid, vid),
               int(m), int(k-K-1),
               tid, vid);
      }
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
         0, -int(K)-1, int(M), -1);
  // A labels
  for (int m =  0, k = -1; m < M; ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, int(k-K-1), m);
  }
  for (int m = -1, k =  0; k < K; ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, int(k-K-1), k);
  }

  clear(filled);

  // B starting at -K-1,0
  for (int tid = 0; tid < size<0>(B); ++tid) {
    for (int vid = 0; vid < size<1>(B); ++vid) {
      auto [n, k] = B(tid, vid);
      if (not filled(0, n, k)) {
        filled(0, n, k) = true;
        printf("\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
               color(tid, vid),
               int(k)-int(K)-1, int(n),
               tid, vid);
      }
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
         -int(K)-1, 0, -1, int(N));
  // B labels
  for (int n =  0, k = -1; n < N; ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", int(k-K-1), n, n);
  }
  for (int n = -1, k =  0; k < K; ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", int(k-K-1), n, k);
  }

  // Footer
  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

} // end namespace detail

// MMA Atom to LaTeX TikZ
template <class... Args, class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex(MMA_Atom<Args...> const& mma_atom,
            TikzColorFn color = {})             // lambda(thr_idx,val_idx) -> tikz color string
{
  print_latex(make_tiled_mma(mma_atom));
}

// TiledMMA to LaTeX TikZ
template <class... Args, class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex(TiledMMA<Args...> const& mma,
            TikzColorFn color = {})             // lambda(thr_idx,val_idx) -> tikz color string
{
  auto tile_mnk = tile_shape(mma);

  Tensor refC = make_identity_tensor(select<0,1>(tile_mnk));
  Tensor tensorC_TV = composition(refC, mma.get_layoutC_TV());

  Tensor refA = make_identity_tensor(select<0,2>(tile_mnk));
  Tensor tensorA_TV = composition(refA, mma.get_layoutA_TV());

  Tensor refB = make_identity_tensor(select<1,2>(tile_mnk));
  Tensor tensorB_TV = composition(refB, mma.get_layoutB_TV());

  detail::print_latex_mma(tensorC_TV, tensorA_TV, tensorB_TV, tile_mnk, color);
}

////////////////////////////
// CopyAtom to LaTeX TikZ //
////////////////////////////

namespace detail {

// Generic TV Layout to LaTeX TikZ
template <class LayoutS_TV, class LayoutD_TV, class Tile_MN,
          class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex_copy(LayoutS_TV const& S,            // (t,v) -> m,n coord
                 LayoutD_TV const& D,            // (t,v) -> m,n coord
                 Tile_MN const& tile_mn,         // (M,N)
                 TikzColorFn       color = {})   // (t,v) -> color
{
  CUTE_STATIC_ASSERT_V(rank(S) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<2>{});

  // Commented prints
  printf("%% Layout S TV: "); print(S); printf("\n");
  printf("%% Layout D TV: "); print(D); printf("\n");

  // Header
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  auto [M, N] = product_each(shape(tile_mn));
  Tensor filled = make_tensor<bool>(make_shape(M, N));
  clear(filled);

  // S starting at 0,0
  for (int tid = 0; tid < size<0>(S); ++tid) {
    for (int vid = 0; vid < size<1>(S); ++vid) {
      auto [m, n] = S(tid, vid);
      if (not filled(m, n)) {
        filled(m, n) = true;
        printf("\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color(tid, vid),
              int(m), int(n),
              tid, vid);
      }
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
         0, 0, int(M), int(N));
  // S Labels
  for (int m =  0, n = -1; m < M; ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, m);
  }
  for (int m = -1, n =  0; n < N; ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, n);
  }

  clear(filled);

  // D starting at 0,N+3
  for (int tid = 0; tid < size<0>(D); ++tid) {
    for (int vid = 0; vid < size<1>(D); ++vid) {
      auto [m, n] = D(tid, vid);
      if (not filled(m, n)) {
        filled(m, n) = true;
        printf("\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color(tid, vid),
              int(m), int(n) + int(N) + 3,
              tid, vid);
      }
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
         0, int(N) + 3, int(M), int(N) + int(N) + 3);
  // D Labels
  for (int m =  0, n = N; m < M; ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, int(n+N+3), m);
  }
  for (int m = -1, n = 0; n < N; ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, int(n+N+3), n);
  }

  // Footer
  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

} // end namespace detail

// TiledCopy to LaTeX TikZ
template <class... Args, class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex(TiledCopy<Args...> const& copy,
            TikzColorFn color = {})              // lambda(tid,vid) -> tikz color string
{
  auto tiler_mn = typename TiledCopy<Args...>::Tiler_MN{};
  auto tile_mn = product_each(shape(logical_divide(make_layout(Shape<_1,_1>{}), tiler_mn)));  // tile_shape

  Tensor refS = make_identity_tensor(tile_mn);
  Tensor layoutS_TV = copy.tidfrg_S(refS)(_,_,Int<0>{});

  Tensor refD = make_identity_tensor(tile_mn);
  Tensor layoutD_TV = copy.tidfrg_D(refD)(_,_,Int<0>{});

  detail::print_latex_copy(layoutS_TV, layoutD_TV, tile_mn, color);
}

} // end namespace cute
