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
/*! \file
  \brief GETT command line parser to gather semantic modes, their stride order, and extents.
*/
#pragma once

#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

#include "cutlass/util/command_line.h"

namespace cutlass {

// Output shortcuts
std::ostream& operator<<(std::ostream& os, std::vector<char> data) {
  for (auto& a : data) os << a;
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> data) {
  for (auto& a : data) os << a << " ";
  return os;
}

struct GettCommandLine {
  struct GettProblem {
    using extent_type = int;
    using stride_type = int64_t;

    // Row modes: appear in A and C/D
    std::vector<extent_type> M;
    std::vector<stride_type> ldAm;
    std::vector<stride_type> ldCm;

    // Column modes: appear in B and C/D
    std::vector<extent_type> N;
    std::vector<stride_type> ldBn;
    std::vector<stride_type> ldCn;  

    // Reduction modes: appear in A and B
    std::vector<extent_type> K;
    std::vector<stride_type> ldAk;
    std::vector<stride_type> ldBk;

    // Batch modes: appear in all in/out tensors
    std::vector<extent_type> L;
    std::vector<stride_type> ldAl;
    std::vector<stride_type> ldBl;
    std::vector<stride_type> ldCl;
  };

  static GettProblem
  parse(int argc, char const* argv[], bool parse_verbose = false) {
    using extent_type = typename GettProblem::extent_type;
    using stride_type = typename GettProblem::stride_type;

    cutlass::CommandLine cmd(argc, argv);

    // modeA
    std::vector<char> a_mode;
    cmd.get_cmd_line_arguments("modeA", a_mode);

    // modeB
    std::vector<char> b_mode;
    cmd.get_cmd_line_arguments("modeB", b_mode);

    // modeC
    std::vector<char> c_mode;
    cmd.get_cmd_line_arguments("modeC", c_mode);


    // mode_sizes
    std::map<char,extent_type> mode_size;
    // First, initialize all modes in a, b, c to make sure they're in map
    for (char a : a_mode) mode_size[a] = 1;
    for (char b : b_mode) mode_size[b] = 1;
    for (char c : c_mode) mode_size[c] = 1;

    // Then, overwrite the ones in -extent
    std::vector<std::pair<std::string, std::string> > extent_tokens;
    cmd.get_cmd_line_argument_pairs("extents", extent_tokens);
    for (auto e : extent_tokens) {
      if (std::get<0>(e).size() > 1) {
        std::cerr << "ERROR: Mode name must only be 1 character long.\n";
        print_usage();
        exit(1);
      }
      char label = std::get<0>(e)[0];
      int  size  = std::stoi(std::get<1>(e));
      mode_size[label] = size;
    }

    // Print out symbolic modes and their extents
    if (parse_verbose) {
      std::cout << "C_" << c_mode << " = A_" << a_mode << " * B_" << b_mode << "\n";
      for (auto e : mode_size) std::cout << "     " << std::get<0>(e) << " : " << std::get<1>(e) << "\n";
    }

    //
    // Collect/Compute strides
    //

    std::map<char,stride_type> mode_ldA;
    std::map<char,stride_type> mode_ldB;
    std::map<char,stride_type> mode_ldC;

    {
      stride_type current;

      current = 1;
      for (char a : a_mode) { mode_ldA[a] = current; current *= mode_size[a]; }

      current = 1;
      for (char b : b_mode) { mode_ldB[b] = current; current *= mode_size[b]; }

      current = 1;
      for (char c : c_mode) { mode_ldC[c] = current; current *= mode_size[c]; }
    }

    //
    // Collect mode categories
    //

    std::vector<char> row_mode;  // rows
    std::vector<char> col_mode;  // columns
    std::vector<char> red_mode;  // reductions
    std::vector<char> bat_mode;  // batches

    {
      std::vector<char> a_label = a_mode;
      std::vector<char> b_label = b_mode;
      std::vector<char> c_label = c_mode;

      std::sort(std::begin(a_label), std::end(a_label));
      std::sort(std::begin(b_label), std::end(b_label));
      std::sort(std::begin(c_label), std::end(c_label));

      // std::set_intersections to find semantic category of each symbolic mode
      std::set_intersection(std::begin(a_label), std::end(a_label),
                            std::begin(c_label), std::end(c_label),
                            std::back_inserter(row_mode));

      std::set_intersection(std::begin(b_label), std::end(b_label),
                            std::begin(c_label), std::end(c_label),
                            std::back_inserter(col_mode));

      std::set_intersection(std::begin(a_label), std::end(a_label),
                            std::begin(b_label), std::end(b_label),
                            std::back_inserter(red_mode));

      std::set_intersection(std::begin(row_mode), std::end(row_mode),
                            std::begin(col_mode), std::end(col_mode),
                            std::back_inserter(bat_mode));

      // std::set_difference to remove batch modes from other semantic modes
      for (char l : bat_mode) {
        row_mode.erase(std::remove(std::begin(row_mode), std::end(row_mode), l), std::end(row_mode));
        col_mode.erase(std::remove(std::begin(col_mode), std::end(col_mode), l), std::end(col_mode));
        red_mode.erase(std::remove(std::begin(red_mode), std::end(red_mode), l), std::end(red_mode));
      }
    }

    // Print out the semantic association of each symbolic mode
    if (parse_verbose) {
      std::cout << "  rows : " << row_mode << '\n';
      std::cout << "  cols : " << col_mode << '\n';
      std::cout << "  reds : " << red_mode << '\n';
      std::cout << "  bats : " << bat_mode << '\n';
    }

    //
    // Permute modes
    //

    // Permute the batched modes to promote coalescing
    // Sort the batched modes by min(ldAl,ldBl) and in case of a tie by the size
    std::sort(std::begin(bat_mode), std::end(bat_mode), [&](char l1, char l2) {
        return std::tie(std::min(mode_ldA[l1],mode_ldB[l1]),mode_size[l1])
             < std::tie(std::min(mode_ldA[l2],mode_ldB[l2]),mode_size[l2]);
      });
    // Compute sizes and strides of ordered reduction modes
    std::vector<extent_type> L;
    std::vector<stride_type> ldAl;
    std::vector<stride_type> ldBl;
    std::vector<stride_type> ldCl;
    for (char l : bat_mode) {
      L.push_back(mode_size[l]);
      ldAl.push_back(mode_ldA[l]);
      ldBl.push_back(mode_ldB[l]);
      ldCl.push_back(mode_ldC[l]);
    }

    // Permute the reduction modes to promote coalescing
    // Sort the reduction modes by min(ldAk,ldBk) and in case of a tie by the size
    std::sort(std::begin(red_mode), std::end(red_mode), [&](char k1, char k2) {
        return std::tie(std::min(mode_ldA[k1],mode_ldB[k1]),mode_size[k1])
             < std::tie(std::min(mode_ldA[k2],mode_ldB[k2]),mode_size[k2]);
      });
    // Compute sizes and strides of ordered reduction modes
    std::vector<extent_type> K;
    std::vector<stride_type> ldAk;
    std::vector<stride_type> ldBk;
    for (char k : red_mode) {
      K.push_back(mode_size[k]);
      ldAk.push_back(mode_ldA[k]);
      ldBk.push_back(mode_ldB[k]);
    }

    // Permute the row modes to promote coalescing
    // Sort the row modes by min(ldAm,ldCm) and in case of a tie by ldAm
    std::sort(std::begin(row_mode), std::end(row_mode), [&](char m1, char m2) {
        return std::tie(std::min(mode_ldA[m1],mode_ldC[m1]),mode_ldA[m1])
             < std::tie(std::min(mode_ldA[m2],mode_ldC[m2]),mode_ldA[m2]);
      });
    // Compute sizes and strides of ordered row modes
    std::vector<extent_type> M;
    std::vector<stride_type> ldAm;
    std::vector<stride_type> ldCm;
    for (char m : row_mode) {
      M.push_back(mode_size[m]);
      ldAm.push_back(mode_ldA[m]);
      ldCm.push_back(mode_ldC[m]);
    }

    // Permute the col modes to promote coalescing
    // Sort the col modes by min(ldBn,ldCn) and in case of a tie by ldBn
    std::sort(std::begin(col_mode), std::end(col_mode), [&](char n1, char n2) {
        return std::tie(std::min(mode_ldB[n1],mode_ldC[n1]),mode_ldB[n1])
             < std::tie(std::min(mode_ldB[n2],mode_ldC[n2]),mode_ldB[n2]);
      });
    // Compute sizes and strides of ordered col modes
    std::vector<extent_type> N;
    std::vector<stride_type> ldBn;
    std::vector<stride_type> ldCn;
    for (char n : col_mode) {
      N.push_back(mode_size[n]);
      ldBn.push_back(mode_ldB[n]);
      ldCn.push_back(mode_ldC[n]);
    }

    if (parse_verbose) {
      std::cout << "C_";
      if (! row_mode.empty()) {
        std::cout << "(" << row_mode << ")";
      }
      if (! col_mode.empty()) {
        std::cout << "(" << col_mode << ")";
      }
      if (! bat_mode.empty()) {
        std::cout << "(" << bat_mode << ")";
      }
      std::cout << " = A_";
      if (! row_mode.empty()) {
        std::cout << "(" << row_mode << ")";
      }
      if (! red_mode.empty()) {
        std::cout << "(" << red_mode << ")";
      }
      if (! bat_mode.empty()) {
        std::cout << "(" << bat_mode << ")";
      }
      std::cout << " * B_";
      if (! col_mode.empty()) {
        std::cout << "(" << col_mode << ")";
      }
      if (! red_mode.empty()) {
        std::cout << "(" << red_mode << ")";
      }
      if (! bat_mode.empty()) {
        std::cout << "(" << bat_mode << ")";
      }
      std::cout << '\n';

      int M_size = std::accumulate(std::begin(M), std::end(M), 1, std::multiplies<>{});
      int N_size = std::accumulate(std::begin(N), std::end(N), 1, std::multiplies<>{});
      int K_size = std::accumulate(std::begin(K), std::end(K), 1, std::multiplies<>{});
      int L_size = std::accumulate(std::begin(L), std::end(L), 1, std::multiplies<>{});

      std::cout << "     M : (" << M_size << ") ";
      for (char m : row_mode) std::cout << m << ":" << mode_size[m] << " ";
      std::cout << '\n';
      std::cout << "     N : (" << N_size << ") ";
      for (char n : col_mode) std::cout << n << ":" << mode_size[n] << " ";
      std::cout << '\n';
      std::cout << "     K : (" << K_size << ") ";
      for (char k : red_mode) std::cout << k << ":" << mode_size[k] << " ";
      std::cout << '\n';
      std::cout << "     L : (" << L_size << ") ";
      for (char l : bat_mode) std::cout << l << ":" << mode_size[l] << " ";
      std::cout << '\n';

      std::cout << "  ldAm : " << ldAm << '\n';
      std::cout << "  ldAk : " << ldAk << '\n';
      std::cout << "  ldAl : " << ldAl << '\n';
      std::cout << "  ldBn : " << ldBn << '\n';
      std::cout << "  ldBk : " << ldBk << '\n';
      std::cout << "  ldBl : " << ldBl << '\n';
      std::cout << "  ldCm : " << ldCm << '\n';
      std::cout << "  ldCn : " << ldCn << '\n';
      std::cout << "  ldCl : " << ldCl << '\n';
    }

    return {M, ldAm, ldCm,
            N, ldBn, ldCn,   
            K, ldAk, ldBk, 
            L, ldAl, ldBl, ldCl}; 
  }

  static void
  print_usage() {
    std::cout <<
      "GETT problem command line parser:\n"
      "  --modeA=<m0,...>\n"
      "    A comma delimited list of characters that correspond to the row, reduction, and batch modes in A tensor.\n"
      "    The semantic association of each symbolic mode is determined automatically.\n\n"

      "  --modeB=<m0,...>\n"
      "    A comma delimited list of characters that correspond to the column, reduction, and batch modes in B tensor.\n"
      "    The semantic association of each symbolic mode is determined automatically.\n\n"

      "  --modeC=<m0,...>\n"
      "    A comma delimited list of characters that correspond to the row, column, and batch modes in B tensor.\n"
      "    The semantic association of each symbolic mode is determined automatically.\n\n"

      "  --extents=<mode:extent,....>\n"
      "    A command delimited list of symbolic mode and its corresponding extent.\n"
      "    Extents are defaulted to 1 if any are not provided.\n\n"

      "Example usage: gett.exe --modeC=m,n,l --modeA=m,k,l --modeB=k,n,l --extents=m:4096,n:4096,k:4096\n";
  }
};

} // namespace cutlass
