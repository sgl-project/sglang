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
  \brief CUTLASS layout visualization tool
*/

#include <map>
#include <iostream>
#include <iomanip>
#include <memory>

#include <cutlass/cutlass.h>

#include "options.h"
#include "register_layout.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

std::map<std::string, std::unique_ptr<VisualizeLayoutBase> > layouts;

/////////////////////////////////////////////////////////////////////////////////////////////////

void print_usage(std::ostream &out) {

  out << "03_visualize_layout <layout> [options]"
    << "\n\n"
    << "  Layouts:\n";

  for (auto const & layout : layouts) {
    out << "    " << layout.first << std::string(46 - layout.first.size(), ' ');
    layout.second->print_help(out);
    out << "\n";
  }

  out << "\n";
    
  Options::print_usage(out);

  out << "\nExamples:\n\n"
      << "$ 03_visualize_layout RowMajor --extent=16,16\n"
      << "$ 03_visualize_layout \"ColumnMajorInterleaved<4>\" --extent=32,8 "
         "--output-shape=16 --vectorize=4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<4,64>\" "
         "--extent=64,64 --vectorize=32 --output-shape=256,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<4,128>\" "
         "--extent=128,32 --vectorize=32 --output-shape=256,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<4,256>\" "
         "--extent=256,16 --vectorize=32 --output-shape=256,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<8,32>\" "
         "--extent=32,64 --vectorize=16 --output-shape=128,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<8,64>\" "
         "--extent=64,32 --vectorize=16 --output-shape=128,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<8,128>\" "
         "--extent=128,16 --vectorize=16 --output-shape=128,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<16,32>\" "
         "--extent=32,32 --vectorize=8 --output-shape=64,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<16,64>\" "
         "--extent=64,16 --vectorize=8 --output-shape=64,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<32,16>\" "
         "--extent=16,32 --vectorize=4 --output-shape=32,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicand<32,32>\" "
         "--extent=32,16 --vectorize=4 --output-shape=32,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicandCongruous<32,32>\" "
         "--extent=32,16 --vectorize=4 --output-shape=32,4\n"
      << "$ 03_visualize_layout \"TensorOpMultiplicandCongruous<64, 16>\" "
         "--extent=16,16 --vectorize=2 --output-shape=16,4\n"
      << "$ 03_visualize_layout \"VoltaTensorOpMultiplicandCrosswise<16,32>\" "
         "--extent=32,64 --vectorize=4 --output-shape=64,4\n"
      << "$ 03_visualize_layout \"VoltaTensorOpMultiplicandCongruous<16>\" "
         "--extent=64,32 --vectorize=8 --output-shape=64,4\n";

  out << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point
int main(int argc, char const *arg[]) {

  RegisterLayouts(layouts);

  if (argc == 1 || (std::string(arg[0]) == "-h" || std::string(arg[1]) == "--help")) {
    print_usage(std::cout);
    return 0;
  }

  // parse command line, skipping layout name
  cutlass::CommandLine cmd_line(argc - 1, arg + 1);
  Options options(cmd_line);

  if (options.help) {
    print_usage(std::cout);
    return 0;
  }

  if (!options.good) {
    return -1;
  }

  std::string layout_name = arg[1];

  auto layout_it = layouts.find(layout_name);
  if (layout_it == layouts.end()) {
    std::cerr << "Layout '" << layout_name << "' not supported." << std::endl;
    return -1;
  }

  bool passed  = layout_it->second->visualize(options);
  if (!passed) {
    return -1;
  }

  layout_it->second->print_csv(std::cout);

  cudaFree(0); // Ensure CUDA is available.

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
