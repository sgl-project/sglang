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

#pragma once

#include <vector>
#include <iostream>

// Cutlass command line parser
#include "cutlass/util/command_line.h"

class Options {
public:

  bool help;
  bool good;
  std::vector<int> extent;          ///< extent of tile to fill
  std::vector<int> stride;          ///< stride vector for layout function
  std::vector<int> output_shape;    ///< output shape
  int vectorize;                    ///< sequences of consecutive output elements are concatenated into a vector
                                    ///  if, and only if, they were consecutive in source memory

public:

  /// Options
  Options(): 
    help(false),
    good(true),
    extent({32, 8}),
    stride({32}),
    output_shape({16, 8}), 
    vectorize(1) { 

  }

  /// Constructs from command line parser
  Options(cutlass::CommandLine const & cmd_line): help(false), good(true) {

    if (cmd_line.check_cmd_line_flag("help") ||
        cmd_line.check_cmd_line_flag("h")) {

      help = true;
    }

    if (cmd_line.check_cmd_line_flag("extent")) {
      cmd_line.get_cmd_line_arguments("extent", extent);
    }
    else {
      extent = {32, 8};
    }

    if (cmd_line.check_cmd_line_flag("stride")) {
      cmd_line.get_cmd_line_arguments("stride", stride);
    }
    
    int default_output_shape[] = {16, 8}; 

    if (cmd_line.check_cmd_line_flag("output-shape")) {
      cmd_line.get_cmd_line_arguments("output-shape", output_shape);
    }

    for (int i = int(output_shape.size()); i < 2; ++i) {
      output_shape.push_back(default_output_shape[i]);
    }

    if (cmd_line.check_cmd_line_flag("vectorize")) {
      cmd_line.get_cmd_line_argument("vectorize", vectorize);
    }
    else {
      vectorize = 1;
    }

    if (output_shape.front() % vectorize) {

      std::cerr << "Error: --vectorize=" << vectorize 
        << " must divide contiguous elements in --output-shape="
        << output_shape.at(0) << "," << output_shape.at(1) << std::endl;

      good = false;
    }
  }

  /// Prints usage statement
  static void print_usage(std::ostream &out) {
    out
      << "  Options:\n"
      << "    --help                              Displays this help message.\n"
      << "    --extent=<extent>                   Specifies the layout-specific extent (as comma-delimited array).\n"
      << "    --stride=<stride>                   Specifies the layout-specific stride vector (comma-delimited array)\n"
      << "    --output-shape=<extent>             Specifies the dimensions of a row-major output matrix. \n"
      << "    --vectorize=<vector length>         If possible, vectorizes the output into vectors of consecutive elements\n";
  }
};
