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
  \brief CUTLASS layout visualization example
*/

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cutlass/coord.h"
#include "cutlass/util/reference/host/tensor_foreach.h"

#include "register_layout.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permits copying dynamic vectors into static-length vectors 
template <typename TensorCoord, int Rank>
struct vector_to_coord {
  
  vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {

    coord[Rank - 1] = vec.at(Rank - 1);
    
    if (Rank > 1) {
      vector_to_coord<TensorCoord, Rank - 1>(coord, vec);
    }
  }
};

/// Permits copying dynamic vectors into static-length vectors 
template <typename TensorCoord>
struct vector_to_coord<TensorCoord, 1> {
  
  vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {

    coord[0] = vec.at(0);
  }
};

/// Permits copying dynamic vectors into static-length vectors 
template <typename TensorCoord>
struct vector_to_coord<TensorCoord, 0> {
  
  vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::ostream &operator<<(std::ostream &out, std::vector<T> const &vec) {
  auto it = vec.begin();
  if (it != vec.end()) {
    out << *it;
    for (++it; it != vec.end(); ++it) {
      out << ", " << *it;
    }
  }
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord, int Rank>
struct coord_to_vector {
  
  coord_to_vector(std::vector<int> &vec, TensorCoord const &coord) {

    vec.at(Rank - 1) = coord[Rank - 1];
    coord_to_vector<TensorCoord, Rank - 1>(vec, coord);
  }
};

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord>
struct coord_to_vector<TensorCoord, 1> {
  
  coord_to_vector(std::vector<int> &vec, TensorCoord const &coord) {

    vec.at(0) = coord[0];
  }
};

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord>
struct coord_to_vector<TensorCoord, 0> {
  
  coord_to_vector(std::vector<int> &vec, TensorCoord const &coord) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure representing an element in source memory
struct Element {

  std::vector<int> coord;     ///< logical coordinate of element (as vector)
  int offset;                 ///< linear offset from source memory
  int color;                  ///< enables coloring each element to indicate

  /// Default ctor
  inline Element(): offset(-1), color(0) { }

  /// Construct from logical coordinate and initial offset
  inline Element(
    std::vector<int> const &coord_, 
    int offset_,
    int color_ = 0
  ): 
    coord(coord_), offset(offset_), color(color_) { }

  /// Returns true if element is in a defined state
  inline bool valid() const {
    return offset >= 0;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visualizes memory layouts by constructing a 'shape' 
template <typename Layout_>
class VisualizeLayout : public VisualizeLayoutBase {
public:

  using Layout = Layout_;
  using TensorCoord = typename Layout::TensorCoord;
  using Stride = typename Layout::Stride;

public:

  Options options;
  Layout layout;
  TensorCoord extent;
  std::vector<Element> elements;
  
public:

  /// Initializes the problem space
  VisualizeLayout() {

  }

  /// visualization method
  bool visualize(Options const &options_) {

    options = options_;
    
    if (options.extent.size() != TensorCoord::kRank) {
      
      std::cerr
        << "--extent must have rank " << TensorCoord::kRank
        << " (given: " << options.extent.size() << ")" << std::endl;

      return false;
    }
    
    vector_to_coord<TensorCoord, TensorCoord::kRank>(extent, options.extent);

    // Construct the layout for a packed tensor
    if (options.stride.empty()) {

      layout = Layout::packed(extent);
    }
    else if (options.stride.size() != Stride::kRank) {

      std::cerr 
        << "--stride must have rank " << Stride::kRank 
        << " (given: " << options.stride.size() << ")" << std::endl;

      return false;
    }
    else {
      // Stride from 
      Stride stride;
      vector_to_coord<Stride, Stride::kRank>(stride, options.stride);

      layout = Layout(stride);
    }

    // Resize elements, setting elements to 'undefined' state
    elements.resize(layout.capacity(extent));

    // enumerate points in tensor space and assign 
    cutlass::reference::host::TensorForEachLambda(
      extent, 
      [&](TensorCoord coord) { 
        
        std::vector<int> coord_vec(TensorCoord::kRank, 0);
        coord_to_vector<TensorCoord, TensorCoord::kRank>(coord_vec, coord);

        int offset = int(layout(coord));

        if (offset >= int(elements.size())) {
          std::cerr
            << "Layout error - " << coord_vec 
            << " is out of range (computed offset: " << offset 
            << ", capacity: " << elements.size() << std::endl;

          throw std::out_of_range("(TensorForEach) layout error - coordinate out of range");
        }

        elements.at(offset) = Element(coord_vec, offset);
      });

    return true;
  }

  /// Verifies the layout satisfies vectorization requirements
  bool verify(bool verbose, std::ostream &out) {
    return true;
  }

private:

  /// returns a pair (is_vectorizable, one_changing_rank) to determine if a
  /// vector exists (consecutive logical coordinates or uniformly invalid)
  /// at the given location. 
  std::pair< bool, int > _is_vectorizable(int i) const {
    // (all elements are invalid) or 
    // (all elements are valid AND 
    //  exactly one rank is changing AND 
    //  elements are consecutive)

    // Don't need vectorization.
    if (options.vectorize <= 2) return std::make_pair(false, -1);

    // Boundary check.
    if (i > int(elements.size()) || (i + options.vectorize - 1) > int(elements.size()))
      return std::make_pair(false, -1);

    // Check if either all elements are valid or invalid.
    bool all_elements_invalid = std::all_of(
        elements.begin() + i, elements.begin() + i + options.vectorize,
        [](Element const &e) { return !e.valid(); });

    bool all_elements_valid = std::all_of(
        elements.begin() + i, elements.begin() + i + options.vectorize,
        [](Element const &e) { return e.valid(); });

    if (!all_elements_invalid && !all_elements_valid)
      return std::make_pair(false, -1);

    // From here, it is vectorizable.
    if (all_elements_invalid) return std::make_pair(true, -1);

    // Check if only exactly one rank is changing.
    int one_changing_rank = -1;
    for (int j = 0; j < options.vectorize; ++j) {
      for (int r = 0; r < TensorCoord::kRank; ++r) {
        if (elements.at(i + j).coord.at(r) != elements.at(i).coord.at(r)) {
          if (one_changing_rank == -1) {
            one_changing_rank = r;
          } else if (one_changing_rank != r) {
            return std::make_pair(false, -1);
          }
        }
      }
    }

    return std::make_pair(true, one_changing_rank);
  }

  /// Prints a vector of elements
  void _print_vector(std::ostream &out, int i, int one_changing_rank) {
    Element const &base_element = elements.at(i);
    if (base_element.valid()) {
      out << "(";
      for (int r = 0; r < TensorCoord::kRank; ++r) {
        if (r) {
          out << ", ";
        }

        if (r == one_changing_rank) {
          out 
            << base_element.coord.at(r) 
            << ".." 
            << (base_element.coord.at(r) + options.vectorize - 1);
        }
        else {
          out << base_element.coord.at(r);
        }
      }
      out << ")";
    }
    else {
      out << " ";
    }
  }

  /// Prints a single element
  void _print_element(std::ostream &out, int k) {
    Element const &element = elements.at(k);
    if (element.valid()) {
      out << "(";
      for (int v = 0; v < TensorCoord::kRank; ++v) {
        out << (v ? ", " : "") << element.coord.at(v);
      }
      out << ")"; 
    }
    else {
      out << " ";
    }
  }

public:

  /// Pretty-prints the layout to the console
  void print_csv(std::ostream &out, char delim = '|', char new_line = '\n') {
    int row = -1;

    for (int i = 0; i < int(elements.size()); i += options.vectorize) {
      if (i % options.output_shape.at(0)) {
        out << delim;
      }
      else {
        if (row >= 0) {
          out << new_line;
        }
        ++row;
        if (row == options.output_shape.at(1)) {
          out << new_line;
          row = 0;
        }
      }

      auto is_vector = _is_vectorizable(i);

      if (is_vector.first) {
        _print_vector(out, i, is_vector.second);        // print a vector starting at element i
      }
      else {
        for (int j = 0; j < options.vectorize; ++j) {   // print individual elements [i..i+j)
          _print_element(out, i + j);
        }
      } 
    }
    
    out << new_line << std::flush;
  }

  /// Help message
  virtual std::ostream &print_help(std::ostream &out) {
    out << "TensorCoord rank " << TensorCoord::kRank << ", Stride rank: " << Stride::kRank;
    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
