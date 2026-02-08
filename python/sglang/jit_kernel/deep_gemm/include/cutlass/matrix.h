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
/*  
  \file
  \brief Matrix classes with value semantics.
*/

#pragma once

#if !defined(__CUDACC_RTC__)
#include <iosfwd>
#include <cmath>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/matrix.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Primary template with partial specializations to follow
template <typename Element, int Rows, int Columns> struct Matrix;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-by-2 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 1, 2> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 1;

  /// Number of columns in matrix
  static int const kColumns = 2;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 2;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 1-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 1-by-2 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1
  ) {

    data[0] = _0_0;  data[1] = _0_1;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> transpose() const {
    Matrix<Element, 2, 1> mt;
    
    mt.data[0] = data[0];
    mt.data[1] = data[1];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 1 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 1 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> row(int i) const {
    return slice_1x2(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 2> const &v, int i = 0) {
    return set_slice_1x2(v, i, 0);
  }
    
  /// Forms a 1-by-2 matrix by horizontally concatenating an Element with an Element
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Element lhs, Element rhs) {
    return Matrix(
      lhs, rhs);
  }
  
  /// Concatenates this matrix with a an Element to form a 1-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> hcat(Element rhs) const {
    return Matrix<Element, 1, 3>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 1-by-2 matrix to form a 1-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> hcat(Matrix<Element, 1, 2> const & rhs) const {
    return Matrix<Element, 1, 4>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 1-by-2 matrix to form a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> vcat(Matrix<Element, 1, 2> const & rhs) const {
    return Matrix<Element, 2, 2>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-2 matrix to form a 3-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> vcat(Matrix<Element, 2, 2> const & rhs) const {
    return Matrix<Element, 3, 2>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 3-by-2 matrix to form a 4-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> vcat(Matrix<Element, 3, 2> const & rhs) const {
    return Matrix<Element, 4, 2>::vcat(*this, rhs);
  }
    
  /// Elementwise add operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];

    return result;
  }
      
  /// Elementwise add operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];

    return *this;
  }
        
  /// Elementwise subtract operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];

    return result;
  }
      
  /// Elementwise subtract operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];

    return *this;
  }
        
  /// Elementwise multiply operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];

    return result;
  }
      
  /// Scalar multiply operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;

    return result;
  }

  /// Scalar multiply operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];

    return result;
  }
      
  /// Scalar divide operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;

    return result;
  }

  /// Scalar divide operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (1-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];

    return m;
  }
  
  /// Matrix product of size 1-by-1-by-2
  CUTLASS_HOST_DEVICE
  Element product(Matrix<Element, 2, 1> const &rhs, Element accum = Element()) const {
    
    // k=0
    accum += data[0] * rhs.data[0];

    // k=1
    accum += data[1] * rhs.data[1];

    return accum;
  }

  /// Matrix product of size 1-by-1-by-2
  CUTLASS_HOST_DEVICE
  Element operator*(Matrix<Element, 2, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> product(
    Matrix<Element, 2, 2> const &rhs,
    Matrix<Element, 1, 2> accum = Matrix<Element, 1, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 1-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> operator*(Matrix<Element, 2, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 2, 2> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 1-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> product(
    Matrix<Element, 2, 3> const &rhs,
    Matrix<Element, 1, 3> accum = Matrix<Element, 1, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 1-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> operator*(Matrix<Element, 2, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> product(
    Matrix<Element, 2, 4> const &rhs,
    Matrix<Element, 1, 4> accum = Matrix<Element, 1, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 1-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> operator*(Matrix<Element, 2, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Dot product of vectors with extent 2
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 2, 1> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    return accum;
  }

  /// Dot product of vectors with extent 2
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 1, 2> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    return accum;
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];

    return accum;
  }
    
};

/// Template alias for 1-by-2 matrix
template <typename Element>
using Matrix1x2 = Matrix<Element, 1, 2>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix1x2<Element> make_Matrix1x2(
    Element _0_0, Element _0_1
) {
  return Matrix1x2<Element>(
  _0_0, _0_1 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-by-3 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 1, 3> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 1;

  /// Number of columns in matrix
  static int const kColumns = 3;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 3;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 1-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 1-by-3 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> transpose() const {
    Matrix<Element, 3, 1> mt;
    
    mt.data[0] = data[0];
    mt.data[1] = data[1];
    mt.data[2] = data[2];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 1 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 1 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> row(int i) const {
    return slice_1x3(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 3> const &v, int i = 0) {
    return set_slice_1x3(v, i, 0);
  }
    
  /// Forms a 1-by-3 matrix by horizontally concatenating an Element with a 1-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Element lhs, Matrix<Element, 1, 2> const & rhs) {
    return Matrix(
      lhs, rhs.at(0, 0), rhs.at(0, 1));
  }
  
  /// Forms a 1-by-3 matrix by horizontally concatenating a 1-by-2 matrix with an Element
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 1, 2> const & lhs, Element rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs);
  }
  
  /// Concatenates this matrix with a an Element to form a 1-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> hcat(Element rhs) const {
    return Matrix<Element, 1, 4>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 1-by-3 matrix to form a 2-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> vcat(Matrix<Element, 1, 3> const & rhs) const {
    return Matrix<Element, 2, 3>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-3 matrix to form a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> vcat(Matrix<Element, 2, 3> const & rhs) const {
    return Matrix<Element, 3, 3>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 3-by-3 matrix to form a 4-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> vcat(Matrix<Element, 3, 3> const & rhs) const {
    return Matrix<Element, 4, 3>::vcat(*this, rhs);
  }
    
  /// Elementwise add operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];

    return result;
  }
      
  /// Elementwise add operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];

    return *this;
  }
        
  /// Elementwise subtract operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];

    return result;
  }
      
  /// Elementwise subtract operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];

    return *this;
  }
        
  /// Elementwise multiply operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];

    return result;
  }
      
  /// Scalar multiply operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;

    return result;
  }

  /// Scalar multiply operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];

    return result;
  }
      
  /// Scalar divide operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;

    return result;
  }

  /// Scalar divide operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (1-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];

    return m;
  }
  
  /// Matrix product of size 1-by-1-by-3
  CUTLASS_HOST_DEVICE
  Element product(Matrix<Element, 3, 1> const &rhs, Element accum = Element()) const {
    
    // k=0
    accum += data[0] * rhs.data[0];

    // k=1
    accum += data[1] * rhs.data[1];

    // k=2
    accum += data[2] * rhs.data[2];

    return accum;
  }

  /// Matrix product of size 1-by-1-by-3
  CUTLASS_HOST_DEVICE
  Element operator*(Matrix<Element, 3, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> product(
    Matrix<Element, 3, 2> const &rhs,
    Matrix<Element, 1, 2> accum = Matrix<Element, 1, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 1-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> operator*(Matrix<Element, 3, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> product(
    Matrix<Element, 3, 3> const &rhs,
    Matrix<Element, 1, 3> accum = Matrix<Element, 1, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];

    return accum;
  }

  /// Matrix product of size 1-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> operator*(Matrix<Element, 3, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 3, 3> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 1-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> product(
    Matrix<Element, 3, 4> const &rhs,
    Matrix<Element, 1, 4> accum = Matrix<Element, 1, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 1-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> operator*(Matrix<Element, 3, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Dot product of vectors with extent 3
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 3, 1> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    return accum;
  }

  /// Dot product of vectors with extent 3
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 1, 3> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    return accum;
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];

    return accum;
  }
    
  /// Cross product
  CUTLASS_HOST_DEVICE
  Matrix cross(Matrix const &rhs) const {
    return Matrix(
      data[1] * rhs.data[2] - data[2] * rhs.data[1],
      data[2] * rhs.data[0] - data[0] * rhs.data[2],
      data[0] * rhs.data[1] - data[1] * rhs.data[0]
    );
  }
  
};

/// Template alias for 1-by-3 matrix
template <typename Element>
using Matrix1x3 = Matrix<Element, 1, 3>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix1x3<Element> make_Matrix1x3(
    Element _0_0, Element _0_1, Element _0_2
) {
  return Matrix1x3<Element>(
  _0_0, _0_1, _0_2 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-by-4 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 1, 4> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 1;

  /// Number of columns in matrix
  static int const kColumns = 4;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 4;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 1-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 1-by-4 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;  data[3] = _0_3;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> transpose() const {
    Matrix<Element, 4, 1> mt;
    
    mt.data[0] = data[0];
    mt.data[1] = data[1];
    mt.data[2] = data[2];
    mt.data[3] = data[3];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 1 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 1 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> slice_1x4(int i = 0, int j = 0) const {
    Matrix<Element, 1, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x4(Matrix<Element, 1, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> row(int i) const {
    return slice_1x4(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 4> const &v, int i = 0) {
    return set_slice_1x4(v, i, 0);
  }
    
  /// Forms a 1-by-4 matrix by horizontally concatenating an Element with a 1-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Element lhs, Matrix<Element, 1, 3> const & rhs) {
    return Matrix(
      lhs, rhs.at(0, 0), rhs.at(0, 1), rhs.at(0, 2));
  }
  
  /// Forms a 1-by-4 matrix by horizontally concatenating a 1-by-2 matrix with a 1-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 1, 2> const & lhs, Matrix<Element, 1, 2> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs.at(0, 0), rhs.at(0, 1));
  }
  
  /// Forms a 1-by-4 matrix by horizontally concatenating a 1-by-3 matrix with an Element
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 1, 3> const & lhs, Element rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), lhs.at(0, 2), rhs);
  }
  
  /// Concatenates this matrix with a a 1-by-4 matrix to form a 2-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> vcat(Matrix<Element, 1, 4> const & rhs) const {
    return Matrix<Element, 2, 4>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-4 matrix to form a 3-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> vcat(Matrix<Element, 2, 4> const & rhs) const {
    return Matrix<Element, 3, 4>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 3-by-4 matrix to form a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> vcat(Matrix<Element, 3, 4> const & rhs) const {
    return Matrix<Element, 4, 4>::vcat(*this, rhs);
  }
    
  /// Elementwise add operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];
    result.data[3] = data[3] + rhs.data[3];

    return result;
  }
      
  /// Elementwise add operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];
    data[3] += rhs.data[3];

    return *this;
  }
        
  /// Elementwise subtract operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];
    result.data[3] = data[3] - rhs.data[3];

    return result;
  }
      
  /// Elementwise subtract operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];
    data[3] -= rhs.data[3];

    return *this;
  }
        
  /// Elementwise multiply operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];
    result.data[3] = data[3] * rhs.data[3];

    return result;
  }
      
  /// Scalar multiply operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;
    result.data[3] = data[3] * s;

    return result;
  }

  /// Scalar multiply operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;
    data[3] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];
    result.data[3] = data[3] / rhs.data[3];

    return result;
  }
      
  /// Scalar divide operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;
    result.data[3] = data[3] / s;

    return result;
  }

  /// Scalar divide operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;
    data[3] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (1-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];
    data[3] /= rhs.data[3];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];

    return m;
  }
  
  /// Matrix product of size 1-by-1-by-4
  CUTLASS_HOST_DEVICE
  Element product(Matrix<Element, 4, 1> const &rhs, Element accum = Element()) const {
    
    // k=0
    accum += data[0] * rhs.data[0];

    // k=1
    accum += data[1] * rhs.data[1];

    // k=2
    accum += data[2] * rhs.data[2];

    // k=3
    accum += data[3] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 1-by-1-by-4
  CUTLASS_HOST_DEVICE
  Element operator*(Matrix<Element, 4, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> product(
    Matrix<Element, 4, 2> const &rhs,
    Matrix<Element, 1, 2> accum = Matrix<Element, 1, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];

    // k=3
    accum.data[0] += data[3] * rhs.data[6];
    accum.data[1] += data[3] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 1-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> operator*(Matrix<Element, 4, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> product(
    Matrix<Element, 4, 3> const &rhs,
    Matrix<Element, 1, 3> accum = Matrix<Element, 1, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];

    // k=3
    accum.data[0] += data[3] * rhs.data[9];
    accum.data[1] += data[3] * rhs.data[10];
    accum.data[2] += data[3] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 1-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> operator*(Matrix<Element, 4, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> product(
    Matrix<Element, 4, 4> const &rhs,
    Matrix<Element, 1, 4> accum = Matrix<Element, 1, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];

    // k=3
    accum.data[0] += data[3] * rhs.data[12];
    accum.data[1] += data[3] * rhs.data[13];
    accum.data[2] += data[3] * rhs.data[14];
    accum.data[3] += data[3] * rhs.data[15];

    return accum;
  }

  /// Matrix product of size 1-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> operator*(Matrix<Element, 4, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 1-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 4, 4> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Dot product of vectors with extent 4
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 4, 1> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    accum += data[3] * rhs.data[3];
    return accum;
  }

  /// Dot product of vectors with extent 4
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 1, 4> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    accum += data[3] * rhs.data[3];
    return accum;
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];

    return accum;
  }
    
};

/// Template alias for 1-by-4 matrix
template <typename Element>
using Matrix1x4 = Matrix<Element, 1, 4>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix1x4<Element> make_Matrix1x4(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3
) {
  return Matrix1x4<Element>(
  _0_0, _0_1, _0_2, _0_3 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 2-by-1 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 2, 1> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 2;

  /// Number of columns in matrix
  static int const kColumns = 1;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 2;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 2-by-1 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 2-by-1 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, 
    Element _1_0
  ) {

    data[0] = _0_0;
    data[1] = _1_0;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> transpose() const {
    Matrix<Element, 1, 2> mt;
    
    mt.data[0] = data[0];
    mt.data[1] = data[1];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 1 + j + 0];
    m.data[1] = data[i * 1 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 1 + j + 0] = m.data[0];
    data[i * 1 + j + 1] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> column(int j) const {
    return slice_2x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 2, 1> const &v, int j =0) {
    return set_slice_2x1(v, 0, j);
  }
    
  /// Concatenates this matrix with a a 2-by-1 matrix to form a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> hcat(Matrix<Element, 2, 1> const & rhs) const {
    return Matrix<Element, 2, 2>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-2 matrix to form a 2-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> hcat(Matrix<Element, 2, 2> const & rhs) const {
    return Matrix<Element, 2, 3>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-3 matrix to form a 2-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> hcat(Matrix<Element, 2, 3> const & rhs) const {
    return Matrix<Element, 2, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 2-by-1 matrix by vertically concatenating an Element with an Element
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Element upper, Element lower) {
    return Matrix(
      upper
      , lower);
  }
  
  /// Concatenates this matrix with a an Element to form a 3-by-1 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> vcat(Element rhs) const {
    return Matrix<Element, 3, 1>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-1 matrix to form a 4-by-1 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> vcat(Matrix<Element, 2, 1> const & rhs) const {
    return Matrix<Element, 4, 1>::vcat(*this, rhs);
  }
    
  /// Elementwise add operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];

    result.data[1] = data[1] + rhs.data[1];

    return result;
  }
      
  /// Elementwise add operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];

    data[1] += rhs.data[1];

    return *this;
  }
        
  /// Elementwise subtract operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];

    result.data[1] = data[1] - rhs.data[1];

    return result;
  }
      
  /// Elementwise subtract operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];

    data[1] -= rhs.data[1];

    return *this;
  }
        
  /// Elementwise multiply operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];

    result.data[1] = data[1] * rhs.data[1];

    return result;
  }
      
  /// Scalar multiply operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;

    result.data[1] = data[1] * s;

    return result;
  }

  /// Scalar multiply operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;

    data[1] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];

    result.data[1] = data[1] / rhs.data[1];

    return result;
  }
      
  /// Scalar divide operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;

    result.data[1] = data[1] / s;

    return result;
  }

  /// Scalar divide operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;

    data[1] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (2-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];

    data[1] /= rhs.data[1];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];

    return m;
  }
  
  /// Matrix product of size 2-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> product(
    Matrix<Element, 1, 1> const &rhs,
    Matrix<Element, 2, 1> accum = Matrix<Element, 2, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[1] * rhs.data[0];

    return accum;
  }

  /// Matrix product of size 2-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> operator*(Matrix<Element, 1, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 1, 1> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 2-by-2-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> product(
    Matrix<Element, 1, 2> const &rhs,
    Matrix<Element, 2, 2> accum = Matrix<Element, 2, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[1] * rhs.data[0];
    accum.data[3] += data[1] * rhs.data[1];

    return accum;
  }

  /// Matrix product of size 2-by-2-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> operator*(Matrix<Element, 1, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-3-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> product(
    Matrix<Element, 1, 3> const &rhs,
    Matrix<Element, 2, 3> accum = Matrix<Element, 2, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[1] * rhs.data[0];
    accum.data[4] += data[1] * rhs.data[1];
    accum.data[5] += data[1] * rhs.data[2];

    return accum;
  }

  /// Matrix product of size 2-by-3-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> operator*(Matrix<Element, 1, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-4-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> product(
    Matrix<Element, 1, 4> const &rhs,
    Matrix<Element, 2, 4> accum = Matrix<Element, 2, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[1] * rhs.data[0];
    accum.data[5] += data[1] * rhs.data[1];
    accum.data[6] += data[1] * rhs.data[2];
    accum.data[7] += data[1] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 2-by-4-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> operator*(Matrix<Element, 1, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Dot product of vectors with extent 2
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 2, 1> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    return accum;
  }

  /// Dot product of vectors with extent 2
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 1, 2> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    return accum;
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];

    return accum;
  }
    
};

/// Template alias for 2-by-1 matrix
template <typename Element>
using Matrix2x1 = Matrix<Element, 2, 1>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix2x1<Element> make_Matrix2x1(
    Element _0_0, 
    Element _1_0
) {
  return Matrix2x1<Element>(
  _0_0, 
  _1_0 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 2-by-2 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 2, 2> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 2;

  /// Number of columns in matrix
  static int const kColumns = 2;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 4;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 2-by-2 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, 
    Element _1_0, Element _1_1
  ) {

    data[0] = _0_0;  data[1] = _0_1;
    data[2] = _1_0;  data[3] = _1_1;
  }
    
  /// Constructs a 2-by-2 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 2> const &row_0,
    Matrix<Element, 1, 2> const &row_1
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_1.data[0];
    data[3] = row_1.data[1];
  }
    
  /// Static method to construct a 2-by-2 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 2, 1> const &column_0,
    Matrix<Element, 2, 1> const &column_1
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_0.data[1];
    result.data[3] = column_1.data[1];
    return result;
  }
    
  /// Constructs an identity matrix
  CUTLASS_HOST_DEVICE
  static Matrix identity() {
    Matrix m;
    
    m.data[0] = Element(1);
    m.data[3] = Element(1);

    return m;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 2, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[3] = diag.data[1];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 2> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[3] = diag.data[1];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> diagonal() const {
    Matrix<Element, 2, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[3];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> transpose() const {
    Matrix<Element, 2, 2> mt;
    
    mt.data[0] = data[0];
    mt.data[2] = data[1];
    mt.data[1] = data[2];
    mt.data[3] = data[3];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> row(int i) const {
    return slice_1x2(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 2> const &v, int i = 0) {
    return set_slice_1x2(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 2] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> column(int j) const {
    return slice_2x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 2, 1> const &v, int j =0) {
    return set_slice_2x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];
    m.data[2] = data[i * 2 + j + 2];
    m.data[3] = data[i * 2 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];
    data[i * 2 + j + 2] = m.data[2];
    data[i * 2 + j + 3] = m.data[3];

    return *this;
  }
    
  /// Forms a 2-by-2 matrix by horizontally concatenating a 2-by-1 matrix with a 2-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 2, 1> const & lhs, Matrix<Element, 2, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0)
      , lhs.at(1, 0), rhs.at(1, 0));
  }
  
  /// Concatenates this matrix with a a 2-by-1 matrix to form a 2-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> hcat(Matrix<Element, 2, 1> const & rhs) const {
    return Matrix<Element, 2, 3>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-2 matrix to form a 2-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> hcat(Matrix<Element, 2, 2> const & rhs) const {
    return Matrix<Element, 2, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 2-by-2 matrix by vertically concatenating a 1-by-2 matrix with a 1-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 2> const & upper, Matrix<Element, 1, 2> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1)
      , lower.at(0, 0), lower.at(0, 1));
  }
  
  /// Concatenates this matrix with a a 1-by-2 matrix to form a 3-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> vcat(Matrix<Element, 1, 2> const & rhs) const {
    return Matrix<Element, 3, 2>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-2 matrix to form a 4-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> vcat(Matrix<Element, 2, 2> const & rhs) const {
    return Matrix<Element, 4, 2>::vcat(*this, rhs);
  }
    
  /// Forms a 2-by-2 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Element                         B,
    Element                         C, Element                         D) {
    return Matrix(
      A, B
      , C, D
    );
  }
  
  /// Elementwise add operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];

    result.data[2] = data[2] + rhs.data[2];
    result.data[3] = data[3] + rhs.data[3];

    return result;
  }
      
  /// Elementwise add operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];

    data[2] += rhs.data[2];
    data[3] += rhs.data[3];

    return *this;
  }
        
  /// Elementwise subtract operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];

    result.data[2] = data[2] - rhs.data[2];
    result.data[3] = data[3] - rhs.data[3];

    return result;
  }
      
  /// Elementwise subtract operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];

    data[2] -= rhs.data[2];
    data[3] -= rhs.data[3];

    return *this;
  }
        
  /// Elementwise multiply operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];

    result.data[2] = data[2] * rhs.data[2];
    result.data[3] = data[3] * rhs.data[3];

    return result;
  }
      
  /// Scalar multiply operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;

    result.data[2] = data[2] * s;
    result.data[3] = data[3] * s;

    return result;
  }

  /// Scalar multiply operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;

    data[2] *= s;
    data[3] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];

    result.data[2] = data[2] / rhs.data[2];
    result.data[3] = data[3] / rhs.data[3];

    return result;
  }
      
  /// Scalar divide operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;

    result.data[2] = data[2] / s;
    result.data[3] = data[3] / s;

    return result;
  }

  /// Scalar divide operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;

    data[2] /= s;
    data[3] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (2-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];

    data[2] /= rhs.data[2];
    data[3] /= rhs.data[3];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];

    return m;
  }
  
  /// Matrix product of size 2-by-1-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> product(
    Matrix<Element, 2, 1> const &rhs,
    Matrix<Element, 2, 1> accum = Matrix<Element, 2, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[2] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[3] * rhs.data[1];

    return accum;
  }

  /// Matrix product of size 2-by-1-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> operator*(Matrix<Element, 2, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> product(
    Matrix<Element, 2, 2> const &rhs,
    Matrix<Element, 2, 2> accum = Matrix<Element, 2, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[2] * rhs.data[0];
    accum.data[3] += data[2] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[3] * rhs.data[2];
    accum.data[3] += data[3] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 2-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> operator*(Matrix<Element, 2, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 2, 2> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 2-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> product(
    Matrix<Element, 2, 3> const &rhs,
    Matrix<Element, 2, 3> accum = Matrix<Element, 2, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[2] * rhs.data[0];
    accum.data[4] += data[2] * rhs.data[1];
    accum.data[5] += data[2] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[3] * rhs.data[3];
    accum.data[4] += data[3] * rhs.data[4];
    accum.data[5] += data[3] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 2-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> operator*(Matrix<Element, 2, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> product(
    Matrix<Element, 2, 4> const &rhs,
    Matrix<Element, 2, 4> accum = Matrix<Element, 2, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[2] * rhs.data[0];
    accum.data[5] += data[2] * rhs.data[1];
    accum.data[6] += data[2] * rhs.data[2];
    accum.data[7] += data[2] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[3] * rhs.data[4];
    accum.data[5] += data[3] * rhs.data[5];
    accum.data[6] += data[3] * rhs.data[6];
    accum.data[7] += data[3] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 2-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> operator*(Matrix<Element, 2, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[3];

    return accum;
  }
    
  /// Returns 2-by-2 rotation matrix
  CUTLASS_HOST_DEVICE
  static Matrix rotation(Element theta) {
    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    return Matrix(
      c, -s,
      s,  c
    );
  }
    
  /// Computes the determinant of a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  Element determinant(Element accum = Element()) const {
        accum += data[0] * data[3] - data[1] * data[2];

    return accum;
  }
  
  /// Computes the inverse of a 2-by-2 matrix given
  /// the matrix's determinant
  CUTLASS_HOST_DEVICE
  Matrix inverse(Element det) const {
    return Matrix(
      data[3], -data[1],
      -data[2], data[0]
    ) * (Element(1) / det); 
  }

  /// Computes the inverse of a 2-by-2 matrix.
  CUTLASS_HOST_DEVICE
  Matrix inverse() const {
    return inverse(determinant());
  }
    
};

/// Template alias for 2-by-2 matrix
template <typename Element>
using Matrix2x2 = Matrix<Element, 2, 2>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix2x2<Element> make_Matrix2x2(
    Element _0_0, Element _0_1, 
    Element _1_0, Element _1_1
) {
  return Matrix2x2<Element>(
  _0_0, _0_1, 
  _1_0, _1_1 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 2-by-3 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 2, 3> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 2;

  /// Number of columns in matrix
  static int const kColumns = 3;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 6;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 2-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 2-by-3 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2, 
    Element _1_0, Element _1_1, Element _1_2
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;
    data[3] = _1_0;  data[4] = _1_1;  data[5] = _1_2;
  }
    
  /// Constructs a 2-by-3 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 3> const &row_0,
    Matrix<Element, 1, 3> const &row_1
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_0.data[2];
    data[3] = row_1.data[0];
    data[4] = row_1.data[1];
    data[5] = row_1.data[2];
  }
    
  /// Static method to construct a 2-by-3 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 3, 1> const &column_0,
    Matrix<Element, 3, 1> const &column_1,
    Matrix<Element, 3, 1> const &column_2
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_2.data[0];
    result.data[3] = column_0.data[1];
    result.data[4] = column_1.data[1];
    result.data[5] = column_2.data[1];
    return result;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 2, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[3] = diag.data[1];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 2> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[3] = diag.data[1];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> diagonal() const {
    Matrix<Element, 2, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[3];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> transpose() const {
    Matrix<Element, 3, 2> mt;
    
    mt.data[0] = data[0];
    mt.data[2] = data[1];
    mt.data[4] = data[2];
    mt.data[1] = data[3];
    mt.data[3] = data[4];
    mt.data[5] = data[5];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> row(int i) const {
    return slice_1x3(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 3> const &v, int i = 0) {
    return set_slice_1x3(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 3] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> column(int j) const {
    return slice_2x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 2, 1> const &v, int j =0) {
    return set_slice_2x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 3];
    m.data[3] = data[i * 3 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 3] = m.data[2];
    data[i * 3 + j + 4] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> slice_2x3(int i = 0, int j = 0) const {
    Matrix<Element, 2, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];
    m.data[3] = data[i * 3 + j + 3];
    m.data[4] = data[i * 3 + j + 4];
    m.data[5] = data[i * 3 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x3(Matrix<Element, 2, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];
    data[i * 3 + j + 3] = m.data[3];
    data[i * 3 + j + 4] = m.data[4];
    data[i * 3 + j + 5] = m.data[5];

    return *this;
  }
    
  /// Forms a 2-by-3 matrix by horizontally concatenating a 2-by-1 matrix with a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 2, 1> const & lhs, Matrix<Element, 2, 2> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0), rhs.at(0, 1)
      , lhs.at(1, 0), rhs.at(1, 0), rhs.at(1, 1));
  }
  
  /// Forms a 2-by-3 matrix by horizontally concatenating a 2-by-2 matrix with a 2-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 2, 2> const & lhs, Matrix<Element, 2, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs.at(0, 0)
      , lhs.at(1, 0), lhs.at(1, 1), rhs.at(1, 0));
  }
  
  /// Concatenates this matrix with a a 2-by-1 matrix to form a 2-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> hcat(Matrix<Element, 2, 1> const & rhs) const {
    return Matrix<Element, 2, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 2-by-3 matrix by vertically concatenating a 1-by-3 matrix with a 1-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 3> const & upper, Matrix<Element, 1, 3> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2));
  }
  
  /// Concatenates this matrix with a a 1-by-3 matrix to form a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> vcat(Matrix<Element, 1, 3> const & rhs) const {
    return Matrix<Element, 3, 3>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-3 matrix to form a 4-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> vcat(Matrix<Element, 2, 3> const & rhs) const {
    return Matrix<Element, 4, 3>::vcat(*this, rhs);
  }
    
  /// Forms a 2-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Matrix<Element, 1, 2> const & B,
    Element                         C, Matrix<Element, 1, 2> const & D) {
    return Matrix(
      A, B.at(0, 0), B.at(0, 1)
      , C, D.at(0, 0), D.at(0, 1)
    );
  }
  
  /// Forms a 2-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 2> const & A, Element                         B,
    Matrix<Element, 1, 2> const & C, Element                         D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B
      , C.at(0, 0), C.at(0, 1), D
    );
  }
  
  /// Elementwise add operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];

    result.data[3] = data[3] + rhs.data[3];
    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];

    return result;
  }
      
  /// Elementwise add operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];

    data[3] += rhs.data[3];
    data[4] += rhs.data[4];
    data[5] += rhs.data[5];

    return *this;
  }
        
  /// Elementwise subtract operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];

    result.data[3] = data[3] - rhs.data[3];
    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];

    return result;
  }
      
  /// Elementwise subtract operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];

    data[3] -= rhs.data[3];
    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];

    return *this;
  }
        
  /// Elementwise multiply operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];

    result.data[3] = data[3] * rhs.data[3];
    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];

    return result;
  }
      
  /// Scalar multiply operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;

    result.data[3] = data[3] * s;
    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;

    return result;
  }

  /// Scalar multiply operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;

    data[3] *= s;
    data[4] *= s;
    data[5] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];

    result.data[3] = data[3] / rhs.data[3];
    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];

    return result;
  }
      
  /// Scalar divide operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;

    result.data[3] = data[3] / s;
    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;

    return result;
  }

  /// Scalar divide operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;

    data[3] /= s;
    data[4] /= s;
    data[5] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (2-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];

    data[3] /= rhs.data[3];
    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];

    return m;
  }
  
  /// Matrix product of size 2-by-1-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> product(
    Matrix<Element, 3, 1> const &rhs,
    Matrix<Element, 2, 1> accum = Matrix<Element, 2, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[3] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[4] * rhs.data[1];

    // k=2
    accum.data[0] += data[2] * rhs.data[2];
    accum.data[1] += data[5] * rhs.data[2];

    return accum;
  }

  /// Matrix product of size 2-by-1-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> operator*(Matrix<Element, 3, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> product(
    Matrix<Element, 3, 2> const &rhs,
    Matrix<Element, 2, 2> accum = Matrix<Element, 2, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[3] * rhs.data[0];
    accum.data[3] += data[3] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[4] * rhs.data[2];
    accum.data[3] += data[4] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];
    accum.data[2] += data[5] * rhs.data[4];
    accum.data[3] += data[5] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 2-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> operator*(Matrix<Element, 3, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> product(
    Matrix<Element, 3, 3> const &rhs,
    Matrix<Element, 2, 3> accum = Matrix<Element, 2, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[3] * rhs.data[0];
    accum.data[4] += data[3] * rhs.data[1];
    accum.data[5] += data[3] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[4] * rhs.data[3];
    accum.data[4] += data[4] * rhs.data[4];
    accum.data[5] += data[4] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];
    accum.data[3] += data[5] * rhs.data[6];
    accum.data[4] += data[5] * rhs.data[7];
    accum.data[5] += data[5] * rhs.data[8];

    return accum;
  }

  /// Matrix product of size 2-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> operator*(Matrix<Element, 3, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 3, 3> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 2-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> product(
    Matrix<Element, 3, 4> const &rhs,
    Matrix<Element, 2, 4> accum = Matrix<Element, 2, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[3] * rhs.data[0];
    accum.data[5] += data[3] * rhs.data[1];
    accum.data[6] += data[3] * rhs.data[2];
    accum.data[7] += data[3] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[4] * rhs.data[4];
    accum.data[5] += data[4] * rhs.data[5];
    accum.data[6] += data[4] * rhs.data[6];
    accum.data[7] += data[4] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];
    accum.data[4] += data[5] * rhs.data[8];
    accum.data[5] += data[5] * rhs.data[9];
    accum.data[6] += data[5] * rhs.data[10];
    accum.data[7] += data[5] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 2-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> operator*(Matrix<Element, 3, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[4];

    return accum;
  }
    
};

/// Template alias for 2-by-3 matrix
template <typename Element>
using Matrix2x3 = Matrix<Element, 2, 3>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix2x3<Element> make_Matrix2x3(
    Element _0_0, Element _0_1, Element _0_2, 
    Element _1_0, Element _1_1, Element _1_2
) {
  return Matrix2x3<Element>(
  _0_0, _0_1, _0_2, 
  _1_0, _1_1, _1_2 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 2-by-4 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 2, 4> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 2;

  /// Number of columns in matrix
  static int const kColumns = 4;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 8;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 2-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 2-by-4 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3, 
    Element _1_0, Element _1_1, Element _1_2, Element _1_3
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;  data[3] = _0_3;
    data[4] = _1_0;  data[5] = _1_1;  data[6] = _1_2;  data[7] = _1_3;
  }
    
  /// Constructs a 2-by-4 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 4> const &row_0,
    Matrix<Element, 1, 4> const &row_1
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_0.data[2];
    data[3] = row_0.data[3];
    data[4] = row_1.data[0];
    data[5] = row_1.data[1];
    data[6] = row_1.data[2];
    data[7] = row_1.data[3];
  }
    
  /// Static method to construct a 2-by-4 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 4, 1> const &column_0,
    Matrix<Element, 4, 1> const &column_1,
    Matrix<Element, 4, 1> const &column_2,
    Matrix<Element, 4, 1> const &column_3
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_2.data[0];
    result.data[3] = column_3.data[0];
    result.data[4] = column_0.data[1];
    result.data[5] = column_1.data[1];
    result.data[6] = column_2.data[1];
    result.data[7] = column_3.data[1];
    return result;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;
    m.data[6] = s;
    m.data[7] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 2, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[3] = diag.data[1];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 2> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[3] = diag.data[1];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> diagonal() const {
    Matrix<Element, 2, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[3];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> transpose() const {
    Matrix<Element, 4, 2> mt;
    
    mt.data[0] = data[0];
    mt.data[2] = data[1];
    mt.data[4] = data[2];
    mt.data[6] = data[3];
    mt.data[1] = data[4];
    mt.data[3] = data[5];
    mt.data[5] = data[6];
    mt.data[7] = data[7];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 2 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> slice_1x4(int i = 0, int j = 0) const {
    Matrix<Element, 1, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x4(Matrix<Element, 1, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> row(int i) const {
    return slice_1x4(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 4> const &v, int i = 0) {
    return set_slice_1x4(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 4] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> column(int j) const {
    return slice_2x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 2, 1> const &v, int j =0) {
    return set_slice_2x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 4];
    m.data[3] = data[i * 4 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 4] = m.data[2];
    data[i * 4 + j + 5] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> slice_2x3(int i = 0, int j = 0) const {
    Matrix<Element, 2, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 4];
    m.data[4] = data[i * 4 + j + 5];
    m.data[5] = data[i * 4 + j + 6];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x3(Matrix<Element, 2, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 4] = m.data[3];
    data[i * 4 + j + 5] = m.data[4];
    data[i * 4 + j + 6] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> slice_2x4(int i = 0, int j = 0) const {
    Matrix<Element, 2, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];
    m.data[4] = data[i * 4 + j + 4];
    m.data[5] = data[i * 4 + j + 5];
    m.data[6] = data[i * 4 + j + 6];
    m.data[7] = data[i * 4 + j + 7];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x4(Matrix<Element, 2, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];
    data[i * 4 + j + 4] = m.data[4];
    data[i * 4 + j + 5] = m.data[5];
    data[i * 4 + j + 6] = m.data[6];
    data[i * 4 + j + 7] = m.data[7];

    return *this;
  }
    
  /// Forms a 2-by-4 matrix by horizontally concatenating a 2-by-1 matrix with a 2-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 2, 1> const & lhs, Matrix<Element, 2, 3> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0), rhs.at(0, 1), rhs.at(0, 2)
      , lhs.at(1, 0), rhs.at(1, 0), rhs.at(1, 1), rhs.at(1, 2));
  }
  
  /// Forms a 2-by-4 matrix by horizontally concatenating a 2-by-2 matrix with a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 2, 2> const & lhs, Matrix<Element, 2, 2> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs.at(0, 0), rhs.at(0, 1)
      , lhs.at(1, 0), lhs.at(1, 1), rhs.at(1, 0), rhs.at(1, 1));
  }
  
  /// Forms a 2-by-4 matrix by horizontally concatenating a 2-by-3 matrix with a 2-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 2, 3> const & lhs, Matrix<Element, 2, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), lhs.at(0, 2), rhs.at(0, 0)
      , lhs.at(1, 0), lhs.at(1, 1), lhs.at(1, 2), rhs.at(1, 0));
  }
  
  /// Forms a 2-by-4 matrix by vertically concatenating a 1-by-4 matrix with a 1-by-4 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 4> const & upper, Matrix<Element, 1, 4> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2), upper.at(0, 3)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2), lower.at(0, 3));
  }
  
  /// Concatenates this matrix with a a 1-by-4 matrix to form a 3-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> vcat(Matrix<Element, 1, 4> const & rhs) const {
    return Matrix<Element, 3, 4>::vcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 2-by-4 matrix to form a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> vcat(Matrix<Element, 2, 4> const & rhs) const {
    return Matrix<Element, 4, 4>::vcat(*this, rhs);
  }
    
  /// Forms a 2-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Matrix<Element, 1, 3> const & B,
    Element                         C, Matrix<Element, 1, 3> const & D) {
    return Matrix(
      A, B.at(0, 0), B.at(0, 1), B.at(0, 2)
      , C, D.at(0, 0), D.at(0, 1), D.at(0, 2)
    );
  }
  
  /// Forms a 2-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 2> const & A, Matrix<Element, 1, 2> const & B,
    Matrix<Element, 1, 2> const & C, Matrix<Element, 1, 2> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0), B.at(0, 1)
      , C.at(0, 0), C.at(0, 1), D.at(0, 0), D.at(0, 1)
    );
  }
  
  /// Forms a 2-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 3> const & A, Element                         B,
    Matrix<Element, 1, 3> const & C, Element                         D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), A.at(0, 2), B
      , C.at(0, 0), C.at(0, 1), C.at(0, 2), D
    );
  }
  
  /// Elementwise add operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];
    result.data[3] = data[3] + rhs.data[3];

    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];
    result.data[6] = data[6] + rhs.data[6];
    result.data[7] = data[7] + rhs.data[7];

    return result;
  }
      
  /// Elementwise add operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];
    data[3] += rhs.data[3];

    data[4] += rhs.data[4];
    data[5] += rhs.data[5];
    data[6] += rhs.data[6];
    data[7] += rhs.data[7];

    return *this;
  }
        
  /// Elementwise subtract operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];
    result.data[3] = data[3] - rhs.data[3];

    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];
    result.data[6] = data[6] - rhs.data[6];
    result.data[7] = data[7] - rhs.data[7];

    return result;
  }
      
  /// Elementwise subtract operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];
    data[3] -= rhs.data[3];

    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];
    data[6] -= rhs.data[6];
    data[7] -= rhs.data[7];

    return *this;
  }
        
  /// Elementwise multiply operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];
    result.data[3] = data[3] * rhs.data[3];

    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];
    result.data[6] = data[6] * rhs.data[6];
    result.data[7] = data[7] * rhs.data[7];

    return result;
  }
      
  /// Scalar multiply operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;
    result.data[3] = data[3] * s;

    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;
    result.data[6] = data[6] * s;
    result.data[7] = data[7] * s;

    return result;
  }

  /// Scalar multiply operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;
    data[3] *= s;

    data[4] *= s;
    data[5] *= s;
    data[6] *= s;
    data[7] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];
    result.data[3] = data[3] / rhs.data[3];

    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];
    result.data[6] = data[6] / rhs.data[6];
    result.data[7] = data[7] / rhs.data[7];

    return result;
  }
      
  /// Scalar divide operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;
    result.data[3] = data[3] / s;

    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;
    result.data[6] = data[6] / s;
    result.data[7] = data[7] / s;

    return result;
  }

  /// Scalar divide operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;
    data[3] /= s;

    data[4] /= s;
    data[5] /= s;
    data[6] /= s;
    data[7] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (2-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];
    data[3] /= rhs.data[3];

    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];
    data[6] /= rhs.data[6];
    data[7] /= rhs.data[7];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];
    m.data[6] = -data[6];
    m.data[7] = -data[7];

    return m;
  }
  
  /// Matrix product of size 2-by-1-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> product(
    Matrix<Element, 4, 1> const &rhs,
    Matrix<Element, 2, 1> accum = Matrix<Element, 2, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[4] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[5] * rhs.data[1];

    // k=2
    accum.data[0] += data[2] * rhs.data[2];
    accum.data[1] += data[6] * rhs.data[2];

    // k=3
    accum.data[0] += data[3] * rhs.data[3];
    accum.data[1] += data[7] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 2-by-1-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> operator*(Matrix<Element, 4, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> product(
    Matrix<Element, 4, 2> const &rhs,
    Matrix<Element, 2, 2> accum = Matrix<Element, 2, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[4] * rhs.data[0];
    accum.data[3] += data[4] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[5] * rhs.data[2];
    accum.data[3] += data[5] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];
    accum.data[2] += data[6] * rhs.data[4];
    accum.data[3] += data[6] * rhs.data[5];

    // k=3
    accum.data[0] += data[3] * rhs.data[6];
    accum.data[1] += data[3] * rhs.data[7];
    accum.data[2] += data[7] * rhs.data[6];
    accum.data[3] += data[7] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 2-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> operator*(Matrix<Element, 4, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> product(
    Matrix<Element, 4, 3> const &rhs,
    Matrix<Element, 2, 3> accum = Matrix<Element, 2, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[4] * rhs.data[0];
    accum.data[4] += data[4] * rhs.data[1];
    accum.data[5] += data[4] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[5] * rhs.data[3];
    accum.data[4] += data[5] * rhs.data[4];
    accum.data[5] += data[5] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];
    accum.data[3] += data[6] * rhs.data[6];
    accum.data[4] += data[6] * rhs.data[7];
    accum.data[5] += data[6] * rhs.data[8];

    // k=3
    accum.data[0] += data[3] * rhs.data[9];
    accum.data[1] += data[3] * rhs.data[10];
    accum.data[2] += data[3] * rhs.data[11];
    accum.data[3] += data[7] * rhs.data[9];
    accum.data[4] += data[7] * rhs.data[10];
    accum.data[5] += data[7] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 2-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> operator*(Matrix<Element, 4, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> product(
    Matrix<Element, 4, 4> const &rhs,
    Matrix<Element, 2, 4> accum = Matrix<Element, 2, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[4] * rhs.data[0];
    accum.data[5] += data[4] * rhs.data[1];
    accum.data[6] += data[4] * rhs.data[2];
    accum.data[7] += data[4] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[5] * rhs.data[4];
    accum.data[5] += data[5] * rhs.data[5];
    accum.data[6] += data[5] * rhs.data[6];
    accum.data[7] += data[5] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];
    accum.data[4] += data[6] * rhs.data[8];
    accum.data[5] += data[6] * rhs.data[9];
    accum.data[6] += data[6] * rhs.data[10];
    accum.data[7] += data[6] * rhs.data[11];

    // k=3
    accum.data[0] += data[3] * rhs.data[12];
    accum.data[1] += data[3] * rhs.data[13];
    accum.data[2] += data[3] * rhs.data[14];
    accum.data[3] += data[3] * rhs.data[15];
    accum.data[4] += data[7] * rhs.data[12];
    accum.data[5] += data[7] * rhs.data[13];
    accum.data[6] += data[7] * rhs.data[14];
    accum.data[7] += data[7] * rhs.data[15];

    return accum;
  }

  /// Matrix product of size 2-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> operator*(Matrix<Element, 4, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 2-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 4, 4> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];
    accum += data[6];
    accum += data[7];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];
    accum += data[6] * data[6];
    accum += data[7] * data[7];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[5];

    return accum;
  }
    
};

/// Template alias for 2-by-4 matrix
template <typename Element>
using Matrix2x4 = Matrix<Element, 2, 4>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix2x4<Element> make_Matrix2x4(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3, 
    Element _1_0, Element _1_1, Element _1_2, Element _1_3
) {
  return Matrix2x4<Element>(
  _0_0, _0_1, _0_2, _0_3, 
  _1_0, _1_1, _1_2, _1_3 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 3-by-1 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 3, 1> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 3;

  /// Number of columns in matrix
  static int const kColumns = 1;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 3;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 3-by-1 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 3-by-1 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, 
    Element _1_0, 
    Element _2_0
  ) {

    data[0] = _0_0;
    data[1] = _1_0;
    data[2] = _2_0;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> transpose() const {
    Matrix<Element, 1, 3> mt;
    
    mt.data[0] = data[0];
    mt.data[1] = data[1];
    mt.data[2] = data[2];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 1 + j + 0];
    m.data[1] = data[i * 1 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 1 + j + 0] = m.data[0];
    data[i * 1 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 1 + j + 0];
    m.data[1] = data[i * 1 + j + 1];
    m.data[2] = data[i * 1 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 1 + j + 0] = m.data[0];
    data[i * 1 + j + 1] = m.data[1];
    data[i * 1 + j + 2] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> column(int j) const {
    return slice_3x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 3, 1> const &v, int j =0) {
    return set_slice_3x1(v, 0, j);
  }
    
  /// Concatenates this matrix with a a 3-by-1 matrix to form a 3-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> hcat(Matrix<Element, 3, 1> const & rhs) const {
    return Matrix<Element, 3, 2>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 3-by-2 matrix to form a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> hcat(Matrix<Element, 3, 2> const & rhs) const {
    return Matrix<Element, 3, 3>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 3-by-3 matrix to form a 3-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> hcat(Matrix<Element, 3, 3> const & rhs) const {
    return Matrix<Element, 3, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 3-by-1 matrix by vertically concatenating an Element with a 2-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Element upper, Matrix<Element, 2, 1> const & lower) {
    return Matrix(
      upper
      , lower.at(0, 0)
      , lower.at(1, 0));
  }
  
  /// Forms a 3-by-1 matrix by vertically concatenating a 2-by-1 matrix with an Element
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 1> const & upper, Element lower) {
    return Matrix(
      upper.at(0, 0)
      , upper.at(1, 0)
      , lower);
  }
  
  /// Concatenates this matrix with a an Element to form a 4-by-1 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> vcat(Element rhs) const {
    return Matrix<Element, 4, 1>::vcat(*this, rhs);
  }
    
  /// Elementwise add operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];

    result.data[1] = data[1] + rhs.data[1];

    result.data[2] = data[2] + rhs.data[2];

    return result;
  }
      
  /// Elementwise add operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];

    data[1] += rhs.data[1];

    data[2] += rhs.data[2];

    return *this;
  }
        
  /// Elementwise subtract operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];

    result.data[1] = data[1] - rhs.data[1];

    result.data[2] = data[2] - rhs.data[2];

    return result;
  }
      
  /// Elementwise subtract operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];

    data[1] -= rhs.data[1];

    data[2] -= rhs.data[2];

    return *this;
  }
        
  /// Elementwise multiply operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];

    result.data[1] = data[1] * rhs.data[1];

    result.data[2] = data[2] * rhs.data[2];

    return result;
  }
      
  /// Scalar multiply operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;

    result.data[1] = data[1] * s;

    result.data[2] = data[2] * s;

    return result;
  }

  /// Scalar multiply operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;

    data[1] *= s;

    data[2] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];

    result.data[1] = data[1] / rhs.data[1];

    result.data[2] = data[2] / rhs.data[2];

    return result;
  }
      
  /// Scalar divide operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;

    result.data[1] = data[1] / s;

    result.data[2] = data[2] / s;

    return result;
  }

  /// Scalar divide operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;

    data[1] /= s;

    data[2] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (3-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];

    data[1] /= rhs.data[1];

    data[2] /= rhs.data[2];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];

    return m;
  }
  
  /// Matrix product of size 3-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> product(
    Matrix<Element, 1, 1> const &rhs,
    Matrix<Element, 3, 1> accum = Matrix<Element, 3, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[1] * rhs.data[0];
    accum.data[2] += data[2] * rhs.data[0];

    return accum;
  }

  /// Matrix product of size 3-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> operator*(Matrix<Element, 1, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 1, 1> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 3-by-2-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> product(
    Matrix<Element, 1, 2> const &rhs,
    Matrix<Element, 3, 2> accum = Matrix<Element, 3, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[1] * rhs.data[0];
    accum.data[3] += data[1] * rhs.data[1];
    accum.data[4] += data[2] * rhs.data[0];
    accum.data[5] += data[2] * rhs.data[1];

    return accum;
  }

  /// Matrix product of size 3-by-2-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> operator*(Matrix<Element, 1, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-3-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> product(
    Matrix<Element, 1, 3> const &rhs,
    Matrix<Element, 3, 3> accum = Matrix<Element, 3, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[1] * rhs.data[0];
    accum.data[4] += data[1] * rhs.data[1];
    accum.data[5] += data[1] * rhs.data[2];
    accum.data[6] += data[2] * rhs.data[0];
    accum.data[7] += data[2] * rhs.data[1];
    accum.data[8] += data[2] * rhs.data[2];

    return accum;
  }

  /// Matrix product of size 3-by-3-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> operator*(Matrix<Element, 1, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-4-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> product(
    Matrix<Element, 1, 4> const &rhs,
    Matrix<Element, 3, 4> accum = Matrix<Element, 3, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[1] * rhs.data[0];
    accum.data[5] += data[1] * rhs.data[1];
    accum.data[6] += data[1] * rhs.data[2];
    accum.data[7] += data[1] * rhs.data[3];
    accum.data[8] += data[2] * rhs.data[0];
    accum.data[9] += data[2] * rhs.data[1];
    accum.data[10] += data[2] * rhs.data[2];
    accum.data[11] += data[2] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 3-by-4-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> operator*(Matrix<Element, 1, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Dot product of vectors with extent 3
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 3, 1> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    return accum;
  }

  /// Dot product of vectors with extent 3
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 1, 3> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    return accum;
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];

    return accum;
  }
    
  /// Cross product
  CUTLASS_HOST_DEVICE
  Matrix cross(Matrix const &rhs) const {
    return Matrix(
      data[1] * rhs.data[2] - data[2] * rhs.data[1],
      data[2] * rhs.data[0] - data[0] * rhs.data[2],
      data[0] * rhs.data[1] - data[1] * rhs.data[0]
    );
  }
  
};

/// Template alias for 3-by-1 matrix
template <typename Element>
using Matrix3x1 = Matrix<Element, 3, 1>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix3x1<Element> make_Matrix3x1(
    Element _0_0, 
    Element _1_0, 
    Element _2_0
) {
  return Matrix3x1<Element>(
  _0_0, 
  _1_0, 
  _2_0 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 3-by-2 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 3, 2> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 3;

  /// Number of columns in matrix
  static int const kColumns = 2;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 6;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 3-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 3-by-2 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, 
    Element _1_0, Element _1_1, 
    Element _2_0, Element _2_1
  ) {

    data[0] = _0_0;  data[1] = _0_1;
    data[2] = _1_0;  data[3] = _1_1;
    data[4] = _2_0;  data[5] = _2_1;
  }
    
  /// Constructs a 3-by-2 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 2> const &row_0,
    Matrix<Element, 1, 2> const &row_1,
    Matrix<Element, 1, 2> const &row_2
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_1.data[0];
    data[3] = row_1.data[1];
    data[4] = row_2.data[0];
    data[5] = row_2.data[1];
  }
    
  /// Static method to construct a 3-by-2 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 2, 1> const &column_0,
    Matrix<Element, 2, 1> const &column_1
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_0.data[1];
    result.data[3] = column_1.data[1];
    result.data[4] = column_0.data[2];
    result.data[5] = column_1.data[2];
    return result;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 2, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[4] = diag.data[1];
    m.data[8] = diag.data[2];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 2> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[4] = diag.data[1];
    m.data[8] = diag.data[2];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> diagonal() const {
    Matrix<Element, 2, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[4];
    diag.data[2] = data[8];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> transpose() const {
    Matrix<Element, 2, 3> mt;
    
    mt.data[0] = data[0];
    mt.data[3] = data[1];
    mt.data[1] = data[2];
    mt.data[4] = data[3];
    mt.data[2] = data[4];
    mt.data[5] = data[5];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> row(int i) const {
    return slice_1x2(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 2> const &v, int i = 0) {
    return set_slice_1x2(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 2] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];
    m.data[2] = data[i * 2 + j + 2];
    m.data[3] = data[i * 2 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];
    data[i * 2 + j + 2] = m.data[2];
    data[i * 2 + j + 3] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 2];
    m.data[2] = data[i * 2 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 2] = m.data[1];
    data[i * 2 + j + 4] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> column(int j) const {
    return slice_3x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 3, 1> const &v, int j =0) {
    return set_slice_3x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> slice_3x2(int i = 0, int j = 0) const {
    Matrix<Element, 3, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];
    m.data[2] = data[i * 2 + j + 2];
    m.data[3] = data[i * 2 + j + 3];
    m.data[4] = data[i * 2 + j + 4];
    m.data[5] = data[i * 2 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x2(Matrix<Element, 3, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];
    data[i * 2 + j + 2] = m.data[2];
    data[i * 2 + j + 3] = m.data[3];
    data[i * 2 + j + 4] = m.data[4];
    data[i * 2 + j + 5] = m.data[5];

    return *this;
  }
    
  /// Forms a 3-by-2 matrix by horizontally concatenating a 3-by-1 matrix with a 3-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 3, 1> const & lhs, Matrix<Element, 3, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0)
      , lhs.at(1, 0), rhs.at(1, 0)
      , lhs.at(2, 0), rhs.at(2, 0));
  }
  
  /// Concatenates this matrix with a a 3-by-1 matrix to form a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> hcat(Matrix<Element, 3, 1> const & rhs) const {
    return Matrix<Element, 3, 3>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 3-by-2 matrix to form a 3-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> hcat(Matrix<Element, 3, 2> const & rhs) const {
    return Matrix<Element, 3, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 3-by-2 matrix by vertically concatenating a 1-by-2 matrix with a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 2> const & upper, Matrix<Element, 2, 2> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1)
      , lower.at(0, 0), lower.at(0, 1)
      , lower.at(1, 0), lower.at(1, 1));
  }
  
  /// Forms a 3-by-2 matrix by vertically concatenating a 2-by-2 matrix with a 1-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 2> const & upper, Matrix<Element, 1, 2> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1)
      , upper.at(1, 0), upper.at(1, 1)
      , lower.at(0, 0), lower.at(0, 1));
  }
  
  /// Concatenates this matrix with a a 1-by-2 matrix to form a 4-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> vcat(Matrix<Element, 1, 2> const & rhs) const {
    return Matrix<Element, 4, 2>::vcat(*this, rhs);
  }
    
  /// Forms a 3-by-2 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Element                         B,
    Matrix<Element, 2, 1> const & C, Matrix<Element, 2, 1> const & D) {
    return Matrix(
      A, B
      , C.at(0, 0), D.at(0, 0)
      , C.at(1, 0), D.at(1, 0)
    );
  }
  
  /// Forms a 3-by-2 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 1> const & A, Matrix<Element, 2, 1> const & B,
    Element                         C, Element                         D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0)
      , A.at(1, 0), B.at(1, 0)
      , C, D
    );
  }
  
  /// Elementwise add operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];

    result.data[2] = data[2] + rhs.data[2];
    result.data[3] = data[3] + rhs.data[3];

    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];

    return result;
  }
      
  /// Elementwise add operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];

    data[2] += rhs.data[2];
    data[3] += rhs.data[3];

    data[4] += rhs.data[4];
    data[5] += rhs.data[5];

    return *this;
  }
        
  /// Elementwise subtract operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];

    result.data[2] = data[2] - rhs.data[2];
    result.data[3] = data[3] - rhs.data[3];

    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];

    return result;
  }
      
  /// Elementwise subtract operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];

    data[2] -= rhs.data[2];
    data[3] -= rhs.data[3];

    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];

    return *this;
  }
        
  /// Elementwise multiply operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];

    result.data[2] = data[2] * rhs.data[2];
    result.data[3] = data[3] * rhs.data[3];

    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];

    return result;
  }
      
  /// Scalar multiply operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;

    result.data[2] = data[2] * s;
    result.data[3] = data[3] * s;

    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;

    return result;
  }

  /// Scalar multiply operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;

    data[2] *= s;
    data[3] *= s;

    data[4] *= s;
    data[5] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];

    result.data[2] = data[2] / rhs.data[2];
    result.data[3] = data[3] / rhs.data[3];

    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];

    return result;
  }
      
  /// Scalar divide operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;

    result.data[2] = data[2] / s;
    result.data[3] = data[3] / s;

    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;

    return result;
  }

  /// Scalar divide operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;

    data[2] /= s;
    data[3] /= s;

    data[4] /= s;
    data[5] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (3-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];

    data[2] /= rhs.data[2];
    data[3] /= rhs.data[3];

    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];

    return m;
  }
  
  /// Matrix product of size 3-by-1-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> product(
    Matrix<Element, 2, 1> const &rhs,
    Matrix<Element, 3, 1> accum = Matrix<Element, 3, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[2] * rhs.data[0];
    accum.data[2] += data[4] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[3] * rhs.data[1];
    accum.data[2] += data[5] * rhs.data[1];

    return accum;
  }

  /// Matrix product of size 3-by-1-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> operator*(Matrix<Element, 2, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> product(
    Matrix<Element, 2, 2> const &rhs,
    Matrix<Element, 3, 2> accum = Matrix<Element, 3, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[2] * rhs.data[0];
    accum.data[3] += data[2] * rhs.data[1];
    accum.data[4] += data[4] * rhs.data[0];
    accum.data[5] += data[4] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[3] * rhs.data[2];
    accum.data[3] += data[3] * rhs.data[3];
    accum.data[4] += data[5] * rhs.data[2];
    accum.data[5] += data[5] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 3-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> operator*(Matrix<Element, 2, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 2, 2> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 3-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> product(
    Matrix<Element, 2, 3> const &rhs,
    Matrix<Element, 3, 3> accum = Matrix<Element, 3, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[2] * rhs.data[0];
    accum.data[4] += data[2] * rhs.data[1];
    accum.data[5] += data[2] * rhs.data[2];
    accum.data[6] += data[4] * rhs.data[0];
    accum.data[7] += data[4] * rhs.data[1];
    accum.data[8] += data[4] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[3] * rhs.data[3];
    accum.data[4] += data[3] * rhs.data[4];
    accum.data[5] += data[3] * rhs.data[5];
    accum.data[6] += data[5] * rhs.data[3];
    accum.data[7] += data[5] * rhs.data[4];
    accum.data[8] += data[5] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 3-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> operator*(Matrix<Element, 2, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> product(
    Matrix<Element, 2, 4> const &rhs,
    Matrix<Element, 3, 4> accum = Matrix<Element, 3, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[2] * rhs.data[0];
    accum.data[5] += data[2] * rhs.data[1];
    accum.data[6] += data[2] * rhs.data[2];
    accum.data[7] += data[2] * rhs.data[3];
    accum.data[8] += data[4] * rhs.data[0];
    accum.data[9] += data[4] * rhs.data[1];
    accum.data[10] += data[4] * rhs.data[2];
    accum.data[11] += data[4] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[3] * rhs.data[4];
    accum.data[5] += data[3] * rhs.data[5];
    accum.data[6] += data[3] * rhs.data[6];
    accum.data[7] += data[3] * rhs.data[7];
    accum.data[8] += data[5] * rhs.data[4];
    accum.data[9] += data[5] * rhs.data[5];
    accum.data[10] += data[5] * rhs.data[6];
    accum.data[11] += data[5] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 3-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> operator*(Matrix<Element, 2, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[3];

    return accum;
  }
    
};

/// Template alias for 3-by-2 matrix
template <typename Element>
using Matrix3x2 = Matrix<Element, 3, 2>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix3x2<Element> make_Matrix3x2(
    Element _0_0, Element _0_1, 
    Element _1_0, Element _1_1, 
    Element _2_0, Element _2_1
) {
  return Matrix3x2<Element>(
  _0_0, _0_1, 
  _1_0, _1_1, 
  _2_0, _2_1 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 3-by-3 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 3, 3> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 3;

  /// Number of columns in matrix
  static int const kColumns = 3;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 9;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 3-by-3 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2, 
    Element _1_0, Element _1_1, Element _1_2, 
    Element _2_0, Element _2_1, Element _2_2
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;
    data[3] = _1_0;  data[4] = _1_1;  data[5] = _1_2;
    data[6] = _2_0;  data[7] = _2_1;  data[8] = _2_2;
  }
    
  /// Constructs a 3-by-3 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 3> const &row_0,
    Matrix<Element, 1, 3> const &row_1,
    Matrix<Element, 1, 3> const &row_2
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_0.data[2];
    data[3] = row_1.data[0];
    data[4] = row_1.data[1];
    data[5] = row_1.data[2];
    data[6] = row_2.data[0];
    data[7] = row_2.data[1];
    data[8] = row_2.data[2];
  }
    
  /// Static method to construct a 3-by-3 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 3, 1> const &column_0,
    Matrix<Element, 3, 1> const &column_1,
    Matrix<Element, 3, 1> const &column_2
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_2.data[0];
    result.data[3] = column_0.data[1];
    result.data[4] = column_1.data[1];
    result.data[5] = column_2.data[1];
    result.data[6] = column_0.data[2];
    result.data[7] = column_1.data[2];
    result.data[8] = column_2.data[2];
    return result;
  }
    
  /// Constructs an identity matrix
  CUTLASS_HOST_DEVICE
  static Matrix identity() {
    Matrix m;
    
    m.data[0] = Element(1);
    m.data[4] = Element(1);
    m.data[8] = Element(1);

    return m;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;
    m.data[6] = s;
    m.data[7] = s;
    m.data[8] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 3, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[4] = diag.data[1];
    m.data[8] = diag.data[2];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 3> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[4] = diag.data[1];
    m.data[8] = diag.data[2];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> diagonal() const {
    Matrix<Element, 3, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[4];
    diag.data[2] = data[8];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> transpose() const {
    Matrix<Element, 3, 3> mt;
    
    mt.data[0] = data[0];
    mt.data[3] = data[1];
    mt.data[6] = data[2];
    mt.data[1] = data[3];
    mt.data[4] = data[4];
    mt.data[7] = data[5];
    mt.data[2] = data[6];
    mt.data[5] = data[7];
    mt.data[8] = data[8];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> row(int i) const {
    return slice_1x3(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 3> const &v, int i = 0) {
    return set_slice_1x3(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 3] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 3];
    m.data[3] = data[i * 3 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 3] = m.data[2];
    data[i * 3 + j + 4] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> slice_2x3(int i = 0, int j = 0) const {
    Matrix<Element, 2, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];
    m.data[3] = data[i * 3 + j + 3];
    m.data[4] = data[i * 3 + j + 4];
    m.data[5] = data[i * 3 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x3(Matrix<Element, 2, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];
    data[i * 3 + j + 3] = m.data[3];
    data[i * 3 + j + 4] = m.data[4];
    data[i * 3 + j + 5] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 3];
    m.data[2] = data[i * 3 + j + 6];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 3] = m.data[1];
    data[i * 3 + j + 6] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> column(int j) const {
    return slice_3x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 3, 1> const &v, int j =0) {
    return set_slice_3x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> slice_3x2(int i = 0, int j = 0) const {
    Matrix<Element, 3, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 3];
    m.data[3] = data[i * 3 + j + 4];
    m.data[4] = data[i * 3 + j + 6];
    m.data[5] = data[i * 3 + j + 7];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x2(Matrix<Element, 3, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 3] = m.data[2];
    data[i * 3 + j + 4] = m.data[3];
    data[i * 3 + j + 6] = m.data[4];
    data[i * 3 + j + 7] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> slice_3x3(int i = 0, int j = 0) const {
    Matrix<Element, 3, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];
    m.data[3] = data[i * 3 + j + 3];
    m.data[4] = data[i * 3 + j + 4];
    m.data[5] = data[i * 3 + j + 5];
    m.data[6] = data[i * 3 + j + 6];
    m.data[7] = data[i * 3 + j + 7];
    m.data[8] = data[i * 3 + j + 8];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x3(Matrix<Element, 3, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];
    data[i * 3 + j + 3] = m.data[3];
    data[i * 3 + j + 4] = m.data[4];
    data[i * 3 + j + 5] = m.data[5];
    data[i * 3 + j + 6] = m.data[6];
    data[i * 3 + j + 7] = m.data[7];
    data[i * 3 + j + 8] = m.data[8];

    return *this;
  }
    
  /// Forms a 3-by-3 matrix by horizontally concatenating a 3-by-1 matrix with a 3-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 3, 1> const & lhs, Matrix<Element, 3, 2> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0), rhs.at(0, 1)
      , lhs.at(1, 0), rhs.at(1, 0), rhs.at(1, 1)
      , lhs.at(2, 0), rhs.at(2, 0), rhs.at(2, 1));
  }
  
  /// Forms a 3-by-3 matrix by horizontally concatenating a 3-by-2 matrix with a 3-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 3, 2> const & lhs, Matrix<Element, 3, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs.at(0, 0)
      , lhs.at(1, 0), lhs.at(1, 1), rhs.at(1, 0)
      , lhs.at(2, 0), lhs.at(2, 1), rhs.at(2, 0));
  }
  
  /// Concatenates this matrix with a a 3-by-1 matrix to form a 3-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> hcat(Matrix<Element, 3, 1> const & rhs) const {
    return Matrix<Element, 3, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 3-by-3 matrix by vertically concatenating a 1-by-3 matrix with a 2-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 3> const & upper, Matrix<Element, 2, 3> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2)
      , lower.at(1, 0), lower.at(1, 1), lower.at(1, 2));
  }
  
  /// Forms a 3-by-3 matrix by vertically concatenating a 2-by-3 matrix with a 1-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 3> const & upper, Matrix<Element, 1, 3> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2)
      , upper.at(1, 0), upper.at(1, 1), upper.at(1, 2)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2));
  }
  
  /// Concatenates this matrix with a a 1-by-3 matrix to form a 4-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> vcat(Matrix<Element, 1, 3> const & rhs) const {
    return Matrix<Element, 4, 3>::vcat(*this, rhs);
  }
    
  /// Forms a 3-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Matrix<Element, 1, 2> const & B,
    Matrix<Element, 2, 1> const & C, Matrix<Element, 2, 2> const & D) {
    return Matrix(
      A, B.at(0, 0), B.at(0, 1)
      , C.at(0, 0), D.at(0, 0), D.at(0, 1)
      , C.at(1, 0), D.at(1, 0), D.at(1, 1)
    );
  }
  
  /// Forms a 3-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 2> const & A, Element                         B,
    Matrix<Element, 2, 2> const & C, Matrix<Element, 2, 1> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B
      , C.at(0, 0), C.at(0, 1), D.at(0, 0)
      , C.at(1, 0), C.at(1, 1), D.at(1, 0)
    );
  }
  
  /// Forms a 3-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 1> const & A, Matrix<Element, 2, 2> const & B,
    Element                         C, Matrix<Element, 1, 2> const & D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0), B.at(0, 1)
      , A.at(1, 0), B.at(1, 0), B.at(1, 1)
      , C, D.at(0, 0), D.at(0, 1)
    );
  }
  
  /// Forms a 3-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 2> const & A, Matrix<Element, 2, 1> const & B,
    Matrix<Element, 1, 2> const & C, Element                         D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0)
      , A.at(1, 0), A.at(1, 1), B.at(1, 0)
      , C.at(0, 0), C.at(0, 1), D
    );
  }
  
  /// Elementwise add operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];

    result.data[3] = data[3] + rhs.data[3];
    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];

    result.data[6] = data[6] + rhs.data[6];
    result.data[7] = data[7] + rhs.data[7];
    result.data[8] = data[8] + rhs.data[8];

    return result;
  }
      
  /// Elementwise add operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];

    data[3] += rhs.data[3];
    data[4] += rhs.data[4];
    data[5] += rhs.data[5];

    data[6] += rhs.data[6];
    data[7] += rhs.data[7];
    data[8] += rhs.data[8];

    return *this;
  }
        
  /// Elementwise subtract operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];

    result.data[3] = data[3] - rhs.data[3];
    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];

    result.data[6] = data[6] - rhs.data[6];
    result.data[7] = data[7] - rhs.data[7];
    result.data[8] = data[8] - rhs.data[8];

    return result;
  }
      
  /// Elementwise subtract operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];

    data[3] -= rhs.data[3];
    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];

    data[6] -= rhs.data[6];
    data[7] -= rhs.data[7];
    data[8] -= rhs.data[8];

    return *this;
  }
        
  /// Elementwise multiply operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];

    result.data[3] = data[3] * rhs.data[3];
    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];

    result.data[6] = data[6] * rhs.data[6];
    result.data[7] = data[7] * rhs.data[7];
    result.data[8] = data[8] * rhs.data[8];

    return result;
  }
      
  /// Scalar multiply operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;

    result.data[3] = data[3] * s;
    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;

    result.data[6] = data[6] * s;
    result.data[7] = data[7] * s;
    result.data[8] = data[8] * s;

    return result;
  }

  /// Scalar multiply operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;

    data[3] *= s;
    data[4] *= s;
    data[5] *= s;

    data[6] *= s;
    data[7] *= s;
    data[8] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];

    result.data[3] = data[3] / rhs.data[3];
    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];

    result.data[6] = data[6] / rhs.data[6];
    result.data[7] = data[7] / rhs.data[7];
    result.data[8] = data[8] / rhs.data[8];

    return result;
  }
      
  /// Scalar divide operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;

    result.data[3] = data[3] / s;
    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;

    result.data[6] = data[6] / s;
    result.data[7] = data[7] / s;
    result.data[8] = data[8] / s;

    return result;
  }

  /// Scalar divide operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;

    data[3] /= s;
    data[4] /= s;
    data[5] /= s;

    data[6] /= s;
    data[7] /= s;
    data[8] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (3-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];

    data[3] /= rhs.data[3];
    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];

    data[6] /= rhs.data[6];
    data[7] /= rhs.data[7];
    data[8] /= rhs.data[8];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];
    m.data[6] = -data[6];
    m.data[7] = -data[7];
    m.data[8] = -data[8];

    return m;
  }
  
  /// Matrix product of size 3-by-1-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> product(
    Matrix<Element, 3, 1> const &rhs,
    Matrix<Element, 3, 1> accum = Matrix<Element, 3, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[3] * rhs.data[0];
    accum.data[2] += data[6] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[4] * rhs.data[1];
    accum.data[2] += data[7] * rhs.data[1];

    // k=2
    accum.data[0] += data[2] * rhs.data[2];
    accum.data[1] += data[5] * rhs.data[2];
    accum.data[2] += data[8] * rhs.data[2];

    return accum;
  }

  /// Matrix product of size 3-by-1-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> operator*(Matrix<Element, 3, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> product(
    Matrix<Element, 3, 2> const &rhs,
    Matrix<Element, 3, 2> accum = Matrix<Element, 3, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[3] * rhs.data[0];
    accum.data[3] += data[3] * rhs.data[1];
    accum.data[4] += data[6] * rhs.data[0];
    accum.data[5] += data[6] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[4] * rhs.data[2];
    accum.data[3] += data[4] * rhs.data[3];
    accum.data[4] += data[7] * rhs.data[2];
    accum.data[5] += data[7] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];
    accum.data[2] += data[5] * rhs.data[4];
    accum.data[3] += data[5] * rhs.data[5];
    accum.data[4] += data[8] * rhs.data[4];
    accum.data[5] += data[8] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 3-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> operator*(Matrix<Element, 3, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> product(
    Matrix<Element, 3, 3> const &rhs,
    Matrix<Element, 3, 3> accum = Matrix<Element, 3, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[3] * rhs.data[0];
    accum.data[4] += data[3] * rhs.data[1];
    accum.data[5] += data[3] * rhs.data[2];
    accum.data[6] += data[6] * rhs.data[0];
    accum.data[7] += data[6] * rhs.data[1];
    accum.data[8] += data[6] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[4] * rhs.data[3];
    accum.data[4] += data[4] * rhs.data[4];
    accum.data[5] += data[4] * rhs.data[5];
    accum.data[6] += data[7] * rhs.data[3];
    accum.data[7] += data[7] * rhs.data[4];
    accum.data[8] += data[7] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];
    accum.data[3] += data[5] * rhs.data[6];
    accum.data[4] += data[5] * rhs.data[7];
    accum.data[5] += data[5] * rhs.data[8];
    accum.data[6] += data[8] * rhs.data[6];
    accum.data[7] += data[8] * rhs.data[7];
    accum.data[8] += data[8] * rhs.data[8];

    return accum;
  }

  /// Matrix product of size 3-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> operator*(Matrix<Element, 3, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 3, 3> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 3-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> product(
    Matrix<Element, 3, 4> const &rhs,
    Matrix<Element, 3, 4> accum = Matrix<Element, 3, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[3] * rhs.data[0];
    accum.data[5] += data[3] * rhs.data[1];
    accum.data[6] += data[3] * rhs.data[2];
    accum.data[7] += data[3] * rhs.data[3];
    accum.data[8] += data[6] * rhs.data[0];
    accum.data[9] += data[6] * rhs.data[1];
    accum.data[10] += data[6] * rhs.data[2];
    accum.data[11] += data[6] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[4] * rhs.data[4];
    accum.data[5] += data[4] * rhs.data[5];
    accum.data[6] += data[4] * rhs.data[6];
    accum.data[7] += data[4] * rhs.data[7];
    accum.data[8] += data[7] * rhs.data[4];
    accum.data[9] += data[7] * rhs.data[5];
    accum.data[10] += data[7] * rhs.data[6];
    accum.data[11] += data[7] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];
    accum.data[4] += data[5] * rhs.data[8];
    accum.data[5] += data[5] * rhs.data[9];
    accum.data[6] += data[5] * rhs.data[10];
    accum.data[7] += data[5] * rhs.data[11];
    accum.data[8] += data[8] * rhs.data[8];
    accum.data[9] += data[8] * rhs.data[9];
    accum.data[10] += data[8] * rhs.data[10];
    accum.data[11] += data[8] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 3-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> operator*(Matrix<Element, 3, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];
    accum += data[6];
    accum += data[7];
    accum += data[8];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];
    accum += data[6] * data[6];
    accum += data[7] * data[7];
    accum += data[8] * data[8];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[4];
    accum += data[8];

    return accum;
  }
    
  /// Returns 3-by-3 rotation matrix around the X axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_X(Element theta) {
    Matrix m = identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(1, 1) = c;
    m.at(1, 2) = -s;
    m.at(2, 1) = s;
    m.at(2, 2) = c;

    return m;
  }

  /// Returns 3-by-3 rotation matrix around the Y axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_Y(Element theta) {
    Matrix m = identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(0, 0) = c;
    m.at(2, 0) = -s;
    m.at(0, 2) = s;
    m.at(2, 2) = c;

    return m;
  }

  /// Returns 3-by-3 rotation matrix around the Z axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_Z(Element theta) {
    Matrix m = Matrix::identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(0, 0) = c;
    m.at(0, 1) = -s;
    m.at(1, 0) = s;
    m.at(1, 1) = c;

    return m;
  }

  /// Returns a 3-by-3 rotation matrix around a unit-length axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation(Element theta, Matrix<Element, 3, 1> const &u) {
    Element x = u.data[0];
    Element y = u.data[1];
    Element z = u.data[2];

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    Element one_minus_cos = Element(1) - fast_cos(theta);

    Matrix m;

    m.set_slice_3x3({
      c + x * x * one_minus_cos, x * y * one_minus_cos - z * s, x * z * one_minus_cos + y * s,
      y * x * one_minus_cos * z * s, c + y * y * one_minus_cos, y * z * one_minus_cos - x * s,
      z * x * one_minus_cos - y * s, z * y * one_minus_cos + x * s, c + z * z * one_minus_cos
    });

    return m;
  }

  /// Returns a 3-by-3 reflection about the plane specified by the 
  /// unit-length normal vector n_unit
  CUTLASS_HOST_DEVICE
  static Matrix reflection(Matrix<Element, 3, 1> const &n_unit) {

    Element a = n_unit.data[0];
    Element b = n_unit.data[1];
    Element c = n_unit.data[2];

    Matrix m = Matrix::identity();

    m.set_slice_3x3({
      Element(1) - Element(2) * a * a, Element(-2) * a * b, Element(-2) * a * c,
      Element(-2) * a * b, Element(1) - Element(2) * b * b, Element(-2) * b * c,
      Element(-2) * a * c, Element(-2) * b * c, Element(1) - Element(2) * c * c
    });

    return m;
  }

  /// Computes the determinant of a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Element determinant(Element accum = Element()) const {
    
    accum += at(0, 0) * Matrix<Element, 2, 2>({ at(1, 1), at(1, 2), at(2, 1), at(2, 2) }).determinant();
    accum -= at(0, 1) * Matrix<Element, 2, 2>({ at(1, 0), at(1, 2), at(2, 0), at(2, 2) }).determinant();
    accum += at(0, 2) * Matrix<Element, 2, 2>({ at(1, 0), at(1, 1), at(2, 0), at(2, 1) }).determinant();

    return accum;
  }
  
  /// Computes the inverse of a 3-by-3 matrix given
  /// the matrix's determinant
  CUTLASS_HOST_DEVICE
  Matrix inverse(Element det) const {
    return Matrix(
      at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1),
      at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2),
      at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1),

      at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2),
      at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0),
      at(0, 2) * at(1, 0) - at(0, 0) * at(1, 2),

      at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0),
      at(0, 1) * at(2, 0) - at(0, 0) * at(2, 1),
      at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0)
    ) * (Element(1) / det);
  }
  /// Computes the inverse of a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix inverse() const {
    return inverse(determinant());
  }
    
};

/// Template alias for 3-by-3 matrix
template <typename Element>
using Matrix3x3 = Matrix<Element, 3, 3>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix3x3<Element> make_Matrix3x3(
    Element _0_0, Element _0_1, Element _0_2, 
    Element _1_0, Element _1_1, Element _1_2, 
    Element _2_0, Element _2_1, Element _2_2
) {
  return Matrix3x3<Element>(
  _0_0, _0_1, _0_2, 
  _1_0, _1_1, _1_2, 
  _2_0, _2_1, _2_2 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 3-by-4 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 3, 4> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 3;

  /// Number of columns in matrix
  static int const kColumns = 4;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 12;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 3-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 3-by-4 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3, 
    Element _1_0, Element _1_1, Element _1_2, Element _1_3, 
    Element _2_0, Element _2_1, Element _2_2, Element _2_3
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;  data[3] = _0_3;
    data[4] = _1_0;  data[5] = _1_1;  data[6] = _1_2;  data[7] = _1_3;
    data[8] = _2_0;  data[9] = _2_1;  data[10] = _2_2;  data[11] = _2_3;
  }
    
  /// Constructs a 3-by-4 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 4> const &row_0,
    Matrix<Element, 1, 4> const &row_1,
    Matrix<Element, 1, 4> const &row_2
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_0.data[2];
    data[3] = row_0.data[3];
    data[4] = row_1.data[0];
    data[5] = row_1.data[1];
    data[6] = row_1.data[2];
    data[7] = row_1.data[3];
    data[8] = row_2.data[0];
    data[9] = row_2.data[1];
    data[10] = row_2.data[2];
    data[11] = row_2.data[3];
  }
    
  /// Static method to construct a 3-by-4 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 4, 1> const &column_0,
    Matrix<Element, 4, 1> const &column_1,
    Matrix<Element, 4, 1> const &column_2,
    Matrix<Element, 4, 1> const &column_3
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_2.data[0];
    result.data[3] = column_3.data[0];
    result.data[4] = column_0.data[1];
    result.data[5] = column_1.data[1];
    result.data[6] = column_2.data[1];
    result.data[7] = column_3.data[1];
    result.data[8] = column_0.data[2];
    result.data[9] = column_1.data[2];
    result.data[10] = column_2.data[2];
    result.data[11] = column_3.data[2];
    return result;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;
    m.data[6] = s;
    m.data[7] = s;
    m.data[8] = s;
    m.data[9] = s;
    m.data[10] = s;
    m.data[11] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 3, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[4] = diag.data[1];
    m.data[8] = diag.data[2];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 3> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[4] = diag.data[1];
    m.data[8] = diag.data[2];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> diagonal() const {
    Matrix<Element, 3, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[4];
    diag.data[2] = data[8];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> transpose() const {
    Matrix<Element, 4, 3> mt;
    
    mt.data[0] = data[0];
    mt.data[3] = data[1];
    mt.data[6] = data[2];
    mt.data[9] = data[3];
    mt.data[1] = data[4];
    mt.data[4] = data[5];
    mt.data[7] = data[6];
    mt.data[10] = data[7];
    mt.data[2] = data[8];
    mt.data[5] = data[9];
    mt.data[8] = data[10];
    mt.data[11] = data[11];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 3 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> slice_1x4(int i = 0, int j = 0) const {
    Matrix<Element, 1, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x4(Matrix<Element, 1, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> row(int i) const {
    return slice_1x4(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 4> const &v, int i = 0) {
    return set_slice_1x4(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 4] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 4];
    m.data[3] = data[i * 4 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 4] = m.data[2];
    data[i * 4 + j + 5] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> slice_2x3(int i = 0, int j = 0) const {
    Matrix<Element, 2, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 4];
    m.data[4] = data[i * 4 + j + 5];
    m.data[5] = data[i * 4 + j + 6];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x3(Matrix<Element, 2, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 4] = m.data[3];
    data[i * 4 + j + 5] = m.data[4];
    data[i * 4 + j + 6] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> slice_2x4(int i = 0, int j = 0) const {
    Matrix<Element, 2, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];
    m.data[4] = data[i * 4 + j + 4];
    m.data[5] = data[i * 4 + j + 5];
    m.data[6] = data[i * 4 + j + 6];
    m.data[7] = data[i * 4 + j + 7];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x4(Matrix<Element, 2, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];
    data[i * 4 + j + 4] = m.data[4];
    data[i * 4 + j + 5] = m.data[5];
    data[i * 4 + j + 6] = m.data[6];
    data[i * 4 + j + 7] = m.data[7];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 4];
    m.data[2] = data[i * 4 + j + 8];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 4] = m.data[1];
    data[i * 4 + j + 8] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> column(int j) const {
    return slice_3x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 3, 1> const &v, int j =0) {
    return set_slice_3x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> slice_3x2(int i = 0, int j = 0) const {
    Matrix<Element, 3, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 4];
    m.data[3] = data[i * 4 + j + 5];
    m.data[4] = data[i * 4 + j + 8];
    m.data[5] = data[i * 4 + j + 9];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x2(Matrix<Element, 3, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 4] = m.data[2];
    data[i * 4 + j + 5] = m.data[3];
    data[i * 4 + j + 8] = m.data[4];
    data[i * 4 + j + 9] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> slice_3x3(int i = 0, int j = 0) const {
    Matrix<Element, 3, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 4];
    m.data[4] = data[i * 4 + j + 5];
    m.data[5] = data[i * 4 + j + 6];
    m.data[6] = data[i * 4 + j + 8];
    m.data[7] = data[i * 4 + j + 9];
    m.data[8] = data[i * 4 + j + 10];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x3(Matrix<Element, 3, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 4] = m.data[3];
    data[i * 4 + j + 5] = m.data[4];
    data[i * 4 + j + 6] = m.data[5];
    data[i * 4 + j + 8] = m.data[6];
    data[i * 4 + j + 9] = m.data[7];
    data[i * 4 + j + 10] = m.data[8];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> slice_3x4(int i = 0, int j = 0) const {
    Matrix<Element, 3, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];
    m.data[4] = data[i * 4 + j + 4];
    m.data[5] = data[i * 4 + j + 5];
    m.data[6] = data[i * 4 + j + 6];
    m.data[7] = data[i * 4 + j + 7];
    m.data[8] = data[i * 4 + j + 8];
    m.data[9] = data[i * 4 + j + 9];
    m.data[10] = data[i * 4 + j + 10];
    m.data[11] = data[i * 4 + j + 11];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x4(Matrix<Element, 3, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];
    data[i * 4 + j + 4] = m.data[4];
    data[i * 4 + j + 5] = m.data[5];
    data[i * 4 + j + 6] = m.data[6];
    data[i * 4 + j + 7] = m.data[7];
    data[i * 4 + j + 8] = m.data[8];
    data[i * 4 + j + 9] = m.data[9];
    data[i * 4 + j + 10] = m.data[10];
    data[i * 4 + j + 11] = m.data[11];

    return *this;
  }
    
  /// Forms a 3-by-4 matrix by horizontally concatenating a 3-by-1 matrix with a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 3, 1> const & lhs, Matrix<Element, 3, 3> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0), rhs.at(0, 1), rhs.at(0, 2)
      , lhs.at(1, 0), rhs.at(1, 0), rhs.at(1, 1), rhs.at(1, 2)
      , lhs.at(2, 0), rhs.at(2, 0), rhs.at(2, 1), rhs.at(2, 2));
  }
  
  /// Forms a 3-by-4 matrix by horizontally concatenating a 3-by-2 matrix with a 3-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 3, 2> const & lhs, Matrix<Element, 3, 2> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs.at(0, 0), rhs.at(0, 1)
      , lhs.at(1, 0), lhs.at(1, 1), rhs.at(1, 0), rhs.at(1, 1)
      , lhs.at(2, 0), lhs.at(2, 1), rhs.at(2, 0), rhs.at(2, 1));
  }
  
  /// Forms a 3-by-4 matrix by horizontally concatenating a 3-by-3 matrix with a 3-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 3, 3> const & lhs, Matrix<Element, 3, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), lhs.at(0, 2), rhs.at(0, 0)
      , lhs.at(1, 0), lhs.at(1, 1), lhs.at(1, 2), rhs.at(1, 0)
      , lhs.at(2, 0), lhs.at(2, 1), lhs.at(2, 2), rhs.at(2, 0));
  }
  
  /// Forms a 3-by-4 matrix by vertically concatenating a 1-by-4 matrix with a 2-by-4 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 4> const & upper, Matrix<Element, 2, 4> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2), upper.at(0, 3)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2), lower.at(0, 3)
      , lower.at(1, 0), lower.at(1, 1), lower.at(1, 2), lower.at(1, 3));
  }
  
  /// Forms a 3-by-4 matrix by vertically concatenating a 2-by-4 matrix with a 1-by-4 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 4> const & upper, Matrix<Element, 1, 4> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2), upper.at(0, 3)
      , upper.at(1, 0), upper.at(1, 1), upper.at(1, 2), upper.at(1, 3)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2), lower.at(0, 3));
  }
  
  /// Concatenates this matrix with a a 1-by-4 matrix to form a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> vcat(Matrix<Element, 1, 4> const & rhs) const {
    return Matrix<Element, 4, 4>::vcat(*this, rhs);
  }
    
  /// Forms a 3-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Matrix<Element, 1, 3> const & B,
    Matrix<Element, 2, 1> const & C, Matrix<Element, 2, 3> const & D) {
    return Matrix(
      A, B.at(0, 0), B.at(0, 1), B.at(0, 2)
      , C.at(0, 0), D.at(0, 0), D.at(0, 1), D.at(0, 2)
      , C.at(1, 0), D.at(1, 0), D.at(1, 1), D.at(1, 2)
    );
  }
  
  /// Forms a 3-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 2> const & A, Matrix<Element, 1, 2> const & B,
    Matrix<Element, 2, 2> const & C, Matrix<Element, 2, 2> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0), B.at(0, 1)
      , C.at(0, 0), C.at(0, 1), D.at(0, 0), D.at(0, 1)
      , C.at(1, 0), C.at(1, 1), D.at(1, 0), D.at(1, 1)
    );
  }
  
  /// Forms a 3-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 3> const & A, Element                         B,
    Matrix<Element, 2, 3> const & C, Matrix<Element, 2, 1> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), A.at(0, 2), B
      , C.at(0, 0), C.at(0, 1), C.at(0, 2), D.at(0, 0)
      , C.at(1, 0), C.at(1, 1), C.at(1, 2), D.at(1, 0)
    );
  }
  
  /// Forms a 3-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 1> const & A, Matrix<Element, 2, 3> const & B,
    Element                         C, Matrix<Element, 1, 3> const & D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0), B.at(0, 1), B.at(0, 2)
      , A.at(1, 0), B.at(1, 0), B.at(1, 1), B.at(1, 2)
      , C, D.at(0, 0), D.at(0, 1), D.at(0, 2)
    );
  }
  
  /// Forms a 3-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 2> const & A, Matrix<Element, 2, 2> const & B,
    Matrix<Element, 1, 2> const & C, Matrix<Element, 1, 2> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0), B.at(0, 1)
      , A.at(1, 0), A.at(1, 1), B.at(1, 0), B.at(1, 1)
      , C.at(0, 0), C.at(0, 1), D.at(0, 0), D.at(0, 1)
    );
  }
  
  /// Forms a 3-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 3> const & A, Matrix<Element, 2, 1> const & B,
    Matrix<Element, 1, 3> const & C, Element                         D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), A.at(0, 2), B.at(0, 0)
      , A.at(1, 0), A.at(1, 1), A.at(1, 2), B.at(1, 0)
      , C.at(0, 0), C.at(0, 1), C.at(0, 2), D
    );
  }
  
  /// Elementwise add operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];
    result.data[3] = data[3] + rhs.data[3];

    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];
    result.data[6] = data[6] + rhs.data[6];
    result.data[7] = data[7] + rhs.data[7];

    result.data[8] = data[8] + rhs.data[8];
    result.data[9] = data[9] + rhs.data[9];
    result.data[10] = data[10] + rhs.data[10];
    result.data[11] = data[11] + rhs.data[11];

    return result;
  }
      
  /// Elementwise add operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];
    data[3] += rhs.data[3];

    data[4] += rhs.data[4];
    data[5] += rhs.data[5];
    data[6] += rhs.data[6];
    data[7] += rhs.data[7];

    data[8] += rhs.data[8];
    data[9] += rhs.data[9];
    data[10] += rhs.data[10];
    data[11] += rhs.data[11];

    return *this;
  }
        
  /// Elementwise subtract operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];
    result.data[3] = data[3] - rhs.data[3];

    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];
    result.data[6] = data[6] - rhs.data[6];
    result.data[7] = data[7] - rhs.data[7];

    result.data[8] = data[8] - rhs.data[8];
    result.data[9] = data[9] - rhs.data[9];
    result.data[10] = data[10] - rhs.data[10];
    result.data[11] = data[11] - rhs.data[11];

    return result;
  }
      
  /// Elementwise subtract operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];
    data[3] -= rhs.data[3];

    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];
    data[6] -= rhs.data[6];
    data[7] -= rhs.data[7];

    data[8] -= rhs.data[8];
    data[9] -= rhs.data[9];
    data[10] -= rhs.data[10];
    data[11] -= rhs.data[11];

    return *this;
  }
        
  /// Elementwise multiply operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];
    result.data[3] = data[3] * rhs.data[3];

    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];
    result.data[6] = data[6] * rhs.data[6];
    result.data[7] = data[7] * rhs.data[7];

    result.data[8] = data[8] * rhs.data[8];
    result.data[9] = data[9] * rhs.data[9];
    result.data[10] = data[10] * rhs.data[10];
    result.data[11] = data[11] * rhs.data[11];

    return result;
  }
      
  /// Scalar multiply operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;
    result.data[3] = data[3] * s;

    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;
    result.data[6] = data[6] * s;
    result.data[7] = data[7] * s;

    result.data[8] = data[8] * s;
    result.data[9] = data[9] * s;
    result.data[10] = data[10] * s;
    result.data[11] = data[11] * s;

    return result;
  }

  /// Scalar multiply operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;
    data[3] *= s;

    data[4] *= s;
    data[5] *= s;
    data[6] *= s;
    data[7] *= s;

    data[8] *= s;
    data[9] *= s;
    data[10] *= s;
    data[11] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];
    result.data[3] = data[3] / rhs.data[3];

    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];
    result.data[6] = data[6] / rhs.data[6];
    result.data[7] = data[7] / rhs.data[7];

    result.data[8] = data[8] / rhs.data[8];
    result.data[9] = data[9] / rhs.data[9];
    result.data[10] = data[10] / rhs.data[10];
    result.data[11] = data[11] / rhs.data[11];

    return result;
  }
      
  /// Scalar divide operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;
    result.data[3] = data[3] / s;

    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;
    result.data[6] = data[6] / s;
    result.data[7] = data[7] / s;

    result.data[8] = data[8] / s;
    result.data[9] = data[9] / s;
    result.data[10] = data[10] / s;
    result.data[11] = data[11] / s;

    return result;
  }

  /// Scalar divide operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;
    data[3] /= s;

    data[4] /= s;
    data[5] /= s;
    data[6] /= s;
    data[7] /= s;

    data[8] /= s;
    data[9] /= s;
    data[10] /= s;
    data[11] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (3-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];
    data[3] /= rhs.data[3];

    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];
    data[6] /= rhs.data[6];
    data[7] /= rhs.data[7];

    data[8] /= rhs.data[8];
    data[9] /= rhs.data[9];
    data[10] /= rhs.data[10];
    data[11] /= rhs.data[11];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];
    m.data[6] = -data[6];
    m.data[7] = -data[7];
    m.data[8] = -data[8];
    m.data[9] = -data[9];
    m.data[10] = -data[10];
    m.data[11] = -data[11];

    return m;
  }
  
  /// Matrix product of size 3-by-1-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> product(
    Matrix<Element, 4, 1> const &rhs,
    Matrix<Element, 3, 1> accum = Matrix<Element, 3, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[4] * rhs.data[0];
    accum.data[2] += data[8] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[5] * rhs.data[1];
    accum.data[2] += data[9] * rhs.data[1];

    // k=2
    accum.data[0] += data[2] * rhs.data[2];
    accum.data[1] += data[6] * rhs.data[2];
    accum.data[2] += data[10] * rhs.data[2];

    // k=3
    accum.data[0] += data[3] * rhs.data[3];
    accum.data[1] += data[7] * rhs.data[3];
    accum.data[2] += data[11] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 3-by-1-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> operator*(Matrix<Element, 4, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> product(
    Matrix<Element, 4, 2> const &rhs,
    Matrix<Element, 3, 2> accum = Matrix<Element, 3, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[4] * rhs.data[0];
    accum.data[3] += data[4] * rhs.data[1];
    accum.data[4] += data[8] * rhs.data[0];
    accum.data[5] += data[8] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[5] * rhs.data[2];
    accum.data[3] += data[5] * rhs.data[3];
    accum.data[4] += data[9] * rhs.data[2];
    accum.data[5] += data[9] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];
    accum.data[2] += data[6] * rhs.data[4];
    accum.data[3] += data[6] * rhs.data[5];
    accum.data[4] += data[10] * rhs.data[4];
    accum.data[5] += data[10] * rhs.data[5];

    // k=3
    accum.data[0] += data[3] * rhs.data[6];
    accum.data[1] += data[3] * rhs.data[7];
    accum.data[2] += data[7] * rhs.data[6];
    accum.data[3] += data[7] * rhs.data[7];
    accum.data[4] += data[11] * rhs.data[6];
    accum.data[5] += data[11] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 3-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> operator*(Matrix<Element, 4, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> product(
    Matrix<Element, 4, 3> const &rhs,
    Matrix<Element, 3, 3> accum = Matrix<Element, 3, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[4] * rhs.data[0];
    accum.data[4] += data[4] * rhs.data[1];
    accum.data[5] += data[4] * rhs.data[2];
    accum.data[6] += data[8] * rhs.data[0];
    accum.data[7] += data[8] * rhs.data[1];
    accum.data[8] += data[8] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[5] * rhs.data[3];
    accum.data[4] += data[5] * rhs.data[4];
    accum.data[5] += data[5] * rhs.data[5];
    accum.data[6] += data[9] * rhs.data[3];
    accum.data[7] += data[9] * rhs.data[4];
    accum.data[8] += data[9] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];
    accum.data[3] += data[6] * rhs.data[6];
    accum.data[4] += data[6] * rhs.data[7];
    accum.data[5] += data[6] * rhs.data[8];
    accum.data[6] += data[10] * rhs.data[6];
    accum.data[7] += data[10] * rhs.data[7];
    accum.data[8] += data[10] * rhs.data[8];

    // k=3
    accum.data[0] += data[3] * rhs.data[9];
    accum.data[1] += data[3] * rhs.data[10];
    accum.data[2] += data[3] * rhs.data[11];
    accum.data[3] += data[7] * rhs.data[9];
    accum.data[4] += data[7] * rhs.data[10];
    accum.data[5] += data[7] * rhs.data[11];
    accum.data[6] += data[11] * rhs.data[9];
    accum.data[7] += data[11] * rhs.data[10];
    accum.data[8] += data[11] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 3-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> operator*(Matrix<Element, 4, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> product(
    Matrix<Element, 4, 4> const &rhs,
    Matrix<Element, 3, 4> accum = Matrix<Element, 3, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[4] * rhs.data[0];
    accum.data[5] += data[4] * rhs.data[1];
    accum.data[6] += data[4] * rhs.data[2];
    accum.data[7] += data[4] * rhs.data[3];
    accum.data[8] += data[8] * rhs.data[0];
    accum.data[9] += data[8] * rhs.data[1];
    accum.data[10] += data[8] * rhs.data[2];
    accum.data[11] += data[8] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[5] * rhs.data[4];
    accum.data[5] += data[5] * rhs.data[5];
    accum.data[6] += data[5] * rhs.data[6];
    accum.data[7] += data[5] * rhs.data[7];
    accum.data[8] += data[9] * rhs.data[4];
    accum.data[9] += data[9] * rhs.data[5];
    accum.data[10] += data[9] * rhs.data[6];
    accum.data[11] += data[9] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];
    accum.data[4] += data[6] * rhs.data[8];
    accum.data[5] += data[6] * rhs.data[9];
    accum.data[6] += data[6] * rhs.data[10];
    accum.data[7] += data[6] * rhs.data[11];
    accum.data[8] += data[10] * rhs.data[8];
    accum.data[9] += data[10] * rhs.data[9];
    accum.data[10] += data[10] * rhs.data[10];
    accum.data[11] += data[10] * rhs.data[11];

    // k=3
    accum.data[0] += data[3] * rhs.data[12];
    accum.data[1] += data[3] * rhs.data[13];
    accum.data[2] += data[3] * rhs.data[14];
    accum.data[3] += data[3] * rhs.data[15];
    accum.data[4] += data[7] * rhs.data[12];
    accum.data[5] += data[7] * rhs.data[13];
    accum.data[6] += data[7] * rhs.data[14];
    accum.data[7] += data[7] * rhs.data[15];
    accum.data[8] += data[11] * rhs.data[12];
    accum.data[9] += data[11] * rhs.data[13];
    accum.data[10] += data[11] * rhs.data[14];
    accum.data[11] += data[11] * rhs.data[15];

    return accum;
  }

  /// Matrix product of size 3-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> operator*(Matrix<Element, 4, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 3-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 4, 4> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];
    accum += data[6];
    accum += data[7];
    accum += data[8];
    accum += data[9];
    accum += data[10];
    accum += data[11];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];
    accum += data[6] * data[6];
    accum += data[7] * data[7];
    accum += data[8] * data[8];
    accum += data[9] * data[9];
    accum += data[10] * data[10];
    accum += data[11] * data[11];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[5];
    accum += data[10];

    return accum;
  }
    
};

/// Template alias for 3-by-4 matrix
template <typename Element>
using Matrix3x4 = Matrix<Element, 3, 4>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix3x4<Element> make_Matrix3x4(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3, 
    Element _1_0, Element _1_1, Element _1_2, Element _1_3, 
    Element _2_0, Element _2_1, Element _2_2, Element _2_3
) {
  return Matrix3x4<Element>(
  _0_0, _0_1, _0_2, _0_3, 
  _1_0, _1_1, _1_2, _1_3, 
  _2_0, _2_1, _2_2, _2_3 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 4-by-1 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 4, 1> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 4;

  /// Number of columns in matrix
  static int const kColumns = 1;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 4;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 4-by-1 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 4-by-1 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, 
    Element _1_0, 
    Element _2_0, 
    Element _3_0
  ) {

    data[0] = _0_0;
    data[1] = _1_0;
    data[2] = _2_0;
    data[3] = _3_0;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> transpose() const {
    Matrix<Element, 1, 4> mt;
    
    mt.data[0] = data[0];
    mt.data[1] = data[1];
    mt.data[2] = data[2];
    mt.data[3] = data[3];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 1 + j + 0];
    m.data[1] = data[i * 1 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 1 + j + 0] = m.data[0];
    data[i * 1 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 1 + j + 0];
    m.data[1] = data[i * 1 + j + 1];
    m.data[2] = data[i * 1 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 1 + j + 0] = m.data[0];
    data[i * 1 + j + 1] = m.data[1];
    data[i * 1 + j + 2] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> slice_4x1(int i = 0, int j = 0) const {
    Matrix<Element, 4, 1> m;
    
    m.data[0] = data[i * 1 + j + 0];
    m.data[1] = data[i * 1 + j + 1];
    m.data[2] = data[i * 1 + j + 2];
    m.data[3] = data[i * 1 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x1(Matrix<Element, 4, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 1 + j + 0] = m.data[0];
    data[i * 1 + j + 1] = m.data[1];
    data[i * 1 + j + 2] = m.data[2];
    data[i * 1 + j + 3] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> column(int j) const {
    return slice_4x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 4, 1> const &v, int j =0) {
    return set_slice_4x1(v, 0, j);
  }
    
  /// Concatenates this matrix with a a 4-by-1 matrix to form a 4-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> hcat(Matrix<Element, 4, 1> const & rhs) const {
    return Matrix<Element, 4, 2>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 4-by-2 matrix to form a 4-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> hcat(Matrix<Element, 4, 2> const & rhs) const {
    return Matrix<Element, 4, 3>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 4-by-3 matrix to form a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> hcat(Matrix<Element, 4, 3> const & rhs) const {
    return Matrix<Element, 4, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 4-by-1 matrix by vertically concatenating an Element with a 3-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Element upper, Matrix<Element, 3, 1> const & lower) {
    return Matrix(
      upper
      , lower.at(0, 0)
      , lower.at(1, 0)
      , lower.at(2, 0));
  }
  
  /// Forms a 4-by-1 matrix by vertically concatenating a 2-by-1 matrix with a 2-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 1> const & upper, Matrix<Element, 2, 1> const & lower) {
    return Matrix(
      upper.at(0, 0)
      , upper.at(1, 0)
      , lower.at(0, 0)
      , lower.at(1, 0));
  }
  
  /// Forms a 4-by-1 matrix by vertically concatenating a 3-by-1 matrix with an Element
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 3, 1> const & upper, Element lower) {
    return Matrix(
      upper.at(0, 0)
      , upper.at(1, 0)
      , upper.at(2, 0)
      , lower);
  }
  
  /// Elementwise add operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];

    result.data[1] = data[1] + rhs.data[1];

    result.data[2] = data[2] + rhs.data[2];

    result.data[3] = data[3] + rhs.data[3];

    return result;
  }
      
  /// Elementwise add operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];

    data[1] += rhs.data[1];

    data[2] += rhs.data[2];

    data[3] += rhs.data[3];

    return *this;
  }
        
  /// Elementwise subtract operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];

    result.data[1] = data[1] - rhs.data[1];

    result.data[2] = data[2] - rhs.data[2];

    result.data[3] = data[3] - rhs.data[3];

    return result;
  }
      
  /// Elementwise subtract operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];

    data[1] -= rhs.data[1];

    data[2] -= rhs.data[2];

    data[3] -= rhs.data[3];

    return *this;
  }
        
  /// Elementwise multiply operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];

    result.data[1] = data[1] * rhs.data[1];

    result.data[2] = data[2] * rhs.data[2];

    result.data[3] = data[3] * rhs.data[3];

    return result;
  }
      
  /// Scalar multiply operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;

    result.data[1] = data[1] * s;

    result.data[2] = data[2] * s;

    result.data[3] = data[3] * s;

    return result;
  }

  /// Scalar multiply operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;

    data[1] *= s;

    data[2] *= s;

    data[3] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];

    result.data[1] = data[1] / rhs.data[1];

    result.data[2] = data[2] / rhs.data[2];

    result.data[3] = data[3] / rhs.data[3];

    return result;
  }
      
  /// Scalar divide operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;

    result.data[1] = data[1] / s;

    result.data[2] = data[2] / s;

    result.data[3] = data[3] / s;

    return result;
  }

  /// Scalar divide operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;

    data[1] /= s;

    data[2] /= s;

    data[3] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (4-by-1)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];

    data[1] /= rhs.data[1];

    data[2] /= rhs.data[2];

    data[3] /= rhs.data[3];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];

    return m;
  }
  
  /// Matrix product of size 4-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> product(
    Matrix<Element, 1, 1> const &rhs,
    Matrix<Element, 4, 1> accum = Matrix<Element, 4, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[1] * rhs.data[0];
    accum.data[2] += data[2] * rhs.data[0];
    accum.data[3] += data[3] * rhs.data[0];

    return accum;
  }

  /// Matrix product of size 4-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> operator*(Matrix<Element, 1, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-1-by-1
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 1, 1> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 4-by-2-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> product(
    Matrix<Element, 1, 2> const &rhs,
    Matrix<Element, 4, 2> accum = Matrix<Element, 4, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[1] * rhs.data[0];
    accum.data[3] += data[1] * rhs.data[1];
    accum.data[4] += data[2] * rhs.data[0];
    accum.data[5] += data[2] * rhs.data[1];
    accum.data[6] += data[3] * rhs.data[0];
    accum.data[7] += data[3] * rhs.data[1];

    return accum;
  }

  /// Matrix product of size 4-by-2-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> operator*(Matrix<Element, 1, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-3-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> product(
    Matrix<Element, 1, 3> const &rhs,
    Matrix<Element, 4, 3> accum = Matrix<Element, 4, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[1] * rhs.data[0];
    accum.data[4] += data[1] * rhs.data[1];
    accum.data[5] += data[1] * rhs.data[2];
    accum.data[6] += data[2] * rhs.data[0];
    accum.data[7] += data[2] * rhs.data[1];
    accum.data[8] += data[2] * rhs.data[2];
    accum.data[9] += data[3] * rhs.data[0];
    accum.data[10] += data[3] * rhs.data[1];
    accum.data[11] += data[3] * rhs.data[2];

    return accum;
  }

  /// Matrix product of size 4-by-3-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> operator*(Matrix<Element, 1, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-4-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> product(
    Matrix<Element, 1, 4> const &rhs,
    Matrix<Element, 4, 4> accum = Matrix<Element, 4, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[1] * rhs.data[0];
    accum.data[5] += data[1] * rhs.data[1];
    accum.data[6] += data[1] * rhs.data[2];
    accum.data[7] += data[1] * rhs.data[3];
    accum.data[8] += data[2] * rhs.data[0];
    accum.data[9] += data[2] * rhs.data[1];
    accum.data[10] += data[2] * rhs.data[2];
    accum.data[11] += data[2] * rhs.data[3];
    accum.data[12] += data[3] * rhs.data[0];
    accum.data[13] += data[3] * rhs.data[1];
    accum.data[14] += data[3] * rhs.data[2];
    accum.data[15] += data[3] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 4-by-4-by-1
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> operator*(Matrix<Element, 1, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Dot product of vectors with extent 4
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 4, 1> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    accum += data[3] * rhs.data[3];
    return accum;
  }

  /// Dot product of vectors with extent 4
  CUTLASS_HOST_DEVICE
  Element dot(Matrix<Element, 1, 4> const &rhs, Element accum = Element()) const {
    
    accum += data[0] * rhs.data[0];
    accum += data[1] * rhs.data[1];
    accum += data[2] * rhs.data[2];
    accum += data[3] * rhs.data[3];
    return accum;
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];

    return accum;
  }
    
};

/// Template alias for 4-by-1 matrix
template <typename Element>
using Matrix4x1 = Matrix<Element, 4, 1>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix4x1<Element> make_Matrix4x1(
    Element _0_0, 
    Element _1_0, 
    Element _2_0, 
    Element _3_0
) {
  return Matrix4x1<Element>(
  _0_0, 
  _1_0, 
  _2_0, 
  _3_0 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 4-by-2 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 4, 2> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 4;

  /// Number of columns in matrix
  static int const kColumns = 2;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 8;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 4-by-2 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 4-by-2 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, 
    Element _1_0, Element _1_1, 
    Element _2_0, Element _2_1, 
    Element _3_0, Element _3_1
  ) {

    data[0] = _0_0;  data[1] = _0_1;
    data[2] = _1_0;  data[3] = _1_1;
    data[4] = _2_0;  data[5] = _2_1;
    data[6] = _3_0;  data[7] = _3_1;
  }
    
  /// Constructs a 4-by-2 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 2> const &row_0,
    Matrix<Element, 1, 2> const &row_1,
    Matrix<Element, 1, 2> const &row_2,
    Matrix<Element, 1, 2> const &row_3
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_1.data[0];
    data[3] = row_1.data[1];
    data[4] = row_2.data[0];
    data[5] = row_2.data[1];
    data[6] = row_3.data[0];
    data[7] = row_3.data[1];
  }
    
  /// Static method to construct a 4-by-2 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 2, 1> const &column_0,
    Matrix<Element, 2, 1> const &column_1
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_0.data[1];
    result.data[3] = column_1.data[1];
    result.data[4] = column_0.data[2];
    result.data[5] = column_1.data[2];
    result.data[6] = column_0.data[3];
    result.data[7] = column_1.data[3];
    return result;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;
    m.data[6] = s;
    m.data[7] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 2, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[5] = diag.data[1];
    m.data[10] = diag.data[2];
    m.data[15] = diag.data[3];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 2> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[5] = diag.data[1];
    m.data[10] = diag.data[2];
    m.data[15] = diag.data[3];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> diagonal() const {
    Matrix<Element, 2, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[5];
    diag.data[2] = data[10];
    diag.data[3] = data[15];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> transpose() const {
    Matrix<Element, 2, 4> mt;
    
    mt.data[0] = data[0];
    mt.data[4] = data[1];
    mt.data[1] = data[2];
    mt.data[5] = data[3];
    mt.data[2] = data[4];
    mt.data[6] = data[5];
    mt.data[3] = data[6];
    mt.data[7] = data[7];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> row(int i) const {
    return slice_1x2(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 2> const &v, int i = 0) {
    return set_slice_1x2(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 2] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];
    m.data[2] = data[i * 2 + j + 2];
    m.data[3] = data[i * 2 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];
    data[i * 2 + j + 2] = m.data[2];
    data[i * 2 + j + 3] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 2];
    m.data[2] = data[i * 2 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 2] = m.data[1];
    data[i * 2 + j + 4] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> slice_3x2(int i = 0, int j = 0) const {
    Matrix<Element, 3, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];
    m.data[2] = data[i * 2 + j + 2];
    m.data[3] = data[i * 2 + j + 3];
    m.data[4] = data[i * 2 + j + 4];
    m.data[5] = data[i * 2 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x2(Matrix<Element, 3, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];
    data[i * 2 + j + 2] = m.data[2];
    data[i * 2 + j + 3] = m.data[3];
    data[i * 2 + j + 4] = m.data[4];
    data[i * 2 + j + 5] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> slice_4x1(int i = 0, int j = 0) const {
    Matrix<Element, 4, 1> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 2];
    m.data[2] = data[i * 2 + j + 4];
    m.data[3] = data[i * 2 + j + 6];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x1(Matrix<Element, 4, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 2] = m.data[1];
    data[i * 2 + j + 4] = m.data[2];
    data[i * 2 + j + 6] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> column(int j) const {
    return slice_4x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 4, 1> const &v, int j =0) {
    return set_slice_4x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> slice_4x2(int i = 0, int j = 0) const {
    Matrix<Element, 4, 2> m;
    
    m.data[0] = data[i * 2 + j + 0];
    m.data[1] = data[i * 2 + j + 1];
    m.data[2] = data[i * 2 + j + 2];
    m.data[3] = data[i * 2 + j + 3];
    m.data[4] = data[i * 2 + j + 4];
    m.data[5] = data[i * 2 + j + 5];
    m.data[6] = data[i * 2 + j + 6];
    m.data[7] = data[i * 2 + j + 7];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x2(Matrix<Element, 4, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 2 + j + 0] = m.data[0];
    data[i * 2 + j + 1] = m.data[1];
    data[i * 2 + j + 2] = m.data[2];
    data[i * 2 + j + 3] = m.data[3];
    data[i * 2 + j + 4] = m.data[4];
    data[i * 2 + j + 5] = m.data[5];
    data[i * 2 + j + 6] = m.data[6];
    data[i * 2 + j + 7] = m.data[7];

    return *this;
  }
    
  /// Forms a 4-by-2 matrix by horizontally concatenating a 4-by-1 matrix with a 4-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 4, 1> const & lhs, Matrix<Element, 4, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0)
      , lhs.at(1, 0), rhs.at(1, 0)
      , lhs.at(2, 0), rhs.at(2, 0)
      , lhs.at(3, 0), rhs.at(3, 0));
  }
  
  /// Concatenates this matrix with a a 4-by-1 matrix to form a 4-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> hcat(Matrix<Element, 4, 1> const & rhs) const {
    return Matrix<Element, 4, 3>::hcat(*this, rhs);
  }
    
  /// Concatenates this matrix with a a 4-by-2 matrix to form a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> hcat(Matrix<Element, 4, 2> const & rhs) const {
    return Matrix<Element, 4, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 4-by-2 matrix by vertically concatenating a 1-by-2 matrix with a 3-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 2> const & upper, Matrix<Element, 3, 2> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1)
      , lower.at(0, 0), lower.at(0, 1)
      , lower.at(1, 0), lower.at(1, 1)
      , lower.at(2, 0), lower.at(2, 1));
  }
  
  /// Forms a 4-by-2 matrix by vertically concatenating a 2-by-2 matrix with a 2-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 2> const & upper, Matrix<Element, 2, 2> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1)
      , upper.at(1, 0), upper.at(1, 1)
      , lower.at(0, 0), lower.at(0, 1)
      , lower.at(1, 0), lower.at(1, 1));
  }
  
  /// Forms a 4-by-2 matrix by vertically concatenating a 3-by-2 matrix with a 1-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 3, 2> const & upper, Matrix<Element, 1, 2> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1)
      , upper.at(1, 0), upper.at(1, 1)
      , upper.at(2, 0), upper.at(2, 1)
      , lower.at(0, 0), lower.at(0, 1));
  }
  
  /// Forms a 4-by-2 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Element                         B,
    Matrix<Element, 3, 1> const & C, Matrix<Element, 3, 1> const & D) {
    return Matrix(
      A, B
      , C.at(0, 0), D.at(0, 0)
      , C.at(1, 0), D.at(1, 0)
      , C.at(2, 0), D.at(2, 0)
    );
  }
  
  /// Forms a 4-by-2 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 1> const & A, Matrix<Element, 2, 1> const & B,
    Matrix<Element, 2, 1> const & C, Matrix<Element, 2, 1> const & D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0)
      , A.at(1, 0), B.at(1, 0)
      , C.at(0, 0), D.at(0, 0)
      , C.at(1, 0), D.at(1, 0)
    );
  }
  
  /// Forms a 4-by-2 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 3, 1> const & A, Matrix<Element, 3, 1> const & B,
    Element                         C, Element                         D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0)
      , A.at(1, 0), B.at(1, 0)
      , A.at(2, 0), B.at(2, 0)
      , C, D
    );
  }
  
  /// Elementwise add operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];

    result.data[2] = data[2] + rhs.data[2];
    result.data[3] = data[3] + rhs.data[3];

    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];

    result.data[6] = data[6] + rhs.data[6];
    result.data[7] = data[7] + rhs.data[7];

    return result;
  }
      
  /// Elementwise add operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];

    data[2] += rhs.data[2];
    data[3] += rhs.data[3];

    data[4] += rhs.data[4];
    data[5] += rhs.data[5];

    data[6] += rhs.data[6];
    data[7] += rhs.data[7];

    return *this;
  }
        
  /// Elementwise subtract operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];

    result.data[2] = data[2] - rhs.data[2];
    result.data[3] = data[3] - rhs.data[3];

    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];

    result.data[6] = data[6] - rhs.data[6];
    result.data[7] = data[7] - rhs.data[7];

    return result;
  }
      
  /// Elementwise subtract operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];

    data[2] -= rhs.data[2];
    data[3] -= rhs.data[3];

    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];

    data[6] -= rhs.data[6];
    data[7] -= rhs.data[7];

    return *this;
  }
        
  /// Elementwise multiply operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];

    result.data[2] = data[2] * rhs.data[2];
    result.data[3] = data[3] * rhs.data[3];

    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];

    result.data[6] = data[6] * rhs.data[6];
    result.data[7] = data[7] * rhs.data[7];

    return result;
  }
      
  /// Scalar multiply operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;

    result.data[2] = data[2] * s;
    result.data[3] = data[3] * s;

    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;

    result.data[6] = data[6] * s;
    result.data[7] = data[7] * s;

    return result;
  }

  /// Scalar multiply operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;

    data[2] *= s;
    data[3] *= s;

    data[4] *= s;
    data[5] *= s;

    data[6] *= s;
    data[7] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];

    result.data[2] = data[2] / rhs.data[2];
    result.data[3] = data[3] / rhs.data[3];

    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];

    result.data[6] = data[6] / rhs.data[6];
    result.data[7] = data[7] / rhs.data[7];

    return result;
  }
      
  /// Scalar divide operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;

    result.data[2] = data[2] / s;
    result.data[3] = data[3] / s;

    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;

    result.data[6] = data[6] / s;
    result.data[7] = data[7] / s;

    return result;
  }

  /// Scalar divide operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;

    data[2] /= s;
    data[3] /= s;

    data[4] /= s;
    data[5] /= s;

    data[6] /= s;
    data[7] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (4-by-2)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];

    data[2] /= rhs.data[2];
    data[3] /= rhs.data[3];

    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];

    data[6] /= rhs.data[6];
    data[7] /= rhs.data[7];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];
    m.data[6] = -data[6];
    m.data[7] = -data[7];

    return m;
  }
  
  /// Matrix product of size 4-by-1-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> product(
    Matrix<Element, 2, 1> const &rhs,
    Matrix<Element, 4, 1> accum = Matrix<Element, 4, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[2] * rhs.data[0];
    accum.data[2] += data[4] * rhs.data[0];
    accum.data[3] += data[6] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[3] * rhs.data[1];
    accum.data[2] += data[5] * rhs.data[1];
    accum.data[3] += data[7] * rhs.data[1];

    return accum;
  }

  /// Matrix product of size 4-by-1-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> operator*(Matrix<Element, 2, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> product(
    Matrix<Element, 2, 2> const &rhs,
    Matrix<Element, 4, 2> accum = Matrix<Element, 4, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[2] * rhs.data[0];
    accum.data[3] += data[2] * rhs.data[1];
    accum.data[4] += data[4] * rhs.data[0];
    accum.data[5] += data[4] * rhs.data[1];
    accum.data[6] += data[6] * rhs.data[0];
    accum.data[7] += data[6] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[3] * rhs.data[2];
    accum.data[3] += data[3] * rhs.data[3];
    accum.data[4] += data[5] * rhs.data[2];
    accum.data[5] += data[5] * rhs.data[3];
    accum.data[6] += data[7] * rhs.data[2];
    accum.data[7] += data[7] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 4-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> operator*(Matrix<Element, 2, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-2-by-2
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 2, 2> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 4-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> product(
    Matrix<Element, 2, 3> const &rhs,
    Matrix<Element, 4, 3> accum = Matrix<Element, 4, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[2] * rhs.data[0];
    accum.data[4] += data[2] * rhs.data[1];
    accum.data[5] += data[2] * rhs.data[2];
    accum.data[6] += data[4] * rhs.data[0];
    accum.data[7] += data[4] * rhs.data[1];
    accum.data[8] += data[4] * rhs.data[2];
    accum.data[9] += data[6] * rhs.data[0];
    accum.data[10] += data[6] * rhs.data[1];
    accum.data[11] += data[6] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[3] * rhs.data[3];
    accum.data[4] += data[3] * rhs.data[4];
    accum.data[5] += data[3] * rhs.data[5];
    accum.data[6] += data[5] * rhs.data[3];
    accum.data[7] += data[5] * rhs.data[4];
    accum.data[8] += data[5] * rhs.data[5];
    accum.data[9] += data[7] * rhs.data[3];
    accum.data[10] += data[7] * rhs.data[4];
    accum.data[11] += data[7] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 4-by-3-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> operator*(Matrix<Element, 2, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> product(
    Matrix<Element, 2, 4> const &rhs,
    Matrix<Element, 4, 4> accum = Matrix<Element, 4, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[2] * rhs.data[0];
    accum.data[5] += data[2] * rhs.data[1];
    accum.data[6] += data[2] * rhs.data[2];
    accum.data[7] += data[2] * rhs.data[3];
    accum.data[8] += data[4] * rhs.data[0];
    accum.data[9] += data[4] * rhs.data[1];
    accum.data[10] += data[4] * rhs.data[2];
    accum.data[11] += data[4] * rhs.data[3];
    accum.data[12] += data[6] * rhs.data[0];
    accum.data[13] += data[6] * rhs.data[1];
    accum.data[14] += data[6] * rhs.data[2];
    accum.data[15] += data[6] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[3] * rhs.data[4];
    accum.data[5] += data[3] * rhs.data[5];
    accum.data[6] += data[3] * rhs.data[6];
    accum.data[7] += data[3] * rhs.data[7];
    accum.data[8] += data[5] * rhs.data[4];
    accum.data[9] += data[5] * rhs.data[5];
    accum.data[10] += data[5] * rhs.data[6];
    accum.data[11] += data[5] * rhs.data[7];
    accum.data[12] += data[7] * rhs.data[4];
    accum.data[13] += data[7] * rhs.data[5];
    accum.data[14] += data[7] * rhs.data[6];
    accum.data[15] += data[7] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 4-by-4-by-2
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> operator*(Matrix<Element, 2, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];
    accum += data[6];
    accum += data[7];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];
    accum += data[6] * data[6];
    accum += data[7] * data[7];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[3];

    return accum;
  }
    
};

/// Template alias for 4-by-2 matrix
template <typename Element>
using Matrix4x2 = Matrix<Element, 4, 2>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix4x2<Element> make_Matrix4x2(
    Element _0_0, Element _0_1, 
    Element _1_0, Element _1_1, 
    Element _2_0, Element _2_1, 
    Element _3_0, Element _3_1
) {
  return Matrix4x2<Element>(
  _0_0, _0_1, 
  _1_0, _1_1, 
  _2_0, _2_1, 
  _3_0, _3_1 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 4-by-3 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 4, 3> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 4;

  /// Number of columns in matrix
  static int const kColumns = 3;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 12;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 4-by-3 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 4-by-3 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2, 
    Element _1_0, Element _1_1, Element _1_2, 
    Element _2_0, Element _2_1, Element _2_2, 
    Element _3_0, Element _3_1, Element _3_2
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;
    data[3] = _1_0;  data[4] = _1_1;  data[5] = _1_2;
    data[6] = _2_0;  data[7] = _2_1;  data[8] = _2_2;
    data[9] = _3_0;  data[10] = _3_1;  data[11] = _3_2;
  }
    
  /// Constructs a 4-by-3 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 3> const &row_0,
    Matrix<Element, 1, 3> const &row_1,
    Matrix<Element, 1, 3> const &row_2,
    Matrix<Element, 1, 3> const &row_3
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_0.data[2];
    data[3] = row_1.data[0];
    data[4] = row_1.data[1];
    data[5] = row_1.data[2];
    data[6] = row_2.data[0];
    data[7] = row_2.data[1];
    data[8] = row_2.data[2];
    data[9] = row_3.data[0];
    data[10] = row_3.data[1];
    data[11] = row_3.data[2];
  }
    
  /// Static method to construct a 4-by-3 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 3, 1> const &column_0,
    Matrix<Element, 3, 1> const &column_1,
    Matrix<Element, 3, 1> const &column_2
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_2.data[0];
    result.data[3] = column_0.data[1];
    result.data[4] = column_1.data[1];
    result.data[5] = column_2.data[1];
    result.data[6] = column_0.data[2];
    result.data[7] = column_1.data[2];
    result.data[8] = column_2.data[2];
    result.data[9] = column_0.data[3];
    result.data[10] = column_1.data[3];
    result.data[11] = column_2.data[3];
    return result;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;
    m.data[6] = s;
    m.data[7] = s;
    m.data[8] = s;
    m.data[9] = s;
    m.data[10] = s;
    m.data[11] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 3, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[5] = diag.data[1];
    m.data[10] = diag.data[2];
    m.data[15] = diag.data[3];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 3> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[5] = diag.data[1];
    m.data[10] = diag.data[2];
    m.data[15] = diag.data[3];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> diagonal() const {
    Matrix<Element, 3, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[5];
    diag.data[2] = data[10];
    diag.data[3] = data[15];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> transpose() const {
    Matrix<Element, 3, 4> mt;
    
    mt.data[0] = data[0];
    mt.data[4] = data[1];
    mt.data[8] = data[2];
    mt.data[1] = data[3];
    mt.data[5] = data[4];
    mt.data[9] = data[5];
    mt.data[2] = data[6];
    mt.data[6] = data[7];
    mt.data[10] = data[8];
    mt.data[3] = data[9];
    mt.data[7] = data[10];
    mt.data[11] = data[11];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> row(int i) const {
    return slice_1x3(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 3> const &v, int i = 0) {
    return set_slice_1x3(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 3] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 3];
    m.data[3] = data[i * 3 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 3] = m.data[2];
    data[i * 3 + j + 4] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> slice_2x3(int i = 0, int j = 0) const {
    Matrix<Element, 2, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];
    m.data[3] = data[i * 3 + j + 3];
    m.data[4] = data[i * 3 + j + 4];
    m.data[5] = data[i * 3 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x3(Matrix<Element, 2, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];
    data[i * 3 + j + 3] = m.data[3];
    data[i * 3 + j + 4] = m.data[4];
    data[i * 3 + j + 5] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 3];
    m.data[2] = data[i * 3 + j + 6];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 3] = m.data[1];
    data[i * 3 + j + 6] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> slice_3x2(int i = 0, int j = 0) const {
    Matrix<Element, 3, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 3];
    m.data[3] = data[i * 3 + j + 4];
    m.data[4] = data[i * 3 + j + 6];
    m.data[5] = data[i * 3 + j + 7];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x2(Matrix<Element, 3, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 3] = m.data[2];
    data[i * 3 + j + 4] = m.data[3];
    data[i * 3 + j + 6] = m.data[4];
    data[i * 3 + j + 7] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> slice_3x3(int i = 0, int j = 0) const {
    Matrix<Element, 3, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];
    m.data[3] = data[i * 3 + j + 3];
    m.data[4] = data[i * 3 + j + 4];
    m.data[5] = data[i * 3 + j + 5];
    m.data[6] = data[i * 3 + j + 6];
    m.data[7] = data[i * 3 + j + 7];
    m.data[8] = data[i * 3 + j + 8];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x3(Matrix<Element, 3, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];
    data[i * 3 + j + 3] = m.data[3];
    data[i * 3 + j + 4] = m.data[4];
    data[i * 3 + j + 5] = m.data[5];
    data[i * 3 + j + 6] = m.data[6];
    data[i * 3 + j + 7] = m.data[7];
    data[i * 3 + j + 8] = m.data[8];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> slice_4x1(int i = 0, int j = 0) const {
    Matrix<Element, 4, 1> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 3];
    m.data[2] = data[i * 3 + j + 6];
    m.data[3] = data[i * 3 + j + 9];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x1(Matrix<Element, 4, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 3] = m.data[1];
    data[i * 3 + j + 6] = m.data[2];
    data[i * 3 + j + 9] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> column(int j) const {
    return slice_4x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 4, 1> const &v, int j =0) {
    return set_slice_4x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> slice_4x2(int i = 0, int j = 0) const {
    Matrix<Element, 4, 2> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 3];
    m.data[3] = data[i * 3 + j + 4];
    m.data[4] = data[i * 3 + j + 6];
    m.data[5] = data[i * 3 + j + 7];
    m.data[6] = data[i * 3 + j + 9];
    m.data[7] = data[i * 3 + j + 10];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x2(Matrix<Element, 4, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 3] = m.data[2];
    data[i * 3 + j + 4] = m.data[3];
    data[i * 3 + j + 6] = m.data[4];
    data[i * 3 + j + 7] = m.data[5];
    data[i * 3 + j + 9] = m.data[6];
    data[i * 3 + j + 10] = m.data[7];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> slice_4x3(int i = 0, int j = 0) const {
    Matrix<Element, 4, 3> m;
    
    m.data[0] = data[i * 3 + j + 0];
    m.data[1] = data[i * 3 + j + 1];
    m.data[2] = data[i * 3 + j + 2];
    m.data[3] = data[i * 3 + j + 3];
    m.data[4] = data[i * 3 + j + 4];
    m.data[5] = data[i * 3 + j + 5];
    m.data[6] = data[i * 3 + j + 6];
    m.data[7] = data[i * 3 + j + 7];
    m.data[8] = data[i * 3 + j + 8];
    m.data[9] = data[i * 3 + j + 9];
    m.data[10] = data[i * 3 + j + 10];
    m.data[11] = data[i * 3 + j + 11];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x3(Matrix<Element, 4, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 3 + j + 0] = m.data[0];
    data[i * 3 + j + 1] = m.data[1];
    data[i * 3 + j + 2] = m.data[2];
    data[i * 3 + j + 3] = m.data[3];
    data[i * 3 + j + 4] = m.data[4];
    data[i * 3 + j + 5] = m.data[5];
    data[i * 3 + j + 6] = m.data[6];
    data[i * 3 + j + 7] = m.data[7];
    data[i * 3 + j + 8] = m.data[8];
    data[i * 3 + j + 9] = m.data[9];
    data[i * 3 + j + 10] = m.data[10];
    data[i * 3 + j + 11] = m.data[11];

    return *this;
  }
    
  /// Forms a 4-by-3 matrix by horizontally concatenating a 4-by-1 matrix with a 4-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 4, 1> const & lhs, Matrix<Element, 4, 2> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0), rhs.at(0, 1)
      , lhs.at(1, 0), rhs.at(1, 0), rhs.at(1, 1)
      , lhs.at(2, 0), rhs.at(2, 0), rhs.at(2, 1)
      , lhs.at(3, 0), rhs.at(3, 0), rhs.at(3, 1));
  }
  
  /// Forms a 4-by-3 matrix by horizontally concatenating a 4-by-2 matrix with a 4-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 4, 2> const & lhs, Matrix<Element, 4, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs.at(0, 0)
      , lhs.at(1, 0), lhs.at(1, 1), rhs.at(1, 0)
      , lhs.at(2, 0), lhs.at(2, 1), rhs.at(2, 0)
      , lhs.at(3, 0), lhs.at(3, 1), rhs.at(3, 0));
  }
  
  /// Concatenates this matrix with a a 4-by-1 matrix to form a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> hcat(Matrix<Element, 4, 1> const & rhs) const {
    return Matrix<Element, 4, 4>::hcat(*this, rhs);
  }
    
  /// Forms a 4-by-3 matrix by vertically concatenating a 1-by-3 matrix with a 3-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 3> const & upper, Matrix<Element, 3, 3> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2)
      , lower.at(1, 0), lower.at(1, 1), lower.at(1, 2)
      , lower.at(2, 0), lower.at(2, 1), lower.at(2, 2));
  }
  
  /// Forms a 4-by-3 matrix by vertically concatenating a 2-by-3 matrix with a 2-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 3> const & upper, Matrix<Element, 2, 3> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2)
      , upper.at(1, 0), upper.at(1, 1), upper.at(1, 2)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2)
      , lower.at(1, 0), lower.at(1, 1), lower.at(1, 2));
  }
  
  /// Forms a 4-by-3 matrix by vertically concatenating a 3-by-3 matrix with a 1-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 3, 3> const & upper, Matrix<Element, 1, 3> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2)
      , upper.at(1, 0), upper.at(1, 1), upper.at(1, 2)
      , upper.at(2, 0), upper.at(2, 1), upper.at(2, 2)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2));
  }
  
  /// Forms a 4-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Matrix<Element, 1, 2> const & B,
    Matrix<Element, 3, 1> const & C, Matrix<Element, 3, 2> const & D) {
    return Matrix(
      A, B.at(0, 0), B.at(0, 1)
      , C.at(0, 0), D.at(0, 0), D.at(0, 1)
      , C.at(1, 0), D.at(1, 0), D.at(1, 1)
      , C.at(2, 0), D.at(2, 0), D.at(2, 1)
    );
  }
  
  /// Forms a 4-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 2> const & A, Element                         B,
    Matrix<Element, 3, 2> const & C, Matrix<Element, 3, 1> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B
      , C.at(0, 0), C.at(0, 1), D.at(0, 0)
      , C.at(1, 0), C.at(1, 1), D.at(1, 0)
      , C.at(2, 0), C.at(2, 1), D.at(2, 0)
    );
  }
  
  /// Forms a 4-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 1> const & A, Matrix<Element, 2, 2> const & B,
    Matrix<Element, 2, 1> const & C, Matrix<Element, 2, 2> const & D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0), B.at(0, 1)
      , A.at(1, 0), B.at(1, 0), B.at(1, 1)
      , C.at(0, 0), D.at(0, 0), D.at(0, 1)
      , C.at(1, 0), D.at(1, 0), D.at(1, 1)
    );
  }
  
  /// Forms a 4-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 2> const & A, Matrix<Element, 2, 1> const & B,
    Matrix<Element, 2, 2> const & C, Matrix<Element, 2, 1> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0)
      , A.at(1, 0), A.at(1, 1), B.at(1, 0)
      , C.at(0, 0), C.at(0, 1), D.at(0, 0)
      , C.at(1, 0), C.at(1, 1), D.at(1, 0)
    );
  }
  
  /// Forms a 4-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 3, 1> const & A, Matrix<Element, 3, 2> const & B,
    Element                         C, Matrix<Element, 1, 2> const & D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0), B.at(0, 1)
      , A.at(1, 0), B.at(1, 0), B.at(1, 1)
      , A.at(2, 0), B.at(2, 0), B.at(2, 1)
      , C, D.at(0, 0), D.at(0, 1)
    );
  }
  
  /// Forms a 4-by-3 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 3, 2> const & A, Matrix<Element, 3, 1> const & B,
    Matrix<Element, 1, 2> const & C, Element                         D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0)
      , A.at(1, 0), A.at(1, 1), B.at(1, 0)
      , A.at(2, 0), A.at(2, 1), B.at(2, 0)
      , C.at(0, 0), C.at(0, 1), D
    );
  }
  
  /// Elementwise add operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];

    result.data[3] = data[3] + rhs.data[3];
    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];

    result.data[6] = data[6] + rhs.data[6];
    result.data[7] = data[7] + rhs.data[7];
    result.data[8] = data[8] + rhs.data[8];

    result.data[9] = data[9] + rhs.data[9];
    result.data[10] = data[10] + rhs.data[10];
    result.data[11] = data[11] + rhs.data[11];

    return result;
  }
      
  /// Elementwise add operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];

    data[3] += rhs.data[3];
    data[4] += rhs.data[4];
    data[5] += rhs.data[5];

    data[6] += rhs.data[6];
    data[7] += rhs.data[7];
    data[8] += rhs.data[8];

    data[9] += rhs.data[9];
    data[10] += rhs.data[10];
    data[11] += rhs.data[11];

    return *this;
  }
        
  /// Elementwise subtract operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];

    result.data[3] = data[3] - rhs.data[3];
    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];

    result.data[6] = data[6] - rhs.data[6];
    result.data[7] = data[7] - rhs.data[7];
    result.data[8] = data[8] - rhs.data[8];

    result.data[9] = data[9] - rhs.data[9];
    result.data[10] = data[10] - rhs.data[10];
    result.data[11] = data[11] - rhs.data[11];

    return result;
  }
      
  /// Elementwise subtract operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];

    data[3] -= rhs.data[3];
    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];

    data[6] -= rhs.data[6];
    data[7] -= rhs.data[7];
    data[8] -= rhs.data[8];

    data[9] -= rhs.data[9];
    data[10] -= rhs.data[10];
    data[11] -= rhs.data[11];

    return *this;
  }
        
  /// Elementwise multiply operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];

    result.data[3] = data[3] * rhs.data[3];
    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];

    result.data[6] = data[6] * rhs.data[6];
    result.data[7] = data[7] * rhs.data[7];
    result.data[8] = data[8] * rhs.data[8];

    result.data[9] = data[9] * rhs.data[9];
    result.data[10] = data[10] * rhs.data[10];
    result.data[11] = data[11] * rhs.data[11];

    return result;
  }
      
  /// Scalar multiply operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;

    result.data[3] = data[3] * s;
    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;

    result.data[6] = data[6] * s;
    result.data[7] = data[7] * s;
    result.data[8] = data[8] * s;

    result.data[9] = data[9] * s;
    result.data[10] = data[10] * s;
    result.data[11] = data[11] * s;

    return result;
  }

  /// Scalar multiply operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;

    data[3] *= s;
    data[4] *= s;
    data[5] *= s;

    data[6] *= s;
    data[7] *= s;
    data[8] *= s;

    data[9] *= s;
    data[10] *= s;
    data[11] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];

    result.data[3] = data[3] / rhs.data[3];
    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];

    result.data[6] = data[6] / rhs.data[6];
    result.data[7] = data[7] / rhs.data[7];
    result.data[8] = data[8] / rhs.data[8];

    result.data[9] = data[9] / rhs.data[9];
    result.data[10] = data[10] / rhs.data[10];
    result.data[11] = data[11] / rhs.data[11];

    return result;
  }
      
  /// Scalar divide operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;

    result.data[3] = data[3] / s;
    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;

    result.data[6] = data[6] / s;
    result.data[7] = data[7] / s;
    result.data[8] = data[8] / s;

    result.data[9] = data[9] / s;
    result.data[10] = data[10] / s;
    result.data[11] = data[11] / s;

    return result;
  }

  /// Scalar divide operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;

    data[3] /= s;
    data[4] /= s;
    data[5] /= s;

    data[6] /= s;
    data[7] /= s;
    data[8] /= s;

    data[9] /= s;
    data[10] /= s;
    data[11] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (4-by-3)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];

    data[3] /= rhs.data[3];
    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];

    data[6] /= rhs.data[6];
    data[7] /= rhs.data[7];
    data[8] /= rhs.data[8];

    data[9] /= rhs.data[9];
    data[10] /= rhs.data[10];
    data[11] /= rhs.data[11];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];
    m.data[6] = -data[6];
    m.data[7] = -data[7];
    m.data[8] = -data[8];
    m.data[9] = -data[9];
    m.data[10] = -data[10];
    m.data[11] = -data[11];

    return m;
  }
  
  /// Matrix product of size 4-by-1-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> product(
    Matrix<Element, 3, 1> const &rhs,
    Matrix<Element, 4, 1> accum = Matrix<Element, 4, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[3] * rhs.data[0];
    accum.data[2] += data[6] * rhs.data[0];
    accum.data[3] += data[9] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[4] * rhs.data[1];
    accum.data[2] += data[7] * rhs.data[1];
    accum.data[3] += data[10] * rhs.data[1];

    // k=2
    accum.data[0] += data[2] * rhs.data[2];
    accum.data[1] += data[5] * rhs.data[2];
    accum.data[2] += data[8] * rhs.data[2];
    accum.data[3] += data[11] * rhs.data[2];

    return accum;
  }

  /// Matrix product of size 4-by-1-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> operator*(Matrix<Element, 3, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> product(
    Matrix<Element, 3, 2> const &rhs,
    Matrix<Element, 4, 2> accum = Matrix<Element, 4, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[3] * rhs.data[0];
    accum.data[3] += data[3] * rhs.data[1];
    accum.data[4] += data[6] * rhs.data[0];
    accum.data[5] += data[6] * rhs.data[1];
    accum.data[6] += data[9] * rhs.data[0];
    accum.data[7] += data[9] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[4] * rhs.data[2];
    accum.data[3] += data[4] * rhs.data[3];
    accum.data[4] += data[7] * rhs.data[2];
    accum.data[5] += data[7] * rhs.data[3];
    accum.data[6] += data[10] * rhs.data[2];
    accum.data[7] += data[10] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];
    accum.data[2] += data[5] * rhs.data[4];
    accum.data[3] += data[5] * rhs.data[5];
    accum.data[4] += data[8] * rhs.data[4];
    accum.data[5] += data[8] * rhs.data[5];
    accum.data[6] += data[11] * rhs.data[4];
    accum.data[7] += data[11] * rhs.data[5];

    return accum;
  }

  /// Matrix product of size 4-by-2-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> operator*(Matrix<Element, 3, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> product(
    Matrix<Element, 3, 3> const &rhs,
    Matrix<Element, 4, 3> accum = Matrix<Element, 4, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[3] * rhs.data[0];
    accum.data[4] += data[3] * rhs.data[1];
    accum.data[5] += data[3] * rhs.data[2];
    accum.data[6] += data[6] * rhs.data[0];
    accum.data[7] += data[6] * rhs.data[1];
    accum.data[8] += data[6] * rhs.data[2];
    accum.data[9] += data[9] * rhs.data[0];
    accum.data[10] += data[9] * rhs.data[1];
    accum.data[11] += data[9] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[4] * rhs.data[3];
    accum.data[4] += data[4] * rhs.data[4];
    accum.data[5] += data[4] * rhs.data[5];
    accum.data[6] += data[7] * rhs.data[3];
    accum.data[7] += data[7] * rhs.data[4];
    accum.data[8] += data[7] * rhs.data[5];
    accum.data[9] += data[10] * rhs.data[3];
    accum.data[10] += data[10] * rhs.data[4];
    accum.data[11] += data[10] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];
    accum.data[3] += data[5] * rhs.data[6];
    accum.data[4] += data[5] * rhs.data[7];
    accum.data[5] += data[5] * rhs.data[8];
    accum.data[6] += data[8] * rhs.data[6];
    accum.data[7] += data[8] * rhs.data[7];
    accum.data[8] += data[8] * rhs.data[8];
    accum.data[9] += data[11] * rhs.data[6];
    accum.data[10] += data[11] * rhs.data[7];
    accum.data[11] += data[11] * rhs.data[8];

    return accum;
  }

  /// Matrix product of size 4-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> operator*(Matrix<Element, 3, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-3-by-3
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 3, 3> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Matrix product of size 4-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> product(
    Matrix<Element, 3, 4> const &rhs,
    Matrix<Element, 4, 4> accum = Matrix<Element, 4, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[3] * rhs.data[0];
    accum.data[5] += data[3] * rhs.data[1];
    accum.data[6] += data[3] * rhs.data[2];
    accum.data[7] += data[3] * rhs.data[3];
    accum.data[8] += data[6] * rhs.data[0];
    accum.data[9] += data[6] * rhs.data[1];
    accum.data[10] += data[6] * rhs.data[2];
    accum.data[11] += data[6] * rhs.data[3];
    accum.data[12] += data[9] * rhs.data[0];
    accum.data[13] += data[9] * rhs.data[1];
    accum.data[14] += data[9] * rhs.data[2];
    accum.data[15] += data[9] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[4] * rhs.data[4];
    accum.data[5] += data[4] * rhs.data[5];
    accum.data[6] += data[4] * rhs.data[6];
    accum.data[7] += data[4] * rhs.data[7];
    accum.data[8] += data[7] * rhs.data[4];
    accum.data[9] += data[7] * rhs.data[5];
    accum.data[10] += data[7] * rhs.data[6];
    accum.data[11] += data[7] * rhs.data[7];
    accum.data[12] += data[10] * rhs.data[4];
    accum.data[13] += data[10] * rhs.data[5];
    accum.data[14] += data[10] * rhs.data[6];
    accum.data[15] += data[10] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];
    accum.data[4] += data[5] * rhs.data[8];
    accum.data[5] += data[5] * rhs.data[9];
    accum.data[6] += data[5] * rhs.data[10];
    accum.data[7] += data[5] * rhs.data[11];
    accum.data[8] += data[8] * rhs.data[8];
    accum.data[9] += data[8] * rhs.data[9];
    accum.data[10] += data[8] * rhs.data[10];
    accum.data[11] += data[8] * rhs.data[11];
    accum.data[12] += data[11] * rhs.data[8];
    accum.data[13] += data[11] * rhs.data[9];
    accum.data[14] += data[11] * rhs.data[10];
    accum.data[15] += data[11] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 4-by-4-by-3
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> operator*(Matrix<Element, 3, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];
    accum += data[6];
    accum += data[7];
    accum += data[8];
    accum += data[9];
    accum += data[10];
    accum += data[11];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];
    accum += data[6] * data[6];
    accum += data[7] * data[7];
    accum += data[8] * data[8];
    accum += data[9] * data[9];
    accum += data[10] * data[10];
    accum += data[11] * data[11];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[4];
    accum += data[8];

    return accum;
  }
    
};

/// Template alias for 4-by-3 matrix
template <typename Element>
using Matrix4x3 = Matrix<Element, 4, 3>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix4x3<Element> make_Matrix4x3(
    Element _0_0, Element _0_1, Element _0_2, 
    Element _1_0, Element _1_1, Element _1_2, 
    Element _2_0, Element _2_1, Element _2_2, 
    Element _3_0, Element _3_1, Element _3_2
) {
  return Matrix4x3<Element>(
  _0_0, _0_1, _0_2, 
  _1_0, _1_1, _1_2, 
  _2_0, _2_1, _2_2, 
  _3_0, _3_1, _3_2 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// 4-by-4 matrix template class definition
template <typename Element_>
struct Matrix<Element_, 4, 4> {

  //
  // Type definitions
  //

  /// Element data type
  using Element = Element_;

  /// Number of rows in matrix
  static int const kRows = 4;

  /// Number of columns in matrix
  static int const kColumns = 4;

  /// Layout of matrix in underlying array
  using Layout = layout::RowMajor;

  /// Number of elements in matrix
  static int const kCount = 16;

  //
  // Data members
  //

  /// Elements of the matrix in row-major layout
  Array<Element, kCount> data;

  //
  // Methods
  //

  /// Constructs a zero matrix
  CUTLASS_HOST_DEVICE
  Matrix() {
    data.clear();
  }
  
  /// Copy constructor for a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Matrix(Matrix const &rhs) {
    data = rhs.data;
  }
    
  /// Constructs a 4-by-4 matrix from scalar elements
  CUTLASS_HOST_DEVICE
  Matrix(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3, 
    Element _1_0, Element _1_1, Element _1_2, Element _1_3, 
    Element _2_0, Element _2_1, Element _2_2, Element _2_3, 
    Element _3_0, Element _3_1, Element _3_2, Element _3_3
  ) {

    data[0] = _0_0;  data[1] = _0_1;  data[2] = _0_2;  data[3] = _0_3;
    data[4] = _1_0;  data[5] = _1_1;  data[6] = _1_2;  data[7] = _1_3;
    data[8] = _2_0;  data[9] = _2_1;  data[10] = _2_2;  data[11] = _2_3;
    data[12] = _3_0;  data[13] = _3_1;  data[14] = _3_2;  data[15] = _3_3;
  }
    
  /// Constructs a 4-by-4 matrix from row vectors
  CUTLASS_HOST_DEVICE
  Matrix(
    Matrix<Element, 1, 4> const &row_0,
    Matrix<Element, 1, 4> const &row_1,
    Matrix<Element, 1, 4> const &row_2,
    Matrix<Element, 1, 4> const &row_3
  ) { 
    data[0] = row_0.data[0];
    data[1] = row_0.data[1];
    data[2] = row_0.data[2];
    data[3] = row_0.data[3];
    data[4] = row_1.data[0];
    data[5] = row_1.data[1];
    data[6] = row_1.data[2];
    data[7] = row_1.data[3];
    data[8] = row_2.data[0];
    data[9] = row_2.data[1];
    data[10] = row_2.data[2];
    data[11] = row_2.data[3];
    data[12] = row_3.data[0];
    data[13] = row_3.data[1];
    data[14] = row_3.data[2];
    data[15] = row_3.data[3];
  }
    
  /// Static method to construct a 4-by-4 matrix from column vectors
  CUTLASS_HOST_DEVICE
  static Matrix from_columns(
    Matrix<Element, 4, 1> const &column_0,
    Matrix<Element, 4, 1> const &column_1,
    Matrix<Element, 4, 1> const &column_2,
    Matrix<Element, 4, 1> const &column_3
  ) { 
    Matrix result;
    
    result.data[0] = column_0.data[0];
    result.data[1] = column_1.data[0];
    result.data[2] = column_2.data[0];
    result.data[3] = column_3.data[0];
    result.data[4] = column_0.data[1];
    result.data[5] = column_1.data[1];
    result.data[6] = column_2.data[1];
    result.data[7] = column_3.data[1];
    result.data[8] = column_0.data[2];
    result.data[9] = column_1.data[2];
    result.data[10] = column_2.data[2];
    result.data[11] = column_3.data[2];
    result.data[12] = column_0.data[3];
    result.data[13] = column_1.data[3];
    result.data[14] = column_2.data[3];
    result.data[15] = column_3.data[3];
    return result;
  }
    
  /// Constructs an identity matrix
  CUTLASS_HOST_DEVICE
  static Matrix identity() {
    Matrix m;
    
    m.data[0] = Element(1);
    m.data[5] = Element(1);
    m.data[10] = Element(1);
    m.data[15] = Element(1);

    return m;
  }
    
  /// Constructs a matrix from a uniform element
  CUTLASS_HOST_DEVICE
  static Matrix uniform(Element s) {
    Matrix m;
    
    m.data[0] = s;
    m.data[1] = s;
    m.data[2] = s;
    m.data[3] = s;
    m.data[4] = s;
    m.data[5] = s;
    m.data[6] = s;
    m.data[7] = s;
    m.data[8] = s;
    m.data[9] = s;
    m.data[10] = s;
    m.data[11] = s;
    m.data[12] = s;
    m.data[13] = s;
    m.data[14] = s;
    m.data[15] = s;

    return m;
  }

  /// Constructs a matrix from a uniform element 1
  CUTLASS_HOST_DEVICE
  static Matrix ones() {
    return uniform(Element(1));
  }

  /// Constructs a matrix from a uniform element 0
  CUTLASS_HOST_DEVICE
  static Matrix zero() {
    return Matrix();
  }
  
  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 4, 1> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[5] = diag.data[1];
    m.data[10] = diag.data[2];
    m.data[15] = diag.data[3];

    return m;
  }

  /// Constructs a matrix from elements along its diagonal
  CUTLASS_HOST_DEVICE
  static Matrix from_diagonal(Matrix<Element, 1, 4> const &diag) {
    Matrix m;
    
    m.data[0] = diag.data[0];
    m.data[5] = diag.data[1];
    m.data[10] = diag.data[2];
    m.data[15] = diag.data[3];

    return m;
  }

  /// Gets an array of diagonal elements
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> diagonal() const {
    Matrix<Element, 4, 1> diag;
    
    diag.data[0] = data[0];
    diag.data[1] = data[5];
    diag.data[2] = data[10];
    diag.data[3] = data[15];

    return diag;
  }
    
  /// Returns a transposed matrix
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> transpose() const {
    Matrix<Element, 4, 4> mt;
    
    mt.data[0] = data[0];
    mt.data[4] = data[1];
    mt.data[8] = data[2];
    mt.data[12] = data[3];
    mt.data[1] = data[4];
    mt.data[5] = data[5];
    mt.data[9] = data[6];
    mt.data[13] = data[7];
    mt.data[2] = data[8];
    mt.data[6] = data[9];
    mt.data[10] = data[10];
    mt.data[14] = data[11];
    mt.data[3] = data[12];
    mt.data[7] = data[13];
    mt.data[11] = data[14];
    mt.data[15] = data[15];

    return mt;
  }
    
  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(int i, int j) const {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(int i, int j) {
    return data[i * 4 + j];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element at(Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & at(Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element &at(int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element at(int offset) const {
    return data[offset];
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element operator[](Coord<2> const &coord) const {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by coordinate
  CUTLASS_HOST_DEVICE
  Element & operator[](Coord<2> const &coord) {
    return at(coord[0], coord[1]);
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element & operator[](int offset) {
    return data[offset];
  }

  /// Accesses an element by offset
  CUTLASS_HOST_DEVICE
  Element operator[](int offset) const {
    return data[offset];
  }
  
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 2> slice_1x2(int i = 0, int j = 0) const {
    Matrix<Element, 1, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x2(Matrix<Element, 1, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 3> slice_1x3(int i = 0, int j = 0) const {
    Matrix<Element, 1, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x3(Matrix<Element, 1, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> slice_1x4(int i = 0, int j = 0) const {
    Matrix<Element, 1, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_1x4(Matrix<Element, 1, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 1, 4> row(int i) const {
    return slice_1x4(i, 0);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_row(Matrix<Element, 1, 4> const &v, int i = 0) {
    return set_slice_1x4(v, i, 0);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 1> slice_2x1(int i = 0, int j = 0) const {
    Matrix<Element, 2, 1> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 4];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x1(Matrix<Element, 2, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 4] = m.data[1];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 2> slice_2x2(int i = 0, int j = 0) const {
    Matrix<Element, 2, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 4];
    m.data[3] = data[i * 4 + j + 5];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x2(Matrix<Element, 2, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 4] = m.data[2];
    data[i * 4 + j + 5] = m.data[3];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 3> slice_2x3(int i = 0, int j = 0) const {
    Matrix<Element, 2, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 4];
    m.data[4] = data[i * 4 + j + 5];
    m.data[5] = data[i * 4 + j + 6];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x3(Matrix<Element, 2, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 4] = m.data[3];
    data[i * 4 + j + 5] = m.data[4];
    data[i * 4 + j + 6] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 2, 4> slice_2x4(int i = 0, int j = 0) const {
    Matrix<Element, 2, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];
    m.data[4] = data[i * 4 + j + 4];
    m.data[5] = data[i * 4 + j + 5];
    m.data[6] = data[i * 4 + j + 6];
    m.data[7] = data[i * 4 + j + 7];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_2x4(Matrix<Element, 2, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];
    data[i * 4 + j + 4] = m.data[4];
    data[i * 4 + j + 5] = m.data[5];
    data[i * 4 + j + 6] = m.data[6];
    data[i * 4 + j + 7] = m.data[7];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 1> slice_3x1(int i = 0, int j = 0) const {
    Matrix<Element, 3, 1> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 4];
    m.data[2] = data[i * 4 + j + 8];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x1(Matrix<Element, 3, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 4] = m.data[1];
    data[i * 4 + j + 8] = m.data[2];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 2> slice_3x2(int i = 0, int j = 0) const {
    Matrix<Element, 3, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 4];
    m.data[3] = data[i * 4 + j + 5];
    m.data[4] = data[i * 4 + j + 8];
    m.data[5] = data[i * 4 + j + 9];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x2(Matrix<Element, 3, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 4] = m.data[2];
    data[i * 4 + j + 5] = m.data[3];
    data[i * 4 + j + 8] = m.data[4];
    data[i * 4 + j + 9] = m.data[5];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 3> slice_3x3(int i = 0, int j = 0) const {
    Matrix<Element, 3, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 4];
    m.data[4] = data[i * 4 + j + 5];
    m.data[5] = data[i * 4 + j + 6];
    m.data[6] = data[i * 4 + j + 8];
    m.data[7] = data[i * 4 + j + 9];
    m.data[8] = data[i * 4 + j + 10];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x3(Matrix<Element, 3, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 4] = m.data[3];
    data[i * 4 + j + 5] = m.data[4];
    data[i * 4 + j + 6] = m.data[5];
    data[i * 4 + j + 8] = m.data[6];
    data[i * 4 + j + 9] = m.data[7];
    data[i * 4 + j + 10] = m.data[8];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 3, 4> slice_3x4(int i = 0, int j = 0) const {
    Matrix<Element, 3, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];
    m.data[4] = data[i * 4 + j + 4];
    m.data[5] = data[i * 4 + j + 5];
    m.data[6] = data[i * 4 + j + 6];
    m.data[7] = data[i * 4 + j + 7];
    m.data[8] = data[i * 4 + j + 8];
    m.data[9] = data[i * 4 + j + 9];
    m.data[10] = data[i * 4 + j + 10];
    m.data[11] = data[i * 4 + j + 11];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_3x4(Matrix<Element, 3, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];
    data[i * 4 + j + 4] = m.data[4];
    data[i * 4 + j + 5] = m.data[5];
    data[i * 4 + j + 6] = m.data[6];
    data[i * 4 + j + 7] = m.data[7];
    data[i * 4 + j + 8] = m.data[8];
    data[i * 4 + j + 9] = m.data[9];
    data[i * 4 + j + 10] = m.data[10];
    data[i * 4 + j + 11] = m.data[11];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> slice_4x1(int i = 0, int j = 0) const {
    Matrix<Element, 4, 1> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 4];
    m.data[2] = data[i * 4 + j + 8];
    m.data[3] = data[i * 4 + j + 12];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x1(Matrix<Element, 4, 1> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 4] = m.data[1];
    data[i * 4 + j + 8] = m.data[2];
    data[i * 4 + j + 12] = m.data[3];

    return *this;
  }
    
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> column(int j) const {
    return slice_4x1(0, j);
  }

  CUTLASS_HOST_DEVICE
  Matrix &set_column(Matrix<Element, 4, 1> const &v, int j =0) {
    return set_slice_4x1(v, 0, j);
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> slice_4x2(int i = 0, int j = 0) const {
    Matrix<Element, 4, 2> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 4];
    m.data[3] = data[i * 4 + j + 5];
    m.data[4] = data[i * 4 + j + 8];
    m.data[5] = data[i * 4 + j + 9];
    m.data[6] = data[i * 4 + j + 12];
    m.data[7] = data[i * 4 + j + 13];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x2(Matrix<Element, 4, 2> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 4] = m.data[2];
    data[i * 4 + j + 5] = m.data[3];
    data[i * 4 + j + 8] = m.data[4];
    data[i * 4 + j + 9] = m.data[5];
    data[i * 4 + j + 12] = m.data[6];
    data[i * 4 + j + 13] = m.data[7];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> slice_4x3(int i = 0, int j = 0) const {
    Matrix<Element, 4, 3> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 4];
    m.data[4] = data[i * 4 + j + 5];
    m.data[5] = data[i * 4 + j + 6];
    m.data[6] = data[i * 4 + j + 8];
    m.data[7] = data[i * 4 + j + 9];
    m.data[8] = data[i * 4 + j + 10];
    m.data[9] = data[i * 4 + j + 12];
    m.data[10] = data[i * 4 + j + 13];
    m.data[11] = data[i * 4 + j + 14];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x3(Matrix<Element, 4, 3> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 4] = m.data[3];
    data[i * 4 + j + 5] = m.data[4];
    data[i * 4 + j + 6] = m.data[5];
    data[i * 4 + j + 8] = m.data[6];
    data[i * 4 + j + 9] = m.data[7];
    data[i * 4 + j + 10] = m.data[8];
    data[i * 4 + j + 12] = m.data[9];
    data[i * 4 + j + 13] = m.data[10];
    data[i * 4 + j + 14] = m.data[11];

    return *this;
  }
    
  /// Gets a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> slice_4x4(int i = 0, int j = 0) const {
    Matrix<Element, 4, 4> m;
    
    m.data[0] = data[i * 4 + j + 0];
    m.data[1] = data[i * 4 + j + 1];
    m.data[2] = data[i * 4 + j + 2];
    m.data[3] = data[i * 4 + j + 3];
    m.data[4] = data[i * 4 + j + 4];
    m.data[5] = data[i * 4 + j + 5];
    m.data[6] = data[i * 4 + j + 6];
    m.data[7] = data[i * 4 + j + 7];
    m.data[8] = data[i * 4 + j + 8];
    m.data[9] = data[i * 4 + j + 9];
    m.data[10] = data[i * 4 + j + 10];
    m.data[11] = data[i * 4 + j + 11];
    m.data[12] = data[i * 4 + j + 12];
    m.data[13] = data[i * 4 + j + 13];
    m.data[14] = data[i * 4 + j + 14];
    m.data[15] = data[i * 4 + j + 15];

    return m;
  }

  /// Overwrites a submatrix with optional offset
  CUTLASS_HOST_DEVICE
  Matrix & set_slice_4x4(Matrix<Element, 4, 4> const &m, int i = 0, int j = 0) {
    
    data[i * 4 + j + 0] = m.data[0];
    data[i * 4 + j + 1] = m.data[1];
    data[i * 4 + j + 2] = m.data[2];
    data[i * 4 + j + 3] = m.data[3];
    data[i * 4 + j + 4] = m.data[4];
    data[i * 4 + j + 5] = m.data[5];
    data[i * 4 + j + 6] = m.data[6];
    data[i * 4 + j + 7] = m.data[7];
    data[i * 4 + j + 8] = m.data[8];
    data[i * 4 + j + 9] = m.data[9];
    data[i * 4 + j + 10] = m.data[10];
    data[i * 4 + j + 11] = m.data[11];
    data[i * 4 + j + 12] = m.data[12];
    data[i * 4 + j + 13] = m.data[13];
    data[i * 4 + j + 14] = m.data[14];
    data[i * 4 + j + 15] = m.data[15];

    return *this;
  }
    
  /// Forms a 4-by-4 matrix by horizontally concatenating a 4-by-1 matrix with a 4-by-3 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 4, 1> const & lhs, Matrix<Element, 4, 3> const & rhs) {
    return Matrix(
      lhs.at(0, 0), rhs.at(0, 0), rhs.at(0, 1), rhs.at(0, 2)
      , lhs.at(1, 0), rhs.at(1, 0), rhs.at(1, 1), rhs.at(1, 2)
      , lhs.at(2, 0), rhs.at(2, 0), rhs.at(2, 1), rhs.at(2, 2)
      , lhs.at(3, 0), rhs.at(3, 0), rhs.at(3, 1), rhs.at(3, 2));
  }
  
  /// Forms a 4-by-4 matrix by horizontally concatenating a 4-by-2 matrix with a 4-by-2 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 4, 2> const & lhs, Matrix<Element, 4, 2> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), rhs.at(0, 0), rhs.at(0, 1)
      , lhs.at(1, 0), lhs.at(1, 1), rhs.at(1, 0), rhs.at(1, 1)
      , lhs.at(2, 0), lhs.at(2, 1), rhs.at(2, 0), rhs.at(2, 1)
      , lhs.at(3, 0), lhs.at(3, 1), rhs.at(3, 0), rhs.at(3, 1));
  }
  
  /// Forms a 4-by-4 matrix by horizontally concatenating a 4-by-3 matrix with a 4-by-1 matrix
  CUTLASS_HOST_DEVICE
  static Matrix hcat(Matrix<Element, 4, 3> const & lhs, Matrix<Element, 4, 1> const & rhs) {
    return Matrix(
      lhs.at(0, 0), lhs.at(0, 1), lhs.at(0, 2), rhs.at(0, 0)
      , lhs.at(1, 0), lhs.at(1, 1), lhs.at(1, 2), rhs.at(1, 0)
      , lhs.at(2, 0), lhs.at(2, 1), lhs.at(2, 2), rhs.at(2, 0)
      , lhs.at(3, 0), lhs.at(3, 1), lhs.at(3, 2), rhs.at(3, 0));
  }
  
  /// Forms a 4-by-4 matrix by vertically concatenating a 1-by-4 matrix with a 3-by-4 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 1, 4> const & upper, Matrix<Element, 3, 4> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2), upper.at(0, 3)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2), lower.at(0, 3)
      , lower.at(1, 0), lower.at(1, 1), lower.at(1, 2), lower.at(1, 3)
      , lower.at(2, 0), lower.at(2, 1), lower.at(2, 2), lower.at(2, 3));
  }
  
  /// Forms a 4-by-4 matrix by vertically concatenating a 2-by-4 matrix with a 2-by-4 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 2, 4> const & upper, Matrix<Element, 2, 4> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2), upper.at(0, 3)
      , upper.at(1, 0), upper.at(1, 1), upper.at(1, 2), upper.at(1, 3)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2), lower.at(0, 3)
      , lower.at(1, 0), lower.at(1, 1), lower.at(1, 2), lower.at(1, 3));
  }
  
  /// Forms a 4-by-4 matrix by vertically concatenating a 3-by-4 matrix with a 1-by-4 matrix
  CUTLASS_HOST_DEVICE
  static Matrix vcat(Matrix<Element, 3, 4> const & upper, Matrix<Element, 1, 4> const & lower) {
    return Matrix(
      upper.at(0, 0), upper.at(0, 1), upper.at(0, 2), upper.at(0, 3)
      , upper.at(1, 0), upper.at(1, 1), upper.at(1, 2), upper.at(1, 3)
      , upper.at(2, 0), upper.at(2, 1), upper.at(2, 2), upper.at(2, 3)
      , lower.at(0, 0), lower.at(0, 1), lower.at(0, 2), lower.at(0, 3));
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Element                         A, Matrix<Element, 1, 3> const & B,
    Matrix<Element, 3, 1> const & C, Matrix<Element, 3, 3> const & D) {
    return Matrix(
      A, B.at(0, 0), B.at(0, 1), B.at(0, 2)
      , C.at(0, 0), D.at(0, 0), D.at(0, 1), D.at(0, 2)
      , C.at(1, 0), D.at(1, 0), D.at(1, 1), D.at(1, 2)
      , C.at(2, 0), D.at(2, 0), D.at(2, 1), D.at(2, 2)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 2> const & A, Matrix<Element, 1, 2> const & B,
    Matrix<Element, 3, 2> const & C, Matrix<Element, 3, 2> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0), B.at(0, 1)
      , C.at(0, 0), C.at(0, 1), D.at(0, 0), D.at(0, 1)
      , C.at(1, 0), C.at(1, 1), D.at(1, 0), D.at(1, 1)
      , C.at(2, 0), C.at(2, 1), D.at(2, 0), D.at(2, 1)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 1, 3> const & A, Element                         B,
    Matrix<Element, 3, 3> const & C, Matrix<Element, 3, 1> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), A.at(0, 2), B
      , C.at(0, 0), C.at(0, 1), C.at(0, 2), D.at(0, 0)
      , C.at(1, 0), C.at(1, 1), C.at(1, 2), D.at(1, 0)
      , C.at(2, 0), C.at(2, 1), C.at(2, 2), D.at(2, 0)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 1> const & A, Matrix<Element, 2, 3> const & B,
    Matrix<Element, 2, 1> const & C, Matrix<Element, 2, 3> const & D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0), B.at(0, 1), B.at(0, 2)
      , A.at(1, 0), B.at(1, 0), B.at(1, 1), B.at(1, 2)
      , C.at(0, 0), D.at(0, 0), D.at(0, 1), D.at(0, 2)
      , C.at(1, 0), D.at(1, 0), D.at(1, 1), D.at(1, 2)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 2> const & A, Matrix<Element, 2, 2> const & B,
    Matrix<Element, 2, 2> const & C, Matrix<Element, 2, 2> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0), B.at(0, 1)
      , A.at(1, 0), A.at(1, 1), B.at(1, 0), B.at(1, 1)
      , C.at(0, 0), C.at(0, 1), D.at(0, 0), D.at(0, 1)
      , C.at(1, 0), C.at(1, 1), D.at(1, 0), D.at(1, 1)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 2, 3> const & A, Matrix<Element, 2, 1> const & B,
    Matrix<Element, 2, 3> const & C, Matrix<Element, 2, 1> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), A.at(0, 2), B.at(0, 0)
      , A.at(1, 0), A.at(1, 1), A.at(1, 2), B.at(1, 0)
      , C.at(0, 0), C.at(0, 1), C.at(0, 2), D.at(0, 0)
      , C.at(1, 0), C.at(1, 1), C.at(1, 2), D.at(1, 0)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 3, 1> const & A, Matrix<Element, 3, 3> const & B,
    Element                         C, Matrix<Element, 1, 3> const & D) {
    return Matrix(
      A.at(0, 0), B.at(0, 0), B.at(0, 1), B.at(0, 2)
      , A.at(1, 0), B.at(1, 0), B.at(1, 1), B.at(1, 2)
      , A.at(2, 0), B.at(2, 0), B.at(2, 1), B.at(2, 2)
      , C, D.at(0, 0), D.at(0, 1), D.at(0, 2)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 3, 2> const & A, Matrix<Element, 3, 2> const & B,
    Matrix<Element, 1, 2> const & C, Matrix<Element, 1, 2> const & D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), B.at(0, 0), B.at(0, 1)
      , A.at(1, 0), A.at(1, 1), B.at(1, 0), B.at(1, 1)
      , A.at(2, 0), A.at(2, 1), B.at(2, 0), B.at(2, 1)
      , C.at(0, 0), C.at(0, 1), D.at(0, 0), D.at(0, 1)
    );
  }
  
  /// Forms a 4-by-4 matrix by concatenating four components
  CUTLASS_HOST_DEVICE
  static Matrix block(
    Matrix<Element, 3, 3> const & A, Matrix<Element, 3, 1> const & B,
    Matrix<Element, 1, 3> const & C, Element                         D) {
    return Matrix(
      A.at(0, 0), A.at(0, 1), A.at(0, 2), B.at(0, 0)
      , A.at(1, 0), A.at(1, 1), A.at(1, 2), B.at(1, 0)
      , A.at(2, 0), A.at(2, 1), A.at(2, 2), B.at(2, 0)
      , C.at(0, 0), C.at(0, 1), C.at(0, 2), D
    );
  }
  
  /// Elementwise add operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix add(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] + rhs.data[0];
    result.data[1] = data[1] + rhs.data[1];
    result.data[2] = data[2] + rhs.data[2];
    result.data[3] = data[3] + rhs.data[3];

    result.data[4] = data[4] + rhs.data[4];
    result.data[5] = data[5] + rhs.data[5];
    result.data[6] = data[6] + rhs.data[6];
    result.data[7] = data[7] + rhs.data[7];

    result.data[8] = data[8] + rhs.data[8];
    result.data[9] = data[9] + rhs.data[9];
    result.data[10] = data[10] + rhs.data[10];
    result.data[11] = data[11] + rhs.data[11];

    result.data[12] = data[12] + rhs.data[12];
    result.data[13] = data[13] + rhs.data[13];
    result.data[14] = data[14] + rhs.data[14];
    result.data[15] = data[15] + rhs.data[15];

    return result;
  }
      
  /// Elementwise add operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator +(Matrix const &rhs) const {
    return add(rhs);
  }

  /// Elementwise add operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator +=(Matrix const &rhs) {
    
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];
    data[3] += rhs.data[3];

    data[4] += rhs.data[4];
    data[5] += rhs.data[5];
    data[6] += rhs.data[6];
    data[7] += rhs.data[7];

    data[8] += rhs.data[8];
    data[9] += rhs.data[9];
    data[10] += rhs.data[10];
    data[11] += rhs.data[11];

    data[12] += rhs.data[12];
    data[13] += rhs.data[13];
    data[14] += rhs.data[14];
    data[15] += rhs.data[15];

    return *this;
  }
        
  /// Elementwise subtract operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix subtract(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] - rhs.data[0];
    result.data[1] = data[1] - rhs.data[1];
    result.data[2] = data[2] - rhs.data[2];
    result.data[3] = data[3] - rhs.data[3];

    result.data[4] = data[4] - rhs.data[4];
    result.data[5] = data[5] - rhs.data[5];
    result.data[6] = data[6] - rhs.data[6];
    result.data[7] = data[7] - rhs.data[7];

    result.data[8] = data[8] - rhs.data[8];
    result.data[9] = data[9] - rhs.data[9];
    result.data[10] = data[10] - rhs.data[10];
    result.data[11] = data[11] - rhs.data[11];

    result.data[12] = data[12] - rhs.data[12];
    result.data[13] = data[13] - rhs.data[13];
    result.data[14] = data[14] - rhs.data[14];
    result.data[15] = data[15] - rhs.data[15];

    return result;
  }
      
  /// Elementwise subtract operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator -(Matrix const &rhs) const {
    return subtract(rhs);
  }

  /// Elementwise subtract operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator -=(Matrix const &rhs) {
    
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];
    data[3] -= rhs.data[3];

    data[4] -= rhs.data[4];
    data[5] -= rhs.data[5];
    data[6] -= rhs.data[6];
    data[7] -= rhs.data[7];

    data[8] -= rhs.data[8];
    data[9] -= rhs.data[9];
    data[10] -= rhs.data[10];
    data[11] -= rhs.data[11];

    data[12] -= rhs.data[12];
    data[13] -= rhs.data[13];
    data[14] -= rhs.data[14];
    data[15] -= rhs.data[15];

    return *this;
  }
        
  /// Elementwise multiply operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] * rhs.data[0];
    result.data[1] = data[1] * rhs.data[1];
    result.data[2] = data[2] * rhs.data[2];
    result.data[3] = data[3] * rhs.data[3];

    result.data[4] = data[4] * rhs.data[4];
    result.data[5] = data[5] * rhs.data[5];
    result.data[6] = data[6] * rhs.data[6];
    result.data[7] = data[7] * rhs.data[7];

    result.data[8] = data[8] * rhs.data[8];
    result.data[9] = data[9] * rhs.data[9];
    result.data[10] = data[10] * rhs.data[10];
    result.data[11] = data[11] * rhs.data[11];

    result.data[12] = data[12] * rhs.data[12];
    result.data[13] = data[13] * rhs.data[13];
    result.data[14] = data[14] * rhs.data[14];
    result.data[15] = data[15] * rhs.data[15];

    return result;
  }
      
  /// Scalar multiply operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix multiply(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] * s;
    result.data[1] = data[1] * s;
    result.data[2] = data[2] * s;
    result.data[3] = data[3] * s;

    result.data[4] = data[4] * s;
    result.data[5] = data[5] * s;
    result.data[6] = data[6] * s;
    result.data[7] = data[7] * s;

    result.data[8] = data[8] * s;
    result.data[9] = data[9] * s;
    result.data[10] = data[10] * s;
    result.data[11] = data[11] * s;

    result.data[12] = data[12] * s;
    result.data[13] = data[13] * s;
    result.data[14] = data[14] * s;
    result.data[15] = data[15] * s;

    return result;
  }

  /// Scalar multiply operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator *(Element const &s) const {
    return multiply(s);
  }

  /// Scalar multiply operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator *=(Element const &s) {
    
    data[0] *= s;
    data[1] *= s;
    data[2] *= s;
    data[3] *= s;

    data[4] *= s;
    data[5] *= s;
    data[6] *= s;
    data[7] *= s;

    data[8] *= s;
    data[9] *= s;
    data[10] *= s;
    data[11] *= s;

    data[12] *= s;
    data[13] *= s;
    data[14] *= s;
    data[15] *= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Matrix const &rhs) const {

    Matrix result;
    
    result.data[0] = data[0] / rhs.data[0];
    result.data[1] = data[1] / rhs.data[1];
    result.data[2] = data[2] / rhs.data[2];
    result.data[3] = data[3] / rhs.data[3];

    result.data[4] = data[4] / rhs.data[4];
    result.data[5] = data[5] / rhs.data[5];
    result.data[6] = data[6] / rhs.data[6];
    result.data[7] = data[7] / rhs.data[7];

    result.data[8] = data[8] / rhs.data[8];
    result.data[9] = data[9] / rhs.data[9];
    result.data[10] = data[10] / rhs.data[10];
    result.data[11] = data[11] / rhs.data[11];

    result.data[12] = data[12] / rhs.data[12];
    result.data[13] = data[13] / rhs.data[13];
    result.data[14] = data[14] / rhs.data[14];
    result.data[15] = data[15] / rhs.data[15];

    return result;
  }
      
  /// Scalar divide operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix divide(Element const &s) const {

    Matrix result;
    
    result.data[0] = data[0] / s;
    result.data[1] = data[1] / s;
    result.data[2] = data[2] / s;
    result.data[3] = data[3] / s;

    result.data[4] = data[4] / s;
    result.data[5] = data[5] / s;
    result.data[6] = data[6] / s;
    result.data[7] = data[7] / s;

    result.data[8] = data[8] / s;
    result.data[9] = data[9] / s;
    result.data[10] = data[10] / s;
    result.data[11] = data[11] / s;

    result.data[12] = data[12] / s;
    result.data[13] = data[13] / s;
    result.data[14] = data[14] / s;
    result.data[15] = data[15] / s;

    return result;
  }

  /// Scalar divide operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Element const &s) const {
    return divide(s);
  }

  /// Scalar divide operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Element const &s) {
    
    data[0] /= s;
    data[1] /= s;
    data[2] /= s;
    data[3] /= s;

    data[4] /= s;
    data[5] /= s;
    data[6] /= s;
    data[7] /= s;

    data[8] /= s;
    data[9] /= s;
    data[10] /= s;
    data[11] /= s;

    data[12] /= s;
    data[13] /= s;
    data[14] /= s;
    data[15] /= s;

    return *this;
  }
        
  /// Elementwise divide operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix operator /(Matrix const &rhs) const {
    return divide(rhs);
  }

  /// Elementwise divide operator (4-by-4)
  CUTLASS_HOST_DEVICE
  Matrix & operator /=(Matrix const &rhs) {
    
    data[0] /= rhs.data[0];
    data[1] /= rhs.data[1];
    data[2] /= rhs.data[2];
    data[3] /= rhs.data[3];

    data[4] /= rhs.data[4];
    data[5] /= rhs.data[5];
    data[6] /= rhs.data[6];
    data[7] /= rhs.data[7];

    data[8] /= rhs.data[8];
    data[9] /= rhs.data[9];
    data[10] /= rhs.data[10];
    data[11] /= rhs.data[11];

    data[12] /= rhs.data[12];
    data[13] /= rhs.data[13];
    data[14] /= rhs.data[14];
    data[15] /= rhs.data[15];

    return *this;
  }
        
  /// Negates each element of the matrix
  CUTLASS_HOST_DEVICE
  Matrix operator-() const {
    Matrix m;
    
    m.data[0] = -data[0];
    m.data[1] = -data[1];
    m.data[2] = -data[2];
    m.data[3] = -data[3];
    m.data[4] = -data[4];
    m.data[5] = -data[5];
    m.data[6] = -data[6];
    m.data[7] = -data[7];
    m.data[8] = -data[8];
    m.data[9] = -data[9];
    m.data[10] = -data[10];
    m.data[11] = -data[11];
    m.data[12] = -data[12];
    m.data[13] = -data[13];
    m.data[14] = -data[14];
    m.data[15] = -data[15];

    return m;
  }
  
  /// Matrix product of size 4-by-1-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> product(
    Matrix<Element, 4, 1> const &rhs,
    Matrix<Element, 4, 1> accum = Matrix<Element, 4, 1>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[4] * rhs.data[0];
    accum.data[2] += data[8] * rhs.data[0];
    accum.data[3] += data[12] * rhs.data[0];

    // k=1
    accum.data[0] += data[1] * rhs.data[1];
    accum.data[1] += data[5] * rhs.data[1];
    accum.data[2] += data[9] * rhs.data[1];
    accum.data[3] += data[13] * rhs.data[1];

    // k=2
    accum.data[0] += data[2] * rhs.data[2];
    accum.data[1] += data[6] * rhs.data[2];
    accum.data[2] += data[10] * rhs.data[2];
    accum.data[3] += data[14] * rhs.data[2];

    // k=3
    accum.data[0] += data[3] * rhs.data[3];
    accum.data[1] += data[7] * rhs.data[3];
    accum.data[2] += data[11] * rhs.data[3];
    accum.data[3] += data[15] * rhs.data[3];

    return accum;
  }

  /// Matrix product of size 4-by-1-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 1> operator*(Matrix<Element, 4, 1> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> product(
    Matrix<Element, 4, 2> const &rhs,
    Matrix<Element, 4, 2> accum = Matrix<Element, 4, 2>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[4] * rhs.data[0];
    accum.data[3] += data[4] * rhs.data[1];
    accum.data[4] += data[8] * rhs.data[0];
    accum.data[5] += data[8] * rhs.data[1];
    accum.data[6] += data[12] * rhs.data[0];
    accum.data[7] += data[12] * rhs.data[1];

    // k=1
    accum.data[0] += data[1] * rhs.data[2];
    accum.data[1] += data[1] * rhs.data[3];
    accum.data[2] += data[5] * rhs.data[2];
    accum.data[3] += data[5] * rhs.data[3];
    accum.data[4] += data[9] * rhs.data[2];
    accum.data[5] += data[9] * rhs.data[3];
    accum.data[6] += data[13] * rhs.data[2];
    accum.data[7] += data[13] * rhs.data[3];

    // k=2
    accum.data[0] += data[2] * rhs.data[4];
    accum.data[1] += data[2] * rhs.data[5];
    accum.data[2] += data[6] * rhs.data[4];
    accum.data[3] += data[6] * rhs.data[5];
    accum.data[4] += data[10] * rhs.data[4];
    accum.data[5] += data[10] * rhs.data[5];
    accum.data[6] += data[14] * rhs.data[4];
    accum.data[7] += data[14] * rhs.data[5];

    // k=3
    accum.data[0] += data[3] * rhs.data[6];
    accum.data[1] += data[3] * rhs.data[7];
    accum.data[2] += data[7] * rhs.data[6];
    accum.data[3] += data[7] * rhs.data[7];
    accum.data[4] += data[11] * rhs.data[6];
    accum.data[5] += data[11] * rhs.data[7];
    accum.data[6] += data[15] * rhs.data[6];
    accum.data[7] += data[15] * rhs.data[7];

    return accum;
  }

  /// Matrix product of size 4-by-2-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 2> operator*(Matrix<Element, 4, 2> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> product(
    Matrix<Element, 4, 3> const &rhs,
    Matrix<Element, 4, 3> accum = Matrix<Element, 4, 3>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[4] * rhs.data[0];
    accum.data[4] += data[4] * rhs.data[1];
    accum.data[5] += data[4] * rhs.data[2];
    accum.data[6] += data[8] * rhs.data[0];
    accum.data[7] += data[8] * rhs.data[1];
    accum.data[8] += data[8] * rhs.data[2];
    accum.data[9] += data[12] * rhs.data[0];
    accum.data[10] += data[12] * rhs.data[1];
    accum.data[11] += data[12] * rhs.data[2];

    // k=1
    accum.data[0] += data[1] * rhs.data[3];
    accum.data[1] += data[1] * rhs.data[4];
    accum.data[2] += data[1] * rhs.data[5];
    accum.data[3] += data[5] * rhs.data[3];
    accum.data[4] += data[5] * rhs.data[4];
    accum.data[5] += data[5] * rhs.data[5];
    accum.data[6] += data[9] * rhs.data[3];
    accum.data[7] += data[9] * rhs.data[4];
    accum.data[8] += data[9] * rhs.data[5];
    accum.data[9] += data[13] * rhs.data[3];
    accum.data[10] += data[13] * rhs.data[4];
    accum.data[11] += data[13] * rhs.data[5];

    // k=2
    accum.data[0] += data[2] * rhs.data[6];
    accum.data[1] += data[2] * rhs.data[7];
    accum.data[2] += data[2] * rhs.data[8];
    accum.data[3] += data[6] * rhs.data[6];
    accum.data[4] += data[6] * rhs.data[7];
    accum.data[5] += data[6] * rhs.data[8];
    accum.data[6] += data[10] * rhs.data[6];
    accum.data[7] += data[10] * rhs.data[7];
    accum.data[8] += data[10] * rhs.data[8];
    accum.data[9] += data[14] * rhs.data[6];
    accum.data[10] += data[14] * rhs.data[7];
    accum.data[11] += data[14] * rhs.data[8];

    // k=3
    accum.data[0] += data[3] * rhs.data[9];
    accum.data[1] += data[3] * rhs.data[10];
    accum.data[2] += data[3] * rhs.data[11];
    accum.data[3] += data[7] * rhs.data[9];
    accum.data[4] += data[7] * rhs.data[10];
    accum.data[5] += data[7] * rhs.data[11];
    accum.data[6] += data[11] * rhs.data[9];
    accum.data[7] += data[11] * rhs.data[10];
    accum.data[8] += data[11] * rhs.data[11];
    accum.data[9] += data[15] * rhs.data[9];
    accum.data[10] += data[15] * rhs.data[10];
    accum.data[11] += data[15] * rhs.data[11];

    return accum;
  }

  /// Matrix product of size 4-by-3-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 3> operator*(Matrix<Element, 4, 3> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> product(
    Matrix<Element, 4, 4> const &rhs,
    Matrix<Element, 4, 4> accum = Matrix<Element, 4, 4>()
  ) const {
    
    // k=0
    accum.data[0] += data[0] * rhs.data[0];
    accum.data[1] += data[0] * rhs.data[1];
    accum.data[2] += data[0] * rhs.data[2];
    accum.data[3] += data[0] * rhs.data[3];
    accum.data[4] += data[4] * rhs.data[0];
    accum.data[5] += data[4] * rhs.data[1];
    accum.data[6] += data[4] * rhs.data[2];
    accum.data[7] += data[4] * rhs.data[3];
    accum.data[8] += data[8] * rhs.data[0];
    accum.data[9] += data[8] * rhs.data[1];
    accum.data[10] += data[8] * rhs.data[2];
    accum.data[11] += data[8] * rhs.data[3];
    accum.data[12] += data[12] * rhs.data[0];
    accum.data[13] += data[12] * rhs.data[1];
    accum.data[14] += data[12] * rhs.data[2];
    accum.data[15] += data[12] * rhs.data[3];

    // k=1
    accum.data[0] += data[1] * rhs.data[4];
    accum.data[1] += data[1] * rhs.data[5];
    accum.data[2] += data[1] * rhs.data[6];
    accum.data[3] += data[1] * rhs.data[7];
    accum.data[4] += data[5] * rhs.data[4];
    accum.data[5] += data[5] * rhs.data[5];
    accum.data[6] += data[5] * rhs.data[6];
    accum.data[7] += data[5] * rhs.data[7];
    accum.data[8] += data[9] * rhs.data[4];
    accum.data[9] += data[9] * rhs.data[5];
    accum.data[10] += data[9] * rhs.data[6];
    accum.data[11] += data[9] * rhs.data[7];
    accum.data[12] += data[13] * rhs.data[4];
    accum.data[13] += data[13] * rhs.data[5];
    accum.data[14] += data[13] * rhs.data[6];
    accum.data[15] += data[13] * rhs.data[7];

    // k=2
    accum.data[0] += data[2] * rhs.data[8];
    accum.data[1] += data[2] * rhs.data[9];
    accum.data[2] += data[2] * rhs.data[10];
    accum.data[3] += data[2] * rhs.data[11];
    accum.data[4] += data[6] * rhs.data[8];
    accum.data[5] += data[6] * rhs.data[9];
    accum.data[6] += data[6] * rhs.data[10];
    accum.data[7] += data[6] * rhs.data[11];
    accum.data[8] += data[10] * rhs.data[8];
    accum.data[9] += data[10] * rhs.data[9];
    accum.data[10] += data[10] * rhs.data[10];
    accum.data[11] += data[10] * rhs.data[11];
    accum.data[12] += data[14] * rhs.data[8];
    accum.data[13] += data[14] * rhs.data[9];
    accum.data[14] += data[14] * rhs.data[10];
    accum.data[15] += data[14] * rhs.data[11];

    // k=3
    accum.data[0] += data[3] * rhs.data[12];
    accum.data[1] += data[3] * rhs.data[13];
    accum.data[2] += data[3] * rhs.data[14];
    accum.data[3] += data[3] * rhs.data[15];
    accum.data[4] += data[7] * rhs.data[12];
    accum.data[5] += data[7] * rhs.data[13];
    accum.data[6] += data[7] * rhs.data[14];
    accum.data[7] += data[7] * rhs.data[15];
    accum.data[8] += data[11] * rhs.data[12];
    accum.data[9] += data[11] * rhs.data[13];
    accum.data[10] += data[11] * rhs.data[14];
    accum.data[11] += data[11] * rhs.data[15];
    accum.data[12] += data[15] * rhs.data[12];
    accum.data[13] += data[15] * rhs.data[13];
    accum.data[14] += data[15] * rhs.data[14];
    accum.data[15] += data[15] * rhs.data[15];

    return accum;
  }

  /// Matrix product of size 4-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix<Element, 4, 4> operator*(Matrix<Element, 4, 4> const &rhs) const {
    return product(rhs);
  }
  
  /// Matrix product of size 4-by-4-by-4
  CUTLASS_HOST_DEVICE
  Matrix & operator*=(Matrix<Element, 4, 4> const &rhs) {
    *this = product(rhs);
    return *this;
  }
    
  /// Returns the sum of elements
  CUTLASS_HOST_DEVICE
  Element sum(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[1];
    accum += data[2];
    accum += data[3];
    accum += data[4];
    accum += data[5];
    accum += data[6];
    accum += data[7];
    accum += data[8];
    accum += data[9];
    accum += data[10];
    accum += data[11];
    accum += data[12];
    accum += data[13];
    accum += data[14];
    accum += data[15];

    return accum;
  }  

  /// Returns the sum of squared elements
  CUTLASS_HOST_DEVICE
  Element norm(Element accum = Element()) const {
    
    accum += data[0] * data[0];
    accum += data[1] * data[1];
    accum += data[2] * data[2];
    accum += data[3] * data[3];
    accum += data[4] * data[4];
    accum += data[5] * data[5];
    accum += data[6] * data[6];
    accum += data[7] * data[7];
    accum += data[8] * data[8];
    accum += data[9] * data[9];
    accum += data[10] * data[10];
    accum += data[11] * data[11];
    accum += data[12] * data[12];
    accum += data[13] * data[13];
    accum += data[14] * data[14];
    accum += data[15] * data[15];

    return accum;
  }

  /// Returns square root of the norm
  CUTLASS_HOST_DEVICE
  Element magnitude() const {
    return fast_sqrt(norm());
  }

  /// Returns the sum of diagonal elements
  CUTLASS_HOST_DEVICE
  Element trace(Element accum = Element()) const {
    
    accum += data[0];
    accum += data[5];
    accum += data[10];
    accum += data[15];

    return accum;
  }
    
  /// Returns 4-by-4 rotation matrix around the X axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_X(Element theta) {
    Matrix m = identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(1, 1) = c;
    m.at(1, 2) = -s;
    m.at(2, 1) = s;
    m.at(2, 2) = c;

    return m;
  }

  /// Returns 4-by-4 rotation matrix around the Y axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_Y(Element theta) {
    Matrix m = identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(0, 0) = c;
    m.at(2, 0) = -s;
    m.at(0, 2) = s;
    m.at(2, 2) = c;

    return m;
  }

  /// Returns 4-by-4 rotation matrix around the Z axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation_Z(Element theta) {
    Matrix m = Matrix::identity();

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    m.at(0, 0) = c;
    m.at(0, 1) = -s;
    m.at(1, 0) = s;
    m.at(1, 1) = c;

    return m;
  }

  /// Returns a 4-by-4 rotation matrix around a unit-length axis
  CUTLASS_HOST_DEVICE
  static Matrix rotation(Element theta, Matrix<Element, 3, 1> const &u) {
    Element x = u.data[0];
    Element y = u.data[1];
    Element z = u.data[2];

    Element c = fast_cos(theta);
    Element s = fast_sin(theta);

    Element one_minus_cos = Element(1) - fast_cos(theta);

    Matrix m;

    m.set_slice_3x3({
      c + x * x * one_minus_cos, x * y * one_minus_cos - z * s, x * z * one_minus_cos + y * s,
      y * x * one_minus_cos * z * s, c + y * y * one_minus_cos, y * z * one_minus_cos - x * s,
      z * x * one_minus_cos - y * s, z * y * one_minus_cos + x * s, c + z * z * one_minus_cos
    });

    return m;
  }

  /// Returns a 4-by-4 reflection about the plane specified by the 
  /// unit-length normal vector n_unit
  CUTLASS_HOST_DEVICE
  static Matrix reflection(Matrix<Element, 3, 1> const &n_unit) {

    Element a = n_unit.data[0];
    Element b = n_unit.data[1];
    Element c = n_unit.data[2];

    Matrix m = Matrix::identity();

    m.set_slice_3x3({
      Element(1) - Element(2) * a * a, Element(-2) * a * b, Element(-2) * a * c,
      Element(-2) * a * b, Element(1) - Element(2) * b * b, Element(-2) * b * c,
      Element(-2) * a * c, Element(-2) * b * c, Element(1) - Element(2) * c * c
    });

    return m;
  }

  /// Returns a perspective projection matrix typical of OpenGL applications
  CUTLASS_HOST_DEVICE
  static Matrix perspective(Element near_plane, Element far_plane, Element fovH, Element fovV) {
    Element aspect = fovH / fovV;
    Element f = Element(cos(fovV)) / Element(fovH);
    Element Q = near_plane - far_plane;

    return Matrix(
      f / aspect, 0,                0,                           0,
      0,          f,                0,                           0,
      0,          0, (near_plane + far_plane) / Q, Element(2) * far_plane * near_plane / Q,
      0,          0,                -1,                          0
    );
  }

  CUTLASS_HOST_DEVICE
  static Matrix translation(Matrix<Element, 3, 1> const &v) {
    return Matrix(
      1, 0, 0, v.data[0],
      0, 1, 0, v.data[1],
      0, 0, 1, v.data[2],
      0, 0, 0, 1
    );
  }
  
  /// Computes the determinant of a 4-by-4 matrix
  CUTLASS_HOST_DEVICE
  Element determinant(Element accum = Element()) const {
    
    accum += at(0, 0) * Matrix<Element, 3, 3>({ at(1, 1), at(1, 2), at(1, 3), at(2, 1), at(2, 2), at(2, 3), at(3, 1), at(3, 2), at(3, 3) }).determinant();
    accum -= at(0, 1) * Matrix<Element, 3, 3>({ at(1, 0), at(1, 2), at(1, 3), at(2, 0), at(2, 2), at(2, 3), at(3, 0), at(3, 2), at(3, 3) }).determinant();
    accum += at(0, 2) * Matrix<Element, 3, 3>({ at(1, 0), at(1, 1), at(1, 3), at(2, 0), at(2, 1), at(2, 3), at(3, 0), at(3, 1), at(3, 3) }).determinant();
    accum -= at(0, 3) * Matrix<Element, 3, 3>({ at(1, 0), at(1, 1), at(1, 2), at(2, 0), at(2, 1), at(2, 2), at(3, 0), at(3, 1), at(3, 2) }).determinant();

    return accum;
  }
  
  /// Computes the inverse of a 4-by-4 matrix (ignores the optional argument)
  CUTLASS_HOST_DEVICE
  Matrix inverse(Element ignore = 1) const {
    Matrix<Element, 2, 2> B = slice_2x2(0, 2);
    Matrix<Element, 2, 2> A = slice_2x2(0, 0);
    Matrix<Element, 2, 2> C = slice_2x2(2, 0);
    Matrix<Element, 2, 2> D = slice_2x2(2, 2);

    Matrix<Element, 2, 2> D_inv = D.inverse();

    Matrix<Element, 2, 2> E = (A - B * D_inv * C).inverse();

    return Matrix::block(
      E,              -E * B * D_inv,
      -D_inv * C * E, D_inv + D_inv * C * E * B * D_inv
    );
  }
    
};

/// Template alias for 4-by-4 matrix
template <typename Element>
using Matrix4x4 = Matrix<Element, 4, 4>;


/// Free function to infer element type from template arguments
template <typename Element>
CUTLASS_HOST_DEVICE Matrix4x4<Element> make_Matrix4x4(
    Element _0_0, Element _0_1, Element _0_2, Element _0_3, 
    Element _1_0, Element _1_1, Element _1_2, Element _1_3, 
    Element _2_0, Element _2_1, Element _2_2, Element _2_3, 
    Element _3_0, Element _3_1, Element _3_2, Element _3_3
) {
  return Matrix4x4<Element>(
  _0_0, _0_1, _0_2, _0_3, 
  _1_0, _1_1, _1_2, _1_3, 
  _2_0, _2_1, _2_2, _2_3, 
  _3_0, _3_1, _3_2, _3_3 
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Elementwise scalar multiplication
template <typename Element, int Rows, int Columns>
CUTLASS_HOST_DEVICE
Matrix<Element, Rows, Columns> operator*(Element s, Matrix<Element, Rows, Columns> const &rhs) {
  return rhs.multiply(s);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
