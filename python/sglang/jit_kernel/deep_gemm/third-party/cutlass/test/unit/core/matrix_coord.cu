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
\brief unit tests for matrix_coord
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/matrix_coord.h"
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace test {
namespace core {

  void test_matrix_coord(cutlass::MatrixCoord::Index row, cutlass::MatrixCoord::Index column) {
    cutlass::MatrixCoord matrix_coord(row, column);

    EXPECT_EQ(matrix_coord.row(), row);
    EXPECT_EQ(matrix_coord.column(), column);
  }

  void test_matrix_coord_operator_addition() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    auto matrix_coord_c = matrix_coord_a + matrix_coord_b;

    EXPECT_EQ(matrix_coord_c.row(), row_a + row_b);
    EXPECT_EQ(matrix_coord_c.column(), column_a + column_b);
  }

  void test_matrix_coord_operator_subtraction() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    auto matrix_coord_c = matrix_coord_a - matrix_coord_b;

    EXPECT_EQ(matrix_coord_c.row(), row_a - row_b);
    EXPECT_EQ(matrix_coord_c.column(), column_a - column_b);
  }

  void test_matrix_coord_operator_multiply() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    auto matrix_coord_c = matrix_coord_a * matrix_coord_b;

    EXPECT_EQ(matrix_coord_c.row(), row_a * row_b);
    EXPECT_EQ(matrix_coord_c.column(), column_a * column_b);
  }

  void test_matrix_coord_operator_division() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    auto matrix_coord_c = matrix_coord_a / matrix_coord_b;

    EXPECT_EQ(matrix_coord_c.row(), row_a / row_b);
    EXPECT_EQ(matrix_coord_c.column(), column_a / column_b);
  }

  void test_matrix_coord_operator_addition_assignment() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    matrix_coord_a += matrix_coord_b;

    EXPECT_EQ(matrix_coord_a.row(), row_a + row_b);
    EXPECT_EQ(matrix_coord_a.column(), column_a + column_b);
  }

  void test_matrix_coord_operator_subtraction_assignment() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    matrix_coord_a -= matrix_coord_b;

    EXPECT_EQ(matrix_coord_a.row(), row_a - row_b);
    EXPECT_EQ(matrix_coord_a.column(), column_a - column_b);
  }

  void test_matrix_coord_operator_multiply_assignment() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    matrix_coord_a *= matrix_coord_b;

    EXPECT_EQ(matrix_coord_a.row(), row_a * row_b);
    EXPECT_EQ(matrix_coord_a.column(), column_a * column_b);
  }

  void test_matrix_coord_operator_division_assignment() {
    cutlass::MatrixCoord::Index row_a = 13;
    cutlass::MatrixCoord::Index column_a = 42;
    cutlass::MatrixCoord::Index row_b = 20;
    cutlass::MatrixCoord::Index column_b = 15;

    cutlass::MatrixCoord matrix_coord_a(row_a, column_a);
    cutlass::MatrixCoord matrix_coord_b(row_b, column_b);

    matrix_coord_a /= matrix_coord_b;

    EXPECT_EQ(matrix_coord_a.row(), row_a / row_b);
    EXPECT_EQ(matrix_coord_a.column(), column_a / column_b);
  }
}
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_row12_column24) {
  cutlass::MatrixCoord::Index row = 12;
  cutlass::MatrixCoord::Index column = 24;
  test::core::test_matrix_coord(row, column);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_addition) {
  test::core::test_matrix_coord_operator_addition();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_subtraction) {
  test::core::test_matrix_coord_operator_subtraction();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_multiply) {
  test::core::test_matrix_coord_operator_multiply();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_division) {
  test::core::test_matrix_coord_operator_division();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_addition_assignment) {
  test::core::test_matrix_coord_operator_addition_assignment();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_subtraction_assignment) {
  test::core::test_matrix_coord_operator_subtraction_assignment();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_multiply_assignment) {
  test::core::test_matrix_coord_operator_multiply_assignment();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Matrix_Coord, basic_operator_division_assignment) {
  test::core::test_matrix_coord_operator_division_assignment();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
