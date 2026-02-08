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
    \brief Unit tests for the CUTLASS Quaternion template class.
*/

#include "../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/quaternion.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/constants.h"
  
/////////////////////////////////////////////////////////////////////////////////////////////////

static float const half_pi = cutlass::constants::half_pi<float>();

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion, add_f32) {

  cutlass::Quaternion<float> q0(1, 1, 1, 1);
  cutlass::Quaternion<float> q1(0, 0, 0, 2);

  cutlass::Quaternion<float> q2 = q0 + q1;

  EXPECT_TRUE(
    q2.x() == 1 &&
    q2.y() == 1 &&
    q2.z() == 1 &&
    q2.w() == 3
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion, rotation) {

  cutlass::Matrix3x1<float> x(1.0f, 0.0f, 0.0f);
  cutlass::Quaternion<float> q = cutlass::Quaternion<float>::rotation(0, 0, 1, half_pi) * 2.0f;
  cutlass::Matrix3x1<float> v = q.rotate(x);

  float epsilon = 0.001f;

  EXPECT_TRUE(
    std::abs(v.at(0)) < epsilon &&
    std::abs(v.at(1)) > (1 - epsilon) &&
    std::abs(v.at(2)) < epsilon
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion, rotation_inv) {
  
  cutlass::Matrix3x1<float> x(1.0f, 0.0f, 0.0f);
  cutlass::Quaternion<float> q = cutlass::Quaternion<float>::rotation(0, 0, 1, half_pi) * 2.0f;
  cutlass::Matrix3x1<float> v = q.rotate(x);

  float epsilon = 0.001f;

  EXPECT_TRUE(
    std::abs(v.at(0)) < epsilon &&
    std::abs(-v.at(1)) > (1 - epsilon) &&
    std::abs(v.at(2)) < epsilon
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion, spinor_rotation) {
  
  cutlass::Matrix3x1<float> x(1.0f, 0.0f, 0.0f);
  cutlass::Quaternion<float> q = cutlass::Quaternion<float>::rotation(0, 0, 1, half_pi);
  cutlass::Matrix3x1<float> v = cutlass::spinor_rotation(q, x);

  float epsilon = 0.001f;

  EXPECT_TRUE(
    std::abs(v.at(0)) < epsilon &&
    std::abs(v.at(1)) > (1 - epsilon) &&
    std::abs(v.at(2)) < epsilon
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion, spinor_rotation_inv) {
  
  cutlass::Matrix3x1<float> x(1.0f, 0.0f, 0.0f);
  cutlass::Quaternion<float> q = cutlass::Quaternion<float>::rotation(0, 0, 1, half_pi);
  cutlass::Matrix3x1<float> v = cutlass::spinor_rotation_inv(q, x);

  float epsilon = 0.001f;

  EXPECT_TRUE(
    std::abs(v.at(0)) < epsilon &&
    std::abs(-v.at(1)) > (1 - epsilon) &&
    std::abs(v.at(2)) < epsilon
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion, as_rotation_matrix3x3) {
  
  cutlass::Matrix3x1<float> x(1.0f, 0.0f, 0.0f);
  cutlass::Quaternion<float> q = cutlass::Quaternion<float>::rotation(0, 0, 1, half_pi);
  cutlass::Matrix3x1<float> v = q.as_rotation_matrix_3x3().product(x);

  float epsilon = 0.001f;

  EXPECT_TRUE(
    std::abs(v.at(0)) < epsilon &&
    std::abs(v.at(1)) > (1 - epsilon) &&
    std::abs(v.at(2)) < epsilon
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Quaternion, as_rotation_matrix4x4) {

  cutlass::Matrix4x1<float> x(1.0f, 0.0f, 0.0f, 1.0f);
  cutlass::Quaternion<float> q = cutlass::Quaternion<float>::rotation(0, 0, 1, half_pi);
  cutlass::Matrix4x1<float> v = q.as_rotation_matrix_4x4().product(x);

  float epsilon = 0.001f;

  EXPECT_TRUE(
    std::abs(v.at(0)) < epsilon &&
    std::abs(v.at(1)) > (1 - epsilon) &&
    std::abs(v.at(2)) < epsilon &&
    std::abs(v.at(3)) > (1 - epsilon)
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

