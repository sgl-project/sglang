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
/*!
  \file

  \brief CUTLASS Library is an object-oriented approach to managing operations implemented by CUTLASS.

  Generally,

    description   - compile-time constant parameters used to instantiate an operation

    configuration - runtime parameters with computationally expensive initialization

    arguments     - runtime parameters that may be passed to an initialized operation with low
                    computational overhead
*/

#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ArchTag, typename OperatorClass> struct ArchMap;

template <> struct ArchMap<arch::Sm50, arch::OpClassSimt> {
  static int const kMin = 50;
  static int const kMax = 1024;
};

template <> struct ArchMap<arch::Sm60, arch::OpClassSimt> {
  static int const kMin = 60;
  static int const kMax = 1024;
};

template <> struct ArchMap<arch::Sm61, arch::OpClassSimt> {
  static int const kMin = 61;
  static int const kMax = 1024;
};

template <> struct ArchMap<arch::Sm70, arch::OpClassWmmaTensorOp> {
  static int const kMin = 70;
  static int const kMax = 1024;
};

template <> struct ArchMap<arch::Sm70, arch::OpClassTensorOp> {
  static int const kMin = 70;
  static int const kMax = 75;
};

template <typename OperatorClass> struct ArchMap<arch::Sm75, OperatorClass> {
  static int const kMin = 75;
  static int const kMax = 1024;
};

template <typename OperatorClass> struct ArchMap<arch::Sm80, OperatorClass> {
  static int const kMin = 80;
  static int const kMax = 1024;
};

template <typename OperatorClass> struct ArchMap<arch::Sm86, OperatorClass> {
  static int const kMin = 86;
  static int const kMax = 1024;
};

template <typename OperatorClass> struct ArchMap<arch::Sm89, OperatorClass> {
  static int const kMin = 89;
  static int const kMax = 100;
};

template <typename OperatorClass> struct ArchMap<arch::Sm90, OperatorClass> {
  static int const kMin = 90;
  static int const kMax = 1024;
};

// Arch conditional WGMMA
template <> struct ArchMap<arch::Sm90, arch::OpClassTensorOp> {
  static int const kMin = 90;
  static int const kMax = 90;
};

// Arch conditional sparse WGMMA
template <> struct ArchMap<arch::Sm90, arch::OpClassSparseTensorOp> {
  static int const kMin = 90;
  static int const kMax = 90;
};


template <typename OperatorClass> struct ArchMap<arch::Sm100, OperatorClass> {
  static int const kMin = 100;
  static int const kMax = 1024;
};

template <> struct ArchMap<arch::Sm100, arch::OpClassTensorOp> {
  static int const kMin = 100;
  #if (__CUDACC_VER_MAJOR__ >= 13)
    static int const kMax = 110;
  #else
      static int const kMax = 103;
  #endif // __CUDACC_VER_MAJOR__ >= 13
};

template <typename OperatorClass> struct ArchMap<arch::Sm103, OperatorClass> {
  static int const kMin = 103;
  static int const kMax = 1024;
};
template <> struct ArchMap<arch::Sm103, arch::OpClassTensorOp> {
  static int const kMin = 103;
  static int const kMax = 103;
};

template <typename OperatorClass> struct ArchMap<arch::Sm120, OperatorClass> {
  static int const kMin = 120;
  static int const kMax = 121;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
