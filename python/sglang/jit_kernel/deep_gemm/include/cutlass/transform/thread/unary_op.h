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

#include "cutlass/cutlass.h"
#include "cutlass/complex.h"

namespace cutlass {
namespace transform {
namespace thread {

namespace UnaryTransform {
    struct Identity;    ///< None (i.e., identity)
    struct Conjugate;   ///< Complex conjugate
}

/// Element-wise unary operator that transforms one element of a fragment at a time
template<
    typename FragmentIn, ///< Input Fragment
    typename FragmentOut,///< Output Fragment
    typename Transform>  ///< Unary transform operator
class UnaryOp
{
    public:
        CUTLASS_DEVICE
        static FragmentOut execute(FragmentIn &in)
        {
            static_assert(FragmentIn::kElements == FragmentOut::kElements, "Number of elements must match.");
            static_assert(platform::is_same<Transform, UnaryTransform::Identity>::value ||
                          platform::is_same<Transform, UnaryTransform::Conjugate>::value,
                          "Unary Operator not supported.");

            FragmentOut out;
            if (platform::is_same<Transform, UnaryTransform::Identity>::value )
            {
                CUTLASS_PRAGMA_UNROLL
                for (int i=0; i < FragmentIn::kElements; ++i){
                   out[i] = static_cast<typename FragmentOut::Element>(in[i]);
                }
            }
            else if (platform::is_same<Transform, UnaryTransform::Conjugate>::value )
            {
                for (int i=0; i < FragmentIn::kElements; ++i){
                   out[i] = conj(static_cast<typename FragmentOut::Element>(in[i]));
                }
            }
            return out;
        }
};

template<typename FragmentIn, typename Transform>
class UnaryOp<FragmentIn, FragmentIn, Transform>
{
    public:
        CUTLASS_DEVICE
        static FragmentIn execute(FragmentIn &in)
        {
            static_assert(platform::is_same<Transform, UnaryTransform::Identity>::value ||
                          platform::is_same<Transform, UnaryTransform::Conjugate>::value,
                          "Unary Operator not supported.");

            if (platform::is_same<Transform, UnaryTransform::Identity>::value )
            {
                return in;
            }
            else if (platform::is_same<Transform, UnaryTransform::Conjugate>::value )
            {
                for(int i=0; i < FragmentIn::kElements; ++i){
                   in[i] = conj(in[i]);
                }
            }
            return in;
        }
      };
    }
  }
}
