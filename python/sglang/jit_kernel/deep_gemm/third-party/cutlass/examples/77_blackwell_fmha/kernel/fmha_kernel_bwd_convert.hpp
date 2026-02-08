/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cute/layout.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

template<class ProblemShape, class Element, class ElementAcc>
struct FmhaKernelBwdConvert {

  struct Arguments {
    ProblemShape problem_shape;

    const ElementAcc* ptr_src_dQ;
    tuple<int, _1, tuple<tuple<int, int>, int>> stride_src_dQ;
    const ElementAcc* ptr_src_dK;
    tuple<int, _1, tuple<tuple<_0, int>, int>> stride_src_dK;
    const ElementAcc* ptr_src_dV;
    tuple<int, _1, tuple<tuple<_0, int>, int>> stride_src_dV;
    
    Element* ptr_dest_dQ;
    tuple<int, _1, tuple<tuple<int, int>, int>> stride_dest_dQ;
    Element* ptr_dest_dK;
    tuple<int, _1, tuple<tuple<_0, int>, int>> stride_dest_dK;
    Element* ptr_dest_dV;
    tuple<int, _1, tuple<tuple<_0, int>, int>> stride_dest_dV;

    ElementAcc scale = 1.0;
  };

  using Params = Arguments;

  using ClusterShape = Shape<_1, _1, _1>;
  static constexpr int SharedStorageSize = 0;

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = 128;
  using ArchTag = cutlass::arch::Sm90;

  static const int kBlockSeq = 8;

  static size_t get_workspace_size(Arguments const& args) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static const int kNumThreadsD = 16;
  static const int kNumThreadsSeq = MaxThreadsPerBlock / kNumThreadsD;
  static const int kElementsPerLoad = 4;

  static const int kIterationsSeq = kBlockSeq / kNumThreadsSeq;

  static bool can_implement(Arguments const& args) {
    return get<2>(args.problem_shape) % kElementsPerLoad == 0 && get<3>(args.problem_shape) % kElementsPerLoad == 0;
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(size<4,0>(params.problem_shape), size<4,1>(params.problem_shape), ceil_div(std::max(size<0>(params.problem_shape), size<1>(params.problem_shape)), kBlockSeq));
    return grid;
  }

  static dim3 get_block_shape() {
    dim3 block(kNumThreadsD, kNumThreadsSeq, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return args;
  }

  template<class StrideSrc, class StrideDest, class Count>
  CUTLASS_DEVICE void copy(Params const& params, const ElementAcc* ptr_src, StrideSrc const& stride_src, Element* ptr_dest, StrideDest const& stride_dest, Count const& count, int d_dim) {
    auto ptr_src_bh = ptr_src + get<2,0,0>(stride_src) * blockIdx.x + get<2,1>(stride_src) * blockIdx.y;
    auto ptr_dest_bh = ptr_dest + get<2,0,0>(stride_dest) * blockIdx.x + get<2,1>(stride_dest) * blockIdx.y;

    int seqlen = count;
    if constexpr (is_variable_length_v<decltype(count)>) {
      int offset = count.cumulative_length[blockIdx.y];
      ptr_dest_bh += offset * get<0>(stride_dest);
      seqlen = count.cumulative_length[blockIdx.y + 1] - offset;
    }

    for (int idx_s_t = threadIdx.y; idx_s_t < kBlockSeq; idx_s_t += kNumThreadsSeq) {
      int idx_s = idx_s_t + kBlockSeq * blockIdx.z;
      if (idx_s >= seqlen) continue;
      auto ptr_src_bhs = ptr_src_bh + idx_s * get<0>(stride_src);
      auto ptr_dest_bhs = ptr_dest_bh + idx_s * get<0>(stride_dest);

      for (int idx_d = threadIdx.x * kElementsPerLoad; idx_d < d_dim; idx_d += kElementsPerLoad * kNumThreadsD) {
        ElementAcc value_src[kElementsPerLoad];
        Element value_dest[kElementsPerLoad];
        
        using VecSrc = uint_bit_t<sizeof_bits_v<ElementAcc> * kElementsPerLoad>;
        using VecDest = uint_bit_t<sizeof_bits_v<Element> * kElementsPerLoad>;
        *reinterpret_cast<VecSrc*>(value_src) = *reinterpret_cast<const VecSrc*>(&ptr_src_bhs[idx_d]);

        for (int v = 0; v < kElementsPerLoad; v++) {
          value_dest[v] = static_cast<Element>(params.scale * value_src[v]);
        }

        *reinterpret_cast<VecDest*>(&ptr_dest_bhs[idx_d]) = *reinterpret_cast<const VecDest*>(value_dest);
      }
    }
  }

  CUTLASS_DEVICE void operator()(const Params &params, char* smem) {
    if (params.ptr_src_dQ != nullptr) {
      copy(params, params.ptr_src_dQ, params.stride_src_dQ, params.ptr_dest_dQ, params.stride_dest_dQ, get<0>(params.problem_shape), get<2>(params.problem_shape));
    }
    if (params.ptr_src_dK != nullptr) {
      copy(params, params.ptr_src_dK, params.stride_src_dK, params.ptr_dest_dK, params.stride_dest_dK, get<1>(params.problem_shape), get<2>(params.problem_shape));
    }
    if (params.ptr_src_dV != nullptr) {
      copy(params, params.ptr_src_dV, params.stride_src_dV, params.ptr_dest_dV, params.stride_dest_dV, get<1>(params.problem_shape), get<3>(params.problem_shape));
    }
  }
};

}  // namespace cutlass::fmha::kernel
