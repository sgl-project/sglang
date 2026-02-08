/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* \file
   \brief Convolution filter format transformation kernel.
*/

#pragma once

#include <algorithm>
#include <random>

#include "cutlass/coord.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cute/int_tuple.hpp"
#include "cute/tensor.hpp"
#include "cute/config.hpp"

namespace cutlass::transform::kernel {

using namespace cute;

enum class FilterFormat {
  CKTRS,
  CTRSK,
  KTRSC
};

template <
  FilterFormat SrcFormat,
  FilterFormat DstFormat,
  int NumDimensions,
  class Element_,
  int AlignmentBytes = 16
>
struct ConvFilterFormatTransformer {
  
  using Element = Element_;
  static_assert(SrcFormat == FilterFormat::CKTRS, "Currently only source format of CKTRS is supported");
  static_assert(DstFormat == FilterFormat::CTRSK || DstFormat == FilterFormat::KTRSC, "Currently only destination format of CTRSK/KTRSC is supported");
  static_assert(AlignmentBytes > 0 && AlignmentBytes % static_cast<int>(sizeof(Element)) == 0, "Invalid alignment setting");

  // In ktrsc order.
  using FilterExtent = array<int, NumDimensions>;

  // Default cta tile shape: 32x32
  static constexpr auto CTATileShape = make_shape(Int<4 * AlignmentBytes / static_cast<int>(sizeof(Element))>{}, Int<32>{});
  // Default thread layout: (4, 32)
  static constexpr auto ThreadLayout = make_layout(make_shape(Int<4>{}, Int<32>{}));

  static constexpr uint32_t MaxThreadsPerBlock = 128;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  using ArchTag = arch::Sm90;

  // Default ctor
  CUTLASS_HOST_DEVICE
  ConvFilterFormatTransformer() {}

  struct Arguments {
    const void *src_ptr;
    void *dst_ptr;
    FilterExtent filter_extent;
  };

  struct Params {
    using TensorSrc = decltype(make_tensor(make_gmem_ptr(recast_ptr<const Element>(nullptr)), make_layout(take<0,NumDimensions>(FilterExtent{}))));
    using TensorDst = decltype(make_tensor(make_gmem_ptr(recast_ptr<Element>(nullptr)), make_layout(make_shape(int32_t(0), int32_t(0)))));

    TensorSrc src;
    TensorDst dst; 
  };

  struct SharedStorage {
    /* empty, no smem needed */
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static Status
  can_implement(Arguments const& args) {
    bool implementable = true;
    // alignment rule
    {
      int contiguous_dim = DstFormat == FilterFormat::CTRSK ? args.filter_extent[0] : args.filter_extent[NumDimensions - 1];
      int align_element = AlignmentBytes / static_cast<int>(sizeof(Element));

      implementable &= (contiguous_dim % align_element == 0);

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Alignment setting is invalid.\n");
        return Status::kInvalid;
      }
    }

    return Status::kSuccess;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static dim3
  get_block_shape() {
    return dim3(size(shape(ThreadLayout)), 1, 1);
  }

  static dim3
  get_grid_shape(Params const& params) {
    auto dim_m = ceil_div(size<0>(shape(params.dst)), get<0>(CTATileShape));
    auto dim_n = ceil_div(size<1>(shape(params.dst)), get<1>(CTATileShape));

    return dim3(dim_m, dim_n, 1);
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    auto k = args.filter_extent[0];
    auto c = args.filter_extent[NumDimensions - 1];
    auto srt = reverse(take<1,NumDimensions - 1>(args.filter_extent));

    // source shape (s,r,t,k,c)
    auto shape_src = flatten(make_shape(srt, k, c));
    auto shape_dst = DstFormat == FilterFormat::CTRSK ? make_shape(k, c * product(srt)) : make_shape(c, k * product(srt));

    auto src = make_tensor(make_gmem_ptr(recast_ptr<const Element>(args.src_ptr)), make_layout(shape_src));
    auto dst = make_tensor(make_gmem_ptr(recast_ptr<Element>(args.dst_ptr)), make_layout(shape_dst));

    return Params{src, dst};
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char *smem_buf) {
    // Tile the input tensor into blocks
    auto block_coord = make_coord(blockIdx.x, blockIdx.y);
    auto block_shape = make_shape(Int<4 * AlignmentBytes / static_cast<int>(sizeof(Element))>{}, Int<32>{});
    // Default thread layout: (4, 32)
    auto thread_layout = make_layout(make_shape(Int<4>{}, Int<32>{}));
    auto vec_layout = make_layout(make_shape(Int<AlignmentBytes / static_cast<int>(sizeof(Element))>{}, Int<1>{}));

    Tensor tile_D = local_tile(params.dst, block_shape, block_coord);

    // Construct tiled copy
    using AccessType = cutlass::AlignedArray<Element, size(vec_layout)>;
    using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

    auto tiled_copy = make_tiled_copy(Atom{}, thread_layout, vec_layout);
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    Tensor thr_tile_D = thr_copy.partition_D(tile_D);

    // shape (s, r, t)
    auto shape_trs = take<0, NumDimensions - 2>(shape(params.src));
    // strided_c = c for format CTRSK, strided_c = k for format KTRSC
    auto strided_c = DstFormat == FilterFormat::CTRSK ? get<NumDimensions - 1>(shape(params.src)) : get<NumDimensions - 2>(shape(params.src));
    // shape (s, r, t, c) for format CTRSK and shape (s, r, t, k) for format KTRSC 
    auto shape_ctrs = append<NumDimensions - 1>(shape_trs, strided_c);
    auto srtc_coord = idx2crd(int(blockIdx.y * get<1>(block_shape) + threadIdx.x / size<0>(thread_layout)), shape_ctrs);
    // index of k for format CTRSK and index of c for format KTRSC
    auto n_layout = make_layout(make_shape(gridDim.x, size<0>(thread_layout)), make_stride(size<0>(block_shape), size<0>(vec_layout)));
    int n_idx = n_layout(make_coord(blockIdx.x, threadIdx.x % size<0>(thread_layout)));

    // Fragment to load from S and store to D
    auto frag = make_fragment_like(thr_tile_D);
    // Predicate tensor.
    Tensor thr_tile_P = make_tensor<bool>(shape(thr_tile_D));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(frag); ++i) {
      auto srt_coord = take<0, NumDimensions - 2>(srtc_coord);
      auto kc_coord = DstFormat == FilterFormat::CTRSK ?
          make_coord(n_idx+i, get<NumDimensions - 2>(srtc_coord)) :
          make_coord(get<NumDimensions - 2>(srtc_coord), n_idx+i);
      auto coord = flatten(make_coord(srt_coord, kc_coord)); 
      thr_tile_P(i) = elem_less(coord, shape(params.src));
      if (thr_tile_P(i)) {
        frag(i) = params.src(coord);
      }
    }

    // Copy from RMEM to GMEM
    copy_if(tiled_copy, thr_tile_P, frag, thr_tile_D);
  }
};

} // namespace cutlass::transform::kernel
