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
/*!
  \file 1-D Distributed GEMM Schedules

  NOTE: This API is __experimental__ and will change heavily over time. Particularly the use of
  CuTe layouts as integer functions in defining iteration-to-tile mappings is over-expressive and
  leaves plenty of room for incorrect/unexpected behavior.
  Please proceed with caution when modifying these schedules or defining new ones.

  Device/iteration mappings are defined with CuTe layouts, 
  since they are functions from integers to integers as well.
  
  Each mapping is defined as a linear function of 2 variables (rank-2 layout):
   First variable (mode) is device index, second variable (mode) is iteration.
   A constant is also added to the final result as an offset value. This is a temporary workaround
   so that identity ownership mappings in the final iteration can be guaranteed for the schedules
   currently implemented.
  How are these mappings defined?
    Each schedule represents a unique parallel matrix multiplication algorithm, which describes how
    matrices/tensors are distributed among TP GPUs.

    Depending on the algorithm, access patterns (GPU to tile or (GPU, iteration) to tile) mappings)
    are not necessarily going to be the identity function.

  Pitfalls:
    The current representation uses CuTe layouts as arbitrary linear functions that map
    (GPU, iteration) to tile indices.
    This approach is over-expressive, and therefore makes a lot of assumptions on the part of the
    developer in how these mappings are defined. This can easily lead to incorrect implementations
    if not handled carefully.

  
  Assumption made in all schedules: TP == number of iterations (stages)
*/

#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

#include "cutlass/experimental/distributed/schedules/dist_gemm_base_schedule.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed::schedules {

// GEMM + Reduce Scatter
// A and B are tiled along the K mode, which means each GPU gets an [M, K / TP]-shaped slice of A,
// and an [N, K / TP] slice of B.
// A is further tiled along the M mode, so that each stage/iteration computes a GEMM of shape
// [M / TP, N, K / TP], and the epilogue will perform the reduction by reading its C tensor directly
// from the left peer's previous D buffer.
//
// Below is an illustration of the tiling and iteration mappings for this pattern in the TP=4 case:
//
//   Rows correspond to the M mode, columns correspond to the K mode for A and B and N mode for 
//   C and D.  Because sharding is done along K, each column of tiles is owned by one GPU.
//   Values in the grid correspond to the iteration/stage accessing the tile.
//   * means the same tile is accessed in all iterations/stages.
//
//         Tensor A                             Tensor B              
//                                                                    
//  GPU0  GPU1  GPU2  GPU3              GPU0  GPU1  GPU2  GPU3        
// |-----|-----|-----|-----|           |-----|-----|-----|-----|      
// |     |     |     |     |           |     |     |     |     |      
// |  3  |  0  |  1  |  2  |           |     |     |     |     |      
// |_____|_____|_____|_____|           |     |     |     |     |      
// |     |     |     |     |           |     |     |     |     |      
// |  2  |  3  |  0  |  1  |           |     |     |     |     |      
// |_____|_____|_____|_____|           |  *  |  *  |  *  |  *  |      
// |     |     |     |     |           |     |     |     |     |      
// |  1  |  2  |  3  |  0  |           |     |     |     |     |      
// |_____|_____|_____|_____|           |     |     |     |     |      
// |     |     |     |     |           |     |     |     |     |      
// |  0  |  1  |  2  |  3  |           |     |     |     |     |      
// |_____|_____|_____|_____|           |_____|_____|_____|_____|      
//                                                                    
//                          M x K                               N x K 
//
//
//              Tensor C                            Tensor D              
//              (Peer's D)
//                                         
//                                                                        
//      |-----------------------|           |-----------------------|     
//      |                       |           |                       |     
// GPU0 |         1,2,3         |      GPU0 |           *           |     
//      |_______________________|           |_______________________|     
//      |                       |           |                       |     
// GPU1 |         1,2,3         |      GPU1 |           *           |     
//      |_______________________|           |_______________________|     
//      |                       |           |                       |     
// GPU2 |         1,2,3         |      GPU2 |           *           |     
//      |_______________________|           |_______________________|     
//      |                       |           |                       |     
// GPU3 |         1,2,3         |      GPU3 |           *           |     
//      |_______________________|           |_______________________|     
//                                                                        
//                               M x N                               M x N
//
//
//  Tensor A's access pattern can be expressed as follows as a function of GPU index and iteration:
//    tile_idx = ((device_idx - 1) - iter + TP) % TP
//  
//  and can be expressed with the following CuTe layout:
//    (TP, TP) : (1, -1)
//  with ProcessorOffset = -1
//
//
//  Note: Since this schedule does not expose any communication, iteration 0 has no reduction step,
//  therefore epilogue is sourceless in iteration 0, and in the rest of the iterations the epilogue
//  source is a remote pointer to Tensor D owned by its left peer.
//
//  Left peer is simply (device_idx - 1 + TP) % TP, which is expressed with the following CuTe layout:
//    (TP, TP) : (1, 0)
//
template <class TP_>
struct ReduceScatter1D_TilingA_RotatingC: BaseSchedule<
    TP_,
    /* ProcessorTiler_ = */ cute::Shape<_1, _1, TP_, _1>,
    /* IterationTiler_ = */ cute::Shape<TP_, _1, _1, _1>,
    /* PeerDeviceMapping_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _0>>,                             // (left neighbor) = (device_idx + ProcessorOffset + TP) % TP, with ProcessorOffset = -1
    /* IterationMappingM_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _m1>>,                            // = (device_idx + ProcessorOffset - iter + TP) % TP, with ProcessorOffset = -1
    /* IterationMappingN_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::N == 1) = 0
    /* IterationMappingK_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::K == 1) = 0
    /* IterationMappingL_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::L == 1) = 0
    /* ProcessorOffset_ = */ _m1,
    /* MemcpyA_ = */ false,
    /* MemcpyB_ = */ false,
    /* KernelWritesArrivalFlag_ = */ true,
    /* NumBuffersA_ = */ 0,
    /* NumBuffersB_ = */ 0,
    /* NumBuffersC_ = */ 0,
    /* NumBuffersD_  = */ TP_{} - 1> {};

// This schedule is similar to ReduceScatter1D_TilingA_RotatingC, but with the second tiling
// done along N instead of M. All other details remain unchanged.
template <class TP_>
struct ReduceScatter1D_TilingB_RotatingC: BaseSchedule<
    TP_,
    /* ProcessorTiler_ = */ cute::Shape<_1, _1, TP_, _1>,
    /* IterationTiler_ = */ cute::Shape<_1, TP_, _1, _1>,
    /* PeerDeviceMapping_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _0>>,                             // (left neighbor) = (device_idx + ProcessorOffset + TP) % TP, with ProcessorOffset = -1
    /* IterationMappingM_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::N == 1) = 0
    /* IterationMappingN_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _m1>>,                            // = (device_idx + ProcessorOffset - iter + TP) % TP, with ProcessorOffset = -1
    /* IterationMappingK_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::K == 1) = 0
    /* IterationMappingL_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::L == 1) = 0
    /* ProcessorOffset_ = */ _m1,
    /* MemcpyA_ = */ false,
    /* MemcpyB_ = */ false,
    /* KernelWritesArrivalFlag_ = */ true,
    /* NumBuffersA_ = */ 0,
    /* NumBuffersB_ = */ 0,
    /* NumBuffersC_ = */ 0,
    /* NumBuffersD_  = */ TP_{} - 1> {};


// AllGather + GEMM
// A and B are tiled along the N mode, which means each GPU allgathers A,
// and operates with an [N / TP, K] slice of B.
// For pipelining, A is further tiled along the M mode, so that each stage/iteration computes a
// GEMM of shape [M / TP, N / TP, K], and concurrently we copy a peer's A slice into a local buffer
// for the next stage/iteration.
//
// Below is an illustration of the tiling and iteration mappings for this pattern in the TP=4 case:
//
//   Rows correspond to the M mode, columns correspond to the K mode for A and B and N mode for 
//   C and D.
//
//   Since this is a pipelined schedule without exposed communication, the first iteration starts
//   off immediately and operates on local slices of A and B. In the rest of the iterations, each
//   GPU accesses a slice of A copied from a peer GPU while it was busy with the last stage.
//
//   Values in the following grids correspond to the peer buffer accessed by each GPU during
//   different iterations:
//
//              Tensor A                         Tensor A               
//               iter 0                           iter 1                
//                                                                      
//      |-----------------------|        |-----------------------|      
//      |                       |        |                       |      
// GPU0 |           0           |        |           1           |      
//      |_______________________|        |_______________________|      
//      |                       |        |                       |      
// GPU1 |           1           |        |           2           |      
//      |_______________________|        |_______________________|      
//      |                       |        |                       |      
// GPU2 |           2           |        |           3           |      
//      |_______________________|        |_______________________|      
//      |                       |        |                       |      
// GPU3 |           3           |        |           0           |      
//      |_______________________|        |_______________________|      
//                                                                      
//                               M x K                            M x K 
//
//              Tensor A                         Tensor A               
//               iter 2                           iter 3                
//                                                                      
//      |-----------------------|        |-----------------------|      
//      |                       |        |                       |      
// GPU0 |           2           |        |           3           |      
//      |_______________________|        |_______________________|      
//      |                       |        |                       |      
// GPU1 |           3           |        |           0           |      
//      |_______________________|        |_______________________|      
//      |                       |        |                       |      
// GPU2 |           0           |        |           1           |      
//      |_______________________|        |_______________________|      
//      |                       |        |                       |      
// GPU3 |           1           |        |           2           |      
//      |_______________________|        |_______________________|      
//                                                                      
//                               M x K                            M x K 
//
//   Values in the following grids correspond to the tile accessed during each iteration.
//   * means the same tile is accessed in all iterations/stages.
//
//              Tensor B                             Tensor C/D               
//                                                                          
//                                                                          
//      |-----------------------|            |-----|-----|-----|-----|      
//      |                       |            |     |     |     |     |      
// GPU0 |           *           |       GPU0 |  0  |  1  |  2  |  3  |      
//      |_______________________|            |_____|_____|_____|_____|      
//      |                       |            |     |     |     |     |      
// GPU1 |           *           |       GPU1 |  3  |  0  |  1  |  2  |      
//      |_______________________|            |_____|_____|_____|_____|      
//      |                       |            |     |     |     |     |      
// GPU2 |           *           |       GPU2 |  2  |  3  |  0  |  1  |      
//      |_______________________|            |_____|_____|_____|_____|      
//      |                       |            |     |     |     |     |      
// GPU3 |           *           |       GPU3 |  1  |  2  |  3  |  0  |      
//      |_______________________|            |_____|_____|_____|_____|      
//                                                                          
//                               N x K                                M x N 
//
//
//  Tensor C/D's access pattern can be expressed as follows as a function of GPU index and iteration:
//    tile_idx = (device_idx + iter) % TP
//  
//  and can be expressed with the following CuTe layout:
//    (TP, TP) : (1, 1)
//
//  This schedule does not need a ProcessorOffset constant.
//
//  Peer devices from which A slices are copied is also expressed with the same function and CuTe
//  layout.
//
template <class TP_>
struct AllGather1D_TilingCD_RotatingA: BaseSchedule<
    TP_,
    /* ProcessorTiler_ = */ cute::Shape<_1, TP_, _1, _1>,
    /* IterationTiler_ = */ cute::Shape<TP_, _1, _1, _1>,
    /* PeerDeviceMapping_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _1>>,                             // = device_idx + iter
    /* IterationMappingM_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _1>>,                             // = device_idx + iter
    /* IterationMappingN_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::N == 1) = 0
    /* IterationMappingK_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::K == 1) = 0
    /* IterationMappingL_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::L == 1) = 0
    /* ProcessorOffset_ = */ _0,
    /* MemcpyA_ = */ true,
    /* MemcpyB_ = */ false,
    /* KernelWritesArrivalFlag_ = */ false,
    /* NumBuffersA_ = */ TP_{} - 1,
    /* NumBuffersB_ = */ 0,
    /* NumBuffersC_ = */ 0,
    /* NumBuffersD_ = */ 0>{};

// This schedule is similar to AllGather1D_TilingCD_RotatingA, but with the order of tiling
// swapped from N then M to M then N. This means slices of B are rotated around GPUs instead of
// slices of A. All other details remain unchanged.
template <class TP_>
struct AllGather1D_TilingCD_RotatingB: BaseSchedule<
    TP_,
    /* ProcessorTiler_ = */ cute::Shape<TP_, _1, _1, _1>,
    /* IterationTiler_ = */ cute::Shape<_1, TP_, _1, _1>,
    /* PeerDeviceMapping_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _1>>,                             // = device_idx + iter
    /* IterationMappingM_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::M == 1) = 0
    /* IterationMappingN_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_1, _1>>,                             // = device_idx + iter
    /* IterationMappingK_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::K == 1) = 0
    /* IterationMappingL_ = */ cute::Layout<cute::Shape<TP_, TP_>, cute::Stride<_0, _0>>,                             // (IterationTiler::L == 1) = 0
    /* ProcessorOffset_ = */ _0,
    /* MemcpyA_ = */ false,
    /* MemcpyB_ = */ true,
    /* KernelWritesArrivalFlag_ = */ false,
    /* NumBuffersA_ = */ 0,
    /* NumBuffersB_ = */ TP_{} - 1,
    /* NumBuffersC_ = */ 0,
    /* NumBuffersD_ = */ 0>{};


} // namespace cutlass::distributed::schedules

///////////////////////////////////////////////////////////////////////////////

