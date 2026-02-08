/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/collective/builders/sm90_gmma_builder.inl"
#include "cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl"
#if !defined(__CUDACC_RTC__) 
#include "cutlass/gemm/collective/builders/sm100_umma_builder.inl"              
#include "cutlass/gemm/collective/builders/sm100_9xBF16_umma_builder.inl"       
#include "cutlass/gemm/collective/builders/sm100_sparse_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl"  
#include "cutlass/gemm/collective/builders/sm100_blockwise_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm100_blockscaled_sparse_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm100_simt_builder.inl"
#include "cutlass/gemm/collective/builders/sm100_mixed_input_umma_builder.inl"       
#include "cutlass/gemm/collective/builders/sm100_cpasync_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm100_mixed_tma_cpasync_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm100_blockscaled_mixed_tma_cpasync_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm103_blockscaled_umma_builder.inl"
#include "cutlass/gemm/collective/builders/sm120_mma_builder.inl"
#include "cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl"
#include "cutlass/gemm/collective/builders/sm120_sparse_mma_builder.inl"
#include "cutlass/gemm/collective/builders/sm120_blockscaled_sparse_mma_builder.inl"
#include "cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
