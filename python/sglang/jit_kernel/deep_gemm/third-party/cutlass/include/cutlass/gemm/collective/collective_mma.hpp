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

#include "cutlass/gemm/collective/collective_mma_decl.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/collective/sm70_mma_twostage.hpp"
#include "cutlass/gemm/collective/sm80_mma_multistage.hpp"
#include "cutlass/gemm/collective/sm80_mma_array_multistage.hpp"
#include "cutlass/gemm/collective/sm90_mma_multistage_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_multistage_gmma_rs_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp" 
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized_fp8.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized_fp8.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp"
#if !defined(__CUDACC_RTC__)
#include "cutlass/gemm/collective/sm100_mma_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_mma_warpspecialized_emulated.hpp"
#include "cutlass/gemm/collective/sm100_mma_array_warpspecialized_emulated.hpp"
#include "cutlass/gemm/collective/sm100_sparse_mma_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_blockscaled_sparse_mma_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp" 
#include "cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp" 
#include "cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp"
#include "cutlass/gemm/collective/sm100_mma_array_warpspecialized_blockwise_scaling.hpp"
#include "cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp"
#include "cutlass/gemm/collective/sm100_mma_cpasync_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_mma_mixed_tma_cpasync_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_blockscaled_mma_mixed_tma_cpasync_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm103_blockscaled_mma_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm103_blockscaled_mma_array_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm120_mma_tma.hpp"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp"
#include "cutlass/gemm/collective/sm120_sparse_mma_tma.hpp"
#include "cutlass/gemm/collective/sm120_blockscaled_sparse_mma_tma.hpp"
#include "cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp"
#include "cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp"
#endif // !defined(__CUDACC_RTC__) 




/////////////////////////////////////////////////////////////////////////////////////////////////
