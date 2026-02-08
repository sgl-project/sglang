# CUTLASS - Programming Examples

> [!IMPORTANT]
> ### ⚠️ **Not for Benchmarking!** ⚠️
> 
> These examples are designed **solely for demonstrating CUTLASS functionality** and may **NOT optimized for performance benchmarking**.
> 
> **For accurate performance measurements**, please use the **[CUTLASS Profiler](../tools/profiler/)** instead (recommended) or manually auto-tune the example, if unavailable via the profiler.
> 


* [00_basic_gemm](00_basic_gemm/)
    
    launches a basic GEMM with single precision inputs and outputs

* [01_cutlass_utilities](01_cutlass_utilities/)

    demonstrates CUTLASS Utilities for allocating and initializing tensors

* [02_dump_reg_smem](02_dump_reg_smem/)

    debugging utilities for printing register and shared memory contents

* [03_visualize_layout](03_visualize_layout/)

    utility for visualizing all layout functions in CUTLASS

* [04_tile_iterator](04_tile_iterator/)

    example demonstrating an iterator over tiles in memory

* [05_batched_gemm](05_batched_gemm/)

    example demonstrating CUTLASS's batched strided GEMM operation

* [06_splitK_gemm](06_splitK_gemm/)

    example demonstrating CUTLASS's Split-K parallel reduction kernel

* [07_volta_tensorop_gemm](07_volta_tensorop_gemm/)

    example demonstrating mixed precision GEMM using Volta Tensor Cores

* [08_turing_tensorop_gemm](08_turing_tensorop_gemm/)

    example demonstrating integer GEMM using Turing Tensor Cores

* [09_turing_tensorop_conv2dfprop](09_turing_tensorop_conv2dfprop/)

    example demonstrating integer implicit GEMM convolution (forward propagation) using Turing Tensor Cores

* [10_planar_complex](10_planar_complex/)

    example demonstrating planar complex GEMM kernels

* [11_planar_complex_array](11_planar_complex_array/)

    example demonstrating planar complex kernels with batch-specific problem sizes

* [12_gemm_bias_relu](12_gemm_bias_relu/)

    example demonstrating GEMM fused with bias and relu

* [13_two_tensor_op_fusion](13_two_tensor_op_fusion/)

    example demonstrating two GEMMs or convolutions fused in one kernel

* [14_ampere_tf32_tensorop_gemm](14_ampere_tf32_tensorop_gemm/)

    example demonstrating FP32 GEMM with implicit TF32 conversion

* [15_ampere_sparse_tensorop_gemm](15_ampere_sparse_tensorop_gemm/)

    example demonstrating usage of Sparse Tensor cores

* [16_ampere_tensorop_conv2dfprop](16_ampere_tensorop_conv2dfprop/)

    example demonstrating forward convolution on tensors of layout NHWC

* [17_fprop_per_channel_bias](17_fprop_per_channel_bias/)

    example demonstrating convolution fused with per channel bias and relu

* [18_ampere_fp64_tensorop_affine2_gemm](18_ampere_fp64_tensorop_affine2_gemm/)

    example demonstrating Affine-2 GEMM

* [19_tensorop_canonical](19_tensorop_canonical/)

    Canonical GEMM using tensor cores

* [20_simt_canonical](20_simt_canonical/)

    Canonical GEMM using SIMT

* [21_quaternion_gemm](21_quaternion_gemm/)

    example demonstrating Quaternion GEMM computations

* [22_quaternion conv](22_quaternion_conv/)

    example demonstrating Quaternion convolution 

* [23_ampere_gemm_operand_reduction_fusion](23_ampere_gemm_operand_reduction_fusion/)

    example demonstrating how to reduce one of the operands of the GEMM along the k-dimension when computing GEMM

* [24_gemm_grouped](24_gemm_grouped/)

    example demonstrating batch of GEMM operations with distinct problem sizes

* [25_ampere_fprop_mainloop_fusion](25_ampere_fprop_mainloop_fusion/)

    example demonstrating fusing activation's per channel scale+bias+relu into the fgrad mainloop

* [26_ampere_wgrad_mainloop_fusion](26_ampere_wgrad_mainloop_fusion/)

    example demonstrating fusing activation's per channel scale+bias+relu into the wgrad mainloop

* [27_ampere_3xtf32_fast_accurate_tensorop_gemm](27_ampere_3xtf32_fast_accurate_tensorop_gemm/)

    example demonstrating emulation of a fast accurate SGEMM with TF32 operations

* [28_ampere_3xtf32_fast_accurate_tensorop_fprop](28_ampere_3xtf32_fast_accurate_tensorop_fprop/)

    example demonstrating emulation of a fast accurate FP32 convolution with TF32 operation 

* [29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm](29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm/)

    example demonstrating emulation of a fast accurate CGEMM with TF32 operation

* [30_wgrad_split_k](30_wgrad_split_k/)

    example demonstrating how to compute conv2d gradient with respect to weight (wgrad) together with split-K

* [31_basic_syrk](31_basic_syrk/)

    example demonstrating Symmetric Rank-K update

* [32_basic_trmm](32_basic_trmm/)

    example demonstrating Triangular Matrix-Matrix multiplication

* [33_ampere_3xtf32_tensorop_symm](33_ampere_3xtf32_tensorop_symm/)

    example demonstrating Symmetric Matrix-Matrix multiplication with FP32 emulation

* [34_transposed_conv2d](34_transposed_conv2d/)

    example demonstrating how to compute 2d transposed convolution, also known as deconvolution, using CUTLASS conv2d Dgrad kernels

* [35_gemm_softmax](35_gemm_softmax/)

    example demonstrating GEMM fused with Softmax in mixed precision using Ampere Tensor Cores

* [36_gather_scatter_fusion](36_gather_scatter_fusion/)

    example demonstrating fuses gather before GEMM and scatter after GEMM into the same GEMM kernel

* [37_gemm_layernorm_gemm_fusion](37_gemm_layernorm_gemm_fusion/)

    example demonstrating fuses gemm->layernorm->gemm into one kernel.

* [38_syr2k_grouped](38_syr2k_grouped/)

    example demonstrating a batch of SYR2K operations with distinct problem sizes

* [39_gemm_permute](39_gemm_permute/)

    example demonstrating batched GEMM operations with output results permuted as reshaped tensors

* [40_cutlass_py](40_cutlass_py/)

    example demonstrating CUTLASS with Python interface

* [41_multi_head_attention](41_multi_head_attention/)

    example demonstrating attention example with non-fixed sequence length input

* [42_ampere_tensorop_group_conv](42_ampere_tensorop_group_conv/)

    example demonstrating how to run group convolution kernels using functions and data structures provided by CUTLASS using tensor cores

* [43_ell_block_sparse_gemm](43_ell_block_sparse_gemm/)

    example demonstrating a Block-Ell sparse gemm

* [44_fused_multi_head_attention](44_fused_multi_head_attention/)

    example demonstrating fused multihead attention (fixed & variable) using shared memory

* [45_dual_gemm](45_dual_gemm/)

    example demonstrating how to fuse two GEMMs sharing the same left input matrix into one kernel 

* [46_depthwise_simt_conv2dfprop](46_depthwise_simt_conv2dfprop/)

    example demonstrating depthwise 2d convolution kernels using functions and data structures provided by CUTLASS using SIMT instruction

* [47_ampere_gemm_universal_streamk](47_ampere_gemm_universal_streamk/)

    example contrasting the Stream-K parallel decomposition for GEMM threadblocks versus the
 "classic data-parallel" and "Split-K" decompositions.

* [48_hopper_warp_specialized_gemm](48_hopper_warp_specialized_gemm/)

    Simple tensorop GEMM example using CUTLASS 3.0 APIs targeting NVIDIA Hopper architecture

* [49_hopper_gemm_schedules_with_collective_builder](49_hopper_gemm_schedules_with_collective_builder/)

    Hopper GEMM example leveraging collective operation builders to showcase the builder API and the various kernel scheduled supported in CUTLASS 3.0 such as warp specialized persistent mainloops.

* [50_hopper_gemm_with_epilogue_swizzle](50_hopper_gemm_with_epilogue_swizzle/)

    Hopper GEMM example to create a GEMM kernel with custom a collective mainloop and a custom vectorized epilogue.

* [51_hopper_gett](51_hopper_gett/)

    Hopper GETT example illustrating the ease with which GETTs can be run due to CUTLASS 3.0's unified micro-kernels and CuTe's hierarchical layouts.

* [52_hopper_gather_scatter_fusion](52_hopper_gather_scatter_fusion/)

    Hopper example that fuses gather before GEMM and scatter after GEMM into the same kernel

* [53_hopper_gemm_permute](53_hopper_gemm_permute/)

    Hopper example demonstrating the fusion of tensor permutation operations with a GEMM kernel

* [54_hopper_fp8_warp_specialized_gemm](54_hopper_fp8_warp_specialized_gemm/)

    Hopper example of instantiating and running an FP8 GEMM kernel

* [55_hopper_mixed_dtype_gemm](55_hopper_mixed_dtype_gemm/)

    Hopper GEMM example with different A and B data types using CUTLASS 3.x APIs for DL kernels with fused dequantization.

* [56_hopper_ptr_array_batched_gemm](56_hopper_ptr_array_batched_gemm/)

    Hopper Ptr-Array Batched GEMM example using CUTLASS 3.x API.

* [57_hopper_grouped_gemm](57_hopper_grouped_gemm/)

    Hopper Grouped GEMM using CUTLASS 3.x API.

* [58_ada_fp8_gemm](58_ada_fp8_gemm/)

    Ada GEMM kernel targetting Ada FP8 tensor cores via the CUTLASS 2.x API.

* [59_ampere_gather_scatter_conv](59_ampere_gather_scatter_conv/)

    CuTe and CUTLASS 3.x based Ampere convolution fprop kernel capable of operating on both affine and gather/scatter tensors,
        showing how kernel authors can re-use CUTLASS 3.x collectives in their custom kernels. 

* [61_hopper_gemm_with_topk_and_softmax](61_hopper_gemm_with_topk_and_softmax/)

    Hopper GEMM kernel with Top-K and softmax epilogue fusion.

* [70_blackwell_gemm](70_blackwell_gemm)

    Simple dense GEMM example targeting the NVIDIA Blackwell SM100 Tensor Core MMA using CUTLASS 3.x APIs.

* [71_blackwell_gemm_with_collective_builder](71_blackwell_gemm_with_collective_builder)

    Blackwell SM100 GEMM example demonstrating compatible mainloop+epilogue builder schedules and epilogue visitor tree (EVT) construction

* [72_blackwell_narrow_precision_gemm](72_blackwell_narrow_precision_gemm/)

    Block-scaled dense GEMM example targeting the NVIDIA Blackwell SM100 Tensor Core MMA using CUTLASS 3.x APIs.

* [73_blackwell_gemm_preferred_cluster](73_blackwell_gemm_preferred_cluster/)

    Blackwell SM100 GEMM kernel with preferred cluster feature.

* [74_blackwell_gemm_streamk](74_blackwell_gemm_streamk/)

    Blackwell SM100 GEMM kernel using the Stream-K scheduler

* [75_blackwell_grouped_gemm](75_blackwell_grouped_gemm)

    Blackwell SM100 grouped GEMM kernel

* [76_blackwell_conv](76_blackwell_conv/)

    Simple convolution(fprop/dgrad/wgrad) example targeting NVIDIA Blackwell SM100 Tensor Core MMA using CUTLASS 3.x APIs.

* [77_blackwell_fmha](77_blackwell_fmha)

    Blackwell SM100 FMHA kernel

* [78_blackwell_emulated_bf16x9_gemm](78_blackwell_emulated_bf16x9_gemm)

    Blackwell SM100 FastFP32 (using BF16 to emulate SGEMM) kernel

* [79_blackwell_geforce_gemm](79_blackwell_geforce_gemm/)

    Blackwell SM120 MMA kernel targeting GeForce RTX 50 series CUDA Cores

* [80_blackwell_geforce_sparse_gemm](80_blackwell_geforce_sparse_gemm/)

    Blackwell SM120 sparse MMA kernel targeting GeForce RTX 50 series CUDA Cores

* [83_blackwell_sparse_gemm](83_blackwell_sparse_gemm)

    Blackwell SM100 Sparse Gemm kernel

* [84_blackwell_narrow_precision_sparse_gemm](84_blackwell_narrow_precision_sparse_gemm)

    Blackwell Block Scaled SM100 Sparse Gemm kernel

# CuTe - Programming Examples

Examples that do not rely on CUTLASS and directly showcase the features of CuTe are located in [cutlass/examples/cute](./cute/).

Additionally, CuTe's core layout and layout algebra have their own test cases within [cutlass/test/unit/cute/core/](../test/unit/cute/core/) that users might find useful as examples of CuTe.

# Python Interface Examples

Examples leveraging CUTLASS's [Python interface](../python/README.md) are located in [cutlass/examples/python](python/).

# Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
