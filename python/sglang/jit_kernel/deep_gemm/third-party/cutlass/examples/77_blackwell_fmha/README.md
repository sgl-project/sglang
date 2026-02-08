# FMHA for Blackwell: Forward

This sample provides code for fused multi-head attention forward, context, or generation phase.
It supports HeadDims of 32, 64, and 128, and fp8, fp16, and bf16 input data types.

For forward or context usage, use an M-blocking (Seqlen-Q) of 256 and an N-blocking (Seqlen-K) of 128.
For generation usage, use an M-blocking (Num-Groups) of 128 (although the limit is currently 32 for actual Num-Groups), and a N-blocking (Seqlen-K) of 64, 128 or 256.

Context loads are done via TMA, whereas generation usage utilized `cp.async` and is thus more amenable to complex load patterns.

For variable sequence length, the code requires a batch of valid (but never used) padding memory ahead of the first output batch. No padding is needed for the input tensor, but it requires that the input tensor contain no NaN or Inf values. Note that users should set `total_length` to the `problem_shape`.

The approach of this implementation is to reuse the selection logic of the collective gemm builder and recombine the result into an FMHA kernel.
The kernel and collective layer are then formulated to be fmha-specific.
The design assigns two tiles to each threadblock, and pingpongs between them in terms of matrix-matrix multiplication and softmax.

The example builds four binaries, showcasing the context and generation usage for fp8 and fp16.
For detailed information on how to invoke them, check out either the tests in `CMakeLists.txt` or the `--help` for them.

To modify the code for fusions, `collective/fmha_fusion.hpp` provides the easiest customization point.
The `apply_mask` function is called with the accumulator of the first GEMM and the logical positions of those elements.
It is well-suited for applying masks or activations.
More complex fusions that require memory loads would require modifying the mainloop collective to orchestrate the load via TMA.

# FMHA for Blackwell: Backward

This sample provides code for fused multi-head attention backward pass.
It supports HeadDims of 64 and 128, and fp8, fp16, and bf16 input data types.
The blocking in sequence length Q and K is 128, loads are done via TMA.
We support causal masking.
The structure of this code is very similar to the forward pass, and the techniques are analogous.

There are three kernels to compute backwards:
1. `FmhaKernelBwdSumOdO` to compute the sum of the outer product of O and dO.
3. `Sm100FmhaBwdKernelTmaWarpSpecialized` to compute the backward pass.
2. `FmhaKernelBwdConvert` to convert the dQ from fp32 to the final output precision.

`Sm100FmhaBwdKernelTmaWarpSpecialized` is the main point of this sample, as it demonstrates how to use tensor cores to achieve a high performance fused kernel.

## MLA Blackwell Backward

The sample also provides the feature of MLA backward(d=192, d_vo=128). To enable MLA backward, please specify `--d=192 --d_vo=128` when running the bwd sample. 

`Sm100FmhaBwdMlaKernelTmaWarpSpecialized`is the main point for MLA backward. The MLA approach is slightly different from the original one to enable high performance with the MLA shape. 

# MLA Inference for Blackwell

This sample provides code for fused multi-head latent attention inference in
the weight-absorbed regime, i.e. for latent head dim 512, and rope head dim 64.
It supports fp16, bf16, and fp8 input and output types.

To accommodate the large output accumulator due to the large latent head dimension,
the sample demonstrates how to leverage 2Sm Blackwell tensor cores.

Loading can be done via TMA (either without paging or with page size 128), or using `cp.async`
for support of any power-of-two page size less than or equal to 128.
With paging, the code also supports variable sequence length.

The approach of this implementation is to reuse the selection logic of the collective gemm builder and recombine the result into an MLA kernel.

The example builds six binaries, showcasing TMA and `cp.async` usage, as well as a back-to-back gemm (essentially turning the softmax into a no-op) for fp8 and fp16.
For detailed information on how to invoke them, check out either the tests in `CMakeLists.txt` or the `--help` for them.

# Changes

* 4.1.0: Enhanced testing of variable sequence length; disabled B2B mode in MLA
  to simplify the sample, clarified that `fmha_gen`  sample only supports head
  dim 128.

* 4.3.0: For variable sequence length, the code requires a batch of valid (but never used) padding memory ahead of the first output batch. No padding is needed for the input tensor, but it requires that the input tensor contain no NaN or Inf values. Note that users should set `total_length` to the `problem_shape`.

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
