# CUTLASS Hopper FMHA Example

This sample showcases how to implement fused multi-head attention (FMHA) using
CUTLASS for the NVIDIA Hopper architecture. At its heart, the forward pass of
FMHA is a GEMM-online softmax-GEMM fusion, whereas the backward pass is a slightly
more complex structure (basically, a GEMM-softmax-2xGEMM-2xGEMM fusion).
For more information please refer to the [Flash Attention 3 paper](https://arxiv.org/abs/2407.08608).

The forward pass kernel supports head dims 32, 64, 128, and 256 for fp16 and bf16 input data types,
and head dims 128, and 256 for fp8.
All kernels use the Tensor Memory Accelerator for loads.
Kernels with head dims 128 and 256 have warp-specialized cooperative schedules.

Backward pass kernels (fp16 only) support head dims 32, 64, and 128, and all support
warp-specialized cooperative schedules.

## Customization

### Mask Fusion

Similar to the [Blackwell FMHA example](../77_blackwell_fmha/README.md), attention masks such as
causal masking can be fused into the kernel. To modify the code for such fusions,
`collective/fmha_fusion.hpp` provides the easiest customization point.
The `before_softmax` function is called with the accumulator of the first GEMM and the logical
positions of those elements. It is well-suited for applying masks or activations.

### MHA Variants

Using CuTe, it is easy to represent the various attention variants.
Where regular multi-head attention's layout for the head dimension is (numHeads:headStride),
for single-head attention it is simply (1:0) everywhere,
for GQA it is normal in Q and (numHeads/numGroups,numGroups:headStride,0) in KV,
and for MQA it is normal for Q and (numHeads:0) in KV.
As such, beyond general stride handling, no additional work is needed to support these,
and the example will just demonstrate regular multi-head attention.

### FP8

The warp-specialized forward kernel supports FP8 computation with both FP32 and FP16
accumulation for the Q*K product. They can be enabled in the runner by defining FP8.

## Performance
Forward pass kernels can generally come close to that of FA3, but backward pass
kernels are more limited in performance and are not expected to reach the same level of performance
as FA3.

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
