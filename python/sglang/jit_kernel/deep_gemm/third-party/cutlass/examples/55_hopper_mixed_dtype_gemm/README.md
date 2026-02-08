This example shows how to do mixed types GEMMs in CUTLASS.

## High level overview 
This example shows how to perform GEMMs on Hopper when A and B have different types. This implementation always passes the type with fewer bits through the register file and upcasts to the type with the higher bit count.

When relying on `KernelScheduleAuto`, the main loop supporting different A and B types will be selected whenever the bit count of A is not equal to the bit count of B. Users can manually select the mixed type main loop and explicitly choose the scheduling policy by specifying one of the following schedules to the `CollectiveBuilder`:  `KernelTmaWarpSpecialized`, `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative`.

This first version only supports mixed type GEMMs using TMA.


## Performance

While the example offers a harness for straightforward benchmarking, this initial implementation isn't optimized for performance in the majority of scenarios. We expect this implementation to be performant for `{fp16, bf16} x {int8, int4, int2}` and `{fp8} x {int4}` for problems that are compute bound. Additionally, we expect good performance for `fp16`, `bf16` or `fp32` scales and zero-points. For best performance, it is ideal to have the scales and zero-points be the same type as mma's type.

The scale only mode for `fp8 x int4` is significantly slower than direct conversion mode. There is a lookup-table workaround targeting this mode, as shown in `55_hopper_int4_fp8_gemm.cu`. To use this feature, use `cutlass::Array<ElementScale, 8>` as the scale type in the collective builder. However, it requires modifications to the encoding of quantized weights and scale factors. Also, scale with zero point mode is not supported for now.

Additionally, it's recommended to reorder the narrow data type tensor such that elements read into register file by the same thread are contiguous in global and shared memory. The user can use the helper function `compute_memory_reordering_atom` and `reorder_tensor` to achieve this. See `55_hopper_int4_fp8_gemm.cu` and `55_hopper_int4_bf16_gemm.cu` for more details.

We are currently optimizing the following cases:
1. Memory bound cases for all types
2. `fp8 x {int2, uint2}` case

## Limitations

* The type that needs to be converted must go through the register file. This means that the collective will swap and transpose whenever the type with fewer bits is the B operand. The user must be aware of when these swaps happen. Note that TMA epilogues currently do not support *implicit* swap + transpose, so non-tma epilogues must be used in this case. We plan to relax this limitation in a future release.

* The layout of the narrow type must be K-major. This means the following:
  * Narrow type is the A operand: Must be Row-Major
  * Narrow type is the B operand: Must be Column-Major

* For 8-bit x 4-bit or 2-bit, both inputs must be K-major.

* TMA requires an alignment of 128 bits. As a result, for a type with `B` bits, `B x TILE_K` must be a multiple of 128 bits.

* The type of the scale and zero-point type must be two bytes or more.

* The group size must be equal to gemm-k size (indicating a broadcast), or it must be a multiple of the threadblock-k size.

## Upcoming features

* Optimizations for memory bound cases.

* Optimizations for scale and zero-point loading when the group size is not equal to the threadblock-k size.

## Copyright

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
