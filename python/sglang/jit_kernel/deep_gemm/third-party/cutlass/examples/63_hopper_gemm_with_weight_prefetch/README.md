# GEMM with L2 weight prefetch

A non-persistent warp specialized GEMM directed at low latency inference.

The kernel can optionally prefetch a portion of weights (operand `A`) into L2 cache while the 
rest of the warps are waiting on the previous kernel to finish writing and flush its memory.
An example of this is normalization or reduction kernels that are immediately followed by a GEMM.

It exposes two runtime parameters:
1. `overlap_ratio`: how early `griddepcontrol.launch_dependent_grids` is issued. 
   Default is `0.5`, meaning after approximately half of K tiles are loaded by DMA warps.
2. `prefetch_ratio`: what percentage of K tiles to prefetch. 
   Default is `-1.0`, meaning prefetching will stop as soon as other DMA warps are past
   `griddepcontrol`.

It is highly recommended to auto-tune these parameters per GEMM and according to some end to end 
runtime (either an entire transformer layer or multiple, but probably not the entire model.)

TMA loads use non-default cache hints: `A` (weights) are loaded with `EvictFirst`, and `B` (activation)
is loaded with `EvictLast`.

## Getting started
To use this kernel in your own target, add this directory to your includes, and include the 
following headers from this example:

```cxx
#include "collective/dispatch_policy_extra.hpp"
#include "collective/builder.hpp"
#include "kernel/sm90_gemm_tma_warpspecialized_with_prefetch.hpp"
```

And then use either one of the new kernel schedules:

```cxx
// Without separate warps for A and B
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccumWithPrefetch;

// With separate warps for A and B
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA;
```

The kernel with separate warps for A and B (
`KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA`)
is expected to be more performant than the other, especially since it allows the kernel to load 
weights into shmem ahead of the `griddepcontrol`.

As for other GEMM parameters, Thread Block Cluster larger than 1 CTA are not yet supported, and
obviously the kernel layer implementation is warp specialized and uses the TMA, and other kernel
layers or collectives require reimplementation.

## Example

Using the example is mostly straightforward.
Just build, and run with your choice of `MNK`:

```bash
./63_hopper_gemm_with_weight_prefetch --m=8192 --n=1 --k=8192
```

You can also disable the overlap or try different overlap and prefetch ratios and see the
difference:

```bash
echo "Without overlap and prefetch"
./63_hopper_gemm_with_weight_prefetch --o=-1.0 --p=-1.0

echo "Overlap ratio of 0.5, best effort prefetch"
./63_hopper_gemm_with_weight_prefetch --o=0.5 --p=-1.0

echo "Overlap ratio of 0.8, prefetch ratio of 0.7"
./63_hopper_gemm_with_weight_prefetch --o=0.8 --p=0.7
```

However, note that the example still runs a single GEMM, and most of the performance improvement
is expected in end to end applications.

## Limitations
* The parameter defaults are typically not good choices, especially `prefetch_ratio`. 
  When `prefetch_ratio` is unspecified (set to `-1.0`), the prefetch warp will `try_wait` on a 
  memory barrier before issuing every single TMA load, and in many cases this will slow down 
  prefetching to the point of being almost ineffective.

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
